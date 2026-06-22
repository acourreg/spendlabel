"""Shared offline training utilities (tronc commun) for the ML paradigms.

Training is scoped to the FIRST ``train_n`` messages (env TRAIN_N, default 5000) so
the benchmark evaluates on a DISJOINT slice (consumer reads offset >= train_n) with
no train/test leakage — the consumer run is the held-out eval, scored by accuracy.py.

Two embedders are supported; each paradigm's training.py picks one, embeds the same
train slice, and trains its head:
  - HuggingFace MiniLM (all-MiniLM-L6-v2) via ONNX  -> 384-d, transformer forward.
  - Model2Vec (potion-base-8M) static vectors        -> 256-d, lookup + mean pool.

Heads: RandomForest (skl2onnx) or MLP (torch->ONNX). The class set (labels.json) is
derived from the train slice and shared across embedders. Requires the offline deps
(consumers/requirements-training.txt), NOT shipped in the slim inference container.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
REPO = Path(__file__).resolve().parents[4]
DATASET = REPO / "data" / "raw" / "dataset.tar.gz"
MIN_COUNT = 50          # drop CPV classes with fewer training samples
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
M2V_NAME = "minishlab/potion-base-8M"


def _clean(v):
    if v is None:
        return None
    v = v.strip()
    return v or None


def _reader(tar, name):
    m = next(x for x in tar.getmembers() if x.name.split("/")[-1] == name)
    return csv.DictReader(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8", errors="replace"))


def read_dataset():
    """Return (texts, labels) — same text + ground-truth derivation as the producer."""
    cpv = {}
    texts, labels = [], []
    with tarfile.open(DATASET, "r:gz") as tar:
        for r in _reader(tar, "tender_additionalClassifications.csv"):
            oc = _clean(r.get("main_ocid"))
            code = _clean(r.get("id"))
            if oc and code and oc not in cpv:
                cpv[oc] = code[:2]
        for r in _reader(tar, "main.csv"):
            oc = _clean(r.get("ocid"))
            if not oc:
                continue
            title = _clean(r.get("tender_title")) or _clean(r.get("title"))
            if not title:
                continue
            label = cpv.get(oc)
            if label is None:
                primary = _clean(r.get("tender_classification_id"))
                label = primary[:2] if primary else None
            if label is None:
                continue
            desc = _clean(r.get("tender_description")) or ""
            texts.append(title + " " + desc)
            labels.append(label)
    return texts, labels


def load_train_slice(pool: int | None = None):
    """Train on the first ``pool`` records; eval is the DISJOINT offset >= pool window.

    No train/test leakage (audit fix B2): training uses messages [0, pool) and the
    benchmark (scripts/rerun.sh) reads from offset ``pool`` onward, so every record
    scored at eval time was unseen in training. This replaces the previous
    even/odd split that deliberately overlapped train and eval (~50% leakage), which
    flattered the ML paradigms relative to the training-free ones.

    Keeps classes with >= MIN_COUNT, writes labels.json. Returns (texts, classes, y).
    """
    if pool is None:
        pool = int(os.getenv("TRAIN_POOL", "5000"))
    print("[slice] reading dataset...")
    texts, labels = read_dataset()
    texts, labels = texts[:pool], labels[:pool]   # first `pool` records (eval = offset>=pool)
    counts = Counter(labels)
    keep = {c for c, n in counts.items() if n >= MIN_COUNT}
    pairs = [(t, l) for t, l in zip(texts, labels) if l in keep]
    texts = [t for t, _ in pairs]
    classes = sorted(keep)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for _, l in pairs], dtype=np.int64)
    (MODELS / "labels.json").write_text(json.dumps(classes))
    print(f"[slice] {len(texts)} training records (first {pool}, eval is offset>={pool}), "
          f"{len(classes)} classes (>= {MIN_COUNT})")
    return texts, classes, y


# --- embedders (offline build of the model files used at inference) ---

def export_minilm():
    """Export the MiniLM encoder + tokenizer to ONNX under models/ (HuggingFace embedder)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    MODELS.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MINILM_NAME)
    model = AutoModel.from_pretrained(MINILM_NAME).eval()

    class Enc(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    d = tok(["hello world"], return_tensors="pt", padding=True, truncation=True, max_length=256)
    torch.onnx.export(
        Enc(model), (d["input_ids"], d["attention_mask"], d["token_type_ids"]),
        str(MODELS / "minilm.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={k: {0: "b", 1: "s"} for k in
                      ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]},
        opset_version=14, dynamo=False,
    )
    tok.save_pretrained(MODELS)
    print("  exported minilm.onnx + tokenizer.json")


def save_model2vec():
    """Download + save the Model2Vec static model under models/model2vec/."""
    from model2vec import StaticModel
    out = MODELS / "model2vec"
    StaticModel.from_pretrained(M2V_NAME).save_pretrained(str(out))
    print(f"  saved model2vec -> {out}")


def parity_check(embedder):
    from sentence_transformers import SentenceTransformer
    samples = ["road resurfacing and drainage works", "NHS community nursing framework",
               "cloud hosting and cyber security support", "school catering services"]
    ref = SentenceTransformer(MINILM_NAME).encode(samples, normalize_embeddings=True)
    got = embedder.embed(samples)
    cos = float(np.min(np.sum(ref * got, axis=1)))
    print(f"  embedding parity (onnx vs sentence-transformers): min cosine = {cos:.5f}")
    assert cos > 0.99, f"embedder mismatch (cos={cos})"


def embed_all(embedder, texts, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        out.append(embedder.embed(texts[i:i + batch]))
        if (i // batch) % 20 == 0:
            print(f"    embedded {min(i + batch, len(texts))}/{len(texts)}")
    return np.vstack(out)


def embed_cached(embedder, texts, tag: str):
    """Embed texts, caching to models/_cache_X_<tag>.npy (so the 2 heads of an embedder
    re-train in seconds)."""
    cache = MODELS / f"_cache_X_{tag}.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(texts):
        print(f"  [cache] {cache.name}")
        return np.load(cache)
    X = embed_all(embedder, texts).astype(np.float32)
    np.save(cache, X)
    return X


# --- heads ---

def train_rf(X, y, out_path):
    """RandomForest head -> ONNX (skl2onnx). depth/leaf caps keep the ensemble small."""
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=8,
                                n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print(f"    RF train accuracy = {rf.score(X, y):.4f} (honest eval = consumer run)")
    onx = to_onnx(rf, initial_types=[("embedding", FloatTensorType([None, X.shape[1]]))],
                  options={id(rf): {"zipmap": False}}, target_opset=14)
    Path(out_path).write_bytes(onx.SerializeToString())
    print(f"[done] wrote {out_path} (dim {X.shape[1]})")


def train_mlp(X, y, K, out_path, epochs=25):
    """MLP head (dim->256->K) -> ONNX (torch)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    dim = X.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(256, K))

        def forward(self, x):
            return self.net(x)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        tr = float((model(torch.from_numpy(X)).argmax(1).numpy() == y).mean())
    print(f"    MLP train accuracy = {tr:.4f} (honest eval = consumer run)")
    torch.onnx.export(model, torch.zeros(1, dim), str(out_path),
                      input_names=["embedding"], output_names=["logits"],
                      dynamic_axes={"embedding": {0: "b"}, "logits": {0: "b"}},
                      opset_version=14, dynamo=False)
    print(f"[done] wrote {out_path} (dim {dim}, {K} classes)")
