"""Shared offline training utilities (tronc commun) for the embedding ML paradigms.

Reads the dataset, exports the shared MiniLM encoder to ONNX, embeds every contract
with the SAME ``Embedder`` used at inference (so train == inference), and returns a
cached train/test split. Each paradigm's ``training.py`` calls ``prepare_dataset()``
then trains its own head:

    python -m consumers.spark_ml.training          # RandomForest -> rf.onnx
    python -m consumers.deeplearning_onnx.training # MLP          -> mlp.onnx

This step DOES use the CPV ground truth (it's training); the consumers never do.
Requires the offline training deps (consumers/requirements-training.txt), NOT shipped
in the slim inference container.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
REPO = Path(__file__).resolve().parents[4]
DATASET = REPO / "data" / "raw" / "dataset.tar.gz"
MIN_COUNT = 50          # drop CPV classes with fewer training samples
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def export_minilm():
    """Export the shared MiniLM encoder + its tokenizer to ONNX under models/."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    MODELS.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

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
    tok.save_pretrained(MODELS)  # writes tokenizer.json (fast tokenizer)
    print("  exported minilm.onnx + tokenizer.json")


def parity_check(embedder):
    from sentence_transformers import SentenceTransformer
    samples = ["road resurfacing and drainage works", "NHS community nursing framework",
               "cloud hosting and cyber security support", "school catering services"]
    ref = SentenceTransformer(MODEL_NAME).encode(samples, normalize_embeddings=True)
    got = embedder.embed(samples)
    cos = float(np.min(np.sum(ref * got, axis=1)))  # both L2-normalized -> dot = cosine
    print(f"  embedding parity (onnx vs sentence-transformers): min cosine = {cos:.5f}")
    assert cos > 0.99, f"embedder mismatch (cos={cos})"


def embed_all(embedder, texts, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        out.append(embedder.embed(texts[i:i + batch]))
        if (i // batch) % 20 == 0:
            print(f"    embedded {min(i + batch, len(texts))}/{len(texts)}")
    return np.vstack(out)


def prepare_dataset():
    """Read + embed the dataset and return (Xtr, Xte, ytr, yte, classes).

    Exports MiniLM, writes labels.json, and caches embeddings in models/_cache_X.npy
    so re-running a head trains in seconds. Both heads share this exact split.
    """
    print("[1] reading dataset...")
    texts, labels = read_dataset()
    print(f"    {len(texts)} labelled records")

    counts = Counter(labels)
    keep = {c for c, n in counts.items() if n >= MIN_COUNT}
    pairs = [(t, l) for t, l in zip(texts, labels) if l in keep]
    texts = [t for t, _ in pairs]
    classes = sorted(keep)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for _, l in pairs], dtype=np.int64)
    print(f"    kept {len(classes)} CPV classes (>= {MIN_COUNT} samples), {len(texts)} records")

    print("[2] exporting MiniLM encoder to ONNX...")
    export_minilm()
    from consumers.embedding import Embedder
    embedder = Embedder(MODELS)

    print("[3] parity check...")
    parity_check(embedder)

    cache_x = MODELS / "_cache_X.npy"
    if cache_x.exists() and np.load(cache_x, mmap_mode="r").shape[0] == len(texts):
        print("[4] loading cached embeddings...")
        X = np.load(cache_x)
    else:
        print("[4] embedding all records with the ONNX embedder...")
        X = embed_all(embedder, texts).astype(np.float32)
        np.save(cache_x, X)

    (MODELS / "labels.json").write_text(json.dumps(classes))

    print("[5] train/test split...")
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    train={len(Xtr)} test={len(Xte)} classes={len(classes)}")
    return Xtr, Xte, ytr, yte, classes
