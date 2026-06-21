"""Shared MiniLM sentence-embedder for the ML paradigms (spark_ml, deeplearning_onnx).

Runs on onnxruntime + tokenizers only (no torch), so the consumer image stays
slim. Reproduces sentence-transformers/all-MiniLM-L6-v2:
    WordPiece tokenize -> transformer (ONNX) -> mean-pool over tokens
    (attention-masked) -> L2 normalize -> 384-d vector.

The exact same class is used at training time, so the embeddings the heads are
trained on match the embeddings seen at inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODELS_DIR = Path(__file__).resolve().parent / "models"
MAX_LEN = 256
EMB_DIM = 384


def make_session(path) -> ort.InferenceSession:
    """ONNX session tuned for a small, memory-constrained container.

    Disables the CPU memory arena (which pre-allocates and holds large pools) and
    caps intra-op threads — keeps onnxruntime from OOM-killing the consumer on a
    Docker Desktop VM with only a few GB shared across kafka/postgres/consumers.
    """
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class Embedder:
    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        models_dir = Path(models_dir)
        self.session = make_session(models_dir / "minilm.onnx")
        self.tok = Tokenizer.from_file(str(models_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.enable_padding()  # pad to the longest sequence in each batch
        self._inputs = {i.name for i in self.session.get_inputs()}

    def embed(self, texts) -> np.ndarray:
        """Embed a list of strings -> float32 array [N, 384]."""
        if isinstance(texts, str):
            texts = [texts]
        encs = self.tok.encode_batch([t or "" for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        last_hidden = self.session.run(None, feeds)[0]  # [B, T, 384]
        m = mask[..., None].astype(np.float32)
        summed = (last_hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        emb = summed / counts                                   # mean pool
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)  # L2
        return emb.astype(np.float32)
