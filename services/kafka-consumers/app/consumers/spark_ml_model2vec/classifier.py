"""spark_ml_model2vec classifier — classification only.

Classical ML head: Model2Vec static embeddings (no transformer) feed a trained
RandomForest (``rf_m2v.onnx``). Same RF recipe as the HuggingFace variant, but on
Model2Vec vectors — ~100x cheaper embedding. No Kafka, no DB (the shared
``consumers/consumer_loop`` drives I/O). Trained offline by ``training.py``.
"""

from __future__ import annotations

import json

from consumers.embedding import MODELS_DIR, make_session
from consumers.embedding_model2vec import Model2VecEmbedder

PARADIGM = "spark_ml_model2vec"
HEAD_FILE = "rf_m2v.onnx"


class Classifier:
    """Model2Vec embeddings -> RandomForest -> 2-digit CPV code."""

    def __init__(self) -> None:
        self.labels = json.loads((MODELS_DIR / "labels.json").read_text())
        self.K = len(self.labels)
        self.embedder = Model2VecEmbedder()
        self.head = make_session(MODELS_DIR / HEAD_FILE)
        self.head_in = self.head.get_inputs()[0].name

    def classify(self, items: list[tuple[str | None, str | None]]) -> list[str]:
        texts = [((it[0] or "") + " " + (it[1] or "")).strip() for it in items]
        embs = self.embedder.embed(texts)
        outs = self.head.run(None, {self.head_in: embs})
        scores = next(o for o in outs if getattr(o, "ndim", 0) == 2 and o.shape[1] == self.K)
        return [self.labels[int(i)] for i in scores.argmax(axis=1)]
