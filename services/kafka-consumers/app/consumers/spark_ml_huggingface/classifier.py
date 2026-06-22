"""spark_ml classifier — classification only.

Classical ML head: the shared MiniLM sentence embedder feeds a trained
RandomForest (``rf.onnx``), run via onnxruntime. Maps a batch of texts to 2-digit
CPV codes and nothing else — no Kafka, no DB (the shared ``consumers/consumer_loop``
drives I/O). The model is trained offline by ``training.py``.
"""

from __future__ import annotations

import json

from consumers.embedding import Embedder, MODELS_DIR, make_session

PARADIGM = "spark_ml_huggingface"
HEAD_FILE = "rf.onnx"


class Classifier:
    """MiniLM embeddings -> RandomForest -> 2-digit CPV code."""

    def __init__(self) -> None:
        self.labels = json.loads((MODELS_DIR / "labels.json").read_text())
        self.K = len(self.labels)
        self.embedder = Embedder()
        self.head = make_session(MODELS_DIR / HEAD_FILE)
        self.head_in = self.head.get_inputs()[0].name

    def classify(self, items: list[tuple[str | None, str | None]]) -> list[str]:
        texts = [((it[0] or "") + " " + (it[1] or "")).strip() for it in items]
        embs = self.embedder.embed(texts)
        outs = self.head.run(None, {self.head_in: embs})
        # skl2onnx RandomForest returns [label[B], probabilities[B, K]] — take the
        # [B, K] probability array and argmax it (always predicts, never abstains).
        scores = next(o for o in outs if getattr(o, "ndim", 0) == 2 and o.shape[1] == self.K)
        return [self.labels[int(i)] for i in scores.argmax(axis=1)]


if __name__ == "__main__":
    clf = Classifier()
    print(clf.classify([("road resurfacing and drainage works", ""), ("NHS community nursing", "")]))
