"""Model2Vec static-embedding embedder (no transformer forward).

A drop-in alternative to the MiniLM ``Embedder``: tokenize -> per-token vector
lookup -> mean pool. ~100-1000x faster on CPU and O(sequence) instead of O(seq^2),
at some quality cost. The static model is baked under models/model2vec/ by training
(``training_common.save_model2vec``). Same ``embed(texts) -> [N, D]`` contract.
"""

from __future__ import annotations

from pathlib import Path

from model2vec import StaticModel

M2V_DIR = Path(__file__).resolve().parent / "models" / "model2vec"


class Model2VecEmbedder:
    def __init__(self, model_dir: Path | str = M2V_DIR) -> None:
        self.model = StaticModel.from_pretrained(str(model_dir))

    def embed(self, texts):
        return self.model.encode([t or "" for t in texts]).astype("float32")
