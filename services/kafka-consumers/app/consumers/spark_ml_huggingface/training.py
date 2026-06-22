"""spark_ml_huggingface training — RandomForest on MiniLM (HuggingFace) embeddings.

    python -m consumers.spark_ml_huggingface.training   # -> models/rf.onnx
"""

from __future__ import annotations

from consumers.training_common import (load_train_slice, embed_cached, export_minilm,
                                        parity_check, train_rf, MODELS)


def main() -> None:
    texts, classes, y = load_train_slice()
    export_minilm()
    from consumers.embedding import Embedder
    emb = Embedder(MODELS)
    parity_check(emb)
    X = embed_cached(emb, texts, "hf")
    train_rf(X, y, MODELS / "rf.onnx")


if __name__ == "__main__":
    main()
