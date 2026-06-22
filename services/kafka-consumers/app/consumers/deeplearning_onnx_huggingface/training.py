"""deeplearning_onnx_huggingface training — MLP on MiniLM (HuggingFace) embeddings.

    python -m consumers.deeplearning_onnx_huggingface.training   # -> models/mlp.onnx
"""

from __future__ import annotations

from consumers.training_common import (load_train_slice, embed_cached, export_minilm,
                                        parity_check, train_mlp, MODELS)


def main() -> None:
    texts, classes, y = load_train_slice()
    export_minilm()
    from consumers.embedding import Embedder
    emb = Embedder(MODELS)
    parity_check(emb)
    X = embed_cached(emb, texts, "hf")
    train_mlp(X, y, len(classes), MODELS / "mlp.onnx")


if __name__ == "__main__":
    main()
