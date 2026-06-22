"""deeplearning_onnx_model2vec training — MLP on Model2Vec static embeddings.

    python -m consumers.deeplearning_onnx_model2vec.training   # -> models/mlp_m2v.onnx
"""

from __future__ import annotations

from consumers.training_common import (load_train_slice, embed_cached, save_model2vec,
                                        train_mlp, MODELS)


def main() -> None:
    texts, classes, y = load_train_slice()
    save_model2vec()
    from consumers.embedding_model2vec import Model2VecEmbedder
    X = embed_cached(Model2VecEmbedder(), texts, "m2v")
    train_mlp(X, y, len(classes), MODELS / "mlp_m2v.onnx")


if __name__ == "__main__":
    main()
