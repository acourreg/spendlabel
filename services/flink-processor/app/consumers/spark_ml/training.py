"""spark_ml training — RandomForest on MiniLM embeddings -> rf.onnx (offline).

Run from the app dir (services/flink-processor/app) with the training venv:

    python -m consumers.spark_ml.training

Uses the shared embeddings/split from consumers.training_common, trains a
RandomForest, and exports it to ONNX via skl2onnx. depth/leaf caps keep the
tree-ensemble small (unbounded -> ~1.2GB ONNX).
"""

from __future__ import annotations

from consumers.training_common import prepare_dataset, MODELS


def main() -> None:
    Xtr, Xte, ytr, yte, classes = prepare_dataset()

    print("[6] RandomForest (spark_ml)...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=8,
                                n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    acc = float(rf.score(Xte, yte))
    print(f"    RF test accuracy = {acc:.4f}")

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    onx = to_onnx(rf, initial_types=[("embedding", FloatTensorType([None, 384]))],
                  options={id(rf): {"zipmap": False}}, target_opset=14)
    out = MODELS / "rf.onnx"
    out.write_bytes(onx.SerializeToString())
    print(f"[done] wrote {out} (RF test acc = {acc:.4f}, {len(classes)} classes)")


if __name__ == "__main__":
    main()
