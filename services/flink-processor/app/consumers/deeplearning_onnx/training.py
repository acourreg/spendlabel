"""deeplearning_onnx training — MLP on MiniLM embeddings -> mlp.onnx (offline).

Run from the app dir (services/flink-processor/app) with the training venv:

    python -m consumers.deeplearning_onnx.training

Uses the shared embeddings/split from consumers.training_common, trains a small
MLP (384 -> 256 -> K), and exports it to ONNX.
"""

from __future__ import annotations

from consumers.training_common import prepare_dataset, MODELS


def main() -> None:
    Xtr, Xte, ytr, yte, classes = prepare_dataset()

    print("[6] MLP (deeplearning_onnx)...")
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    K = len(classes)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(256, K))

        def forward(self, x):
            return self.net(x)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    for _ in range(25):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    acc = float((pred == yte).mean())
    print(f"    MLP test accuracy = {acc:.4f}")

    out = MODELS / "mlp.onnx"
    torch.onnx.export(model, torch.zeros(1, 384), str(out),
                      input_names=["embedding"], output_names=["logits"],
                      dynamic_axes={"embedding": {0: "b"}, "logits": {0: "b"}},
                      opset_version=14, dynamo=False)
    print(f"[done] wrote {out} (MLP test acc = {acc:.4f}, {K} classes)")


if __name__ == "__main__":
    main()
