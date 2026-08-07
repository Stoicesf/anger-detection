"""Train the DS-CNN in PyTorch on the RTX 4060 (GPU), then export ONNX."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from anger_detection.common import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TIME_FRAMES,
    ensure_dirs,
)
from anger_detection.models import DSCNN, SEED, set_seed

N_MELS = 40


class AudioDataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X = X  # (N, 98, 40, 1) float32
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i, :, :, 0]  # (98, 40)
        if self.train:
            pad = 16
            x = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
            off = np.random.randint(0, 2 * pad + 1)
            x = x[off : off + TIME_FRAMES]
            x = x + np.random.normal(0, 0.002, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x).reshape(1, TIME_FRAMES, N_MELS)), int(self.y[i])


def main() -> None:
    ensure_dirs()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    file_ids = np.load(PROCESSED_DIR / "file_ids.npy")
    print(f"Loaded X={X.shape}")

    unique_ids = np.unique(file_ids)
    file_label = np.array([y[np.where(file_ids == fid)[0][0]] for fid in unique_ids])
    ids_train, ids_tmp, _, _ = train_test_split(
        unique_ids, file_label, test_size=0.2, stratify=file_label, random_state=SEED
    )
    label_map = dict(zip(unique_ids, file_label))
    y_tmp = np.array([label_map[fid] for fid in ids_tmp])
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    train_mask = np.isin(file_ids, ids_train)
    val_mask = np.isin(file_ids, ids_val)
    test_mask = np.isin(file_ids, ids_test)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"files: train={len(ids_train)} val={len(ids_val)} test={len(ids_test)}")
    print(f"crops: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights = np.clip(weights, None, 10.0)
    print("class weights:", weights.round(3))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    train_ds = AudioDataset(X_train, y_train, train=True)
    val_ds = AudioDataset(X_val, y_val, train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = DSCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-5
    )

    best_val = 0.0
    best_state = None
    history = []
    patience = 15
    wait = 0
    t0 = time.time()
    for epoch in range(60):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        train_acc = correct / total

        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_loss += criterion(out, yb).item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += len(yb)
        val_acc = v_correct / v_total
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
                        "train_loss": loss_sum / total, "val_loss": v_loss / v_total, "lr": lr})
        print(f"epoch {epoch:02d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.2e} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), MODELS_DIR / "dscnn_torch_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at epoch {epoch}")
                break

    print(f"Training finished in {(time.time()-t0)/60:.1f} min, best val_acc={best_val:.4f}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "dscnn_torch_final.pt")

    # test evaluation
    test_ds = AudioDataset(X_test, y_test, train=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    model.eval()
    t_correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            t_correct += (model(xb).argmax(1) == yb).sum().item()
    test_acc = t_correct / len(X_test)
    print(f"Test accuracy: {test_acc:.4f}")

    with open(OUTPUT_DIR / "history_torch.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "best_val_accuracy": float(best_val),
                "epochs_run": len(history),
                "device": str(device),
                "class_names": CLASS_NAMES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # export ONNX (with softmax for a probability output)
    model.eval()
    export_model = nn.Sequential(model, nn.Softmax(dim=1)).to("cpu")
    dummy = torch.randn(1, 1, TIME_FRAMES, N_MELS)
    torch.onnx.export(
        export_model,
        dummy,
        str(MODELS_DIR / "dscnn.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )
    print("ONNX exported ->", MODELS_DIR / "dscnn.onnx")


if __name__ == "__main__":
    main()
