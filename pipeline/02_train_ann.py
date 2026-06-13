"""
02_train_ann.py
===============
Trains a deep MLP (ANN) using PyTorch on the pre-saved split.
Collects all IEEE-required metrics + learning curves.

Usage:
    python 02_train_ann.py --split_dir ./split_data --output_dir ./results/ann

Outputs:
    metrics.json
    confusion_matrix.png
    roc_curves.png
    learning_curves.png      ← train/val loss and accuracy per epoch
    model.pt

Requirements:
    pip install torch scikit-learn matplotlib numpy
"""

import argparse
import json
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─── Model Definition ──────────────────────────────────────────────────────────

class SpeciesMLP(nn.Module):
    """
    4-layer MLP with BatchNorm + Dropout.
    Architecture deliberately separate from TabTransformer to
    represent a classical deep-learning baseline.
    """
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims=(512, 256, 128, 64),
                 dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Metric helpers ────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_prob, k):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss",  linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("ANN — Learning Curves (Loss)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc",  linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("ANN — Learning Curves (Accuracy)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title("ANN (MLP) — Confusion Matrix (90 Species)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_roc_curves(y_true, y_prob, n_classes, out_path, top_n=10):
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_macro = np.linspace(0, 1, 200)
    tpr_list = []
    aucs = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        aucs.append((auc_i, i, fpr_i, tpr_i))
        tpr_list.append(np.interp(fpr_macro, fpr_i, tpr_i))

    fig, ax = plt.subplots(figsize=(10, 8))
    for auc_i, i, fpr_i, tpr_i in sorted(aucs, key=lambda x: -x[0])[:top_n]:
        ax.plot(fpr_i, tpr_i, lw=0.8, alpha=0.6, label=f"Class {i} (AUC={auc_i:.2f})")
    mean_tpr = np.mean(tpr_list, axis=0)
    macro_auc = np.trapz(mean_tpr, fpr_macro)
    ax.plot(fpr_macro, mean_tpr, "k--", lw=2.5, label=f"Macro Avg (AUC={macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ANN (MLP) — ROC Curves")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


# ─── Training Loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
    return total_loss / total, correct / total


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load split
    X_train = np.load(os.path.join(args.split_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.split_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.split_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.split_dir, "y_test.npy"))
    with open(os.path.join(args.split_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    n_classes  = len(le.classes_)
    input_dim  = X_train.shape[1]
    print(f"[INFO] input_dim={input_dim}  n_classes={n_classes}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Model, optimizer, scheduler
    model     = SpeciesMLP(input_dim, n_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("[INFO] Training ANN (MLP)...")
    t0 = time.time()
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, test_loader,  criterion, optimizer, device, train=False)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"TrainLoss={tr_loss:.4f}  ValLoss={vl_loss:.4f}  "
                  f"TrainAcc={tr_acc:.4f}  ValAcc={vl_acc:.4f}")

    train_time = time.time() - t0
    print(f"[INFO] Training time: {train_time:.2f}s")

    # Load best weights
    model.load_state_dict(best_state)

    # ── Inference + full metric collection ────────────────────────────────────
    model.eval()
    all_logits, all_labels = [], []
    t1 = time.time()
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(yb)
    inference_time = (time.time() - t1) / len(X_test) * 1000

    logits_np = torch.cat(all_logits).numpy()
    y_prob    = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    y_pred    = np.argmax(y_prob, axis=1)
    y_true    = np.concatenate([b.numpy() for b in all_labels])

    report  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc     = accuracy_score(y_true, y_pred)
    top3    = top_k_accuracy(y_true, y_prob, 3)
    top5    = top_k_accuracy(y_true, y_prob, 5)
    y_bin   = label_binarize(y_true, classes=list(range(n_classes)))
    roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

    metrics = {
        "model": "ANN_MLP",
        "accuracy":           round(acc,  4),
        "precision_macro":    round(report["macro avg"]["precision"], 4),
        "recall_macro":       round(report["macro avg"]["recall"],    4),
        "f1_macro":           round(report["macro avg"]["f1-score"],  4),
        "top3_accuracy":      round(top3,    4),
        "top5_accuracy":      round(top5,    4),
        "roc_auc_macro_ovr":  round(roc_auc, 4),
        "train_time_sec":     round(train_time, 2),
        "inference_time_ms_per_sample": round(inference_time, 4),
        "best_val_acc":       round(best_val_acc, 4),
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "lr":                 args.lr,
        "architecture":       "512-256-128-64 + BN + Dropout(0.3)",
    }

    print("\n─── ANN Results ───────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_learning_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(args.output_dir, "learning_curves.png")
    )
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, n_classes, os.path.join(args.output_dir, "roc_curves.png"))

    # Save model
    torch.save(best_state, os.path.join(args.output_dir, "model.pt"))
    print(f"\n[DONE] All ANN outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",  default="./split_data")
    parser.add_argument("--output_dir", default="./results/ann")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    main(parser.parse_args())
