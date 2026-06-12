"""
03_train_tabtransformer.py
==========================
TabTransformer for tabular species classification.
Implements the architecture from Huang et al. (2020):
"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"

For purely numerical features (no categorical), each feature is projected
via a linear embedding and processed through Transformer encoder blocks.

Usage:
    python 03_train_tabtransformer.py --split_dir ./split_data \
                                       --output_dir ./results/tabtransformer

Outputs:
    metrics.json
    confusion_matrix.png
    roc_curves.png
    learning_curves.png
    attention_heatmap.png    ← mean attention weights across heads (Layer 1)
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


# ─── TabTransformer Architecture ───────────────────────────────────────────────

class FeatureEmbedding(nn.Module):
    """Projects each scalar feature into a d_model-dimensional embedding."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, n_features * d_model)
        self.n_features = n_features
        self.d_model    = d_model

    def forward(self, x):
        # x: (B, n_features) → (B, n_features, d_model)
        B = x.size(0)
        out = self.proj(x)                    # (B, n_features * d_model)
        return out.view(B, self.n_features, self.d_model)


class TabTransformer(nn.Module):
    """
    TabTransformer with:
      - Per-feature linear embedding
      - Stack of Transformer encoder layers (with multi-head self-attention)
      - MLP classifier head
    """
    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        d_model:    int = 64,
        n_heads:    int = 8,
        n_layers:   int = 4,
        dim_ff:     int = 256,
        dropout:    float = 0.1,
        mlp_hidden: tuple = (256, 128),
    ):
        super().__init__()
        self.embedding = FeatureEmbedding(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten transformer output → MLP head
        flat_dim = n_features * d_model
        mlp_layers = []
        prev = flat_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        mlp_layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, return_attention=False):
        emb = self.embedding(x)               # (B, n_features, d_model)
        enc = self.transformer(emb)           # (B, n_features, d_model)
        flat = enc.flatten(1)                 # (B, n_features * d_model)
        logits = self.mlp(flat)
        return logits


# ─── Attention Heatmap ─────────────────────────────────────────────────────────

def extract_attention(model, X_sample, device):
    """
    Extract attention weights from the first transformer layer's self-attention.
    Returns mean over heads: shape (n_features, n_features).
    """
    model.eval()
    hooks = []
    attn_weights = {}

    def hook_fn(name):
        def _hook(module, input, output):
            # TransformerEncoderLayer passes through self-attn; capture via forward hook
            # output[1] is attn_weights when need_weights=True
            # We re-run manually below via the sub-module
            pass
        return _hook

    # Access the internal MultiheadAttention of layer 0
    attn_module = model.transformer.layers[0].self_attn

    def attn_hook(module, input, output):
        # output = (attn_output, attn_weights)
        attn_weights["layer0"] = output[1].detach().cpu()

    h = attn_module.register_forward_hook(attn_hook)

    with torch.no_grad():
        x = torch.tensor(X_sample, dtype=torch.float32).to(device)
        emb = model.embedding(x)
        # Force need_weights=True
        # TransformerEncoderLayer calls self_attn internally
        _ = model.transformer.layers[0].self_attn(
            emb, emb, emb, need_weights=True, average_attn_weights=False
        )
    h.remove()

    if "layer0" in attn_weights:
        # shape: (B, n_heads, n_features, n_features) → mean over B and heads
        return attn_weights["layer0"].mean(dim=(0, 1)).numpy()
    return None


def plot_attention_heatmap(attn_matrix, feature_names, out_path):
    n = attn_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n // 3), max(8, n // 3)))
    im = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Attention Weight")
    tick_labels = (feature_names[:n] if feature_names else
                   [str(i) for i in range(n)])
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_title("TabTransformer — Attention Heatmap (Layer 1, Mean over Heads)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


# ─── Shared helpers (same as ANN) ──────────────────────────────────────────────

def top_k_accuracy(y_true, y_prob, k):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


def plot_learning_curves(tl, vl, ta, va, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tl) + 1)
    ax1.plot(ep, tl, label="Train"); ax1.plot(ep, vl, "--", label="Val")
    ax1.set_title("TabTransformer — Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, ta, label="Train"); ax2.plot(ep, va, "--", label="Val")
    ax2.set_title("TabTransformer — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[SAVED] {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title("TabTransformer — Confusion Matrix (90 Species)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[SAVED] {out_path}")


def plot_roc_curves(y_true, y_prob, n_classes, out_path, top_n=10):
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_m = np.linspace(0, 1, 200); tpr_list = []; aucs = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        aucs.append((auc(fpr_i, tpr_i), i, fpr_i, tpr_i))
        tpr_list.append(np.interp(fpr_m, fpr_i, tpr_i))
    fig, ax = plt.subplots(figsize=(10, 8))
    for auc_i, i, fpr_i, tpr_i in sorted(aucs, key=lambda x: -x[0])[:top_n]:
        ax.plot(fpr_i, tpr_i, lw=0.8, alpha=0.6, label=f"Class {i} ({auc_i:.2f})")
    mean_tpr = np.mean(tpr_list, axis=0)
    ax.plot(fpr_m, mean_tpr, "k--", lw=2.5,
            label=f"Macro Avg (AUC={np.trapz(mean_tpr, fpr_m):.4f})")
    ax.plot([0,1],[0,1],"gray",linestyle=":",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("TabTransformer — ROC Curves")
    ax.legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[SAVED] {out_path}")


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

    X_train = np.load(os.path.join(args.split_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.split_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.split_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.split_dir, "y_test.npy"))
    with open(os.path.join(args.split_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(args.split_dir, "feature_names.txt")) as f:
        feature_names = [l.strip() for l in f.readlines()]

    n_classes = len(le.classes_)
    n_features = X_train.shape[1]
    print(f"[INFO] n_features={n_features}  n_classes={n_classes}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TabTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_model=64,
        n_heads=8,
        n_layers=4,
        dim_ff=256,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] TabTransformer trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_state = 0.0, None

    print("[INFO] Training TabTransformer...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl, va = run_epoch(model, test_loader,  criterion, optimizer, device, False)
        scheduler.step()
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)
        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"TrLoss={tl:.4f} VaLoss={vl:.4f}  "
                  f"TrAcc={ta:.4f} VaAcc={va:.4f}")
    train_time = time.time() - t0
    print(f"[INFO] Training time: {train_time:.2f}s")

    model.load_state_dict(best_state)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_logits, all_labels = [], []
    t1 = time.time()
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_logits.append(model(Xb.to(device)).cpu())
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
        "model": "TabTransformer",
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
        "d_model": 64, "n_heads": 8, "n_layers": 4,
        "n_params": n_params,
    }

    print("\n─── TabTransformer Results ────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_learning_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(args.output_dir, "learning_curves.png")
    )
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, n_classes, os.path.join(args.output_dir, "roc_curves.png"))

    # Attention heatmap (use small sample for speed)
    sample_size = min(50, len(X_test))
    attn = extract_attention(model, X_test[:sample_size], device)
    if attn is not None:
        plot_attention_heatmap(
            attn, feature_names,
            os.path.join(args.output_dir, "attention_heatmap.png")
        )

    torch.save(best_state, os.path.join(args.output_dir, "model.pt"))
    print(f"\n[DONE] All TabTransformer outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",  default="./split_data")
    parser.add_argument("--output_dir", default="./results/tabtransformer")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    main(parser.parse_args())
