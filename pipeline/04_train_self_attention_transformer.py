"""
04_train_self_attention_transformer.py
=======================================
12-Head Self-Attention Transformer for 90-species tabular classification.
This is the proposed novel architecture — the key differentiator
from TabTransformer is:
  - 12 attention heads (vs 8 in TabTransformer)
  - Deeper stack (6 layers vs 4)
  - Larger d_model (128 vs 64)
  - Learnable CLS token (classification aggregation, BERT-style)
  - Pre-LayerNorm (more stable training than post-LN)

Usage:
    python 04_train_self_attention_transformer.py \
        --split_dir ./split_data \
        --output_dir ./results/self_attention_12head

Outputs:
    metrics.json
    confusion_matrix.png
    roc_curves.png
    learning_curves.png
    attention_heatmap.png    ← Layer 1 mean attention (12 heads averaged)
    attention_heatmap_per_head.png ← All 12 heads in a grid
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
import math

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


# ─── 12-Head Self-Attention Transformer ────────────────────────────────────────

class PreLNTransformerLayer(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block.
    Pre-LN (LN before attention) is empirically more stable than post-LN
    for deep models. Reference: Liu et al. (2020).
    """
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        # Self-attention sublayer (Pre-LN)
        residual = x
        x_ln = self.ln1(x)
        attn_out, attn_weights = self.attn(
            x_ln, x_ln, x_ln, need_weights=True, average_attn_weights=False
        )
        x = residual + attn_out

        # FFN sublayer (Pre-LN)
        x = x + self.ff(self.ln2(x))

        if return_attn:
            return x, attn_weights   # attn_weights: (B, n_heads, seq, seq)
        return x


class SelfAttentionTransformer12Head(nn.Module):
    """
    12-Head Self-Attention Transformer for tabular data.
    Key design choices:
      - CLS token for global classification (BERT-style)
      - Per-feature linear projections to d_model
      - Pre-LayerNorm for training stability
      - 12 attention heads, 6 layers, d_model=128
    """
    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        d_model:    int   = 128,
        n_heads:    int   = 12,
        n_layers:   int   = 6,
        dim_ff:     int   = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model    = d_model
        self.n_features = n_features

        # Project each feature scalar → d_model vector
        self.feature_proj = nn.Linear(n_features, n_features * d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Learnable positional embeddings (n_features + 1 for CLS)
        self.pos_emb = nn.Parameter(torch.randn(1, n_features + 1, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)

        # Classifier head on CLS token
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        B = x.size(0)

        # Feature embeddings: (B, n_features, d_model)
        feat = self.feature_proj(x).view(B, self.n_features, self.d_model)

        # Prepend CLS token: (B, n_features+1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, feat], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_emb

        # Transformer layers
        attn_maps = []
        for layer in self.layers:
            if return_attn:
                tokens, attn_w = layer(tokens, return_attn=True)
                attn_maps.append(attn_w)
            else:
                tokens = layer(tokens)

        tokens = self.ln_final(tokens)

        # Use CLS token for classification
        cls_out = tokens[:, 0, :]          # (B, d_model)
        logits  = self.classifier(cls_out)

        if return_attn:
            return logits, attn_maps
        return logits


# ─── Attention visualization ───────────────────────────────────────────────────

def get_attention_maps(model, X_sample, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_sample, dtype=torch.float32).to(device)
        _, attn_maps = model(x, return_attn=True)
    # attn_maps: list of (B, n_heads, seq_len, seq_len) tensors
    # Average over batch → (n_heads, seq_len, seq_len)
    return [m.mean(dim=0).cpu().numpy() for m in attn_maps]


def plot_attention_heatmap_mean(attn_layer0, feature_names, out_path):
    """Mean attention over 12 heads for Layer 1."""
    mean_attn = attn_layer0.mean(axis=0)   # (seq_len, seq_len)
    n = mean_attn.shape[0]
    labels = ["[CLS]"] + (feature_names[:n-1] if feature_names else
                           [f"F{i}" for i in range(n-1)])

    fig, ax = plt.subplots(figsize=(max(12, n // 2), max(10, n // 2)))
    im = ax.imshow(mean_attn, cmap="plasma", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("12-Head Self-Attention Transformer\n"
                 "Layer 1 — Mean Attention Heatmap (12 Heads Averaged)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_attention_per_head(attn_layer0, out_path, max_heads=12):
    """Grid of all 12 attention heads for Layer 1."""
    n_heads = min(attn_layer0.shape[0], max_heads)
    cols = 4
    rows = math.ceil(n_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for h in range(n_heads):
        im = axes[h].imshow(attn_layer0[h], cmap="plasma", aspect="auto")
        axes[h].set_title(f"Head {h+1}", fontsize=9)
        axes[h].axis("off")

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle("12-Head Self-Attention Transformer — Layer 1, Per-Head Attention Maps",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ─── Shared helpers ────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_prob, k):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


def plot_learning_curves(tl, vl, ta, va, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tl) + 1)
    ax1.plot(ep, tl, label="Train"); ax1.plot(ep, vl, "--", label="Val")
    ax1.set_title("12-Head Transformer — Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, ta, label="Train"); ax2.plot(ep, va, "--", label="Val")
    ax2.set_title("12-Head Transformer — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[SAVED] {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title("12-Head Self-Attention Transformer — Confusion Matrix (90 Species)")
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
    ax.set_title("12-Head Self-Attention Transformer — ROC Curves")
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

    n_classes  = len(le.classes_)
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SelfAttentionTransformer12Head(
        n_features=n_features,
        n_classes=n_classes,
        d_model=60,    # 60 = 12 × 5, divisible by n_heads=12, faster on CPU
        n_heads=12,
        n_layers=4,
        dim_ff=256,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 12-Head Transformer trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup + cosine annealing
    def lr_lambda(step):
        warmup = args.warmup_epochs
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_state = 0.0, None

    print("[INFO] Training 12-Head Self-Attention Transformer...")
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
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs}  lr={lr_now:.2e}  "
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
        "model": "12-Head_Self-Attention_Transformer",
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
        "d_model": 60, "n_heads": 12, "n_layers": 4, "dim_ff": 256,
        "n_params": n_params,
        "label_smoothing": 0.1,
        "warmup_epochs": args.warmup_epochs,
    }

    print("\n─── 12-Head Transformer Results ───────────────────")
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

    # Attention maps
    sample_size = min(32, len(X_test))
    attn_maps = get_attention_maps(model, X_test[:sample_size], device)
    if attn_maps:
        attn_l0 = attn_maps[0]   # (n_heads, seq_len, seq_len)
        plot_attention_heatmap_mean(
            attn_l0, ["[CLS]"] + feature_names,
            os.path.join(args.output_dir, "attention_heatmap.png")
        )
        plot_attention_per_head(
            attn_l0,
            os.path.join(args.output_dir, "attention_heatmap_per_head.png")
        )

    torch.save(best_state, os.path.join(args.output_dir, "model.pt"))
    print(f"\n[DONE] All 12-Head Transformer outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",     default="./split_data")
    parser.add_argument("--output_dir",    default="./results/self_attention_12head")
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int,   default=10)
    main(parser.parse_args())
