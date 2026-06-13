"""
01_train_xgboost.py
===================
Trains XGBoost on the pre-saved train/test split and collects ALL metrics
required for the IEEE benchmark table.

Usage:
    python 01_train_xgboost.py --split_dir ./split_data --output_dir ./results/xgboost

Outputs:
    metrics.json          ← accuracy, precision, recall, F1, top-3/5 acc, ROC-AUC,
                            training time, inference time
    confusion_matrix.png
    roc_curves.png
    shap_summary.png
    shap_importance.png
    model.pkl

Requirements:
    pip install xgboost scikit-learn shap matplotlib numpy
"""

import argparse
import json
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ─── Metric helpers ────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_prob, k):
    """Fraction of samples where true label is in top-k predicted."""
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title("XGBoost — Confusion Matrix (90 Species)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_roc_curves(y_true, y_prob, n_classes, out_path, top_n=10):
    """Plot macro-averaged ROC + top-N per-class curves."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_macro = np.linspace(0, 1, 200)
    tpr_macro_list = []

    fig, ax = plt.subplots(figsize=(10, 8))
    from sklearn.metrics import roc_curve, auc

    # Plot top_n class curves
    aucs = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        aucs.append((auc_i, i, fpr_i, tpr_i))
        tpr_macro_list.append(np.interp(fpr_macro, fpr_i, tpr_i))

    aucs_sorted = sorted(aucs, key=lambda x: x[0], reverse=True)
    for auc_i, i, fpr_i, tpr_i in aucs_sorted[:top_n]:
        ax.plot(fpr_i, tpr_i, lw=0.8, alpha=0.6, label=f"Class {i} (AUC={auc_i:.2f})")

    # Macro average
    mean_tpr = np.mean(tpr_macro_list, axis=0)
    macro_auc = np.trapz(mean_tpr, fpr_macro)
    ax.plot(fpr_macro, mean_tpr, "k--", lw=2.5, label=f"Macro Avg (AUC={macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost — ROC Curves (90 Species)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_shap(model, X_test, feature_names, out_dir):
    print("[INFO] Computing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed if dataset is large
    sample = X_test[:300] if len(X_test) > 300 else X_test
    shap_values = explainer.shap_values(sample)

    # For multi-class, shap_values may be a list (one array per class)
    # or a 3D array (n_samples, n_features, n_classes). Normalise to 2D
    # by averaging absolute SHAP values across classes.
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) → stack and mean over classes
        import numpy as _np
        shap_2d = _np.abs(_np.stack(shap_values, axis=0)).mean(axis=0)
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        import numpy as _np
        if shap_values.shape[2] == sample.shape[1]:
            # shape: (n_samples, n_classes, n_features) – take mean over classes
            shap_2d = _np.abs(shap_values).mean(axis=1)
        else:
            shap_2d = _np.abs(shap_values).mean(axis=2)
    else:
        shap_2d = shap_values

    # Summary plot (beeswarm) — use mean-abs shap_values
    plt.figure()
    shap.summary_plot(
        shap_2d, sample,
        feature_names=feature_names,
        show=False, max_display=20
    )
    plt.title("XGBoost — SHAP Summary Plot (Mean |SHAP| across classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Importance (bar) plot
    plt.figure()
    shap.summary_plot(
        shap_2d, sample,
        feature_names=feature_names,
        plot_type="bar", show=False, max_display=20
    )
    plt.title("XGBoost — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] SHAP plots to {out_dir}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load split
    X_train = np.load(os.path.join(args.split_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.split_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.split_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.split_dir, "y_test.npy"))
    with open(os.path.join(args.split_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(args.split_dir, "feature_names.txt")) as f:
        feature_names = [l.strip() for l in f.readlines()]

    n_classes = len(le.classes_)
    print(f"[INFO] n_classes={n_classes}  n_features={X_train.shape[1]}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",   # GPU: change to "gpu_hist"
    )

    print("[INFO] Training XGBoost...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    train_time = time.time() - t0
    print(f"[INFO] Training time: {train_time:.2f}s")

    # ── Inference ─────────────────────────────────────────────────────────────
    t1 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    inference_time = (time.time() - t1) / len(X_test) * 1000   # ms per sample

    # ── Metrics ───────────────────────────────────────────────────────────────
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc    = accuracy_score(y_test, y_pred)
    top3   = top_k_accuracy(y_test, y_prob, 3)
    top5   = top_k_accuracy(y_test, y_prob, 5)
    y_bin  = label_binarize(y_test, classes=list(range(n_classes)))
    roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

    metrics = {
        "model": "XGBoost",
        "accuracy":           round(acc,  4),
        "precision_macro":    round(report["macro avg"]["precision"], 4),
        "recall_macro":       round(report["macro avg"]["recall"],    4),
        "f1_macro":           round(report["macro avg"]["f1-score"],  4),
        "top3_accuracy":      round(top3,    4),
        "top5_accuracy":      round(top5,    4),
        "roc_auc_macro_ovr":  round(roc_auc, 4),
        "train_time_sec":     round(train_time, 2),
        "inference_time_ms_per_sample": round(inference_time, 4),
        "n_estimators":       500,
        "max_depth":          6,
        "learning_rate":      0.05,
    }

    print("\n─── XGBoost Results ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_test, y_pred, le.classes_,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_test, y_prob, n_classes,
        os.path.join(args.output_dir, "roc_curves.png")
    )
    plot_shap(model, X_test, feature_names, args.output_dir)

    # ── Save model ────────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"\n[DONE] All XGBoost outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",  default="./split_data")
    parser.add_argument("--output_dir", default="./results/xgboost")
    main(parser.parse_args())
