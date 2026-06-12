"""
05_aggregate_results.py
========================
After all models are trained, run this to:
  1. Collect all metrics.json files into a single IEEE-formatted table
  2. Run statistical significance tests (McNemar's test) between the
     proposed model and baselines
  3. Generate combined comparison plots

Usage:
    python 05_aggregate_results.py --results_root ./results \
                                    --split_dir   ./split_data \
                                    --output_dir  ./results/summary

IMPORTANT: Only run after ALL models have been trained and their
metrics.json + classification_report.json files exist.

Outputs:
    ieee_benchmark_table.csv    ← paste directly into LaTeX
    ieee_benchmark_table.tex    ← ready-to-use \tabular block
    mcnemar_tests.json          ← p-values for significance claims
    comparison_bar_chart.png
    top_k_comparison.png
    timing_comparison.png
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

# ─── Config ────────────────────────────────────────────────────────────────────

# Maps subfolder name → display name for the table
MODEL_DIRS = {
    "random_forest":         "Random Forest (Baseline)",
    "xgboost":               "XGBoost",
    "ann":                   "ANN (MLP)",
    "tabtransformer":        "TabTransformer",
    "self_attention_12head": "12-Head Self-Attention Transformer (Proposed)",
}

METRIC_COLS = [
    ("accuracy",           "Accuracy"),
    ("precision_macro",    "Precision (Macro)"),
    ("recall_macro",       "Recall (Macro)"),
    ("f1_macro",           "F1-Score (Macro)"),
    ("top3_accuracy",      "Top-3 Accuracy"),
    ("top5_accuracy",      "Top-5 Accuracy"),
    ("roc_auc_macro_ovr",  "ROC-AUC (Macro OvR)"),
    ("train_time_sec",     "Training Time (s)"),
    ("inference_time_ms_per_sample", "Inference (ms/sample)"),
]


# ─── Collect metrics ───────────────────────────────────────────────────────────

def load_all_metrics(results_root):
    rows = []
    for folder, display_name in MODEL_DIRS.items():
        path = os.path.join(results_root, folder, "metrics.json")
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}  → skipping {display_name}")
            continue
        with open(path) as f:
            m = json.load(f)
        row = {"Model": display_name}
        for key, _ in METRIC_COLS:
            row[key] = m.get(key, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ─── LaTeX table ───────────────────────────────────────────────────────────────

def df_to_latex(df):
    col_format = "l" + "r" * len(METRIC_COLS)
    header_names = ["Model"] + [label for _, label in METRIC_COLS]

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Performance Comparison of Classification Models on the 90-Species Dataset}",
        r"\label{tab:benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_format + r"}",
        r"\toprule",
        " & ".join(header_names) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        cells = [row["Model"]]
        for key, _ in METRIC_COLS:
            v = row[key]
            if key in ("train_time_sec", "inference_time_ms_per_sample"):
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:.4f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ─── McNemar's Test ────────────────────────────────────────────────────────────

def run_mcnemar_tests(results_root, split_dir):
    """
    McNemar's test compares the PROPOSED model's predictions
    against each baseline on the same test set.

    Requires saved y_pred arrays. If not saved, prints a warning.
    Add  np.save('y_pred.npy', y_pred)  to each training script
    if you want to run this automatically.
    """
    proposed_pred_path = os.path.join(
        results_root, "self_attention_12head", "y_pred.npy"
    )
    if not os.path.exists(proposed_pred_path):
        print("[WARN] y_pred.npy not found for proposed model. "
              "Add  np.save(os.path.join(output_dir,'y_pred.npy'), y_pred) "
              "at the end of 04_train_self_attention_transformer.py and re-run.")
        return {}

    y_test   = np.load(os.path.join(split_dir, "y_test.npy"))
    y_prop   = np.load(proposed_pred_path)
    correct_prop = (y_prop == y_test)

    results = {}
    for folder, display_name in MODEL_DIRS.items():
        if folder == "self_attention_12head":
            continue
        pred_path = os.path.join(results_root, folder, "y_pred.npy")
        if not os.path.exists(pred_path):
            continue
        y_base = np.load(pred_path)
        correct_base = (y_base == y_test)

        # Contingency table
        n00 = np.sum(~correct_prop & ~correct_base)
        n01 = np.sum(~correct_prop &  correct_base)
        n10 = np.sum( correct_prop & ~correct_base)
        n11 = np.sum( correct_prop &  correct_base)

        table = np.array([[n11, n10], [n01, n00]])
        result = mcnemar(table, exact=True)
        results[display_name] = {
            "statistic": float(result.statistic),
            "p_value":   float(result.pvalue),
            "significant_at_0.05": bool(result.pvalue < 0.05),
            "contingency_table": table.tolist(),
        }
        print(f"  McNemar vs {display_name}: p={result.pvalue:.6f}  "
              f"{'*SIGNIFICANT*' if result.pvalue < 0.05 else 'not significant'}")

    return results


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_comparison_bar(df, out_path):
    metrics_to_plot = ["accuracy", "f1_macro", "roc_auc_macro_ovr"]
    labels          = ["Accuracy", "F1 (Macro)", "ROC-AUC (Macro)"]
    n_models = len(df)
    x = np.arange(len(metrics_to_plot))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=row["Model"],
                      color=colors[i], edgecolor="k", linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison — Key Metrics")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_top_k(df, out_path):
    metrics = ["accuracy", "top3_accuracy", "top5_accuracy"]
    labels  = ["Top-1", "Top-3", "Top-5"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        vals = [row[m] for m in metrics]
        ax.plot(labels, vals, marker="o", label=row["Model"])

    ax.set_ylabel("Accuracy"); ax.set_title("Top-k Accuracy Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_timing(df, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    ax1.barh(df["Model"], df["train_time_sec"], color=colors)
    ax1.set_xlabel("Training Time (s)")
    ax1.set_title("Training Time")

    ax2.barh(df["Model"], df["inference_time_ms_per_sample"], color=colors)
    ax2.set_xlabel("Inference Time (ms/sample)")
    ax2.set_title("Inference Latency")

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[STEP 1] Loading metrics from all models...")
    df = load_all_metrics(args.results_root)

    if df.empty:
        print("[ERROR] No metrics found. Train at least one model first.")
        return

    print("\n─── Collected Benchmark Table ─────────────────────")
    print(df.to_string(index=False))

    # CSV
    csv_path = os.path.join(args.output_dir, "ieee_benchmark_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # LaTeX
    tex = df_to_latex(df)
    tex_path = os.path.join(args.output_dir, "ieee_benchmark_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"[SAVED] {tex_path}")
    print("\n─── LaTeX Table ────────────────────────────────────")
    print(tex)

    # Significance tests
    print("\n[STEP 2] McNemar's significance tests...")
    mcnemar_results = run_mcnemar_tests(args.results_root, args.split_dir)
    if mcnemar_results:
        mn_path = os.path.join(args.output_dir, "mcnemar_tests.json")
        with open(mn_path, "w") as f:
            json.dump(mcnemar_results, f, indent=2)
        print(f"[SAVED] {mn_path}")

    # Plots
    print("\n[STEP 3] Generating comparison plots...")
    plot_comparison_bar(df, os.path.join(args.output_dir, "comparison_bar_chart.png"))
    plot_top_k(df,          os.path.join(args.output_dir, "top_k_comparison.png"))
    plot_timing(df,         os.path.join(args.output_dir, "timing_comparison.png"))

    print(f"\n[DONE] All summary outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--split_dir",    default="./split_data")
    parser.add_argument("--output_dir",   default="./results/summary")
    main(parser.parse_args())
