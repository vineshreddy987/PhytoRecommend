"""
00_setup_and_split.py
=====================
STEP 1 — Data loading, preprocessing, and deterministic train/test split.
Run this FIRST. It saves the split to disk so every model uses
the IDENTICAL split (reproducibility requirement for IEEE submission).

Usage:
    python 00_setup_and_split.py --data path/to/your_dataset.csv \
                                  --label species_column_name \
                                  --output_dir ./split_data

Outputs (saved to --output_dir):
    X_train.npy, X_test.npy
    y_train.npy, y_test.npy
    label_encoder.pkl
    feature_names.txt
    split_meta.json        ← records random_state, n_classes, n_features, sizes
"""

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42   # Fixed seed — must never change across experiments


def load_and_preprocess(data_path: str, label_col: str):
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Drop rows with missing label ──────────────────────────────────────────
    df = df.dropna(subset=[label_col])

    # ── Drop classes with fewer than 2 samples (can't stratify-split) ─────────
    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= 2].index
    removed = counts[counts < 2].index.tolist()
    if removed:
        print(f"[INFO] Removed {len(removed)} class(es) with <2 samples: {removed}")
    df = df[df[label_col].isin(valid_classes)].copy()
    print(f"[INFO] Rows after class filter: {df.shape[0]}")

    # ── Separate features and label ───────────────────────────────────────────
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # ── Drop free-text / ID columns that add no signal ────────────────────────
    drop_cols = [c for c in X.columns if c.lower() in ('source', 'doi', 'notes',
                 'variety', 'location', 'date', 'treatment', 'oxidative_state',
                 'chemical_form', 'concentration_units')]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # ── Coerce numeric-looking object columns to float ───────────────────────
    # Force-coerce all object columns to numeric; non-parseable values → NaN
    # This handles columns like bcf/soil_ph that contain sentinel text
    # (e.g. "Non précisé") mixed with numeric data.
    for col in list(X.select_dtypes(include='object').columns):
        converted = pd.to_numeric(X[col], errors='coerce')
        # Keep conversion if the column name suggests numeric data
        # OR if at least 20% of non-null values are actually numeric
        non_null = X[col].notna().sum()
        converted_non_null = converted.notna().sum()
        numeric_ratio = converted_non_null / non_null if non_null > 0 else 0
        # Columns that are clearly categorical (e.g. chemical symbols, soil type)
        # will have ~0 numeric ratio; force-coerce everything else
        if numeric_ratio > 0.0 or col in (
            'bcf', 'soil_ph', 'organic_matter_pct',
            'soil_concentration', 'plant_concentration', 'duration_days'
        ):
            X[col] = converted
            print(f"[INFO] Converted '{col}' to numeric (numeric_ratio={numeric_ratio:.2f})")
        # else: leave as object for one-hot encoding

    # ── Handle missing values in numeric features (median imputation) ─────────
    X[X.select_dtypes(include='number').columns] = X.select_dtypes(include='number').fillna(
        X.select_dtypes(include='number').median()
    )

    # ── One-hot encode remaining categorical columns ───────────────────────────
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    if cat_cols:
        print(f"[INFO] One-hot encoding {len(cat_cols)} categorical columns: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    X = X.fillna(0)  # any remaining NaNs

    feature_names = list(X.columns)
    print(f"[INFO] Features retained after encoding: {len(feature_names)}")

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"[INFO] Number of species (classes): {n_classes}")

    return X.values.astype(np.float32), y_enc, le, feature_names


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    X, y, le, feature_names = load_and_preprocess(args.data, args.label)

    # ── Stratified split: 80 % train / 20 % test ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Standard-scale features (fit on train only) ───────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Save everything ───────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"),  y_test)

    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(args.output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    meta = {
        "random_state": RANDOM_STATE,
        "n_classes": int(len(le.classes_)),
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "test_size": 0.20,
        "scaling": "StandardScaler",
        "label_column": args.label,
        "source_file": args.data,
    }
    with open(os.path.join(args.output_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Split saved to: {args.output_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True, help="Path to CSV dataset")
    parser.add_argument("--label",      required=True, help="Name of the species/label column")
    parser.add_argument("--output_dir", default="./split_data")
    main(parser.parse_args())
