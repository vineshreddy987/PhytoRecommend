"""
retrain_per_element.py
======================
Train ONE XGBoost classifier per element.
Each model only sees data for that element, so it learns
species-specific BCF/pH/concentration patterns for that contaminant.
This eliminates the cross-element dominance bias entirely.
"""

import os, json, pickle, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import xgboost as xgb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Paths (relative to repo root) ─────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(REPO_ROOT, "data", "phytoremediation_dataset.csv")
OUT_DIR   = os.path.join(REPO_ROOT, "models", "per_element_models")
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_ELEMENTS = ['Pb','Cd','Zn','Cu','As','Ni','Cr','Co','Hg','Se','Mn','Fe','Tl','Mo','Sb']
NUMERIC_FEATS = ['soil_concentration','plant_concentration','bcf',
                 'soil_ph','organic_matter_pct','duration_days']

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
for col in NUMERIC_FEATS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

models_meta = {}

for element in MAIN_ELEMENTS:
    print(f"\n{'='*55}")
    print(f"  Training model for element: {element}")
    print(f"{'='*55}")

    sub = df[df['element'] == element].copy()
    print(f"  Raw samples: {len(sub)}")

    # Drop species with < 5 samples for this element
    sp_counts = sub['species'].value_counts()
    valid_sp  = sp_counts[sp_counts >= 5].index
    sub = sub[sub['species'].isin(valid_sp)]
    n_classes = sub['species'].nunique()
    print(f"  Species after filter: {n_classes}")

    if n_classes < 3:
        print(f"  [SKIP] Too few classes for {element}")
        continue

    # Balance: cap max, oversample min — within this element
    MIN_S = max(8,  min(20,  sp_counts.min()))
    MAX_S = max(40, min(150, int(sp_counts.median() + sp_counts.std())))

    parts = []
    for sp, sdf in sub.groupby('species'):
        n = len(sdf)
        if n < MIN_S:
            sdf = resample(sdf, replace=True,  n_samples=MIN_S, random_state=RANDOM_STATE)
        elif n > MAX_S:
            sdf = resample(sdf, replace=False, n_samples=MAX_S, random_state=RANDOM_STATE)
        parts.append(sdf)
    sub_bal = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  After balance: {len(sub_bal)} samples, "
          f"min/max per class: {sub_bal['species'].value_counts().min()}/"
          f"{sub_bal['species'].value_counts().max()}")

    # Feature engineering
    X = sub_bal[NUMERIC_FEATS].copy()
    y_raw = sub_bal['species']

    # Log-transform skewed columns
    for col in ['soil_concentration','plant_concentration','bcf']:
        X[col] = np.log1p(X[col].fillna(0))

    # Interaction: pH × log(BCF) — key separator
    X['ph_x_log_bcf']   = X['soil_ph'].fillna(7.0) * X['bcf']
    # BCF bucket: <0.1, 0.1-1, >1
    X['bcf_gt1']        = (X['bcf'] > np.log1p(1.0)).astype(float)
    X['bcf_lt01']       = (X['bcf'] < np.log1p(0.1)).astype(float)

    X = X.fillna(X.median(numeric_only=True))
    feature_names = list(X.columns)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    if len(np.unique(y)) < 3:
        print(f"  [SKIP] Only {len(np.unique(y))} classes after encoding")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values.astype(np.float32), y,
        test_size=0.20, random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # Inverse-frequency sample weights
    cls_cnt = np.bincount(y_tr, minlength=len(le.classes_)).astype(float)
    cls_w   = 1.0 / np.maximum(cls_cnt, 1)
    cls_w  /= cls_w.mean()
    sw      = cls_w[y_tr]

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=2.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=30,
        verbosity=0,
    )

    t0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_te, y_te)], verbose=False)
    elapsed = time.time() - t0

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    top3   = np.mean([y_te[i] in np.argsort(model.predict_proba(X_te[i:i+1])[0])[-3:]
                      for i in range(len(y_te))])

    print(f"  Accuracy: {acc:.3f}  Top-3: {top3:.3f}  Time: {elapsed:.1f}s  "
          f"best_iter={model.best_iteration}")

    # Quick diversity test
    def q_predict(sc_v, pc_v, bcf_v, ph, org, dur, top=3):
        row = {c: 0.0 for c in feature_names}
        log_b = np.log1p(bcf_v)
        row['soil_concentration']  = np.log1p(sc_v)
        row['plant_concentration'] = np.log1p(pc_v)
        row['bcf']                 = log_b
        row['soil_ph']             = ph
        row['organic_matter_pct']  = org
        row['duration_days']       = dur
        row['ph_x_log_bcf']        = ph * log_b
        row['bcf_gt1']             = float(log_b > np.log1p(1.0))
        row['bcf_lt01']            = float(log_b < np.log1p(0.1))
        v = sc.transform(np.array([[row[c] for c in feature_names]], dtype=np.float32))
        p = model.predict_proba(v)[0]
        return [(le.classes_[i], round(float(p[i])*100,1))
                for i in np.argsort(p)[::-1][:top]]

    # 3 varied inputs per element
    test_inputs = [
        (50,   5.0, 0.10, 6.5, 2.5,  60, 'medium BCF'),
        (500, 500,  1.0,  5.0, 1.5,  30, 'high BCF acid'),
        (10,   0.1, 0.01, 7.5, 4.0, 120, 'low BCF alkaline'),
    ]
    for sc_v, pc_v, bcf_v, ph, org, dur, lbl in test_inputs:
        r = q_predict(sc_v, pc_v, bcf_v, ph, org, dur)
        print(f"  [{lbl}] -> {r[0][0]}({r[0][1]}%)  {r[1][0]}({r[1][1]}%)  {r[2][0]}({r[2][1]}%)")

    # Save model artefacts
    el_dir = os.path.join(OUT_DIR, element)
    os.makedirs(el_dir, exist_ok=True)
    with open(os.path.join(el_dir, "model.pkl"),         "wb") as f: pickle.dump(model, f)
    with open(os.path.join(el_dir, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
    with open(os.path.join(el_dir, "scaler.pkl"),        "wb") as f: pickle.dump(sc, f)
    with open(os.path.join(el_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    models_meta[element] = {
        "n_classes":    int(len(le.classes_)),
        "n_features":   int(len(feature_names)),
        "accuracy":     round(acc, 4),
        "top3_accuracy":round(top3, 4),
        "best_iter":    int(model.best_iteration),
        "classes":      list(le.classes_),
    }

# Save master index
with open(os.path.join(OUT_DIR, "models_meta.json"), "w") as f:
    json.dump(models_meta, f, indent=2)

print("\n" + "="*55)
print("  ALL MODELS TRAINED")
print("="*55)
for el, m in models_meta.items():
    print(f"  {el:4s}: {m['n_classes']:2d} classes  acc={m['accuracy']:.3f}  top3={m['top3_accuracy']:.3f}")
print(f"\n[DONE] Models saved to {OUT_DIR}")
