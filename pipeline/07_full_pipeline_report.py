"""
full_pipeline_report.py
========================
Complete ML pipeline for phytoremediation plant recommendation.
Covers: class analysis → rare-class removal → class-weight training →
        SMOTE → retrain → compare → feature importance → test predictions → final report.
"""

import os, json, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BASE      = REPO_ROOT
DATA      = os.path.join(REPO_ROOT, "data", "phytoremediation_dataset.csv")
OUT       = os.path.join(REPO_ROOT, "reports")
os.makedirs(OUT, exist_ok=True)

NUMERIC  = ['soil_concentration', 'plant_concentration', 'bcf',
            'soil_ph', 'organic_matter_pct', 'duration_days']
MIN_RECORDS = 50   # drop species with fewer records than this

print("="*70)
print("  PHYTOREMEDIATION PLANT RECOMMENDATION — FULL PIPELINE REPORT")
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load & analyse class distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 1] Class Distribution Analysis")
print("-"*50)

df_raw = pd.read_csv(DATA)
for c in NUMERIC:
    df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

sp_counts = df_raw['species'].value_counts()
sp_pct    = (sp_counts / len(df_raw) * 100).round(2)

print(f"  Total records   : {len(df_raw):,}")
print(f"  Total species   : {sp_counts.shape[0]}")
print(f"  Max class count : {sp_counts.max():,}  ({sp_counts.index[0]})")
print(f"  Min class count : {sp_counts.min():,}  ({sp_counts.index[-1]})")
print(f"  Imbalance ratio : {sp_counts.max()/sp_counts.min():.0f}:1")
print(f"\n  Full species distribution:")
dist_df = pd.DataFrame({'Count': sp_counts, 'Percentage': sp_pct})
print(dist_df.to_string())

# Save distribution table
dist_df.to_csv(os.path.join(OUT, "class_distribution.csv"))

# ── Figure 1: Class imbalance bar chart ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(18, 14))

colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sp_counts)))[::-1]
bars = axes[0].bar(range(len(sp_counts)), sp_counts.values, color=colors, edgecolor='white', linewidth=0.4)
axes[0].axhline(MIN_RECORDS, color='red', linestyle='--', linewidth=1.5,
                label=f'Min threshold ({MIN_RECORDS} records)')
axes[0].set_xticks(range(len(sp_counts)))
axes[0].set_xticklabels(sp_counts.index, rotation=90, fontsize=6)
axes[0].set_ylabel('Record Count', fontsize=11)
axes[0].set_title('Class Distribution — All Species (before filtering)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Percentage pie for top-10 vs rest
top10 = sp_counts.head(10)
rest  = pd.Series({'Others (remaining species)': sp_counts.iloc[10:].sum()})
pie_data = pd.concat([top10, rest])
wedge_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
axes[1].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
            colors=wedge_colors, startangle=140, textprops={'fontsize': 8})
axes[1].set_title('Top-10 Species vs Rest (%)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_class_imbalance.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [SAVED] fig1_class_imbalance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Filter rare species
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[SECTION 2] Remove species with < {MIN_RECORDS} records")
print("-"*50)

rare = sp_counts[sp_counts < MIN_RECORDS]
kept = sp_counts[sp_counts >= MIN_RECORDS]
print(f"  Removed {len(rare)} rare species: {list(rare.index)}")
print(f"  Kept {len(kept)} species")

df = df_raw[df_raw['species'].isin(kept.index)].copy()
print(f"  Records after filter: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature engineering (shared by all models)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 3] Feature Engineering")
print("-"*50)

def engineer_features(df_in):
    X = df_in[NUMERIC + ['element']].copy()
    # One-hot encode element
    X = pd.get_dummies(X, columns=['element'], dtype=float)
    # Log-transform skewed columns
    for c in ['soil_concentration', 'plant_concentration', 'bcf']:
        X[c] = np.log1p(X[c].fillna(0))
    # Interaction features
    X['ph_x_log_bcf']     = X['soil_ph'].fillna(7.0)        * X['bcf']
    X['logconc_x_logbcf'] = X['soil_concentration']          * X['bcf']
    X['bcf_category']     = pd.cut(
        np.expm1(X['bcf']),
        bins=[-0.001, 0.1, 1.0, 999],
        labels=[0, 1, 2]              # 0=low, 1=moderate, 2=hyperacc
    ).astype(float)
    X['log_duration']     = np.log1p(X['duration_days'].fillna(60))
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])
    return X

X_all = engineer_features(df)
feature_names = list(X_all.columns)
print(f"  Features created : {len(feature_names)}")
print(f"  Feature list: {feature_names}")

le_main = LabelEncoder()
y_all   = le_main.fit_transform(df['species'])
n_cls   = len(le_main.classes_)
print(f"  Classes          : {n_cls}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all.values.astype(np.float32), y_all,
    test_size=0.2, random_state=42, stratify=y_all)

sc_main = StandardScaler()
X_tr_sc = sc_main.fit_transform(X_tr)
X_te_sc = sc_main.transform(X_te)
print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BASELINE model (no balancing)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 4] Baseline Model (no class balancing)")
print("-"*50)

def get_metrics(y_true, y_pred, y_prob, n_classes, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"  {'─'*45}")
    print(f"  {label}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall (macro)    : {rec:.4f}")
    print(f"  F1 Score (macro)  : {f1:.4f}")
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

base_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, tree_method='hist',
    early_stopping_rounds=30, verbosity=0)

t0 = time.time()
base_model.fit(X_tr_sc, y_tr, eval_set=[(X_te_sc, y_te)], verbose=False)
base_time = time.time() - t0

y_pred_base = base_model.predict(X_te_sc)
y_prob_base = base_model.predict_proba(X_te_sc)
metrics_base = get_metrics(y_te, y_pred_base, y_prob_base, n_cls, "BASELINE (no balancing)")
print(f"  Train time        : {base_time:.1f}s")

# Per-class report baseline
report_base = classification_report(y_te, y_pred_base,
    target_names=le_main.classes_, zero_division=0, output_dict=True)

# Sunflower dominance check
sf_idx = list(le_main.classes_).index('Sunflower') if 'Sunflower' in le_main.classes_ else -1
unique_pred, pred_counts = np.unique(y_pred_base, return_counts=True)
pred_count_dict = dict(zip(unique_pred, pred_counts))
sf_pred_pct = pred_count_dict.get(sf_idx, 0) / len(y_pred_base) * 100
print(f"  Sunflower predicted as #1 : {pred_count_dict.get(sf_idx,0)} / {len(y_pred_base)} = {sf_pred_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CLASS WEIGHTS model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 5] Model with Class Weights")
print("-"*50)

classes_arr = np.unique(y_tr)
cw = compute_class_weight('balanced', classes=classes_arr, y=y_tr)
cw_dict = dict(zip(classes_arr, cw))
sw_tr = np.array([cw_dict[c] for c in y_tr])

cw_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, tree_method='hist',
    early_stopping_rounds=30, verbosity=0)

t0 = time.time()
cw_model.fit(X_tr_sc, y_tr, sample_weight=sw_tr,
             eval_set=[(X_te_sc, y_te)], verbose=False)
cw_time = time.time() - t0

y_pred_cw = cw_model.predict(X_te_sc)
y_prob_cw = cw_model.predict_proba(X_te_sc)
metrics_cw = get_metrics(y_te, y_pred_cw, y_prob_cw, n_cls, "CLASS WEIGHTS model")
print(f"  Train time        : {cw_time:.1f}s")

unique_pred_cw, pred_counts_cw = np.unique(y_pred_cw, return_counts=True)
pred_cw_dict = dict(zip(unique_pred_cw, pred_counts_cw))
sf_cw_pct = pred_cw_dict.get(sf_idx, 0) / len(y_pred_cw) * 100
print(f"  Sunflower predicted as #1 : {pred_cw_dict.get(sf_idx,0)} / {len(y_pred_cw)} = {sf_cw_pct:.1f}%")

# Decide whether SMOTE is needed
still_biased = sf_cw_pct > 20.0
print(f"\n  Sunflower still > 20% of predictions? {'YES — applying SMOTE' if still_biased else 'NO — class weights sufficient'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SMOTE + Class Weights model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 6] SMOTE Oversampling + Class Weights")
print("-"*50)

# SMOTE: cap max class size to avoid over-synthetic samples
counts_tr = pd.Series(y_tr).value_counts()
# Strategy: bring all classes up to max(median, min_threshold) but cap at current_max/2
target_n = min(int(counts_tr.median() * 2), counts_tr.max())
smote_strategy = {c: max(counts_tr.get(c, 0), min(counts_tr.get(c, 0)*3, target_n))
                  for c in range(n_cls) if counts_tr.get(c, 0) > 0}

print(f"  SMOTE target per class (approx): {target_n}")
print(f"  Classes that will be oversampled: "
      f"{sum(1 for c,n in smote_strategy.items() if counts_tr.get(c,0) < n)}")

smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=3)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc, y_tr)
print(f"  After SMOTE: {len(X_tr_sm):,} training samples (was {len(X_tr):,})")

# Recompute weights on SMOTE'd data
cw2 = compute_class_weight('balanced', classes=np.unique(y_tr_sm), y=y_tr_sm)
sw_sm = np.array([cw2[c] for c in y_tr_sm])

smote_model = xgb.XGBClassifier(
    n_estimators=600, max_depth=7, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.2, reg_lambda=2.0,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, tree_method='hist',
    early_stopping_rounds=40, verbosity=0)

t0 = time.time()
smote_model.fit(X_tr_sm, y_tr_sm, sample_weight=sw_sm,
                eval_set=[(X_te_sc, y_te)], verbose=False)
smote_time = time.time() - t0

y_pred_sm = smote_model.predict(X_te_sc)
y_prob_sm = smote_model.predict_proba(X_te_sc)
metrics_sm = get_metrics(y_te, y_pred_sm, y_prob_sm, n_cls, "SMOTE + Class Weights model")
print(f"  Train time        : {smote_time:.1f}s")

unique_pred_sm, pred_counts_sm = np.unique(y_pred_sm, return_counts=True)
pred_sm_dict = dict(zip(unique_pred_sm, pred_counts_sm))
sf_sm_pct = pred_sm_dict.get(sf_idx, 0) / len(y_pred_sm) * 100
print(f"  Sunflower predicted as #1 : {pred_sm_dict.get(sf_idx,0)} / {len(y_pred_sm)} = {sf_sm_pct:.1f}%")

# Choose best model
best_model  = smote_model
best_scaler = sc_main
best_le     = le_main
best_fn     = feature_names
best_label  = "SMOTE+ClassWeights"
best_preds  = y_pred_sm
best_probs  = y_prob_sm
metrics_best= metrics_sm

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Confusion matrices comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 7] Confusion Matrix Comparison")
print("-"*50)

fig, axes = plt.subplots(1, 3, figsize=(36, 14))
for ax, y_p, title in [
    (axes[0], y_pred_base, "Baseline (No Balancing)"),
    (axes[1], y_pred_cw,   "Class Weights"),
    (axes[2], y_pred_sm,   "SMOTE + Class Weights (Best)"),
]:
    cm = confusion_matrix(y_te, y_p)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    n = len(le_main.classes_)
    ax.set_xticks(range(n)); ax.set_xticklabels(le_main.classes_, rotation=90, fontsize=4)
    ax.set_yticks(range(n)); ax.set_yticklabels(le_main.classes_, fontsize=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_confusion_matrices.png"), dpi=120, bbox_inches='tight')
plt.close()
print("  [SAVED] fig2_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Per-class performance report
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 8] Per-Class Performance Report (Best Model)")
print("-"*50)

report_sm = classification_report(y_te, y_pred_sm,
    target_names=le_main.classes_, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_sm).T
report_df = report_df[report_df.index.isin(le_main.classes_)]
report_df = report_df.sort_values('f1-score', ascending=False)
print(report_df[['precision','recall','f1-score','support']].round(3).to_string())
report_df.to_csv(os.path.join(OUT, "per_class_report.csv"))

# ── Figure 3: Per-class F1 comparison ────────────────────────────────────────
report_base_df = pd.DataFrame(report_base).T
report_base_df = report_base_df[report_base_df.index.isin(le_main.classes_)]

classes_sorted = report_df.index.tolist()
f1_base = [report_base_df.loc[c,'f1-score'] if c in report_base_df.index else 0
           for c in classes_sorted]
f1_sm   = [report_df.loc[c,'f1-score'] for c in classes_sorted]

x = np.arange(len(classes_sorted))
fig, ax = plt.subplots(figsize=(22, 7))
w = 0.38
ax.bar(x - w/2, f1_base, w, label='Baseline', color='#ef9a9a', edgecolor='white', linewidth=0.3)
ax.bar(x + w/2, f1_sm,   w, label='SMOTE+CW', color='#66bb6a', edgecolor='white', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(classes_sorted, rotation=90, fontsize=6)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('Per-Class F1 Score: Baseline vs SMOTE+ClassWeights', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_per_class_f1.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] fig3_per_class_f1.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 9] Feature Importance Analysis")
print("-"*50)

fi = best_model.feature_importances_
fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
fi_df['cumulative_pct'] = fi_df['importance'].cumsum() / fi_df['importance'].sum() * 100

print("  Top 15 features:")
print(fi_df.head(15).to_string(index=False))

# Group importance by category
def categorize(feat):
    if feat in ['soil_concentration','log_soil_conc']:    return 'Soil Concentration'
    if feat in ['plant_concentration']:                   return 'Plant Concentration'
    if feat in ['bcf','bcf_category']:                    return 'BCF'
    if feat == 'soil_ph':                                 return 'Soil pH'
    if feat == 'organic_matter_pct':                      return 'Organic Matter'
    if feat in ['duration_days','log_duration']:          return 'Exposure Duration'
    if feat.startswith('element_'):                       return 'Contaminant Element'
    if 'bcf' in feat or 'conc' in feat:                   return 'Interaction Features'
    return 'Other'

fi_df['category'] = fi_df['feature'].apply(categorize)
cat_imp = fi_df.groupby('category')['importance'].sum().sort_values(ascending=False)
cat_pct = (cat_imp / cat_imp.sum() * 100).round(1)

print("\n  Importance by category:")
for cat, pct in cat_pct.items():
    bar = '█' * int(pct / 2)
    print(f"  {cat:<25}: {pct:5.1f}%  {bar}")

# Check if element dominates
elem_dom = cat_pct.get('Contaminant Element', 0)
if elem_dom > 70:
    print(f"\n  ⚠️  Element importance = {elem_dom:.1f}% (>70%) — see Section 11 for fix")

# ── Figure 4: Feature importance ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top-20 features
top20 = fi_df.head(20)
colors_fi = plt.cm.viridis(np.linspace(0.2, 0.85, len(top20)))[::-1]
ax1.barh(range(len(top20)), top20['importance'].values[::-1], color=colors_fi)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['feature'].values[::-1], fontsize=8)
ax1.set_xlabel('Importance Score', fontsize=10)
ax1.set_title('Top-20 Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Category pie
wedge_c = plt.cm.Set2(np.linspace(0, 1, len(cat_imp)))
ax2.pie(cat_imp.values, labels=cat_imp.index, autopct='%1.1f%%',
        colors=wedge_c, startangle=140, textprops={'fontsize': 9})
ax2.set_title('Feature Importance by Category', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [SAVED] fig4_feature_importance.png")

# ── SHAP values ───────────────────────────────────────────────────────────────
print("  Computing SHAP values (sample of 400)...")
explainer   = shap.TreeExplainer(best_model)
sample_idx  = np.random.choice(len(X_te_sc), min(400, len(X_te_sc)), replace=False)
shap_values = explainer.shap_values(X_te_sc[sample_idx])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_te_sc[sample_idx],
                  feature_names=feature_names, show=False,
                  max_display=20, plot_type='bar')
plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_shap_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] fig5_shap_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Test predictions: extreme scenarios
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 10] Test Predictions — Extreme & Varied Scenarios")
print("-"*50)

# Re-identify element columns
elem_cols_all = [f for f in feature_names if f.startswith('element_')]
all_elements  = [c.replace('element_','') for c in elem_cols_all]

def make_pred(soil_c, plant_c, bcf_v, ph, org, dur, el, top=5):
    row = {col: 0.0 for col in feature_names}
    log_b = np.log1p(bcf_v)
    row['soil_concentration']  = np.log1p(soil_c)
    row['plant_concentration'] = np.log1p(plant_c)
    row['bcf']                 = log_b
    row['soil_ph']             = ph
    row['organic_matter_pct']  = org
    row['duration_days']       = dur
    row['ph_x_log_bcf']        = ph * log_b
    row['logconc_x_logbcf']    = np.log1p(soil_c) * log_b
    row['bcf_category']        = 0.0 if bcf_v < 0.1 else (1.0 if bcf_v < 1.0 else 2.0)
    row['log_duration']        = np.log1p(dur)
    ecol = 'element_' + el
    if ecol in row: row[ecol] = 1.0
    v = best_scaler.transform(np.array([[row[f] for f in feature_names]], dtype=np.float32))
    p = best_model.predict_proba(v)[0]
    return [(best_le.classes_[i], round(float(p[i])*100, 1))
            for i in np.argsort(p)[::-1][:top]]

# Scenario groups
scenario_groups = {
    "BCF Variation (Pb, neutral pH=7)": [
        ("Very low BCF=0.001",  50,   0.05, 0.001, 7.0, 2.5,  60, 'Pb'),
        ("Low BCF=0.05",        50,   2.5,  0.05,  7.0, 2.5,  60, 'Pb'),
        ("Moderate BCF=0.5",    50,  25.0,  0.5,   7.0, 2.5,  60, 'Pb'),
        ("High BCF=2.0",        50, 100.0,  2.0,   7.0, 2.5,  60, 'Pb'),
        ("Hyperaccum BCF=5.0",  50, 250.0,  5.0,   7.0, 2.5,  60, 'Pb'),
    ],
    "pH Variation (Cd, BCF=0.5)": [
        ("Very acidic pH=3.5",  30,  15.0,  0.5,   3.5, 2.0,  60, 'Cd'),
        ("Acidic pH=5.0",       30,  15.0,  0.5,   5.0, 2.0,  60, 'Cd'),
        ("Neutral pH=7.0",      30,  15.0,  0.5,   7.0, 2.0,  60, 'Cd'),
        ("Alkaline pH=8.0",     30,  15.0,  0.5,   8.0, 2.0,  60, 'Cd'),
        ("Very alkaline pH=9",  30,  15.0,  0.5,   9.0, 2.0,  60, 'Cd'),
    ],
    "Exposure Duration (Zn, BCF=1.0, pH=6.5)": [
        ("14 days",             100, 100.0, 1.0,   6.5, 3.0,  14, 'Zn'),
        ("30 days",             100, 100.0, 1.0,   6.5, 3.0,  30, 'Zn'),
        ("60 days",             100, 100.0, 1.0,   6.5, 3.0,  60, 'Zn'),
        ("120 days",            100, 100.0, 1.0,   6.5, 3.0, 120, 'Zn'),
        ("365 days",            100, 100.0, 1.0,   6.5, 3.0, 365, 'Zn'),
    ],
    "Contaminant Element (BCF=0.3, pH=6.5, medium conc)": [
        ("Lead (Pb)",           50,  15.0,  0.3,   6.5, 2.5,  60, 'Pb'),
        ("Cadmium (Cd)",        50,  15.0,  0.3,   6.5, 2.5,  60, 'Cd'),
        ("Zinc (Zn)",           50,  15.0,  0.3,   6.5, 2.5,  60, 'Zn'),
        ("Arsenic (As)",        50,  15.0,  0.3,   6.5, 2.5,  60, 'As'),
        ("Chromium (Cr)",       50,  15.0,  0.3,   6.5, 2.5,  60, 'Cr'),
        ("Cobalt (Co)",         50,  15.0,  0.3,   6.5, 2.5,  60, 'Co'),
        ("Mercury (Hg)",        50,  15.0,  0.3,   6.5, 2.5,  60, 'Hg'),
        ("Selenium (Se)",       50,  15.0,  0.3,   6.5, 2.5,  60, 'Se'),
    ],
    "Organic Matter Variation (Ni, BCF=0.2, pH=6.8)": [
        ("Very low OM=0.5%",   120,  24.0,  0.2,   6.8, 0.5,  75, 'Ni'),
        ("Low OM=1.5%",        120,  24.0,  0.2,   6.8, 1.5,  75, 'Ni'),
        ("Medium OM=3.0%",     120,  24.0,  0.2,   6.8, 3.0,  75, 'Ni'),
        ("High OM=6.0%",       120,  24.0,  0.2,   6.8, 6.0,  75, 'Ni'),
        ("Very high OM=15%",   120,  24.0,  0.2,   6.8,15.0,  75, 'Ni'),
    ],
}

pred_results = {}
all_top1 = []

for group_name, scenarios in scenario_groups.items():
    print(f"\n  ── {group_name}")
    pred_results[group_name] = []
    for label, sc_v, pc_v, bcf_v, ph, org, dur, el in scenarios:
        r = make_pred(sc_v, pc_v, bcf_v, ph, org, dur, el)
        all_top1.append(r[0][0])
        pred_results[group_name].append({
            "scenario": label, "top1": r[0][0], "top1_conf": r[0][1],
            "top2": r[1][0], "top2_conf": r[1][1],
            "top3": r[2][0], "top3_conf": r[2][1],
        })
        print(f"    {label:<30} -> #{1}: {r[0][0]:<30} ({r[0][1]:5.1f}%)"
              f"  #{2}: {r[1][0]:<25} ({r[1][1]:5.1f}%)")

# Diversity summary
unique_top1 = list(set(all_top1))
print(f"\n  Unique species predicted as #1 across ALL scenarios: {len(unique_top1)}")
print(f"  Species: {sorted(unique_top1)}")
from collections import Counter
top1_freq = Counter(all_top1)
print(f"  Frequency breakdown:")
for sp, cnt in top1_freq.most_common():
    bar = '█' * cnt
    print(f"    {sp:<40}: {cnt:2d}x  {bar}")

# ── Figure 6: Prediction diversity bar ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
freq_items = top1_freq.most_common()
sp_labels  = [x[0] for x in freq_items]
sp_vals    = [x[1] for x in freq_items]
colors_pred = ['#c62828' if v > 5 else '#ef6c00' if v > 2 else '#2e7d32' for v in sp_vals]
ax.bar(sp_labels, sp_vals, color=colors_pred, edgecolor='white')
ax.axhline(3, color='gray', linestyle='--', linewidth=1, label='Threshold: 3 predictions')
ax.set_xticklabels(sp_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Times predicted as #1', fontsize=10)
ax.set_title(f'Prediction Diversity — {len(unique_top1)} unique species across {len(all_top1)} test scenarios',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_prediction_diversity.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [SAVED] fig6_prediction_diversity.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Additional feature engineering recommendations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 11] Feature Engineering Recommendations")
print("-"*50)

elem_importance = cat_pct.get('Contaminant Element', 0)
print(f"  Contaminant Element importance: {elem_importance:.1f}%")
if elem_importance > 70:
    print("""
  ⚠️  Element dominates predictions (>70%). Recommended additional features:

  1. BCF × pH interaction  [ALREADY ADDED — ph_x_log_bcf]
  2. log(soil_conc) × log(BCF)  [ALREADY ADDED — logconc_x_logbcf]
  3. BCF category bucket (low/moderate/hyperacc)  [ALREADY ADDED]
  4. Toxicity class per element (numeric: 1=Low, 2=Medium, 3=High)
  5. Duration × BCF interaction: plants differ by uptake rate over time
  6. pH × organic_matter: soil chemistry synergy affects bioavailability
  7. Concentration quintile rank: relative contamination severity
  8. Train per-element models (isolates element-specific patterns)
""")
else:
    print(f"  ✓ Element importance ({elem_importance:.1f}%) is within acceptable range (<70%).")
    print("  Environmental variables are contributing meaningfully to predictions.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Save best model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 12] Saving Best Model")
print("-"*50)

model_out = os.path.join(REPO_ROOT, "models", "best_balanced_model")
os.makedirs(model_out, exist_ok=True)
with open(os.path.join(model_out, "model.pkl"), "wb") as f:         pickle.dump(best_model, f)
with open(os.path.join(model_out, "label_encoder.pkl"), "wb") as f: pickle.dump(best_le, f)
with open(os.path.join(model_out, "scaler.pkl"), "wb") as f:        pickle.dump(best_scaler, f)
with open(os.path.join(model_out, "feature_names.txt"), "w") as f:
    f.write("\n".join(best_fn))

print(f"  Model saved to: {model_out}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL REPORT SUMMARY")
print("="*70)

print(f"""
1. ORIGINAL CLASS DISTRIBUTION
   ─────────────────────────────
   Total records   : {len(df_raw):,}
   Total species   : {sp_counts.shape[0]}
   Imbalance ratio : {sp_counts.max()/sp_counts.min():.0f}:1
   Most common     : {sp_counts.index[0]} ({sp_counts.iloc[0]:,} records, {sp_counts.iloc[0]/len(df_raw)*100:.1f}%)
   Least common    : {sp_counts.index[-1]} ({sp_counts.iloc[-1]:,} records)
   Species removed (< {MIN_RECORDS} records): {len(rare)} species

2. BALANCING METHOD USED
   ──────────────────────
   Step 1: Class Weights (sklearn compute_class_weight='balanced')
   Step 2: SMOTE oversampling (minority classes → target {target_n} samples)
   Combined with stronger regularisation (reg_alpha=0.2, reg_lambda=2.0)
   Training size after SMOTE: {len(X_tr_sm):,} samples

3. MODEL PERFORMANCE COMPARISON
   ───────────────────────────────
   {'Metric':<22} {'Baseline':>12} {'ClassWeights':>14} {'SMOTE+CW':>12}
   {'─'*62}
   {'Accuracy':<22} {metrics_base['accuracy']:>12.4f} {metrics_cw['accuracy']:>14.4f} {metrics_sm['accuracy']:>12.4f}
   {'Precision (macro)':<22} {metrics_base['precision']:>12.4f} {metrics_cw['precision']:>14.4f} {metrics_sm['precision']:>12.4f}
   {'Recall (macro)':<22} {metrics_base['recall']:>12.4f} {metrics_cw['recall']:>14.4f} {metrics_sm['recall']:>12.4f}
   {'F1 Score (macro)':<22} {metrics_base['f1']:>12.4f} {metrics_cw['f1']:>14.4f} {metrics_sm['f1']:>12.4f}
   {'Sunflower bias %':<22} {sf_pred_pct:>11.1f}% {sf_cw_pct:>13.1f}% {sf_sm_pct:>11.1f}%

4. TOP PREDICTIVE FEATURES
   ──────────────────────────""")

for i, row in fi_df.head(10).iterrows():
    print(f"   #{i+1:<2} {row['feature']:<30} importance={row['importance']:.4f}  category={row['category']}")

print(f"""
   Feature category breakdown:""")
for cat, pct in cat_pct.items():
    print(f"   {'  '+cat:<28}: {pct:5.1f}%")

print(f"""
5. EXAMPLE PREDICTION COMPARISONS
   ──────────────────────────────────
   Same conditions (Cobalt, BCF=0.3, pH=6.5) in baseline vs best model:""")

if 'Co' in all_elements:
    r_base_co = []
    row_co = {col: 0.0 for col in feature_names}
    log_b = np.log1p(0.3)
    row_co['soil_concentration']  = np.log1p(50)
    row_co['plant_concentration'] = np.log1p(15)
    row_co['bcf']                 = log_b
    row_co['soil_ph']             = 6.5
    row_co['organic_matter_pct']  = 2.5
    row_co['duration_days']       = 60
    row_co['ph_x_log_bcf']        = 6.5 * log_b
    row_co['logconc_x_logbcf']    = np.log1p(50) * log_b
    row_co['bcf_category']        = 1.0
    row_co['log_duration']        = np.log1p(60)
    if 'element_Co' in row_co: row_co['element_Co'] = 1.0
    v_co = sc_main.transform(np.array([[row_co[f] for f in feature_names]], dtype=np.float32))

    p_base_co = base_model.predict_proba(v_co)[0]
    p_sm_co   = best_model.predict_proba(v_co)[0]
    t_base    = [(le_main.classes_[i], round(float(p_base_co[i])*100,1))
                 for i in np.argsort(p_base_co)[::-1][:5]]
    t_sm      = [(le_main.classes_[i], round(float(p_sm_co[i])*100,1))
                 for i in np.argsort(p_sm_co)[::-1][:5]]

    print(f"   Baseline   -> #1:{t_base[0][0]}({t_base[0][1]}%)  #2:{t_base[1][0]}({t_base[1][1]}%)  #3:{t_base[2][0]}({t_base[2][1]}%)")
    print(f"   SMOTE+CW   -> #1:{t_sm[0][0]}({t_sm[0][1]}%)  #2:{t_sm[1][0]}({t_sm[1][1]}%)  #3:{t_sm[2][0]}({t_sm[2][1]}%)")

print(f"""
6. DIVERSITY ASSESSMENT
   ─────────────────────
   Unique species as top-1 across {len(all_top1)} test scenarios : {len(unique_top1)}
   (Higher is better — goal: >5 unique species)
   {'✓ GOOD diversity' if len(unique_top1) >= 5 else '⚠ Still limited diversity — consider per-element models'}

7. OUTPUT FILES
   ─────────────
   pipeline_report/fig1_class_imbalance.png
   pipeline_report/fig2_confusion_matrices.png
   pipeline_report/fig3_per_class_f1.png
   pipeline_report/fig4_feature_importance.png
   pipeline_report/fig5_shap_importance.png
   pipeline_report/fig6_prediction_diversity.png
   pipeline_report/class_distribution.csv
   pipeline_report/per_class_report.csv
   results/best_balanced_model/  (model, scaler, encoder)
""")

print("="*70)
print("  PIPELINE COMPLETE")
print("="*70)
