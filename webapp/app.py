"""
app.py  —  Flask backend for PhytoRecommend
Uses per-element XGBoost models trained by pipeline/06_retrain_per_element.py
Run from the repo root:  python webapp/app.py
"""

import os, pickle, json
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── Resolve paths relative to this file ──────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(HERE)
MODELS_DIR = os.path.join(REPO_ROOT, "models", "per_element_models")

if not os.path.exists(MODELS_DIR):
    raise RuntimeError(
        f"Models directory not found: {MODELS_DIR}\n"
        "Run  python pipeline/06_retrain_per_element.py  first."
    )

# ── Load all per-element models at startup ────────────────────────────────────
element_models = {}

with open(os.path.join(MODELS_DIR, "models_meta.json")) as f:
    models_meta = json.load(f)

for element in models_meta:
    el_dir = os.path.join(MODELS_DIR, element)
    with open(os.path.join(el_dir, "model.pkl"),         "rb") as f: m  = pickle.load(f)
    with open(os.path.join(el_dir, "label_encoder.pkl"), "rb") as f: le = pickle.load(f)
    with open(os.path.join(el_dir, "scaler.pkl"),        "rb") as f: sc = pickle.load(f)
    with open(os.path.join(el_dir, "feature_names.txt"))       as f:
        fn = [l.strip() for l in f.readlines()]
    element_models[element] = {"model": m, "le": le, "scaler": sc, "feature_names": fn}

print(f"[INFO] Loaded {len(element_models)} per-element models: {sorted(element_models)}")

# ── Element metadata ──────────────────────────────────────────────────────────
ELEMENTS = sorted(element_models.keys())

ELEMENT_NAMES = {
    'As':'Arsenic (As)',   'Cd':'Cadmium (Cd)',   'Co':'Cobalt (Co)',
    'Cr':'Chromium (Cr)',  'Cu':'Copper (Cu)',    'Fe':'Iron (Fe)',
    'Hg':'Mercury (Hg)',   'Mn':'Manganese (Mn)', 'Mo':'Molybdenum (Mo)',
    'Ni':'Nickel (Ni)',    'Pb':'Lead (Pb)',       'Sb':'Antimony (Sb)',
    'Se':'Selenium (Se)',  'Tl':'Thallium (Tl)',  'Zn':'Zinc (Zn)',
}

TOXICITY = {
    'Pb':'High','Cd':'High','Hg':'High','As':'High','Cr':'High','Tl':'High',
    'Sb':'Medium','Ni':'Medium','Cu':'Medium','Zn':'Medium','Co':'Medium',
    'Mo':'Medium','Se':'Medium','Mn':'Low','Fe':'Low',
}

app = Flask(__name__)


def build_and_predict(soil_conc, plant_conc, bcf, soil_ph,
                      organic_pct, duration, element, top_k=5):
    if element not in element_models:
        raise ValueError(f"No model for element: {element}")

    m_data        = element_models[element]
    model         = m_data["model"]
    le            = m_data["le"]
    scaler        = m_data["scaler"]
    feature_names = m_data["feature_names"]

    log_bcf = np.log1p(bcf)
    row = {col: 0.0 for col in feature_names}
    row['soil_concentration']  = np.log1p(soil_conc)
    row['plant_concentration'] = np.log1p(plant_conc)
    row['bcf']                 = log_bcf
    row['soil_ph']             = soil_ph
    row['organic_matter_pct']  = organic_pct
    row['duration_days']       = duration
    if 'ph_x_log_bcf' in row:   row['ph_x_log_bcf']   = soil_ph * log_bcf
    if 'bcf_gt1'  in row:       row['bcf_gt1']         = float(log_bcf > np.log1p(1.0))
    if 'bcf_lt01' in row:       row['bcf_lt01']        = float(log_bcf < np.log1p(0.1))

    vec    = np.array([[row[f] for f in feature_names]], dtype=np.float32)
    scaled = scaler.transform(vec)
    proba  = model.predict_proba(scaled)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]

    return [{
        "rank":       rank + 1,
        "species":    le.classes_[i],
        "confidence": round(float(proba[i]) * 100, 1),
    } for rank, i in enumerate(top_idx)]


@app.route("/")
def index():
    return render_template("index.html",
                           elements=ELEMENTS,
                           element_names=ELEMENT_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json()
        element = data["element"]
        results = build_and_predict(
            soil_conc   = float(data["soil_concentration"]),
            plant_conc  = float(data["plant_concentration"]),
            bcf         = float(data["bcf"]),
            soil_ph     = float(data["soil_ph"]),
            organic_pct = float(data["organic_matter_pct"]),
            duration    = float(data["duration_days"]),
            element     = element,
        )
        return jsonify({
            "status":    "ok",
            "results":   results,
            "toxicity":  TOXICITY.get(element, "Medium"),
            "element":   element,
            "n_species": models_meta[element]["n_classes"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
