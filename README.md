# 🌿 PhytoRecommend — AI Phytoremediation Plant Recommender

An AI-powered web application that recommends the best plant species for
remediating contaminated soil. Given environmental conditions such as pollutant
element, BCF, soil pH, organic matter, and exposure duration, the model
predicts the top-5 most suitable plant species for phytoremediation.

**Trained on 36,599 real phytoremediation studies | 56 plant species | 15 contaminant elements**

---

## 📁 Project Structure

```
phytoremediation-ai/
│
├── data/
│   └── phytoremediation_dataset.csv
│           The cleaned and merged dataset containing 36,599 records
│           with columns: species, element, soil_concentration,
│           plant_concentration, bcf, soil_ph, organic_matter_pct,
│           duration_days, source
│
├── models/
│   └── per_element_models/
│       ├── As/                    Arsenic model files
│       ├── Cd/                    Cadmium model files
│       ├── Co/                    Cobalt model files
│       ├── Cr/                    Chromium model files
│       ├── Cu/                    Copper model files
│       ├── Fe/                    Iron model files
│       ├── Hg/                    Mercury model files
│       ├── Mn/                    Manganese model files
│       ├── Mo/                    Molybdenum model files
│       ├── Ni/                    Nickel model files
│       ├── Pb/                    Lead model files
│       ├── Sb/                    Antimony model files
│       ├── Se/                    Selenium model files
│       ├── Tl/                    Thallium model files
│       ├── Zn/                    Zinc model files
│       └── models_meta.json       Metadata (accuracy, classes, features per model)
│
│       Each element folder contains:
│           model.pkl              Trained XGBoost classifier
│           label_encoder.pkl      Encodes/decodes species names
│           scaler.pkl             StandardScaler for input features
│           feature_names.txt      List of feature names used during training
│
├── pipeline/
│   ├── 00_setup_and_split.py
│   │       Loads the raw dataset, preprocesses it, performs stratified
│   │       80/20 train-test split and saves split arrays to disk.
│   │       Run this FIRST before any benchmark training.
│   │
│   ├── 01_train_xgboost.py
│   │       Trains a global XGBoost classifier on the full split.
│   │       Generates metrics, confusion matrix, ROC curves and SHAP plots.
│   │
│   ├── 02_train_ann.py
│   │       Trains a deep MLP (ANN) using PyTorch.
│   │       Generates metrics, learning curves, confusion matrix, ROC curves.
│   │
│   ├── 03_train_tabtransformer.py
│   │       Trains a TabTransformer (Huang et al. 2020) on the tabular data.
│   │       Generates metrics, attention heatmap, learning curves.
│   │
│   ├── 04_train_self_attention_transformer.py
│   │       Trains the proposed 12-Head Self-Attention Transformer with
│   │       CLS token, Pre-LayerNorm, warmup + cosine LR schedule.
│   │       Generates per-head attention maps.
│   │
│   ├── 05_aggregate_results.py
│   │       Collects all metrics.json files from trained models and produces
│   │       an IEEE-formatted benchmark comparison table (CSV + LaTeX),
│   │       bar charts, top-k comparison, and timing plots.
│   │
│   ├── 06_retrain_per_element.py      ⭐ MAIN MODEL TRAINING SCRIPT
│   │       Trains one XGBoost classifier per contaminant element (15 models).
│   │       Applies element-wise class balancing, log-transforms, interaction
│   │       features, and inverse-frequency sample weights.
│   │       Saves all models to models/per_element_models/
│   │       This is the script that produces the models used by the webapp.
│   │
│   └── 07_full_pipeline_report.py
│           Runs a complete class imbalance analysis, applies SMOTE + class
│           weights, compares baseline vs balanced model, generates SHAP
│           feature importance, and produces a full PDF-ready report with
│           6 figures and 2 CSV tables.
│
├── webapp/
│   ├── app.py
│   │       Flask backend. Loads all 15 per-element models at startup.
│   │       Exposes two routes:
│   │         GET  /          → serves the HTML frontend
│   │         POST /predict   → accepts JSON input, returns top-5 predictions
│   │
│   └── templates/
│       └── index.html
│               Single-page frontend. BCF auto-calculates from soil and plant
│               concentrations. Displays ranked results with confidence bars,
│               toxicity badge, and known best accumulators from literature.
│
├── Dockerfile
│       Builds a Docker container with Python 3.11-slim, installs all
│       dependencies, and starts the app using Gunicorn.
│
├── .dockerignore
│       Excludes __pycache__, .git, .env and log files from Docker image.
│
├── Procfile
│       Tells Render how to start the app: gunicorn webapp.app:app
│
├── runtime.txt
│       Pins Python version to 3.11.9 for Render deployment.
│
├── requirements.txt
│       All Python dependencies with pinned versions.
│
├── .gitignore
│       Excludes venv, __pycache__, .npy arrays, .xlsx files, logs.
│
└── README.md
        This file.
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/vineshreddy987/PhytoRecommend.git
cd PhytoRecommend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app (models are already included)
python webapp/app.py

# 4. Open in browser
http://localhost:5000
```

> Models are pre-trained and included in `models/`. No training needed to run the app.

---

## 🔁 Retrain Models from Scratch

```bash
# Train the 15 per-element models used by the webapp
python pipeline/06_retrain_per_element.py

# Optional: run full benchmark comparison (XGBoost, ANN, TabTransformer, Transformer)
python pipeline/00_setup_and_split.py --data data/phytoremediation_dataset.csv --label species
python pipeline/01_train_xgboost.py
python pipeline/02_train_ann.py
python pipeline/03_train_tabtransformer.py
python pipeline/04_train_self_attention_transformer.py
python pipeline/05_aggregate_results.py

# Optional: class imbalance analysis and SMOTE report
python pipeline/07_full_pipeline_report.py
```

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect `vineshreddy987/PhytoRecommend`
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn webapp.app:app`
   - **Instance Type:** Free
5. Click **Deploy**

---

## 🐳 Deploy with Docker

```bash
# Build image
docker build -t phytorecommend .

# Run locally
docker run -p 5000:5000 phytorecommend

# Push to Docker Hub
docker tag phytorecommend vineshreddy987/phytorecommend:v1
docker push vineshreddy987/phytorecommend:v1
```

---

## 📥 Inputs

| Field | Description | Example |
|---|---|---|
| Contaminant Element | Pollutant in the soil | `Pb` (Lead) |
| Soil Concentration | Amount in soil (mg/kg) | `250` |
| Plant Concentration | Amount absorbed by plant (mg/kg) | `45` |
| BCF | Bioconcentration Factor = plant ÷ soil | `0.18` (auto-calculated) |
| Soil pH | pH of contaminated soil | `5.2` |
| Organic Matter | % organic matter in soil | `1.5` |
| Exposure Duration | Days the plant grows in the soil | `60` |

**BCF > 1** = hyperaccumulator — plant concentrates more than the soil has. Best for phytoremediation.

## 📤 Output

Top-5 recommended plant species ranked by confidence score for the given site conditions.

---

## 🧠 Model Architecture

- **15 per-element XGBoost classifiers** — one model per contaminant element
- Each model trained only on data for that element, eliminating cross-element bias
- Features: `log(BCF)`, `log(soil_conc)`, `log(plant_conc)`, `soil_pH`, `organic_matter`, `log(duration)`, `pH × log(BCF)` interaction, `BCF category`
- Class imbalance handled with: SMOTE oversampling + inverse-frequency sample weights

## 📊 Benchmark Results (Global Model Comparison)

| Model | Accuracy | Top-3 | ROC-AUC |
|---|---|---|---|
| **XGBoost** | **49.5%** | **69.3%** | **0.948** |
| TabTransformer | 38.9% | 57.8% | 0.915 |
| ANN (MLP) | 24.2% | 43.5% | 0.818 |
| 12-Head Self-Attention Transformer | 19.9% | 42.0% | 0.720 |

---

## 🛠 Tech Stack

- **Backend:** Python 3.11, Flask, XGBoost, scikit-learn, NumPy, pandas
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Deployment:** Render / Docker
- **Training:** XGBoost, PyTorch (ANN, TabTransformer, Transformer)
