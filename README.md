# 🌿 PhytoRecommend — AI Phytoremediation Plant Recommender

An AI-powered web app that recommends the best plant species for remediating
contaminated soil. Given site conditions (pollutant element, BCF, soil pH,
organic matter, exposure duration), the model predicts the top-5 plant species
most suitable for phytoremediation.

Trained on **36,599 real phytoremediation studies** across **56 plant species**
and **15 contaminant elements** (Pb, Cd, Zn, Cu, As, Ni, Cr, Co, Hg, Se, Mn,
Fe, Tl, Mo, Sb).

---

## Project Structure

```
phytoremediation-ai/
│
├── data/
│   └── phytoremediation_dataset.csv   ← Cleaned dataset (36k records)
│
├── models/
│   └── per_element_models/            ← 15 trained XGBoost models (one per element)
│       ├── Pb/  Cd/  Zn/  Cu/  As/
│       ├── Ni/  Cr/  Co/  Hg/  Se/
│       ├── Mn/  Fe/  Tl/  Mo/  Sb/
│       └── models_meta.json
│
├── pipeline/
│   └── 06_retrain_per_element.py      ← Script to retrain all models
│
├── webapp/
│   ├── app.py                         ← Flask backend
│   └── templates/
│       └── index.html                 ← Frontend UI
│
├── Procfile                           ← Render deployment config
├── runtime.txt                        ← Python 3.11 pin
├── requirements.txt                   ← Dependencies
└── README.md
```

---

## Live Demo

Deployed on Render: [https://phytorecommend.onrender.com](https://phytorecommend.onrender.com)

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vineshreddy987/PhytoRecommend.git
cd PhytoRecommend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the web app
python webapp/app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Inputs

| Field | Description | Example |
|---|---|---|
| Contaminant Element | Pollutant in the soil | `Pb` (Lead) |
| Soil Concentration | Amount in soil (mg/kg) | `250` |
| Plant Concentration | Amount absorbed by plant (mg/kg) | `45` |
| BCF | Bioconcentration Factor = plant/soil | `0.18` (auto-calculated) |
| Soil pH | pH of contaminated soil | `5.2` |
| Organic Matter | % organic matter in soil | `1.5` |
| Exposure Duration | Days plant grows in soil | `60` |

**BCF > 1** = hyperaccumulator — plant takes up more than the soil has, ideal for phytoremediation.

## Output

Top-5 recommended plant species with confidence scores, ranked by suitability
for the given contamination conditions.

---

## Retrain Models

To retrain all 15 per-element models from scratch:

```bash
python pipeline/06_retrain_per_element.py
```

Models are saved to `models/per_element_models/`.

---

## Tech Stack

- **Backend:** Python, Flask, XGBoost, scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render
- **Models:** 15 per-element XGBoost classifiers with SMOTE + class weights
