# FPL Decision Support System

A Machine Learning-Based Decision Support System for Fantasy Premier League, developed as a Final Degree Project at the Escola Tècnica Superior d'Enginyeria Informàtica (ETSINF), Universitat Politècnica de València.

---

## Overview

This system predicts FPL player points using AutoML and deep learning, selects optimal squads via integer linear programming, and delivers recommendations through an interactive Streamlit dashboard. A season simulation over GW15–29 of the 2025/26 season achieved **797 points vs a 702 average** — an estimated global rank of **top 50,000 out of 10M+ managers (top 0.5%)**.

---

## Project Structure
The structure below reflects the repository after all scripts and notebooks have been run.

```
fpl-dss/
├── data/
│   ├── raw_historical/          ← vaastav CSVs for 2023/24 and 2024/25
│   ├── raw_live/                ← vaastav CSV for 2025/26 (GW1–29)
│   └── processed/
│       ├── final_feature_set.csv
│       ├── nb2_regression_featured/
│       │   ├── predictions.csv
│       │   └── predictions_pergw.csv
│       └── nb3_position_models/
│           └── predictions.csv
├── models/
│   ├── nb1_regression_all/
│   │   ├── pycaret_best_model.pkl
│   │   └── autokeras_best_model.keras
│   ├── nb2_regression_featured/
│   │   ├── huber_model.pkl
│   │   ├── scaler.pkl
│   │   ├── imputer.pkl
│   │   └── autokeras_best_model.keras
│   └── nb3_position_models/
│       ├── huber_{GK,DEF,MID,FWD}.pkl
│       ├── scaler_{GK,DEF,MID,FWD}.pkl
│       └── imputer_{GK,DEF,MID,FWD}.pkl
├── notebooks/
│   ├── nb1_regression_alldata.ipynb       ← Experiment 1: all columns
│   ├── nb2_regression_featured.ipynb      ← Experiment 2: engineered features + weighting
│   ├── nb3_position_models.ipynb          ← Experiment 3: position-specific models
│   ├── nb4_comparison.ipynb               ← Comparative Evaluation
│   └── nb5_simulation.ipynb               ← Season Simulation
├── outputs/
├── src/
│   ├── filter_data.py                     ← Season concatenation & preprocessing
│   ├── feature_engineering.py             ← 27 engineered features
│   └── dashboard.py                       ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Azizbhs/Fantasy-Premier-League-Decision-Support-System
cd fpl-dss
```

### 2. Install dependencies

Python 3.10 is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** PyCaret and AutoKeras have conflicting TensorFlow dependencies. Train them in separate virtual environments or separate Google Colab sessions. The dashboard only requires the saved model files and does not need either library installed.

### 3. Prepare the data

Download the vaastav dataset for each season from [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) and place the CSV files in:

```
data/raw_historical/    ← 2023/24 and 2024/25 season files
data/raw_live/          ← 2025/26 season file
```

Then run the preprocessing pipeline:

```bash
python src/filter_data.py
python src/feature_engineering.py
```

This produces `data/processed/final_feature_set.csv`.

### 4. Run the notebooks

Run the notebooks in order:

| Notebook | Description |
|----------|-------------|
| `nb1_regression_alldata.ipynb` | Experiment 1 — PyCaret + AutoKeras on all columns |
| `nb2_regression_featured.ipynb` | Experiment 2 — engineered features + sample weighting |
| `nb3_position_models.ipynb` | Experiment 3 — position-specific Huber models |
| `nb4_comparison.ipynb` | Comparative evaluation across all configurations |
| `nb5_simulation.ipynb` | Season simulation (GW15–29) |

> Notebooks were developed on **Google Colab** with a T4 GPU. All model artefacts are saved to `models/` and all output figures to `outputs/`.

### 5. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard reads predictions from `models/nb2_regression_featured/predictions.csv` and connects to the live FPL API automatically. No model files need to be loaded at runtime.

---

## Experimental Results

| Configuration | Model | MAE | R² | vs Baseline |
|--------------|-------|-----|----|-------------|
| Naïve Baseline | — | 1.505 | — | — |
| NB1 (All columns) | Huber Regressor | **0.834** | 0.248 | −44.6% |
| NB1 (All columns) | AutoKeras | 1.076 | **0.338** | −28.5% |
| NB2 (Featured + weighted) | Huber Regressor | 0.864 | 0.213 | −42.6% |
| NB2 (Featured + weighted) | AutoKeras | 1.022 | 0.317 | −32.1% |

NB2 Huber was selected as the final model based on FPL-relevant metrics: **MAE@15 = 7.34**, **Precision@15 = 0.181**.

### Season Simulation (GW15–29, 2025/26)

| Metric | Our Model | Average Manager |
|--------|-----------|----------------|
| Total points | **797** | 702 |
| Points advantage | +95 | — |
| Mean pts/GW | **56.9** | 50.1 |
| Estimated rank | **Top 50,000 (top 0.5%)** | — |

---

## Dashboard Pages

- **Overview** — System metrics and model performance comparison
- **Best Squad** — ILP-optimal 15-player squad for the upcoming gameweek
- **My Squad** — Import your FPL team via team ID and get transfer recommendations
- **Player Predictions** — Filterable table of all 820 players ranked by predicted score
- **Captain Picks** — Top 10 captaincy recommendations with double-points projections
- **Fixtures** — Live fixture difficulty ratings for the upcoming gameweek
- **Season Simulation** — GW15–29 results vs average FPL manager

---

## Requirements

```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.2
pycaret==3.3.2
autokeras==3.0.0
tensorflow==2.18.0
lightgbm==4.6.0
pulp==3.3.0
streamlit==1.56.0
requests
matplotlib
scipy
```

---

## Data Sources

- **vaastav/Fantasy-Premier-League** — [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- **FPL Official API** — [https://fantasy.premierleague.com/api/](https://fantasy.premierleague.com/api/)

---

## License

This project was developed for academic purposes as a Final Degree Project at UPV. All data sources are used in accordance with their respective licences.
