# F1CAST: Formula 1 Race Ranking & DNF Predictor

F1CAST is a machine learning pipeline designed to predict the final finishing positions of Formula 1 drivers and estimate their risk of DNF (Did Not Finish) during a race. It uses historical race data, track-specific metrics, and a two-stage modeling approach (classification + learning-to-rank).

---

## 📊 Pipeline Architecture

The pipeline consists of two stages implemented in **`xgboost`**:

1. **Stage 1 (DNF Classifier)**: An `XGBClassifier` predicts the binary probability that a driver will DNF due to mechanical failure, crashes, or track conditions.
2. **Stage 2 (Grid Ranker)**: An `XGBRanker` (optimizing for NDCG) takes the starting grid, team/driver rolling form, circuit metadata, and the **DNF probability** from Stage 1 to predict the final relative finishing positions of all drivers in a race.

```
                  ┌──────────────────────┐
                  │  Engineered Features  │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌──────────────────────┐            ┌──────────────────┐
│  Stage 1: Classifier │            │  Stage 2: Ranker │
│   (Predicts DNF)     │            │   (XGBRanker)    │
└──────────┬───────────┘            └────────┬─────────┘
           │                                 ▲
           └─────────► DNF_Risk ─────────────┘
                       (Probability)
```

---

## 📂 Project Structure

```text
├── data/                       # Dataset directories (ignored by git)
│   ├── raw/                    # Raw Parquet outputs from ingestion
│   └── engineered/             # Final engineered Parquet features
├── cache/                      # FastF1 download and sqlite caches (ignored by git)
├── logs/                       # Prediction csv exports (ignored by git)
├── scripts/                    # Python pipeline scripts
│   ├── ingestor.py             # Fetches Ergast/FastF1 API data
│   ├── metadata.py             # Defines static circuit characteristics
│   ├── engineer.py             # Computes rolling forms & merges track metadata
│   ├── model_pipeline.py       # Two-stage XGBoost model implementation
│   ├── main.py                 # Splits data, trains model, and evaluates metrics
│   └── test.py                 # FastF1 plotting demonstration
├── .gitignore                  # Git ignore file (excludes caches/data/logs)
└── README.md                   # Project documentation
```

---

## ⚙️ Engineered Features

F1CAST uses a variety of engineered features to capture driver momentum, reliability, and track profile:
* **GridPosition**: The driver's starting grid slot.
* **DriverForm / TeamForm**: 5-race rolling average of finishing positions (shifted to prevent lookahead leakage).
* **Team_DNF_Rate**: 10-race rolling average of team DNF history.
* **Track Profile**: Circuit characteristics including tyre stress, asphalt abrasion, overtake difficulty, downforce configuration, brake severity, and altitude.
* **Era_Code**: Categorical split indicating the technical car regulation eras (e.g., 2014, 2017, 2022, 2026+).
* **SampleWeight**: Time-decay weight applied during training to prioritize recent race outcomes.

---

## 🚀 Getting Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install pandas numpy xgboost fastf1 scipy scikit-learn matplotlib
```

### 2. Ingest Historical Data
Downloads and consolidates F1 race histories from the API:
```bash
python scripts/ingestor.py
```

### 3. Generate Features
Computes the rolling metrics and merges track profile metadata:
```bash
python scripts/engineer.py
```

### 4. Train & Evaluate
Trains the XGBoost models and evaluates predictions for 2026 Rounds 8 & 9:
```bash
python scripts/main.py
```

---

## 📈 Evaluation Performance

Performance is evaluated using the **Spearman Rank Correlation Coefficient** (comparing predicted rank to actual finishing position):
* **2026 Round 8**: `0.913`
* **2026 Round 9**: `0.598`
* **Average Correlation**: `0.756`

Predictions for the test sets are saved locally in `logs/2026_test_results.csv`.
