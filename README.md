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
├── .github/workflows/         # GitHub Actions workflows
│   └── f1_automation.yml      # Serverless pipeline cron & manual trigger
├── data/                       # Dataset directories
│   ├── raw/                # Raw Parquet outputs (pro_f1_raw.parquet whitelisted)
│   └── engineered/         # Final engineered Parquet features
├── cache/                      # FastF1 download and sqlite caches (ignored by git)
├── predictions/                # Pre-race prediction grid JSON outputs
├── comparisons/                # Post-race prediction accuracy JSON outputs
├── logs/                       # Prediction csv exports and charts (ignored by git)
├── scripts/                    # Python pipeline scripts
│   ├── ingestor.py             # Fetches Ergast/FastF1 API data
│   ├── metadata.py             # Defines static circuit characteristics
│   ├── engineer.py             # Computes rolling forms & merges track metadata
│   ├── model_pipeline.py       # Two-stage XGBoost model implementation
│   ├── main.py                 # Splits data, trains model, and evaluates metrics
│   ├── state.py            # Idempotency and runner state manager
│   ├── notifier.py         # Formats and dispatches Resend HTML emails
│   └── run_pipeline.py     # Orchestrator running Stage 1-4
├── state.json                  # Tracks dispatched notification rounds
├── requirements.txt            # Python dependencies lists
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
 
## 🤖 Serverless Automation (GitHub Actions)
 
F1CAST includes an automated serverless workflow running daily on **GitHub Actions** (`.github/workflows/f1_automation.yml`):
* **Trigger Schedule**: Runs daily at **`09:00 AM PKT`** (04:00 UTC).
* **Pre-Race Flow**: Automatically detects if Qualifying results are available for an upcoming GP. If a grid is found and has not been predicted yet:
  * Trains the model on all historical races, runs predictions for the upcoming race, writes `predictions/pred_*.json`, and sends a pre-race prediction email.
* **Post-Race Flow**: Detects when race results are official. If the comparison has not been sent:
  * Downloads the race classification, appends to the dataset, retrains the model, generates comparison reports, writes `comparisons/comp_*.json`, and sends a post-race results analysis email.
* **Auto-Commit**: Automatically pushes updated data files (`data/raw/pro_f1_raw.parquet`), state tracking (`state.json`), predictions, and comparisons back to your repository.
 
### 🔑 Repository Secrets Configuration
To enable email dispatches via Resend API:
1. Register on [Resend.com](https://resend.com) (free tier includes 3,000 emails/month).
2. Set the following secrets under **Settings** → **Secrets and variables** → **Actions**:
   * `RESEND_API_KEY`: Your Resend API Key (`re_...`).
   * `NOTIFICATION_EMAIL`: Recipient email address for notifications.
   * `SENDER_EMAIL`: *(Optional)* Custom verified sender email (defaults to `onboarding@resend.dev`).
 
---
 
## 🚀 Getting Started
 
### 1. Installation
Install the necessary python dependencies using requirements.txt:
```bash
pip install -r requirements.txt
```
 
### 2. Run Automated Pipeline (Single Command)
Runs ingestion, feature engineering, and model training/evaluation automatically in sequence:
```bash
python scripts/run_pipeline.py
```
 
#### Custom Pipeline Options:
```bash
# Ingest up to latest available races and evaluate on the single most recent race (trains on all prior races)
python scripts/run_pipeline.py
 
# Force re-downloading race data
python scripts/run_pipeline.py --force-ingest
 
# Skip ingestion & feature engineering if data is already prepared
python scripts/run_pipeline.py --skip-ingest --skip-engineer
 
# Explicitly test on specific season and rounds
python scripts/run_pipeline.py --test-season 2026 --test-rounds 8 9 10
```
 
### 3. Modular Step-by-Step Execution (Optional)
If you prefer running individual pipeline modules manually:
```bash
python scripts/ingestor.py     # Ingest raw FastF1 race data
python scripts/engineer.py     # Generate rolling form & track features
python scripts/main.py         # Train model & calculate predictions
```
 
---
 
## 📈 Evaluation Performance
 
Performance is evaluated using the **Spearman Rank Correlation Coefficient** (comparing predicted rank to actual finishing position).
 
Predictions for test sets are saved automatically in `logs/latest_test_results.csv` and visualization charts are saved to `logs/actual_vs_predicted_ranks.png`.
