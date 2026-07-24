# scripts/run_pipeline.py
"""
F1CAST End-to-End Pipeline Orchestrator

Automates the complete workflow:
1. Data Ingestion (FastF1 API -> Raw Parquet)
2. Feature Engineering (Raw Parquet -> Engineered Parquet with rolling metrics)
3. Model Training & Ranking (Engineered Parquet -> Stage 1 Classifier + Stage 2 Ranker -> Metrics & Plots)
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Ensure project root and scripts directory are in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from ingestor import run_ingestion, get_upcoming_race_grid
from engineer import build_features
from main import run_pipeline as run_model_pipeline, format_gp_name
import state as state_mgr
import notifier

def main():
    parser = argparse.ArgumentParser(
        description="F1CAST: Run automated end-to-end F1 race prediction pipeline."
    )
    parser.add_argument(
        '--start-year', type=int, default=2018,
        help="Earliest season to ingest data for (default: 2018)"
    )
    parser.add_argument(
        '--end-year', type=int, default=None,
        help="Latest season to ingest data for (default: current year)"
    )
    parser.add_argument(
        '--force-ingest', action='store_true',
        help="Force re-fetching of all race data from FastF1"
    )
    parser.add_argument(
        '--skip-ingest', action='store_true',
        help="Skip raw data ingestion step"
    )
    parser.add_argument(
        '--skip-engineer', action='store_true',
        help="Skip feature engineering step"
    )
    parser.add_argument(
        '--test-season', type=int, default=None,
        help="Target season for model evaluation split (e.g., 2026)"
    )
    parser.add_argument(
        '--test-rounds', type=int, nargs='+', default=None,
        help="Target round numbers for model evaluation split (e.g., 8 9 10)"
    )
    parser.add_argument(
        '--num-test-rounds', type=int, default=1,
        help="Number of latest completed rounds to use as evaluation test set if explicit rounds are not specified (default: 1)"
    )
    parser.add_argument(
        '--show-plot', action='store_true',
        help="Display evaluation plot interactive window"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("F1CAST AUTOMATED PIPELINE STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Ingestion
    if not args.skip_ingest:
        print("\n--- STAGE 1: RAW DATA INGESTION ---")
        ingest_res = run_ingestion(
            start_year=args.start_year,
            end_year=args.end_year,
            force_refresh=args.force_ingest
        )
        if ingest_res.get('up_to_date'):
            print(f"Ingestion complete: Dataset is already up to date. No new races to fetch (Total dataset races: {ingest_res['total_races']}).")
        else:
            print(f"Ingestion complete: {ingest_res['new_races_ingested']} new race(s) fetched. Total dataset races: {ingest_res['total_races']}.")
    else:
        print("\n--- STAGE 1: RAW DATA INGESTION (SKIPPED) ---")

    # 1.5 Pre-Race Prediction Grid Check
    upcoming_grid_df = None
    if not args.skip_ingest:
        print("\n--- CHECKING FOR UPCOMING RACES starting grid ---")
        try:
            upcoming_grid_df = get_upcoming_race_grid()
        except Exception as e:
            print(f"Notice: Could not fetch upcoming race grid ({e})")

    # If we have an upcoming starting grid, check if we need to predict it
    if upcoming_grid_df is not None:
        try:
            season = int(upcoming_grid_df['Season'].iloc[0])
            round_num = int(upcoming_grid_df['Round'].iloc[0])
            gp_name = format_gp_name(upcoming_grid_df)

            if not state_mgr.is_prediction_sent(season, round_num):
                print(f"\n[PRE-RACE] Starting grid detected for {season} R{round_num} ({gp_name}). Generating predictions...")
                
                # Combine grid temporarily with raw parquet
                raw_path = 'data/raw/pro_f1_raw.parquet'
                temp_raw_path = 'data/raw/pro_f1_raw_temp.parquet'
                temp_eng_path = 'data/engineered/pro_f1_engineered_temp.parquet'

                if os.path.exists(raw_path):
                    raw_df = pd.read_parquet(raw_path)
                    combined_df = pd.concat([raw_df, upcoming_grid_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['Season', 'Round', 'Driver'], keep='last')
                    combined_df.to_parquet(temp_raw_path, index=False)
                    
                    # Run feature engineering temporarily
                    print("[PRE-RACE] Building features for upcoming race...")
                    build_features(raw_path=temp_raw_path, output_path=temp_eng_path)
                    
                    # Run pipeline prediction temporarily
                    print("[PRE-RACE] Running model predictions...")
                    pre_eval = run_model_pipeline(
                        data_path=temp_eng_path,
                        test_season=season,
                        test_rounds=round_num,
                        num_test_rounds=1,
                        show_plot=False
                    )
                    
                    if pre_eval and pre_eval.get('last_pred_json'):
                        pred_file = pre_eval['last_pred_json']
                        with open(pred_file, 'r', encoding='utf-8') as f:
                            pred_data = json.load(f)
                        
                        subject = f"F1CAST Prediction: {season} Round {round_num} ({gp_name})"
                        html = notifier.format_prediction_email(season, round_num, gp_name, pred_data['predictions'])
                        
                        print("[PRE-RACE] Dispatching prediction email...")
                        dispatched = notifier.send_email(subject, html)
                        if dispatched:
                            state_mgr.mark_prediction_sent(season, round_num)
                            print(f"[STATE] Marked pre-race prediction state as SENT for {season} R{round_num}.")
                        else:
                            print(f"[STATE WARNING] Pre-race email dispatch failed. Leaving state pending.")
                    
                    # Clean up temp files
                    for path in [temp_raw_path, temp_eng_path]:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                else:
                    print("[PRE-RACE ERROR] Raw parquet file not found. Skipping pre-race predictions.")
            else:
                print(f"\n[PRE-RACE] Prediction email already sent for {season} R{round_num} ({gp_name}). Skipping.")
        except Exception as pre_err:
            print(f"[PRE-RACE ERROR] Failed during pre-race prediction flow: {pre_err}")

    # 2. Feature Engineering
    if not args.skip_engineer:
        print("\n--- STAGE 2: FEATURE ENGINEERING ---")
        eng_res = build_features()
        if not eng_res:
            print("ERROR: Feature engineering failed. Exiting.")
            sys.exit(1)
        print(f"Feature engineering complete: {eng_res['rows']} rows across {eng_res['total_races']} races saved to {eng_res['output_path']}.")
    else:
        print("\n--- STAGE 2: FEATURE ENGINEERING (SKIPPED) ---")

    # 3. Model Training & Evaluation
    print("\n--- STAGE 3: MODEL TRAINING & EVALUATION ---")
    eval_res = run_model_pipeline(
        test_season=args.test_season,
        test_rounds=args.test_rounds,
        num_test_rounds=args.num_test_rounds,
        show_plot=args.show_plot
    )

    if not eval_res:
        print("ERROR: Model training/evaluation failed. Exiting.")
        sys.exit(1)

    # 4. State Management & Email Dispatches
    print("\n--- STAGE 4: STATE MANAGEMENT & EMAIL NOTIFICATIONS ---")
    comp_file = eval_res.get('last_comp_json')
    pred_file = eval_res.get('last_pred_json')

    if comp_file and os.path.exists(comp_file):
        try:
            with open(comp_file, 'r', encoding='utf-8') as f:
                comp_data = json.load(f)
            
            season = comp_data['season']
            round_num = comp_data['round']
            gp_name = comp_data['gp_name']
            spearman_corr = comp_data['spearman_correlation']
            comp_list = comp_data['comparison']

            if not state_mgr.is_comparison_sent(season, round_num):
                print(f"[STATE] Post-race comparison email pending for {season} R{round_num} ({gp_name}). Processing report...")
                subject = f"F1CAST Results: {season} Round {round_num} ({gp_name})"
                html = notifier.format_comparison_email(season, round_num, gp_name, comp_list, spearman_corr)
                dispatched = notifier.send_email(subject, html)
                if dispatched:
                    state_mgr.mark_comparison_sent(season, round_num)
                    print(f"[STATE] Marked post-race comparison state for {season} R{round_num}.")
                else:
                    print(f"[STATE WARNING] Email notification dispatch failed. Leaving state as pending.")
            else:
                print(f"[STATE] Post-race comparison email already sent for {season} R{round_num} ({gp_name}). Skipping dispatch.")
        except Exception as err:
            print(f"[STATE ERROR] Could not process comparison notification: {err}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Training races: {eval_res['train_races']} ({eval_res['train_rows']} rows)")
    print(f"Test races:     {eval_res['test_races']} ({eval_res['test_rows']} rows)")
    print(f"Average Spearman Correlation: {eval_res['avg_spearman']:.3f}")
    print(f"Results CSV:    {eval_res['results_csv']}")
    print(f"Evaluation Plot:{eval_res['plot_png']}")
    if eval_res.get('last_pred_json'):
        print(f"Prediction JSON:{eval_res['last_pred_json']}")
    if eval_res.get('last_comp_json'):
        print(f"Comparison JSON:{eval_res['last_comp_json']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

