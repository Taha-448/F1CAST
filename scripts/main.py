# scripts/main.py
import os
import pandas as pd
from scipy.stats import spearmanr
from model_pipeline import F1RankingModel

def run_pipeline():
    data_path = 'data/engineered/pro_f1_engineered.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run ingestor.py and engineer.py first.")
        return

    df = pd.read_parquet(data_path)
    
    features = [
        'GridPosition', 'DriverForm', 'TeamForm', 'Team_DNF_Rate', 
        'Track_Cat_Code', 'Era_Code', 'Team_Code', 'stress', 'abrasion', 
        'overtake_diff', 'speed_cat', 'downforce', 'brake_severity', 'altitude_m'
    ]
    
    model = F1RankingModel(features)

    # Train split: 2018 to 2026 Round 7
    train_df = df[(df['Season'] < 2026) | ((df['Season'] == 2026) & (df['Round'] <= 7))].copy()
    # Test split: 2026 Rounds 8 and 9
    test_df = df[(df['Season'] == 2026) & (df['Round'].isin([8, 9]))].copy()

    if train_df.empty:
        print("Error: Train dataset is empty. Ensure you have ingested the historical data.")
        return
    if test_df.empty:
        print("Error: Test dataset is empty. Ensure 2026 Rounds 8 and 9 have been ingested.")
        return

    # Sort training data by Season and Round to ensure group grouping works correctly in XGBRanker
    train_df = train_df.sort_values(['Season', 'Round'])
    test_df = test_df.sort_values(['Season', 'Round'])

    print(f"Training model on {len(train_df)} rows (2018 to 2026 Round 7)...")
    model.fit(train_df)
    
    print(f"Evaluating model on {len(test_df)} rows (2026 Rounds 8 and 9)...")
    scores, dnf_p = model.predict(test_df)
    
    test_df['Predicted_Score'] = scores
    test_df['DNF_Risk'] = dnf_p
    
    # Calculate ranks for each round separately (higher Predicted_Score means better rank/lower rank number)
    test_df['Predicted_Rank'] = test_df.groupby('Round')['Predicted_Score'].rank(ascending=False)

    os.makedirs('logs', exist_ok=True)
    results_path = 'logs/2026_test_results.csv'
    test_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # Display evaluation metrics
    print("\n=== EVALUATION RESULTS ===")
    test_rounds = sorted(test_df['Round'].unique())
    correlations = []
    
    for r in test_rounds:
        round_data = test_df[test_df['Round'] == r]
        # Calculate Spearman Correlation for this race
        coef, _ = spearmanr(round_data['Predicted_Rank'], round_data['FinishPos'])
        print(f"2026 Round {r} Spearman Correlation: {coef:.3f}")
        correlations.append(coef)
        
    if correlations:
        print(f"Average Spearman Correlation: {sum(correlations) / len(correlations):.3f}")

if __name__ == "__main__":
    run_pipeline()