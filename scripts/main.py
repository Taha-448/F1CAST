# scripts/main.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # Train split: 2018 to 2026 Round 5
    train_df = df[(df['Season'] < 2026) | ((df['Season'] == 2026) & (df['Round'] <= 5))].copy()
    # Test split: 2026 Rounds 6 to 9
    test_df = df[(df['Season'] == 2026) & (df['Round'].isin([6, 7, 8, 9]))].copy()

    if train_df.empty:
        print("Error: Train dataset is empty. Ensure you have ingested the historical data.")
        return
    if test_df.empty:
        print("Error: Test dataset is empty. Ensure 2026 Rounds 6 to 9 have been ingested.")
        return

    # Sort training data by Season and Round to ensure group grouping works correctly in XGBRanker
    train_df = train_df.sort_values(['Season', 'Round'])
    test_df = test_df.sort_values(['Season', 'Round'])

    print(f"Training model on {len(train_df)} rows (2018 to 2026 Round 5)...")
    model.fit(train_df)
    
    print(f"Evaluating model on {len(test_df)} rows (2026 Rounds 6 to 9)...")
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

    # Plot results
    plot_results(test_df)

def plot_results(test_df):
    rounds = sorted(test_df['Round'].unique())
    num_rounds = len(rounds)
    
    # Elegant styling setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, num_rounds, figsize=(6 * num_rounds, 6), sharey=True)
    if num_rounds == 1:
        axes = [axes]
        
    for i, r in enumerate(rounds):
        ax = axes[i]
        round_data = test_df[test_df['Round'] == r].copy()
        
        # Sort by actual finish position
        round_data = round_data.sort_values('FinishPos')
        
        # Dynamic axis limits
        max_val = max(int(round_data['FinishPos'].max()), int(round_data['Predicted_Rank'].max()))
        if max_val < 20:
            max_val = 20
            
        # Perfect prediction reference diagonal
        ax.plot([1, max_val], [1, max_val], color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5, label='Perfect Prediction')
        
        # Absolute prediction error for color coding points
        errors = np.abs(round_data['FinishPos'] - round_data['Predicted_Rank'])
        
        # Scatter actual vs predicted positions
        scatter = ax.scatter(
            round_data['FinishPos'], 
            round_data['Predicted_Rank'], 
            c=errors, 
            cmap='viridis_r', 
            s=120, 
            edgecolors='black', 
            linewidths=0.8,
            zorder=3, 
            alpha=0.9
        )
        
        # Annotate each driver's abbreviation
        for _, row in round_data.iterrows():
            ax.annotate(
                row['Driver'], 
                (row['FinishPos'], row['Predicted_Rank']),
                textcoords="offset points", 
                xytext=(0, 8), 
                ha='center', 
                va='bottom',
                fontsize=8.5,
                weight='semibold',
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)
            )
            
        ax.set_title(f"2026 Round {r} Results", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Actual Finish Position", fontsize=11)
        if i == 0:
            ax.set_ylabel("Predicted Rank", fontsize=11)
            
        # Configure layout (invert Y-axis so rank 1 is at the top)
        ax.set_xlim(0.5, max_val + 0.5)
        ax.set_ylim(max_val + 0.5, 0.5)
        ax.set_xticks(range(1, max_val + 1, 2 if max_val > 10 else 1))
        ax.set_yticks(range(1, max_val + 1, 2 if max_val > 10 else 1))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
            
    fig.suptitle("F1CAST Model Performance: Actual vs. Predicted Ranks", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save image
    plot_path = 'logs/actual_vs_predicted_ranks.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Saved results plot to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()