# scripts/main.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

try:
    from model_pipeline import F1RankingModel
except ImportError:
    from scripts.model_pipeline import F1RankingModel

LOCATION_TO_GP = {
    'Spa-Francorchamps': 'Belgian GP',
    'Spa': 'Belgian GP',
    'Sakhir': 'Bahrain GP',
    'Bahrain': 'Bahrain GP',
    'Jeddah': 'Saudi Arabian GP',
    'Melbourne': 'Australian GP',
    'Suzuka': 'Japanese GP',
    'Shanghai': 'Chinese GP',
    'Miami': 'Miami GP',
    'Imola': 'Emilia Romagna GP',
    'Monte Carlo': 'Monaco GP',
    'Monaco': 'Monaco GP',
    'Montreal': 'Canadian GP',
    'Barcelona': 'Spanish GP',
    'Spielberg': 'Austrian GP',
    'Silverstone': 'British GP',
    'Budapest': 'Hungarian GP',
    'Zandvoort': 'Dutch GP',
    'Monza': 'Italian GP',
    'Baku': 'Azerbaijan GP',
    'Singapore': 'Singapore GP',
    'Austin': 'United States GP',
    'Mexico City': 'Mexico City GP',
    'São Paulo': 'São Paulo GP',
    'Interlagos': 'São Paulo GP',
    'Las Vegas': 'Las Vegas GP',
    'Lusail': 'Qatar GP',
    'Yas Marina': 'Abu Dhabi GP',
    'Abu Dhabi': 'Abu Dhabi GP',
    'Paul Ricard': 'French GP',
    'Le Castellet': 'French GP',
    'Portimão': 'Portuguese GP',
    'Nürburgring': 'Eifel GP',
    'Mugello': 'Tuscan GP',
    'Sochi': 'Russian GP',
}

def format_gp_name(row_group):
    if 'EventName' in row_group.columns:
        evt = row_group['EventName'].iloc[0]
        if evt and pd.notnull(evt) and str(evt).strip():
            evt_str = str(evt).strip()
            gp_clean = evt_str.replace("Grand Prix", "GP").strip()
            if not gp_clean.endswith("GP"):
                gp_clean += " GP"
            return gp_clean

    if 'Circuit' in row_group.columns:
        circ = str(row_group['Circuit'].iloc[0]).strip()
        if circ in LOCATION_TO_GP:
            return LOCATION_TO_GP[circ]
        for key, name in LOCATION_TO_GP.items():
            if key.lower() in circ.lower():
                return name
        return f"{circ} GP"

    return ""

def run_pipeline(
    data_path='data/engineered/pro_f1_engineered.parquet', 
    test_season=None, 
    test_rounds=None, 
    num_test_rounds=1, 
    show_plot=False
):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run ingestor.py and engineer.py first.")
        return None

    df = pd.read_parquet(data_path)
    if df.empty:
        print("Error: Dataset is empty.")
        return None
        
    features = [
        'GridPosition', 'DriverForm', 'TeamForm', 'Team_DNF_Rate', 
        'Track_Cat_Code', 'Era_Code', 'Team_Code', 'stress', 'abrasion', 
        'overtake_diff', 'speed_cat', 'downforce', 'brake_severity', 'altitude_m'
    ]
    
    model = F1RankingModel(features)

    # Get all unique (Season, Round) pairs in chronological order
    races_df = df[['Season', 'Round']].drop_duplicates().sort_values(['Season', 'Round'])
    all_races = list(zip(races_df['Season'], races_df['Round']))

    if len(all_races) <= num_test_rounds:
        print(f"Error: Not enough races ({len(all_races)}) to split for training and testing ({num_test_rounds} test rounds requested).")
        return None

    # Determine train/test split
    if test_season is not None and test_rounds is not None:
        if isinstance(test_rounds, int):
            test_rounds = [test_rounds]
        test_races = [(test_season, r) for r in test_rounds]
    else:
        test_races = all_races[-num_test_rounds:]

    test_set = set(test_races)
    
    df['is_test'] = df.apply(lambda row: (row['Season'], row['Round']) in test_set, axis=1)
    
    train_df = df[~df['is_test']].copy()
    test_df = df[df['is_test']].copy()

    if train_df.empty:
        print("Error: Train dataset is empty.")
        return None
    if test_df.empty:
        print("Error: Test dataset is empty. Check split parameters.")
        return None

    # Sort training data by Season and Round to ensure group grouping works correctly in XGBRanker
    train_df = train_df.sort_values(['Season', 'Round'])
    test_df = test_df.sort_values(['Season', 'Round'])

    train_races_cnt = len(train_df[['Season', 'Round']].drop_duplicates())
    test_races_cnt = len(test_df[['Season', 'Round']].drop_duplicates())

    print(f"Training model on {len(train_df)} rows across {train_races_cnt} races...")
    model.fit(train_df)
    
    print(f"Evaluating model on {len(test_df)} rows across {test_races_cnt} test races...")
    scores, dnf_p = model.predict(test_df)
    
    test_df['Predicted_Score'] = scores
    test_df['DNF_Risk'] = dnf_p
    
    # Calculate ranks for each round separately (higher Predicted_Score means better rank/lower rank number)
    test_df['Predicted_Rank'] = test_df.groupby(['Season', 'Round'])['Predicted_Score'].rank(ascending=False)

    os.makedirs('logs', exist_ok=True)
    results_path = 'logs/latest_test_results.csv'
    test_df.drop(columns=['is_test']).to_csv(results_path, index=False)
    print(f"Saved test predictions to {results_path}")

    # Display evaluation metrics
    print("\n=== EVALUATION RESULTS ===")
    correlations = {}
    
    for (season, round_num), group in test_df.groupby(['Season', 'Round']):
        gp_name = format_gp_name(group)
        coef, _ = spearmanr(group['Predicted_Rank'], group['FinishPos'])
        if np.isnan(coef):
            coef = 0.0
        
        label_str = f"Season {season} Round {round_num}"
        if gp_name:
            label_str += f" ({gp_name})"
        print(f"{label_str} Spearman Correlation: {coef:.3f}")
        correlations[f"{season}_R{round_num}"] = coef
        
    avg_corr = sum(correlations.values()) / len(correlations) if correlations else 0.0
    print(f"Average Spearman Correlation: {avg_corr:.3f}")

    # Plot results
    plot_path = plot_results(test_df, show_plot=show_plot)

    return {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_races': train_races_cnt,
        'test_races': test_races_cnt,
        'correlations': correlations,
        'avg_spearman': avg_corr,
        'results_csv': results_path,
        'plot_png': plot_path
    }

def plot_results(test_df, show_plot=False):
    unique_races = sorted(list(set(zip(test_df['Season'], test_df['Round']))))
    num_rounds = len(unique_races)
    
    # Elegant styling setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, num_rounds, figsize=(6 * num_rounds, 6), sharey=True)
    if num_rounds == 1:
        axes = [axes]
        
    for i, (season, r) in enumerate(unique_races):
        ax = axes[i]
        round_data = test_df[(test_df['Season'] == season) & (test_df['Round'] == r)].copy()
        gp_name = format_gp_name(round_data)
        
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
        ax.scatter(
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
            
        title_text = f"{season} Round {r}"
        if gp_name:
            title_text += f"\n({gp_name})"
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
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
    os.makedirs('logs', exist_ok=True)
    plot_path = 'logs/actual_vs_predicted_ranks.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Saved results plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close(fig)
    return plot_path

if __name__ == "__main__":
    run_pipeline(show_plot=True)