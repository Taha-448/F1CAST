# scripts/engineer.py
import os
import pandas as pd
import numpy as np
from metadata import get_circuit_metadata

def build_features():
    raw_path = 'data/raw/pro_f1_raw.parquet'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please run ingestor.py first.")
        return
        
    df = pd.read_parquet(raw_path)
    df['FinishPos'] = pd.to_numeric(df['FinishPos'], errors='coerce').fillna(20)
    
    # Label Era
    def get_era(year):
        if 2014 <= year <= 2016: return 'A'
        if 2017 <= year <= 2021: return 'B'
        if 2022 <= year <= 2025: return 'C'
        return 'D'
    df['Era'] = df['Season'].apply(get_era)

    # DNF & Rank Labels
    # DNF Logic: If not classified as Finished/Laps down, mark as 1.
    df['is_DNF'] = df['Status'].apply(lambda x: 0 if 'Finished' in str(x) or 'Lap' in str(x) else 1)
    
    # Sort data chronologically, and within each race, sort by:
    # 1. is_DNF (0 first, 1 last)
    # 2. FinishPos (1 first)
    # 3. LapsCompleted (more laps first for DNFs)
    df = df.sort_values(
        ['Season', 'Round', 'is_DNF', 'FinishPos', 'LapsCompleted'],
        ascending=[True, True, True, True, False]
    )
    
    # Within each race (Season + Round), assign a sequential rank (1 to N)
    df['Rank'] = df.groupby(['Season', 'Round']).cumcount() + 1
    
    # Relevance Label: Higher is better, must be integer (0 or positive) for XGBRanker
    group_sizes = df.groupby(['Season', 'Round'])['Season'].transform('size')
    df['RelevanceLabel'] = (group_sizes - df['Rank']).astype(int)
    
    # Rolling Form (Grouped by Driver and Team)
    df['DriverForm'] = df.groupby('Driver')['FinishPos'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['TeamForm'] = df.groupby('Team')['FinishPos'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['Team_DNF_Rate'] = df.groupby('Team')['is_DNF'].transform(lambda x: x.shift(1).rolling(10).mean())

    # Fill NaNs of rolling metrics with sensible defaults to avoid chronological/cross-driver leakage
    df['DriverForm'] = df['DriverForm'].fillna(10.5)
    df['TeamForm'] = df['TeamForm'].fillna(10.5)
    df['Team_DNF_Rate'] = df['Team_DNF_Rate'].fillna(0.1)

    # Metadata Merge using name normalization
    meta_df = df['Circuit'].apply(get_circuit_metadata).apply(pd.Series)
    df = pd.concat([df, meta_df], axis=1)

    # Sample Weighting (Recency decay)
    # Most recent races have weight 1.0, older races decay toward 0.1
    max_year = df['Season'].max()
    df['SampleWeight'] = 1 / (1 + (max_year - df['Season']))

    # Encoding
    df['Track_Cat_Code'] = df['cat'].astype('category').cat.codes
    df['Team_Code'] = df['Team'].astype('category').cat.codes
    df['Era_Code'] = df['Era'].map({'A':0, 'B':1, 'C':2, 'D':3})

    # Fill any other potential NaNs (e.g. GridPosition) and safety check
    df['GridPosition'] = df['GridPosition'].fillna(20)
    df.fillna(0, inplace=True)
    
    engineered_dir = 'data/engineered'
    os.makedirs(engineered_dir, exist_ok=True)
    output_path = os.path.join(engineered_dir, 'pro_f1_engineered.parquet')
    df.to_parquet(output_path, index=False)
    print(f"Successfully saved engineered features to {output_path}")

if __name__ == "__main__":
    build_features()