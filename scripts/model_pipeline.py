# scripts/model_pipeline.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class F1RankingModel:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        # Stage 1: DNF Classifier
        self.dnf_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=200)
        # Stage 2: Grid Ranker (XGBRanker)
        self.ranker = xgb.XGBRanker(objective='rank:ndcg', n_estimators=500, learning_rate=0.05)

    def fit(self, train_df):
        # 1. Train DNF on full train_df
        self.dnf_model.fit(train_df[self.feature_cols], train_df['is_DNF'], sample_weight=train_df['SampleWeight'])
        
        # 2. Out-of-fold predictions to prevent target leakage for DNF_Risk
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        dnf_probs = np.zeros(len(train_df))
        
        for train_idx, val_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            
            fold_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=200)
            fold_model.fit(fold_train[self.feature_cols], fold_train['is_DNF'], sample_weight=fold_train['SampleWeight'])
            
            dnf_probs[val_idx] = fold_model.predict_proba(fold_val[self.feature_cols])[:, 1]
            
        X_rank = train_df[self.feature_cols].copy()
        X_rank['DNF_Risk'] = dnf_probs
        
        # 3. Train Ranker (Grouped by Season+Round)
        groups = train_df.groupby(['Season', 'Round']).size().values
        self.ranker.fit(X_rank, train_df['RelevanceLabel'], group=groups, sample_weight=train_df['SampleWeight'])

    def predict(self, test_df):
        dnf_probs = self.dnf_model.predict_proba(test_df[self.feature_cols])[:, 1]
        X_rank = test_df[self.feature_cols].copy()
        X_rank['DNF_Risk'] = dnf_probs
        
        # Get Ranking scores (higher predicted score means higher relevance/better position)
        scores = self.ranker.predict(X_rank)
        return scores, dnf_probs