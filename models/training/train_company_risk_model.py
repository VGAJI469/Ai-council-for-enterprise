"""
Train XGBoost model on company financial health data.
This model learns to predict financial distress risk
from 33 financial indicators across 5000 companies.
"""

import joblib
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, f1_score
)
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train():
    logger.info('Loading company financial health dataset...')
    df = pd.read_csv('data/processed/company_financial_features.csv')

    drop_cols = ['company_id', 'sector', 'financial_distress']
    features  = [c for c in df.columns if c not in drop_cols]
    target    = 'financial_distress'

    X = df[features]
    y = df[target]

    logger.info(f'Dataset: {len(df)} companies, {len(features)} features')
    logger.info(f'Distress rate: {y.mean():.2%}')

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info('Training XGBoost on company financial data...')
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        early_stopping_rounds=25,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info('='*55)
    logger.info('  COMPANY RISK MODEL RESULTS')
    logger.info('='*55)
    logger.info(f'  Accuracy  : {accuracy_score(y_test, y_pred):.4f}')
    logger.info(f'  F1 Score  : {f1_score(y_test, y_pred):.4f}')
    logger.info(f'  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}')
    logger.info('')
    logger.info(classification_report(y_test, y_pred,
        target_names=['Financially Healthy', 'Financial Distress']))

    importance = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    logger.info('  TOP 10 MOST IMPORTANT FINANCIAL INDICATORS:')
    for feat, score in importance[:10]:
        bar = '=' * int(score * 200)
        logger.info(f'  {feat:<30} {bar}  {score:.4f}')

    Path('models').mkdir(exist_ok=True)
    joblib.dump({
        'model':    model,
        'scaler':   scaler,
        'features': features,
        'target':   target,
        'model_type': 'company_financial_risk'
    }, 'models/company_risk_model.pkl')

    logger.info('')
    logger.info('  Model saved to models/company_risk_model.pkl')
    return model, scaler, features


if __name__ == '__main__':
    train()
