"""
Base Model Training - XGBoost shared model for all council agents.
All agents use this same trained model, differentiated by role-specific weighting.
"""

import logging, joblib, yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def train(data_path="data/processed/credit_risk_features.csv",
          model_path="models/base_model.pkl",
          config_path="config/model.yaml"):
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return None, None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    df = pd.read_csv(data_path)
    target = "default_flag"
    features = [c for c in df.columns if c not in [target, "loan_id"]]
    X = df[features]; y = df[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=cfg["training"]["test_split"],
        random_state=cfg["training"]["random_state"], stratify=y)

    model = xgb.XGBClassifier(
        **cfg["base"]["hyperparameters"],
        eval_metric="logloss",
        early_stopping_rounds=cfg["training"]["early_stopping_rounds"],
        random_state=cfg["training"]["random_state"]
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump({"model": model, "scaler": scaler, "features": features}, model_path)
    logger.info(f"Model saved: {model_path}")
    return model, scaler

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
      