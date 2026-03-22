"""Model Wrapper - Standardized interface for base model inference."""
import joblib
import numpy as np
import logging
logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "borrower_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "loan_term_months", "employment_years", "default_history"
]

class BaseModelWrapper:
    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.features = artifact["features"]

    def predict_proba(self, features: dict) -> float:
        row = np.array([[features.get(f, 0.0) for f in self.features]])
        row_scaled = self.scaler.transform(row)
        prob = self.model.predict_proba(row_scaled)[0][1]
        return float(prob)

class MockModel:
    """Fallback model for testing without a trained artifact."""
    def predict_proba(self, features: dict) -> float:
        dti = features.get("debt_to_income_ratio", 0.35)
        cs = features.get("credit_score", 650)
        return min(max((dti * 0.6 + (850 - cs) / 850 * 0.4), 0.0), 1.0)
