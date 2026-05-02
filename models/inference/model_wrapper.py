"""
Model Wrapper - Standardized interface for base model inference.

Upgrade note (Item 10):
  MockModel.predict_proba() previously used a dumb 2-variable linear formula:
    DTI * 0.6 + (850 - credit_score) / 850 * 0.4

  Replaced with a proper heuristic that:
    - Uses all 7 available feature fields
    - Applies a non-linear DTI risk acceleration above the 0.45 threshold
      (DTI risk does accelerate sharply past this point in real credit models)
    - Handles missing features gracefully with realistic defaults
    - Is weighted across dimensions so no single feature can dominate unreasonably
    - Behaves like a real preliminary credit model, not a two-variable proxy
"""

import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "borrower_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "loan_term_months", "employment_years", "default_history"
]


class BaseModelWrapper:
    """Production wrapper around a trained XGBoost artifact."""

    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.model   = artifact["model"]
        self.scaler  = artifact["scaler"]
        self.features = artifact["features"]

    def predict_proba(self, features: dict) -> float:
        """Run scaled inference through the trained XGBoost model."""
        row        = np.array([[features.get(f, 0.0) for f in self.features]])
        row_scaled = self.scaler.transform(row)
        prob       = self.model.predict_proba(row_scaled)[0][1]
        return float(prob)


class MockModel:
    """
    Intelligent fallback model for testing or when no trained artifact is available.

    Uses all 7 standard feature fields with a weighted heuristic. DTI risk is
    non-linear — it accelerates sharply above 0.45 to reflect real credit behaviour
    where moderate leverage is manageable but high leverage compounds rapidly.

    Feature weights (sum to 1.0):
      DTI            0.30  — primary leverage signal, non-linear above 0.45
      Credit score   0.25  — normalised over [300, 850] range
      Default history 0.20 — binary prior, strongest single predictor
      Loan-to-income  0.10 — affordability ratio
      Employment years 0.10 — job stability proxy
      Loan term       0.05 — longer terms carry slightly more uncertainty
    """

    def predict_proba(self, features: dict) -> float:
        """
        Compute a risk probability using a full multi-feature heuristic.

        Replaces: DTI * 0.6 + (850 - credit_score) / 850 * 0.4
        Which ignored 5 of the 7 available fields and applied a flat linear
        relationship to DTI despite its known non-linear behaviour.
        """
        # --- Raw feature extraction with safe defaults ---
        dti_raw  = features.get("debt_to_income_ratio", 0.35)
        cs       = features.get("credit_score", 650)
        dh       = float(features.get("default_history", 0))
        income   = features.get("borrower_income", 60_000)
        loan     = features.get("loan_amount", 200_000)
        emp      = features.get("employment_years", 5.0)
        term     = features.get("loan_term_months", 360)

        # --- Non-linear DTI risk ---
        # Below 0.45: risk is proportional to DTI
        # Above 0.45: each additional point of DTI causes 2.5× more risk
        # This reflects the empirical finding that DTI > 0.45 is a hard tipping point
        if dti_raw < 0.45:
            dti_risk = dti_raw
        else:
            dti_risk = 0.45 + (dti_raw - 0.45) * 2.5
        dti_risk = min(dti_risk, 1.0)

        # --- Normalised sub-scores ---
        # Credit score: lower score → higher risk; normalised over [300, 850]
        cs_risk = max(0.0, (850.0 - cs) / 550.0)

        # Default history: binary 0/1 — prior default is the strongest single predictor
        dh_risk = dh

        # Loan-to-income: affordability, capped at 1.0 (>10× income is extreme)
        income_risk = min(loan / max(income, 1.0) / 10.0, 1.0)

        # Employment stability: new employees carry more risk; plateau at 20 years
        emp_risk = max(0.0, 1.0 - emp / 20.0)

        # Loan term: very long terms (>480 months) carry marginal extra uncertainty
        term_risk = min(term / 480.0, 1.0)

        # --- Weighted composite ---
        score = (
            0.30 * dti_risk
            + 0.25 * cs_risk
            + 0.20 * dh_risk
            + 0.10 * income_risk
            + 0.10 * emp_risk
            + 0.05 * term_risk
        )

        return float(min(max(score, 0.0), 1.0))
