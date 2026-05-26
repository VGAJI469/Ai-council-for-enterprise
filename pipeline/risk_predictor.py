import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RiskPredictor:
    def __init__(self, model_path="pipeline/risk_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.features = [
            "debt_to_income_ratio",
            "market_growth_rate",
            "competitive_risk",
            "liquidity_ratio_inv",
            "cash_flow_risk"
        ]

    def _generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)

        # Generating features
        dti = np.random.uniform(0.1, 0.8, n_samples)
        mgr = np.random.uniform(0.01, 0.15, n_samples)
        cr = np.random.uniform(0.1, 0.9, n_samples)
        lri = np.random.uniform(0.1, 0.8, n_samples)
        cfr = np.random.uniform(0.1, 0.9, n_samples)

        # Risk score calculation formula to generate pseudo-realistic ground truth
        risk_score = (dti * 0.3) + ((1 - mgr) * 0.2) + (cr * 0.2) + (lri * 0.15) + (cfr * 0.15)
        # Add some noise
        risk_score += np.random.normal(0, 0.05, n_samples)

        # Binary target: 1 if risk_score > 0.5 else 0
        target = (risk_score > 0.5).astype(int)

        df = pd.DataFrame({
            "debt_to_income_ratio": dti,
            "market_growth_rate": mgr,
            "competitive_risk": cr,
            "liquidity_ratio_inv": lri,
            "cash_flow_risk": cfr,
            "default_risk": target
        })

        return df

    def train_or_load_model(self):
        """Loads the model if it exists, otherwise trains a new one using synthetic data."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            df = self._generate_synthetic_data()
            X = df[self.features]
            y = df["default_risk"]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
            ])

            pipeline.fit(X, y)
            self.model = pipeline

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def predict_risk_percentage(self, data: dict) -> float:
        """
        Takes the input payload from the API and outputs a default probability percentage (0-100).
        """
        if self.model is None:
            self.train_or_load_model()

        # Extract features from data, defaulting to safe median values if missing
        input_data = pd.DataFrame([{
            "debt_to_income_ratio": data.get("debt_to_income_ratio", 0.35),
            "market_growth_rate": data.get("market_growth_rate", 0.05),
            "competitive_risk": data.get("competitive_risk", 0.4),
            "liquidity_ratio_inv": data.get("liquidity_ratio_inv", 0.4),
            "cash_flow_risk": data.get("cash_flow_risk", 0.4)
        }])

        # Output probability of class 1 (default)
        prob = self.model.predict_proba(input_data)[0][1]

        # Return as percentage
        return float(prob * 100)

# Instantiate a global instance for the API
risk_engine = RiskPredictor()
