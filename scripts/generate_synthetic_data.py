"""
Generate synthetic financial datasets for pipeline testing.
Creates credit_risk.csv, macroeconomic.csv, market_volatility.csv
in the data/raw/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 200

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

credit_scores = np.random.randint(500, 820, N)
dti = np.round(np.random.uniform(0.1, 0.65, N), 2)
income = np.random.randint(30000, 180000, N)
loan_amount = np.random.randint(10000, 500000, N)
employment_years = np.round(np.random.uniform(0.5, 25, N), 1)
loan_term = np.random.choice([60, 120, 180, 240, 360], N)
default_history = np.random.choice([0, 0, 0, 1, 2], N)

default_prob = (dti * 0.5 + (850 - credit_scores) / 850 * 0.5)
default_flag = (default_prob + np.random.normal(0, 0.1, N) > 0.45).astype(int)

credit_df = pd.DataFrame({
    "loan_id":              [f"LOAN_{i:04d}" for i in range(N)],
    "borrower_income":      income,
    "debt_to_income_ratio": dti,
    "credit_score":         credit_scores,
    "loan_amount":          loan_amount,
    "loan_term_months":     loan_term,
    "employment_years":     employment_years,
    "default_history":      default_history,
    "collateral_value":     np.round(loan_amount * np.random.uniform(0.8, 1.5, N), 2),
    "default_flag":         default_flag
})
credit_df.to_csv("data/raw/credit_risk.csv", index=False)
print(f"credit_risk.csv: {len(credit_df)} records, {credit_df['default_flag'].sum()} defaults")

credit_df.to_csv("data/processed/credit_risk_features.csv", index=False)

dates = pd.date_range("2023-01-01", periods=24, freq="ME")
macro_df = pd.DataFrame({
    "date":                       dates.strftime("%Y-%m-%d"),
    "gdp_growth_rate":            np.round(np.random.uniform(0.01, 0.04, 24), 4),
    "inflation_rate":             np.round(np.random.uniform(0.02, 0.07, 24), 4),
    "interest_rate":              np.round(np.random.uniform(0.04, 0.07, 24), 4),
    "unemployment_rate":          np.round(np.random.uniform(0.035, 0.065, 24), 4),
    "consumer_confidence_index":  np.round(np.random.uniform(85, 115, 24), 2),
    "industrial_production_index":np.round(np.random.uniform(95, 108, 24), 2),
})
macro_df.to_csv("data/raw/macroeconomic.csv", index=False)
print(f"macroeconomic.csv: {len(macro_df)} records")

market_df = pd.DataFrame({
    "date":             dates.strftime("%Y-%m-%d"),
    "volatility_index": np.round(np.random.uniform(12, 35, 24), 2),
    "index_return":     np.round(np.random.uniform(-0.05, 0.08, 24), 4),
    "vix":              np.round(np.random.uniform(14, 40, 24), 2),
    "spread":           np.round(np.random.uniform(0.5, 3.5, 24), 3),
})
market_df.to_csv("data/raw/market_volatility.csv", index=False)
print(f"market_volatility.csv: {len(market_df)} records")

print("\nAll datasets generated in data/raw/")
print("Ready for pipeline ingestion.")
