import joblib
import numpy as np

artifact = joblib.load("models/base_model.pkl")
model    = artifact["model"]
scaler   = artifact["scaler"]
features = artifact["features"]

print(f"Model type: {type(model).__name__}")
print(f"Features ({len(features)}): {features}")

sample = np.array([[45000, 0.55, 580, 120000, 360, 1.5, 2, 90000]])
sample_scaled = scaler.transform(sample)
prob = model.predict_proba(sample_scaled)[0][1]
print(f"\nHigh-risk borrower default probability: {prob:.4f}")

sample2 = np.array([[120000, 0.18, 780, 80000, 180, 10.0, 0, 150000]])
sample2_scaled = scaler.transform(sample2)
prob2 = model.predict_proba(sample2_scaled)[0][1]
print(f"Low-risk borrower default probability:  {prob2:.4f}")
