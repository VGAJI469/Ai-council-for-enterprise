"""Feature Builder - Unified feature vector construction from multi-source data."""

import pandas as pd
import logging
logger = logging.getLogger(__name__)


class FeatureBuilder:
    def build(self, datasets: dict) -> list:
        credit = datasets.get("credit_risk", pd.DataFrame())
        macro = datasets.get("macroeconomic", pd.DataFrame())
        market = datasets.get("market_volatility", pd.DataFrame())

        if credit.empty:
            logger.warning("No credit risk data. Using synthetic demo record.")
            credit = pd.DataFrame([{
                "loan_id": "DEMO_001", "borrower_income": 75000,
                "debt_to_income_ratio": 0.35, "credit_score": 680,
                "loan_amount": 250000, "loan_term_months": 360,
                "employment_years": 5.0, "default_history": 0
            }])

        macro_f = self._macro_features(macro)
        market_f = self._market_features(market)

        records = []
        for _, row in credit.iterrows():
            f = row.to_dict()
            f.update(macro_f); f.update(market_f)
            f = self._engineer(f)
            records.append(f)
        return records

    def _macro_features(self, df):
        if df.empty:
            return {"gdp_growth_rate": 0.025, "inflation_rate": 0.032, "interest_rate": 0.052, "unemployment_rate": 0.058}
        r = df.iloc[-1]
        return {k: r.get(k, 0) for k in ["gdp_growth_rate","inflation_rate","interest_rate","unemployment_rate"]}

    def _market_features(self, df):
        if df.empty:
            return {"market_volatility": 0.18, "market_growth_rate": 0.06}
        return {"market_volatility": df.get("volatility_index", pd.Series([0.18])).mean(),
                "market_growth_rate": df.get("index_return", pd.Series([0.06])).mean()}

    def _engineer(self, f: dict) -> dict:
        dti = f.get("debt_to_income_ratio", 0.35)
        cs = f.get("credit_score", 650)
        f["liquidity_ratio_inv"] = min(dti * 1.5, 1.0)
        f["default_probability"] = max(0, (850 - cs) / 550)
        f["competitive_risk"] = f.get("market_volatility", 0.2)
        f["sentiment_risk"] = 0.28
        f["brand_risk"] = 0.22
        f["media_risk"] = 0.18
        f["stakeholder_risk"] = 0.20
        f["regulatory_violation_prob"] = 0.12
        f["policy_risk"] = 0.18
        f["legal_risk"] = 0.14
        f["compliance_score"] = 0.82
        f["customer_churn_risk"] = 0.25
        f["cash_flow_risk"] = dti
        return f

    def build_company_profile(self, company: dict) -> dict:
        """
        Build a feature vector from a company financial profile
        for use in the council debate system.
        """
        f = company.copy()

        # Derived risk signals for agents
        f['default_probability']        = max(0, min(1,
            (f.get('debt_to_equity_ratio', 1.0) / 5.0) * 0.4 +
            max(0, -f.get('net_profit_margin', 0)) * 0.3 +
            (1 - min(f.get('current_ratio', 1.0) / 4.0, 1.0)) * 0.3
        ))
        f['liquidity_ratio_inv']        = max(0, 1 - f.get('current_ratio', 1.0) / 4.0)
        f['competitive_risk']           = max(0, 1 - f.get('market_share_percent', 5.0) / 35.0)
        f['market_growth_rate']         = f.get('market_growth_rate', 0.05)
        f['cash_flow_risk']             = max(0, -f.get('free_cash_flow', 0) / (f.get('annual_revenue', 1) + 1))
        f['sentiment_risk']             = max(0, 1 - f.get('esg_score', 50) / 90.0)
        f['brand_risk']                 = max(0, 1 - f.get('customer_retention_rate', 0.8))
        f['media_risk']                 = min(f.get('regulatory_violations', 0) * 0.15, 0.9)
        f['stakeholder_risk']           = max(0, 1 - f.get('management_score', 5) / 10.0)
        f['regulatory_violation_prob']  = min(f.get('regulatory_violations', 0) * 0.10, 0.9)
        f['policy_risk']                = 0.15
        f['legal_risk']                 = min(f.get('regulatory_violations', 0) * 0.12, 0.9)
        f['compliance_score']           = max(0, 1 - f.get('regulatory_violations', 0) * 0.15)
        f['customer_churn_risk']        = max(0, 1 - f.get('customer_retention_rate', 0.8))
        f['debt_to_income_ratio']       = f.get('debt_to_revenue_ratio', 0.3)

        return f
