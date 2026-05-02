"""
CFO Agent — Financial stability, capital preservation, liquidity risk.
Preferred model: deepseek-r1 (strong numerical + reasoning chains)
"""
from agents.base.base_agent import BaseAgent


class CFOAgent(BaseAgent):

    def _static_prompt(self) -> str:
        """
        Static persona prompt. The base class get_system_prompt() appends a
        compact memory suffix (credibility, recent history, macro snapshot)
        so the LLM has situational awareness without exceeding token budget.
        """
        return (
            "You are the CFO of a major enterprise on a financial risk council. "
            "Your mandate is capital preservation, liquidity management, and avoiding default risk. "
            "You are conservative and data-driven. You require strong financial fundamentals. "
            "Reference specific ratios (DTI, credit score, cash flow) in your reasoning. "
            "Be precise and quantitative. Respond in 2-3 sentences. No bullet points."
        )

    def compute_role_risk(self, features: dict) -> float:
        w = self.weights
        score = (
            w.get("liquidity_risk",       0.35) * features.get("liquidity_ratio_inv",  0.5)
            + w.get("capital_preservation", 0.30) * features.get("debt_to_income_ratio", 0.5)
            + w.get("default_probability",  0.25) * features.get("default_probability",  0.5)
            + w.get("cash_flow",            0.10) * features.get("cash_flow_risk",        0.5)
        )
        return min(max(score, 0.0), 1.0)
