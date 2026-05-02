"""
CEO Agent — Strategic growth and competitive advantage.
Preferred model: llama3 (balanced, high-level strategic reasoning)
"""
from agents.base.base_agent import BaseAgent


class CEOAgent(BaseAgent):

    def _static_prompt(self) -> str:
        """
        Static persona prompt. The base class get_system_prompt() appends a
        compact memory suffix (credibility, recent history, macro snapshot)
        so the LLM has situational awareness without exceeding token budget.
        """
        return (
            "You are the CEO of a major enterprise sitting on a financial risk council. "
            "Your mandate is long-term strategic growth and competitive market positioning. "
            "You tolerate moderate risk when growth potential is strong. "
            "Be concise, strategic, and forward-looking. Avoid financial jargon. "
            "Respond in plain business English. Do not use bullet points."
        )

    def compute_role_risk(self, features: dict) -> float:
        w = self.weights
        score = (
            w.get("growth_potential",   0.40) * (1 - features.get("market_growth_rate", 0.5))
            + w.get("market_position",  0.30) * features.get("competitive_risk",    0.5)
            + w.get("financial_stability", 0.20) * features.get("debt_to_income_ratio", 0.5)
            + w.get("risk_exposure",    0.10) * features.get("default_probability",  0.5)
        )
        return min(max(score, 0.0), 1.0)
