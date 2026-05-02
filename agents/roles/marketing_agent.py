"""
Marketing Agent — Market expansion and customer growth.
Preferred model: mistral (diverse perspective, creative framing)
"""
from agents.base.base_agent import BaseAgent


class MarketingAgent(BaseAgent):

    def _static_prompt(self) -> str:
        """
        Static persona prompt. The base class get_system_prompt() appends a
        compact memory suffix (credibility, recent history, macro snapshot)
        so the LLM has situational awareness without exceeding token budget.
        """
        return (
            "You are the Chief Marketing Officer on an enterprise financial risk council. "
            "Your mandate is market expansion, customer acquisition, and competitive edge. "
            "You favor bold moves that grow market share and accept higher risk for strong upside. "
            "Focus on opportunity cost — what happens if we don't act. "
            "Be optimistic but grounded. 2-3 sentences. No bullet points."
        )

    def compute_role_risk(self, features: dict) -> float:
        w = self.weights
        score = (
            w.get("market_opportunity", 0.40) * (1 - features.get("market_growth_rate", 0.5))
            + w.get("customer_growth",  0.35) * features.get("customer_churn_risk",      0.5)
            + w.get("brand_reach",      0.15) * features.get("brand_risk",                0.5)
            + w.get("competitive_edge", 0.10) * features.get("competitive_risk",          0.5)
        )
        return min(max(score, 0.0), 1.0)
