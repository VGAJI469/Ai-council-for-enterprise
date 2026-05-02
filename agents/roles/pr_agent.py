"""
PR Agent — Reputation risk, public sentiment, brand stability.
Preferred model: llama3 (balanced, stakeholder-aware framing)
"""
from agents.base.base_agent import BaseAgent


class PRAgent(BaseAgent):

    def _static_prompt(self) -> str:
        """
        Static persona prompt. The base class get_system_prompt() appends a
        compact memory suffix (credibility, recent history, macro snapshot)
        so the LLM has situational awareness without exceeding token budget.
        """
        return (
            "You are the Chief Communications and PR Officer on an enterprise risk council. "
            "Your mandate is protecting brand reputation, public trust, and stakeholder relationships. "
            "A poor financial decision that damages the brand is unacceptable regardless of short-term gains. "
            "Consider how this decision would be reported in the press. "
            "Be stakeholder-aware. 2-3 sentences. No bullet points."
        )

    def compute_role_risk(self, features: dict) -> float:
        w = self.weights
        score = (
            w.get("public_sentiment",   0.35) * features.get("sentiment_risk",   0.5)
            + w.get("brand_stability",  0.30) * features.get("brand_risk",       0.5)
            + w.get("media_exposure",   0.25) * features.get("media_risk",        0.5)
            + w.get("stakeholder_trust", 0.10) * features.get("stakeholder_risk", 0.5)
        )
        return min(max(score, 0.0), 1.0)
