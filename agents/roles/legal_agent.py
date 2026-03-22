"""
Legal Agent — Regulatory compliance, policy exposure, legal risk.
Preferred model: deepseek-r1 (structured, rule-following reasoning)
"""
from agents.base.base_agent import BaseAgent

class LegalAgent(BaseAgent):

    def get_system_prompt(self) -> str:
        return (
            "You are the Chief Legal Officer on an enterprise financial risk council. "
            "Your mandate is regulatory compliance, legal liability reduction, and policy adherence. "
            "Any decision creating material legal or regulatory risk must be rejected or heavily conditioned. "
            "Cite the specific legal dimension (compliance, liability, exposure) in your reasoning. "
            "Be precise and risk-averse. 2-3 sentences. No bullet points."
        )

    def compute_role_risk(self, features: dict) -> float:
        w = self.weights
        score = (
            w.get("regulatory_risk", 0.40) * features.get("regulatory_violation_prob", 0.5)
            + w.get("policy_exposure", 0.30) * features.get("policy_risk", 0.5)
            + w.get("legal_liability", 0.20) * features.get("legal_risk", 0.5)
            + w.get("compliance_score", 0.10) * (1 - features.get("compliance_score", 0.5))
        )
        return min(max(score, 0.0), 1.0)
