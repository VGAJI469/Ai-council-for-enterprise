"""
Legal Agent — Regulatory compliance, policy exposure, legal risk.
Preferred model: deepseek-r1 (structured, rule-following reasoning)

Stability improvements:
  - Override predict() to ensure Legal Council output is always complete
  - Add special validation for empty responses (since Legal is critical)
  - Apply stricter minimum length requirements  
  - Log failures with high severity for monitoring
"""
import logging
from agents.base.base_agent import BaseAgent, AgentPrediction, FALLBACK_REASONING

logger = logging.getLogger(__name__)


class LegalAgent(BaseAgent):

    def _static_prompt(self) -> str:
        """
        Static persona prompt. The base class get_system_prompt() appends a
        compact memory suffix (credibility, recent history, macro snapshot)
        so the LLM has situational awareness without exceeding token budget.
        """
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
            w.get("regulatory_risk",   0.40) * features.get("regulatory_violation_prob", 0.5)
            + w.get("policy_exposure", 0.30) * features.get("policy_risk",               0.5)
            + w.get("legal_liability", 0.20) * features.get("legal_risk",                0.5)
            + w.get("compliance_score", 0.10) * (1 - features.get("compliance_score",    0.5))
        )
        return min(max(score, 0.0), 1.0)

    def predict(self, financial_data: dict) -> AgentPrediction:
        """
        Override predict() to add extra validation for Legal Counsel.
        Ensures that legal warnings are never lost due to LLM failures.
        """
        prediction = super().predict(financial_data)
        
        # Extra validation for Legal Counsel: stricter minimum reasoning length
        if not prediction.reasoning or len(prediction.reasoning.strip()) < 50:
            logger.critical(
                "LEGAL_AGENT_VALIDATION_FAILED | agent=%s | reasoning_len=%d | "
                "risk_score=%.3f | decision=%s — applying enhanced fallback",
                self.agent_id,
                len(prediction.reasoning.strip()) if prediction.reasoning else 0,
                prediction.risk_score,
                prediction.decision,
            )
            
            # Apply more conservative fallback for Legal
            prediction.reasoning = (
                "LEGAL COUNSEL ALERT: Unable to generate a complete legal analysis. "
                "This decision warrants extra regulatory review by outside counsel before proceeding. "
                "Recommend conditional approval pending legal review."
            )
            prediction.confidence = max(0.2, prediction.confidence)  # Lower confidence on fallback
            if prediction.decision == "APPROVE":
                prediction.decision = "CONDITIONAL_APPROVE"  # More cautious
        
        return prediction
