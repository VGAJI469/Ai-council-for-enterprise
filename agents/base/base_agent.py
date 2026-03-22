"""
Base Agent - Foundation class for all council agents.
Uses locally hosted LLMs via Ollama for reasoning generation.
Models available: llama3, phi, deepseek-r1, mistral
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from agents.base.llm_client import LocalLLMClient, ROLE_MODEL_MAP

logger = logging.getLogger(__name__)


@dataclass
class AgentPrediction:
    agent_id: str
    agent_role: str
    risk_score: float
    confidence: float
    decision: str              # APPROVE / REJECT / CONDITIONAL_APPROVE
    reasoning: str
    model_used: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    credibility: float = 1.0


class BaseAgent(ABC):
    """
    Abstract base class for all enterprise council agents.
    Each agent wraps a shared predictive ML model with role-specific
    decision biases and locally-hosted LLM reasoning.
    """

    def __init__(self, agent_id: str, config: dict, model=None,
                 llm_client: Optional[LocalLLMClient] = None):
        self.agent_id = agent_id
        self.config = config
        self.model = model
        self.credibility = config.get("initial_credibility", 1.0)
        self.role = config["focus"]
        self.risk_threshold = config["risk_threshold"]
        self.weights = config["weights"]
        self.prediction_history = []

        # Shared LLM client (injected or created fresh)
        self.llm = llm_client or LocalLLMClient()
        self.preferred_model = ROLE_MODEL_MAP.get(self.role, "llama3")

    @abstractmethod
    def compute_role_risk(self, features: dict) -> float:
        """Apply role-specific weighting to raw features."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the role-specific system prompt for LLM reasoning."""
        pass

    def predict(self, financial_data: dict) -> AgentPrediction:
        """
        Full prediction pipeline:
        1. Base ML model risk score
        2. Role-specific risk adjustment
        3. Decision thresholding
        4. Local LLM reasoning generation
        """
        # Base ML score (from XGBoost) or feature-derived heuristic
        if self.model is not None:
            base_risk = self.model.predict_proba(financial_data)
        else:
            base_risk = financial_data.get("default_probability", 0.4)

        role_risk = self.compute_role_risk(financial_data)
        # Add small per-call jitter so identical inputs still yield varied scores
        jitter    = random.gauss(0, 0.04)
        final_risk = 0.6 * base_risk + 0.4 * role_risk + jitter

        final_risk = min(max(final_risk, 0.0), 1.0)
        decision   = self._threshold_decision(final_risk)
        confidence = self._compute_confidence(final_risk)
        reasoning, model_used = self._generate_reasoning(financial_data, final_risk, decision)

        prediction = AgentPrediction(
            agent_id=self.agent_id,
            agent_role=self.role,
            risk_score=round(final_risk, 4),
            confidence=round(confidence, 4),
            decision=decision,
            reasoning=reasoning,
            model_used=model_used,
            credibility=self.credibility
        )
        self.prediction_history.append(prediction)
        return prediction

    def _threshold_decision(self, risk_score: float) -> str:
        if risk_score < self.risk_threshold * 0.7:
            return "APPROVE"
        elif risk_score > self.risk_threshold * 1.2:
            return "REJECT"
        else:
            return "CONDITIONAL_APPROVE"

    def _compute_confidence(self, risk_score: float) -> float:
        distance = abs(risk_score - self.risk_threshold)
        return min(0.5 + distance, 0.99)

    def _generate_reasoning(self, data: dict, risk: float, decision: str):
        """Generate role-specific reasoning using a locally hosted LLM."""
        dti  = data.get("debt_to_income_ratio", "N/A")
        cs   = data.get("credit_score", "N/A")
        loan = data.get("loan_amount", "N/A")
        gdp  = data.get("gdp_growth_rate", "N/A")

        prompt = (
            f"Financial case — DTI: {dti}, Credit score: {cs}, "
            f"Loan amount: {loan}, GDP growth: {gdp}\n"
            f"Computed risk score: {risk:.4f} | Recommended decision: {decision}\n\n"
            f"As the {self.role.replace('_', ' ').upper()} representative on the enterprise risk council, "
            f"give 2-3 concise sentences explaining your position on this decision. "
            f"Focus only on metrics relevant to your mandate."
        )

        # Randomise temperature and seed per call to avoid deterministic LLM output
        temperature = round(random.uniform(0.55, 0.90), 2)
        seed        = random.randint(1, 99999)
        print(f"[LLM] {self.preferred_model} | role={self.role} | temp={temperature} | seed={seed}")

        reasoning = self.llm.generate(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            model=self.preferred_model,
            max_tokens=250,
            temperature=temperature,
            seed=seed,
        )

        return reasoning, self.preferred_model

    def update_credibility(self, alpha, beta, gamma, delta,
                           performance, agreement, historical_error):
        """
        Credibility(t+1) = α·C(t) + β·Performance + γ·Agreement − δ·HistoricalError
        """
        self.credibility = (
            alpha * self.credibility
            + beta * performance
            + gamma * agreement
            - delta * historical_error
        )
        self.credibility = max(0.01, min(1.0, self.credibility))
        return self.credibility

    def to_dict(self) -> dict:
        return {
            "agent_id":        self.agent_id,
            "role":            self.role,
            "credibility":     round(self.credibility, 4),
            "risk_threshold":  self.risk_threshold,
            "preferred_model": self.preferred_model,
            "predictions":     len(self.prediction_history),
        }
