"""Credibility Manager - Dynamic agent credibility tracking."""

import logging
from typing import List, Dict
from agents.base.base_agent import AgentPrediction

logger = logging.getLogger(__name__)


class CredibilityManager:
    def __init__(self, alpha=0.7, beta=0.15, gamma=0.10, delta=0.05):
        self.alpha = alpha; self.beta = beta
        self.gamma = gamma; self.delta = delta
        self.history: List[Dict] = []

    def update_all(self, agents: list, predictions: List[AgentPrediction],
                   final_decision: str, ground_truth: str = None) -> Dict:
        updates = {}
        for agent, prediction in zip(agents, predictions):
            performance = 1.0 if (ground_truth and prediction.decision == ground_truth) else 0.5
            agreement = 1.0 if prediction.decision == final_decision else 0.0
            hist_error = self._historical_error(agent)
            old = agent.credibility
            new_cred = agent.update_credibility(
                self.alpha, self.beta, self.gamma, self.delta,
                performance, agreement, hist_error
            )
            updates[agent.role] = {
                "old": round(old, 4), "new": round(new_cred, 4),
                "performance": round(performance, 4), "agreement": agreement
            }
            logger.info(f"  [{agent.role}] credibility {old:.3f} → {new_cred:.3f}")
        self.history.append({"cycle": len(self.history)+1, "updates": updates})
        return updates

    def _historical_error(self, agent) -> float:
        if len(agent.prediction_history) < 5:
            return 0.0
        recent = agent.prediction_history[-5:]
        return 1.0 - sum(abs(p.risk_score - 0.5) for p in recent) / len(recent)

    def get_lowest(self, agents): return min(agents, key=lambda a: a.credibility)
    def get_highest(self, agents): return max(agents, key=lambda a: a.credibility)
