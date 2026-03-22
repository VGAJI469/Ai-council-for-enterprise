"""
Weighted Aggregator - Credibility-weighted voting with dynamic risk and confidence scoring.

Risk and confidence are computed from actual agent prediction data per run,
not from a static formula — so results vary meaningfully with different inputs.
"""

import logging
from typing import List, Dict
from agents.base.base_agent import AgentPrediction

logger = logging.getLogger(__name__)

DECISION_SCORES = {"APPROVE": 0.0, "CONDITIONAL_APPROVE": 0.5, "REJECT": 1.0}

# Map risk_score float ranges → qualitative risk bands used in compute_risk_score
def _risk_band(risk_score: float) -> float:
    """Convert a continuous risk score into a risk band float for aggregation."""
    if risk_score < 0.35:
        return 0.20   # LOW
    elif risk_score < 0.60:
        return 0.50   # MODERATE
    else:
        return 0.90   # HIGH


def compute_risk_score(predictions: List[AgentPrediction]) -> float:
    """
    Compute aggregate risk from actual agent predictions each run.

    Formula:
      base_risk        = credibility-weighted average of per-agent risk bands
      conf_penalty     = 1 - credibility-weighted confidence (low confidence → higher risk)
      dissent_factor   = (reject_votes / total_votes) * 0.3

      final_risk = base_risk * 0.5 + conf_penalty * 0.3 + dissent_factor * 0.2
    """
    if not predictions:
        return 0.5

    total_cred = sum(p.credibility for p in predictions)
    if total_cred == 0:
        total_cred = len(predictions)

    base_risk = sum(
        _risk_band(p.risk_score) * p.credibility for p in predictions
    ) / total_cred

    weighted_conf = sum(p.confidence * p.credibility for p in predictions) / total_cred
    conf_penalty  = 1.0 - weighted_conf

    reject_count   = sum(1 for p in predictions if p.decision == "REJECT")
    dissent_factor = (reject_count / len(predictions)) * 0.3

    final_risk = (base_risk * 0.5) + (conf_penalty * 0.3) + (dissent_factor * 0.2)
    return round(min(max(final_risk, 0.0), 1.0), 4)


def compute_council_confidence(predictions: List[AgentPrediction]) -> float:
    """
    Compute council confidence from actual agreement level per run.

    Formula:
      agreement_ratio    = dominant_position_count / total_agents
      avg_agent_conf     = credibility-weighted mean of per-agent confidences

      council_confidence = agreement_ratio * 0.6 + avg_agent_conf * 0.4
    """
    if not predictions:
        return 0.5

    total_cred = sum(p.credibility for p in predictions)
    if total_cred == 0:
        total_cred = len(predictions)

    # Position diversity
    from collections import Counter
    position_counts = Counter(p.decision for p in predictions)
    dominant_count  = max(position_counts.values())
    agreement_ratio = dominant_count / len(predictions)

    avg_conf = sum(p.confidence * p.credibility for p in predictions) / total_cred

    council_conf = (agreement_ratio * 0.6) + (avg_conf * 0.4)
    return round(min(max(council_conf, 0.0), 1.0), 4)


class WeightedAggregator:

    def aggregate(self, predictions: List[AgentPrediction]) -> Dict:
        if not predictions:
            raise ValueError("No predictions to aggregate.")

        total_cred = sum(p.credibility for p in predictions)
        if total_cred == 0:
            total_cred = len(predictions)

        # Weighted decision score determines verdict
        weighted_decision = sum(
            DECISION_SCORES[p.decision] * p.credibility for p in predictions
        ) / total_cred

        if weighted_decision < 0.33:
            final_decision = "APPROVE"
        elif weighted_decision < 0.66:
            final_decision = "CONDITIONAL_APPROVE"
        else:
            final_decision = "REJECT"

        # Dynamic scores — computed fresh from actual agent data each run
        aggregate_risk  = compute_risk_score(predictions)
        council_conf    = compute_council_confidence(predictions)

        # Sanity checks
        assert 0.0 <= aggregate_risk  <= 1.0, f"risk out of bounds: {aggregate_risk}"
        assert 0.0 <= council_conf    <= 1.0, f"confidence out of bounds: {council_conf}"

        print(f"[Council] verdict={final_decision}")
        print(f"[Council] risk={aggregate_risk}  confidence={council_conf}")

        vote_breakdown = {
            p.agent_role: {
                "decision":    p.decision,
                "risk_score":  round(p.risk_score, 4),
                "credibility": round(p.credibility, 4),
                "weight":      round(p.credibility / total_cred, 4),
                "confidence":  round(p.confidence, 4),
            }
            for p in predictions
        }

        return {
            "final_decision":         final_decision,
            "aggregate_risk_score":   aggregate_risk,
            "ensemble_confidence":    council_conf,
            "total_credibility_pool": round(total_cred, 4),
            "vote_breakdown":         vote_breakdown,
            "quorum_size":            len(predictions),
        }
