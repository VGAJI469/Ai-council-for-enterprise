"""Unit tests for weighted decision aggregator."""
import pytest
from datetime import datetime
from council.voting.weighted_aggregator import WeightedAggregator
from agents.base.base_agent import AgentPrediction

def make_pred(role, decision, risk, credibility):
    return AgentPrediction(
        agent_id=f"{role}_001", agent_role=role, risk_score=risk,
        confidence=0.85, decision=decision, reasoning="test", credibility=credibility
    )

def test_unanimous_approve():
    agg = WeightedAggregator()
    preds = [make_pred("ceo", "APPROVE", 0.2, 1.0),
             make_pred("cfo", "APPROVE", 0.22, 1.0),
             make_pred("legal", "APPROVE", 0.18, 1.0)]
    result = agg.aggregate(preds)
    assert result["final_decision"] == "APPROVE"

def test_unanimous_reject():
    agg = WeightedAggregator()
    preds = [make_pred("ceo", "REJECT", 0.85, 1.0),
             make_pred("cfo", "REJECT", 0.9, 1.0),
             make_pred("legal", "REJECT", 0.88, 1.0)]
    result = agg.aggregate(preds)
    assert result["final_decision"] == "REJECT"

def test_high_credibility_dominates():
    agg = WeightedAggregator()
    preds = [make_pred("cfo", "REJECT", 0.8, 5.0),   # high credibility
             make_pred("ceo", "APPROVE", 0.2, 0.1),   # low credibility
             make_pred("marketing", "APPROVE", 0.25, 0.1)]
    result = agg.aggregate(preds)
    assert result["final_decision"] == "REJECT"
