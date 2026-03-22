"""Unit tests for credibility update formula."""
import pytest
from council.credibility.credibility_manager import CredibilityManager

class MockAgent:
    def __init__(self, role, credibility):
        self.role = role
        self.credibility = credibility
        self.prediction_history = []
    def update_credibility(self, alpha, beta, gamma, delta, perf, agree, err):
        self.credibility = alpha*self.credibility + beta*perf + gamma*agree - delta*err
        self.credibility = max(0.01, min(1.0, self.credibility))
        return self.credibility

def test_good_performance_raises_credibility():
    agent = MockAgent("ceo", 0.7)
    agent.update_credibility(0.7, 0.15, 0.10, 0.05, 1.0, 1.0, 0.0)
    assert agent.credibility > 0.7

def test_poor_performance_lowers_credibility():
    agent = MockAgent("cfo", 0.8)
    agent.update_credibility(0.7, 0.15, 0.10, 0.05, 0.0, 0.0, 1.0)
    assert agent.credibility < 0.8

def test_credibility_bounded_above():
    agent = MockAgent("legal", 1.0)
    for _ in range(10):
        agent.update_credibility(0.7, 0.15, 0.10, 0.05, 1.0, 1.0, 0.0)
    assert agent.credibility <= 1.0

def test_credibility_bounded_below():
    agent = MockAgent("pr", 0.05)
    for _ in range(10):
        agent.update_credibility(0.7, 0.15, 0.10, 0.05, 0.0, 0.0, 1.0)
    assert agent.credibility >= 0.01
