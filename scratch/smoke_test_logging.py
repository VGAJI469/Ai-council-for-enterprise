from council.debate.council_session import CouncilSession
from agents.base.base_agent import AgentPrediction

class MockAgent:
    def __init__(self, agent_id, role):
        self.agent_id = agent_id
        self.role = role
        self.credibility = 1.0
    def predict(self, ctx):
        return AgentPrediction(
            agent_id=self.agent_id,
            agent_role=self.role,
            risk_score=0.42,
            confidence=0.72,
            decision="CONDITIONAL_APPROVE",
            reasoning="Mock reasoning for testing structured logging and audit.",
            model_used="mock-model",
        )

if __name__ == '__main__':
    agents = [MockAgent('a1','financial_stability'), MockAgent('a2','strategic_growth')]
    session = CouncilSession('testsession123', agents)
    preds = session.run({'default_probability': 0.33})
    print('Predictions:', [p.decision for p in preds])
    # write audit report
    try:
        rp = session.audit.write_report()
        print('Wrote report to', rp)
    except Exception as e:
        print('Failed to write report', e)
