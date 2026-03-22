"""Council Session - Orchestrates multi-agent debate."""

import logging
from typing import List, Dict
from datetime import datetime
from agents.base.base_agent import AgentPrediction

logger = logging.getLogger(__name__)


class CouncilSession:
    def __init__(self, session_id: str, agents: list):
        self.session_id = session_id
        self.agents = agents
        self.predictions: List[AgentPrediction] = []
        self.debate_log: List[Dict] = []
        self.timestamp = datetime.utcnow()

    def run(self, financial_data: dict) -> List[AgentPrediction]:
        logger.info(f"Council session {self.session_id} — {len(self.agents)} agents")
        for agent in self.agents:
            try:
                prediction = agent.predict(financial_data)
                self.predictions.append(prediction)
                self.debate_log.append({
                    "agent": agent.role, "decision": prediction.decision,
                    "risk_score": prediction.risk_score,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning
                })
                logger.info(f"  [{agent.role}] {prediction.decision} | risk={prediction.risk_score:.4f}")
            except Exception as e:
                logger.error(f"Agent {agent.role} failed: {e}")
        return self.predictions

    def get_transcript(self) -> dict:
        return {"session_id": self.session_id,
                "timestamp": self.timestamp.isoformat(),
                "debate": self.debate_log}
