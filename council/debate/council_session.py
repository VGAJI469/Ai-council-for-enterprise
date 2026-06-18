"""Council Session - Orchestrates multi-agent debate with history management."""

import logging
from typing import List, Dict
from datetime import datetime
from agents.base.base_agent import AgentPrediction
from utils.logger import get_logger
from utils.audit import AuditRecorder

logger = get_logger(__name__)

# Limit debate history to prevent token overflow
MAX_DEBATE_LOG_SIZE = 500  # Max entries in debate_log
MAX_PREDICTION_HISTORY = 100  # Max per-agent prediction history


class CouncilSession:
    def __init__(self, session_id: str, agents: list):
        self.session_id = session_id
        self.agents = agents
        self.predictions: List[AgentPrediction] = []
        self.debate_log: List[Dict] = []
        self.timestamp = datetime.utcnow()
        self.audit = AuditRecorder(self.session_id)
        # record session start
        try:
            self.audit.record_event("session_start", {"agent_count": len(self.agents)})
        except Exception:
            logger.exception("Failed to record session start audit")

    def run(self, financial_data: dict) -> List[AgentPrediction]:
        logger.info(f"Council session {self.session_id} — {len(self.agents)} agents")
        for agent in self.agents:
            try:
                prediction = agent.predict(financial_data)
                self.predictions.append(prediction)
                
                # Add to debate log with size limit
                log_entry = {
                    "agent": agent.role, 
                    "decision": prediction.decision,
                    "risk_score": prediction.risk_score,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning[:200] if prediction.reasoning else "",  # Truncate reasoning
                }
                self.debate_log.append(log_entry)
                # record agent decision to audit
                try:
                    self.audit.record_event("agent_decision", {"agent": agent.role, "decision": prediction.decision, "risk_score": prediction.risk_score, "confidence": prediction.confidence})
                except Exception:
                    logger.exception("Failed to record agent decision to audit")
                
                # Enforce size limit on debate log
                if len(self.debate_log) > MAX_DEBATE_LOG_SIZE:
                    self.debate_log = self.debate_log[-MAX_DEBATE_LOG_SIZE:]
                    logger.warning(f"Debate log exceeded {MAX_DEBATE_LOG_SIZE} entries — trimmed to last {MAX_DEBATE_LOG_SIZE}")
                
                logger.info(f"  [{agent.role}] {prediction.decision} | risk={prediction.risk_score:.4f} | cred={prediction.credibility:.3f}")
            except Exception as e:
                logger.exception(f"Agent {agent.role} failed during predict: {e}")
                # Create fallback prediction instead of crashing
                fallback_pred = self._create_fallback_prediction(agent)
                self.predictions.append(fallback_pred)
                logger.warning(f"  [{agent.role}] FALLBACK | Empty response fallback triggered")
                try:
                    self.audit.record_error(e, {"agent": agent.role})
                except Exception:
                    logger.exception("Failed to record error to audit")
        
        return self.predictions
    def _propagate_consensus(self, consensus_risk: float) -> None:
        for agent in self.agents:
            agent.last_consensus_risk = consensus_risk
        logger.info(
            f"Council session {self.session_id} — consensus risk propagated: {consensus_risk:.4f}"
    )

    def _create_fallback_prediction(self, agent) -> AgentPrediction:
        """Create a safe fallback prediction when agent fails."""
        from agents.base.base_agent import AgentPrediction, FALLBACK_REASONING
        return AgentPrediction(
            agent_id=agent.agent_id,
            agent_role=agent.role,
            risk_score=0.5,
            confidence=0.2,
            decision="CONDITIONAL_APPROVE",
            reasoning=FALLBACK_REASONING,
            model_used="fallback",
            credibility=agent.credibility,
            base_risk=0.5,
            role_risk=0.5,
        )

    def get_transcript(self) -> dict:
        """Return a trimmed transcript to avoid memory bloat."""
        # Limit reasoning length in transcript
        trimmed_log = [
            {
                "agent": entry["agent"],
                "decision": entry["decision"],
                "risk_score": entry["risk_score"],
                "confidence": entry["confidence"],
                "reasoning": entry.get("reasoning", "")[:150],  # Max 150 chars per entry
            }
            for entry in self.debate_log[-50:]  # Keep only last 50 entries
        ]
        
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "debate": trimmed_log,
            "total_entries": len(self.debate_log),
        }
