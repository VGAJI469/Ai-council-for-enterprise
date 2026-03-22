"""Council REST API routes."""
import uuid
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from agents.evolution.agent_factory import AgentFactory
from council.voting.weighted_aggregator import WeightedAggregator

router = APIRouter(prefix="/council", tags=["council"])

# ── Shared factory (single Ollama connection for all requests) ────────────────
_factory = None
_agents  = None

def _get_agents():
    global _factory, _agents
    if _agents is None:
        with open("config/agents.yaml") as f:
            cfg = yaml.safe_load(f)
        _factory = AgentFactory()
        _agents  = _factory.build_council(cfg)
    return _agents


# ── Request/Response models ───────────────────────────────────────────────────

class CaseRequest(BaseModel):
    case: str
    context: Optional[Dict[str, Any]] = None

class FinancialRecord(BaseModel):
    loan_id: str
    borrower_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    loan_term_months: int
    employment_years: float
    default_history: int = 0

class CouncilDecision(BaseModel):
    session_id: str
    final_decision: str
    aggregate_risk_score: float
    ensemble_confidence: float
    vote_breakdown: Dict[str, Any]
    quorum_size: int


# ── POST /council/run ─────────────────────────────────────────────────────────

@router.post("/run")
async def run_council(req: CaseRequest):
    """
    Run all 5 council agents on a case string.
    Returns the final verdict, aggregate risk score, council confidence,
    and per-agent vote breakdown.
    """
    try:
        agents = _get_agents()
        agg    = WeightedAggregator()

        # Build context from optional payload or sensible defaults
        context = req.context or {
            "decision_topic":        req.case,
            "debt_to_income_ratio":  0.40,
            "credit_score":          660,
            "loan_amount":           120000,
            "default_probability":   0.30,
            "market_growth_rate":    0.05,
            "competitive_risk":      0.50,
            "liquidity_ratio_inv":   0.40,
            "cash_flow_risk":        0.40,
            "regulatory_violation_prob": 0.15,
            "policy_risk":           0.20,
            "legal_risk":            0.15,
            "compliance_score":      0.70,
            "sentiment_risk":        0.30,
            "brand_risk":            0.25,
            "media_risk":            0.28,
            "stakeholder_risk":      0.30,
            "customer_churn_risk":   0.28,
            "market_opportunity":    0.50,
            "investment_amount":     10000000,
            "payback_period_years":  3,
        }

        predictions = []
        for agent in agents:
            pred = agent.predict(context.copy())
            predictions.append(pred)

        result = agg.aggregate(predictions)

        return {
            "session_id":            str(uuid.uuid4()),
            "case":                  req.case,
            "verdict":               result["final_decision"],
            "aggregate_risk_score":  result["aggregate_risk_score"],
            "council_confidence":    result["ensemble_confidence"],
            "vote_breakdown":        result["vote_breakdown"],
            "quorum_size":           result["quorum_size"],
        }

    except Exception as e:
        raise HTTPException(500, f"Council evaluation failed: {e}")


# ── POST /council/evaluate (legacy stub) ──────────────────────────────────────

@router.post("/evaluate", response_model=CouncilDecision)
async def evaluate(record: FinancialRecord):
    """Submit a financial record for council evaluation."""
    raise HTTPException(503, "Wire up pipeline.run_pipeline.run_council_cycle() here.")


# ── GET /council/agents/status ────────────────────────────────────────────────

@router.get("/agents/status")
async def agent_status():
    """Live credibility scores and model assignments for all council agents."""
    try:
        agents = _get_agents()
        return {
            "agents": [a.to_dict() for a in agents],
        }
    except Exception as e:
        return {
            "agents": [],
            "error": str(e),
        }


# ── GET /council/llm/models ──────────────────────────────────────────────────

@router.get("/llm/models")
async def list_models():
    """Check which Ollama models are currently available."""
    try:
        from agents.base.llm_client import LocalLLMClient
        client = LocalLLMClient()
        return {"available_models": client.list_models(), "ollama_url": client.base_url}
    except Exception as e:
        raise HTTPException(503, f"Ollama not reachable: {e}")


# ── GET /council/evolution/log ────────────────────────────────────────────────

@router.get("/evolution/log")
async def evolution_log():
    """Agent evolution history and replacement events."""
    return {"message": "Connect to EvolutionController.get_evolution_log()"}
