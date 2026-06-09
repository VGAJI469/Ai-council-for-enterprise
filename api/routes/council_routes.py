"""Council REST API routes."""
import uuid
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from agents.evolution.agent_factory import AgentFactory
from council.voting.weighted_aggregator import WeightedAggregator
from utils.logger import get_logger
from utils.audit import AuditRecorder

router = APIRouter(prefix="/council", tags=["council"])

# ── Shared singletons (single Ollama connection for all requests) ─────────────
_factory               = None
_agents                = None
_oversight_board       = None
_supervision_controller = None
_performance_tracker   = None
_ceo_mutator           = None


def _get_agents():
    global _factory, _agents
    if _agents is None:
        with open("config/agents.yaml") as f:
            cfg = yaml.safe_load(f)
        _factory = AgentFactory()
        _agents  = _factory.build_council(cfg)
    return _agents


def _get_oversight_components():
    """
    Lazy-initialise the CEO oversight board, performance tracker, mutator,
    and supervision controller.  All four are singletons — shared across
    every request so state accumulates across debate sessions.

    Returns (oversight_board, supervision_controller).
    """
    global _oversight_board, _supervision_controller, _performance_tracker, _ceo_mutator

    if _oversight_board is None:
        from council.oversight.ceo_oversight    import CEOOversightBoard
        from council.oversight.ceo_performance  import CEOPerformanceTracker
        from evolution.mutation.ceo_mutator     import CEOMutator
        from evolution.selection.ceo_supervision_controller import CEOSupervisionController

        agents = _get_agents()
        ceo_agent = next(
            (a for a in agents if a.role == "strategic_growth"), agents[0]
        )

        with open("config/agents.yaml") as f:
            cfg = yaml.safe_load(f)
        sup_cfg = cfg.get("ceo_supervision", {})

        _oversight_board     = CEOOversightBoard()
        _performance_tracker = CEOPerformanceTracker()
        _ceo_mutator         = CEOMutator()
        _supervision_controller = CEOSupervisionController(
            ceo_agent             = ceo_agent,
            oversight_board       = _oversight_board,
            performance_tracker   = _performance_tracker,
            ceo_mutator           = _ceo_mutator,
            agent_factory         = _factory,
            supervision_threshold = sup_cfg.get("supervision_threshold", 3),
            performance_floor     = sup_cfg.get("performance_floor", 0.35),
        )

    return _oversight_board, _supervision_controller


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
    logger = get_logger("api.routes.council")
    try:
        session_id = str(uuid.uuid4())
        audit = AuditRecorder(session_id)

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
        audit.record_event("session_request", {"case": req.case})
        for agent in agents:
            pred = agent.predict(context.copy())
            predictions.append(pred)
            try:
                audit.record_event("agent_prediction", {"agent": pred.agent_role, "decision": pred.decision, "risk_score": pred.risk_score, "confidence": pred.confidence})
            except Exception:
                logger.exception("Failed to record agent prediction to audit")

        result = agg.aggregate(predictions)

        payload = {
            "session_id":            session_id,
            "case":                  req.case,
            "verdict":               result["final_decision"],
            "aggregate_risk_score":  result["aggregate_risk_score"],
            "council_confidence":    result["ensemble_confidence"],
            "vote_breakdown":        result["vote_breakdown"],
            "quorum_size":           result["quorum_size"],
        }
        try:
            audit.record_event("session_end", payload)
            report_path = audit.write_report()
            logger.info("Generated audit report", extra={"extra": {"session_id": session_id, "report_path": report_path}})
        except Exception:
            logger.exception("Failed to write audit report")

        return payload

    except Exception as e:
        logger = get_logger("api.routes.council")
        logger.exception("Council evaluation failed")
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


# ── GET /council/ceo/supervision ─────────────────────────────────────────────

@router.get("/ceo/supervision")
async def ceo_supervision():
    """
    Return the CEO oversight dashboard: generation, supervision score,
    override statistics, performance score, and mutation log.

    All values accumulate across debate sessions as long as the server is
    running.  Use POST /council/ceo/reset-supervision to clear state for
    a fresh demo run.
    """
    try:
        board, controller = _get_oversight_components()
        summary  = board.get_pattern_summary()
        status   = controller.get_status()
        mut_log  = _ceo_mutator.get_mutation_log() if _ceo_mutator else []
        last_mut_reason = mut_log[-1]["mutation_reason"] if mut_log else None

        return {
            "ceo_generation":       status["ceo_generation"],
            "supervision_score":    status["supervision_score"],
            "override_streak":      status["override_streak"],
            "performance_score":    status["performance_score"],
            "total_decisions":      summary["total_decisions"],
            "total_overrides":      summary["total_overrides"],
            "override_rate":        summary["override_rate"],
            "dominant_pattern":     summary["dominant_pattern"],
            "last_mutation_reason": last_mut_reason,
            "mutation_log":         mut_log,
        }
    except Exception as e:
        raise HTTPException(500, f"CEO supervision query failed: {e}")


# ── POST /council/ceo/reset-supervision ──────────────────────────────────────

@router.post("/ceo/reset-supervision")
async def reset_ceo_supervision():
    """
    Clear CEO oversight board history and reset the performance tracker.
    Useful for demo resets between independent debate runs.

    Does NOT restart the CEO agent or change its generation — it only
    wipes the accumulated override and performance history so the next
    session starts with a clean slate.
    """
    global _oversight_board, _performance_tracker, _supervision_controller, _ceo_mutator

    try:
        if _oversight_board is not None:
            from council.oversight.ceo_oversight   import CEOOversightBoard
            from council.oversight.ceo_performance import CEOPerformanceTracker
            from evolution.mutation.ceo_mutator    import CEOMutator

            _oversight_board     = CEOOversightBoard()
            _performance_tracker = CEOPerformanceTracker()
            _ceo_mutator         = CEOMutator()

            # Update supervision controller references without replacing the agent
            if _supervision_controller is not None:
                _supervision_controller.oversight_board     = _oversight_board
                _supervision_controller.performance_tracker = _performance_tracker

        return {
            "status":  "reset_complete",
            "message": "CEO oversight board and performance tracker cleared.",
        }
    except Exception as e:
        raise HTTPException(500, f"CEO supervision reset failed: {e}")


# ── GET /council/debate (Streaming SSE) ───────────────────────────────────────
from fastapi.responses import StreamingResponse
from council.debate.boardroom_stream import run_debate_stream

@router.get("/debate")
async def stream_debate(
    motion: str,
    dti: float,
    creditScore: float,
    defaultProbability: float,
):
    """
    Start a live, multi-round boardroom debate and stream execution updates
    as Server-Sent Events (SSE).
    """
    context = {
        "debt_to_income_ratio":      dti / 100.0,
        "credit_score":              creditScore,
        "default_probability":       defaultProbability / 100.0,
        "market_growth_rate":        0.05,
        "competitive_risk":          0.50,
        "liquidity_ratio_inv":       0.40,
        "cash_flow_risk":            0.40,
        "regulatory_violation_prob": 0.15,
        "policy_risk":               0.20,
        "legal_risk":                0.15,
        "compliance_score":          0.70,
        "sentiment_risk":            0.30,
        "brand_risk":                0.25,
        "media_risk":                0.28,
        "stakeholder_risk":          0.30,
        "customer_churn_risk":       0.28,
        "market_opportunity":        0.50,
        "investment_amount":         10000000,
        "payback_period_years":      3,
    }

    agents = _get_agents()
    board, controller = _get_oversight_components()

    return StreamingResponse(
        run_debate_stream(motion, context, agents, board, controller),
        media_type="text/event-stream"
    )


