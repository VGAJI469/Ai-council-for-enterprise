"""
boardroom_stream.py
===================
Asynchronous debate engine that streams debate rounds via Server-Sent Events (SSE).
Wraps the boardroom debate logic in an async generator, using asyncio.to_thread
for non-blocking execution of agent decisions and speak tasks.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List

from council.debate.boardroom_debate import (
    ROLE_TITLES,
    ROLE_EMOJI,
    speak,
    _build_role_context_block,
    _truncate,
    _build_history_block,
    _compute_council_majority,
    DEBATE_MAX_TOKENS,
)
from council.voting.weighted_aggregator import WeightedAggregator
from council.credibility.credibility_manager import CredibilityManager

logger = logging.getLogger("boardroom_stream")

# Map Python role keys (agent focus) to frontend Casing
ROLE_MAP_PY_TO_JS = {
    'strategic_growth': 'CEO',
    'financial_stability': 'CFO',
    'regulatory_compliance': 'LEGAL',
    'market_expansion': 'CMO',
    'reputation_risk': 'PR',
}

ROLE_MAP_JS_TO_PY = {v: k for k, v in ROLE_MAP_PY_TO_JS.items()}


def parse_ceo_verdict(verdict_text: str) -> dict:
    """Parse the CEO's structured text response to extract closing statement and action plan."""
    action_plan = []
    closing_statement = ""

    # Parse Action Plan
    action_plan_match = re.search(r"ACTION PLAN:(.*?)(?:CLOSING STATEMENT:|$)", verdict_text, re.DOTALL | re.IGNORECASE)
    if action_plan_match:
        plan_block = action_plan_match.group(1).strip()
        lines = [line.strip() for line in plan_block.split("\n") if line.strip()]
        for line in lines[:5]:
            # Guess owner based on line content
            owner = "CEO"
            for role in ["CFO", "LEGAL", "CMO", "PR", "CEO"]:
                if role in line.upper():
                    owner = role
                    break
            # Remove line numbering and owner titles from string
            step_text = re.sub(r"^\d+[\.\-\s]+", "", line)
            step_text = re.sub(r"^(CFO|LEGAL|CMO|PR|CEO|GC)[\:\-\s]+", "", step_text, flags=re.IGNORECASE)
            action_plan.append({"step": step_text.strip(), "owner": owner})

    # If parsing failed or action plan is empty, populate clean fallbacks
    if not action_plan:
        action_plan = [
            {"step": "Establish monthly key indicator check gates.", "owner": "CFO"},
            {"step": "Conduct regular compliance audit reviews.", "owner": "LEGAL"},
            {"step": "Monitor user growth and customer acquisition costs.", "owner": "CMO"},
            {"step": "Draft media communications strategy.", "owner": "PR"},
            {"step": "Oversee rollout and pilot operations.", "owner": "CEO"},
        ]

    # Parse Closing Statement
    closing_match = re.search(r"CLOSING STATEMENT:(.*?)$", verdict_text, re.DOTALL | re.IGNORECASE)
    if closing_match:
        closing_statement = closing_match.group(1).strip()
    else:
        closing_statement = verdict_text[-300:].strip()  # Take the end as fallback

    return {"action_plan": action_plan, "closing_statement": closing_statement}


async def run_debate_stream(
    motion: str,
    context: Dict[str, Any],
    agents: List,
    oversight_board,
    supervision_controller,
) -> AsyncGenerator[str, None]:
    """
    Run multi-round debate asynchronously and yield SSE JSON blocks.
    Uses asyncio.to_thread to run synchronous Ollama inference without blocking.
    """
    agg = WeightedAggregator()
    session_id = uuid.uuid4().hex[:12]

    # 1. Yield session_start
    yield f"data: {json.dumps({'type': 'session_start', 'config': {'motion': motion, 'dti': round(context.get('debt_to_income_ratio', 0) * 100), 'creditScore': context.get('credit_score', 0), 'defaultProbability': round(context.get('default_probability', 0) * 100)}})}\n\n"
    await asyncio.sleep(0.1)

    motion_short = _truncate(motion, 450)
    opening_statements = {}
    predictions = {}

    # ── ROUND 1: Opening Statements ──
    for agent in agents:
        js_role = ROLE_MAP_PY_TO_JS.get(agent.role, "CEO")
        title = ROLE_TITLES.get(agent.role, agent.role)
        role_context_block = _build_role_context_block(agent, context)
        prompt = f"""
The board is debating the following motion:
"{motion}"

{role_context_block}

You are the {title}. This is your opening statement to the full executive council.
Give your complete, detailed position on this motion.
Cover every angle relevant to your role — opportunities, risks, financials,
strategy, compliance, reputation — whatever matters most to your mandate.
Be direct, opinionated, and thorough. This is not a summary. This is your argument.
Speak in first person as if addressing the room.
Minimum 250 words. End with a clear, complete concluding sentence.
"""
        # Run speak & predict in background threads
        response = await asyncio.to_thread(speak, agent, prompt, DEBATE_MAX_TOKENS, 0.55, 1)
        pred = await asyncio.to_thread(agent.predict, context)

        opening_statements[agent.role] = response
        predictions[agent.role] = pred

        # Yield speech message
        msg = {
            "type": "message",
            "message": {
                "id": str(uuid.uuid4()),
                "role": js_role,
                "text": response,
                "prediction": {
                    "riskScore": round(pred.risk_score * 100),
                    "confidence": round(pred.confidence * 100),
                    "decision": "CONDITIONAL" if pred.decision == "CONDITIONAL_APPROVE" else pred.decision
                },
                "round": 1,
                "timestamp": int(time.time() * 1000)
            }
        }
        yield f"data: {json.dumps(msg)}\n\n"
        await asyncio.sleep(0.5)

    # Yield initial credibility update
    cred_event = {
        "type": "credibility",
        "scores": [{"role": ROLE_MAP_PY_TO_JS.get(a.role, "CEO"), "score": round(a.credibility * 100)} for a in agents]
    }
    yield f"data: {json.dumps(cred_event)}\n\n"
    await asyncio.sleep(0.1)

    # ── ROUND 2: Cross Examination ──
    all_openings = _build_history_block(opening_statements, "said")
    cross_statements = {}
    for agent in agents:
        js_role = ROLE_MAP_PY_TO_JS.get(agent.role, "CEO")
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board is debating: "{motion_short}"

Here is what every executive said in their opening statement:

{all_openings}

You are the {title}. You have just heard all of your colleagues speak.
Now respond directly to the room. 
- Pick out specific things others said that you agree or strongly disagree with
- Name the person you are responding to by their title
- Challenge any assumptions you think are wrong
- Defend your own position if someone contradicted you
- Bring new arguments or evidence to the table that strengthens your case
- Be conversational, direct, and unafraid to push back
Speak as if you are interrupting the meeting to make your point heard.
Minimum 200 words. End with a clear, complete concluding sentence.
"""
        response = await asyncio.to_thread(speak, agent, prompt, DEBATE_MAX_TOKENS, 0.50, 2)
        cross_statements[agent.role] = response

        # Re-use Round 1 prediction for metadata in later rounds
        pred = predictions[agent.role]
        msg = {
            "type": "message",
            "message": {
                "id": str(uuid.uuid4()),
                "role": js_role,
                "text": response,
                "prediction": {
                    "riskScore": round(pred.risk_score * 100),
                    "confidence": round(pred.confidence * 100),
                    "decision": "CONDITIONAL" if pred.decision == "CONDITIONAL_APPROVE" else pred.decision
                },
                "round": 2,
                "timestamp": int(time.time() * 1000)
            }
        }
        yield f"data: {json.dumps(msg)}\n\n"
        await asyncio.sleep(0.5)

    # ── ROUND 3: Rebuttals ──
    all_cross = _build_history_block(cross_statements, "responded")
    rebuttal_statements = {}
    for agent in agents:
        js_role = ROLE_MAP_PY_TO_JS.get(agent.role, "CEO")
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board is still debating: "{motion_short}"

Here is what each executive said in the cross examination round:
{all_cross}

You are the {title}. This is the rebuttal round.
The debate has gotten more intense. People have challenged your position.
Now fight back. 
- Directly address the most aggressive challenge made against your position
- Expose the weakest argument made by one of your colleagues
- Make your strongest single argument — the one point you will not back down from
- If you have shifted your position slightly based on what you heard, explain why
- End with a clear statement of where you stand
Be passionate, sharp, and decisive. No more than 150 words but make every word count.
End with a definitive concluding statement.
"""
        response = await asyncio.to_thread(speak, agent, prompt, 800, 0.50, 3)
        rebuttal_statements[agent.role] = response

        pred = predictions[agent.role]
        msg = {
            "type": "message",
            "message": {
                "id": str(uuid.uuid4()),
                "role": js_role,
                "text": response,
                "prediction": {
                    "riskScore": round(pred.risk_score * 100),
                    "confidence": round(pred.confidence * 100),
                    "decision": "CONDITIONAL" if pred.decision == "CONDITIONAL_APPROVE" else pred.decision
                },
                "round": 3,
                "timestamp": int(time.time() * 1000)
            }
        }
        yield f"data: {json.dumps(msg)}\n\n"
        await asyncio.sleep(0.5)

    # ── ROUND 4: Finding Common Ground ──
    all_rebuttals = _build_history_block(rebuttal_statements, "rebutted")
    common_ground = {}
    for agent in agents:
        js_role = ROLE_MAP_PY_TO_JS.get(agent.role, "CEO")
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board has been debating: "{motion_short}"

After three rounds of debate, the chair is asking each executive:
What can you agree on? Where is there common ground?

Here is what each executive said in the rebuttal round:
{all_rebuttals}

You are the {title}.
- State the one or two things every executive in this room seems to agree on
- State clearly what your non-negotiable condition is before you support this motion
- Suggest one specific compromise that could bring the council together
- End with your final lean: are you FOR, AGAINST, or CONDITIONAL on this motion?
Be concise and constructive. Maximum 120 words.
End with a complete sentence stating your position.
"""
        response = await asyncio.to_thread(speak, agent, prompt, 600, 0.50, 4)
        common_ground[agent.role] = response

        pred = predictions[agent.role]
        msg = {
            "type": "message",
            "message": {
                "id": str(uuid.uuid4()),
                "role": js_role,
                "text": response,
                "prediction": {
                    "riskScore": round(pred.risk_score * 100),
                    "confidence": round(pred.confidence * 100),
                    "decision": "CONDITIONAL" if pred.decision == "CONDITIONAL_APPROVE" else pred.decision
                },
                "round": 4,
                "timestamp": int(time.time() * 1000)
            }
        }
        yield f"data: {json.dumps(msg)}\n\n"
        await asyncio.sleep(0.5)

    # ── FINAL VERDICT: CEO closes ──
    all_common = _build_history_block(common_ground, "concluded")
    ceo = next(a for a in agents if a.role == 'strategic_growth')
    verdict_prompt = f"""
This board has debated the motion: "{motion_short}"

You have heard four full rounds of debate from your CFO, CMO, PR Director, and Legal Counsel.

Here is where everyone landed:
{all_common}

You are the CEO. The debate is now closed. Deliver the final verdict.
Structure your closing exactly like this:

RISK CLASSIFICATION: (Low / Medium / High) — one sentence on why.

WHAT THE COUNCIL GOT RIGHT: Name each executive and one key insight they contributed.

KEY POINT OF CONTENTION: What was the biggest disagreement and how do you resolve it?

FINAL DECISION: State clearly whether the company will proceed, reject, or conditionally approve this motion.

ACTION PLAN: Five specific steps the company will take, with who owns each step.

CLOSING STATEMENT: One powerful paragraph that captures the spirit of this decision and what it means for the company.

Be authoritative, decisive, and clear. This is the binding decision. Minimum 350 words.
End with a powerful, complete closing sentence.
"""
    verdict = await asyncio.to_thread(speak, ceo, verdict_prompt, 1200, 0.5, 5)

    # Build and yield CEO final verdict message
    ceo_pred = predictions[ceo.role]
    msg = {
        "type": "message",
        "message": {
            "id": str(uuid.uuid4()),
            "role": "CEO",
            "text": verdict,
            "prediction": {
                "riskScore": round(ceo_pred.risk_score * 100),
                "confidence": round(ceo_pred.confidence * 100),
                "decision": "CONDITIONAL" if ceo_pred.decision == "CONDITIONAL_APPROVE" else ceo_pred.decision
            },
            "round": 5,
            "timestamp": int(time.time() * 1000)
        }
    }
    yield f"data: {json.dumps(msg)}\n\n"
    await asyncio.sleep(0.5)


    # ── Vote aggregation & results ──
    prediction_list = list(predictions.values())
    result = agg.aggregate(prediction_list)

    non_ceo_preds = [p for p in prediction_list if p.agent_role != 'strategic_growth']
    council_majority = _compute_council_majority(non_ceo_preds) if non_ceo_preds else _compute_council_majority(prediction_list)
    ceo_pred_obj = next((p for p in prediction_list if p.agent_role == 'strategic_growth'), None)

    # Record to oversight board
    if oversight_board is not None and ceo_pred_obj is not None:
        oversight_board.record(
            session_id=session_id,
            aggregate_risk=result['aggregate_risk_score'],
            council_majority=council_majority,
            ceo_prediction=ceo_pred_obj,
        )

    # Evaluate for mutation
    if supervision_controller is not None:
        # Evaluate streak and performance
        supervision_controller.evaluate_after_session()

    # Update credibility scores
    cred_manager = CredibilityManager()
    cred_manager.update_all(
        agents, prediction_list, result["final_decision"],
        consensus_risk=result['aggregate_risk_score']
    )

    # Yield credibility update
    cred_event = {
        "type": "credibility",
        "scores": [{"role": ROLE_MAP_PY_TO_JS.get(a.role, "CEO"), "score": round(a.credibility * 100)} for a in agents]
    }
    yield f"data: {json.dumps(cred_event)}\n\n"
    await asyncio.sleep(0.1)

    # Yield oversight status update
    oversight_event = {
        "type": "oversight",
        "state": {
            "supervisionScore": round(oversight_board.get_supervision_score() * 100),
            "overrideStreak": oversight_board.count_consecutive_override_failures(),
            "ceoGeneration": ceo.config.get("generation", 1),
            "dominantPattern": oversight_board.get_pattern_summary().get("dominant_pattern", "NONE")
        }
    }
    yield f"data: {json.dumps(oversight_event)}\n\n"
    await asyncio.sleep(0.1)

    # Yield consensus outcome
    parsed_verdict = parse_ceo_verdict(verdict)
    outcome = {
        "verdict": "CONDITIONAL APPROVE" if result["final_decision"] == "CONDITIONAL_APPROVE" else result["final_decision"],
        "councilConfidence": round(result["ensemble_confidence"] * 100),
        "aggregateRisk": round(result["aggregate_risk_score"] * 100),
        "actionPlan": parsed_verdict["action_plan"],
        "closingStatement": parsed_verdict["closing_statement"]
    }
    yield f"data: {json.dumps({'type': 'consensus', 'outcome': outcome})}\n\n"
    await asyncio.sleep(0.1)

    # Yield end
    yield f"data: {json.dumps({'type': 'session_end'})}\n\n"
