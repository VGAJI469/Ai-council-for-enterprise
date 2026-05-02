"""
Boardroom Debate Engine
Runs a full multi-round conversational debate between all 5 council agents.
Each agent speaks in their own voice, responds to others, challenges positions,
and the CEO delivers a final binding verdict.

Upgrade note (Item 7):
  Opening statement prompts now include a role-specific context block that:
  - Selects the 4 most relevant metrics for each agent's role from the context dict
  - Flags any metric that breaches a critical danger threshold with a warning marker
  This replaces the generic single-paragraph context injection that gave every
  agent the same undifferentiated view of the full context dict.

Stability patch:
  - Prompt size control: motion text truncated to 600 chars; debate history
    limited to LAST round only (not cumulative) to prevent token overflow
  - Response validation: speak_with_retry() retries up to 3 times on empty/
    short/truncated responses before inserting a structured fallback
  - Temperature reduced across all rounds (opening 0.6→, cross 0.6→, rebuttal 0.65→)
  - Per-agent logging: role, model, response length, retry count, truncation flag
  - Deterministic seeds per agent per round for reproducibility
"""

import sys
import os
import io
import hashlib
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import uuid
import logging
from collections import Counter
from typing import Optional
from agents.evolution.agent_factory import AgentFactory
from council.voting.weighted_aggregator import WeightedAggregator
from agents.base.base_agent import ROLE_METRICS, DANGER_THRESHOLDS

# Oversight imports — lazy-typed to avoid hard coupling at module load time
# (CEOOversightBoard and CEOSupervisionController are injected, not imported globally)

# Suppress all logging to show only debate output
logging.basicConfig(level=logging.CRITICAL)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.disable(logging.CRITICAL)
logger = logging.getLogger("boardroom_debate")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_MOTION_CHARS       = 450      # Truncate motion text to ensure room for response
MAX_STATEMENT_CHARS    = 600      # Truncate per-agent statements in history
MAX_SPEAK_RETRIES      = 3        # Retry speak() on empty/truncated output
MIN_RESPONSE_LENGTH    = 60       # Minimum chars for valid debate response (raised from 40)
FALLBACK_TEMPERATURE   = 0.45     # Lower temp used on retry attempts
MAX_DEBATE_HISTORY     = 12       # Limit debate rounds to prevent token overflow
DEBATE_MAX_TOKENS      = 1200     # Max tokens per agent per round (increased from 900)

ROLE_TITLES = {
    'strategic_growth':      'CEO',
    'financial_stability':   'CFO',
    'market_expansion':      'Chief Marketing Officer',
    'reputation_risk':       'PR Director',
    'regulatory_compliance': 'Legal Counsel',
}

ROLE_EMOJI = {
    'strategic_growth':      '[CEO]',
    'financial_stability':   '[CFO]',
    'market_expansion':      '[CMO]',
    'reputation_risk':       '[PRD]',
    'regulatory_compliance': '[LC]',
}

def load_config(path='config/agents.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Prompt utilities ──────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a sentence boundary when possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at the last sentence boundary
    for marker in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
        idx = truncated.rfind(marker)
        if idx > max_chars * 0.6:
            return truncated[:idx + 1]
    return truncated.rstrip() + '…'


def _build_history_block(statements: dict, label: str = "said") -> str:
    """
    Build a compact history block from a round's statements.
    Each statement is truncated to MAX_STATEMENT_CHARS to control prompt size.
    """
    parts = []
    for role, stmt in statements.items():
        title = ROLE_TITLES.get(role, role)
        trimmed = _truncate(stmt, MAX_STATEMENT_CHARS)
        parts.append(f"{title} {label}:\n{trimmed}")
    return '\n\n'.join(parts)


def _build_role_context_block(agent, context: dict) -> str:
    """
    Build a role-specific context summary for opening statement prompts.

    Selects the 4 metrics most relevant to the agent's role and flags any that
    breach critical danger thresholds. Replaces the raw context dict dump which
    gave every agent an identical, undifferentiated view of all metrics.

    Used in boardroom_debate.py opening statements (700-token budget) to
    complement the same logic in base_agent._generate_reasoning().
    """
    role_keys = ROLE_METRICS.get(agent.role, list(context.keys())[:4])
    lines = ["Key metrics for your role:"]
    danger_alerts = []

    for key in role_keys:
        val = context.get(key)
        if val is None:
            continue
        label = key.replace('_', ' ').title()
        line  = f"  {label}: {val}"

        if key in DANGER_THRESHOLDS:
            op, threshold = DANGER_THRESHOLDS[key]
            try:
                triggered = (op == '>' and float(val) > threshold) or \
                            (op == '<' and float(val) < threshold)
                if triggered:
                    line += f"  [!] CRITICAL (threshold {op}{threshold})"
                    danger_alerts.append(f"{label}={val}")
            except (TypeError, ValueError):
                pass
        lines.append(line)

    if danger_alerts:
        lines.append(f"\n[!] DANGER ZONE: {', '.join(danger_alerts)}")
        lines.append("Address these critical signals explicitly in your statement.")

    return "\n".join(lines)


# ── Agent seed utility ────────────────────────────────────────────────────────

def _agent_seed(agent, round_num: int) -> int:
    """Deterministic seed per agent per round for reproducibility."""
    raw = f"{agent.agent_id}:{agent.role}:round{round_num}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) % 100000


# ── Core speak function with retry + validation ──────────────────────────────

def speak(agent, prompt, max_tokens=None, temperature=0.55, round_num=0):
    """
    Send a prompt to the agent's LLM and return the response.
    Uses generate_with_validation() for automatic retry on empty/truncated output.
    
    Args:
        agent: The agent speaking
        prompt: The prompt text
        max_tokens: Max tokens (default: DEBATE_MAX_TOKENS = 1200)
        temperature: Temperature for generation (default: 0.55)
        round_num: Round number for deterministic seeding
    """
    max_tokens = max_tokens or DEBATE_MAX_TOKENS
    seed = _agent_seed(agent, round_num)
    title = ROLE_TITLES.get(agent.role, agent.role)

    logger.info(
        "SPEAK | round=%d | role=%s | model=%s | max_tokens=%d | temp=%.2f | seed=%d",
        round_num, title, agent.preferred_model, max_tokens, temperature, seed,
    )

    response = agent.llm.generate_with_validation(
        prompt=prompt,
        system_prompt=agent.get_system_prompt(),
        model=agent.preferred_model,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        role=agent.role,
        min_length=MIN_RESPONSE_LENGTH,
    )

    logger.info(
        "RESPONSE | round=%d | role=%s | length=%d",
        round_num, title, len(response),
    )

    return response


def _compute_council_majority(predictions: list) -> str:
    """
    Return the plurality decision among the given predictions.
    In case of an exact tie, returns CONDITIONAL_APPROVE as a safe neutral.
    """
    counts = Counter(p.decision for p in predictions)
    if not counts:
        return "CONDITIONAL_APPROVE"
    top_count = max(counts.values())
    top_decisions = [d for d, c in counts.items() if c == top_count]
    if len(top_decisions) == 1:
        return top_decisions[0]
    # Tie-breaker: prefer CONDITIONAL_APPROVE as a neutral outcome
    return "CONDITIONAL_APPROVE" if "CONDITIONAL_APPROVE" in top_decisions else top_decisions[0]


def print_speaker(agent, text):
    title = ROLE_TITLES.get(agent.role, agent.role)
    emoji = ROLE_EMOJI.get(agent.role, '🎙️')
    print()
    print(f'  {emoji} [ {title.upper()} ]')
    print('-' * 60)
    print(text.strip())
    print()


# ── Main debate engine ────────────────────────────────────────────────────────

def run_debate(
    motion: str,
    context: dict,
    oversight_board=None,        # Optional[CEOOversightBoard]
    supervision_controller=None, # Optional[CEOSupervisionController]
):
    cfg        = load_config()
    factory    = AgentFactory()
    agents     = factory.build_council(cfg)
    agg        = WeightedAggregator()
    session_id = uuid.uuid4().hex[:12]   # unique identifier for oversight board records

    # Truncated motion for use in later round prompts (keeps prompt size bounded)
    motion_short = _truncate(motion, MAX_MOTION_CHARS)

    print()
    print('=' * 70)
    print('       ENTERPRISE BOARDROOM DEBATE')
    print('=' * 70)
    print(f'  MOTION: {motion}')
    print('=' * 70)

    # ── ROUND 1: Opening Statements ───────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 1 — OPENING STATEMENTS')
    print('  Each executive gives their full position on the motion.')
    print('=' * 70)

    opening_statements = {}
    predictions = []

    for agent in agents:
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
        response = speak(agent, prompt, max_tokens=DEBATE_MAX_TOKENS, temperature=0.55, round_num=1)
        opening_statements[agent.role] = response
        print_speaker(agent, response)
        pred = agent.predict(context)
        predictions.append(pred)

    # ── ROUND 2: Cross Examination ─────────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 2 — CROSS EXAMINATION')
    print('  Each executive responds to what they just heard.')
    print('  They challenge, agree, push back, and add new arguments.')
    print('=' * 70)

    # Only pass opening statements (last 1 round), truncated per agent
    all_openings = _build_history_block(opening_statements, "said")

    cross_statements = {}
    for agent in agents:
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
        response = speak(agent, prompt, max_tokens=DEBATE_MAX_TOKENS, temperature=0.50, round_num=2)
        cross_statements[agent.role] = response
        print_speaker(agent, response)

    # ── ROUND 3: Heated Rebuttals ──────────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 3 — REBUTTALS')
    print('  Tensions rise. Each executive defends their corner.')
    print('=' * 70)

    # KEY FIX: Only pass the LAST round (cross examination), NOT full history
    # This prevents token overflow that was causing empty responses
    all_cross = _build_history_block(cross_statements, "responded")

    rebuttal_statements = {}
    for agent in agents:
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
        response = speak(agent, prompt, max_tokens=800, temperature=0.50, round_num=3)
        rebuttal_statements[agent.role] = response
        print_speaker(agent, response)

    # ── ROUND 4: Finding Common Ground ────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 4 — COMMON GROUND')
    print('  The chair asks each executive what they can agree on.')
    print('=' * 70)

    # Only pass rebuttals (last round), not full cumulative history
    all_rebuttals = _build_history_block(rebuttal_statements, "rebutted")

    common_ground = {}
    for agent in agents:
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
        response = speak(agent, prompt, max_tokens=600, temperature=0.50, round_num=4)
        common_ground[agent.role] = response
        print_speaker(agent, response)

    # ── FINAL VERDICT: CEO closes ──────────────────────────────────────
    print()
    print('=' * 70)
    print('  FINAL VERDICT — CEO CLOSES THE DEBATE')
    print('=' * 70)

    # Pass only the common ground round (most compact, final positions)
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

    verdict = speak(ceo, verdict_prompt, max_tokens=1200, temperature=0.5, round_num=5)
    print_speaker(ceo, verdict)

    # ── Vote Tally ─────────────────────────────────────────────────────
    result = agg.aggregate(predictions)

    print()
    print('=' * 70)
    print('  WEIGHTED VOTE TALLY')
    print('=' * 70)
    for role, v in result['vote_breakdown'].items():
        title  = ROLE_TITLES.get(role, role)
        bar    = '=' * int(v['weight'] * 40)
        space  = '-' * (40 - int(v['weight'] * 40))
        print(f'  {title:<26} {v["decision"]:<22} [{bar}{space}]')
    print()
    print(f'  FINAL WEIGHTED DECISION : {result["final_decision"]}')
    print(f'  AGGREGATE RISK SCORE    : {result["aggregate_risk_score"]}')
    print(f'  COUNCIL CONFIDENCE      : {result["ensemble_confidence"]}')
    print('=' * 70)

    # ── CEO Oversight Recording ────────────────────────────────────────
    # Compute council majority from all agent predictions (non-CEO agents
    # drive the majority; CEO prediction is also in predictions list so we
    # re-use the full list and let the count rule settle it naturally).
    non_ceo_preds   = [p for p in predictions if p.agent_role != 'strategic_growth']
    council_majority = _compute_council_majority(non_ceo_preds) if non_ceo_preds \
                       else _compute_council_majority(predictions)

    # CEO's own prediction object (from predict() called after its opening statement)
    ceo_pred_obj = next(
        (p for p in predictions if p.agent_role == 'strategic_growth'), None
    )

    # Oversight board fields — defaults when oversight is not wired in
    ceo_override_detected  = False
    ceo_override_type      = None
    ceo_supervision_score  = 1.0
    ceo_generation         = ceo.config.get('generation', 1)
    ceo_performance_score  = 0.5
    mutation_triggered     = False

    if oversight_board is not None and ceo_pred_obj is not None:
        record = oversight_board.record(
            session_id       = session_id,
            aggregate_risk   = result['aggregate_risk_score'],
            council_majority = council_majority,
            ceo_prediction   = ceo_pred_obj,
        )
        ceo_override_detected = record.override_detected
        ceo_override_type     = record.override_direction
        ceo_supervision_score = oversight_board.get_supervision_score()

    if supervision_controller is not None:
        new_ceo = supervision_controller.evaluate_after_session()
        mutation_triggered = (new_ceo.agent_id != ceo.agent_id)
        ceo_generation     = new_ceo.config.get('generation', 1)
        ceo_performance_score = supervision_controller.performance_tracker.get_score()

    # ── Supervision summary line ───────────────────────────────────────
    print()
    print('=' * 70)
    print('  CEO SUPERVISION SUMMARY')
    print('=' * 70)
    print(f'  Override detected   : {"YES — " + str(ceo_override_type) if ceo_override_detected else "No"}')
    print(f'  Supervision score   : {ceo_supervision_score:.4f}')
    print(f'  CEO generation      : {ceo_generation}')
    print(f'  Performance score   : {ceo_performance_score:.4f}')
    print(f'  Mutation triggered  : {"YES" if mutation_triggered else "No"}')
    print('=' * 70)

    result.update({
        'ceo_override_detected':  ceo_override_detected,
        'ceo_override_type':      ceo_override_type,
        'ceo_supervision_score':  ceo_supervision_score,
        'ceo_generation':         ceo_generation,
        'ceo_performance_score':  ceo_performance_score,
    })

    return result


if __name__ == '__main__':
    # Adding this to ensure module imports from within the script work when called from the root dir via python council/debate/boardroom_debate.py
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    motion = "The board is considering deploying an autonomous AI underwriting system that will replace 340 human loan officers across 12 regional offices, reduce loan processing time from 11 days to 4 hours, and is projected to increase annual loan volume by $2.1 billion — however the system has a documented 8.3% error rate on minority applicant profiles, is currently under pre-emptive investigation by the CFPB for potential Fair Lending Act violations, our legal team estimates a 60% probability of class-action litigation within 18 months, the three largest competitor banks have already deployed similar systems and are capturing market share at 3x our current growth rate, and our own internal employee survey shows 71% of staff believe this deployment would 'fundamentally damage the company\'s integrity' — should the board approve full deployment, implement a limited pilot in 2 regional offices, or halt the project entirely and invest the $180 million implementation budget into augmenting human officers with AI assistance tools instead?"

    # Context for AI autonomous underwriting system scenario
    context = {
        # Financial
        "debt_to_income_ratio":         0.31,
        "loan_amount":                  180_000_000,
        "investment_amount":            180_000_000,
        "payback_period_years":         4,
        "default_probability":          0.18,
        "cash_flow_risk":               0.35,
        "liquidity_ratio_inv":          0.28,

        # Market
        "market_growth_rate":           0.61,
        "competitive_risk":             0.78,
        "market_opportunity":           0.85,
        "customer_churn_risk":          0.33,

        # Legal / Regulatory
        "regulatory_violation_prob":    0.72,
        "policy_risk":                  0.81,
        "legal_risk":                   0.74,
        "compliance_score":             0.29,

        # Reputation
        "sentiment_risk":               0.68,
        "brand_risk":                   0.61,
        "media_risk":                   0.76,
        "stakeholder_risk":             0.64,

        # Specifics
        "ai_error_rate_minority":       0.083,
        "litigation_probability_18mo":  0.60,
        "staff_integrity_concern_pct":  0.71,
        "competitor_growth_multiplier": 3.0,
        "loan_volume_increase":         2_100_000_000,
        "processing_time_reduction":    0.964,
        "headcount_reduction":          340,
    }

    run_debate(motion, context)
