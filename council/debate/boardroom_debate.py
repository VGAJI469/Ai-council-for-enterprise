"""
Boardroom Debate Engine
Runs a full multi-round conversational debate between all 5 council agents.
Each agent speaks in their own voice, responds to others, challenges positions,
and the CEO delivers a final binding verdict.
"""

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import uuid
import logging
from agents.evolution.agent_factory import AgentFactory
from council.voting.weighted_aggregator import WeightedAggregator

logging.basicConfig(level=logging.WARNING)

ROLE_TITLES = {
    'strategic_growth':      'CEO',
    'financial_stability':   'CFO',
    'market_expansion':      'Chief Marketing Officer',
    'reputation_risk':       'PR Director',
    'regulatory_compliance': 'Legal Counsel',
}

ROLE_EMOJI = {
    'strategic_growth':      '👔',
    'financial_stability':   '💰',
    'market_expansion':      '📈',
    'reputation_risk':       '📣',
    'regulatory_compliance': '⚖️',
}

def load_config(path='config/agents.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def speak(agent, prompt, max_tokens=700, temperature=0.75):
    return agent.llm.generate(
        prompt=prompt,
        system_prompt=agent.get_system_prompt(),
        model=agent.preferred_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def print_speaker(agent, text):
    title = ROLE_TITLES.get(agent.role, agent.role)
    print()
    print(f'  [ {title.upper()} ]')
    print('-' * 60)
    print(text.strip())
    print()

def run_debate(motion: str, context: dict):
    cfg     = load_config()
    factory = AgentFactory()
    agents  = factory.build_council(cfg)
    agg     = WeightedAggregator()

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
        prompt = f"""
The board is debating the following motion:
"{motion}"

Additional context: {context}

You are the {title}. This is your opening statement to the full executive council.
Give your complete, detailed position on this motion.
Cover every angle relevant to your role — opportunities, risks, financials,
strategy, compliance, reputation — whatever matters most to your mandate.
Be direct, opinionated, and thorough. This is not a summary. This is your argument.
Speak in first person as if addressing the room.
Minimum 250 words.
"""
        response = speak(agent, prompt, max_tokens=700)
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

    all_openings = '\n\n'.join([
        f"{ROLE_TITLES.get(role, role)} said:\n{stmt}"
        for role, stmt in opening_statements.items()
    ])

    cross_statements = {}
    for agent in agents:
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board is debating: "{motion}"

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
Minimum 200 words.
"""
        response = speak(agent, prompt, max_tokens=600)
        cross_statements[agent.role] = response
        print_speaker(agent, response)

    # ── ROUND 3: Heated Rebuttals ──────────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 3 — REBUTTALS')
    print('  Tensions rise. Each executive defends their corner.')
    print('=' * 70)

    all_cross = '\n\n'.join([
        f"{ROLE_TITLES.get(role, role)} responded:\n{stmt}"
        for role, stmt in cross_statements.items()
    ])

    rebuttal_statements = {}
    for agent in agents:
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board is still debating: "{motion}"

Opening statements:
{all_openings}

Cross examination round:
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
"""
        response = speak(agent, prompt, max_tokens=400, temperature=0.8)
        rebuttal_statements[agent.role] = response
        print_speaker(agent, response)

    # ── ROUND 4: Finding Common Ground ────────────────────────────────
    print()
    print('=' * 70)
    print('  ROUND 4 — COMMON GROUND')
    print('  The chair asks each executive what they can agree on.')
    print('=' * 70)

    all_rebuttals = '\n\n'.join([
        f"{ROLE_TITLES.get(role, role)} rebutted:\n{stmt}"
        for role, stmt in rebuttal_statements.items()
    ])

    common_ground = {}
    for agent in agents:
        title = ROLE_TITLES.get(agent.role, agent.role)
        prompt = f"""
The board has been debating: "{motion}"

After three rounds of debate, the chair is asking each executive:
What can you agree on? Where is there common ground?

You are the {title}.
- State the one or two things every executive in this room seems to agree on
- State clearly what your non-negotiable condition is before you support this motion
- Suggest one specific compromise that could bring the council together
- End with your final lean: are you FOR, AGAINST, or CONDITIONAL on this motion?
Be concise and constructive. Maximum 120 words.
"""
        response = speak(agent, prompt, max_tokens=300, temperature=0.6)
        common_ground[agent.role] = response
        print_speaker(agent, response)

    # ── FINAL VERDICT: CEO closes ──────────────────────────────────────
    print()
    print('=' * 70)
    print('  FINAL VERDICT — CEO CLOSES THE DEBATE')
    print('=' * 70)

    all_common = '\n\n'.join([
        f"{ROLE_TITLES.get(role, role)}:\n{stmt}"
        for role, stmt in common_ground.items()
    ])

    ceo = next(a for a in agents if a.role == 'strategic_growth')
    verdict_prompt = f"""
This board has debated the motion: "{motion}"

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
"""

    verdict = speak(ceo, verdict_prompt, max_tokens=900, temperature=0.5)
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

    return result


if __name__ == '__main__':
    # Adding this to ensure module imports from within the script work when called from the root dir via python council/debate/boardroom_debate.py
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    motion = 'The board is considering a $240 million all-cash acquisition of NeuralEdge AI, a fast-growing artificial intelligence startup with 18 patents in generative AI and a $52 million annual recurring revenue growing at 110% year-over-year, but the target has a monthly burn rate of $6.8 million, faces pending regulatory scrutiny from the EU AI Act, has lost two key engineers to competitors in the last quarter, and our own company currently holds only $310 million in liquid reserves -- should the board approve this acquisition at the proposed valuation or walk away and risk losing first-mover advantage in enterprise AI?'

    # Context reflects the high-stakes AI acquisition scenario
    context = {
        'debt_to_income_ratio':       0.38,   # moderate — company is stable but not flush
        'liquidity_ratio_inv':        0.55,   # concerning — $240M would consume 77% of reserves
        'default_probability':        0.22,   # low-moderate — healthy acquirer
        'cash_flow_risk':             0.48,   # moderate — large cash outlay tightens runway
        'competitive_risk':           0.72,   # high — rivals are also eyeing AI acquisitions
        'market_growth_rate':         0.35,   # strong — AI sector booming
        'regulatory_violation_prob':  0.45,   # elevated — EU AI Act uncertainty
        'policy_risk':                0.52,   # high — evolving AI regulation globally
        'legal_risk':                 0.40,   # moderate — IP due diligence risks
        'compliance_score':           0.55,   # below average — target has compliance gaps
        'sentiment_risk':             0.35,   # moderate — public wary of big AI deals
        'brand_risk':                 0.30,   # moderate — acquisition could boost or hurt brand
        'media_risk':                 0.58,   # high — AI acquisitions attract media scrutiny
        'stakeholder_risk':           0.50,   # moderate — shareholders may question valuation
        'customer_churn_risk':        0.20,   # low — AI capabilities could improve retention
        'market_opportunity':         0.82,   # very high — first-mover advantage in enterprise AI
        'investment_amount':          240000000,
        'payback_period_years':       5,
    }

    run_debate(motion, context)
