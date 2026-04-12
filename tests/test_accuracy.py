"""
tests/test_accuracy.py

Accuracy & Reliability Test Suite for the Adaptive AI Enterprise Council.

Sections:
  1. Baseline Accuracy / Consistency Test
  2. Cross-Agent Conflict / Differentiation Test
  3. JSON Reliability Test
  4. Confidence Calibration Test
  5. Response Time Benchmark
  6. Output Variance Test (determinism check)
  7. Full Accuracy Report (all sections + summary table + JSON export)

Run full suite:
    python tests/test_accuracy.py

Single-agent mode:
    python tests/test_accuracy.py --agent cfo

Quick mode (1 iteration instead of 3-5):
    python tests/test_accuracy.py --quick

Combine:
    python tests/test_accuracy.py --agent ceo --quick
"""

import sys
import os
import json
import time
import argparse
import logging
import datetime
import statistics

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN  + Style.BRIGHT
    RED    = Fore.RED    + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    CYAN   = Fore.CYAN   + Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = RESET = ""
    print("[WARNING] colorama not installed. Run: pip install colorama")

logging.basicConfig(level=logging.WARNING)

import pytest

from agents.evolution.agent_factory import AgentFactory
from agents.base.llm_client import LocalLLMClient, ROLE_MODEL_MAP

# ── Friendly label maps ───────────────────────────────────────────────────────
ROLE_TITLE = {
    "strategic_growth":      "CEO",
    "financial_stability":   "CFO",
    "market_expansion":      "Marketing",
    "reputation_risk":       "PR",
    "regulatory_compliance": "Legal",
}

AGENT_KEY_MAP = {
    "ceo":       "strategic_growth",
    "cfo":       "financial_stability",
    "marketing": "market_expansion",
    "pr":        "reputation_risk",
    "legal":     "regulatory_compliance",
}

MODEL_TIMEOUT_THRESHOLDS = {
    "phi3":        15,
    "phi":         15,
    "llama3":      30,
    "mistral":     60,
    "deepseek-r1": 45,
}


# ── Loader ────────────────────────────────────────────────────────────────────
def load_agents(filter_role: str = None):
    """Load council agents from config, optionally filtered to one role."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "agents.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    factory = AgentFactory()
    all_agents = factory.build_council(cfg)

    if filter_role:
        return [a for a in all_agents if a.role == filter_role]
    return all_agents


@pytest.fixture
def agents():
    """Pytest fixture that provides all council agents."""
    return load_agents()


# ── Test data ─────────────────────────────────────────────────────────────────
CONSISTENCY_CASE = {
    "decision_topic":        "Should we expand into Tier-2 cities with unsecured consumer loans at aggressive interest rates given rising interest rates and an NPA rate of 4.2%?",
    "debt_to_income_ratio":  0.58,
    "credit_score":          610,
    "loan_amount":           150000,
    "default_probability":   0.38,
    "market_growth_rate":    0.25,
    "competitive_risk":      0.65,
    "liquidity_ratio_inv":   0.42,
    "cash_flow_risk":        0.50,
    "regulatory_violation_prob": 0.20,
    "policy_risk":           0.30,
    "legal_risk":            0.18,
    "compliance_score":      0.68,
    "sentiment_risk":        0.40,
    "brand_risk":            0.30,
    "media_risk":            0.35,
    "stakeholder_risk":      0.42,
    "customer_churn_risk":   0.35,
    "market_opportunity":    0.60,
    "investment_amount":     20000000,
    "payback_period_years":  3,
}

CALIBRATION_CASES = {
    "risky": {
        "label": "CLEARLY RISKY",
        "expected_positions": {"REJECT", "CONDITIONAL_APPROVE"},
        "expected_confidence_band": "HIGH",
        "data": {
            "decision_topic":        "Should we give unsecured loans to unemployed borrowers with credit score below 300 during a recession?",
            "debt_to_income_ratio":  0.95,
            "credit_score":          270,
            "loan_amount":           300000,
            "default_probability":   0.85,
            "market_growth_rate":    -0.05,
            "competitive_risk":      0.90,
            "liquidity_ratio_inv":   0.80,
            "cash_flow_risk":        0.90,
            "regulatory_violation_prob": 0.50,
            "policy_risk":           0.60,
            "legal_risk":            0.55,
            "compliance_score":      0.30,
            "sentiment_risk":        0.75,
            "brand_risk":            0.70,
            "media_risk":            0.80,
            "stakeholder_risk":      0.78,
            "customer_churn_risk":   0.80,
            "market_opportunity":    0.10,
            "investment_amount":     5000000,
            "payback_period_years":  5,
        },
    },
    "safe": {
        "label": "CLEARLY SAFE",
        "expected_positions": {"APPROVE"},
        "expected_confidence_band": "HIGH",
        "data": {
            "decision_topic":        "Should we offer secured home loans to salaried employees with credit score above 750 during GDP growth?",
            "debt_to_income_ratio":  0.15,
            "credit_score":          800,
            "loan_amount":           80000,
            "default_probability":   0.04,
            "market_growth_rate":    0.06,
            "competitive_risk":      0.15,
            "liquidity_ratio_inv":   0.10,
            "cash_flow_risk":        0.08,
            "regulatory_violation_prob": 0.02,
            "policy_risk":           0.05,
            "legal_risk":            0.03,
            "compliance_score":      0.95,
            "sentiment_risk":        0.05,
            "brand_risk":            0.03,
            "media_risk":            0.04,
            "stakeholder_risk":      0.06,
            "customer_churn_risk":   0.04,
            "market_opportunity":    0.90,
            "investment_amount":     50000000,
            "payback_period_years":  2,
        },
    },
    "ambiguous": {
        "label": "AMBIGUOUS",
        "expected_positions": {"CONDITIONAL_APPROVE"},
        "expected_confidence_band": "MEDIUM",
        "data": {
            "decision_topic":        "Should we expand into Tier-2 cities with moderate credit checks during stable economic conditions?",
            "debt_to_income_ratio":  0.40,
            "credit_score":          660,
            "loan_amount":           120000,
            "default_probability":   0.28,
            "market_growth_rate":    0.04,
            "competitive_risk":      0.45,
            "liquidity_ratio_inv":   0.38,
            "cash_flow_risk":        0.35,
            "regulatory_violation_prob": 0.12,
            "policy_risk":           0.15,
            "legal_risk":            0.12,
            "compliance_score":      0.75,
            "sentiment_risk":        0.20,
            "brand_risk":            0.18,
            "media_risk":            0.22,
            "stakeholder_risk":      0.25,
            "customer_churn_risk":   0.20,
            "market_opportunity":    0.55,
            "investment_amount":     15000000,
            "payback_period_years":  3,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BASELINE CONSISTENCY TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_agent_consistency(agents, iterations: int = 3):
    """
    Send the same case to every agent N times. Check:
      - Decision stability
      - Confidence variance < 0.15
      - Risk score stability
    """
    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 1 — BASELINE CONSISTENCY TEST")
    print(f"  Sending identical case to each agent {iterations}x")
    print(f"{'='*65}{RESET}")

    results = {}

    for agent in agents:
        title = ROLE_TITLE.get(agent.role, agent.role)
        decisions   = []
        confidences = []
        risk_levels = []

        print(f"\n  Testing {CYAN}{title}{RESET} ({agent.preferred_model}) × {iterations} runs...")

        for i in range(iterations):
            try:
                pred = agent.predict(CONSISTENCY_CASE.copy())
                decisions.append(pred.decision)
                confidences.append(pred.confidence)
                risk_levels.append("HIGH" if pred.risk_score > 0.6
                                   else "MEDIUM" if pred.risk_score > 0.35
                                   else "LOW")
                print(f"    Run {i+1}: {pred.decision:<22} conf={pred.confidence:.3f}  risk={pred.risk_score:.3f}")
            except Exception as e:
                print(f"    {RED}Run {i+1}: ERROR — {e}{RESET}")
                decisions.append("ERROR")
                confidences.append(0.0)
                risk_levels.append("ERROR")

        # Calculate scores
        valid_decisions = [d for d in decisions if d != "ERROR"]
        if valid_decisions:
            most_common = max(set(valid_decisions), key=valid_decisions.count)
            matching    = valid_decisions.count(most_common)
            consistency = (matching / iterations) * 100
        else:
            consistency = 0.0

        conf_values = [c for c in confidences if c > 0]
        conf_variance = (max(conf_values) - min(conf_values)) if len(conf_values) > 1 else 0.0

        risk_stable = len(set(risk_levels)) <= 2  # allow at most 2 distinct risk bands

        # Print analysis
        stable_flag = consistency >= 66
        conf_ok     = conf_variance < 0.15
        status      = (GREEN + "STABLE" if stable_flag else RED + "UNSTABLE") + RESET
        conf_status = (GREEN + "OK"     if conf_ok     else YELLOW + "VARIABLE") + RESET

        print(f"    Consistency : {consistency:.0f}%  → {status}")
        print(f"    Conf var    : {conf_variance:.3f}  → {conf_status}")
        print(f"    Risk stable : {'Yes' if risk_stable else 'No'}")

        results[agent.role] = {
            "title":        title,
            "consistency":  round(consistency, 1),
            "conf_variance": round(conf_variance, 3),
            "risk_stable":  risk_stable,
            "stable":       stable_flag,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CROSS-AGENT DIFFERENTIATION TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_agent_differentiation(agents):
    """
    Run all agents on the same case. Verify they produce at least 2 different
    positions (agents should NOT all agree).
    """
    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 2 — CROSS-AGENT CONFLICT / DIFFERENTIATION TEST")
    print(f"{'='*65}{RESET}")

    positions = {}
    for agent in agents:
        title = ROLE_TITLE.get(agent.role, agent.role)
        try:
            pred = agent.predict(CONSISTENCY_CASE.copy())
            positions[agent.role] = pred.decision
            print(f"  {title:<14} → {pred.decision}")
        except Exception as e:
            positions[agent.role] = "ERROR"
            print(f"  {title:<14} → {RED}ERROR: {e}{RESET}")

    unique = set(v for v in positions.values() if v != "ERROR")
    diversity = len(unique) / max(len(agents), 1)

    passed = diversity >= 0.4
    status = GREEN + "PASS" if passed else RED + "FAIL"
    print(f"\n  Unique positions  : {unique}")
    print(f"  Diversity score   : {diversity:.2f}  (target ≥ 0.40)")
    print(f"  Result            : {status}{RESET}")

    return {
        "unique_positions": list(unique),
        "diversity_score":  round(diversity, 2),
        "passed":           passed,
        "breakdown":        positions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: JSON RELIABILITY TEST
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = {"position", "risk_level", "confidence", "argument",
                   "key_points", "sources_used"}

def _ask_for_json(agent, data: dict) -> dict:
    """
    Ask the agent's LLM to respond in structured JSON with required fields.
    Returns the parsed dict or raises an exception.
    """
    prompt = f"""
You are the {ROLE_TITLE.get(agent.role, agent.role)} in an enterprise risk council.

Case: {data.get('decision_topic', 'Financial decision case')}
DTI: {data.get('debt_to_income_ratio', 'N/A')}
Credit Score: {data.get('credit_score', 'N/A')}
Default Probability: {data.get('default_probability', 'N/A')}

Respond ONLY with a valid JSON object containing exactly these fields:
{{
  "position": "APPROVE" | "REJECT" | "CONDITIONAL_APPROVE",
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "confidence": <float 0.0-1.0>,
  "argument": "<2-3 sentence rationale>",
  "key_points": ["<point1>", "<point2>"],
  "sources_used": ["<metric1>", "<metric2>"]
}}

Output ONLY the JSON. No preamble, no markdown fences.
"""
    raw = agent.llm.generate(
        prompt=prompt,
        system_prompt=agent.get_system_prompt(),
        model=agent.preferred_model,
        max_tokens=400,
        temperature=0.3,
    )

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])

    # Find first JSON object in the response
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(text[start:end])


def test_json_reliability(agents, iterations: int = 5):
    """
    Run each agent N times asking for structured JSON.
    Score = valid JSON with all required fields / total attempts.
    """
    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 3 — JSON RELIABILITY TEST")
    print(f"  Each agent invoked {iterations}x for structured JSON output")
    print(f"{'='*65}{RESET}")

    results = {}

    for agent in agents:
        title = ROLE_TITLE.get(agent.role, agent.role)
        valid_json   = 0
        valid_fields = 0

        print(f"\n  Testing {CYAN}{title}{RESET}...")

        for i in range(iterations):
            try:
                parsed = _ask_for_json(agent, CONSISTENCY_CASE.copy())
                valid_json += 1
                missing = REQUIRED_FIELDS - set(parsed.keys())
                if not missing:
                    valid_fields += 1
                    print(f"    Run {i+1}: {GREEN}VALID JSON + all fields present{RESET}")
                else:
                    print(f"    Run {i+1}: {YELLOW}VALID JSON but missing: {missing}{RESET}")
            except json.JSONDecodeError as e:
                print(f"    Run {i+1}: {RED}BAD JSON — {e}{RESET}")
            except Exception as e:
                print(f"    Run {i+1}: {RED}ERROR — {e}{RESET}")

        reliability = (valid_fields / iterations) * 100
        ok = reliability >= 80
        status = (GREEN + "RELIABLE" if ok else RED + "UNRELIABLE") + RESET
        print(f"    Score: {reliability:.0f}%  ({valid_fields}/{iterations} fully valid)  → {status}")

        results[agent.role] = {
            "title":       title,
            "reliability": round(reliability, 1),
            "reliable":    ok,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CONFIDENCE CALIBRATION TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_confidence_calibration(agents):
    """
    Run 3 predefined cases per agent:
      - Clearly risky   → expect REJECT / CONDITIONAL_APPROVE + HIGH confidence
      - Clearly safe    → expect APPROVE + HIGH confidence
      - Ambiguous       → expect CONDITIONAL_APPROVE + MEDIUM confidence
    """
    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 4 — CONFIDENCE CALIBRATION TEST")
    print(f"{'='*65}{RESET}")

    results = {}

    for agent in agents:
        title = ROLE_TITLE.get(agent.role, agent.role)
        correct = 0
        total   = len(CALIBRATION_CASES)
        detail  = {}

        print(f"\n  [{CYAN}{title}{RESET}]")

        for case_key, case in CALIBRATION_CASES.items():
            try:
                pred = agent.predict(case["data"].copy())

                pos_match  = pred.decision in case["expected_positions"]
                high_conf  = pred.confidence > 0.75
                medium_conf = 0.40 <= pred.confidence <= 0.75

                if case["expected_confidence_band"] == "HIGH":
                    conf_match = high_conf
                else:  # MEDIUM
                    conf_match = medium_conf

                passed = pos_match and conf_match
                if passed:
                    correct += 1

                tag = (GREEN + "OK" if passed else
                       YELLOW + "PARTIAL" if pos_match or conf_match else
                       RED + "FAIL") + RESET

                print(f"    {case['label']:<18}  decision={pred.decision:<22} "
                      f"conf={pred.confidence:.3f}  {tag}")

                detail[case_key] = {
                    "decision":  pred.decision,
                    "confidence": pred.confidence,
                    "passed":    passed,
                }

            except Exception as e:
                print(f"    {case['label']:<18}  {RED}ERROR: {e}{RESET}")
                detail[case_key] = {"passed": False, "error": str(e)}

        cal_score = (correct / total) * 100
        print(f"    Calibration score: {cal_score:.0f}% ({correct}/{total})")

        results[agent.role] = {
            "title":        title,
            "calibration":  round(cal_score, 1),
            "detail":       detail,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RESPONSE TIME BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def test_response_time(agents, iterations: int = 3):
    """
    Time each model N times and take the average.
    Compare against threshold per model family.
    """
    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 5 — RESPONSE TIME BENCHMARK")
    print(f"  Each model timed {iterations} times")
    print(f"{'='*65}{RESET}")

    results = {}

    for agent in agents:
        title = ROLE_TITLE.get(agent.role, agent.role)
        model = agent.preferred_model
        times = []

        print(f"\n  {CYAN}{title}{RESET} ({model})")

        for i in range(iterations):
            try:
                t0 = time.perf_counter()
                agent.predict(CONSISTENCY_CASE.copy())
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                print(f"    Run {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: {RED}ERROR — {e}{RESET}")

        if times:
            avg = statistics.mean(times)
            threshold = MODEL_TIMEOUT_THRESHOLDS.get(model, 45)
            passed = avg <= threshold
            status = (GREEN + "PASS" if passed else RED + "FAIL") + RESET
            print(f"    Avg: {avg:.2f}s  (threshold: {threshold}s)  → {status}")
        else:
            avg = None
            passed = False
            print(f"    {RED}No successful runs{RESET}")

        results[agent.role] = {
            "title":     title,
            "model":     model,
            "avg_time":  round(avg, 2) if avg else None,
            "passed":    passed,
        }

    # Speed ranking
    ranked = sorted(
        [(r["title"], r["avg_time"]) for r in results.values() if r["avg_time"]],
        key=lambda x: x[1]
    )
    if ranked:
        print(f"\n  {CYAN}Speed ranking (fastest → slowest):{RESET}")
        for rank, (title, avg) in enumerate(ranked, 1):
            print(f"    {rank}. {title:<14} {avg:.2f}s")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: OUTPUT VARIANCE TEST (determinism check)
# ─────────────────────────────────────────────────────────────────────────────

# Deliberately ambiguous case — council should NOT always agree
AMBIGUOUS_VARIANCE_CASE = {
    "decision_topic":        "Should we moderately expand with mixed credit checks in a stable economy with neutral competition?",
    "debt_to_income_ratio":  0.40,
    "credit_score":          640,
    "loan_amount":           100000,
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


def test_output_variance(agents, iterations: int = 3):
    """
    Run the full council (all agents) on an ambiguous case N times.
    Verify that risk score and confidence vary across runs (not deterministic).

    Checks:
      - risk_score variance > 0.02  across runs
      - confidence variance > 0.02  across runs
      - verdict is NOT identical on all runs (for an ambiguous motion)
    """
    from council.voting.weighted_aggregator import WeightedAggregator

    print(f"\n{CYAN}{'='*65}")
    print("  SECTION 6 — OUTPUT VARIANCE TEST")
    print(f"  Council run {iterations}x on the same ambiguous case")
    print(f"  Checking: risk variance, confidence variance, verdict variety")
    print(f"{'='*65}{RESET}")

    agg      = WeightedAggregator()
    verdicts = []
    risks    = []
    confs    = []

    for i in range(iterations):
        print(f"\n  Run {i+1}...")
        predictions = []
        for agent in agents:
            try:
                pred = agent.predict(AMBIGUOUS_VARIANCE_CASE.copy())
                predictions.append(pred)
                title = ROLE_TITLE.get(agent.role, agent.role)
                print(f"    {title:<14} decision={pred.decision:<22} "
                      f"risk={pred.risk_score:.3f}  conf={pred.confidence:.3f}")
            except Exception as e:
                title = ROLE_TITLE.get(agent.role, agent.role)
                print(f"    {title:<14} {RED}ERROR: {e}{RESET}")

        if predictions:
            try:
                result = agg.aggregate(predictions)
                verdicts.append(result["final_decision"])
                risks.append(result["aggregate_risk_score"])
                confs.append(result["ensemble_confidence"])
                print(f"    --> verdict={result['final_decision']}  "
                      f"risk={result['aggregate_risk_score']}  "
                      f"conf={result['ensemble_confidence']}")
            except Exception as e:
                print(f"    {RED}Aggregation error: {e}{RESET}")

    print(f"\n  Results across {iterations} runs:")
    print(f"  Verdicts  : {verdicts}")
    print(f"  Risks     : {risks}")
    print(f"  Confs     : {confs}")

    # Evaluate variance
    results = {}

    if len(risks) >= 2:
        risk_var = max(risks) - min(risks)
        conf_var = max(confs) - min(confs)
        risk_ok  = risk_var > 0.02
        conf_ok  = conf_var > 0.02

        r_status = (GREEN + "PASS" if risk_ok  else RED + "FAIL") + RESET
        c_status = (GREEN + "PASS" if conf_ok  else RED + "FAIL") + RESET
        print(f"\n  Risk variance  : {risk_var:.4f}  (need > 0.02)  {r_status}")
        print(f"  Conf variance  : {conf_var:.4f}  (need > 0.02)  {c_status}")

        unique_verdicts = set(verdicts)
        if len(unique_verdicts) > 1:
            v_status = GREEN + "PASS — verdicts varied" + RESET
        else:
            v_status = YELLOW + f"NOTE — all {iterations} runs returned {list(unique_verdicts)[0] if unique_verdicts else '?'}" + RESET
        print(f"  Verdict variety: {unique_verdicts}  {v_status}")

        results = {
            "risk_variance":    round(risk_var, 4),
            "conf_variance":    round(conf_var, 4),
            "risk_varied":      risk_ok,
            "conf_varied":      conf_ok,
            "unique_verdicts":  list(unique_verdicts),
            "verdict_varied":   len(unique_verdicts) > 1,
        }
    else:
        print(f"  {RED}Not enough successful runs to compute variance{RESET}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FULL ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def run_full_accuracy_report(filter_agent: str = None, quick: bool = False,
                              variance_only: bool = False):
    """
    Runs all test sections, prints a summary table, and saves
    accuracy_report.json in the project root.
    """
    iters_main  = 1 if quick else 3
    iters_json  = 1 if quick else 5
    iters_time  = 1 if quick else 3
    iters_var   = 1 if quick else 3

    mode_label = f"{YELLOW}QUICK MODE{RESET}" if quick else f"{GREEN}FULL MODE{RESET}"
    print(f"\n{CYAN}{'='*65}")
    print("  AI COUNCIL — ACCURACY TEST SUITE")
    print(f"  Mode: {mode_label}")
    if filter_agent:
        print(f"  Agent filter: {filter_agent.upper()}")
    print(f"{'='*65}{RESET}")

    agents = load_agents(AGENT_KEY_MAP.get(filter_agent) if filter_agent else None)
    if not agents:
        print(f"{RED}No agents found. Check --agent flag value.{RESET}")
        return

    if variance_only:
        variance_result = test_output_variance(agents, iterations=iters_var)
        return {"variance": variance_result}

    # Run all sections
    consistency_results    = test_agent_consistency(agents, iterations=iters_main)
    differentiation_result = test_agent_differentiation(agents)
    json_results           = test_json_reliability(agents, iterations=iters_json)
    calibration_results    = test_confidence_calibration(agents)
    time_results           = test_response_time(agents, iterations=iters_time)
    variance_result        = test_output_variance(agents, iterations=iters_var)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print(f"\n{CYAN}{'='*65}")
    print("  FINAL SUMMARY TABLE")
    print(f"{'='*65}{RESET}")

    header = f"  {'Agent':<14} {'Consist.':>9} {'JSON Rel.':>10} {'Calibrat.':>10} {'Avg Time':>10}"
    sep    = "  " + "-" * 60
    print(header)
    print(sep)

    all_roles = list({a.role for a in agents})
    for role in all_roles:
        title   = ROLE_TITLE.get(role, role)
        cons    = consistency_results.get(role, {}).get("consistency", "N/A")
        jrel    = json_results.get(role, {}).get("reliability", "N/A")
        cal     = calibration_results.get(role, {}).get("calibration", "N/A")
        avg_t   = time_results.get(role, {}).get("avg_time")
        time_s  = f"{avg_t:.1f}s" if avg_t is not None else "N/A"

        cons_c  = GREEN if isinstance(cons, float) and cons >= 66 else RED if isinstance(cons, float) else ""
        jrel_c  = GREEN if isinstance(jrel, float) and jrel >= 80 else RED if isinstance(jrel, float) else ""
        cal_c   = GREEN if isinstance(cal,  float) and cal  >= 66 else RED if isinstance(cal,  float) else ""

        cons_s  = f"{cons_c}{cons}%{RESET}" if isinstance(cons, float) else str(cons)
        jrel_s  = f"{jrel_c}{jrel}%{RESET}" if isinstance(jrel, float) else str(jrel)
        cal_s   = f"{cal_c}{cal}%{RESET}"   if isinstance(cal,  float) else str(cal)

        print(f"  {title:<14} {cons_s:>16} {jrel_s:>17} {cal_s:>17} {time_s:>10}")

    print(sep)
    print(f"  Diversity score: {differentiation_result['diversity_score']:.2f}  "
          f"{'(' + GREEN + 'PASS' + RESET + ')' if differentiation_result['passed'] else '(' + RED + 'FAIL' + RESET + ')'}")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "timestamp":        datetime.datetime.utcnow().isoformat() + "Z",
        "quick_mode":       quick,
        "agent_filter":     filter_agent,
        "consistency":      consistency_results,
        "differentiation":  differentiation_result,
        "json_reliability": json_results,
        "calibration":      calibration_results,
        "response_time":    time_results,
        "variance":         variance_result,
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "accuracy_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  {GREEN}Full report saved: {out_path}{RESET}\n")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Council Accuracy & Reliability Test Suite"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=list(AGENT_KEY_MAP.keys()),
        help="Test a single agent only (ceo | cfo | marketing | pr | legal)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 1 iteration per test for fast spot-checking",
    )
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Run only the output variance test (Section 6) — fastest determinism check",
    )
    args = parser.parse_args()

    run_full_accuracy_report(
        filter_agent=args.agent,
        quick=args.quick,
        variance_only=args.variance,
    )
