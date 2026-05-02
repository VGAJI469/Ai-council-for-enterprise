"""
scratch/test_ceo_mutation.py
Quick smoke test for the CEO oversight + mutation pipeline.
No Ollama required — simulates sessions directly.
"""
import sys
sys.path.insert(0, 'e:/adaptive-ai-council')

from council.oversight.ceo_oversight import CEOOversightBoard
from council.oversight.ceo_performance import CEOPerformanceTracker
from evolution.mutation.ceo_mutator import CEOMutator

# ── Simulate a parent CEO config (mirrors agents.yaml) ────────────────────────
parent_config = {
    "agent_id": "ceo_gen1_initial",
    "name": "CEO Agent",
    "focus": "strategic_growth",
    "risk_threshold": 0.55,
    "weights": {
        "growth_potential":    0.40,
        "market_position":     0.30,
        "financial_stability": 0.20,
        "risk_exposure":       0.10,
    },
    "initial_credibility": 1.0,
    "generation": 1,
}


class FakePred:
    """Minimal AgentPrediction stand-in."""
    def __init__(self, decision, confidence=0.70,
                 reasoning="strategic risk compliance capital market shareholder"):
        self.decision   = decision
        self.confidence = confidence
        self.reasoning  = reasoning
        self.agent_role = "strategic_growth"


# ── Simulate 5 sessions: 4 LENIENT overrides, 1 aligned ──────────────────────
sessions = [
    ("sess001", 0.60, "REJECT",  "APPROVE"),   # lenient
    ("sess002", 0.62, "REJECT",  "APPROVE"),   # lenient
    ("sess003", 0.58, "REJECT",  "APPROVE"),   # lenient
    ("sess004", 0.65, "APPROVE", "APPROVE"),   # aligned
    ("sess005", 0.57, "REJECT",  "APPROVE"),   # lenient again
]

board   = CEOOversightBoard()
tracker = CEOPerformanceTracker()

print("=" * 65)
print("  CEO MUTATION SMOKE TEST")
print("=" * 65)
print()
print("  Simulating sessions …")

for sid, risk, council, ceo_dec in sessions:
    pred = FakePred(ceo_dec)
    rec  = board.record(sid, risk, council, pred)
    tracker.update(pred, council, risk)
    tag  = f"({rec.override_direction})" if rec.override_detected else "(aligned)"
    print(f"    {sid} | council={council:<8} ceo={ceo_dec:<8} override={str(rec.override_detected):<6} {tag}")

# ── Pattern summary ───────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  OVERRIDE PATTERN SUMMARY")
print("─" * 65)
summary = board.get_pattern_summary()
for k, v in summary.items():
    print(f"    {k:<22}: {v}")

streak   = board.count_consecutive_override_failures()
sup_s    = board.get_supervision_score()
perf_s   = tracker.get_score()
print(f"    {'consecutive_streak':<22}: {streak}")
print(f"    {'supervision_score':<22}: {sup_s}")
print(f"    {'performance_score':<22}: {perf_s}")

# ── Mutation trigger check ────────────────────────────────────────────────────
SUPERVISION_THRESHOLD = 3
PERFORMANCE_FLOOR     = 0.35

will_mutate = streak >= SUPERVISION_THRESHOLD or perf_s < PERFORMANCE_FLOOR
print()
print("─" * 65)
print("  MUTATION TRIGGER CHECK")
print("─" * 65)
print(f"    streak >= threshold ({streak} >= {SUPERVISION_THRESHOLD}): {streak >= SUPERVISION_THRESHOLD}")
print(f"    perf < floor  ({perf_s:.3f} < {PERFORMANCE_FLOOR}):       {perf_s < PERFORMANCE_FLOOR}")
print(f"    WILL MUTATE: {will_mutate}")

# ── Run CEOMutator ────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  CEOMutator.mutate() OUTPUT")
print("─" * 65)

mutator          = CEOMutator()
override_history = board.get_override_history()
print(f"    Override records fed to mutator: {len(override_history)}")

child_id, child_cfg = mutator.mutate(parent_config, override_history)

old_th = parent_config["risk_threshold"]
new_th = child_cfg["risk_threshold"]
print(f"    child_id        : {child_id}")
print(f"    generation      : {child_cfg['generation']}")
print(f"    mutation_reason : {child_cfg['mutation_reason']}")
print(f"    risk_threshold  : {old_th} → {new_th}  (delta={round(new_th - old_th, 4)})")
print(f"    weights         :")
for k, v in child_cfg["weights"].items():
    old_v = parent_config["weights"].get(k, "n/a")
    print(f"      {k:<24}: {old_v} → {v}")

log = mutator.get_mutation_log()
print(f"    mutation_log entries: {len(log)}")

# ── Generation-cap reset scenario ─────────────────────────────────────────────
print()
print("─" * 65)
print("  GENERATION CAP RESET SCENARIO  (gen=5 → should reset to gen=1)")
print("─" * 65)

gen5_config = {**parent_config, "generation": 5, "agent_id": "ceo_gen5_abc"}
cap_id, cap_cfg = mutator.mutate(gen5_config, override_history)
print(f"    child_id        : {cap_id}")
print(f"    generation      : {cap_cfg['generation']}")
print(f"    mutation_reason : {cap_cfg['mutation_reason']}")
print(f"    risk_threshold  : {cap_cfg['risk_threshold']}")

print()
print("=" * 65)
print("  ALL CHECKS COMPLETE")
print("=" * 65)
