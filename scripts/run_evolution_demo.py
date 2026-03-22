"""
Evolution Demo - runs 15 cycles to trigger agent replacement.
The lowest-performing agent after 3 consecutive poor cycles
is replaced by a mutated version of the top performer.
"""

import sys
import yaml
import uuid
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from pipeline.aggregation.feature_builder import FeatureBuilder
from council.debate.council_session import CouncilSession
from council.voting.weighted_aggregator import WeightedAggregator
from council.credibility.credibility_manager import CredibilityManager
from evolution.mutation.agent_mutator import AgentMutator
from evolution.selection.evolution_controller import EvolutionController
from agents.evolution.agent_factory import AgentFactory

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

agent_cfg  = load_yaml("config/agents.yaml")
factory    = AgentFactory()
agents     = factory.build_council(agent_cfg)
cred_cfg   = agent_cfg["credibility"]

credibility_manager = CredibilityManager(
    alpha=cred_cfg["alpha"], beta=cred_cfg["beta"],
    gamma=cred_cfg["gamma"], delta=cred_cfg["delta"]
)
mutator    = AgentMutator()
evolution  = EvolutionController(evolution_threshold=3)
aggregator = WeightedAggregator()
builder    = FeatureBuilder()

record = {
    "loan_id":              "DEMO_EVOLUTION",
    "borrower_income":      65000,
    "debt_to_income_ratio": 0.42,
    "credit_score":         630,
    "loan_amount":          180000,
    "loan_term_months":     360,
    "employment_years":     3.5,
    "default_history":      1,
}
features = builder._engineer(record)

print("\n" + "="*60)
print("  EVOLUTION DEMO - 15 council cycles")
print("="*60)

for cycle in range(1, 16):
    session     = CouncilSession(uuid.uuid4().hex[:6], agents)
    predictions = session.run(features)
    result      = aggregator.aggregate(predictions)

    credibility_manager.update_all(agents, predictions, result["final_decision"])
    agents = evolution.evaluate_council(
        agents, credibility_manager, mutator, factory,
        {a.role: a.config for a in agents}
    )

    creds = {a.role: round(a.credibility, 3) for a in agents}
    print(f"Cycle {cycle:02d} | {result['final_decision']:<22} | Credibilities: {creds}")

elog = evolution.get_evolution_log()
print("\n" + "="*60)
if elog:
    print("  EVOLUTION EVENTS:")
    for e in elog:
        print(f"  Removed: {e['removed']} -> Replaced by child of {e['parent']} (gen {e['generation']})")
else:
    print("  No evolution events triggered in 15 cycles.")
print("="*60)
