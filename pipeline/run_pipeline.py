"""
Main Pipeline Runner — Full automation loop for the Enterprise Council.
Uses locally hosted LLMs via Ollama for agent reasoning.

Loop:
  1. Ingest financial data
  2. Build feature vectors
  3. Run council session (agent predictions + local LLM reasoning)
  4. Aggregate credibility-weighted decision
  5. Update agent credibility scores
  6. Evolve council if an agent underperforms
  7. Repeat
"""

import logging
import yaml
import uuid
import json
from datetime import datetime
from pathlib import Path

from pipeline.ingestion.data_ingester import DataIngester
from pipeline.aggregation.feature_builder import FeatureBuilder
from council.debate.council_session import CouncilSession
from council.voting.weighted_aggregator import WeightedAggregator
from council.credibility.credibility_manager import CredibilityManager
from evolution.mutation.agent_mutator import AgentMutator
from evolution.selection.evolution_controller import EvolutionController
from agents.evolution.agent_factory import AgentFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_result(cycle: int, result: dict):
    decision = result["final_decision"]
    risk = result["aggregate_risk_score"]
    conf = result["ensemble_confidence"]
    icons = {"APPROVE": "✓", "REJECT": "✗", "CONDITIONAL_APPROVE": "~"}
    icon = icons.get(decision, "?")
    logger.info(
        f"  {icon} Decision: {decision:<22} Risk: {risk:.4f}  Confidence: {conf:.4f}"
    )


def main(max_records: int = 5):
    logger.info("=" * 60)
    logger.info("  Adaptive AI Enterprise Council — Starting")
    logger.info("  LLM Backend: Ollama (local)")
    logger.info("=" * 60)

    pipeline_cfg = load_yaml("config/pipeline.yaml")
    agent_cfg    = load_yaml("config/agents.yaml")
    model_cfg    = load_yaml("config/model.yaml")

    ollama_url = model_cfg.get("llm", {}).get("base_url", "http://localhost:11434")

    ingester   = DataIngester(pipeline_cfg["pipeline"]["ingestion"])
    builder    = FeatureBuilder()
    aggregator = WeightedAggregator()
    factory    = AgentFactory(ollama_url=ollama_url)

    cred_cfg = agent_cfg["credibility"]
    credibility_manager = CredibilityManager(
        alpha=cred_cfg["alpha"],
        beta=cred_cfg["beta"],
        gamma=cred_cfg["gamma"],
        delta=cred_cfg["delta"],
    )
    mutator   = AgentMutator()
    evolution = EvolutionController(
        evolution_threshold=cred_cfg["evolution_threshold"]
    )

    # Ingest data and build features
    datasets = ingester.ingest()
    features = builder.build(datasets)
    agents   = factory.build_council(agent_cfg)

    logger.info(
        f"Council active: {[a.role for a in agents]}"
    )
    logger.info(
        f"Model assignments: {[f'{a.role}→{a.preferred_model}' for a in agents]}"
    )
    logger.info(f"Processing {min(max_records, len(features))} records...\n")

    all_results = []

    for i, record in enumerate(features[:max_records]):
        logger.info(f"--- Cycle {i+1} | Loan: {record.get('loan_id', f'RECORD_{i+1}')} ---")

        session = CouncilSession(uuid.uuid4().hex[:8], agents)
        predictions = session.run(record)
        result = aggregator.aggregate(predictions)
        print_result(i + 1, result)

        credibility_manager.update_all(
            agents, predictions, result["final_decision"]
        )

        agents = evolution.evaluate_council(
            agents, credibility_manager, mutator, factory,
            {a.role: a.config for a in agents}
        )

        all_results.append({
            "cycle":               i + 1,
            "loan_id":             record.get("loan_id", f"RECORD_{i+1}"),
            "final_decision":      result["final_decision"],
            "aggregate_risk_score":result["aggregate_risk_score"],
            "ensemble_confidence": result["ensemble_confidence"],
            "vote_breakdown":      result["vote_breakdown"],
            "evolution_log":       evolution.get_evolution_log(),
        })

    Path("outputs").mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = f"outputs/council_results_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Pipeline complete. Results saved: {out}")
    logger.info(f"{'=' * 60}")
    return all_results


if __name__ == "__main__":
    main()
