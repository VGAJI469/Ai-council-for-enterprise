import uuid
import yaml
import logging
# Suppress all logging output
logging.disable(logging.CRITICAL)

from pipeline.ingestion.data_ingester import DataIngester
from pipeline.aggregation.feature_builder import FeatureBuilder
from agents.evolution.agent_factory import AgentFactory
from council.debate.council_session import CouncilSession
from council.voting.weighted_aggregator import WeightedAggregator

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    pipeline_cfg = load_yaml("config/pipeline.yaml")
    agent_cfg    = load_yaml("config/agents.yaml")
    model_cfg    = load_yaml("config/model.yaml")
    ollama_url = model_cfg.get("llm", {}).get("base_url", "http://localhost:11434")

    ingester   = DataIngester(pipeline_cfg["pipeline"]["ingestion"])
    builder    = FeatureBuilder()
    factory    = AgentFactory(ollama_url=ollama_url)

    datasets = ingester.ingest()
    features = builder.build(datasets)
    agents   = factory.build_council(agent_cfg)
    
    record = features[0]
    print(f"\nEvaluating Loan: {record.get('loan_id', 'Unknown')}")
    print("-" * 40)
    
    session = CouncilSession(uuid.uuid4().hex[:8], agents)
    predictions = session.run(record)
    
    aggregator = WeightedAggregator()
    result = aggregator.aggregate(predictions)
    
    print("\n" + "="*80)
    print("AI COUNCIL DEBATE TRANSCRIPT")
    print("="*80)
    for p in predictions:
        print(f"\n[{p.agent_role.upper()}] Decision: {p.decision} | Risk: {p.risk_score} | Confidence: {p.confidence}")
        print(f"Reasoning: {p.reasoning}")
    
    print("\n" + "="*80)
    print("FINAL COUNCIL VERDICT")
    print("="*80)
    print(f"Decision: {result['final_decision']}")
    print(f"Aggregate Risk Score: {result['aggregate_risk_score']}")
    print(f"Council Confidence: {result['ensemble_confidence']}")
    print("="*80)
