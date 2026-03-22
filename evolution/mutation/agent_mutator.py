"""Agent Mutator - Generates evolved agents from high-performing parents."""

import uuid, copy, random, logging
logger = logging.getLogger(__name__)

MUTATION_RATE = 0.15
WEIGHT_PERTURB = 0.1


class AgentMutator:
    def mutate(self, parent_agent, parent_config: dict):
        child_config = copy.deepcopy(parent_config)
        child_id = f"{parent_agent.role}_gen{child_config.get('generation',1)+1}_{uuid.uuid4().hex[:6]}"

        mutated = {}
        for k, v in child_config["weights"].items():
            delta = random.uniform(-WEIGHT_PERTURB, WEIGHT_PERTURB) if random.random() < MUTATION_RATE else 0
            mutated[k] = min(max(v + delta, 0.05), 0.95)
        total = sum(mutated.values())
        child_config["weights"] = {k: v/total for k, v in mutated.items()}

        child_config["risk_threshold"] = min(max(
            child_config["risk_threshold"] + random.uniform(-0.05, 0.05), 0.1), 0.9)
        child_config["initial_credibility"] = 0.5
        child_config["parent_id"] = parent_agent.agent_id
        child_config["generation"] = child_config.get("generation", 1) + 1

        logger.info(f"Evolved agent: {child_id} (gen {child_config['generation']})")
        return child_id, child_config
