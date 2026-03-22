"""Evolution Controller - Agent lifecycle management."""

import logging
from typing import List, Dict
logger = logging.getLogger(__name__)


class EvolutionController:
    def __init__(self, evolution_threshold: int = 3):
        self.evolution_threshold = evolution_threshold
        self.underperformance_streak: Dict[str, int] = {}
        self.evolution_log: List[Dict] = []

    def evaluate_council(self, agents, credibility_manager, mutator, agent_factory, configs) -> list:
        weakest = credibility_manager.get_lowest(agents)
        strongest = credibility_manager.get_highest(agents)

        role = weakest.role
        self.underperformance_streak[role] = self.underperformance_streak.get(role, 0) + 1

        if self.underperformance_streak[role] >= self.evolution_threshold:
            logger.warning(f"[EVOLUTION] Replacing {role} after {self.evolution_threshold} poor cycles")
            agents = [a for a in agents if a.role != role]
            parent_config = configs.get(strongest.role, strongest.config)
            new_id, new_config = mutator.mutate(strongest, parent_config)
            new_agent = agent_factory.create_from_config(new_id, new_config, strongest.model)
            agents.append(new_agent)
            self.evolution_log.append({
                "removed": role, "parent": strongest.role,
                "new_id": new_id, "generation": new_config.get("generation", 2)
            })
            self.underperformance_streak[role] = 0

        return agents

    def get_evolution_log(self): return self.evolution_log
