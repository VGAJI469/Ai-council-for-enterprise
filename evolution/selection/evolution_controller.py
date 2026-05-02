"""
Evolution Controller - Agent lifecycle management.

Upgrade note (Item 8):
  FitnessScorer added — evaluates agents on four composite dimensions instead of
  relying solely on credibility (which is a lagging, single-dimension signal):

    1. Accuracy (40%):    average confidence of predictions — agents that make
       clear calls (far from 0.5) score higher than indecisive ones.
    2. Diversity (25%):   fraction of predictions that dissented from the council
       majority — homogeneous councils lose the benefit of multi-agent deliberation.
    3. Reasoning quality (20%): fraction of role-relevant keywords found in the
       agent's reasoning strings — proxy for substantive LLM output vs. stubs.
    4. Recovery speed (15%): credibility slope over last 5 cycles — agents that
       bounce back quickly after bad cycles are more valuable than persistent sinkers.

  The replacement parent is selected by FitnessScorer.rank() rather than by raw
  credibility, so a highly credible but intellectually homogeneous agent doesn't
  automatically dominate the gene pool.
"""

import logging
import statistics
from typing import List, Dict, Optional

from agents.base.base_agent import ROLE_KEYWORDS

logger = logging.getLogger(__name__)


class FitnessScorer:
    """
    Multi-dimensional fitness evaluator for council agents.

    Replaces the implicit "highest credibility = best parent" assumption in the
    original EvolutionController, which could entrench a single dominant agent
    style and reduce council diversity over generations.
    """

    # Composite weights
    W_ACCURACY  = 0.40
    W_DIVERSITY = 0.25
    W_QUALITY   = 0.20
    W_RECOVERY  = 0.15

    def score(self, agent, all_predictions_per_cycle: List[List] = None) -> float:
        """
        Compute a composite fitness score in [0, 1] for an agent.

        Args:
            agent:                   BaseAgent instance to evaluate.
            all_predictions_per_cycle: list of per-cycle prediction lists from
                                       the full council, used to compute diversity.
                                       If None, diversity defaults to neutral 0.5.
        """
        history = agent.prediction_history
        if not history:
            return 0.5   # no data yet — neutral score

        # 1. Accuracy: mean prediction confidence (distance from 0.5 × 2 → [0,1])
        accuracy = min(
            sum(abs(p.risk_score - 0.5) * 2.0 for p in history) / len(history),
            1.0
        )

        # 2. Diversity: fraction of predictions that differed from council majority
        diversity = self._compute_diversity(agent, all_predictions_per_cycle)

        # 3. Reasoning quality: role-keyword presence in LLM output
        quality = self._compute_quality(agent)

        # 4. Recovery speed: credibility slope over last 5 cycles
        recovery = self._compute_recovery(agent)

        composite = (
            self.W_ACCURACY  * accuracy
            + self.W_DIVERSITY * diversity
            + self.W_QUALITY   * quality
            + self.W_RECOVERY  * recovery
        )
        logger.debug(
            f"[FITNESS] {agent.role}: acc={accuracy:.3f} div={diversity:.3f} "
            f"qual={quality:.3f} rec={recovery:.3f} → {composite:.3f}"
        )
        return round(min(max(composite, 0.0), 1.0), 4)

    def _compute_diversity(self, agent,
                           all_predictions_per_cycle: Optional[List[List]]) -> float:
        """
        Compute what fraction of an agent's predictions dissented from the
        council majority. An agent that always agrees contributes no diversity
        value; one that thoughtfully disagrees ~30% of the time is valuable.

        If per-cycle data isn't available, returns 0.5 (neutral).
        """
        if not all_predictions_per_cycle:
            return 0.5

        dissent_count = 0
        total = 0
        for cycle_preds in all_predictions_per_cycle:
            if not cycle_preds:
                continue
            # Find the majority decision in this cycle
            from collections import Counter
            counts   = Counter(p.decision for p in cycle_preds)
            majority = counts.most_common(1)[0][0]
            # Find this agent's prediction in this cycle
            agent_pred = next(
                (p for p in cycle_preds if p.agent_role == agent.role), None
            )
            if agent_pred:
                total += 1
                if agent_pred.decision != majority:
                    dissent_count += 1

        if total == 0:
            return 0.5
        dissent_ratio = dissent_count / total
        # Optimal diversity is ~25-35% dissent; too much or too little is penalised
        # Map: 0%→0.2, 30%→1.0, 60%→0.5, 100%→0.0
        if dissent_ratio <= 0.30:
            return 0.2 + (dissent_ratio / 0.30) * 0.8
        else:
            return max(0.0, 1.0 - (dissent_ratio - 0.30) / 0.70)

    def _compute_quality(self, agent) -> float:
        """
        Score reasoning quality by checking how many role-relevant keywords
        appear in the agent's reasoning strings.

        Replaces character-count heuristic which falsely rewarded verbose-but-wrong
        responses and couldn't distinguish a timeout stub from real reasoning.
        """
        keywords = ROLE_KEYWORDS.get(agent.role, set())
        if not keywords or not agent.prediction_history:
            return 0.5

        recent = agent.prediction_history[-10:] if len(agent.prediction_history) >= 5 \
                 else agent.prediction_history
        scores = []
        for p in recent:
            reasoning_lower = p.reasoning.lower()
            found = sum(1 for kw in keywords if kw in reasoning_lower)
            scores.append(found / len(keywords))

        return round(sum(scores) / len(scores), 4) if scores else 0.5

    def _compute_recovery(self, agent) -> float:
        """
        Measure recovery speed as the credibility slope over the last 5 predictions.
        Positive slope = improving; negative = declining.
        Mapped to [0, 1] with 0 = declining, 0.5 = flat, 1 = recovering fast.
        """
        history = agent.prediction_history
        recent  = history[-5:] if len(history) >= 2 else history

        if len(recent) < 2:
            return 0.5

        # Use stored credibility on each prediction as the trend signal
        scores = [p.credibility for p in recent]
        # Slope via simple rise/run
        n     = len(scores)
        xs    = list(range(n))
        x_bar = sum(xs) / n
        y_bar = sum(scores) / n
        numer = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, scores))
        denom = sum((x - x_bar) ** 2 for x in xs)
        slope = (numer / denom) if denom != 0 else 0.0

        # Map slope to [0, 1]: slope of +0.05/cycle → 1.0; -0.05/cycle → 0.0
        mapped = 0.5 + (slope / 0.05) * 0.5
        return round(min(max(mapped, 0.0), 1.0), 4)

    def rank(self, agents: list,
             all_predictions_per_cycle: Optional[List[List]] = None) -> list:
        """Return agents sorted by composite fitness score (highest first)."""
        scored = [(a, self.score(a, all_predictions_per_cycle)) for a in agents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scored]


class EvolutionController:
    """
    Agent lifecycle manager. Replaces underperforming agents with evolved
    offspring from the fittest parent (selected by FitnessScorer).
    """

    def __init__(self, evolution_threshold: int = 3):
        self.evolution_threshold     = evolution_threshold
        self.underperformance_streak: Dict[str, int] = {}
        self.evolution_log:           List[Dict] = []
        self.fitness_scorer          = FitnessScorer()
        self._cycle_predictions:     List[List] = []   # history for diversity scoring

    def record_cycle_predictions(self, predictions: list) -> None:
        """
        Store per-cycle prediction snapshots so FitnessScorer can compute
        diversity scores across cycles.
        """
        self._cycle_predictions.append(list(predictions))
        # Keep last 20 cycles to bound memory usage
        if len(self._cycle_predictions) > 20:
            self._cycle_predictions = self._cycle_predictions[-20:]

    def evaluate_council(self, agents: list, credibility_manager,
                         mutator, agent_factory, configs: dict) -> list:
        """
        Evaluate council health and trigger evolution when warranted.

        Weakest agent is still the one with the lowest credibility (as before),
        but the replacement PARENT is now chosen by composite fitness score rather
        than raw credibility, ensuring the best all-round agent propagates.
        """
        weakest  = credibility_manager.get_lowest(agents)
        role     = weakest.role
        self.underperformance_streak[role] = self.underperformance_streak.get(role, 0) + 1

        if self.underperformance_streak[role] >= self.evolution_threshold:
            logger.warning(
                f"[EVOLUTION] Replacing {role} after {self.evolution_threshold} poor cycles"
            )

            # Select parent by composite fitness, not raw credibility
            remaining   = [a for a in agents if a.role != role]
            fitness_ranked = self.fitness_scorer.rank(remaining, self._cycle_predictions)
            best_parent = fitness_ranked[0] if fitness_ranked else \
                          credibility_manager.get_highest(agents)

            logger.info(
                f"[EVOLUTION] Parent selected: {best_parent.role} "
                f"(fitness={self.fitness_scorer.score(best_parent, self._cycle_predictions):.3f})"
            )

            agents      = remaining
            parent_cfg  = configs.get(best_parent.role, best_parent.config)
            new_id, new_cfg = mutator.mutate(best_parent, parent_cfg)
            new_agent   = agent_factory.create_from_config(new_id, new_cfg, best_parent.model)
            new_agent._recently_evolved = True   # Item 6: flag for memory suffix
            agents.append(new_agent)

            self.evolution_log.append({
                "removed":    role,
                "parent":     best_parent.role,
                "new_id":     new_id,
                "generation": new_cfg.get("generation", 2),
                "parent_fitness": self.fitness_scorer.score(
                    best_parent, self._cycle_predictions
                ),
            })
            self.underperformance_streak[role] = 0

        return agents

    def get_evolution_log(self) -> List[Dict]:
        return self.evolution_log
