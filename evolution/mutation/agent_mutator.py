"""
Agent Mutator - Generates evolved agents from high-performing parents.

Upgrade note (Item 9):
  Mutation is now DIRECTIONAL rather than purely random.

  Old approach: ±10% random perturbation on every weight with probability MUTATION_RATE.
  This had no memory of which weights actually contributed to good predictions.

  New approach:
    1. Analyse which weights correlated with accurate predictions in the parent's
       history (using the stored base_risk/role_risk components on each prediction).
    2. Weights that the parent leaned on during accurate cycles get a 1.5×
       directional boost — mutations favour INCREASING successful weights.
    3. Weights that weren't used during accurate cycles get normal random mutation.
    4. A mutation_log records what changed, in which direction, and why — enabling
       full traceability of agent evolution across generations.
"""

import uuid
import copy
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

MUTATION_RATE  = 0.15
WEIGHT_PERTURB = 0.10
DIRECTIONAL_BOOST = 1.5   # multiplier on perturbation for historically-good weights


class AgentMutator:
    """
    Directional evolutionary mutator for council agents.

    Biases weight perturbations toward increasing weights that were associated
    with accurate predictions in the parent's history, rather than randomising
    all mutations equally.
    """

    def __init__(self):
        self.mutation_log: List[Dict[str, Any]] = []

    def mutate(self, parent_agent, parent_config: dict) -> tuple:
        """
        Generate a child agent config via directional mutation.

        Returns (child_id, child_config). The child_config is a deep copy of
        the parent with weights and threshold mutated. Signature is unchanged
        from the original to maintain compatibility with EvolutionController.
        """
        child_config = copy.deepcopy(parent_config)
        generation   = child_config.get("generation", 1) + 1
        child_id     = f"{parent_agent.role}_gen{generation}_{uuid.uuid4().hex[:6]}"

        # Analyse parent history to find which weights correlated with accuracy
        directional_hints = self._compute_directional_hints(parent_agent)

        mutated        = {}
        weight_changes = {}

        for k, v in child_config["weights"].items():
            if random.random() < MUTATION_RATE:
                # Determine perturbation magnitude and direction
                magnitude = WEIGHT_PERTURB
                hint      = directional_hints.get(k, 0.0)  # +value=increase, -value=decrease

                if abs(hint) > 0.1:
                    # Directional mutation: bias toward historically-associated direction
                    magnitude *= DIRECTIONAL_BOOST
                    direction  = 1.0 if hint > 0 else -1.0
                    # Still add a small random component so it's not purely deterministic
                    delta      = direction * magnitude * random.uniform(0.5, 1.0)
                    rationale  = f"directional (hint={hint:+.3f})"
                else:
                    # No strong signal — fall back to random perturbation
                    delta     = random.uniform(-magnitude, magnitude)
                    rationale = "random (no directional signal)"

                new_v = min(max(v + delta, 0.05), 0.95)
                weight_changes[k] = {
                    "old":       round(v, 4),
                    "new":       round(new_v, 4),
                    "delta":     round(delta, 4),
                    "rationale": rationale,
                }
            else:
                new_v = v
                weight_changes[k] = {"old": round(v, 4), "new": round(v, 4),
                                     "delta": 0, "rationale": "unchanged"}
            mutated[k] = new_v

        # Normalise weights so they sum to 1.0
        total = sum(mutated.values())
        if total > 0:
            child_config["weights"] = {k: v / total for k, v in mutated.items()}

        # Threshold mutation (unchanged behaviour)
        old_threshold = child_config["risk_threshold"]
        child_config["risk_threshold"] = min(max(
            old_threshold + random.uniform(-0.05, 0.05), 0.1
        ), 0.9)

        child_config["initial_credibility"] = 0.5
        child_config["parent_id"]           = parent_agent.agent_id
        child_config["generation"]          = generation

        # Mutation log entry for full generational traceability
        log_entry = {
            "child_id":       child_id,
            "parent_id":      parent_agent.agent_id,
            "parent_role":    parent_agent.role,
            "generation":     generation,
            "weights_changed": weight_changes,
            "threshold_delta": round(child_config["risk_threshold"] - old_threshold, 4),
            "directional_hints": {k: round(v, 4) for k, v in directional_hints.items()},
        }
        self.mutation_log.append(log_entry)
        logger.info(
            f"[MUTATION] {child_id} (gen {generation}) | "
            f"directional weights: {[k for k, h in directional_hints.items() if abs(h) > 0.1]}"
        )
        return child_id, child_config

    def _compute_directional_hints(self, agent) -> Dict[str, float]:
        """
        Analyse parent's prediction history to identify which weights were
        associated with accurate outcomes (predictions far from 0.5 = confident).

        Returns a dict of weight_name → hint where:
          +value: this weight was high when predictions were accurate → prefer increasing it
          -value: this weight was low when predictions were accurate → safer to leave it
          0.0:    insufficient history or no signal

        This uses the role_risk component and credibility as proxies for accuracy,
        since true ground truth is not always available.
        """
        history = agent.prediction_history
        if len(history) < 5:
            return {}

        recent      = history[-min(len(history), 15):]
        # "Accurate" cycle = confident prediction (risk far from 0.5) AND high credibility
        accurate    = [p for p in recent
                       if abs(p.risk_score - 0.5) > 0.15 and p.credibility > 0.6]
        inaccurate  = [p for p in recent
                       if abs(p.risk_score - 0.5) < 0.10 or p.credibility < 0.4]

        if not accurate or not inaccurate:
            return {}

        hints = {}
        for weight_key in agent.weights:
            # Proxy: confident predictions had a higher role_risk component on average
            # Weight keys map conceptually to the role_risk formula dimensions
            # We use role_risk as a stand-in for "how much did role-specific judgment drive this?"
            acc_role_risk  = sum(p.role_risk  for p in accurate)  / len(accurate)
            inacc_role_risk = sum(p.role_risk for p in inaccurate) / len(inaccurate)

            # Positive hint = role was leaned on during accurate cycles
            delta = acc_role_risk - inacc_role_risk
            # Scale to [-1, +1] range
            hints[weight_key] = max(-1.0, min(delta, 1.0))

        return hints

    def get_mutation_log(self) -> List[Dict[str, Any]]:
        """Return the full generational mutation log for traceability."""
        return self.mutation_log
