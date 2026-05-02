"""
evolution/mutation/ceo_mutator.py
==================================
Directional mutator specifically for the CEO agent.

Unlike AgentMutator (which handles the other four council roles and uses
weight-level directional hints from prediction history), CEOMutator operates
at the *configuration* level and drives mutation based on the override pattern
reported by CEOOversightBoard — not on individual prediction weights.

The logic is:
  - If the CEO has been systematically LENIENT (approved when it should have
    rejected), tighten the risk_threshold and reduce strategic weight.
  - If the CEO has been systematically AGGRESSIVE (rejected when it should have
    approved), loosen the risk_threshold and increase strategic weight.
  - Otherwise apply a small random recalibration so the CEO never stagnates
    through a fixed config across many sessions.

A generation cap of 5 prevents unbounded drift: when the CEO reaches
generation 5 the config resets to the baseline values read from agents.yaml
rather than compounding 5 rounds of directional correction.
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Generation ceiling — reset to baseline if reached
MAX_CEO_GENERATION = 5

# Baseline config used when generation cap is hit (mirrors agents.yaml values)
_CEO_BASELINE = {
    "name":                "CEO Agent",
    "focus":               "strategic_growth",
    "risk_threshold":      0.55,
    "weights": {
        "growth_potential":     0.40,
        "market_position":      0.30,
        "financial_stability":  0.20,
        "risk_exposure":        0.10,
    },
    "initial_credibility": 1.0,
    "generation":          1,
}

# Threshold + weight correction magnitudes
LENIENCY_RISK_CORRECTION    = 0.05   # decrease risk_threshold when too lenient
CONSERVATISM_RISK_CORRECTION = 0.05  # increase risk_threshold when too aggressive
STRATEGIC_WEIGHT_DRIFT       = 0.08
QUANTITATIVE_WEIGHT_DRIFT    = 0.08

# Minimum override count (in last 10 entries) to trigger directional correction
DIRECTIONAL_TRIGGER = 3

# Clamp bounds
RISK_THRESHOLD_MIN = 0.30
RISK_THRESHOLD_MAX = 0.75
WEIGHT_MIN         = 0.05
WEIGHT_MAX         = 0.85


class CEOMutator:
    """
    Directional mutator for the CEO agent configuration.

    Unlike the general AgentMutator which mutates weight arrays using
    prediction history, CEOMutator corrects the CEO's decision thresholds
    and strategic weighting based on observed override patterns reported
    by CEOOversightBoard.  This ensures that a biased CEO drifts toward
    calibrated behaviour rather than simply being replaced with a random
    variant.

    Usage
    -----
    >>> mutator = CEOMutator()
    >>> new_id, new_cfg = mutator.mutate(parent_config, override_history)
    >>> log = mutator.get_mutation_log()
    """

    def __init__(self) -> None:
        self._mutation_log: List[Dict[str, Any]] = []

    def mutate(
        self,
        parent_config:    Dict[str, Any],
        override_history: List,           # list of CEODecisionRecord
    ) -> tuple:
        """
        Produce a new CEO agent config via directional mutation.

        Mutation direction is determined by counting LENIENT and AGGRESSIVE
        overrides in the most recent 10 entries of override_history:

          lenient_overrides = LENIENT + LENIENT_RISK_MISMATCH in last 10
          aggressive_overrides = AGGRESSIVE in last 10

          if lenient_overrides >= 3:
              risk_threshold  -= 0.05   (make CEO stricter)
              strategic_weight -= 0.08
              quantitative_weight += 0.08
              reason = "Correcting systematic leniency bias"

          elif aggressive_overrides >= 3:
              risk_threshold  += 0.05   (make CEO more permissive)
              strategic_weight += 0.06
              quantitative_weight -= 0.06
              reason = "Correcting systematic conservatism bias"

          else:
              risk_threshold += uniform(-0.03, +0.03)
              reason = "Scheduled recalibration — no dominant pattern"

        After mutation all values are clamped and weights are renormalised
        to sum to 1.0.

        If the resulting generation would reach MAX_CEO_GENERATION (5), the
        config is reset to the baseline instead of continuing to drift.

        Parameters
        ----------
        parent_config    : Deep-copyable dict matching the agents.yaml agent block.
        override_history : List of CEODecisionRecord objects from the oversight board.

        Returns
        -------
        (child_id: str, child_config: dict)
        """
        child_config = copy.deepcopy(parent_config)
        current_gen  = child_config.get("generation", 1)
        parent_id    = child_config.get("agent_id", "ceo_unknown")

        # ── Generation cap: reset instead of continuing to drift ──────────
        if current_gen >= MAX_CEO_GENERATION:
            logger.warning(
                "[CEO_MUTATOR] Generation cap %d reached — resetting to baseline",
                MAX_CEO_GENERATION,
            )
            child_config     = copy.deepcopy(_CEO_BASELINE)
            child_config["generation"]          = 1
            child_config["parent_id"]           = parent_id
            child_config["mutation_reason"]     = "Generation cap reached — reset to baseline"
            child_config["mutation_timestamp"]  = datetime.now(timezone.utc).isoformat()
            child_id = f"ceo_gen1_reset_{uuid.uuid4().hex[:6]}"
            self._log_mutation(child_id, parent_id, 1, child_config, "Generation cap reset")
            return child_id, child_config

        # ── Analyse last 10 override records ─────────────────────────────
        recent         = override_history[-10:]
        lenient_count  = sum(
            1 for r in recent
            if getattr(r, "override_direction", None) in ("LENIENT", "LENIENT_RISK_MISMATCH")
        )
        aggressive_count = sum(
            1 for r in recent
            if getattr(r, "override_direction", None) == "AGGRESSIVE"
        )

        # ── Determine mutation direction ───────────────────────────────────
        old_threshold = child_config.get("risk_threshold", 0.55)
        weights       = child_config.get("weights", {})
        weight_keys   = list(weights.keys())

        # Identify "strategic" and "quantitative" proxy keys from the weight dict
        # strategic proxy  → growth_potential / market_position (first two keys)
        # quantitative proxy → financial_stability / risk_exposure (last two keys)
        strategic_keys    = weight_keys[:2]   if len(weight_keys) >= 2 else weight_keys
        quantitative_keys = weight_keys[2:]   if len(weight_keys) >= 3 else []

        if lenient_count >= DIRECTIONAL_TRIGGER:
            # CEO has been too permissive — tighten threshold, reduce strategic weight
            new_threshold    = old_threshold - LENIENCY_RISK_CORRECTION
            mutation_reason  = "Correcting systematic leniency bias"
            _shift_weights(weights, strategic_keys,    -STRATEGIC_WEIGHT_DRIFT)
            _shift_weights(weights, quantitative_keys, +QUANTITATIVE_WEIGHT_DRIFT)

        elif aggressive_count >= DIRECTIONAL_TRIGGER:
            # CEO has been too conservative — loosen threshold, increase strategic weight
            new_threshold    = old_threshold + CONSERVATISM_RISK_CORRECTION
            mutation_reason  = "Correcting systematic conservatism bias"
            _shift_weights(weights, strategic_keys,    +STRATEGIC_WEIGHT_DRIFT)
            _shift_weights(weights, quantitative_keys, -QUANTITATIVE_WEIGHT_DRIFT)

        else:
            # No dominant pattern — small random recalibration
            new_threshold   = old_threshold + random.uniform(-0.03, 0.03)
            mutation_reason = "Scheduled recalibration — no dominant pattern"

        # ── Apply constraints ─────────────────────────────────────────────
        new_threshold = round(max(RISK_THRESHOLD_MIN, min(new_threshold, RISK_THRESHOLD_MAX)), 4)
        for k in weights:
            weights[k] = round(max(WEIGHT_MIN, min(weights[k], WEIGHT_MAX)), 4)

        # Renormalise weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 6) for k, v in weights.items()}

        new_gen   = current_gen + 1
        child_id  = f"ceo_gen{new_gen}_{uuid.uuid4().hex[:6]}"

        child_config["risk_threshold"]      = new_threshold
        child_config["weights"]             = weights
        child_config["generation"]          = new_gen
        child_config["parent_id"]           = parent_id
        child_config["initial_credibility"] = 0.5
        child_config["mutation_reason"]     = mutation_reason
        child_config["mutation_timestamp"]  = datetime.now(timezone.utc).isoformat()

        logger.info(
            "[CEO_MUTATOR] %s | gen=%d | reason=%s | "
            "threshold: %.3f → %.3f | lenient=%d | aggressive=%d",
            child_id, new_gen, mutation_reason,
            old_threshold, new_threshold, lenient_count, aggressive_count,
        )

        self._log_mutation(child_id, parent_id, new_gen, child_config, mutation_reason)
        return child_id, child_config

    def get_mutation_log(self) -> List[Dict[str, Any]]:
        """
        Return the full history of all CEO mutations.

        Each entry records child_id, parent_id, generation, mutation_reason,
        mutation_timestamp, and the resulting config snapshot.
        """
        return list(self._mutation_log)

    # ── Private ───────────────────────────────────────────────────────────────

    def _log_mutation(
        self,
        child_id:       str,
        parent_id:      str,
        generation:     int,
        child_config:   Dict[str, Any],
        mutation_reason: str,
    ) -> None:
        """Append a structured entry to the internal mutation log."""
        self._mutation_log.append({
            "child_id":           child_id,
            "parent_id":          parent_id,
            "generation":         generation,
            "mutation_reason":    mutation_reason,
            "mutation_timestamp": child_config.get("mutation_timestamp", ""),
            "risk_threshold":     child_config.get("risk_threshold"),
            "weights":            copy.deepcopy(child_config.get("weights", {})),
        })


# ── Module-level helper ───────────────────────────────────────────────────────

def _shift_weights(
    weights:     Dict[str, float],
    target_keys: List[str],
    delta:       float,
) -> None:
    """
    Add *delta* to each weight in *target_keys* in-place.
    Clamping to [WEIGHT_MIN, WEIGHT_MAX] is applied after all shifts.
    Renormalisation happens in the caller after all groups have been shifted.
    """
    for k in target_keys:
        if k in weights:
            weights[k] = weights[k] + delta
