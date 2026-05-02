"""
evolution/selection/ceo_supervision_controller.py
===================================================
CEO supervision controller — decides when to trigger CEO mutation.

The standard EvolutionController evaluates all five agents uniformly using
the credibility ladder.  The CEO is not replaced by that system because it
is the decision-maker, not a peer agent.  This controller adds a separate
accountability loop that triggers CEO mutation when:

  1. The consecutive override streak exceeds supervision_threshold, OR
  2. The 4-component CEO performance score drops below performance_floor.

When either condition is met the controller calls CEOMutator.mutate() and
rebuilds the CEO via AgentFactory.  The previous CEO agent is retired and
the new one carries the next generation number and the mutation reason.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CEOSupervisionController:
    """
    Supervises the CEO agent and replaces it when persistent override bias
    or poor performance is detected.

    Designed to be instantiable independently from the other components —
    all dependencies are injected at construction time but can also be
    updated after construction without breaking the interface.

    Parameters
    ----------
    ceo_agent              : Current CEO agent (BaseAgent subclass).
    oversight_board        : CEOOversightBoard instance tracking CEO decisions.
    performance_tracker    : CEOPerformanceTracker instance.
    ceo_mutator            : CEOMutator instance for generating replacement configs.
    agent_factory          : AgentFactory instance for building the replacement CEO.
    supervision_threshold  : Consecutive override streak that triggers mutation.
                             Defaults to 3 (matches agents.yaml ceo_supervision block).
    performance_floor      : Minimum acceptable composite performance score.
                             Defaults to 0.35.
    """

    def __init__(
        self,
        ceo_agent,
        oversight_board,
        performance_tracker,
        ceo_mutator,
        agent_factory,
        supervision_threshold: int   = 3,
        performance_floor:     float = 0.35,
    ) -> None:
        self.ceo_agent              = ceo_agent
        self.oversight_board        = oversight_board
        self.performance_tracker    = performance_tracker
        self.ceo_mutator            = ceo_mutator
        self.agent_factory          = agent_factory
        self.supervision_threshold  = supervision_threshold
        self.performance_floor      = performance_floor

        self._supervision_log: List[Dict[str, Any]] = []

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate_after_session(self):
        """
        Evaluate the CEO after every debate session and trigger mutation if
        either trigger condition is met.

        Trigger conditions (checked in this order):
          1. consecutive override streak >= supervision_threshold
          2. performance_tracker.get_score() < performance_floor

        When triggered:
          - CEOMutator.mutate() is called with the CEO's current config and
            the full override history from the oversight board.
          - AgentFactory.create_from_config() builds the replacement CEO.
          - The replacement agent is marked as recently evolved.
          - A supervision log entry is created with full diagnostic context.
          - self.ceo_agent is updated to the new CEO.

        Returns
        -------
        The current CEO agent (either the existing one or the new replacement).
        Each caller receives the agent it should use going forward.
        """
        streak     = self.oversight_board.count_consecutive_override_failures()
        perf_score = self.performance_tracker.get_score()
        sup_score  = self.oversight_board.get_supervision_score()

        trigger_reason: Optional[str] = None

        if streak >= self.supervision_threshold:
            trigger_reason = (
                f"Consecutive override streak ({streak}) reached threshold "
                f"({self.supervision_threshold})"
            )
        elif perf_score < self.performance_floor:
            trigger_reason = (
                f"Performance score ({perf_score:.3f}) dropped below floor "
                f"({self.performance_floor})"
            )

        if trigger_reason is None:
            logger.debug(
                "[CEO_SUPERVISION] No mutation triggered | streak=%d | perf=%.3f | sup=%.3f",
                streak, perf_score, sup_score,
            )
            return self.ceo_agent

        # ── Trigger mutation ──────────────────────────────────────────────
        logger.warning(
            "[CEO_SUPERVISION] Mutation triggered | reason=%s | streak=%d | perf=%.3f",
            trigger_reason, streak, perf_score,
        )

        old_id  = self.ceo_agent.agent_id
        parent_config    = self.ceo_agent.config
        override_history = self.oversight_board.get_override_history()

        new_id, new_config = self.ceo_mutator.mutate(parent_config, override_history)

        new_agent = self.agent_factory.create_from_config(
            new_id, new_config, model=None
        )
        new_agent._recently_evolved = True

        new_gen = new_config.get("generation", "?")
        log_entry = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "old_agent_id":      old_id,
            "new_agent_id":      new_id,
            "trigger_reason":    trigger_reason,
            "override_streak":   streak,
            "performance_score": round(perf_score, 4),
            "supervision_score": round(sup_score, 4),
            "generation":        new_gen,
            "mutation_reason":   new_config.get("mutation_reason", ""),
        }
        self._supervision_log.append(log_entry)

        logger.info(
            "[CEO_SUPERVISION] CEO replaced | %s → %s | gen=%s | reason=%s",
            old_id, new_id, new_gen, new_config.get("mutation_reason", ""),
        )

        self.ceo_agent = new_agent
        return self.ceo_agent

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_supervision_log(self) -> List[Dict[str, Any]]:
        """
        Return all CEO replacement events recorded by this controller.

        Each entry includes: timestamp, old_agent_id, new_agent_id,
        trigger_reason, override_streak, performance_score,
        supervision_score, generation, and mutation_reason.
        """
        return list(self._supervision_log)

    def get_status(self) -> Dict[str, Any]:
        """
        Return a current-state snapshot of all CEO accountability signals.

        Keys
        ----
        ceo_generation     : Generation number from the CEO's config.
        ceo_agent_id       : Current CEO agent ID.
        supervision_score  : Oversight board's health score (0–1).
        override_streak    : Current consecutive override streak.
        performance_score  : Current 4-component performance score (0–1).
        total_mutations    : How many times the CEO has been replaced this run.
        """
        gen = self.ceo_agent.config.get("generation", 1)
        return {
            "ceo_generation":    gen,
            "ceo_agent_id":      self.ceo_agent.agent_id,
            "supervision_score": self.oversight_board.get_supervision_score(),
            "override_streak":   self.oversight_board.count_consecutive_override_failures(),
            "performance_score": self.performance_tracker.get_score(),
            "total_mutations":   len(self._supervision_log),
        }
