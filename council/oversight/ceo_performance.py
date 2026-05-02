"""
council/oversight/ceo_performance.py
=====================================
CEO-specific 4-component performance tracker.

The standard credibility formula (alpha * C + beta * P + gamma * A - delta * E)
applies the same formula to all five agents without distinguishing the CEO's
unique role as final decision-maker.  This tracker measures the CEO on four
dimensions that specifically reflect executive accountability:

  1. Prediction accuracy   — did the CEO call it correctly vs ground truth?
  2. Override justification quality — when the CEO broke from the council,
     was the reasoning substantive?
  3. Council alignment     — how often did the CEO align with the majority?
  4. Outcome tracking      — rolling correctness rate over known outcomes.

All four components are scored 0–1 and combined using the weights defined
in the CEOPerformanceTracker class constants.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Keywords expected in high-quality CEO override reasoning
CEO_REASONING_KEYWORDS = frozenset([
    "risk", "compliance", "strategic", "litigation", "capital",
    "regulatory", "shareholder", "market", "legal", "financial",
])

# Component weights — must sum to 1.0
W_PREDICTION_ACCURACY            = 0.35
W_OVERRIDE_JUSTIFICATION_QUALITY = 0.25
W_COUNCIL_ALIGNMENT              = 0.20
W_OUTCOME_TRACKING               = 0.20

# How many past outcomes to include in the rolling outcome average
OUTCOME_WINDOW = 10


class CEOPerformanceTracker:
    """
    Maintains a per-session history of CEO performance across 4 dimensions
    and exposes a single composite score used by CEOSupervisionController
    to decide when mutation is warranted.

    Designed to be instantiable independently — no coupling to oversight board
    or supervision controller at construction time.

    Usage
    -----
    >>> tracker = CEOPerformanceTracker()
    >>> tracker.update(ceo_prediction, council_majority, aggregate_risk)
    >>> tracker.get_score()
    0.71
    """

    def __init__(self) -> None:
        self._history:  List[Dict[str, Any]] = []   # one dict per session
        self._outcomes: List[float] = []             # rolling correctness for outcome tracking

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        ceo_prediction,             # AgentPrediction — typed loosely to avoid circular import
        council_majority: str,
        aggregate_risk:   float,
        ground_truth:     Optional[str] = None,
    ) -> None:
        """
        Record one debate session and update all four performance components.

        Parameters
        ----------
        ceo_prediction   : AgentPrediction returned by the CEO agent.
        council_majority : Plurality decision among the non-CEO agents.
        aggregate_risk   : Weighted aggregate risk score from the council (0–1).
        ground_truth     : Actual outcome label if known
                           (APPROVE / CONDITIONAL_APPROVE / REJECT), else None.
        """
        ceo_decision = ceo_prediction.decision
        reasoning    = ceo_prediction.reasoning or ""
        override     = (ceo_decision != council_majority)

        pred_acc = self._score_prediction_accuracy(ceo_decision, ground_truth)
        ovr_qual = self._score_override_justification(override, reasoning)
        alignment = self._score_council_alignment(ceo_decision, council_majority)
        outcome_q = self._score_outcome_tracking(ceo_decision, ground_truth)

        composite = (
            W_PREDICTION_ACCURACY            * pred_acc
            + W_OVERRIDE_JUSTIFICATION_QUALITY * ovr_qual
            + W_COUNCIL_ALIGNMENT              * alignment
            + W_OUTCOME_TRACKING               * outcome_q
        )
        composite = round(max(0.0, min(composite, 1.0)), 4)

        entry: Dict[str, Any] = {
            "ceo_decision":          ceo_decision,
            "council_majority":      council_majority,
            "aggregate_risk":        round(aggregate_risk, 4),
            "ground_truth":          ground_truth,
            "override_detected":     override,
            "prediction_accuracy":   round(pred_acc, 4),
            "override_quality":      round(ovr_qual, 4),
            "council_alignment":     round(alignment, 4),
            "outcome_tracking":      round(outcome_q, 4),
            "composite_score":       composite,
        }
        self._history.append(entry)

        logger.info(
            "[CEO_PERF] session=%d | pred_acc=%.3f | ovr_qual=%.3f | "
            "alignment=%.3f | outcome=%.3f | composite=%.3f",
            len(self._history), pred_acc, ovr_qual, alignment, outcome_q, composite,
        )

    def get_score(self) -> float:
        """
        Return the rolling composite performance score over the last 10 sessions.
        Returns 0.5 (neutral) if no history is available yet.

        The score is the simple arithmetic mean of composite_score values across
        the most recent sessions — not a single-session snapshot — so that one
        outlier session cannot dominate the signal.
        """
        if not self._history:
            return 0.5
        window = self._history[-OUTCOME_WINDOW:]
        return round(sum(e["composite_score"] for e in window) / len(window), 4)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return the full per-session scoring history as a list of dicts.
        Each dict contains all four component scores plus the composite.
        """
        return list(self._history)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a high-level summary of CEO performance across all recorded
        sessions.

        Keys
        ----
        total_sessions        : Total sessions evaluated.
        composite_score       : Current rolling composite (last 10 sessions).
        avg_prediction_acc    : Mean prediction accuracy across all sessions.
        avg_override_quality  : Mean override justification quality.
        avg_council_alignment : Mean council alignment.
        avg_outcome_tracking  : Mean outcome tracking score.
        override_sessions     : Count of sessions where CEO diverged.
        """
        if not self._history:
            return {
                "total_sessions":        0,
                "composite_score":       0.5,
                "avg_prediction_acc":    0.0,
                "avg_override_quality":  0.0,
                "avg_council_alignment": 0.0,
                "avg_outcome_tracking":  0.0,
                "override_sessions":     0,
            }

        n = len(self._history)
        return {
            "total_sessions":        n,
            "composite_score":       self.get_score(),
            "avg_prediction_acc":    round(sum(e["prediction_accuracy"]  for e in self._history) / n, 4),
            "avg_override_quality":  round(sum(e["override_quality"]     for e in self._history) / n, 4),
            "avg_council_alignment": round(sum(e["council_alignment"]    for e in self._history) / n, 4),
            "avg_outcome_tracking":  round(sum(e["outcome_tracking"]     for e in self._history) / n, 4),
            "override_sessions":     sum(1 for e in self._history if e["override_detected"]),
        }

    # ── Scoring components ────────────────────────────────────────────────────

    @staticmethod
    def _score_prediction_accuracy(
        ceo_decision: str,
        ground_truth: Optional[str],
    ) -> float:
        """
        Score the CEO's decision against a known outcome.

          1.0  — decision matched ground truth exactly.
          0.5  — ground truth not yet known (deferred).
          0.0  — decision was wrong.

        Parameters
        ----------
        ceo_decision : CEO's decision string.
        ground_truth : Actual outcome, or None if not yet available.
        """
        if ground_truth is None:
            return 0.5
        return 1.0 if ceo_decision == ground_truth else 0.0

    @staticmethod
    def _score_override_justification(override: bool, reasoning: str) -> float:
        """
        Score the quality of the CEO's reasoning when it diverged from the
        council majority.

        When no override occurred the full score (1.0) is awarded — the CEO
        only needs to justify itself when it breaks from the council.

        When an override DID occur:
          score = min(len(reasoning) / 800, 1.0) * 0.6
                + keyword_density * 0.4

          keyword_density = fraction of CEO_REASONING_KEYWORDS present in the
          reasoning text (case-insensitive).

        Parameters
        ----------
        override  : True if the CEO diverged from the council majority.
        reasoning : Raw reasoning text from the CEO prediction.
        """
        if not override:
            return 1.0

        # Length component: 800+ chars → full 0.6 credit
        length_score = min(len(reasoning) / 800.0, 1.0) * 0.6

        # Keyword density component
        lower = reasoning.lower()
        found = sum(1 for kw in CEO_REASONING_KEYWORDS if kw in lower)
        keyword_density = found / len(CEO_REASONING_KEYWORDS)
        kw_score = keyword_density * 0.4

        return round(length_score + kw_score, 4)

    @staticmethod
    def _score_council_alignment(ceo_decision: str, council_majority: str) -> float:
        """
        Score how well the CEO's decision aligned with the council majority.

          1.0  — CEO matched the council majority exactly.
          0.5  — CEO chose CONDITIONAL_APPROVE when others were split
                 (partial alignment — acknowledges uncertainty).
          0.0  — Hard override: CEO directly contradicted the majority.

        Parameters
        ----------
        ceo_decision     : CEO's decision string.
        council_majority : Plurality decision of the non-CEO agents.
        """
        if ceo_decision == council_majority:
            return 1.0
        if ceo_decision == "CONDITIONAL_APPROVE":
            return 0.5
        return 0.0

    def _score_outcome_tracking(
        self,
        ceo_decision: str,
        ground_truth: Optional[str],
    ) -> float:
        """
        Update the rolling outcome window and return the current outcome
        tracking score.

        Only sessions with known ground truth contribute to the window.
        Sessions where ground_truth is None carry over the last known average
        (neutral 0.5 if no outcomes are known yet).

        Parameters
        ----------
        ceo_decision : CEO's decision string.
        ground_truth : Actual outcome, or None.
        """
        if ground_truth is not None:
            correct = 1.0 if ceo_decision == ground_truth else 0.0
            self._outcomes.append(correct)
            if len(self._outcomes) > OUTCOME_WINDOW:
                self._outcomes = self._outcomes[-OUTCOME_WINDOW:]

        if not self._outcomes:
            return 0.5
        return round(sum(self._outcomes) / len(self._outcomes), 4)
