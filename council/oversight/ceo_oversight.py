"""
council/oversight/ceo_oversight.py
===================================
CEO oversight and accountability layer.

The boardroom debate currently has no mechanism to detect when the CEO
overrides the council majority, count how often it happens, or penalise
repeated patterns of bias.  This module introduces:

  - CEODecisionRecord  — immutable snapshot of one debate outcome
  - CEOOversightBoard  — accumulates records, detects overrides, derives a
    supervision score, and persists history to JSON for audit purposes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class CEODecisionRecord:
    """
    Immutable snapshot of the CEO's decision for a single debate session.

    Fields
    ------
    session_id            : Unique debate session identifier.
    timestamp             : UTC ISO-8601 string when the record was created.
    aggregate_risk        : Council-wide aggregate risk score (0–1).
    council_majority      : Decision held by the plurality of council agents
                            (APPROVE / CONDITIONAL_APPROVE / REJECT).
    ceo_decision          : The CEO's actual decision.
    ceo_confidence        : CEO prediction confidence score (0–1).
    ceo_reasoning_length  : Character count of the CEO's reasoning text.
    override_detected     : True if CEO diverged from council majority in a
                            material way or ignored a high-risk signal.
    override_direction    : One of "LENIENT", "AGGRESSIVE",
                            "LENIENT_RISK_MISMATCH", or None.
    outcome               : Ground-truth outcome label, filled later when known.
    """

    session_id:            str
    timestamp:             str
    aggregate_risk:        float
    council_majority:      str
    ceo_decision:          str
    ceo_confidence:        float
    ceo_reasoning_length:  int
    override_detected:     bool
    override_direction:    Optional[str]   # "LENIENT" | "AGGRESSIVE" | "LENIENT_RISK_MISMATCH" | None
    outcome:               Optional[str]   # filled post-hoc if ground truth is available


class CEOOversightBoard:
    """
    Tracks all CEO decisions across debate sessions and computes oversight
    signals that can trigger mutation through CEOSupervisionController.

    The board operates independently from the credibility formula — it adds
    a second accountability axis specifically for the CEO role, measuring
    directional bias and override frequency rather than generic credibility.

    Usage
    -----
    >>> board = CEOOversightBoard()
    >>> board.record(session_id, aggregate_risk, council_majority, ceo_prediction)
    >>> summary = board.get_pattern_summary()
    """

    def __init__(self) -> None:
        self._records: List[CEODecisionRecord] = []

    # ── Core recording ────────────────────────────────────────────────────────

    def record(
        self,
        session_id: str,
        aggregate_risk: float,
        council_majority: str,
        ceo_prediction,          # AgentPrediction — typed loosely to avoid circular import
    ) -> CEODecisionRecord:
        """
        Create a CEODecisionRecord for the given session and append it to the
        board history.

        Override detection logic:
          - LENIENT              : council said REJECT, CEO said APPROVE.
          - AGGRESSIVE           : council said APPROVE, CEO said REJECT.
          - LENIENT_RISK_MISMATCH: CEO approved despite aggregate_risk > 0.55.
          - None                 : no material divergence detected.

        Parameters
        ----------
        session_id       : Unique identifier for the debate session.
        aggregate_risk   : Weighted council risk score (0–1).
        council_majority : Plurality decision of the non-CEO agents.
        ceo_prediction   : AgentPrediction returned by the CEO agent.

        Returns
        -------
        The newly created CEODecisionRecord.
        """
        ceo_decision  = ceo_prediction.decision
        ceo_conf      = ceo_prediction.confidence
        ceo_reasoning = ceo_prediction.reasoning or ""

        override_detected, override_direction = self._detect_override(
            council_majority, ceo_decision, aggregate_risk
        )

        record = CEODecisionRecord(
            session_id           = session_id,
            timestamp            = datetime.now(timezone.utc).isoformat(),
            aggregate_risk       = round(aggregate_risk, 4),
            council_majority     = council_majority,
            ceo_decision         = ceo_decision,
            ceo_confidence       = round(ceo_conf, 4),
            ceo_reasoning_length = len(ceo_reasoning),
            override_detected    = override_detected,
            override_direction   = override_direction,
            outcome              = None,
        )
        self._records.append(record)

        if override_detected:
            logger.warning(
                "[CEO_OVERSIGHT] Override detected | session=%s | direction=%s | "
                "council=%s | ceo=%s | risk=%.3f",
                session_id, override_direction, council_majority,
                ceo_decision, aggregate_risk,
            )
        else:
            logger.info(
                "[CEO_OVERSIGHT] No override | session=%s | ceo=%s",
                session_id, ceo_decision,
            )

        return record

    # ── Retrieval helpers ─────────────────────────────────────────────────────

    def get_recent_records(self, n: int = 10) -> List[CEODecisionRecord]:
        """
        Return the last *n* decision records (most-recent last).

        Parameters
        ----------
        n : Maximum number of records to return. Defaults to 10.
        """
        return self._records[-n:]

    def get_override_history(self) -> List[CEODecisionRecord]:
        """
        Return all records where an override was detected (any direction).
        Useful for auditing systematic patterns of CEO dissent.
        """
        return [r for r in self._records if r.override_detected]

    # ── Pattern analysis ──────────────────────────────────────────────────────

    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Return an aggregated summary of CEO decision behaviour.

        Keys in the returned dict
        -------------------------
        total_decisions   : Total sessions recorded.
        total_overrides   : Sessions where an override was detected.
        lenient_count     : Count of LENIENT overrides.
        aggressive_count  : Count of AGGRESSIVE overrides.
        mismatch_count    : Count of LENIENT_RISK_MISMATCH overrides.
        override_rate     : Fraction of sessions with an override (0–1).
        dominant_pattern  : The most frequent override direction, or "NONE".
        """
        total        = len(self._records)
        overrides    = [r for r in self._records if r.override_detected]
        lenient      = sum(1 for r in overrides if r.override_direction == "LENIENT")
        aggressive   = sum(1 for r in overrides if r.override_direction == "AGGRESSIVE")
        mismatch     = sum(1 for r in overrides if r.override_direction == "LENIENT_RISK_MISMATCH")
        total_ovr    = len(overrides)
        rate         = round(total_ovr / total, 4) if total else 0.0

        if total_ovr == 0:
            dominant = "NONE"
        else:
            counts   = {"LENIENT": lenient, "AGGRESSIVE": aggressive,
                        "LENIENT_RISK_MISMATCH": mismatch}
            dominant = max(counts, key=counts.get)
            if counts[dominant] == 0:
                dominant = "NONE"

        return {
            "total_decisions":  total,
            "total_overrides":  total_ovr,
            "lenient_count":    lenient,
            "aggressive_count": aggressive,
            "mismatch_count":   mismatch,
            "override_rate":    rate,
            "dominant_pattern": dominant,
        }

    def count_consecutive_override_failures(self) -> int:
        """
        Count the current streak of consecutive sessions where an override was
        detected.  Resets to 0 when a non-override session breaks the streak.

        A streak of ≥ supervision_threshold signals that the CEO is exhibiting
        a persistent pattern that warrants mutation, not just a one-off outlier.
        """
        streak = 0
        for record in reversed(self._records):
            if record.override_detected:
                streak += 1
            else:
                break
        return streak

    def get_supervision_score(self) -> float:
        """
        Compute a composite supervision health score in [0, 1].

        Formula
        -------
          score = 1.0 - (override_rate * 0.5) - (consecutive_streak * 0.1)
          clamped to [0.0, 1.0]

        A score of 1.0 means no overrides and no streak (fully aligned CEO).
        A score approaching 0.0 indicates chronic override behaviour.
        """
        summary = self.get_pattern_summary()
        streak  = self.count_consecutive_override_failures()
        raw     = 1.0 - (summary["override_rate"] * 0.5) - (streak * 0.1)
        return round(max(0.0, min(raw, 1.0)), 4)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_log(self, path: str) -> None:
        """
        Serialise the full decision history to a JSON file.

        Parameters
        ----------
        path : Filesystem path to write the JSON log.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = [asdict(r) for r in self._records]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("[CEO_OVERSIGHT] Log saved to %s (%d records)", path, len(payload))

    def load_log(self, path: str) -> None:
        """
        Deserialise a previously saved JSON log and replace the current history.

        Parameters
        ----------
        path : Filesystem path to read the JSON log from.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._records = [CEODecisionRecord(**entry) for entry in data]
        logger.info("[CEO_OVERSIGHT] Log loaded from %s (%d records)", path, len(self._records))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_override(
        council_majority: str,
        ceo_decision:     str,
        aggregate_risk:   float,
    ):
        """
        Classify the CEO's decision relative to the council majority.

        Returns (override_detected: bool, override_direction: str | None).

        Rules (evaluated in priority order)
        ------------------------------------
        1. LENIENT              : council=REJECT  AND ceo=APPROVE
        2. AGGRESSIVE           : council=APPROVE AND ceo=REJECT
        3. LENIENT_RISK_MISMATCH: ceo=APPROVE AND aggregate_risk > 0.55
        4. None                 : no material divergence
        """
        if council_majority == "REJECT" and ceo_decision == "APPROVE":
            return True, "LENIENT"
        if council_majority == "APPROVE" and ceo_decision == "REJECT":
            return True, "AGGRESSIVE"
        if ceo_decision == "APPROVE" and aggregate_risk > 0.55:
            return True, "LENIENT_RISK_MISMATCH"
        return False, None
