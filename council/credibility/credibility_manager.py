"""
Credibility Manager - Dynamic agent credibility tracking.

Upgrade note (Item 3):
  CredibilityTuner added — adjusts alpha/beta/gamma/delta every 10 cycles based
  on council performance. Uses council consensus risk variance as the outcome
  signal (always available, no ground truth labels required).

  Tuning logic (concrete, not vague):
    - gamma DOWN if: in cycles where most agents agreed (high agreement_ratio),
      the resulting consensus risk showed high variance across those cycles.
      Agreements that produce inconsistent outcomes mean gamma over-rewards
      agreement, so it should decrease.
    - delta UP if: agents with high historical_error in cycle N also had high
      error in cycle N+1 — i.e. historical error IS predictive, so weight it more.
    - alpha/beta adjusted proportionally to keep parameters bounded.
  All parameters clamped to [0.02, 0.85] and their sum is kept ≤ 1.0.
"""

import logging
import statistics
from typing import List, Dict, Optional

from agents.base.base_agent import AgentPrediction

logger = logging.getLogger(__name__)

# Parameter bounds
_PARAM_MIN = 0.02
_PARAM_MAX = 0.85
_TUNE_EVERY = 10   # cycles between tuning runs


class CredibilityTuner:
    """
    Self-tuning controller for credibility formula parameters.

    Monitors cycle outcomes and adjusts alpha/beta/gamma/delta so the formula
    stays calibrated as agent behaviour and market conditions evolve.

    Outcome signal: council consensus risk variance — derived from aggregate
    risk scores produced each cycle, always available without ground truth labels.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.15,
                 gamma: float = 0.10, delta: float = 0.05):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self._cycle_records: List[Dict] = []   # one entry per update_all() call
        self._tune_count = 0

    def record_cycle(self, updates: Dict, consensus_risk: float,
                     agreement_fractions: List[float],
                     historical_errors: List[float]) -> None:
        """
        Record one cycle's data for later tuning analysis.

        Args:
            updates:             return value of CredibilityManager.update_all()
            consensus_risk:      aggregate risk score from WeightedAggregator
            agreement_fractions: per-agent agreement (1.0=agreed, 0.0=dissented)
            historical_errors:   per-agent historical error values from that cycle
        """
        self._cycle_records.append({
            "consensus_risk":       consensus_risk,
            "agreement_fractions":  agreement_fractions,
            "historical_errors":    historical_errors,
        })

    def maybe_tune(self) -> bool:
        """
        Run a tuning pass if enough cycles have accumulated. Returns True if tuned.
        """
        if len(self._cycle_records) < _TUNE_EVERY:
            return False

        window = self._cycle_records[-_TUNE_EVERY:]

        # ── Gamma tuning ─────────────────────────────────────────────────────
        # Identify high-agreement cycles (mean agreement > 0.6) and check if
        # consensus risk in those cycles was volatile (high stdev).
        # Volatile consensus despite high agreement → gamma is over-rewarding
        # conformity that doesn't actually produce stable outcomes.
        high_agree_risks = [
            r["consensus_risk"]
            for r in window
            if r["agreement_fractions"] and
               sum(r["agreement_fractions"]) / len(r["agreement_fractions"]) > 0.60
        ]
        if len(high_agree_risks) >= 3:
            risk_variance = statistics.stdev(high_agree_risks)
            if risk_variance > 0.10:      # high volatility despite agreement
                old_gamma  = self.gamma
                self.gamma = max(_PARAM_MIN, self.gamma - 0.01)
                if self.gamma != old_gamma:
                    logger.info(f"[TUNER] gamma {old_gamma:.3f}→{self.gamma:.3f} "
                                f"(agreement risk variance={risk_variance:.3f})")

        # ── Delta tuning ─────────────────────────────────────────────────────
        # Check if high historical_error in one cycle predicts high error in the
        # next cycle. If the correlation is strong, delta should increase.
        hist_errors_seq = [
            r["historical_errors"] for r in window if r["historical_errors"]
        ]
        if len(hist_errors_seq) >= 4:
            avg_errors = [sum(e) / len(e) for e in hist_errors_seq]
            # Simple lag-1 correlation: do high-error cycles follow high-error cycles?
            pairs = list(zip(avg_errors[:-1], avg_errors[1:]))
            if pairs:
                # If both consecutive errors are above median, error IS auto-correlated
                med = statistics.median(avg_errors)
                both_high = sum(1 for a, b in pairs if a > med and b > med)
                autocorr_ratio = both_high / len(pairs)
                if autocorr_ratio > 0.50:
                    old_delta  = self.delta
                    self.delta = min(_PARAM_MAX, self.delta + 0.01)
                    if self.delta != old_delta:
                        logger.info(f"[TUNER] delta {old_delta:.3f}→{self.delta:.3f} "
                                    f"(error auto-correlation={autocorr_ratio:.2f})")

        # ── Alpha/beta rebalance ──────────────────────────────────────────────
        # Keep alpha dominant but not extreme; shrink if gamma+delta grew.
        overflow = (self.alpha + self.beta + self.gamma + self.delta) - 1.0
        if overflow > 0:
            self.alpha = max(_PARAM_MIN, self.alpha - overflow * 0.7)
            self.beta  = max(_PARAM_MIN, self.beta  - overflow * 0.3)

        self._tune_count += 1
        logger.info(
            f"[TUNER] Pass #{self._tune_count}: "
            f"α={self.alpha:.3f} β={self.beta:.3f} "
            f"γ={self.gamma:.3f} δ={self.delta:.3f}"
        )
        return True

    def get_params(self) -> Dict:
        return {
            "alpha": self.alpha, "beta": self.beta,
            "gamma": self.gamma, "delta": self.delta,
        }


class CredibilityManager:
    """
    Tracks and updates agent credibility scores across cycles.
    Optionally integrates a CredibilityTuner that self-adjusts formula parameters.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.15,
                 gamma: float = 0.10, delta: float = 0.05,
                 tuner: Optional[CredibilityTuner] = None):
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.delta  = delta
        self.tuner  = tuner
        self.history: List[Dict] = []
        self._cycle_count = 0

    def update_all(self, agents: list, predictions: List[AgentPrediction],
                   final_decision: str,
                   ground_truth: str = None,
                   consensus_risk: float = 0.50) -> Dict:
        """
        Update all agent credibility scores for one cycle.

        Args:
            agents:         list of BaseAgent instances
            predictions:    corresponding AgentPrediction objects
            final_decision: winning decision from WeightedAggregator
            ground_truth:   actual outcome if known (usually None)
            consensus_risk: aggregate risk score from this cycle (for tuner)
        """
        updates = {}
        agreement_fractions = []
        historical_errors   = []

        for agent, prediction in zip(agents, predictions):
            performance = 1.0 if (ground_truth and prediction.decision == ground_truth) else 0.5
            agreement   = 1.0 if prediction.decision == final_decision else 0.0
            hist_error  = self._historical_error(agent)

            agreement_fractions.append(agreement)
            historical_errors.append(hist_error)

            old      = agent.credibility
            new_cred = agent.update_credibility(
                self.alpha, self.beta, self.gamma, self.delta,
                performance, agreement, hist_error
            )
            updates[agent.role] = {
                "old":         round(old, 4),
                "new":         round(new_cred, 4),
                "performance": round(performance, 4),
                "agreement":   agreement,
            }
            logger.info(f"  [{agent.role}] credibility {old:.3f} → {new_cred:.3f}")

        self.history.append({"cycle": len(self.history) + 1, "updates": updates})
        self._cycle_count += 1

        # Feed this cycle's data to the tuner (if wired)
        if self.tuner is not None:
            self.tuner.record_cycle(
                updates=updates,
                consensus_risk=consensus_risk,
                agreement_fractions=agreement_fractions,
                historical_errors=historical_errors,
            )
            if self._cycle_count % _TUNE_EVERY == 0:
                tuned = self.tuner.maybe_tune()
                if tuned:
                    # Sync parameters back from tuner
                    p = self.tuner.get_params()
                    self.alpha = p["alpha"]
                    self.beta  = p["beta"]
                    self.gamma = p["gamma"]
                    self.delta = p["delta"]

        return updates

    def _historical_error(self, agent) -> float:
        """
        Estimate agent's historical error as how close its recent risk scores
        are to ambiguous territory. Confident decisions (far from 0.5) have
        lower historical error; wishy-washy predictions near 0.5 score higher.
        """
        if len(agent.prediction_history) < 5:
            return 0.0
        recent = agent.prediction_history[-5:]
        return 1.0 - sum(abs(p.risk_score - 0.5) for p in recent) / len(recent)

    def get_lowest(self, agents):
        return min(agents, key=lambda a: a.credibility)

    def get_highest(self, agents):
        return max(agents, key=lambda a: a.credibility)
