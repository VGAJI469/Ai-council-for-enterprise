"""
Base Agent - Foundation class for all council agents.
Uses locally hosted LLMs via Ollama for reasoning generation.
Models available: llama3, phi, deepseek-r1, mistral

Upgrade notes:
  Item 4 — Dynamic blend ratio: 0.6/0.4 ML/role split replaced with per-agent
    adaptive weights that shift based on which component tracked the council
    consensus more accurately over the last 10 predictions. A slow decay toward
    the 0.6/0.4 baseline prevents permanent drift.

  Item 5 — Calibrated jitter: gauss(0, 0.04) replaced with _compute_jitter()
    whose sigma scales with input uncertainty (DTI, credit quality, volatility).
    High-risk turbulent inputs get larger sigma; clean inputs get smaller sigma.
    This makes the system express epistemic uncertainty rather than arbitrary noise.

  Item 6 — Memory-aware prompts: _build_memory_suffix() appends a compact dynamic
    block to every system prompt, giving the LLM situational awareness: its current
    credibility, last 5 decisions, whether it was recently evolved, and macro snapshot.

  Item 7 — Domain-intelligent debate prompts: _generate_reasoning() dynamically
    selects the 4-6 most relevant metrics for each agent's role, flags any metric
    in a critical danger zone, and includes council credibility distribution.

  Stability patch:
    - Temperature range narrowed to [0.50, 0.65] (was 0.55–0.90)
    - Deterministic seed derived from agent_id hash (was random.randint)
    - Output validation: decision, risk_score, reasoning checked after prediction
    - Retry on invalid prediction output (up to 2 re-generations)
    - max_tokens raised to 500 for reasoning (was 320)
"""

import logging
import random
import hashlib
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from agents.base.llm_client import LocalLLMClient, ROLE_MODEL_MAP

logger = logging.getLogger(__name__)

# ── Role-relevant metric selection (Item 7) ───────────────────────────────────

ROLE_METRICS = {
    "financial_stability":   [
        "debt_to_income_ratio", "liquidity_ratio_inv",
        "cash_flow_risk", "default_probability",
    ],
    "regulatory_compliance": [
        "regulatory_violation_prob", "compliance_score",
        "policy_risk", "legal_risk",
    ],
    "market_expansion":      [
        "market_growth_rate", "customer_churn_risk",
        "competitive_risk", "brand_risk",
    ],
    "reputation_risk":       [
        "sentiment_risk", "brand_risk",
        "media_risk", "stakeholder_risk",
    ],
    "strategic_growth":      [
        "market_growth_rate", "competitive_risk",
        "debt_to_income_ratio", "default_probability",
    ],
}

# Thresholds that trigger danger flags in the prompt
DANGER_THRESHOLDS = {
    "debt_to_income_ratio":       (">",  0.55),
    "credit_score":               ("<",  580.0),
    "market_volatility":          (">",  0.30),
    "regulatory_violation_prob":  (">",  0.60),
    "default_probability":        (">",  0.70),
    "compliance_score":           ("<",  0.40),
    "liquidity_ratio_inv":        (">",  0.70),
}

# Role-relevant keywords for reasoning quality scoring later (used by FitnessScorer)
ROLE_KEYWORDS = {
    "financial_stability":   {"dti", "liquidity", "cash", "capital", "debt",
                               "ratio", "default", "leverage", "solvency"},
    "regulatory_compliance": {"compliance", "regulatory", "liability", "policy",
                               "legal", "violation", "regulation", "statute"},
    "market_expansion":      {"market", "growth", "customer", "brand",
                               "competitive", "opportunity", "acquisition", "share"},
    "reputation_risk":       {"reputation", "sentiment", "brand", "media",
                               "stakeholder", "public", "trust", "perception"},
    "strategic_growth":      {"strategy", "growth", "competitive", "risk",
                               "market", "position", "long-term", "vision"},
}

# ── Stability constants ──────────────────────────────────────────────────────
REASONING_MAX_TOKENS    = 1000    # Increased from 500 to prevent truncation
REASONING_TEMP_LOW      = 0.48    # Tighter range for consistency
REASONING_TEMP_HIGH     = 0.58    # Reduced from 0.65 for stability
MAX_REASONING_RETRIES   = 3       # Retry up to 3 times on failure
FALLBACK_REASONING      = "Analysis deferred — LLM did not return valid reasoning within retry budget."


@dataclass
class AgentPrediction:
    agent_id:    str
    agent_role:  str
    risk_score:  float
    confidence:  float
    decision:    str              # APPROVE / REJECT / CONDITIONAL_APPROVE
    reasoning:   str
    model_used:  str = ""
    timestamp:   datetime = field(default_factory=datetime.utcnow)
    credibility: float = 1.0
    base_risk:   float = 0.0     # stored for blend ratio learning
    role_risk:   float = 0.0     # stored for blend ratio learning


class BaseAgent(ABC):
    """
    Abstract base class for all enterprise council agents.
    Each agent wraps a shared predictive ML model with role-specific
    decision biases and locally-hosted LLM reasoning.
    """

    def __init__(self, agent_id: str, config: dict, model=None,
                 llm_client: Optional[LocalLLMClient] = None):
        self.agent_id        = agent_id
        self.config          = config
        self.model           = model
        self.credibility     = config.get("initial_credibility", 1.0)
        self.role            = config["focus"]
        self.risk_threshold  = config["risk_threshold"]
        self.weights         = config["weights"]
        self.prediction_history: list = []

        # Shared LLM client (injected or created fresh)
        self.llm             = llm_client or LocalLLMClient()
        self.preferred_model = ROLE_MODEL_MAP.get(self.role, "llama3")

        # Item 4 — dynamic blend ratio (initialised at the canonical 0.6/0.4)
        self._ml_weight      = 0.60
        self._role_weight    = 0.40
        self._blend_history: list = []     # (base_error, role_error) per cycle
        self.last_consensus_risk: float = 0.50   # set externally by pipeline after each cycle

        # Item 6 — evolution flag (set externally by EvolutionController after replacement)
        self._recently_evolved: bool = False

        # Deterministic seed derived from agent ID for reproducibility
        self._base_seed = int(hashlib.md5(agent_id.encode()).hexdigest()[:8], 16)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def compute_role_risk(self, features: dict) -> float:
        """Apply role-specific weighting to raw features."""

    @abstractmethod
    def _static_prompt(self) -> str:
        """Return the role-specific system prompt (static portion)."""

    def get_system_prompt(self) -> str:
        """
        Full system prompt = static role persona + compact memory suffix.
        The memory suffix (Item 6) gives the LLM situational awareness without
        exceeding the token budget (~80 tokens maximum for the suffix).
        """
        return self._static_prompt() + self._build_memory_suffix()

    # ── Prediction pipeline ───────────────────────────────────────────────────

    def predict(self, financial_data: dict) -> AgentPrediction:
        """
        Full prediction pipeline:
        1. Base ML model risk score
        2. Role-specific risk adjustment
        3. Dynamic ML/role blend (Item 4) + calibrated jitter (Item 5)
        4. Decision thresholding
        5. Local LLM reasoning generation (with validation + retry)
        6. Blend ratio update
        """
        # Base ML score (from XGBoost) or feature-derived heuristic
        if self.model is not None:
            # base_risk = self.model.predict_proba(financial_data) -> here financial_data is a dict
            input_df = pd.DataFrame([financial_data])
            base_risk = float(self.model.predict_proba(input_df)[0][1])

        else:
            base_risk = financial_data.get("default_probability", 0.4)

        role_risk  = self.compute_role_risk(financial_data)

        # Item 5: context-aware jitter — sigma scales with input uncertainty
        jitter     = self._compute_jitter(financial_data)

        # Item 4: dynamic blend weights (instead of fixed 0.6 / 0.4)
        final_risk = (
            self._ml_weight   * base_risk
            + self._role_weight * role_risk
            + jitter
        )
        final_risk = min(max(final_risk, 0.0), 1.0)

        decision   = self._threshold_decision(final_risk)
        confidence = self._compute_confidence(final_risk)
        reasoning, model_used = self._generate_reasoning(financial_data, final_risk, decision)

        # ── Output validation ─────────────────────────────────────────────
        prediction = self._build_validated_prediction(
            base_risk, role_risk, final_risk, decision, confidence,
            reasoning, model_used,
        )
        self.prediction_history.append(prediction)

        # Item 4: update blend ratio using most recent consensus as ground signal
        self._update_blend_ratio(base_risk, role_risk, self.last_consensus_risk)

        return prediction

    def _build_validated_prediction(
        self, base_risk, role_risk, final_risk, decision, confidence,
        reasoning, model_used,
    ) -> AgentPrediction:
        """
        Validate that the prediction contains all required fields.
        If reasoning is missing or decision is invalid, apply fallbacks.
        """
        valid_decisions = {"APPROVE", "REJECT", "CONDITIONAL_APPROVE"}
        if decision not in valid_decisions:
            logger.warning(
                "Agent %s produced invalid decision '%s' — defaulting to CONDITIONAL_APPROVE",
                self.agent_id, decision,
            )
            decision = "CONDITIONAL_APPROVE"

        if not reasoning or len(reasoning.strip()) < 10:
            logger.warning(
                "Agent %s produced empty/short reasoning (len=%d) — using fallback",
                self.agent_id, len(reasoning.strip()) if reasoning else 0,
            )
            reasoning = FALLBACK_REASONING

        return AgentPrediction(
            agent_id=self.agent_id,
            agent_role=self.role,
            risk_score=round(final_risk, 4),
            confidence=round(confidence, 4),
            decision=decision,
            reasoning=reasoning,
            model_used=model_used,
            credibility=self.credibility,
            base_risk=round(base_risk, 4),
            role_risk=round(role_risk, 4),
        )

    # ── Item 4 — Dynamic blend ratio ─────────────────────────────────────────

    def _update_blend_ratio(self, base_risk: float, role_risk: float,
                            consensus_risk: float) -> None:
        """
        Adjust the ML/role blend weights based on which component tracked the
        council consensus more accurately over the last 10 predictions.

        Ground truth proxy: council consensus risk (always available — no labels
        needed). The component that was closer to consensus on average gains
        weight; the other loses weight symmetrically.

        A slow 0.5%/cycle decay toward the 0.6/0.4 baseline prevents permanent
        drift if the system dynamics change and a previously dominant component
        starts to fail.

        Replaces the hardcoded 0.6 * base_risk + 0.4 * role_risk which never
        adapted regardless of which component was performing better.
        """
        base_error = abs(base_risk - consensus_risk)
        role_error = abs(role_risk - consensus_risk)
        self._blend_history.append((base_error, role_error))

        # Only update after collecting 10 data points
        if len(self._blend_history) >= 10:
            window = self._blend_history[-10:]
            avg_base_err = sum(e[0] for e in window) / 10
            avg_role_err = sum(e[1] for e in window) / 10

            step = 0.03
            if avg_base_err < avg_role_err:
                # ML component is closer to consensus → increase its weight
                self._ml_weight   = min(self._ml_weight   + step, 0.85)
                self._role_weight = max(self._role_weight - step, 0.15)
            else:
                # Role-specific component is closer → increase its weight
                self._role_weight = min(self._role_weight + step, 0.85)
                self._ml_weight   = max(self._ml_weight   - step, 0.15)

        # Slow decay toward baseline (0.6/0.4) — prevents irreversible drift
        decay = 0.005
        self._ml_weight   += decay * (0.60 - self._ml_weight)
        self._role_weight += decay * (0.40 - self._role_weight)

    # ── Item 5 — Calibrated jitter ────────────────────────────────────────────

    def _compute_jitter(self, features: dict) -> float:
        """
        Context-aware noise whose magnitude scales with input uncertainty.

        Replaces gauss(0, 0.04) which applied the same noise magnitude to a
        pristine low-risk borrower as to a distressed high-DTI defaulter.
        High epistemic uncertainty (stressed DTI, poor credit, volatile market)
        deserves larger sigma; clean inputs deserve smaller sigma.

        sigma range: [0.02, 0.08]
          - Clean input (low DTI, good credit, calm market): ~0.02
          - Stressed input (all three flags active):         ~0.08
        """
        dti    = features.get("debt_to_income_ratio", 0.35)
        cs     = features.get("credit_score", 650)
        vol    = features.get("market_volatility", 0.18)

        cs_norm    = max(0.0, min((cs - 300) / 550.0, 1.0))
        dti_flag   = 1.0 if dti > 0.45 else (dti / 0.45)
        cs_flag    = 1.0 if cs_norm < 0.40 else (0.40 - cs_norm + 0.40) / 0.40
        cs_flag    = max(0.0, cs_flag)
        vol_norm   = min(vol / 0.40, 1.0)

        uncertainty = 0.40 * dti_flag + 0.30 * cs_flag + 0.30 * vol_norm
        sigma       = 0.02 + 0.06 * uncertainty
        return random.gauss(0, sigma)

    # ── Item 6 — Memory-aware system prompt suffix ────────────────────────────

    def _build_memory_suffix(self) -> str:
        """
        Build a compact dynamic context block appended to the static system prompt.

        Gives the LLM situational awareness on every call — without this, the LLM
        always reasons from a blank-slate persona with no memory of prior cycles.

        Kept deliberately compact (~80 tokens) to avoid crowding out reasoning within
        the max_tokens budget. Format: one line per piece of information.
        """
        lines = ["\n--- Agent Context ---"]
        lines.append(
            f"Credibility: {self.credibility:.3f} | "
            f"Blend: ML={self._ml_weight:.2f}/Role={self._role_weight:.2f} | "
            f"Evolved: {'Yes' if self._recently_evolved else 'No'}"
        )

        # Last 5 decisions (compact format)
        recent = self.prediction_history[-5:] if self.prediction_history else []
        if recent:
            hist_str = " | ".join(
                f"[C{i+1}] {p.decision[:4]} r={p.risk_score:.2f}"
                for i, p in enumerate(recent)
            )
            lines.append(f"Recent: {hist_str}")

        # Macro snapshot from last prediction data (if available)
        if self.prediction_history:
            last = self.prediction_history[-1]
            # Macro fields aren't on AgentPrediction directly; use stored consensus
            lines.append(f"Council consensus risk: {self.last_consensus_risk:.3f}")

        lines.append("--- End Context ---")
        return "\n".join(lines)

    # ── Item 7 — Domain-intelligent reasoning prompt ──────────────────────────

    def _generate_reasoning(self, data: dict, risk: float, decision: str):
        """
        Generate role-specific reasoning using a locally hosted LLM.

        Replaces the generic 4-metric prompt (DTI, credit score, loan amount,
        GDP growth) which was identical for every role. Now:
          - Dynamically selects the 4–6 most relevant metrics for this agent's role
          - Flags any metric in a critical danger zone with a ⚠ marker
          - Includes a compact council credibility distribution so the LLM knows
            whose influence dominates the current session
          - max_tokens raised to 500 to accommodate the richer prompt context
          - Retries up to MAX_REASONING_RETRIES times on empty/short output

        Stability improvements:
          - Temperature narrowed to [0.50, 0.65] for more consistent output
          - Deterministic seed derived from agent_id for reproducibility
          - Retry logic with progressive token budget increase
        """
        # Role-relevant metric selection
        role_metric_keys = ROLE_METRICS.get(self.role, [
            "debt_to_income_ratio", "credit_score", "loan_amount", "gdp_growth_rate"
        ])

        metric_lines = []
        danger_flags = []
        for key in role_metric_keys:
            val = data.get(key, "N/A")
            label = key.replace("_", " ").title()
            line  = f"  {label}: {val}"

            if key in DANGER_THRESHOLDS and val != "N/A":
                op, threshold = DANGER_THRESHOLDS[key]
                triggered = (op == ">" and float(val) > threshold) or \
                            (op == "<" and float(val) < threshold)
                if triggered:
                    line  += f"  [!] CRITICAL (threshold {op}{threshold})"
                    danger_flags.append(f"{label}={val}")

            metric_lines.append(line)

        metrics_block = "\n".join(metric_lines)
        danger_block  = (
            f"\n[!] DANGER ZONE ALERTS: {', '.join(danger_flags)}"
            if danger_flags else ""
        )

        # Credibility distribution (compact)
        cred_note = (
            f"Your credibility weight: {self.credibility:.3f}"
            f" | Council consensus: {self.last_consensus_risk:.3f}"
        )

        prompt = (
            f"Financial case — key metrics for your role ({self.role.replace('_', ' ').upper()}):\n"
            f"{metrics_block}"
            f"{danger_block}\n\n"
            f"Computed risk score: {risk:.4f} | Recommended decision: {decision}\n"
            f"{cred_note}\n\n"
            f"As the {self.role.replace('_', ' ').upper()} representative on the enterprise risk council, "
            f"give 2-3 concise sentences explaining your position on this decision. "
            f"Focus only on metrics relevant to your mandate. "
            f"If any danger zone alerts are present, address them directly. "
            f"End with a clear, complete concluding sentence."
        )

        # Deterministic seed from agent ID + prediction count
        seed = (self._base_seed + len(self.prediction_history)) % 100000
        temperature = round(random.uniform(REASONING_TEMP_LOW, REASONING_TEMP_HIGH), 2)

        # Generation happens via validated generate_with_validation below

        # Use validated generation with retry
        reasoning = self.llm.generate_with_validation(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            model=self.preferred_model,
            max_tokens=REASONING_MAX_TOKENS,
            temperature=temperature,
            seed=seed,
            role=self.role,
            min_length=30,
        )

        return reasoning, self.preferred_model

    # ── Standard helpers ──────────────────────────────────────────────────────

    def _threshold_decision(self, risk_score: float) -> str:
        if risk_score < self.risk_threshold * 0.7:
            return "APPROVE"
        elif risk_score > self.risk_threshold * 1.2:
            return "REJECT"
        else:
            return "CONDITIONAL_APPROVE"

    def _compute_confidence(self, risk_score: float) -> float:
        distance = abs(risk_score - self.risk_threshold)
        return min(0.5 + distance, 0.99)

    def update_credibility(self, alpha, beta, gamma, delta,
                           performance, agreement, historical_error):
        """
        Credibility(t+1) = α·C(t) + β·Performance + γ·Agreement − δ·HistoricalError
        """
        self.credibility = (
            alpha * self.credibility
            + beta  * performance
            + gamma * agreement
            - delta * historical_error
        )
        self.credibility = max(0.01, min(1.0, self.credibility))
        return self.credibility

    def to_dict(self) -> dict:
        return {
            "agent_id":        self.agent_id,
            "role":            self.role,
            "credibility":     round(self.credibility, 4),
            "risk_threshold":  self.risk_threshold,
            "preferred_model": self.preferred_model,
            "predictions":     len(self.prediction_history),
            "ml_weight":       round(self._ml_weight, 4),
            "role_weight":     round(self._role_weight, 4),
        }
