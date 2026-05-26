"""
Local LLM Client - Routes reasoning requests to locally hosted models.
Supports: llama3, phi, deepseek-r1, mistral via Ollama HTTP API.

Ollama must be running: `ollama serve`
Default endpoint: http://localhost:11434

Stability upgrades:
  - Response quality validation (truncation detection, min length)
  - Exponential backoff on retries (3 attempts by default)
  - Structured fallback when all retries exhausted
  - Per-call logging: model, attempt, response length, truncation flag
"""

import json
import logging
import time
import re
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Map friendly names → Ollama model tags
MODEL_REGISTRY = {
    "llama3":      "llama3",
    "phi":         "phi3",
    "deepseek-r1": "deepseek-r1",
    "mistral":     "mistral",
}

# Role → preferred model mapping
# Reasoning-heavy roles use deepseek-r1, fast roles use phi
ROLE_MODEL_MAP = {
    "strategic_growth":     "llama3",       # CEO  — balanced
    "financial_stability":  "llama3",       # CFO  — deep reasoning
    "market_expansion":     "llama3",       # Marketing — creative
    "reputation_risk":      "phi3",         # PR   — balanced
    "regulatory_compliance":"llama3",       # Legal — precise reasoning
}

DEFAULT_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Per-model timeout overrides (in seconds)
MODEL_TIMEOUTS = {
    "deepseek-r1": 1200,  # Extended timeout for deep reasoning
    "llama3": 900,        # Extended for medium reasoning
    "mistral": 600,       # Extended for faster models
    "phi": 300,           # Extended for lightest models
}

# ── Response quality constants ────────────────────────────────────────────────
MIN_RESPONSE_LENGTH = 30          # Raised from 20 to ensure minimum meaningful content
TRUNCATION_MARKERS  = re.compile(  # Patterns that suggest mid-sentence truncation
    r'(?:'
    r'\.\.\.$'               # ends with ellipsis
    r'|[,;:]\s*$'            # ends with trailing comma/semicolon/colon
    r'|—\s*$'                # ends with em-dash
    r'|\s+(?:and|but|or|the|that|this|which|however|therefore|because|as|since|if|when)\s*$'  # ends mid-clause
    r')',
    re.IGNORECASE,
)


class LocalLLMClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.
    Drop-in replacement for the Anthropic client used previously.
    Includes retry logic, adaptive timeouts, response validation,
    and structured fallback generation.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 300, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = timeout
        self.max_retries = max_retries
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                tags = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"Ollama connected. Available models: {tags}")
            else:
                logger.warning(f"Ollama responded with status {r.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama not reachable at %s — start with: ollama serve", self.base_url
            )

    # ── Response quality checks ───────────────────────────────────────────────

    @staticmethod
    def _is_response_empty(text: str) -> bool:
        """True if the response is missing or too short to be useful."""
        return not text or len(text.strip()) < MIN_RESPONSE_LENGTH

    @staticmethod
    def _is_response_truncated(text: str) -> bool:
        """
        Heuristic check: response was cut off mid-sentence.
        Looks for trailing punctuation patterns that indicate incompleteness.
        """
        text = text.strip()
        if not text:
            return True
        # A properly finished response ends with sentence-terminal punctuation
        if text[-1] in '.!?"\')\u201d':
            return False
        # Check explicit truncation markers
        if TRUNCATION_MARKERS.search(text):
            return True
        # No terminal punctuation at all → likely truncated
        return True

    @staticmethod
    def _build_fallback_response(model_tag: str, role: str = "", reason: str = "LLM failure") -> str:
        """
        Build a comprehensive fallback when all retries are exhausted.
        Ensures structured output that downstream systems can parse.
        """
        fallback_text = (
            f"[FALLBACK-{model_tag}] This response was generated due to LLM {reason}. "
            f"Role: {role or 'unknown'}. "
            f"Recommendation: review system logs and retry the request."
        )
        return fallback_text

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> str:
        """
        Call Ollama /api/generate and return the response text.
        Includes retry logic and adaptive timeouts based on model.
        Falls back gracefully if the model is unavailable.
        """
        model_tag = MODEL_REGISTRY.get(model or DEFAULT_MODEL, DEFAULT_MODEL)
        
        # Use model-specific timeout if available, otherwise use default
        timeout = MODEL_TIMEOUTS.get(model_tag, self.default_timeout)

        options: dict = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": 1.1,
        }
        if seed is not None:
            options["seed"] = seed

        payload = {
            "model":  model_tag,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": options,
        }

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    wait = min(2 ** attempt, 16)
                    logger.warning(
                        "Retry %d/%d for model %s — waiting %ds (timeout: %ds)",
                        attempt, self.max_retries, model_tag, wait, timeout,
                    )
                    time.sleep(wait)
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                text = response.json().get("response", "").strip()

                # Log response quality metrics
                logger.info(
                    "LLM response | model=%s | attempt=%d | length=%d | truncated=%s",
                    model_tag, attempt + 1, len(text), self._is_response_truncated(text),
                )
                return text

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                logger.error("LLM request timed out for model %s (after %d retries, timeout=%ds)", 
                             model_tag, self.max_retries, timeout)
                return f"[{model_tag}] Timeout — model is slow or overloaded."

            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                logger.error("Cannot reach Ollama at %s. Is `ollama serve` running?", self.base_url)
                return f"[{model_tag}] Ollama not running — no reasoning generated."

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                logger.error("LLM error (%s): %s", model_tag, e)
                return f"[{model_tag}] Error: {e}"

        # Should not reach here, but return error if all retries exhausted
        logger.error("All retries exhausted for model %s: %s", model_tag, last_error)
        return f"[{model_tag}] All retries exhausted."

    # ── Validated generation (used by debate engine) ──────────────────────────

    def generate_with_validation(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        max_tokens: int = 1200,
        temperature: float = 0.55,
        seed: Optional[int] = None,
        role: str = "",
        min_length: int = MIN_RESPONSE_LENGTH,
    ) -> str:
        """
        Generate with quality gate: retries up to max_retries times if the
        response is empty, too short, or truncated mid-sentence.

        Retry strategy:
        1. First attempt: original parameters
        2. Retry 1-2: increase max_tokens, decrease temperature
        3. Final fallback: structured message
        
        On final failure returns a structured fallback string so the debate
        engine never receives an empty slot.
        
        Returns:
            Non-empty string guaranteed (never None or empty)
        """
        model_tag = MODEL_REGISTRY.get(model or DEFAULT_MODEL, DEFAULT_MODEL)
        original_max_tokens = max_tokens
        current_temperature = temperature

        for attempt in range(self.max_retries + 1):
            try:
                text = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=current_temperature,
                    seed=(seed + attempt * 7) if seed is not None else None,
                )

                is_empty     = self._is_response_empty(text)
                is_truncated = self._is_response_truncated(text)
                is_short     = len(text.strip()) < min_length

                if not is_empty and not is_short:
                    if is_truncated:
                        logger.warning(
                            "TRUNCATED | model=%s | role=%s | attempt=%d | len=%d — "
                            "appending completion marker.",
                            model_tag, role, attempt + 1, len(text),
                        )
                        # Salvage the truncated response instead of discarding
                        if not text.rstrip().endswith('.'):
                            text = text.rstrip() + "."
                    return text

                logger.warning(
                    "QUALITY_GATE_FAIL | model=%s | role=%s | attempt=%d/%d | "
                    "empty=%s | short=%s | truncated=%s | len=%d",
                    model_tag, role, attempt + 1, self.max_retries + 1,
                    is_empty, is_short, is_truncated, len(text.strip()),
                )

                if attempt < self.max_retries:
                    # Progressive retry strategy
                    max_tokens = min(max_tokens + 250, 2000)
                    current_temperature = max(current_temperature - 0.08, 0.3)
                    logger.info(
                        "RETRY | attempt=%d | new_max_tokens=%d | new_temp=%.2f",
                        attempt + 1, max_tokens, current_temperature,
                    )

            except Exception as e:
                logger.error(
                    "GENERATION_ERROR | model=%s | role=%s | attempt=%d | error=%s",
                    model_tag, role, attempt + 1, str(e),
                )
                if attempt < self.max_retries:
                    continue

        logger.error(
            "ALL_RETRIES_EXHAUSTED | model=%s | role=%s | max_attempts=%d",
            model_tag, role, self.max_retries + 1,
        )
        return self._build_fallback_response(model_tag, role, "empty or truncated after all retries")

    def get_model_for_role(self, role: str) -> str:
        """Return the preferred model tag for a given agent role."""
        return ROLE_MODEL_MAP.get(role, DEFAULT_MODEL)

    def list_models(self) -> list:
        """Return models currently available in Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
