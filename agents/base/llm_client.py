"""
Local LLM Client - Routes reasoning requests to locally hosted models.
Supports: llama3, phi, deepseek-r1, mistral via Ollama HTTP API.

Ollama must be running: `ollama serve`
Default endpoint: http://localhost:11434
"""

import json
import logging
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
    "financial_stability":  "deepseek-r1",  # CFO  — deep reasoning
    "market_expansion":     "mistral",      # Marketing — creative
    "reputation_risk":      "llama3",       # PR   — balanced
    "regulatory_compliance":"deepseek-r1",  # Legal — precise reasoning
}

DEFAULT_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"


class LocalLLMClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.
    Drop-in replacement for the Anthropic client used previously.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
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
        Falls back gracefully if the model is unavailable.
        """
        model_tag = MODEL_REGISTRY.get(model or DEFAULT_MODEL, DEFAULT_MODEL)

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

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out for model %s", model_tag)
            return f"[{model_tag}] Timeout — no reasoning generated."

        except requests.exceptions.ConnectionError:
            logger.error("Cannot reach Ollama. Is `ollama serve` running?")
            return f"[{model_tag}] Ollama not running — no reasoning generated."

        except Exception as e:
            logger.error("LLM error (%s): %s", model_tag, e)
            return f"[{model_tag}] Error: {e}"

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
