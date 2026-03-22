"""
check_ollama.py — Verify Ollama is running and required models are available.
Run this before starting the pipeline: python scripts/check_ollama.py
"""

import sys
import requests

OLLAMA_URL = "http://localhost:11434"
REQUIRED_MODELS = ["llama3", "deepseek-r1", "mistral", "phi3"]


def check():
    print("Checking Ollama connection...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("  ERROR: Ollama is not running.")
        print("  Start it with:  ollama serve")
        sys.exit(1)

    available = [m["name"] for m in r.json().get("models", [])]
    print(f"  Connected to Ollama at {OLLAMA_URL}")
    print(f"  Available models: {available}\n")

    missing = []
    for model in REQUIRED_MODELS:
        found = any(model in m for m in available)
        status = "OK" if found else "MISSING"
        print(f"  [{status}] {model}")
        if not found:
            missing.append(model)

    if missing:
        print(f"\n  Pull missing models with:")
        for m in missing:
            print(f"    ollama pull {m}")
        print()
        print("  Council will use llama3 as fallback for missing models.")
    else:
        print("\n  All required models available.")

    print("\n  Quick reasoning test (llama3)...")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3", "prompt": "In one sentence, what is financial risk?",
                  "stream": False, "options": {"num_predict": 60}},
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        print(f"  Response: {answer[:120]}")
        print("\n  Ollama is ready.")
    except Exception as e:
        print(f"  Test failed: {e}")


if __name__ == "__main__":
    check()
