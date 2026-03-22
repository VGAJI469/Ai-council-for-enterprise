#!/bin/bash
# pull_models.sh — Pull all required models into Ollama
# Run once before first use: bash scripts/pull_models.sh

echo "Pulling required models into Ollama..."
echo "(This may take several minutes depending on your connection)"

ollama pull llama3
ollama pull deepseek-r1
ollama pull mistral
ollama pull phi3

echo ""
echo "Done. Verify with: python scripts/check_ollama.py"
