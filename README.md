# Adaptive AI Enterprise Council — Financial Risk Governance

A multi-agent AI system simulating an enterprise boardroom for financial risk analysis.
Five C-suite agents independently evaluate financial data using locally hosted LLMs,
then vote via credibility-weighted aggregation to produce a final decision.

## Local LLM Stack (Ollama — no cloud API needed)

| Agent | Role | Model |
|-------|------|-------|
| CEO | Strategic growth | llama3 |
| CFO | Capital preservation | deepseek-r1 |
| Marketing | Market expansion | mixtral |
| PR | Reputation risk | llama3 |
| Legal | Regulatory compliance | deepseek-r1 |

## Quick Start

### 1. Install Ollama
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

### 2. Pull the required models
```bash
bash scripts/pull_models.sh
# or individually:
ollama pull llama3
ollama pull deepseek-r1
ollama pull mixtral
ollama pull phi3
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify setup
```bash
python scripts/check_ollama.py
```

### 5. Run the council pipeline
```bash
python -m pipeline.run_pipeline
```

### 6. Run the API server
```bash
uvicorn api.main:app --reload
# Docs: http://localhost:8000/docs
```

### 7. Run tests
```bash
pytest tests/
```

## Architecture
See `docs/architecture.md` for full system design.
