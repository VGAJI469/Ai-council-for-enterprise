# Quick Start Guide

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Configure environment
```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

## 3. Setup project
```bash
python scripts/setup.py
```

## 4. Train base model (optional — requires credit_risk.csv in data/raw/)
```bash
python models/training/train_base_model.py
```

## 5. Run the council pipeline
```bash
cd /path/to/adaptive-ai-council
python -m pipeline.run_pipeline
```

## 6. Run API server
```bash
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

## 7. Run tests
```bash
pytest tests/
```
