<div align="center">

# 🏛️ AI Enterprise Council

### *Where AI Agents Deliberate. Humans Decide.*

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black?style=for-the-badge)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **A production-grade, multi-agent decision intelligence system** — five specialist AI agents with distinct personas, dedicated LLM models, and RAG-grounded domain knowledge — that independently analyse, debate, vote on, and synthesise enterprise decisions. Fully local. Zero cloud dependency. Self-improving through outcome feedback.

<br/>

```
┌─────────────────────────────────────────────────────────────┐
│                  COUNCIL SESSION #ECS-4F2A                  │
│  Case: ₹500Cr Tier-2 City Expansion via Unsecured Loans    │
│                                                             │
│  FINAL VERDICT  ▶  CONDITIONAL APPROVE                     │
│  Council Confidence    ████████░░  74%                      │
│  Aggregate Risk        █████░░░░░  48%                      │
│                                                             │
│  CFO (30% weight) → DISSENTS: "Liquidity risk is           │
│  underweighted. Rising rates compress margins in 12 mo."   │
│                                                             │
│  Fresh Eyes Flag → "Assumes stable RBI stance.             │
│  Regulatory risk not fully priced into conditions."        │
└─────────────────────────────────────────────────────────────┘
```

</div>

---

## 🔒 Logging and Audit

This project includes a centralized structured logging and audit framework to help trace council sessions, debate rounds, agent decisions, and runtime failures.

- Log files are written to the `logs/` directory: `app.log`, `error.log`, and `audit.log`.
- Session audit reports are written as JSON to `reports/session_reports/session_<id>.json`.
- Logs are JSON-structured and include timestamps, logger names and optional metadata (e.g. `session_id`, `agent`).

Usage notes:

1. Start the FastAPI app as normal; the app initializes the structured logging on startup.
2. After a `/council/run` request completes, a session audit report is generated automatically.
3. Inspect `logs/app.log` and `logs/error.log` for operational events and warnings.


---

## 📌 The Problem This Solves

Enterprise decisions fail — not from lack of data, but from structural flaws in how decisions are made:

| Problem | Reality | Impact |
|---------|---------|--------|
| **Single decision maker** | One brain = one blind spot | Cognitive bias unchecked |
| **No devil's advocate** | Boardrooms suffer groupthink | Risks go unchallenged |
| **Gut feel over ground truth** | CFO guesses NPA benchmarks | Decisions not grounded in domain facts |
| **No accountability loop** | Poor advisors stay at the table | Bad advice keeps repeating |

**The AI Enterprise Council solves all four** — structurally, algorithmically, and automatically.

---

## ✅ What This Project Covers

This system demonstrates all four enterprise AI use cases:

```
✅  Process Orchestration Agents
    8-layer automated pipeline: input → RAG → analysis → debate
    → vote → synthesis → validation → feedback loop

✅  Meeting Intelligence Systems
    Extracts positions, creates conditions, assigns accountability,
    tracks escalation — without any manual follow-up

✅  Multi-Agent Collaboration
    4 specialist LLMs working in sequence and parallel:
    DeepSeek R1 (strategy) → Mixtral (critique) →
    LLaMA 3 (execution) → Phi-3 (evaluation)

✅  Workflow Health Monitors
    performance_tracker.py monitors agent accuracy across sessions,
    detects drift, and triggers automatic replacement before
    council reliability degrades

✅  CEO Governance & Oversight Layer (NEW)
    CEOOversightBoard detects override patterns (LENIENT /
    AGGRESSIVE / LENIENT_RISK_MISMATCH) across debate sessions.
    CEOMutator applies directional threshold corrections — no
    random replacement, targeted recalibration of the CEO config.
    CEOSupervisionController wires board + tracker + mutator
    into an automated governance loop with a generation cap of 5.
```

---

## 🏗️ System Architecture

```
                        USER SUBMITS DECISION
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend    │
                    │  POST /council/run  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   LAYER 2: RAG      │
                    │  Each agent queries │
                    │  its own ChromaDB   │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │    LAYER 3: ISOLATED ANALYSIS    │
              │     asyncio.gather() — all 5     │
              │   agents run simultaneously,     │
              │   zero cross-contamination       │
              └────────────────┬────────────────┘
                               │
    ┌──────┬──────────┬────────┴────────┬──────────┬──────┐
    │ CEO  │   CFO    │     Legal       │  Mktg    │  PR  │
    │DeepSeek│Mixtral │   LLaMA 3      │ LLaMA 3  │ Phi-3│
    └──────┴──────────┴────────┬────────┴──────────┴──────┘
                               │
                    ┌──────────▼──────────┐
                    │  LAYER 4: DEBATE    │
                    │  Max 3 rounds       │
                    │  Evidence required  │
                    │  Sycophancy blocked │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LAYER 5: VOTE      │
                    │  Credibility-       │
                    │  weighted outcome   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LAYER 6: SYNTHESIS │
                    │  Verdict, conditions│
                    │  risks, next steps  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LAYER 7: FRESH     │
                    │  EYES VALIDATION    │
                    │  Zero debate context│
                    │  Anti-groupthink    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LAYER 8: FEEDBACK  │
                    │  Outcomes update    │
                    │  credibility scores │
                    │  Poor agents auto-  │
                    │  replaced from bench│
                    └─────────────────────┘
```

---

## 🤖 The Council Agents

Five specialist agents sit on the default council. Each has a dedicated LLM model selected for its architectural strength.

<table>
<thead>
<tr>
<th>Agent</th>
<th>Role</th>
<th>LLM Model</th>
<th>Weight</th>
<th>Mandate</th>
</tr>
</thead>
<tbody>
<tr>
<td>🔴 <b>CEO</b></td>
<td>Chief Executive Officer</td>
<td>DeepSeek R1 7B</td>
<td>25%</td>
<td>Market position, valuation, first-mover advantage. Calculated risk is acceptable.</td>
</tr>
<tr>
<td>🟢 <b>CFO</b></td>
<td>Chief Financial Officer</td>
<td>Mixtral 8x7B</td>
<td><b>30% ★</b></td>
<td>Cash flow, NPA rates, cost of capital. Numbers do not lie. Optimism alarms me.</td>
</tr>
<tr>
<td>🔵 <b>Legal</b></td>
<td>Chief Legal Officer</td>
<td>LLaMA 3 8B</td>
<td>20%</td>
<td>RBI compliance, consumer protection, litigation risk. Non-compliance destroys companies.</td>
</tr>
<tr>
<td>🟡 <b>Marketing</b></td>
<td>Chief Marketing Officer</td>
<td>LLaMA 3 8B</td>
<td>15%</td>
<td>Customer acquisition, demand signals, market windows. Delay always has a cost.</td>
</tr>
<tr>
<td>🩷 <b>PR</b></td>
<td>Chief Comms Officer</td>
<td>Phi-3 Mini</td>
<td>10%</td>
<td>Brand trust, media narrative, reputational crises. Trust takes years to build.</td>
</tr>
</tbody>
</table>

> ★ CFO holds the highest credibility weight by default — financial risk grounding is the most critical constraint in the fintech domain this system was built for.

### Why Different LLMs Per Agent?

| Model | Architecture Strength | Why It Fits the Role |
|-------|----------------------|----------------------|
| **DeepSeek R1** | Chain-of-thought reasoning | CEO needs to decompose complex strategic tradeoffs step by step |
| **Mixtral 8x7B** | Mixture-of-experts | CFO needs to hold multiple risk dimensions simultaneously |
| **LLaMA 3 8B** | Strong instruction-following | Legal and Marketing need precise structured JSON output |
| **Phi-3 Mini** | Fast, efficient scoring | PR/Evaluator needs speed — it runs last and scores everything |

---

## 📚 RAG Knowledge System

Every agent retrieves real domain documents before forming its position. **No hallucination on domain facts.**

```
council/rag/knowledge_base/
│
├── cfo/              RBI NPA guidelines · cost-of-capital benchmarks
│                     fintech lending risk reports · default rate analysis
│
├── legal/            RBI digital lending circular 2022 · Consumer Protection Act 2019
│                     NBFC fair practices code · interest rate cap regulations
│
├── strategy/         Tier-2 market sizing reports · competitor analysis
│                     first-mover advantage case studies · India digital credit growth
│
├── marketing/        CAC benchmarks · Tier-2 consumer behaviour studies
│                     digital adoption in rural India · brand positioning frameworks
│
└── pr/               Fintech media sentiment analysis · responsible lending framework
                      predatory lending crisis case studies · reputational risk playbooks
```

### Retrieval Flow

```python
# At query time — for every agent, every session:
1. Agent's case text is embedded → nomic-embed-text via Ollama
2. ChromaDB similarity search → agent's own collection
3. Top 5 most relevant chunks retrieved with source filenames
4. Injected into prompt under DOMAIN KNOWLEDGE header
5. Agent instructed: "reason from retrieved text, cite sources"
```

**Result:** Every agent argument includes a `sources_used` field listing the actual documents that drove its conclusion.

---

## ⚔️ Debate Protocol

Based on the [focuslead AI Council Framework](https://github.com/focuslead/ai-council-framework) — research-backed methodology for multi-AI deliberation.

### Five Non-Negotiable Rules

```
┌─────────────────────────────────────────────────────────────┐
│  RULE 1 — ISOLATED ROUND 1                                  │
│  No agent sees any other agent's analysis before forming    │
│  its own position. asyncio.gather() ensures zero            │
│  cross-contamination. Eliminates anchoring bias.            │
├─────────────────────────────────────────────────────────────┤
│  RULE 2 — EVIDENCE-REQUIRED POSITION CHANGES               │
│  An agent may only shift its vote if it cites new evidence. │
│  Unsupported capitulation is algorithmically rejected.      │
│  position_changed=True without evidence_cited → REVERTED.   │
├─────────────────────────────────────────────────────────────┤
│  RULE 3 — 3-ROUND HARD LIMIT                               │
│  Extended deliberation → confidence rises, accuracy falls   │
│  (Xiong et al., 2025). Hard cap prevents exhaustion-driven  │
│  capitulation. No exceptions. No overrides.                 │
├─────────────────────────────────────────────────────────────┤
│  RULE 4 — PROTECTED DISSENT                                │
│  Minority positions are NEVER erased. Every agent whose     │
│  vote differs from the final outcome has its full argument  │
│  preserved in dissenting_views on the decision object.      │
├─────────────────────────────────────────────────────────────┤
│  RULE 5 — EARLY EXIT                                        │
│  If zero agents change position in a round, debate ends     │
│  immediately. Consensus is stable. No wasted LLM calls.     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗳️ Voting & Credibility System

### How Outcome Is Computed

```
weighted_score per agent = agent_credibility ÷ Σ all_credibilities

Final tally:
  ≥ 0.65  →  APPROVE
  0.35–0.64  →  CONDITIONAL APPROVE
  < 0.35  →  REJECT
```

### How Credibility Evolves

Submit real-world outcomes after decisions play out:

```bash
curl -X POST "http://localhost:8000/council/feedback?\
  outcome=worse_than_expected&\
  correct_agents=cfo&correct_agents=legal&\
  wrong_agents=marketing"
```

| Event | Adjustment |
|-------|------------|
| Named as `correct_agent` | **+5%** |
| Named as `wrong_agent` | **−5%** |
| `outcome=worse_than_expected` | CFO +3%, Legal +2%, Marketing −3% |
| `outcome=better_than_expected` | CEO +3%, Marketing +2%, CFO −2% |

All scores are re-normalised to sum to 100% after every update.

---

## 🔄 Agent Replacement System

Poor performers lose their seat. The best available candidate takes it.

### Replacement Triggers

```
Agent is flagged if ANY of these conditions are met:

  ① Credibility drops below 5%          ← credibility floor
  ② 3 consecutive wrong predictions      ← pattern of failure
  ③ Accuracy below 30% over 5+ sessions ← systemic underperformance

Minimum 3 sessions before any agent is eligible for replacement.
Maximum 1 replacement per session — worst performer first.
```

### The Bench — 8 Replacement Candidates

| ID | Name | Role | Speciality |
|----|------|------|------------|
| `risk_analyst` | Risk Analyst | Enterprise Risk Manager | Quantitative risk modelling, stress testing |
| `operations` | COO | Chief Operating Officer | Execution feasibility, capacity planning |
| `data_scientist` | CDO | Chief Data Officer | Evidence-based analysis, model risk |
| `investor_relations` | IR Head | Investor Relations Director | Analyst perception, board alignment |
| `product` | CPO | Chief Product Officer | Product-market fit, user adoption curves |
| `cybersecurity` | CISO | Chief Information Security Officer | Fraud risk, data security, system resilience |
| `sustainability` | CSO | Chief Sustainability Officer | ESG compliance, social licence |
| `economist` | Economist | Chief Economist | Macro forecasting, monetary policy impact |

### Human Override

```bash
# Reinstate any benched agent (human override)
curl -X POST http://localhost:8000/council/reinstate/marketing

# View full replacement audit trail
curl http://localhost:8000/council/vote-out-history

# See who is currently on the bench
curl http://localhost:8000/council/bench
```

---

## 🛡️ CEO Oversight & Governance System *(New)*

The council adds a dedicated accountability layer specifically for the CEO — the one agent whose strategic bias can override financial or legal signals.

### How Override Detection Works

```
Every debate session, CEOOversightBoard classifies the CEO's decision:

  LENIENT              → council said REJECT,  CEO said APPROVE
  AGGRESSIVE           → council said APPROVE,  CEO said REJECT
  LENIENT_RISK_MISMATCH→ CEO approved despite aggregate_risk > 0.55
  None                 → no material divergence — aligned decision
```

### Supervision Score

```
score = 1.0 − (override_rate × 0.5) − (consecutive_streak × 0.1)
clamped to [0.0, 1.0]

1.0  → fully aligned, no overrides
0.0  → chronic override behaviour across all recent sessions
```

### Directional Mutation (not random replacement)

When the override streak hits the threshold (default: 3), `CEOSupervisionController` triggers `CEOMutator`:

```
Lenient bias (≥ 3 LENIENT in last 10):
  risk_threshold  −= 0.05   (make CEO stricter)
  strategic weights −= 0.08 → quantitative weights += 0.08

Aggressive bias (≥ 3 AGGRESSIVE in last 10):
  risk_threshold  += 0.05   (loosen threshold)
  strategic weights += 0.06 → quantitative weights −= 0.06

No dominant pattern:
  risk_threshold += uniform(−0.03, +0.03)   (scheduled recalibration)
```

### Generation Cap

After **5 mutation generations** the CEO config resets to the `agents.yaml` baseline instead of compounding drift. This prevents the threshold from ever escaping its operational range.

### API

```bash
# CEO oversight dashboard (generation, score, override stats, mutation log)
curl http://localhost:8000/council/ceo/supervision

# Reset CEO state for a fresh demo (does NOT restart the agent)
curl -X POST http://localhost:8000/council/ceo/reset-supervision
```

---

## 📁 Project Structure

```
adaptive-ai-council/
│
├── 📄 requirements.txt
├── 📄 accuracy_report.json             Latest per-agent accuracy snapshot
├── 📄 test_quick_run.py                Smoke test — single council cycle
│
├── 📂 api/
│   ├── main.py                         FastAPI entry point
│   ├── 📂 routes/
│   │   └── council_routes.py           All /council/* endpoints
│   ├── 📂 schemas/                     Pydantic request/response models
│   └── 📂 middleware/
│
├── 📂 agents/
│   ├── 📂 base/
│   │   ├── base_agent.py               Abstract agent + predict() contract
│   │   └── llm_client.py               httpx → Ollama /api/generate
│   ├── 📂 roles/
│   │   ├── ceo_agent.py                DeepSeek R1 — strategic growth
│   │   ├── cfo_agent.py                Mixtral — financial risk
│   │   ├── legal_agent.py              LLaMA 3 — compliance
│   │   ├── marketing_agent.py          LLaMA 3 — market opportunity
│   │   └── pr_agent.py                 Phi-3 — reputation risk
│   └── 📂 evolution/
│       └── agent_factory.py            Builds council from agents.yaml
│
├── 📂 council/
│   ├── 📂 credibility/
│   │   └── credibility_manager.py      Update, clamp, normalise credibility
│   ├── 📂 debate/
│   │   ├── boardroom_debate.py         3-round debate loop + sycophancy guard
│   │   ├── company_debate.py           Financial-record-level debate runner
│   │   └── council_session.py          Session orchestrator — all 8 layers
│   ├── 📂 voting/
│   │   └── weighted_aggregator.py      Credibility-weighted vote + outcome
│   └── 📂 oversight/                   ★ NEW — CEO Governance Layer
│       ├── ceo_oversight.py            CEOOversightBoard + CEODecisionRecord
│       └── ceo_performance.py          CEOPerformanceTracker — accuracy streaks
│
├── 📂 evolution/
│   ├── 📂 mutation/
│   │   ├── agent_mutator.py            General directional mutator (4 agents)
│   │   └── ceo_mutator.py              ★ NEW — CEO-specific threshold mutator
│   ├── 📂 selection/
│   │   ├── evolution_controller.py     General evolution orchestrator
│   │   └── ceo_supervision_controller.py  ★ NEW — CEO governance loop
│   └── 📂 registry/
│
├── 📂 pipeline/
│   ├── run_pipeline.py                 End-to-end financial pipeline runner
│   ├── 📂 ingestion/
│   ├── 📂 preprocessing/
│   └── 📂 aggregation/
│       └── feature_builder.py
│
├── 📂 models/
│   ├── base_model.pkl                  Trained ML base model
│   └── 📂 inference/
│       └── model_wrapper.py
│
├── 📂 config/
│   ├── agents.yaml                     Agent definitions + ceo_supervision block
│   └── model.yaml                      LLM model assignments
│
├── 📂 data/
│   ├── 📂 raw/
│   └── 📂 processed/
│
├── 📂 scripts/
│   ├── generate_company_financial_data.py
│   ├── generate_synthetic_data.py
│   └── check_ollama.py
│
└── 📂 tests/
    └── (unit + integration tests)
```

---

## 🧰 Tech Stack

<table>
<tr>
<th>Layer</th>
<th>Technology</th>
<th>Why This Choice</th>
</tr>
<tr>
<td><b>API</b></td>
<td>FastAPI + Python 3.13</td>
<td>Native async support — critical for parallel agent execution</td>
</tr>
<tr>
<td><b>LLM Runtime</b></td>
<td>Ollama</td>
<td>Runs all models locally — zero data leaves the machine</td>
</tr>
<tr>
<td><b>Vector Store</b></td>
<td>ChromaDB</td>
<td>Persistent, embedded, per-agent collections with no server needed</td>
</tr>
<tr>
<td><b>Embeddings</b></td>
<td>nomic-embed-text</td>
<td>Best-in-class local embedding model via Ollama</td>
</tr>
<tr>
<td><b>Strategy LLM</b></td>
<td>DeepSeek R1 7B</td>
<td>Chain-of-thought reasoning for complex decomposition</td>
</tr>
<tr>
<td><b>Critique LLM</b></td>
<td>Mixtral 8x7B</td>
<td>Mixture-of-experts for multi-perspective risk analysis</td>
</tr>
<tr>
<td><b>Execution LLM</b></td>
<td>LLaMA 3 8B</td>
<td>Strong instruction-following for structured planning</td>
</tr>
<tr>
<td><b>Evaluation LLM</b></td>
<td>Phi-3 Mini</td>
<td>Fast, efficient scoring — ideal for final evaluation</td>
</tr>
<tr>
<td><b>HTTP Client</b></td>
<td>httpx</td>
<td>Async-first — non-blocking Ollama calls per agent</td>
</tr>
<tr>
<td><b>Validation</b></td>
<td>Pydantic v2</td>
<td>Strict schema enforcement on every LLM response</td>
</tr>
<tr>
<td><b>PDF Parsing</b></td>
<td>PyMuPDF</td>
<td>Extracts text from regulatory PDFs for knowledge base ingestion</td>
</tr>
<tr>
<td><b>Concurrency</b></td>
<td>asyncio</td>
<td>True parallel agent execution — all 5 agents run simultaneously</td>
</tr>
</table>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- 16 GB RAM minimum (32 GB recommended for Mixtral)
- 25 GB free disk space

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/VGAJI469/Ai-council-for-enterprise.git
cd ai-council

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Pull all required models
ollama pull deepseek-r1:7b
ollama pull mixtral:8x7b
ollama pull llama3:8b
ollama pull phi3:mini
ollama pull nomic-embed-text

# 4. Configure environment
cp .env.example .env

# 5. Add domain documents to knowledge base folders
cp your_rbi_circular.pdf council/rag/knowledge_base/legal/
cp your_npa_report.txt   council/rag/knowledge_base/cfo/

# 6. Build the knowledge base (run once)
python ingest.py

# 7. Start the server
uvicorn main:app --reload --port 8000
```

### Run Your First Council Session

```bash
curl -X POST http://localhost:8000/council/run \
  -H "Content-Type: application/json" \
  -d '{
    "case": "Should we approve ₹500Cr expansion into Tier-2 cities via unsecured consumer loans at aggressive interest rates?",
    "macro": "GDP Slowing",
    "rates": "Rising",
    "npa": "4.2%",
    "competition": "Aggressively Expanding"
  }'
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/council/run` | Run a full council session |
| `POST` | `/council/feedback` | Submit outcome to update credibility scores |
| `GET` | `/council/agents/status` | Live credibility scores + model assignments |
| `GET` | `/council/llm/models` | List available Ollama models |
| `GET` | `/council/evolution/log` | Agent evolution history and replacement events |
| `GET` | `/council/ceo/supervision` | ★ CEO oversight dashboard — generation, score, override stats, mutation log |
| `POST` | `/council/ceo/reset-supervision` | ★ Clear CEO oversight state for a fresh demo run |
| `GET` | `/health` | Health check |

---

## 📊 Sample Council Output

```jsonc
{
  "session_id": "ECS-4F2A",
  "final_verdict": "Conditional Pilot Approval",
  "aggregate_risk_score": 0.4831,
  "council_confidence": 0.7412,

  "decision": {
    "verdict": "Conditional Pilot Approval",
    "summary": "Credibility-weighted majority supports a limited pilot. CFO's high-confidence rejection mandates strict risk controls before any full rollout.",
    "conditions": [
      "Pilot capped at 30% of proposed scale",
      "Minimum credit score threshold raised for underwriting",
      "Monthly NPA gate reviews with automatic halt triggers"
    ],
    "risks": [
      "NPA rate could breach 7% within 12 months under rising rate scenario",
      "RBI regulatory scrutiny likely if aggressive pricing continues"
    ],
    "dissenting_views": [
      "CFO: Liquidity risk is underweighted. Rising rates will compress margins within 12 months. Full rejection recommended."
    ]
  },

  "fresh_eyes": {
    "missed_risks": [
      "Assumes stable RBI stance — post-2022 tightening not fully priced in",
      "Pilot selection bias could mask systemic default risk at full scale"
    ],
    "fragile_assumption": "That pilot NPA rate will be representative of full rollout",
    "confidence_in_verdict": 7,
    "recommendation": "Proceed"
  }
}
```

---

## 🧪 Accuracy Testing

```bash
# Full test suite — all agents, all checks
python tests/test_accuracy.py

# Quick mode — 1 iteration per test
python tests/test_accuracy.py --quick

# Single agent test
python tests/test_accuracy.py --agent cfo
```

| Test | What It Validates | Pass Threshold |
|------|------------------|----------------|
| **Consistency** | Same case, 3 runs — position stable | ≥ 66% match |
| **Differentiation** | Council should not unanimously agree | Diversity ≥ 0.4 |
| **JSON Reliability** | Valid output with all required fields | ≥ 80% success rate |
| **Confidence Calibration** | High confidence on clear cases, medium on ambiguous | Per-tier check |
| **Response Time** | Each model within acceptable latency | Model-specific thresholds |
| **Output Variance** | Risk score varies across different cases | Variance > 0.02 |

Results saved to `accuracy_report.json` with full timestamp and per-agent breakdown.

---

## ⚙️ Configuration

```env
OLLAMA_BASE=http://localhost:11434   # Ollama API URL
CHROMA_PATH=./chroma_store           # Vector DB storage path
KNOWLEDGE_PATH=./council/rag/knowledge_base
MAX_DEBATE_ROUNDS=3                  # Hard cap — do not increase
EMBED_MODEL=nomic-embed-text         # Local embedding model
```

---

## 🗺️ Roadmap

- [x] CEO Oversight Board — override detection across sessions
- [x] CEO Performance Tracker — accuracy streaks and floor enforcement
- [x] CEO Directional Mutator — targeted threshold correction (not random replacement)
- [x] CEO Supervision Controller — automated governance loop with generation cap
- [x] Management API — `/council/ceo/supervision` + `/council/ceo/reset-supervision`
- [ ] Persistent session storage with SQLite
- [ ] Streaming API — token-level streaming per agent
- [ ] React web dashboard for live council sessions
- [ ] Custom agent builder via API — no code changes needed
- [ ] Fine-tuned domain models for CFO and Legal agents
- [ ] Multi-council support — parallel sessions on different decisions
- [ ] Webhook integration — notify external systems on decision completion
- [ ] Multi-language council support

---

## 📖 Research Foundation

This system is built on peer-reviewed methodology:

- **[focuslead/ai-council-framework](https://github.com/focuslead/ai-council-framework)** — structured debate protocol, protected dissent, and credibility-weighted consensus
- **Xiong et al., 2025 — *Talk Isn't Always Cheap*** — evidence for the 3-round hard limit: extended deliberation increases AI confidence while decreasing accuracy, producing sycophancy through exhaustion

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the Hackathon · 100% Local · Zero Cloud Dependency**

*Five agents enter. The best decision wins.*

</div>
