# DevPath: Tech Career Roadmap Agent

> A RAG + agentic assistant that answers developer career questions grounded in data from 65,000+ real developers and global labor market research.

**Live app:** [devpath-tech-career-roadmap-agent.streamlit.app](https://devpath-tech-career-roadmap-agent.streamlit.app/)

![Landing Page](media/devpath-title.png)

> Note: First load may take 60 seconds as Streamlit Community Cloud wakes up on demand. Please be patient when accessing the site.

Built as a final project for [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) by DataTalksClub.

---

## The Problem

Developers often don't know what to learn next or how to plan their careers. Most advice online is:

- **Too generic** - "Learn Python and SQL" without explaining when, why, or where those skills are actually used.
- **US-focused** - Salary data, hiring trends, and recommended tools often reflect the US market, making them less useful for developers elsewhere.
- **Based on opinions** - Many recommendations come from blogs or personal experiences instead of real data.

Someone in the Philippines asking, "How do I become a data engineer?" needs different guidance than someone in Germany. Local job markets, salaries, and skill demand vary from country to country.

DevPath provides recommendations backed by real developer survey data, official occupation standards, and regional job market forecasts, with clear source citations so users can see where the information comes from.

---

## Demo

![DevPath Demo](media/demo.gif)

Try asking:
- *"I know Python and SQL. How do I become a data engineer?"*
- *"What tools do ML engineers use most in 2024?"*
- *"I am in the Philippines - is data science growing in Southeast Asia?"*
- *"What skills do I need to transition from data analyst to data scientist?"*

**Example answer:**

> To become a data engineer with Python and SQL, here are the key steps based on real data:
>
> According to the Stack Overflow Developer Survey 2024, 65.1% of data engineers use Python and 57.4% use SQL as primary languages (65,437 respondents worldwide). This confirms your current skills are well-aligned.
>
> According to O*NET 29.0 (US Dept of Labor), data engineers typically require skills in: Programming, Systems Analysis, Technology Design, and Database Management. Key tools include Apache Spark, Hadoop, AWS, and dbt...

---

## How It Works

DevPath uses a **RAG + agentic** architecture:

```
You type a question
       |
       v
Streamlit chat interface
       |
       v
AgentShim (Groq openai/gpt-oss-120b)
  - Decides what to search and how many times
  - Uses 2 tools: search() and search_by_source()
       |
       v
Knowledge base (187 chunks in DuckDB)
  - Stack Overflow Survey 2024 (34 role chunks)
  - Stack Overflow Survey 2025 (32 role chunks)
  - JetBrains Developer Ecosystem Survey 2025 (18 chunks)
  - O*NET 29.0 occupation data (7 tech role chunks)
  - WEF Future of Jobs 2025 (96 regional forecast chunks)
       |
       v
Hybrid retrieval (text + vector, RRF) + cross-encoder rerank
       |
       v
Answer with specific statistics and source citations
```

**Why AgentShim instead of Pydantic AI directly?**
Pydantic AI 2.22.0 has a confirmed bug with Groq tool calling, as it generates malformed function call format. AgentShim is a raw Groq client with a manual tool loop that produces correct tool calls. Pydantic is still used for `BaseModel` and `dataclass` type safety.

**Retrieval**: hybrid search (RRF over text + vector search) followed by local cross-encoder reranking (`sentence-transformers`, `ms-marco-MiniLM`) — see Retrieval Evaluation below for why this method was selected.

---

## Dataset

| Source | Organization | License | Coverage | Why |
|--------|-------------|---------|----------|-----|
| [Stack Overflow Survey, 2024](https://survey.stackoverflow.co/2024/) | Stack Overflow | ODbL | 65,437 devs, 185 countries | Real tool adoption - what developers actually use |
| [Stack Overflow Survey, 2025](https://survey.stackoverflow.co/2025/) | Stack Overflow | ODbL | 49,191 devs, 177 countries | Year-over-year shift in tool/language adoption |
| [JetBrains Developer Ecosystem Survey 2025](https://www.jetbrains.com/lp/devecosystem-2025/) | JetBrains | Free w/ attribution | 24,534 devs, 194 countries | Cross-checks SO data with an independently-sampled developer population |
| [O*NET 29.0](https://www.onetcenter.org/database.html) | U.S. Dept of Labor | CC BY 4.0 | 900+ occupations | Formal skill requirements - career roadmap data |
| [WEF Future of Jobs 2025](https://www.weforum.org/publications/the-future-of-jobs-report-2025/) | World Economic Forum | Free w/ attribution | 55 economies | Regional job growth forecasts - non-US context |

All datasets are **static snapshots**, meaning the knowledge base remains the same between runs. This makes the results reproducible and keeps evaluation metrics consistent.

**Why combine these five sources?** No single dataset can answer every career-related question:

- **Stack Overflow Developer Surveys (2024 + 2025)** show what technologies developers actually use, and how that's shifting year over year, but not the formal requirements for a role or future demand.
- **JetBrains Developer Ecosystem Survey** provides an independent cross-check on tool/language adoption from a differently-sampled developer population.
- **O*NET** defines the skills and responsibilities associated with occupations, but not which tools and technologies are most commonly used in practice.
- **WEF Future of Jobs** provides insights into regional labor market trends and demand, but not the specific technical skills needed for each role.

---

## Architecture

```mermaid
flowchart TD
    U([User]) --> ST[Streamlit UI\nstreamlit_app.py]
    ST --> AG[AgentShim\nRaw Groq Client + Manual Tool Loop]
    ST --> FB[Feedback\nthumbs up/down]
    FB --> MON[(DuckDB\nfeedback_log + query_times)]
    AG --> LLM[openai/gpt-oss-120b\nGroq free tier]
    AG --> LF[Logfire\nagent_run / llm_call / tool_call spans]
    LLM -->|search tool| IDX[index.py\ntext + vector + hybrid RRF + cross-encoder rerank]
    LLM -->|search_by_source tool| IDX
    IDX --> RR[Cross-encoder reranker\nsentence-transformers, CPU]
    IDX --> DB[(DuckDB\n187 chunks)]
    DB --> SO24[Stack Overflow 2024\n34 role chunks\n185 countries]
    DB --> SO25[Stack Overflow 2025\n32 role chunks\n177 countries]
    DB --> JB[JetBrains Ecosystem 2025\n18 chunks\n194 countries]
    DB --> ON[O*NET 29.0\n7 tech occupation chunks\nCC BY 4.0]
    DB --> WEF[WEF Future of Jobs 2025\n96 regional chunks\n55 economies]
```

---

## Retrieval Evaluation

Ground truth: **90 questions**, stratified across all 5 source prefixes (`jb2025`, `onet`, `so`, `so2025`, `wef`; 6 chunks sampled per source, 3 questions per chunk) so every dataset — including the newly-added Stack Overflow 2025 and JetBrains 2025 surveys — is represented, not just the first N chunks. Evaluation uses exact chunk-ID matching.

| Method | Hit Rate | MRR | Selected |
|--------|----------|-----|----------|
| text_search | 0.4556 | 0.3800 | |
| vector_search | 0.5778 | 0.4702 | |
| hybrid_search (RRF) | 0.6667 | 0.4893 | |
| **hybrid_search + rerank** | **0.6889** | **0.6276** | YES |

**hybrid_reranked selected** as the agent's primary retrieval method (best Hit Rate and MRR). Hybrid RRF retrieval pulls a larger candidate pool, then a local cross-encoder (`ms-marco-MiniLM`, via `sentence-transformers`) reranks down to the final top-n. This is also what `agent.py` uses in production.

**LLM-as-judge prompt comparison** (10 sampled questions, scored 1-5 by an LLM judge):

| Prompt Variant | Avg Score | Selected |
|----------------|-----------|----------|
| V1: concise | 2.4/5 | |
| V2: detailed + citations | **2.9/5** | YES |

To reproduce:
```bash
rm -f data/processed/ground_truth.json data/processed/eval_results.json
uv run python rag/evaluate.py
```
(Ground truth and eval results are cached to `data/processed/`; delete them to force a full regeneration. Without deleting, a rerun loads the cached files and completes in seconds.)

---

## Monitoring

DevPath uses [Logfire](https://logfire.dev) for observability. Every agent run produces a trace with child spans for each LLM call, tool call, and fallback recovery.

![Logfire Traces](media/logfire_traces.png)

**What is tracked per run:**
- `agent_run` : full question-to-answer duration (avg ~2.5s)
- `llm_call` : each Groq API call with token usage
- `tool_call` : each search tool invocation with query and source
- `llm_fallback` : triggered when Groq generates malformed tool calls (auto-recovery)

Configure `LOGFIRE_TOKEN` in `.env` to enable tracing.
View live traces at [logfire.dev](https://logfire.dev).

User feedback (thumbs up / thumbs down) is collected per answer via the Streamlit UI.

---

## Project Structure

```
devpath/
├── agent.py                  # Raw Groq agent + AgentShim manual tool loop (uses hybrid_search_reranked)
├── index.py                  # text + vector + hybrid search (RRF) + cross-encoder reranking
├── embedder.py                # ONNX embedder (all-MiniLM-L6-v2), thread-safe tokenizer access
├── streamlit_app.py          # Streamlit Cloud entry (calls agent directly)
├── monitoring.py             # Persists feedback + query times to DuckDB for the Stats dashboard
├── requirements.txt          # Streamlit Cloud dependencies
├── pyproject.toml            # uv / Docker dependencies (torch pinned to CPU-only wheel)
├── uv.lock                   # Locked dependency versions
├── Dockerfile.api            # FastAPI container
├── Dockerfile.ui             # Streamlit container
├── docker-compose.yml        # Local: API + UI orchestration
├── devpath_pipeline.duckdb   # DuckDB knowledge base (committed)
├── api/main.py               # FastAPI: /ask /feedback /health /stats
├── ui/app.py                 # Streamlit UI for local Docker run
├── ingestion/
│   ├── clean_so.py            # SO Survey 2024 -> so_chunks.json
│   ├── clean_so2025.py        # SO Survey 2025 -> so2025_chunks.json (NEW)
│   ├── clean_jetbrains.py     # JetBrains Ecosystem 2025 -> jetbrains_chunks.json (NEW)
│   ├── clean_onet.py          # O*NET -> onet_chunks.json
│   ├── extract_wef.py         # WEF PDF -> wef_chunks.json
│   └── pipeline.py            # dlt: JSON chunks -> DuckDB
├── rag/
│   ├── download.py           # Downloads ONNX model
│   └── evaluate.py           # Hit Rate + MRR + LLM-as-judge prompt comparison, with 429/TPD-aware retry + model fallback
├── models/                   # ONNX model files (committed, ~90MB)
└── data/processed/           # JSON chunks + eval results (committed)
    ├── so_chunks.json        # 34 developer role chunks (SO 2024)
    ├── so2025_chunks.json    # 32 developer role chunks (SO 2025)
    ├── jetbrains_chunks.json # 18 chunks (JetBrains Ecosystem 2025)
    ├── onet_chunks.json      # 7 tech occupation chunks
    ├── wef_chunks.json       # 96 regional forecast chunks
    ├── ground_truth.json     # 90 evaluation Q&A pairs (stratified across all 5 sources)
    └── eval_results.json     # Hit Rate + MRR + RAG prompt comparison results
```

---

## Setup

### Requirements
- Python 3.11+
- [uv](https://astral.sh/uv) (package manager)
- Docker Desktop (for local containerized run)
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Logfire token (free at [logfire.dev](https://logfire.dev))
- (Optional) Hugging Face token — set `HF_TOKEN` to avoid HF Hub's anonymous rate limit when the cross-encoder reranker downloads on first use

### Installation

```bash
git clone https://github.com/CalistaJajalla/devpath.git
cd devpath
uv sync
cp .env.example .env
# Edit .env and add your GROQ_API_KEY, LOGFIRE_TOKEN, and (optional) HF_TOKEN
```

### Download datasets

**Stack Overflow Survey 2024:**
```bash
mkdir -p data/raw
curl -L -o data/raw/survey_results_public.csv \
  'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2024/2024-09-03/stackoverflow_survey_single_response.csv'
```

**O*NET 29.0:**
```bash
BASE='https://www.onetcenter.org/dl_files/database/db_29_0_excel'
curl -L -o data/raw/onet_occupations.xlsx "$BASE/Occupation%20Data.xlsx"
curl -L -o data/raw/onet_skills.xlsx "$BASE/Skills.xlsx"
curl -L -o data/raw/onet_tech_skills.xlsx "$BASE/Technology%20Skills.xlsx"
```

**WEF Future of Jobs 2025** (must download manually):
1. Go to [weforum.org/publications/the-future-of-jobs-report-2025/](https://www.weforum.org/publications/the-future-of-jobs-report-2025/)
2. Download the PDF
3. Save as `data/raw/wef_2025.pdf` (~18MB)

### Run ingestion pipeline

```bash
uv add openpyxl
uv run python ingestion/clean_so.py      # -> data/processed/so_chunks.json
uv run python ingestion/clean_onet.py   # -> data/processed/onet_chunks.json
uv run python ingestion/extract_wef.py  # -> data/processed/wef_chunks.json
uv run python ingestion/pipeline.py     # -> devpath_pipeline.duckdb
```

> Note: Processed chunks are already committed to the repo. You only need to run ingestion if you want to rebuild from raw data.

### Run locally

**With Docker (recommended):**
```bash
docker compose up
# Open http://localhost:8501
```

**Without Docker:**
```bash
# Terminal 1: API
uv run uvicorn api.main:app --port 8000

# Terminal 2: UI
API_URL=http://localhost:8000 uv run streamlit run ui/app.py
```

### Environment variables

```bash
# .env (copy from .env.example)
GROQ_API_KEY=gsk_your-groq-key-here
LOGFIRE_TOKEN=your-logfire-write-token
LOGFIRE_READ_TOKEN=your-logfire-read-token
API_URL=http://localhost:8000
HF_TOKEN=hf_your-huggingface-token-here   # optional, avoids anonymous HF Hub rate limits for the cross-encoder
TOKENIZERS_PARALLELISM=false               # avoids "Already borrowed" tokenizer errors under concurrent access
```

---

## Evaluation Criteria Checklist

For peer reviewers - here is where to find each criterion:

| Criterion | Where |
|-----------|-------|
| Problem description | This README: Problem and Dataset sections |
| Retrieval flow | `index.py`, `agent.py`: RAG + agentic with 2 search tools, hybrid + cross-encoder rerank |
| Retrieval evaluation | `rag/evaluate.py`, `data/processed/eval_results.json`, table above |
| LLM evaluation | `rag/evaluate.py`: prompt V1 vs V2 comparison, LLM-as-judge |
| Interface | Live at [devpath-tech-career-roadmap-agent.streamlit.app](https://devpath-tech-career-roadmap-agent.streamlit.app/) / locally via docker compose up |
| Ingestion pipeline | `ingestion/` folder: 4 scripts + dlt pipeline to DuckDB |
| Monitoring | Logfire traces at logfire.dev + thumbs feedback in UI |
| Containerization | `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.ui` |
| Reproducibility | This README setup section + `.env.example` + `uv.lock` |

---

## Known Limitations

- **Pydantic AI compatibility**: The version used has issues with tool calling when paired with Groq, so the application uses the native Groq client as a workaround.
- **WEF data extraction**: The quality of extracted text depends on the PDF structure. If extraction is incomplete, the system falls back to using only the Stack Overflow Survey and O*NET datasets.
- **Memory usage**: The ONNX embedding model requires around 300 MB of memory. On Streamlit Cloud's 1 GB limit, the initial load may take longer. The cross-encoder reranker (`sentence-transformers`/torch, CPU-only) adds additional memory and cold-start time — expect a slower first load especially on Streamlit Community Cloud's free tier.
- **Groq rate limits**: The free tier may occasionally throttle requests, including daily (not just per-minute) token quotas on some models. `rag/evaluate.py` detects daily-quota errors and falls back to an alternate model rather than retrying indefinitely.
- **Evaluation metrics**: Hit Rate/MRR reflect exact chunk-ID matching across a heterogeneous, multi-source knowledge base; hybrid search with cross-encoder reranking gave the largest improvement over single-method retrieval (see table above).