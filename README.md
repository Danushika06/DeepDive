# DeepDive — Agentic Research Synthesis Assistant

**A LangGraph pipeline that takes a research question, plans sub-questions, searches the live web, builds a FAISS index over what it finds, extracts quote-grounded findings, flags cross-source conflicts and writes a cited markdown report, with an eval harness and a FastAPI service around it.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat) ![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat&logo=meta&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)

## Overview

A single prompt asking an LLM to "write a cited research report" tends to produce confident text with citations nobody checked. DeepDive breaks the job into stages, gives each stage its own role and model, and checks the output at the points where it is easiest to cheat: quotes must appear in the source text, citation numbers are assigned by code rather than by the model, and conflicting sources are surfaced instead of averaged away.

I built it as a hands-on companion to DeepLearning.AI's *Building Agentic AI* course, so the repo is organised around the four patterns the course covers (planning, tool use, reflection, multi-agent) plus its evaluation module. `ARCHITECTURE.md` maps files to course concepts; this README describes what the code does.

Status: a working prototype and learning project. The graph, API, web UI, CLI and eval harness are all implemented and the graph routing is covered by tests that run with no network. The repo does not contain any recorded eval results, so I make no accuracy claims. The eval datasets are small gold sets, and the AWS deployment mentioned in the old README was never built.

## Key Features

- **Custom LangGraph `StateGraph`** — six hand-written nodes (`planner`, `searcher`, `retriever`, `extractor`, `critic`, `synthesizer`) over one typed `DeepDiveState`, wired in `agents/graph.py` with two conditional edges.
- **Two bounded loops** — a re-plan loop when retrieval is thin and a re-retrieve loop when the critic finds an uncovered sub-question. Both are capped by counters in state, so the graph exits gracefully before LangGraph's `recursion_limit`.
- **Two model handles** — `llama-3.3-70b-versatile` on Groq for planning, conflict judging and writing; the cheaper `llama-3.1-8b-instant` for the high-volume per-chunk extraction. Model ids come from config.
- **Pluggable search** — a `SearchProvider` ABC with Tavily and keyless DuckDuckGo implementations; `SEARCH_PROVIDER=auto` uses Tavily if a key is set and DuckDuckGo otherwise.
- **Quote-grounded extraction with a retry loop** — each chunk is sent to the extractor in Groq JSON mode. Schema errors and quotes that are not in the chunk are fed back to the model; a finding whose quote cannot be grounded after the retries is dropped.
- **Hybrid conflict detection** — an LLM judge per sub-question, topped up with rule-based checks for differing years and differing numbers across sources.
- **Deterministic citations** — `source_id` to `[n]` numbering and the References section are built in code (`synthesizer.py`); the model only places inline markers. A post-check computes `citation_faithfulness`, the share of inline markers that resolve to a real source.
- **Per-step trace** — every node is wrapped by `@traced_node`, which records latency, tokens, estimated USD cost and any exception into a `Span`; a node failure is recorded in `state["errors"]` instead of killing the run.
- **Injectable dependencies** — a `Deps` dataclass holds the LLMs, search provider and heavy functions, so tests run the whole graph with fakes.
- **Eval harness** — retrieval, extraction and citation suites, a five-criterion binary LLM judge, and judge-versus-human accuracy and Cohen's kappa.
- **API and no-build web UI** — async job plus polling, a sync endpoint, a Server-Sent Events stream of the agent flow, eval and history endpoints, and a static HTML/JS front end served by FastAPI.
- **Container** — multi-stage Dockerfile with CPU-only torch, the embedding model baked in, a non-root user and a healthcheck.

## How It Works

```
START -> planner -> searcher -> retriever --route_after_retrieval--> extractor -> critic --route_after_reflection--> synthesizer -> END
            ^                                    | "replan"                                  | "redo"
            +------------------------------------+                    retriever <-----------+
```

`pipeline.run_research()` builds default deps, compiles the graph with `build_graph(deps)`, seeds the state with `initial_state()` and invokes it with `recursion_limit=40`. `_assemble_report()` then packs the final state into a `ResearchReport` (report, citations, conflicts, findings, metrics, run trace).

### State and node plumbing

`agents/state.py` defines `DeepDiveState` as a `TypedDict`. `trace` and `errors` use an `operator.add` reducer so each node appends; every other field is last-write-wins. Loop counters (`plan_iterations`, `reflection_retries`) live in state. `agents/graph.py` binds `Deps` into each node with a small closure so LangGraph sees a plain `node(state)` signature. The runtime knobs (`top_k`, `min_sources_per_subq`, `max_plan_iters`, `max_reflection_iters` and so on) are passed in `state["config"]` by `Settings.runtime_config()`.

### The nodes

- **`planner_node`** asks the reasoning model for a structured `QueryPlan` (3 to 6 sub-questions, 2 to 3 queries each) via `with_structured_output`. Ids are rewritten to `sq1..sqN`. If the structured call fails it falls back to a single sub-question equal to the original question. On a re-plan the prompt says how few usable sources the last attempt found and asks for broader queries.
- **`searcher_node`** is not an LLM call. It runs `gather_sources()` from `retrieval/search.py`: run each query through the provider, de-duplicate by normalised URL, cap at `max_sources` (default 6), assign ids `S1..Sn`, and fill in text from Tavily's `raw_content` or from `fetch_and_clean()` (trafilatura). If a fetch fails but a snippet exists, the snippet is kept as the source text.
- **`retriever_node`** calls `build_index()` (`retrieval/index.py`): `RecursiveCharacterTextSplitter` (800 characters, 120 overlap), local `sentence-transformers/all-MiniLM-L6-v2` embeddings through `langchain-huggingface`, and an in-memory FAISS store built fresh for each request. It retrieves `top_k * (1 + reflection_retries)` chunks per sub-question, so a critic-triggered redo reads deeper into the same index, then de-duplicates chunks.
- **`extractor_node`** delegates to `extract_all()` in `extraction/extract.py`. It keeps the best 20 chunks by FAISS distance (`max_extract_chunks`), then calls `extract_from_chunk()` per chunk. That function binds `response_format={"type": "json_object"}` (the code comments say the 8B model often fails Groq's tool-calling path), validates into `RawExtraction`, and retries up to `extract_max_retries` times. Retry feedback is either the Pydantic validation error or the list of quotes that are not grounded. `_quote_is_grounded()` accepts a normalised substring match or at least 60% token overlap. Provenance (`source_id`, `url`, `title`) is attached afterwards in `FindingRecord.from_raw`, so the model never writes a citation target. Per-chunk failures are isolated and findings are de-duplicated.
- **`critic_node`** calls `detect_conflicts()` (`extraction/conflicts.py`). Findings are grouped by sub-question, and only groups with two or more distinct sources are examined. The LLM judge returns a `ConflictReport`; `_rule_based()` then adds a `date` conflict if sources mention different years and a `numeric` conflict if they give different numbers, unless the LLM already reported that type for that sub-question. The critic also checks coverage: any sub-question with zero findings sets `coverage_sufficient = False` and increments `reflection_retries`.
- **`synthesizer_node`** numbers citations from the findings (`_build_citations`), prompts the reasoning model to write the report using only those findings and markers, appends the References section itself, and computes `citation_faithfulness`. With no findings it returns a short "no sufficient evidence" stub.

### Routing

- `route_after_retrieval`: if unique sources are fewer than `min_sources_per_subq * number of sub-questions` and `plan_iterations < max_plan_iters` (2), go back to the planner; otherwise extract.
- `route_after_reflection`: if coverage is insufficient and `reflection_retries <= max_reflection_iters` (1), go back to the retriever; otherwise write. Because the critic increments the counter before routing, a persistent gap gives one extra retrieval pass and ends with `reflection_retries == 2`, which `tests/test_graph_flow.py` asserts.

### Tracing and cost

`benchmark.timed_step` times a node and builds a `Span` with an estimated cost from a hard-coded Groq price table (`PRICES`, labelled approximate in the code). `RunTrace.from_spans` totals latency, tokens and cost.

### Evals

`evals/` holds three suites, loaded from JSONL files in `data/eval/` and validated by Pydantic models in `evals/datasets.py`:

- **Retrieval** (`metrics_retrieval.py`): runs the live search provider for each question and scores retrieved domains against a gold domain list (precision, recall, F1, hit rate at k).
- **Extraction** (`metrics_extraction.py`): feeds fixed source text to `extract_from_chunk` and reports schema-valid rate and answer accuracy. It needs a Groq key but no web search.
- **Citation** (`metrics_citation.py`, `judge.py`): a substring baseline plus an LLM judge scoring five binary criteria (`quote_present`, `quote_supports_claim`, `no_overreach`, `correct_attribution`, `no_fabrication`); a case is faithful at score 4 or more. Each case has a human label, so the runner also reports judge-versus-human accuracy and Cohen's kappa.

`runner.py` prints a rich table and can dump per-case results with `--out`.

### API and UI

`api/app.py` creates the FastAPI app, warms the embedding model in the lifespan hook (skip with `DEEPDIVE_SKIP_WARMUP=1`) and serves `src/deepdive/web/` at `/`. `api/streaming.py` runs the graph with `stream_mode="values"` and emits `start`, `step`, `done` and `error` SSE events, one `step` per node span with running counts; completed runs are added to an in-memory `HistoryStore` (last 50). Jobs for `POST /research` are kept in an in-memory `JobStore`, so they are lost on restart and not shared between workers.

## Tech Stack

- **Language:** Python (`requires-python >=3.11`; the Dockerfile uses 3.13)
- **Orchestration:** LangGraph, LangChain Core
- **LLMs:** Groq via `langchain-groq` (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
- **RAG:** FAISS (`faiss-cpu`), `sentence-transformers` (MiniLM), `langchain-huggingface`, `langchain-text-splitters`
- **Search and fetch:** Tavily, `ddgs` (DuckDuckGo), trafilatura, httpx, tenacity
- **Schemas and config:** Pydantic v2, pydantic-settings
- **API and UI:** FastAPI, Uvicorn, Server-Sent Events, vanilla HTML/CSS/JS (marked and DOMPurify from a CDN)
- **CLI and evals:** rich, argparse
- **Tooling:** pytest, pytest-asyncio, ruff, mypy, Docker, Docker Compose

## Getting Started

### Prerequisites

- Python 3.11 or newer (the README and Dockerfile use 3.13)
- A Groq API key (`GROQ_API_KEY`)
- Optional: a Tavily API key for full-content search; without it DuckDuckGo is used
- Optional: Docker and Docker Compose

### Installation

```bash
git clone https://github.com/Danushika06/DeepDive.git
cd DeepDive
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps
cp .env.example .env             # Windows: copy .env.example .env
# then set GROQ_API_KEY in .env (TAVILY_API_KEY is optional)
```

### Running

CLI:

```bash
python -m deepdive.pipeline "What are the main trade-offs between solid-state and lithium-ion batteries?"
deepdive "..." --json --max-sources 8
```

API and web UI:

```bash
uvicorn deepdive.api.app:app --reload
# http://localhost:8000   (API docs at /docs)
```

Docker:

```bash
docker compose up --build
curl http://localhost:8000/health
```

Tests and evals:

```bash
pytest
python -m evals.runner --suite all
python -m evals.runner --suite citation --no-judge
python -m evals.runner --suite citation --out eval_results.json
```

## Usage

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/` | GET | The web UI (Research, Evals and History tabs) |
| `/health` | GET | Liveness plus the active models and search provider |
| `/research` | POST | Start a background job, returns `202` with `job_id` |
| `/research/{job_id}` | GET | Poll a job (`queued`, `running`, `done`, `error`) |
| `/research/sync` | POST | Run inline and return the report (times out after 4x `request_timeout_s`) |
| `/research/stream?question=...` | GET | SSE stream of the agent flow, one event per node |
| `/history`, `/history/{id}` | GET | In-memory run history for this process |
| `/evals/last` | GET | Contents of `eval_results.json`, if it exists |
| `/evals/run?suite=...` | POST | Run an eval suite in the background, poll `GET /evals/{job_id}` |

```bash
curl -X POST http://localhost:8000/research/sync \
  -H "Content-Type: application/json" \
  -d '{"question":"What causes inflation?","max_sources":6}'
```

Configuration lives in `src/deepdive/config.py` and can be overridden through `.env`: `GROQ_REASONING_MODEL`, `GROQ_EXTRACT_MODEL`, `EMBEDDING_MODEL`, `SEARCH_PROVIDER` (`auto`, `tavily`, `ddg`) and `LOG_LEVEL`.

## Project Structure

```
DeepDive/
├── src/deepdive/
│   ├── pipeline.py            # run_research(), report assembly, `deepdive` CLI
│   ├── config.py              # pydantic-settings: models, chunking, loop bounds
│   ├── llm.py                 # ChatGroq factories (reasoning vs extract)
│   ├── benchmark.py           # timed_step, price table, per-step cost
│   ├── agents/                # state, graph, routers, deps, @traced_node, six node files
│   ├── retrieval/             # search (Tavily/DDG), fetch (trafilatura), index (FAISS)
│   ├── extraction/            # extract (JSON mode + retry), conflicts (LLM + rules)
│   ├── schemas/               # Pydantic contracts: plan, findings, report, trace, sources
│   ├── api/                   # app, routes, SSE streaming, jobs, history, evals bridge
│   └── web/                   # index.html, app.js, styles.css (no build step)
├── evals/                     # datasets, three metric suites, LLM judge, runner CLI
├── data/eval/                 # retrieval / extraction / citation JSONL gold sets
├── tests/                     # mocked graph routing, API, schemas, offline evals
├── ARCHITECTURE.md            # design write-up tied to the course modules
├── Dockerfile                 # multi-stage, CPU torch, baked-in embedding model
├── docker-compose.yml
└── requirements*.txt
```

---

## Contact

<table>
  <tr><td><b>GitHub:</b> <a href="https://github.com/Danushika06">Danushika06</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/danushika-n-323b4628b/">danushika-n</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:danushikanagarajan@gmail.com">danushikanagarajan@gmail.com</a></td></tr>
</table>

Built by **Danushika N**
