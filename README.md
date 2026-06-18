# DeepDive — Agentic Research Synthesis Assistant

DeepDive takes a research question, **decomposes** it, **searches the live web**, builds a
**FAISS** index over what it finds (RAG), **extracts strict-JSON findings**, **flags
cross-source conflicts**, and writes a **citation-backed markdown report** — then **measures**
its own retrieval, extraction, and citation-faithfulness. It's served behind a **Dockerized
FastAPI** API.

It was built as a hands-on companion to DeepLearning.AI's *Building Agentic AI* course
(Andrew Ng). Every module maps to one of the four agentic design patterns plus the Module-4
eval discipline — see **[ARCHITECTURE.md](ARCHITECTURE.md)** for the file-by-file map.

```
  Planning            Tool use (web)     RAG (FAISS)     structured     Reflection      Multi-agent
     │                      │                 │            output           │               │
START ─▶ planner ─▶ searcher ─▶ retriever ──?──▶ extractor ─▶ critic ──?──▶ synthesizer ─▶ END
            ▲                      │  thin?                       │  gap?
            └──── replan ──────────┘            re-retrieve ──────┘
            (broaden queries, bounded)          (wider k, bounded)
```

- **LLM:** Groq (OpenAI-compatible) — `llama-3.3-70b-versatile` for reasoning, the cheaper
  `llama-3.1-8b-instant` for high-volume extraction.
- **Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2` (Groq has no embeddings API).
- **Search:** keyless **DuckDuckGo** by default; **Tavily** auto-used if `TAVILY_API_KEY` is set.

---

## Quickstart (Windows / PowerShell)

### 1. Set up the environment
```powershell
py -V:3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

### 2. Add your key
```powershell
copy .env.example .env
# open .env and set GROQ_API_KEY=...  (free at https://console.groq.com/keys)
# optional: set TAVILY_API_KEY=...     (free at https://app.tavily.com — better search)
```

### 3. Run a research query (CLI)
```powershell
python -m deepdive.pipeline "How would I build a rocket company to compete with SpaceX?"
# or the installed console script:
deepdive "What are the main trade-offs between solid-state and lithium-ion batteries?"
# full structured output:
deepdive "..." --json --max-sources 8
```
You'll get the markdown report, then a metrics + per-step latency/cost trace.

### 4. Run the API + Web UI
```powershell
uvicorn deepdive.api.app:app --reload
```
Then open **`http://localhost:8000`** for the web app — a professional console that
**streams the agent flow live**: each node (planner → searcher → … → synthesizer) lights
up as it runs, with a real-time per-step trace (latency / tokens / cost), the final cited
report, highlighted cross-source conflicts, an **Evals dashboard**, and **run history**.

Endpoints:
- `GET  /` — the web UI (served by FastAPI, no build step)
- `GET  /research/stream?question=...` — **Server-Sent Events**: one event per node (live flow)
- `GET  /health` — liveness + active models/provider
- `POST /research` → `202 {job_id}`, then poll `GET /research/{job_id}`
- `POST /research/sync` — run inline (handy for demos)
- `GET  /evals/last` · `POST /evals/run` · `GET /history` — dashboard + history data
- Interactive API docs at `http://localhost:8000/docs`

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/research/sync -H "Content-Type: application/json" -d '{"question":"What causes inflation?"}'
```

### 5. Run the evals (Module 4)
```powershell
python -m evals.runner --suite all                 # retrieval + extraction + citation
python -m evals.runner --suite citation --out eval_results.json
python -m evals.runner --suite extraction          # offline (no web), needs GROQ key
```
Prints a metrics table and dumps full per-case results for error analysis.

> **Linux/macOS:** same commands; activate the venv with `source .venv/bin/activate` and use
> `cp` instead of `copy`.

---

## Run in Docker
```powershell
docker compose up --build
curl http://localhost:8000/health
```
The image installs **CPU-only torch** (~1.7 GB instead of ~8 GB) and **bakes in the embedding
model** at build time, so the first request is fast and the container runs offline (apart from
web search + Groq). Provide keys via `.env` (read by compose) or `-e GROQ_API_KEY=...`.

---

## What the metrics mean (the resume bullet, made real)

| Metric | What it measures | Where |
|---|---|---|
| **Retrieval** (precision / recall / F1 / hit-rate@k) | Do retrieved source domains match a gold set? | `evals/metrics_retrieval.py` |
| **Extraction** (schema-valid rate, answer accuracy) | Does extraction return valid JSON with the right answer? | `evals/metrics_extraction.py` |
| **Citation faithfulness** (substring baseline + 5-criterion LLM judge + judge-vs-human κ) | Is each cited quote actually in / supportive of the source? | `evals/metrics_citation.py`, `evals/judge.py` |
| **Latency & cost per step** | Time + tokens + \$ for every node | `src/deepdive/benchmark.py` → `RunTrace` |

The citation judge uses **five binary (0/1) rubric criteria summed to 0–5** — deliberately not a
raw 1–5 score, which the course flags as unreliable and position-biased.

---

## Project layout
```
src/deepdive/
  config.py · llm.py · benchmark.py · pipeline.py     # config, Groq clients, cost trace, orchestrator
  schemas/        # Pydantic v2 contracts (the "strict JSON schema")
  agents/         # LangGraph: state, graph, routers + planner/searcher/retriever/extractor/critic/synthesizer
  retrieval/      # search (DDG/Tavily) · fetch (trafilatura) · index (FAISS + local embeddings)
  extraction/     # extract (strict JSON + reflection loop) · conflicts (rule + LLM-judge)
  api/            # FastAPI app, routes, job store
evals/            # datasets loader, 3 metric families, LLM judge, runner CLI
data/eval/        # retrieval.jsonl · extraction.jsonl · citation.jsonl (small gold sets)
tests/            # graph routing/loops (mocked), schemas, API, offline evals
Dockerfile · docker-compose.yml
```

---

## Deploying to AWS (documented next step — not built here)

The container is AWS-ready. To run it on ECS/Fargate:

1. `docker build --platform linux/amd64 -t deepdive:latest .` (Fargate is x86).
2. Push to **ECR**: create a repo, `aws ecr get-login-password | docker login ...`, tag, `docker push`.
3. Store `GROQ_API_KEY` / `TAVILY_API_KEY` in **AWS Secrets Manager**; reference them as `secrets`
   in the ECS task definition (never plaintext env).
4. **ECS/Fargate** service: 512–1024 CPU / 2–4 GB RAM (torch headroom), container port 8000,
   `/health` as the ALB target-group health check. The async `/research` + polling design keeps
   requests short; raise the ALB idle timeout only if you rely on `/research/sync`.
5. Logs → CloudWatch via the `awslogs` driver.

---

## Tech stack
Python 3.13 · LangGraph · LangChain (Groq, HuggingFace, Community/FAISS) · FAISS · sentence-transformers ·
Tavily / ddgs · trafilatura · Pydantic v2 · FastAPI · Docker.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the deep dive and the full course-concept map.
