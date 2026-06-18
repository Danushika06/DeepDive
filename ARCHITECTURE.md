# DeepDive — Architecture & Course Walkthrough

This doc explains **what was built and why**, tying each module back to a lecture from
DeepLearning.AI's *Building Agentic AI* (Andrew Ng). If you read one file alongside the
course, read this one.

The whole system is one application of the course's **core skill**: *take a complex task and
decompose it into small steps, each doable by an LLM or a tool.* "Write a cited research
report" is too big for one prompt — so we broke it into plan → search → retrieve → extract →
critique → synthesize, and gave each step the right tool or model.

---

## 1. The pipeline as a state machine

DeepDive is a **custom LangGraph `StateGraph`** (not the prebuilt `create_react_agent`). One
typed state object flows through six nodes; two conditional edges create bounded loops.

```
START ─▶ planner ─▶ searcher ─▶ retriever ──route?──▶ extractor ─▶ critic ──route?──▶ synthesizer ─▶ END
            ▲                        │  (thin)                          │  (gap)
            └─────── replan ─────────┘              re-retrieve ────────┘
```

`src/deepdive/agents/state.py` defines the shared `DeepDiveState`. Two list fields — `trace`
and `errors` — use an `operator.add` reducer so each node **appends** to them; everything else
is last-write-wins. `initial_state()` seeds the counters and empty accumulators.

Why custom `StateGraph` over `create_react_agent`? Each stage has a **distinct role, a distinct
model, a typed output, and explicit loops** back to earlier stages. A single prebuilt
agent-with-tools hides all of that in an opaque message scratchpad. Hand-written nodes make the
node list *be* the design-pattern map — which is the point of a learning build. ReAct still
appears, but where it belongs: **inside the planner** (reason → emit a query plan → observe on
re-plan).

---

## 2. File → course concept map

| File | Role in DeepDive | Course concept |
|---|---|---|
| `agents/planner.py` | Decompose the question into sub-questions + search queries | **Planning** (M5); ReAct |
| `agents/searcher.py` | Run queries through the active SearchProvider | **Tool use** (M3) |
| `retrieval/search.py` | Pluggable DDG/Tavily search behind one ABC | Tool use + **modularity** (M1.3) |
| `retrieval/fetch.py` | Fetch + clean pages (trafilatura) | Tool use (other AI/util) |
| `retrieval/index.py` | Chunk → embed (local) → FAISS → retrieve | **RAG** (M3 building blocks) |
| `agents/extractor.py` + `extraction/extract.py` | Strict-JSON findings + **reflection retry** | **Reflection** (M2.5) + structured output |
| `extraction/conflicts.py` + `agents/critic.py` | Cross-source conflict detection + coverage check | **Reflection** (M2) |
| `agents/synthesizer.py` | Write the citation-backed report | **Multi-agent** writer (M5) |
| `agents/graph.py` | Wire roles into a `StateGraph` | Multi-agent orchestration (M5) |
| `agents/routers.py` | Bounded replan / re-retrieve loops | `max_turns` idea (M3) |
| `benchmark.py` + `schemas/trace.py` | Per-step latency + token + cost spans | **Module 4.5** cost/latency |
| `evals/metrics_retrieval.py` | precision / recall / F1 / hit-rate vs gold domains | **M4** component, objective eval |
| `evals/metrics_extraction.py` | schema-valid rate + answer accuracy (offline) | **M4** objective eval |
| `evals/metrics_citation.py` + `judge.py` | substring baseline + 5-criterion LLM judge + judge-vs-human κ | **M2.4/M4** rubric judge, position bias |
| `evals/runner.py` | Metrics table + per-case JSON dump | **M4** error analysis |
| `api/*`, `Dockerfile`, `docker-compose.yml` | Deployed service | Deployment (FastAPI + Docker) |

---

## 3. The four patterns, concretely

### Planning (`agents/planner.py`)
The developer does **not** hard-code the steps. The planner LLM returns a structured
`QueryPlan` (3–6 sub-questions, each with 2–3 queries). Ids are canonicalized to `sq1..sqN` so
downstream nodes have stable keys. On a re-plan it's told how few sources the last attempt
found and asked to broaden — the *observe* half of ReAct.

### Tool use (`agents/searcher.py`, `retrieval/`)
Search is a tool the agent calls. `retrieval/search.py` defines a `SearchProvider` ABC with two
interchangeable implementations — Tavily and DuckDuckGo — selected by config. This is the
course's **modularity** point: swap the search engine without touching the pipeline. Tavily can
return clean page content directly (skipping the fetch step); DDG returns snippets that
`fetch.py` turns into full text via trafilatura. The **RAG** half (`index.py`) embeds the fetched
text locally (Groq has no embeddings endpoint) and retrieves the top-k chunks per sub-question,
each carrying `source_id`/`url`/`title` so citations stay traceable.

### Reflection (`extraction/extract.py`, `extraction/conflicts.py`, `agents/critic.py`)
Two reflective mechanisms, both using the course's key insight — **reflection is strongest with
new external information, not the model re-reading itself**:
- **Extraction retry loop:** the external feedback is a *validator*. If `with_structured_output`
  reports a schema error, or a `supporting_quote` isn't verbatim from the chunk, that exact error
  is fed back to the model to fix. A quote that can't be grounded after retries is dropped, so we
  never emit a fabricated citation.
- **Conflict detection (the headline deliverable):** the critic groups findings by sub-question
  and flags genuine cross-source disagreements via a **hybrid** of rule-based numeric/date checks
  (cheap, precise) and an LLM judge (semantic contradictions). It also does a coverage check; an
  uncovered sub-question is an evidence gap that triggers a bounded re-retrieve.

### Multi-agent (`agents/`)
Distinct role-played agents over one shared state: a **planner**, an **extractor**, a **critic**,
and a **synthesis** writer — the course's "hire a team, not one person." The synthesizer assigns
citations deterministically (`source_id → [n]`) so the References list is always correct; the LLM
only places inline `[n]` markers. `citation_faithfulness` is then a cheap post-check: the share of
inline markers that resolve to a real retrieved source.

---

## 4. Evals & error analysis (Module 4 — "the most important module")

The course's claim is that disciplined evals are the #1 predictor of success. DeepDive treats
evals as first-class:

- **Two axes** (how you score × whether there's per-example ground truth) are all represented:
  retrieval (code, gold domains), extraction (code, gold answers), citation (LLM-judge + human
  labels).
- **Component vs end-to-end:** retrieval and extraction are *component* evals (isolate one piece,
  cheap, low-noise); citation faithfulness is an *end-to-end* signal.
- **Rubric over Likert:** the citation judge sums five **binary** criteria (0–5), with a fixed
  field order to blunt **position bias** — exactly the course's fix for unreliable 1–5 scoring.
- **Evaluate your evaluator:** because the citation dataset carries human labels, the runner
  reports judge-vs-human **accuracy + Cohen's κ**. The substring baseline deliberately disagrees
  with humans on *overreach* cases (quote present but claim unfaithful) — that gap is the concrete
  reason an LLM judge is needed.
- **Read individual failures:** `runner.py --out results.json` dumps every per-case result, not
  just aggregates, for error analysis.
- **Cost/latency:** `benchmark.timed_step` records each node's latency, tokens, and \$ into the
  `RunTrace` returned with every report (Module 4.5: time *and* price each step).

The development loop the course prescribes — *build → look at traces → add a small eval → tune* —
is exactly what this repo's structure supports.

---

## 5. Key design decisions & trade-offs

| Decision | Why |
|---|---|
| Custom `StateGraph` over `create_react_agent` | Transparency, per-node models, typed state, explicit loops — and the deprecated prebuilt is opaque. |
| Two model handles (70B reason / 8B extract) | Extraction is the highest-volume call; the 8B model is ~10× cheaper. The reflection loop + verbatim-quote check guard its reliability. Swap to 70B via one config value if needed. |
| Local sentence-transformer embeddings | Groq has no embeddings endpoint; local MiniLM is free, offline, and fast. |
| Ephemeral per-request FAISS index | The corpus is built fresh per question from live search — nothing to persist or invalidate. |
| DDG keyless default, Tavily auto-upgrade | Runs out-of-the-box on a Groq key alone; upgrades to clean full-content search if a Tavily key is present. |
| Bounded loops via state counters | Graceful exit (proceed with a caveat) before LangGraph's hard `recursion_limit` ever trips — the `max_turns` idea. |
| Injectable `Deps` | Lets tests exercise the whole graph (routing + loop bounds) with fake LLMs/search and zero network. |
| Async job + polling API | The realistic pattern for long-running agent work; a sync endpoint is kept for demos. |

---

## 6. Data flow in one paragraph

A question enters `pipeline.run_research` → `initial_state` → the compiled graph. **planner**
writes `sub_questions` + `queries`; **searcher** turns queries into `FetchedPage`s (with stable
`S1..Sn` ids); **retriever** builds a FAISS index and writes top-k `Chunk`s (and replans if too
few unique sources); **extractor** turns chunks into grounded `FindingRecord`s (strict JSON +
reflection); **critic** writes `conflicts` and a coverage verdict (re-retrieving on a gap);
**synthesizer** writes `report_markdown` + `citations` + `citation_faithfulness`. Every node
appends a `Span`. `pipeline._assemble_report` packages it all — report, citations, conflicts,
findings, metrics, and the full `RunTrace` — into a `ResearchReport`, which is what both the CLI
and the API return.
