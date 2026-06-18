"""API routes.

The research pipeline is long-running, so the default path is async + polling:
  POST /research        -> 202 + job_id  (runs in a background task)
  GET  /research/{id}   -> poll status / result
  POST /research/sync   -> run inline with a timeout (handy for demos)
  GET  /health          -> liveness + active model/provider config

The pipeline is sync + CPU/IO heavy, so it runs via ``asyncio.to_thread`` and never
blocks the event loop.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..logging_conf import get_logger
from ..pipeline import run_research
from .evals_api import eval_jobs, load_last
from .history import history
from .jobs import store
from .models import HealthResponse, ResearchAccepted, ResearchRequest, ResearchResult
from .streaming import research_event_stream

log = get_logger(__name__)
router = APIRouter()


def _active_provider() -> str:
    s = get_settings()
    if s.search_provider == "tavily" or (s.search_provider == "auto" and s.tavily_api_key):
        return "tavily"
    return "ddg"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        reasoning_model=s.reasoning_model,
        extract_model=s.extract_model,
        embedding_model=s.embedding_model,
        search_provider=_active_provider(),
    )


async def _run_job(job_id: str, question: str, max_sources: int | None) -> None:
    await store.set_status(job_id, "running")
    try:
        report = await asyncio.to_thread(run_research, question, max_sources)
        await store.complete(job_id, report)
    except Exception as exc:  # noqa: BLE001 - surface as job error, not a 500
        log.exception("research job %s failed", job_id)
        await store.fail(job_id, f"{type(exc).__name__}: {exc}")


@router.post("/research", response_model=ResearchAccepted, status_code=202)
async def create_research(req: ResearchRequest, background_tasks: BackgroundTasks) -> ResearchAccepted:
    job = await store.create()
    background_tasks.add_task(_run_job, job.job_id, req.question, req.max_sources)
    return ResearchAccepted(job_id=job.job_id, status="queued")


@router.get("/research/stream")
async def research_stream(
    question: str = Query(..., min_length=3, max_length=2000),
    max_sources: int | None = Query(None, ge=1, le=15),
) -> StreamingResponse:
    """Live Server-Sent Events stream of the agent flow (one event per node).

    Defined before /research/{job_id} so the literal 'stream' path is not captured
    by the {job_id} path parameter.
    """
    return StreamingResponse(
        research_event_stream(question, max_sources),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/research/{job_id}", response_model=ResearchResult)
async def get_research(job_id: str) -> ResearchResult:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return ResearchResult(job_id=job.job_id, status=job.status, report=job.report, error=job.error)


@router.post("/research/sync", response_model=ResearchResult)
async def research_sync(req: ResearchRequest) -> ResearchResult:
    timeout = get_settings().request_timeout_s * 4
    try:
        report = await asyncio.wait_for(
            asyncio.to_thread(run_research, req.question, req.max_sources), timeout=timeout
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="research timed out") from exc
    return ResearchResult(job_id="sync", status="done", report=report)


# --- Run history --------------------------------------------------------------


@router.get("/history")
async def list_history() -> list[dict]:
    return history.list()


@router.get("/history/{entry_id}")
async def get_history(entry_id: str):
    report = history.get(entry_id)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown history id")
    return report


# --- Evals dashboard ----------------------------------------------------------


@router.get("/evals/last")
async def evals_last():
    return {"results": load_last()}


@router.post("/evals/run", status_code=202)
async def evals_run(
    background_tasks: BackgroundTasks,
    suite: str = Query("all", pattern="^(all|retrieval|extraction|citation)$"),
    use_judge: bool = Query(True),
) -> dict:
    job_id = await eval_jobs.create()
    background_tasks.add_task(eval_jobs.run, job_id, suite, use_judge)
    return {"job_id": job_id, "status": "queued"}


@router.get("/evals/{job_id}")
async def evals_status(job_id: str) -> dict:
    job = await eval_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown eval job")
    return job
