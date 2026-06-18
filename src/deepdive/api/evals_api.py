"""Bridge from the API to the top-level ``evals`` package for the dashboard.

The ``evals`` package lives at the repo root (a graded learning artifact), so we add
the repo root to ``sys.path`` before importing it. Provides:
  * ``run_eval_suites`` — run one/all suites and return the same dicts the CLI prints.
  * ``load_last`` — read a previously written ``eval_results.json`` (instant dashboard).
  * an async job store so ``POST /evals/run`` can run in the background and be polled.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from ..logging_conf import get_logger

log = get_logger(__name__)

# .../src/deepdive/api/evals_api.py -> parents[3] == repo root (where evals/ & data/ live)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _safe(fn, suite: str) -> dict:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - isolate suite failures
        return {"suite": suite, "level": "-", "n": 0, "error": f"{type(exc).__name__}: {exc}"}


def run_eval_suites(suite: str = "all", use_judge: bool = True) -> list[dict]:
    from evals import metrics_citation, metrics_extraction, metrics_retrieval

    results: list[dict] = []
    if suite in ("all", "retrieval"):
        results.append(_safe(lambda: metrics_retrieval.run(), "retrieval"))
    if suite in ("all", "extraction"):
        results.append(_safe(lambda: metrics_extraction.run(), "extraction"))
    if suite in ("all", "citation"):
        results.append(_safe(lambda: metrics_citation.run(use_judge=use_judge), "citation"))
    return results


def load_last() -> list[dict] | None:
    path = _REPO_ROOT / "eval_results.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to read eval_results.json: %s", exc)
    return None


class EvalJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> str:
        async with self._lock:
            job_id = uuid.uuid4().hex[:8]
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "results": None, "error": None}
            return job_id

    async def get(self, job_id: str) -> dict | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def run(self, job_id: str, suite: str, use_judge: bool) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"
        try:
            results = await asyncio.to_thread(run_eval_suites, suite, use_judge)
            async with self._lock:
                self._jobs[job_id].update(status="done", results=results)
        except Exception as exc:  # noqa: BLE001
            log.exception("eval job %s failed", job_id)
            async with self._lock:
                self._jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


eval_jobs = EvalJobStore()
