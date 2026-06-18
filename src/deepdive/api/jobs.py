"""In-memory async job store for the background ``/research`` endpoint.

Deliberately simple for the MVP — a dict behind an ``asyncio.Lock``. The plan notes
this is the swap point for Redis/RQ/Celery in a multi-worker production deployment.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from ..schemas.report import ResearchReport


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | running | done | error
    report: ResearchReport | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        async with self._lock:
            job = Job(job_id=uuid.uuid4().hex[:12])
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def set_status(self, job_id: str, status: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = status

    async def complete(self, job_id: str, report: ResearchReport) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "done"
                self._jobs[job_id].report = report

    async def fail(self, job_id: str, error: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "error"
                self._jobs[job_id].error = error


# Module-level singleton shared by the routes.
store = JobStore()
