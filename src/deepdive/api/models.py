"""Request/response models for the API (Pydantic v2). They wrap the shared
``ResearchReport`` schema so the API and the pipeline never drift."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas.report import ResearchReport


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    max_sources: int | None = Field(default=None, ge=1, le=15)


class ResearchAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]


class ResearchResult(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    report: ResearchReport | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    reasoning_model: str
    extract_model: str
    embedding_model: str
    search_provider: str
