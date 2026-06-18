"""Final output schemas: the citation-backed report and its metrics."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .extraction import Conflict, FindingRecord
from .trace import RunTrace


class Citation(BaseModel):
    """A numbered reference in the final report: marker ``[n]`` -> a source."""

    marker: str  # e.g. "[1]"
    source_id: str
    url: str
    title: str


class Metrics(BaseModel):
    """In-run metrics surfaced alongside the report (the resume bullet's
    'retrieval, extraction and citation-faithfulness metrics')."""

    n_sources_retrieved: int = 0
    n_unique_sources: int = 0
    n_findings: int = 0
    n_conflicts: int = 0
    extraction_valid_rate: float | None = None  # share of chunks that yielded valid JSON
    citation_faithfulness: float | None = None  # share of [n] markers resolving to a real source


class ResearchReport(BaseModel):
    """The full result returned by the pipeline / API."""

    question: str
    sub_questions: list[str] = Field(default_factory=list)
    report_markdown: str = ""
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    findings: list[FindingRecord] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    trace: RunTrace = Field(default_factory=RunTrace)
