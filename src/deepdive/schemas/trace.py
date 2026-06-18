"""Tracing schemas (course Module 4: trace = whole run, span = one step).

Every graph node emits a ``Span`` that is appended into ``state['trace']``. The
``RunTrace`` aggregate is computed at the end and returned in the API response —
this is both the error-analysis trace and the latency/cost benchmark (Module 4.5).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Span(BaseModel):
    """One step's trace record."""

    node: str
    model: str | None = None
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    est_cost_usd: float | None = None
    detail: str | None = None  # short human-readable summary of what the node did
    error: str | None = None


class RunTrace(BaseModel):
    """Aggregate of all spans for one research run."""

    spans: list[Span] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0

    @classmethod
    def from_spans(cls, spans: list[Span]) -> RunTrace:
        return cls(
            spans=list(spans),
            total_latency_ms=round(sum(s.latency_ms for s in spans), 2),
            total_tokens_in=sum(s.tokens_in or 0 for s in spans),
            total_tokens_out=sum(s.tokens_out or 0 for s in spans),
            total_cost_usd=round(sum(s.est_cost_usd or 0.0 for s in spans), 6),
        )
