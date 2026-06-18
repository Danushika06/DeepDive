"""Latency + cost benchmarking (course Module 4.5: time AND price each step).

``timed_step`` is a context manager that measures wall-clock latency and (if the
node fills in token usage) estimates cost from a per-model price table, then appends
a :class:`Span` to the provided list. Nodes use it so the resulting ``RunTrace`` is
simultaneously the error-analysis trace and the cost/latency benchmark.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from .schemas.trace import Span

# Groq pricing in USD per 1M tokens: (input, output). Approximate — for the cost
# benchmark only. Refresh from https://groq.com/pricing when model IDs change.
PRICES: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.10, 0.30),
    "llama-3.1-70b-versatile": (0.59, 0.79),
}


def estimate_cost(model: str | None, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if not model or model not in PRICES:
        return None
    in_rate, out_rate = PRICES[model]
    ti = tokens_in or 0
    to = tokens_out or 0
    return round((ti * in_rate + to * out_rate) / 1_000_000, 6)


class StepUsage:
    """Mutable handle a node fills in inside a ``with timed_step(...) as usage:`` block."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.tokens_in: int | None = None
        self.tokens_out: int | None = None
        self.detail: str | None = None

    def set(
        self,
        *,
        model: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        detail: str | None = None,
    ) -> None:
        if model is not None:
            self.model = model
        if tokens_in is not None:
            self.tokens_in = tokens_in
        if tokens_out is not None:
            self.tokens_out = tokens_out
        if detail is not None:
            self.detail = detail


@contextmanager
def timed_step(spans: list[Span], node: str) -> Iterator[StepUsage]:
    """Time a step, capture token usage the body records, append a Span. Re-raises
    on error but still records a Span with the error message first."""
    usage = StepUsage()
    t0 = time.perf_counter()
    error: str | None = None
    try:
        yield usage
    except Exception as exc:  # noqa: BLE001 - we record then re-raise
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        spans.append(
            Span(
                node=node,
                model=usage.model,
                latency_ms=latency_ms,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                est_cost_usd=estimate_cost(usage.model, usage.tokens_in, usage.tokens_out),
                detail=usage.detail,
                error=error,
            )
        )
