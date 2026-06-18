"""Extractor node — structured output. Delegates to ``extraction.extract_all`` (strict
JSON + Reflection retry loop) and records the aggregate token usage as one span."""

from __future__ import annotations

from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node


@traced_node("extractor")
def extractor_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    chunks = state.get("retrieved_chunks", [])
    sub_questions = state.get("sub_questions", [])
    stats = deps.extract_all_fn(deps.extract_llm, chunks, sub_questions, deps.settings)

    usage.set(
        model=stats.model,
        tokens_in=stats.tokens_in,
        tokens_out=stats.tokens_out,
        detail=f"{len(stats.findings)} findings, valid_rate={stats.valid_rate}",
    )
    return {"findings": stats.findings, "extraction_valid_rate": stats.valid_rate}
