"""The shared graph state threaded through every node.

A ``TypedDict`` (LangGraph-native). List fields that accumulate across nodes use an
``operator.add`` reducer so partial updates APPEND rather than overwrite (``trace``,
``errors``). Everything else is last-write-wins, which is correct because only one
node writes each field per step.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..schemas.extraction import Conflict, FindingRecord
from ..schemas.report import Citation
from ..schemas.sources import Chunk, FetchedPage
from ..schemas.trace import Span


class DeepDiveState(TypedDict, total=False):
    # --- inputs ---
    research_question: str
    config: dict

    # --- planning (Planning pattern) ---
    sub_questions: list[dict]  # [{"id": "sq1", "question": "..."}]
    query_plan: list[dict]  # [{"id", "question", "queries": [...]}]
    queries: list[str]

    # --- retrieval (Tool use + RAG) ---
    sources: list[FetchedPage]
    retrieved_chunks: list[Chunk]
    n_unique_sources: int

    # --- extraction (structured output) ---
    findings: list[FindingRecord]
    extraction_valid_rate: float | None

    # --- reflection (Reflection pattern) ---
    conflicts: list[Conflict]
    coverage_sufficient: bool

    # --- synthesis (Multi-agent writer) ---
    citations: list[Citation]
    report_markdown: str
    citation_faithfulness: float | None

    # --- control / loop bounds ---
    plan_iterations: int
    reflection_retries: int
    route_reason: str | None

    # --- accumulators (Module 4 tracing) ---
    trace: Annotated[list[Span], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(question: str, config: dict) -> DeepDiveState:
    return {
        "research_question": question,
        "config": config,
        "plan_iterations": 0,
        "reflection_retries": 0,
        "route_reason": None,
        "trace": [],
        "errors": [],
    }
