"""Critic node — Reflection pattern (course Module 2).

Two reflective jobs:
  1. Cross-source conflict detection (the headline deliverable: "flagging cross-source
     conflicts") via ``extraction.detect_conflicts``.
  2. A coverage check: any sub-question with zero findings is an evidence gap. If gaps
     remain (and the retry budget allows), the router sends control back to the
     retriever to widen retrieval — a reflect-then-act loop.
"""

from __future__ import annotations

from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node


@traced_node("critic")
def critic_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    settings = deps.settings
    findings = state.get("findings", [])
    sub_questions = state.get("sub_questions", [])

    conflicts, ti, to = deps.detect_conflicts_fn(deps.reasoning_llm, findings, settings)

    covered = {f.sub_question_id for f in findings}
    missing = [sq["id"] for sq in sub_questions if sq["id"] not in covered]
    sufficient = (len(missing) == 0) or (not sub_questions)

    update: dict = {"conflicts": conflicts, "coverage_sufficient": sufficient}
    if not sufficient:
        update["reflection_retries"] = state.get("reflection_retries", 0) + 1
        update["route_reason"] = f"coverage gap on {missing}"

    usage.set(
        model=settings.reasoning_model if conflicts or findings else None,
        tokens_in=ti,
        tokens_out=to,
        detail=f"{len(conflicts)} conflicts; missing sub-questions={missing}",
    )
    return update
