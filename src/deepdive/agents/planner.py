"""Planner node — Planning pattern (course Module 5) + ReAct inside one node.

Reason: decompose the research question into focused sub-questions.
Act: emit concrete web-search queries (as a structured ``QueryPlan``).
Observe: on a re-plan (``plan_iterations`` > 0, triggered by thin retrieval), it is
told how few sources the last attempt found and asked to broaden the queries.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import usage_from_message
from ..logging_conf import get_logger
from ..schemas.extraction import QueryPlan
from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node

log = get_logger(__name__)

PLANNER_SYSTEM = (
    "You are a research planner. Decompose the user's research question into 3-6 focused, "
    "non-overlapping sub-questions that together fully answer it. For each sub-question, "
    "propose 2-3 specific web-search queries. Prefer sub-questions that can be answered with "
    "evidence from independent sources."
)


def _fallback_plan(question: str) -> QueryPlan:
    from ..schemas.extraction import SubQuestionPlan

    return QueryPlan(
        sub_questions=[
            SubQuestionPlan(sub_question_id="sq1", sub_question=question, queries=[question])
        ]
    )


@traced_node("planner")
def planner_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    question = state["research_question"]
    plan_iter = state.get("plan_iterations", 0)

    human = f"Research question:\n{question}\n\n"
    if plan_iter > 0:
        n = state.get("n_unique_sources", 0)
        human += (
            f"NOTE: a previous attempt retrieved only {n} usable source(s). Produce BROADER, "
            "alternative queries (different phrasings, synonyms, adjacent topics) to surface more sources.\n\n"
        )
    human += "Return a structured query plan."

    structured = deps.reasoning_llm.with_structured_output(QueryPlan, include_raw=True)
    ti = to = 0
    plan: QueryPlan | None = None
    try:
        result = structured.invoke([SystemMessage(PLANNER_SYSTEM), HumanMessage(human)])
        ti, to = usage_from_message(result.get("raw"))
        plan = result.get("parsed")
    except Exception as exc:  # noqa: BLE001 - fall back to a single-question plan
        log.warning("planner structured call failed, using fallback: %s", exc)
    plan = plan or _fallback_plan(question)

    # Canonicalize ids to sq1..sqN so downstream nodes have stable, unique keys.
    sub_questions: list[dict] = []
    query_plan: list[dict] = []
    queries: list[str] = []
    for i, sq in enumerate(plan.sub_questions, start=1):
        sid = f"sq{i}"
        sub_questions.append({"id": sid, "question": sq.sub_question})
        qs = [q for q in sq.queries if q.strip()] or [sq.sub_question]
        query_plan.append({"id": sid, "question": sq.sub_question, "queries": qs})
        queries.extend(qs)

    usage.set(
        model=deps.settings.reasoning_model,
        tokens_in=ti,
        tokens_out=to,
        detail=f"{len(sub_questions)} sub-questions, {len(queries)} queries (iter {plan_iter + 1})",
    )
    return {
        "sub_questions": sub_questions,
        "query_plan": query_plan,
        "queries": queries,
        "plan_iterations": plan_iter + 1,
        "route_reason": None,
    }
