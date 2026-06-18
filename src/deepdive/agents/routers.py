"""Conditional-edge routers (pure functions of state -> next-node key).

They implement the two bounded loops. Loop bounding = the course's ``max_turns`` idea:
domain counters give a graceful exit (proceed with a caveat) before LangGraph's hard
``recursion_limit`` backstop ever trips.
"""

from __future__ import annotations

from typing import Literal

from .state import DeepDiveState


def route_after_retrieval(state: DeepDiveState) -> Literal["replan", "extract"]:
    """Replan (broaden queries) if retrieval is thin and the plan-iteration budget allows."""
    cfg = state.get("config", {})
    sub_questions = state.get("sub_questions", [])
    n_unique = state.get("n_unique_sources", 0)
    need = cfg.get("min_sources_per_subq", 2) * max(1, len(sub_questions))
    thin = n_unique < need
    if thin and state.get("plan_iterations", 0) < cfg.get("max_plan_iters", 2):
        return "replan"
    return "extract"


def route_after_reflection(state: DeepDiveState) -> Literal["redo", "write"]:
    """Redo retrieval (wider k) if the critic found an evidence gap and budget allows."""
    cfg = state.get("config", {})
    if not state.get("coverage_sufficient", True) and state.get("reflection_retries", 0) <= cfg.get(
        "max_reflection_iters", 1
    ):
        return "redo"
    return "write"
