"""Graph routing + loop-bound tests (no network, no LLM)."""

from __future__ import annotations

from conftest import (
    chunk,
    finding,
    make_deps,
    make_settings,
    source,
    two_subq_plan,
)
from deepdive.agents.graph import build_graph
from deepdive.agents.state import initial_state


def _run(deps, settings):
    app = build_graph(deps)
    state0 = initial_state("What is X and Y?", settings.runtime_config())
    return app.invoke(state0, config={"recursion_limit": settings.recursion_limit})


def test_happy_path_single_pass():
    settings = make_settings(min_sources_per_subq=1)
    deps = make_deps(
        settings=settings,
        plan=two_subq_plan(),
        sources=[source("S1"), source("S2")],
        chunks_by_subq={"sq1": [chunk("S1", "sq1")], "sq2": [chunk("S2", "sq2")]},
        findings=[finding("S1", "sq1"), finding("S2", "sq2")],  # both sub-questions covered
    )
    final = _run(deps, settings)

    assert final["plan_iterations"] == 1  # no replan
    assert final["coverage_sufficient"] is True
    assert "report_markdown" in final and final["report_markdown"]
    assert len(final["citations"]) == 2
    nodes = [s.node for s in final["trace"]]
    assert nodes == ["planner", "searcher", "retriever", "extractor", "critic", "synthesizer"]


def test_replan_loop_is_bounded():
    # Require many sources per sub-question so retrieval is always "thin" -> replan
    # until plan_iterations hits max_plan_iters, then proceed.
    settings = make_settings(min_sources_per_subq=5, max_plan_iters=2)
    deps = make_deps(
        settings=settings,
        plan=two_subq_plan(),
        sources=[source("S1")],  # only 1 unique source < required
        chunks_by_subq={"sq1": [chunk("S1", "sq1")], "sq2": [chunk("S1", "sq2")]},
        findings=[finding("S1", "sq1"), finding("S1", "sq2")],
    )
    final = _run(deps, settings)

    assert final["plan_iterations"] == 2  # bounded, did not loop forever
    assert final["report_markdown"]


def test_reflection_loop_is_bounded():
    # Findings cover only sq1, never sq2 -> coverage gap -> redo once, then write.
    settings = make_settings(min_sources_per_subq=1, max_reflection_iters=1)
    deps = make_deps(
        settings=settings,
        plan=two_subq_plan(),
        sources=[source("S1"), source("S2")],
        chunks_by_subq={"sq1": [chunk("S1", "sq1")], "sq2": [chunk("S2", "sq2")]},
        findings=[finding("S1", "sq1")],  # sq2 has zero findings -> gap
    )
    final = _run(deps, settings)

    assert final["coverage_sufficient"] is False
    assert final["reflection_retries"] == 2  # incremented twice, bounded by router
    # retriever ran more than once because of the redo loop
    assert sum(1 for s in final["trace"] if s.node == "retriever") >= 2
