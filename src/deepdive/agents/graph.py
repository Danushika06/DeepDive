"""Wire the nodes into a compiled LangGraph ``StateGraph``.

    START -> planner -> searcher -> retriever --?--> extractor -> critic --?--> synthesizer -> END
                ^                        |                            |
                +----- replan (thin) ---+              redo (gap) ----+--> retriever

Deps are bound into each node via a tiny single-arg closure so LangGraph sees a clean
``node(state)`` signature.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from .critic import critic_node
from .deps import Deps
from .extractor import extractor_node
from .planner import planner_node
from .retriever import retriever_node
from .routers import route_after_reflection, route_after_retrieval
from .searcher import searcher_node
from .state import DeepDiveState
from .synthesizer import synthesizer_node


def _bind(node: Callable, deps: Deps) -> Callable:
    def run(state: DeepDiveState) -> dict:
        return node(state, deps)

    return run


def build_graph(deps: Deps):
    g = StateGraph(DeepDiveState)

    g.add_node("planner", _bind(planner_node, deps))
    g.add_node("searcher", _bind(searcher_node, deps))
    g.add_node("retriever", _bind(retriever_node, deps))
    g.add_node("extractor", _bind(extractor_node, deps))
    g.add_node("critic", _bind(critic_node, deps))
    g.add_node("synthesizer", _bind(synthesizer_node, deps))

    g.add_edge(START, "planner")
    g.add_edge("planner", "searcher")
    g.add_edge("searcher", "retriever")
    g.add_conditional_edges(
        "retriever", route_after_retrieval, {"replan": "planner", "extract": "extractor"}
    )
    g.add_edge("extractor", "critic")
    g.add_conditional_edges(
        "critic", route_after_reflection, {"redo": "retriever", "write": "synthesizer"}
    )
    g.add_edge("synthesizer", END)

    return g.compile()
