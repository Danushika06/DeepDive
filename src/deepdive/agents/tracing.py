"""``@traced_node`` — wraps a node so it (a) times itself + records token/cost into a
Span (Module 4.5 benchmark) and (b) never crashes the whole graph: an exception is
caught, recorded on the span, and surfaced via ``state['errors']`` so the run can
finish gracefully with a partial result.

Node functions have signature ``fn(state, deps, usage) -> dict``; the wrapper presents
the LangGraph-expected ``node(state)`` after ``deps`` is bound in graph.py.
"""

from __future__ import annotations

from collections.abc import Callable

from ..benchmark import timed_step
from .deps import Deps
from .state import DeepDiveState


def traced_node(name: str) -> Callable:
    def decorator(fn: Callable[[DeepDiveState, Deps, object], dict]) -> Callable:
        def wrapper(state: DeepDiveState, deps: Deps) -> dict:
            spans: list = []
            update: dict = {}
            try:
                with timed_step(spans, name) as usage:
                    update = fn(state, deps, usage) or {}
            except Exception as exc:  # noqa: BLE001 - record + continue, don't kill the graph
                update = {"errors": [f"{name}: {type(exc).__name__}: {exc}"]}
            update["trace"] = spans  # appended via the operator.add reducer
            return update

        wrapper.__name__ = f"traced_{name}"
        return wrapper

    return decorator
