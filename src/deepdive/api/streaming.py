"""Server-Sent Events stream of a research run.

Runs the compiled LangGraph with ``stream_mode="values"`` and emits one SSE ``step``
event per node execution (carrying that node's trace span), then a final ``done`` event
with the assembled report. The generator is synchronous; Starlette runs it in a
threadpool, so the blocking LLM/search work never stalls the event loop.

SSE event types: ``start`` | ``step`` | ``done`` | ``error``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from ..agents.deps import build_default_deps
from ..agents.graph import build_graph
from ..agents.state import initial_state
from ..config import get_settings
from ..logging_conf import get_logger
from ..pipeline import _assemble_report
from .history import history

log = get_logger(__name__)

# The canonical node order, sent up-front so the UI can pre-render the flow graph.
GRAPH_NODES = ["planner", "searcher", "retriever", "extractor", "critic", "synthesizer"]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def research_event_stream(question: str, max_sources: int | None = None) -> Iterator[str]:
    settings = get_settings()
    if max_sources is not None:
        settings = settings.model_copy(update={"max_sources": max_sources})

    try:
        deps = build_default_deps(settings)
    except Exception as exc:  # noqa: BLE001 - e.g. missing GROQ_API_KEY
        yield _sse("error", {"message": str(exc)})
        return

    app = build_graph(deps)
    state0 = initial_state(question, settings.runtime_config())
    yield _sse("start", {"question": question, "nodes": GRAPH_NODES})

    seen = 0
    final_state: dict | None = None
    try:
        for snapshot in app.stream(
            state0, stream_mode="values", config={"recursion_limit": settings.recursion_limit}
        ):
            final_state = snapshot
            spans = snapshot.get("trace", [])
            while seen < len(spans):
                span = spans[seen]
                seen += 1
                yield _sse(
                    "step",
                    {
                        "index": seen,
                        "span": span.model_dump(),
                        "counts": {
                            "sources": len(snapshot.get("sources", [])),
                            "chunks": len(snapshot.get("retrieved_chunks", [])),
                            "findings": len(snapshot.get("findings", [])),
                            "conflicts": len(snapshot.get("conflicts", [])),
                        },
                    },
                )
    except Exception as exc:  # noqa: BLE001 - surface as an SSE error event
        log.exception("streaming research failed")
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    if final_state is None:
        yield _sse("error", {"message": "no result produced"})
        return

    report = _assemble_report(question, final_state)
    run_id = history.add(question, report)
    yield _sse("done", {"run_id": run_id, "report": report.model_dump(mode="json")})
