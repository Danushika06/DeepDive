"""Searcher node — Tool use (web search). Not an LLM call; a deterministic adapter to
the active SearchProvider. Runs all planned queries, dedupes, fetches full text, and
yields citation-ready ``FetchedPage`` objects with stable source ids."""

from __future__ import annotations

from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node


@traced_node("searcher")
def searcher_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    queries = state.get("queries", [])
    pages = deps.gather_sources_fn(deps.search_provider, queries, deps.settings)
    n_ok = sum(1 for p in pages if p.ok)
    usage.set(detail=f"{n_ok}/{len(pages)} sources with usable text (provider={getattr(deps.search_provider, 'name', '?')})")
    return {"sources": pages}
