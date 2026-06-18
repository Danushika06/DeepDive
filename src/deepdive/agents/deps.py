"""Dependency container injected into every node.

Holding the LLMs, the search provider, and the heavy pipeline functions here (rather
than importing them inside nodes) is what makes the graph testable: a unit test builds
``Deps`` with fake LLMs / a fake search provider / stub functions and asserts routing
and loop bounds with zero network or model calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..extraction.conflicts import detect_conflicts
from ..extraction.extract import extract_all
from ..retrieval.index import build_index
from ..retrieval.search import gather_sources, get_search_provider


@dataclass
class Deps:
    settings: Settings
    reasoning_llm: Any = None
    extract_llm: Any = None
    search_provider: Any = None
    # Heavy operations as swappable hooks (defaults = the real implementations).
    gather_sources_fn: Callable = field(default=gather_sources)
    build_index_fn: Callable = field(default=build_index)
    extract_all_fn: Callable = field(default=extract_all)
    detect_conflicts_fn: Callable = field(default=detect_conflicts)


def build_default_deps(settings: Settings | None = None) -> Deps:
    """Construct production Deps (real Groq clients + chosen search provider).

    Imported lazily so that ``import deepdive.agents`` does not require API keys.
    """
    from ..llm import build_extract_llm, build_reasoning_llm

    settings = settings or get_settings()
    return Deps(
        settings=settings,
        reasoning_llm=build_reasoning_llm(settings),
        extract_llm=build_extract_llm(settings),
        search_provider=get_search_provider(settings),
    )
