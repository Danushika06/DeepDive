"""Pluggable web search (course Module 1.3: modularity — swap search engines freely).

A single ``SearchProvider`` ABC with two interchangeable implementations:
  * Tavily   — returns clean full-page content (skips the fetch step). Needs a key.
  * DuckDuckGo (ddgs) — keyless default; returns snippets + URLs that we then fetch.

``get_search_provider`` picks one from config: ``auto`` uses Tavily when a key is
present, otherwise DuckDuckGo. ``gather_sources`` is the search->fetch orchestration
that yields citation-ready :class:`FetchedPage` objects with stable source ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings
from ..logging_conf import get_logger
from ..schemas.sources import FetchedPage, SearchResult
from .fetch import fetch_and_clean

log = get_logger(__name__)

# Tavily raw_content shorter than this is treated as unusable -> fall back to fetch.
_MIN_RAW_CONTENT = 200


class SearchProvider(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        ...


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str, include_raw_content: bool = True) -> None:
        from tavily import TavilyClient  # imported lazily so DDG-only users need no dep load

        self._client = TavilyClient(api_key=api_key)
        self._include_raw = include_raw_content

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        resp = self._client.search(
            query,
            max_results=max_results,
            include_raw_content=self._include_raw,
            search_depth="advanced",
        )
        results: list[SearchResult] = []
        for r in resp.get("results", []):
            url = r.get("url", "")
            results.append(
                SearchResult(
                    url=url,
                    title=r.get("title") or url,
                    snippet=r.get("content", "") or "",
                    raw_content=r.get("raw_content"),
                    score=r.get("score"),
                    provider=self.name,
                )
            )
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    name = "ddg"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        from ddgs import DDGS  # `ddgs`, not the frozen `duckduckgo_search`

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                results.append(
                    SearchResult(
                        url=url,
                        title=r.get("title") or url,
                        snippet=r.get("body", "") or "",
                        raw_content=None,
                        provider=self.name,
                    )
                )
        return results


def get_search_provider(settings: Settings | None = None) -> SearchProvider:
    settings = settings or get_settings()
    choice = settings.search_provider.lower()
    use_tavily = choice == "tavily" or (choice == "auto" and bool(settings.tavily_api_key))
    if use_tavily:
        if not settings.tavily_api_key:
            raise ValueError("SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set.")
        log.info("Search provider: Tavily")
        return TavilySearchProvider(settings.tavily_api_key)
    log.info("Search provider: DuckDuckGo (keyless)")
    return DuckDuckGoSearchProvider()


def _normalize_url(url: str) -> str:
    return url.rstrip("/").split("#", 1)[0].lower()


def gather_sources(
    provider: SearchProvider,
    queries: list[str],
    settings: Settings | None = None,
) -> list[FetchedPage]:
    """Run every query, dedupe by URL, cap at ``max_sources``, assign stable source
    ids (S1, S2, ...), and materialize full text (from Tavily raw_content or a fetch)."""
    settings = settings or get_settings()
    seen: set[str] = set()
    hits: list[SearchResult] = []
    for q in queries:
        try:
            for r in provider.search(q, max_results=settings.results_per_query):
                if not r.url:
                    continue
                key = _normalize_url(r.url)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(r)
        except Exception as exc:  # noqa: BLE001 - a failing query shouldn't abort the run
            log.warning("search failed for query %r: %s", q, exc)
        if len(hits) >= settings.max_sources:
            break

    hits = hits[: settings.max_sources]
    pages: list[FetchedPage] = []
    for i, r in enumerate(hits, start=1):
        source_id = f"S{i}"
        if r.raw_content and len(r.raw_content) >= _MIN_RAW_CONTENT:
            pages.append(
                FetchedPage(
                    source_id=source_id, url=r.url, title=r.title,
                    text=r.raw_content.strip(), ok=True,
                )
            )
        else:
            page = fetch_and_clean(r.url, source_id=source_id, title=r.title)
            # If the fetch produced nothing, keep the snippet so the source isn't lost.
            if not page.ok and r.snippet:
                page = FetchedPage(
                    source_id=source_id, url=r.url, title=r.title,
                    text=r.snippet.strip(), ok=True, error="using snippet fallback",
                )
            pages.append(page)
    return pages
