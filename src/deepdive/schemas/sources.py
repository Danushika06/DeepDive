"""Schemas for web sources as they flow through search -> fetch -> chunk -> retrieve.

URLs are plain ``str`` (not ``HttpUrl``) on purpose: live web URLs are messy and we
do not want pydantic validation to drop an odd-but-valid source mid-pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single hit from a SearchProvider."""

    url: str
    title: str
    snippet: str = ""
    raw_content: str | None = None  # full page text if the provider supplied it (Tavily)
    score: float | None = None
    provider: str = ""


class FetchedPage(BaseModel):
    """A source after we have its full cleaned text (directly from Tavily, or via fetch)."""

    source_id: str  # stable id, e.g. "S1" — used as the citation key
    url: str
    title: str
    text: str
    ok: bool = True
    error: str | None = None


class Chunk(BaseModel):
    """A retrieved chunk with provenance so citations stay traceable."""

    source_id: str
    url: str
    title: str
    text: str
    score: float = 0.0  # similarity score from FAISS (lower L2 distance = closer)
    sub_question_id: str | None = None
