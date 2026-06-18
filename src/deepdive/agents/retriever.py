"""Retriever node — RAG. Builds an ephemeral FAISS index from the fetched sources and
retrieves the top-k chunks per sub-question. On a reflection redo (``reflection_retries``
> 0) it widens k to pull deeper into the same index for the under-covered sub-questions."""

from __future__ import annotations

import re

from ..schemas.sources import Chunk
from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, str]] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = (c.source_id, re.sub(r"\s+", " ", c.text[:200]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


@traced_node("retriever")
def retriever_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    settings = deps.settings
    sources = state.get("sources", [])
    index = deps.build_index_fn(sources, settings)

    sub_questions = state.get("sub_questions", [])
    k = settings.top_k * (1 + state.get("reflection_retries", 0))
    chunks: list[Chunk] = []
    if index is not None:
        for sq in sub_questions:
            chunks.extend(index.retrieve(sq["question"], k=k, sub_question_id=sq["id"]))
    chunks = _dedupe_chunks(chunks)
    n_unique = len({c.source_id for c in chunks})

    usage.set(detail=f"{len(chunks)} chunks from {n_unique} sources (k={k})")
    return {"retrieved_chunks": chunks, "n_unique_sources": n_unique}
