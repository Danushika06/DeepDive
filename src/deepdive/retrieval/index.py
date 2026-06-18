"""Chunk -> embed -> FAISS index -> retrieve (RAG).

Embeddings are LOCAL (sentence-transformers via langchain-huggingface) because Groq
has no embeddings endpoint. The index is built fresh per research run from the live
sources, used, and discarded (ephemeral), so there is nothing to persist or invalidate.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings, get_settings
from ..logging_conf import get_logger
from ..schemas.sources import Chunk, FetchedPage

log = get_logger(__name__)


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> HuggingFaceEmbeddings:
    """Load the embedding model once per model name (lazy singleton)."""
    log.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},  # cosine-like behavior
    )


def _split(pages: list[FetchedPage], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: list[Document] = []
    for page in pages:
        if not (page.ok and page.text.strip()):
            continue
        for idx, piece in enumerate(splitter.split_text(page.text)):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source_id": page.source_id,
                        "url": page.url,
                        "title": page.title,
                        "chunk_index": idx,
                    },
                )
            )
    return docs


class VectorIndex:
    """Thin wrapper over a FAISS store that returns provenance-carrying Chunks."""

    def __init__(self, store: FAISS) -> None:
        self._store = store

    def retrieve(self, query: str, k: int, sub_question_id: str | None = None) -> list[Chunk]:
        hits = self._store.similarity_search_with_score(query, k=k)
        chunks: list[Chunk] = []
        for doc, score in hits:
            m = doc.metadata
            chunks.append(
                Chunk(
                    source_id=m.get("source_id", ""),
                    url=m.get("url", ""),
                    title=m.get("title", ""),
                    text=doc.page_content,
                    score=float(score),
                    sub_question_id=sub_question_id,
                )
            )
        return chunks


def build_index(pages: list[FetchedPage], settings: Settings | None = None) -> VectorIndex | None:
    """Build an in-memory FAISS index from fetched pages. Returns None if there is
    no usable text (e.g. every fetch failed)."""
    settings = settings or get_settings()
    docs = _split(pages, settings.chunk_size, settings.chunk_overlap)
    if not docs:
        log.warning("No chunks to index (no usable source text).")
        return None
    embedder = get_embedder(settings.embedding_model)
    store = FAISS.from_documents(docs, embedder)
    log.info("Indexed %d chunks from %d sources.", len(docs), len({d.metadata['source_id'] for d in docs}))
    return VectorIndex(store)
