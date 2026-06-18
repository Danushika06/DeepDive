"""Central configuration (pydantic-settings).

All tunables live here so the rest of the code never reads ``os.environ`` directly.
Model IDs in particular live here so a Groq model deprecation is a one-line change
(course Module 4: "model IDs live in config").
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Credentials -------------------------------------------------------
    # Optional at import time so tests / offline eval datasets work without keys.
    # llm.py raises a clear error if a real LLM call is attempted without a key.
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    # --- Models ------------------------------------------------------------
    reasoning_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_REASONING_MODEL")
    extract_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_EXTRACT_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    temperature_reasoning: float = 0.2
    temperature_extract: float = 0.0
    request_timeout_s: int = 90

    # --- Retrieval / RAG ---------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    max_sources: int = 6
    min_sources_per_subq: int = 2
    # Soft cap on chunks sent to the (per-chunk) extractor — bounds worst-case
    # latency/cost (course Module 4.5). Keeps the best-scoring chunks.
    max_extract_chunks: int = 20

    # --- Web search --------------------------------------------------------
    # auto = Tavily when TAVILY_API_KEY present, else keyless DuckDuckGo.
    search_provider: str = Field(default="auto", alias="SEARCH_PROVIDER")
    results_per_query: int = 4
    fetch_timeout_s: float = 15.0

    # --- Loop bounds (the course's `max_turns` analogue) -------------------
    max_plan_iters: int = 2
    max_reflection_iters: int = 1
    recursion_limit: int = 40
    extract_max_retries: int = 2

    # --- App ---------------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def runtime_config(self) -> dict:
        """The subset threaded into the LangGraph state under ``state['config']``."""
        return {
            "top_k": self.top_k,
            "max_sources": self.max_sources,
            "min_sources_per_subq": self.min_sources_per_subq,
            "results_per_query": self.results_per_query,
            "max_plan_iters": self.max_plan_iters,
            "max_reflection_iters": self.max_reflection_iters,
            "extract_max_retries": self.extract_max_retries,
        }


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor."""
    return Settings()
