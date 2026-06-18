"""Groq LLM client factories.

Two handles, per the plan:
  * reasoning  -> llama-3.3-70b-versatile  (planner, critic, synthesizer, judge)
  * extract    -> llama-3.1-8b-instant     (high-volume, cheaper extraction calls)

Groq has no embeddings endpoint, so embeddings live in ``retrieval/index.py`` (local
sentence-transformers). This module is only about the chat models.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_groq import ChatGroq

from .config import Settings, get_settings


class MissingAPIKey(RuntimeError):
    """Raised when an LLM call is attempted but GROQ_API_KEY is not set."""


def _ensure_key(settings: Settings) -> None:
    if not settings.groq_api_key:
        raise MissingAPIKey(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )
    # ChatGroq reads GROQ_API_KEY from the environment; pydantic-settings loaded it
    # from .env into the Settings object but not into os.environ — bridge it here.
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)


def build_reasoning_llm(settings: Settings | None = None) -> ChatGroq:
    settings = settings or get_settings()
    _ensure_key(settings)
    return ChatGroq(
        model=settings.reasoning_model,
        temperature=settings.temperature_reasoning,
        timeout=settings.request_timeout_s,
        max_retries=2,
    )


def build_extract_llm(settings: Settings | None = None) -> ChatGroq:
    settings = settings or get_settings()
    _ensure_key(settings)
    return ChatGroq(
        model=settings.extract_model,
        temperature=settings.temperature_extract,
        timeout=settings.request_timeout_s,
        max_retries=2,
    )


def usage_from_message(msg: Any) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from a LangChain AIMessage, tolerant of
    the two shapes ChatGroq may use across versions."""
    if msg is None:
        return None, None
    usage = getattr(msg, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return usage.get("input_tokens"), usage.get("output_tokens")
    meta = getattr(msg, "response_metadata", None) or {}
    tok = meta.get("token_usage") or meta.get("usage") or {}
    if tok:
        return tok.get("prompt_tokens"), tok.get("completion_tokens")
    return None, None
