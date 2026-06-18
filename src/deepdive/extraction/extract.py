"""Strict-JSON extraction with a Reflection-with-external-feedback retry loop.

For each retrieved chunk we ask the (cheap) extractor model to return a ``RawExtraction``.
We use Groq **JSON mode** (``response_format={"type":"json_object"}``) rather than
function-calling: the small ``llama-3.1-8b-instant`` model frequently trips Groq's
server-side ``tool_use_failed`` (HTTP 400) when emitting tool calls, even though the JSON
it produces is valid. JSON mode avoids the tool-calling path entirely.

Two external-feedback signals drive retries (course Module 2.5 — reflection is strongest
with NEW information, not re-reading):

  1. Schema validity — if the JSON fails Pydantic validation, the error is fed back.
  2. Verbatim quotes — every ``supporting_quote`` must actually appear in the chunk.
     A fabricated quote is fed back; if still bad on the last attempt the offending
     finding is dropped so we never emit an unfaithful citation.

Every API/parse failure is isolated per chunk, so one bad chunk never zeroes the batch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..llm import usage_from_message
from ..logging_conf import get_logger
from ..schemas.extraction import FindingRecord, RawExtraction
from ..schemas.sources import Chunk

log = get_logger(__name__)

EXTRACT_SYSTEM = (
    "You are a meticulous research extraction agent. You are given ONE text excerpt "
    "from a single web source, plus a list of sub-questions. Extract only factual claims "
    "that the excerpt DIRECTLY supports. For each claim, copy a VERBATIM supporting quote "
    "from the excerpt (exact words). If the excerpt answers none of the sub-questions, "
    "return an empty findings list. Never invent information that is not in the excerpt."
)

SCHEMA_HINT = (
    "Respond ONLY with a JSON object of exactly this shape (replace every <...> placeholder):\n"
    '{"findings": [{"sub_question_id": "<id such as sq1>", '
    '"claim": "<one-sentence factual claim>", "answer": "<the direct answer>", '
    '"supporting_quote": "<text copied verbatim from the excerpt>", '
    '"confidence": <number between 0 and 1>}]}\n'
    'Use the sub_question_id values shown in brackets. Return {"findings": []} if the '
    "excerpt answers none of the sub-questions."
)


@dataclass
class ExtractStats:
    findings: list[FindingRecord] = field(default_factory=list)
    n_chunks: int = 0
    n_valid_chunks: int = 0
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def valid_rate(self) -> float | None:
        return round(self.n_valid_chunks / self.n_chunks, 4) if self.n_chunks else None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_is_grounded(quote: str, source_text: str) -> bool:
    """True if the quote is verbatim (normalized substring) or strongly overlaps."""
    if not quote.strip():
        return False
    nq, ns = _normalize(quote), _normalize(source_text)
    if nq in ns:
        return True
    q_tokens = set(nq.split())
    if not q_tokens:
        return False
    overlap = len(q_tokens & set(ns.split())) / len(q_tokens)
    return overlap >= 0.6


def _build_user_prompt(chunk: Chunk, sub_questions: list[dict]) -> str:
    sq_lines = "\n".join(f"- [{sq['id']}] {sq['question']}" for sq in sub_questions)
    hint = (
        f"\n(This excerpt was retrieved for sub-question {chunk.sub_question_id}.)"
        if chunk.sub_question_id
        else ""
    )
    return (
        f"SUB-QUESTIONS:\n{sq_lines}\n\n"
        f"SOURCE EXCERPT (source_id={chunk.source_id}, title={chunk.title!r}):{hint}\n"
        f'"""\n{chunk.text}\n"""\n\n'
        f"{SCHEMA_HINT}"
    )


def _parse(content: str) -> RawExtraction:
    """Validate JSON content into RawExtraction, salvaging the outermost {...} if needed."""
    try:
        return RawExtraction.model_validate_json(content)
    except ValidationError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            return RawExtraction.model_validate_json(content[start : end + 1])
        raise


def extract_from_chunk(
    llm,
    chunk: Chunk,
    sub_questions: list[dict],
    max_retries: int = 2,
) -> tuple[list[FindingRecord], bool, int, int]:
    """Extract grounded findings from one chunk. Returns
    (findings, valid, tokens_in, tokens_out). Never raises."""
    json_llm = llm.bind(response_format={"type": "json_object"})
    messages: list = [
        SystemMessage(EXTRACT_SYSTEM),
        HumanMessage(_build_user_prompt(chunk, sub_questions)),
    ]
    tokens_in = tokens_out = 0
    valid = False

    for attempt in range(max_retries + 1):
        try:
            msg = json_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - API hiccup: skip this chunk gracefully
            log.warning("extraction call failed for %s: %s", chunk.source_id, exc)
            return [], valid, tokens_in, tokens_out

        ti, to = usage_from_message(msg)
        tokens_in += ti or 0
        tokens_out += to or 0
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        try:
            parsed = _parse(content)
        except ValidationError as exc:
            if attempt < max_retries:
                messages.append(msg)
                messages.append(
                    HumanMessage(f"Your JSON failed validation: {exc}. Return corrected JSON only.")
                )
                continue
            return [], False, tokens_in, tokens_out

        valid = True
        grounded, ungrounded = [], []
        for raw in parsed.findings:
            (grounded if _quote_is_grounded(raw.supporting_quote, chunk.text) else ungrounded).append(raw)

        if ungrounded and attempt < max_retries:
            messages.append(msg)
            bad = "; ".join(f'"{f.supporting_quote[:80]}"' for f in ungrounded)
            messages.append(
                HumanMessage(
                    f"These supporting_quote values are NOT verbatim from the excerpt: {bad}. "
                    "Re-extract using only quotes copied exactly from the excerpt; drop claims you cannot quote."
                )
            )
            continue

        findings = [
            FindingRecord.from_raw(raw, source_id=chunk.source_id, url=chunk.url, title=chunk.title)
            for raw in grounded
        ]
        return findings, valid, tokens_in, tokens_out

    return [], valid, tokens_in, tokens_out


def _dedupe(findings: list[FindingRecord]) -> list[FindingRecord]:
    seen: set[tuple[str, str]] = set()
    out: list[FindingRecord] = []
    for f in findings:
        key = (_normalize(f.claim), f.source_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def extract_all(
    llm,
    chunks: list[Chunk],
    sub_questions: list[dict],
    settings: Settings | None = None,
) -> ExtractStats:
    """Extract findings across all chunks. ``sub_questions`` is a list of
    ``{'id': 'sq1', 'question': '...'}`` dicts. Per-chunk failures are isolated."""
    settings = settings or get_settings()
    # Keep the best-scoring chunks (lower FAISS L2 distance = closer) to bound cost/latency.
    if settings.max_extract_chunks and len(chunks) > settings.max_extract_chunks:
        chunks = sorted(chunks, key=lambda c: c.score)[: settings.max_extract_chunks]
    stats = ExtractStats(model=settings.extract_model)
    for chunk in chunks:
        stats.n_chunks += 1
        try:
            findings, valid, ti, to = extract_from_chunk(
                llm, chunk, sub_questions, max_retries=settings.extract_max_retries
            )
        except Exception as exc:  # noqa: BLE001 - belt-and-suspenders; never abort the batch
            log.warning("chunk %s extraction errored: %s", chunk.source_id, exc)
            continue
        stats.tokens_in += ti
        stats.tokens_out += to
        if valid:
            stats.n_valid_chunks += 1
        stats.findings.extend(findings)
    stats.findings = _dedupe(stats.findings)
    log.info(
        "Extraction: %d findings from %d/%d valid chunks.",
        len(stats.findings), stats.n_valid_chunks, stats.n_chunks,
    )
    return stats
