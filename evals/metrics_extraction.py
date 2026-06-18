"""Extraction eval (component-level, objective, offline w.r.t. the web).

Feeds fixed ``source_text`` to the extractor and checks (a) schema-valid rate — did
``with_structured_output`` yield valid JSON — and (b) answer accuracy — does any
finding's answer contain an expected substring. Needs GROQ_API_KEY (no network/search).
"""

from __future__ import annotations

import re

from deepdive.config import get_settings
from deepdive.extraction.extract import extract_from_chunk
from deepdive.llm import build_extract_llm
from deepdive.schemas.sources import Chunk

from .datasets import ExtractionCase, load_extraction


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def score_case(case: ExtractionCase, llm, settings=None) -> dict:
    settings = settings or get_settings()
    chunk = Chunk(
        source_id="S1",
        url="https://eval.local/" + case.id,
        title=case.id,
        text=case.source_text,
        sub_question_id=case.sub_question_id,
    )
    sub_questions = [{"id": case.sub_question_id, "question": case.sub_question}]
    findings, valid, _ti, _to = extract_from_chunk(
        llm, chunk, sub_questions, max_retries=settings.extract_max_retries
    )

    extracted = bool(findings)
    if not case.expected_answers:
        correct = extracted == case.must_extract
    else:
        answers_norm = [_norm(f.answer) for f in findings]
        correct = any(_norm(exp) in a for exp in case.expected_answers for a in answers_norm)

    return {
        "id": case.id,
        "schema_valid": 1 if valid else 0,
        "extracted": 1 if extracted else 0,
        "correct": 1 if correct else 0,
        "n_findings": len(findings),
    }


def run(settings=None) -> dict:
    settings = settings or get_settings()
    cases = load_extraction()
    llm = build_extract_llm(settings)
    per_case = [score_case(c, llm, settings) for c in cases]
    n = len(per_case) or 1
    return {
        "suite": "extraction",
        "level": "component",
        "n": len(per_case),
        "schema_valid_rate": round(sum(r["schema_valid"] for r in per_case) / n, 4),
        "answer_accuracy": round(sum(r["correct"] for r in per_case) / n, 4),
        "per_case": per_case,
    }
