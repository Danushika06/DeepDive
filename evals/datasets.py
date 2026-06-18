"""Typed eval-case loaders. Each dataset is JSONL (one case per line); malformed
rows fail fast via Pydantic validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


class RetrievalCase(BaseModel):
    id: str
    question: str
    gold_domains: list[str]
    gold_urls: list[str] = Field(default_factory=list)
    k: int = 5


class ExtractionCase(BaseModel):
    id: str
    source_text: str
    sub_question: str
    sub_question_id: str = "sq1"
    expected_answers: list[str] = Field(default_factory=list)  # any substring match = correct
    must_extract: bool = True


class CitationCase(BaseModel):
    id: str
    claim: str
    supporting_quote: str
    source_text: str
    label_faithful: int  # human gold label: 1 = faithful, 0 = not


def _load(path: Path, model: type[BaseModel]) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {path}")
    cases = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(model.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path.name} line {i}: {exc}") from exc
    return cases


def load_retrieval() -> list[RetrievalCase]:
    return _load(DATA_DIR / "retrieval.jsonl", RetrievalCase)


def load_extraction() -> list[ExtractionCase]:
    return _load(DATA_DIR / "extraction.jsonl", ExtractionCase)


def load_citation() -> list[CitationCase]:
    return _load(DATA_DIR / "citation.jsonl", CitationCase)
