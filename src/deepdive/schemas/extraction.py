"""Strict-JSON schemas for the LLM agents (planner, extractor, critic, eval judge).

These are the "strict JSON schema" half of the project's RAG + structured-extraction
pipeline. The planner/extractor/critic all return one of these via
``ChatGroq.with_structured_output(...)``, and the extraction loop re-prompts on any
Pydantic ``ValidationError`` (Reflection-with-external-feedback, course Module 2.5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Planner output (Planning pattern)
# ---------------------------------------------------------------------------


class SubQuestionPlan(BaseModel):
    sub_question_id: str = Field(description="Short stable id, e.g. 'sq1'.")
    sub_question: str = Field(description="One focused sub-question.")
    queries: list[str] = Field(
        description="2-3 concrete web-search queries that would answer this sub-question."
    )


class QueryPlan(BaseModel):
    """The planner decomposes the research question into this plan."""

    sub_questions: list[SubQuestionPlan] = Field(description="3-6 sub-questions with queries.")


# ---------------------------------------------------------------------------
# Extraction output (structured output / Tool use)
# ---------------------------------------------------------------------------


class RawFinding(BaseModel):
    """What the LLM extracts from ONE chunk. No source fields here — we attach those
    ourselves afterward so the model cannot fabricate a citation target."""

    sub_question_id: str = Field(description="Which sub-question this finding answers.")
    claim: str = Field(description="One-sentence factual claim.")
    answer: str = Field(description="Direct answer/value for the sub-question.")
    supporting_quote: str = Field(
        description="VERBATIM sentence(s) copied from the provided text that support the claim."
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RawExtraction(BaseModel):
    findings: list[RawFinding] = Field(default_factory=list)


class FindingRecord(BaseModel):
    """A RawFinding enriched with provenance (filled in by our pipeline)."""

    sub_question_id: str
    claim: str
    answer: str
    supporting_quote: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_id: str = ""
    url: str = ""
    title: str = ""

    @classmethod
    def from_raw(cls, raw: RawFinding, *, source_id: str, url: str, title: str) -> FindingRecord:
        return cls(
            sub_question_id=raw.sub_question_id,
            claim=raw.claim,
            answer=raw.answer,
            supporting_quote=raw.supporting_quote,
            confidence=raw.confidence,
            source_id=source_id,
            url=url,
            title=title,
        )


# ---------------------------------------------------------------------------
# Conflict detection output (Reflection pattern)
# ---------------------------------------------------------------------------


class Conflict(BaseModel):
    sub_question_id: str
    topic: str
    conflict_type: Literal["numeric", "date", "categorical", "contradiction"]
    summary: str = Field(description="Plain-language description of how the sources disagree.")
    source_ids: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


class ConflictReport(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)


class CoverageVerdict(BaseModel):
    """Critic's judgement on whether evidence is sufficient (drives the reflect loop)."""

    sufficient: bool = Field(description="True if the evidence adequately covers all sub-questions.")
    missing_sub_question_ids: list[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# LLM-as-judge output for the citation-faithfulness eval (Module 4)
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Five binary (0/1) rubric criteria, summed to 0-5 — NOT a raw 1-5 Likert score
    (course Module 2.4/4: rubric-based grading beats single scores; avoids position bias)."""

    quote_present: int = Field(ge=0, le=1, description="Is the quote actually in the source text?")
    quote_supports_claim: int = Field(ge=0, le=1, description="Does the quote support the claim?")
    no_overreach: int = Field(ge=0, le=1, description="Claim asserts no more than the source warrants?")
    correct_attribution: int = Field(ge=0, le=1, description="Cited source is the one with the quote?")
    no_fabrication: int = Field(ge=0, le=1, description="Nothing invented relative to the source?")
    rationale: str = ""

    @property
    def score(self) -> int:
        return (
            self.quote_present
            + self.quote_supports_claim
            + self.no_overreach
            + self.correct_attribution
            + self.no_fabrication
        )

    @property
    def faithful(self) -> bool:
        return self.score >= 4
