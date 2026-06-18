"""Schema round-trip + helper tests."""

from __future__ import annotations

from deepdive.schemas.extraction import FindingRecord, JudgeVerdict, RawFinding
from deepdive.schemas.trace import RunTrace, Span


def test_judge_score_and_faithful():
    v = JudgeVerdict(
        quote_present=1, quote_supports_claim=1, no_overreach=1, correct_attribution=1, no_fabrication=0
    )
    assert v.score == 4
    assert v.faithful is True

    v2 = JudgeVerdict(
        quote_present=1, quote_supports_claim=0, no_overreach=1, correct_attribution=1, no_fabrication=0
    )
    assert v2.score == 3
    assert v2.faithful is False


def test_finding_from_raw_attaches_provenance():
    raw = RawFinding(sub_question_id="sq1", claim="c", answer="a", supporting_quote="q", confidence=0.9)
    f = FindingRecord.from_raw(raw, source_id="S1", url="https://ex.com", title="T")
    assert f.source_id == "S1"
    assert f.url == "https://ex.com"
    assert f.confidence == 0.9


def test_runtrace_aggregates_spans():
    spans = [
        Span(node="a", latency_ms=10, tokens_in=5, tokens_out=2, est_cost_usd=0.001),
        Span(node="b", latency_ms=20, tokens_in=3, tokens_out=1, est_cost_usd=0.002),
    ]
    t = RunTrace.from_spans(spans)
    assert t.total_latency_ms == 30
    assert t.total_tokens_in == 8
    assert t.total_tokens_out == 3
    assert round(t.total_cost_usd, 3) == 0.003
