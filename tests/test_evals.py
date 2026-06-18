"""Offline eval tests (no LLM / no network)."""

from __future__ import annotations

from evals import datasets, metrics_citation
from evals.metrics_retrieval import _domain, _matches


def test_cohen_kappa_perfect_agreement():
    assert metrics_citation._cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_cohen_kappa_no_better_than_chance():
    # all-agree predictions vs balanced labels -> kappa near 0
    assert metrics_citation._cohen_kappa([1, 1, 1, 1], [1, 0, 1, 0]) == 0.0


def test_substring_baseline_offline():
    res = metrics_citation.run(use_judge=False)
    assert res["suite"] == "citation"
    assert 0.0 <= res["substring_faithful_rate"] <= 1.0
    # The substring baseline agrees with humans on most cases, but NOT all — overreach
    # cases (quote present yet claim unfaithful) are exactly why we also use an LLM judge.
    assert res["substring_vs_human_acc"] >= 0.6


def test_domain_matching_helpers():
    assert _domain("https://www.nih.gov/health") == "nih.gov"
    assert _matches("https://en.wikipedia.org/wiki/Statin", ["wikipedia.org"]) is True
    assert _matches("https://example.com/x", ["wikipedia.org"]) is False


def test_datasets_load_and_validate():
    assert len(datasets.load_extraction()) >= 10
    assert len(datasets.load_citation()) >= 10
    assert len(datasets.load_retrieval()) >= 10
