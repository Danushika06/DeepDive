"""Retrieval eval (component-level, objective). For each question we run the live
search provider and measure how well the retrieved source domains match a gold set.
Domain-level matching is robust to live-URL churn (course Module 4 note)."""

from __future__ import annotations

from urllib.parse import urlparse

from deepdive.config import get_settings
from deepdive.retrieval.search import gather_sources, get_search_provider

from .datasets import RetrievalCase, load_retrieval


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _matches(url: str, gold_domains: list[str]) -> bool:
    d = _domain(url)
    return any(g.lower() in d or d.endswith(g.lower()) for g in gold_domains)


def score_case(case: RetrievalCase, settings=None) -> dict:
    settings = settings or get_settings()
    provider = get_search_provider(settings)
    pages = gather_sources(provider, [case.question], settings)
    urls = [p.url for p in pages]
    retrieved = len(urls)
    hits = sum(1 for u in urls if _matches(u, case.gold_domains))
    matched_gold = sum(
        1 for g in case.gold_domains if any(_matches(u, [g]) for u in urls)
    )
    precision = hits / retrieved if retrieved else 0.0
    recall = matched_gold / len(case.gold_domains) if case.gold_domains else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "id": case.id,
        "retrieved": retrieved,
        "hits": hits,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit_at_k": 1 if hits > 0 else 0,
    }


def run(settings=None) -> dict:
    cases = load_retrieval()
    per_case = [score_case(c, settings) for c in cases]
    n = len(per_case) or 1
    agg = {
        "suite": "retrieval",
        "level": "component",
        "n": len(per_case),
        "precision": round(sum(r["precision"] for r in per_case) / n, 4),
        "recall": round(sum(r["recall"] for r in per_case) / n, 4),
        "f1": round(sum(r["f1"] for r in per_case) / n, 4),
        "hit_rate@k": round(sum(r["hit_at_k"] for r in per_case) / n, 4),
        "per_case": per_case,
    }
    return agg
