"""Citation-faithfulness eval (end-to-end signal).

Two graders, reported side by side:
  1. Substring baseline (objective, free): is the supporting quote actually in the source?
  2. LLM-as-judge rubric (5 binary criteria, faithful = score >= 4).

Because each case carries a human ``label_faithful``, we also report judge-vs-human
agreement (accuracy + Cohen's kappa) — the course's "evaluate your evaluator" loop.
"""

from __future__ import annotations

import re

from deepdive.config import get_settings
from deepdive.llm import build_reasoning_llm

from .datasets import CitationCase, load_citation
from .judge import judge_citation


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _substring_faithful(case: CitationCase) -> int:
    return 1 if _norm(case.supporting_quote) in _norm(case.source_text) else 0


def _cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return round((agree - pe) / (1 - pe), 4)


def run(settings=None, use_judge: bool = True) -> dict:
    settings = settings or get_settings()
    cases = load_citation()
    llm = build_reasoning_llm(settings) if use_judge else None

    per_case = []
    labels, judge_preds, substr_preds = [], [], []
    for c in cases:
        substr = _substring_faithful(c)
        row = {"id": c.id, "label": c.label_faithful, "substring_faithful": substr}
        substr_preds.append(substr)
        labels.append(c.label_faithful)
        if use_judge:
            verdict = judge_citation(llm, c.claim, c.supporting_quote, c.source_text)
            jf = 1 if verdict.faithful else 0
            row["judge_score"] = verdict.score
            row["judge_faithful"] = jf
            judge_preds.append(jf)
        per_case.append(row)

    n = len(cases) or 1
    result = {
        "suite": "citation",
        "level": "e2e",
        "n": len(cases),
        "substring_faithful_rate": round(sum(substr_preds) / n, 4),
        "substring_vs_human_acc": round(
            sum(1 for x, y in zip(substr_preds, labels, strict=False) if x == y) / n, 4
        ),
        "per_case": per_case,
    }
    if use_judge:
        result["judge_faithful_rate"] = round(sum(judge_preds) / n, 4)
        result["judge_vs_human_acc"] = round(
            sum(1 for x, y in zip(judge_preds, labels, strict=False) if x == y) / n, 4
        )
        result["judge_vs_human_kappa"] = _cohen_kappa(judge_preds, labels)
    return result
