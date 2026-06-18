"""Cross-source conflict detection (Reflection pattern).

Hybrid approach:
  * Rule-based pre-filter — cheap, no hallucination: within one sub-question, flag
    sources whose numeric values or years disagree.
  * LLM-as-judge — catches semantic contradictions rules can't, returning a structured
    ``ConflictReport``.
Findings are grouped by sub-question (claims must address the same question to conflict),
and only groups with >= 2 distinct sources are examined.
"""

from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..llm import usage_from_message
from ..logging_conf import get_logger
from ..schemas.extraction import Conflict, ConflictReport, FindingRecord

log = get_logger(__name__)

CONFLICT_SYSTEM = (
    "You are a critical research analyst. You are given several findings from DIFFERENT "
    "sources that all answer the same sub-question. Identify genuine disagreements between "
    "sources — contradictory facts, incompatible numbers/dates, or opposing conclusions. "
    "Do NOT report differences that are merely wording, scope, or detail. If sources agree, "
    "return an empty conflicts list."
)

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for m in _NUM_RE.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _years(text: str) -> set[str]:
    return set(_YEAR_RE.findall(text))


def _group_by_subq(findings: list[FindingRecord]) -> dict[str, list[FindingRecord]]:
    groups: dict[str, list[FindingRecord]] = defaultdict(list)
    for f in findings:
        groups[f.sub_question_id].append(f)
    return groups


def _rule_based(sqid: str, group: list[FindingRecord]) -> list[Conflict]:
    """Flag numeric/date disagreement across distinct sources."""
    conflicts: list[Conflict] = []
    years_by_src = {f.source_id: _years(f.answer) for f in group}
    distinct_years = {y for ys in years_by_src.values() for y in ys}
    if len({y for ys in years_by_src.values() for y in ys}) > 1 and len(years_by_src) > 1:
        # more than one distinct year mentioned across sources
        if len(distinct_years) > 1:
            conflicts.append(
                Conflict(
                    sub_question_id=sqid,
                    topic=group[0].sub_question_id,
                    conflict_type="date",
                    summary=f"Sources cite different years: {sorted(distinct_years)}.",
                    source_ids=sorted(years_by_src),
                    urls=sorted({f.url for f in group}),
                    severity="medium",
                )
            )
    # numeric disagreement: distinct numeric answers across >=2 sources
    nums_by_src = {f.source_id: _numbers(f.answer) for f in group}
    all_nums = {n for ns in nums_by_src.values() for n in ns}
    sources_with_nums = [s for s, ns in nums_by_src.items() if ns]
    if len(sources_with_nums) > 1 and len(all_nums) > 1:
        conflicts.append(
            Conflict(
                sub_question_id=sqid,
                topic=group[0].sub_question_id,
                conflict_type="numeric",
                summary=f"Sources report differing numeric values: {sorted(all_nums)}.",
                source_ids=sorted(sources_with_nums),
                urls=sorted({f.url for f in group}),
                severity="medium",
            )
        )
    return conflicts


def _llm_judge(llm, sqid: str, group: list[FindingRecord]) -> tuple[list[Conflict], int, int]:
    lines = "\n".join(
        f"- ({f.source_id}) claim: {f.claim} | answer: {f.answer}" for f in group
    )
    structured = llm.with_structured_output(ConflictReport, include_raw=True)
    result = structured.invoke(
        [
            SystemMessage(CONFLICT_SYSTEM),
            HumanMessage(
                f"Sub-question id: {sqid}\nFindings from different sources:\n{lines}\n\n"
                "Return a structured conflicts list (empty if the sources agree)."
            ),
        ]
    )
    ti, to = usage_from_message(result.get("raw"))
    report: ConflictReport | None = result.get("parsed")
    conflicts = report.conflicts if report else []
    # backfill ids the model may have left blank
    for c in conflicts:
        if not c.sub_question_id:
            c.sub_question_id = sqid
        if not c.source_ids:
            c.source_ids = sorted({f.source_id for f in group})
        if not c.urls:
            c.urls = sorted({f.url for f in group})
    return conflicts, ti or 0, to or 0


def _covered(candidate: Conflict, existing: list[Conflict]) -> bool:
    return any(
        c.sub_question_id == candidate.sub_question_id and c.conflict_type == candidate.conflict_type
        for c in existing
    )


def detect_conflicts(
    llm,
    findings: list[FindingRecord],
    settings: Settings | None = None,
) -> tuple[list[Conflict], int, int]:
    """Returns (conflicts, tokens_in, tokens_out)."""
    settings = settings or get_settings()
    conflicts: list[Conflict] = []
    tokens_in = tokens_out = 0
    for sqid, group in _group_by_subq(findings).items():
        if len({f.source_id for f in group}) < 2:
            continue
        try:
            judged, ti, to = _llm_judge(llm, sqid, group)
        except Exception as exc:  # noqa: BLE001 - skip this group, keep rule-based below
            log.warning("conflict judge failed for %s: %s", sqid, exc)
            judged, ti, to = [], 0, 0
        tokens_in += ti
        tokens_out += to
        conflicts.extend(judged)
        for cand in _rule_based(sqid, group):
            if not _covered(cand, conflicts):
                conflicts.append(cand)
    log.info("Conflict detection: %d conflicts across sub-questions.", len(conflicts))
    return conflicts, tokens_in, tokens_out
