"""Synthesizer node — the "synthesis agent" (Multi-agent, course Module 5).

Writes the final citation-backed markdown report. Citations are assigned
deterministically (source_id -> [n]) so the References section is always correct; the
LLM only places the inline [n] markers in prose. ``citation_faithfulness`` is a cheap
post-check: the share of inline markers that resolve to a real retrieved source.
"""

from __future__ import annotations

import re

from ..llm import usage_from_message
from ..schemas.extraction import Conflict, FindingRecord
from ..schemas.report import Citation
from .deps import Deps
from .state import DeepDiveState
from .tracing import traced_node

SYNTH_SYSTEM = (
    "You are a research writer. Using ONLY the provided findings, write a clear, well-structured "
    "markdown report that answers the research question. Group related points into themed sections. "
    "Cite every factual statement inline using the [n] source markers EXACTLY as given — never invent "
    "a marker that is not in the SOURCES list. Include a '## Conflicting evidence' section that honestly "
    "summarizes the listed conflicts (write 'No material conflicts were found.' if the list is empty). "
    "Do NOT add a References section; it will be appended automatically."
)

_MARKER_RE = re.compile(r"\[(\d+)\]")


def _build_citations(findings: list[FindingRecord]) -> tuple[list[Citation], dict[str, str]]:
    marker_by_source: dict[str, str] = {}
    citations: list[Citation] = []
    meta: dict[str, FindingRecord] = {}
    for f in findings:
        if f.source_id and f.source_id not in marker_by_source:
            marker = f"[{len(marker_by_source) + 1}]"
            marker_by_source[f.source_id] = marker
            meta[f.source_id] = f
    for source_id, marker in marker_by_source.items():
        f = meta[source_id]
        citations.append(Citation(marker=marker, source_id=source_id, url=f.url, title=f.title or f.url))
    return citations, marker_by_source


def _sources_block(citations: list[Citation]) -> str:
    return "\n".join(f"{c.marker} {c.title} — {c.url}" for c in citations)


def _findings_block(findings: list[FindingRecord], marker_by_source: dict[str, str]) -> str:
    lines = []
    for f in findings:
        marker = marker_by_source.get(f.source_id, "")
        lines.append(f"- ({f.sub_question_id}) {f.claim} — {f.answer} {marker}")
    return "\n".join(lines)


def _conflicts_block(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "(none)"
    return "\n".join(
        f"- [{c.conflict_type}] {c.summary} (sources: {', '.join(c.source_ids)})" for c in conflicts
    )


def _references_md(citations: list[Citation]) -> str:
    return "\n".join(f"- {c.marker} {c.title} — {c.url}" for c in citations)


def _faithfulness(body: str, valid_markers: set[str]) -> float | None:
    used = _MARKER_RE.findall(body)
    if not used:
        return None
    valid = sum(1 for n in used if f"[{n}]" in valid_markers)
    return round(valid / len(used), 4)


@traced_node("synthesizer")
def synthesizer_node(state: DeepDiveState, deps: Deps, usage) -> dict:
    settings = deps.settings
    question = state["research_question"]
    findings = state.get("findings", [])
    conflicts = state.get("conflicts", [])
    sub_questions = state.get("sub_questions", [])

    citations, marker_by_source = _build_citations(findings)

    if not findings:
        report = (
            f"# Research report: {question}\n\n"
            "_No sufficient evidence was found from the available web sources to answer this "
            "question reliably. Try rephrasing the question or adding a Tavily API key for "
            "higher-quality search._"
        )
        usage.set(detail="no findings — stub report")
        return {"report_markdown": report, "citations": [], "citation_faithfulness": None}

    sq_text = "\n".join(f"- [{sq['id']}] {sq['question']}" for sq in sub_questions)
    human = (
        f"RESEARCH QUESTION:\n{question}\n\n"
        f"SUB-QUESTIONS:\n{sq_text}\n\n"
        f"SOURCES:\n{_sources_block(citations)}\n\n"
        f"FINDINGS:\n{_findings_block(findings, marker_by_source)}\n\n"
        f"CONFLICTS:\n{_conflicts_block(conflicts)}\n\n"
        "Write the markdown report now."
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    msg = deps.reasoning_llm.invoke([SystemMessage(SYNTH_SYSTEM), HumanMessage(human)])
    body = msg.content if isinstance(msg.content, str) else str(msg.content)
    ti, to = usage_from_message(msg)

    faith = _faithfulness(body, {c.marker for c in citations})
    report = body.strip() + "\n\n## References\n" + _references_md(citations)

    usage.set(
        model=settings.reasoning_model,
        tokens_in=ti,
        tokens_out=to,
        detail=f"{len(citations)} citations, faithfulness={faith}",
    )
    return {"report_markdown": report, "citations": citations, "citation_faithfulness": faith}
