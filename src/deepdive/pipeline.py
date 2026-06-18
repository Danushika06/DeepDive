"""Top-level orchestrator + CLI.

``run_research(question)`` builds the graph, invokes it, and assembles a
:class:`ResearchReport`. ``main()`` is the ``deepdive`` console entry point.
"""

from __future__ import annotations

import argparse
import sys

from .agents.deps import Deps, build_default_deps
from .agents.graph import build_graph
from .agents.state import initial_state
from .config import Settings, get_settings
from .logging_conf import configure_logging, get_logger
from .schemas.report import Metrics, ResearchReport
from .schemas.trace import RunTrace

log = get_logger(__name__)


def _assemble_report(question: str, final: dict) -> ResearchReport:
    metrics = Metrics(
        n_sources_retrieved=len(final.get("sources", [])),
        n_unique_sources=final.get("n_unique_sources", 0),
        n_findings=len(final.get("findings", [])),
        n_conflicts=len(final.get("conflicts", [])),
        extraction_valid_rate=final.get("extraction_valid_rate"),
        citation_faithfulness=final.get("citation_faithfulness"),
    )
    return ResearchReport(
        question=question,
        sub_questions=[sq["question"] for sq in final.get("sub_questions", [])],
        report_markdown=final.get("report_markdown", ""),
        citations=final.get("citations", []),
        conflicts=final.get("conflicts", []),
        findings=final.get("findings", []),
        metrics=metrics,
        trace=RunTrace.from_spans(final.get("trace", [])),
    )


def run_research(
    question: str,
    max_sources: int | None = None,
    settings: Settings | None = None,
    deps: Deps | None = None,
) -> ResearchReport:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    if max_sources is not None:
        settings = settings.model_copy(update={"max_sources": max_sources})
    deps = deps or build_default_deps(settings)

    app = build_graph(deps)
    state0 = initial_state(question, settings.runtime_config())
    final = app.invoke(state0, config={"recursion_limit": settings.recursion_limit})
    return _assemble_report(question, final)


def _print_human(report: ResearchReport) -> None:
    m = report.metrics
    print(report.report_markdown)
    print("\n" + "=" * 70)
    print("METRICS")
    print(f"  sources retrieved / unique : {m.n_sources_retrieved} / {m.n_unique_sources}")
    print(f"  findings / conflicts       : {m.n_findings} / {m.n_conflicts}")
    print(f"  extraction valid rate      : {m.extraction_valid_rate}")
    print(f"  citation faithfulness      : {m.citation_faithfulness}")
    t = report.trace
    print(
        f"  trace: {len(t.spans)} steps, {t.total_latency_ms:.0f} ms, "
        f"{t.total_tokens_in}+{t.total_tokens_out} tokens, ${t.total_cost_usd:.4f}"
    )
    for s in t.spans:
        cost = f"${s.est_cost_usd:.4f}" if s.est_cost_usd is not None else "—"
        print(f"    - {s.node:<12} {s.latency_ms:>7.0f} ms  {cost:<9} {s.detail or ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepDive — agentic research synthesis.")
    parser.add_argument("question", nargs="+", help="The research question.")
    parser.add_argument("--max-sources", type=int, default=None, help="Cap on web sources.")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    args = parser.parse_args(argv)

    question = " ".join(args.question)
    report = run_research(question, max_sources=args.max_sources)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
