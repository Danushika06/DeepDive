"""Eval runner CLI.

  python -m evals.runner --suite all
  python -m evals.runner --suite extraction
  python -m evals.runner --suite citation --no-judge
  python -m evals.runner --suite all --out eval_results.json

Each suite is isolated: a failure (e.g. missing key / no network) is reported as an
error row instead of aborting the others. Full per-case results are dumped to JSON for
error analysis (the course stresses reading individual failures, not just aggregates).
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from deepdive.config import get_settings
from deepdive.logging_conf import configure_logging

from . import metrics_citation, metrics_extraction, metrics_retrieval

_META_KEYS = {"suite", "level", "n", "per_case", "error"}


def _safe_run(fn) -> dict:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - isolate suite failures
        return {"suite": getattr(fn, "_suite", "?"), "level": "-", "n": 0, "error": f"{type(exc).__name__}: {exc}"}


def _render(results: list[dict]) -> None:
    console = Console()
    table = Table(title="DeepDive — Eval Metrics", show_lines=False)
    table.add_column("Suite", style="cyan")
    table.add_column("Level")
    table.add_column("Metric")
    table.add_column("Value", justify="right", style="green")
    table.add_column("N", justify="right")

    for res in results:
        suite = res.get("suite", "?")
        level = res.get("level", "-")
        n = str(res.get("n", 0))
        if res.get("error"):
            table.add_row(suite, level, "ERROR", res["error"][:60], n)
            continue
        for key, val in res.items():
            if key in _META_KEYS:
                continue
            table.add_row(suite, level, key, f"{val}", n)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DeepDive evals.")
    parser.add_argument(
        "--suite", choices=["all", "retrieval", "extraction", "citation"], default="all"
    )
    parser.add_argument("--no-judge", action="store_true", help="Citation suite: skip the LLM judge.")
    parser.add_argument("--out", default=None, help="Write full per-case results to this JSON file.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    results: list[dict] = []
    if args.suite in ("all", "retrieval"):
        results.append(_safe_run(lambda: metrics_retrieval.run(settings)))
    if args.suite in ("all", "extraction"):
        results.append(_safe_run(lambda: metrics_extraction.run(settings)))
    if args.suite in ("all", "citation"):
        results.append(_safe_run(lambda: metrics_citation.run(settings, use_judge=not args.no_judge)))

    _render(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote full results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
