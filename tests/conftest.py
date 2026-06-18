"""Test fakes — let us exercise the whole graph with zero network or LLM calls.

The graph is built with ``Deps`` whose heavy operations (search, indexing, extraction,
conflict detection) are stubbed, and whose LLMs are ``FakeLLM`` objects. This is what
the injectable-Deps design buys us: fast, deterministic routing/loop-bound tests.
"""

from __future__ import annotations

from deepdive.agents.deps import Deps
from deepdive.config import Settings
from deepdive.extraction.extract import ExtractStats
from deepdive.schemas.extraction import FindingRecord, QueryPlan, SubQuestionPlan
from deepdive.schemas.sources import Chunk, FetchedPage


class FakeMsg:
    def __init__(self, content: str = "", tin: int = 10, tout: int = 5) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": tin, "output_tokens": tout}
        self.response_metadata = {}


class FakeStructured:
    def __init__(self, parsed) -> None:
        self._parsed = parsed

    def invoke(self, messages):
        return {"raw": FakeMsg(), "parsed": self._parsed, "parsing_error": None}


class FakeLLM:
    """Supports both ``.with_structured_output(Schema).invoke()`` and ``.invoke()``."""

    def __init__(self, by_schema: dict | None = None, content: str = "# Report\n\nA finding [1].\n") -> None:
        self._by_schema = by_schema or {}
        self._content = content

    def with_structured_output(self, schema, include_raw: bool = False, **kw):
        return FakeStructured(self._by_schema.get(schema))

    def invoke(self, messages):
        return FakeMsg(content=self._content)


class FakeIndex:
    def __init__(self, chunks_by_subq: dict[str, list[Chunk]]) -> None:
        self._chunks = chunks_by_subq

    def retrieve(self, query: str, k: int, sub_question_id: str | None = None) -> list[Chunk]:
        return self._chunks.get(sub_question_id, [])


def make_settings(**overrides) -> Settings:
    base = dict(min_sources_per_subq=1, max_plan_iters=2, max_reflection_iters=1, top_k=3)
    base.update(overrides)
    return Settings(**base)


def make_deps(
    *,
    settings: Settings,
    plan: QueryPlan,
    sources: list[FetchedPage],
    chunks_by_subq: dict[str, list[Chunk]],
    findings: list[FindingRecord],
    conflicts: list | None = None,
) -> Deps:
    reasoning = FakeLLM(by_schema={QueryPlan: plan})

    def gather(provider, queries, s):
        return sources

    def build_index(srcs, s):
        return FakeIndex(chunks_by_subq) if any(p.ok for p in srcs) else None

    def extract_all(llm, chs, subqs, s):
        return ExtractStats(
            findings=list(findings), n_chunks=len(chs),
            n_valid_chunks=len(chs), model="fake", tokens_in=1, tokens_out=1,
        )

    def detect_conflicts(llm, fs, s):
        return (list(conflicts or []), 0, 0)

    return Deps(
        settings=settings,
        reasoning_llm=reasoning,
        extract_llm=FakeLLM(),
        search_provider=type("P", (), {"name": "fake"})(),
        gather_sources_fn=gather,
        build_index_fn=build_index,
        extract_all_fn=extract_all,
        detect_conflicts_fn=detect_conflicts,
    )


def two_subq_plan() -> QueryPlan:
    return QueryPlan(
        sub_questions=[
            SubQuestionPlan(sub_question_id="sq1", sub_question="What is X?", queries=["X"]),
            SubQuestionPlan(sub_question_id="sq2", sub_question="What is Y?", queries=["Y"]),
        ]
    )


def source(sid: str) -> FetchedPage:
    return FetchedPage(source_id=sid, url=f"https://ex.com/{sid}", title=sid, text="some text", ok=True)


def chunk(sid: str, sqid: str) -> Chunk:
    return Chunk(source_id=sid, url=f"https://ex.com/{sid}", title=sid, text="t", score=0.1, sub_question_id=sqid)


def finding(sid: str, sqid: str) -> FindingRecord:
    return FindingRecord(
        sub_question_id=sqid, claim="c", answer="a", supporting_quote="q",
        source_id=sid, url=f"https://ex.com/{sid}", title=sid,
    )
