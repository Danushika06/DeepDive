"""API tests with a mocked pipeline (no network / no LLM)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deepdive.schemas.report import Metrics, ResearchReport


def _fake_report(question: str, max_sources=None) -> ResearchReport:
    return ResearchReport(question=question, report_markdown="# ok", metrics=Metrics(n_findings=1))


def test_health(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SKIP_WARMUP", "1")
    from deepdive.api.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["search_provider"] in ("tavily", "ddg")


def test_sync_research(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SKIP_WARMUP", "1")
    import deepdive.api.routes as routes

    monkeypatch.setattr(routes, "run_research", _fake_report)
    from deepdive.api.app import app

    with TestClient(app) as client:
        r = client.post("/research/sync", json={"question": "what is x?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert body["report"]["report_markdown"] == "# ok"


def test_async_job_lifecycle(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SKIP_WARMUP", "1")
    import deepdive.api.routes as routes

    monkeypatch.setattr(routes, "run_research", _fake_report)
    from deepdive.api.app import app

    with TestClient(app) as client:
        r = client.post("/research", json={"question": "what is x?"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert job_id

        r2 = client.get(f"/research/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] in ("queued", "running", "done")

        r3 = client.get("/research/does-not-exist")
        assert r3.status_code == 404
