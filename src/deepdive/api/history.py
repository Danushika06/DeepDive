"""In-memory run history (this process / session). Populated by the streaming
research endpoint on completion; read by the Run History view."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime

from ..schemas.report import ResearchReport


@dataclass
class HistoryEntry:
    id: str
    question: str
    created_at: str
    report: ResearchReport


class HistoryStore:
    def __init__(self, cap: int = 50) -> None:
        self._items: list[HistoryEntry] = []
        self._lock = threading.Lock()
        self._cap = cap

    def add(self, question: str, report: ResearchReport) -> str:
        with self._lock:
            entry = HistoryEntry(
                id=uuid.uuid4().hex[:8],
                question=question,
                created_at=datetime.now().isoformat(timespec="seconds"),
                report=report,
            )
            self._items.insert(0, entry)  # newest first
            del self._items[self._cap :]
            return entry.id

    def list(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": e.id,
                    "question": e.question,
                    "created_at": e.created_at,
                    "n_findings": e.report.metrics.n_findings,
                    "n_conflicts": e.report.metrics.n_conflicts,
                    "citation_faithfulness": e.report.metrics.citation_faithfulness,
                }
                for e in self._items
            ]

    def get(self, entry_id: str) -> ResearchReport | None:
        with self._lock:
            for e in self._items:
                if e.id == entry_id:
                    return e.report
        return None


# Module-level singleton.
history = HistoryStore()
