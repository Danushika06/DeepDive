"""Fetch + clean a web page into readable text (trafilatura).

Used for DuckDuckGo results (which give only URL + snippet). Tavily can return full
page content directly, so those skip this step. One bad URL must never kill a run:
all failures are caught and returned as ``ok=False`` pages.
"""

from __future__ import annotations

import json

import trafilatura

from ..logging_conf import get_logger
from ..schemas.sources import FetchedPage

log = get_logger(__name__)


def fetch_and_clean(url: str, source_id: str, title: str = "") -> FetchedPage:
    fallback_title = title or url
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return FetchedPage(
                source_id=source_id, url=url, title=fallback_title, text="",
                ok=False, error="download failed",
            )
        extracted = trafilatura.extract(
            downloaded,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )
        if not extracted:
            return FetchedPage(
                source_id=source_id, url=url, title=fallback_title, text="",
                ok=False, error="no extractable text",
            )
        data = json.loads(extracted)
        text = (data.get("text") or "").strip()
        page_title = data.get("title") or fallback_title
        return FetchedPage(
            source_id=source_id, url=url, title=page_title, text=text, ok=bool(text),
            error=None if text else "empty text",
        )
    except Exception as exc:  # noqa: BLE001 - isolate per-URL failures
        log.warning("fetch failed for %s: %s", url, exc)
        return FetchedPage(
            source_id=source_id, url=url, title=fallback_title, text="",
            ok=False, error=str(exc),
        )
