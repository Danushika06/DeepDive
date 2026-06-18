"""FastAPI application factory.

The lifespan warms the local embedding model on startup (it is downloaded at image
build time, loaded into RAM here) so the first request doesn't pay the load cost.
Set ``DEEPDIVE_SKIP_WARMUP=1`` to skip it (used by hermetic tests).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import get_settings
from ..logging_conf import configure_logging, get_logger
from .routes import router

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    if os.getenv("DEEPDIVE_SKIP_WARMUP") != "1":
        try:
            from ..retrieval.index import get_embedder

            get_embedder(settings.embedding_model)
            log.info("Embedding model warmed: %s", settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            log.warning("Embedding warm-up skipped: %s", exc)
    yield


app = FastAPI(
    title="DeepDive — Agentic Research Synthesis",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)

# Serve the no-build frontend (static assets at /static, SPA shell at /).
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
