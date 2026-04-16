"""FastAPI application entry point."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .retrieval import get_retriever
from .routes import ingest as ingest_routes
from .routes import query as query_routes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="PDF RAG Pipeline",
    description=(
        "A minimal Retrieval-Augmented Generation backend for PDF knowledge "
        "bases, powered by Mistral AI."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_routes.router)
app.include_router(query_routes.router)


# ─────────────────────────── Health ───────────────────────────
@app.get("/health", tags=["meta"])
async def health() -> dict:
    retr = get_retriever()
    return {
        "status": "ok",
        "indexed_chunks": retr.size,
        "chat_model": settings.mistral_chat_model,
        "embed_model": settings.mistral_embed_model,
    }


# ─────────────────────────── UI ───────────────────────────────
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")


@app.on_event("startup")
async def _startup() -> None:
    # Warm the retriever singleton on startup so the first query is fast.
    retriever = get_retriever()
    logger.info("RAG pipeline ready — %d chunks in index", retriever.size)
