# -*- coding: utf-8 -*-
"""
FastAPI application entrypoint for the Arabic writing-teaching app.

Run (dev):
    python database.py            # create tables
    python seed_reference_books.py   # curriculum content (optional)
    python seed_placement.py         # placement-test questions (optional)
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs for the Flutter team:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc
    http://localhost:8000/openapi.json   -> generate a Dart client if wanted
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from routers import ai, attempts, auth, content, placement, progress

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Backend API for the Arabic writing-teaching app (تعلم الكتابة العربية).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static media (dictation audio, etc.) served at /media/...
# e.g. Exercise.content["audio_url"] = "/media/dictation/ara_0689.wav"
MEDIA_DIR = Path(__file__).resolve().parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.on_event("startup")
def _startup() -> None:
    # Convenience for dev; use Alembic migrations in production.
    init_db()
    if settings.AI_WARM_UP:
        _warm_up_models()


def _warm_up_models() -> None:
    """Load the ML models in the background so the first student submission
    doesn't pay for it (the AES checkpoint, the relevance CrossEncoder and the
    two GEC taggers take a while to come up). The server accepts requests
    immediately; anything that arrives early simply waits for its model."""
    import threading

    def _run() -> None:
        # Load one at a time and report each, so a model that fails to come up
        # (usually out of memory — the four checkpoints want ~3 GB between
        # them) is visible in the log at boot instead of surfacing later as a
        # failed student submission. Each warm_up swallows its own exception
        # and returns False; the model stays unloaded and is retried lazily.
        for enabled, name, load in (
            (settings.GEC_ENABLED, "gec", lambda: __import__("gec_service").warm_up()),
            (settings.RELEVANCE_ENABLED, "relevance",
             lambda: __import__("relevance_scorer").warm_up()),
            (settings.AES_ENABLED, "aes", lambda: __import__("aes_model").warm_up()),
            # Last on purpose: it is the only TensorFlow model, so if the box
            # is going to run out of memory it does so after the three text
            # models are safely up (copying then degrades to the trace score).
            (settings.HANDWRITING_ENABLED, "handwriting",
             lambda: __import__("handwriting_model").warm_up()),
        ):
            if not enabled:
                continue
            try:
                ok = load()
            except Exception as exc:
                ok, detail = False, f" ({type(exc).__name__}: {exc})"
            else:
                detail = ""
            print(f"[warm-up] {name}: {'ok' if ok else 'FAILED'}{detail}", flush=True)

    threading.Thread(target=_run, name="ai-warm-up", daemon=True).start()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "ai_mock": settings.AI_MOCK,
        "aes": settings.AES_ENABLED,
        "relevance": settings.RELEVANCE_ENABLED,
        "gec": settings.GEC_ENABLED,
        "handwriting": settings.HANDWRITING_ENABLED,
    }


P = settings.API_PREFIX
app.include_router(auth.router, prefix=P)
app.include_router(content.router, prefix=P)
app.include_router(progress.router, prefix=P)
app.include_router(attempts.router, prefix=P)
app.include_router(placement.router, prefix=P)
app.include_router(ai.router, prefix=P)
