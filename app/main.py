"""
main.py — FastAPI application shell for the LLM Wargame Middleware.

Responsibilities (only these — everything else is delegated):
  - Configure logging AT IMPORT TIME (before app submodules are imported) so
    their module-level loggers and any import-time messages use our format.
  - Lifespan startup:  init HTTP client -> load acronym CSV -> init image cache.
  - Lifespan shutdown: close the HTTP client (guarded).
  - Expose POST /v1/chat/completions (OpenAI-compatible), delegating to the
    pipeline, and return the streamed SSE response.
  - Expose GET /health with per-subsystem readiness for a single-curl check.

Run (inside the container):
  uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging

# ------------------------------------------------------------------ #
# IMPORTANT: configure logging BEFORE importing the app submodules.
#
# Python runs a module's top-level code (including `logger = getLogger(__name__)`
# and any import-time log calls) at import time. If we configured logging inside
# lifespan (which runs AFTER all imports), those early logs would use Python's
# default config or be dropped. We therefore:
#   1) import only `logging` and our settings first,
#   2) configure the root logger,
#   3) THEN import the app submodules.
#
# Do NOT move the app-submodule imports above _configure_logging(); a linter or
# PEP8 "imports at top" auto-fix would silently reintroduce the late-config bug.
# ------------------------------------------------------------------ #
from app.config import settings  # noqa: E402  (settings only; no logging done at import)


def _configure_logging() -> None:
    """
    Configure root logging from settings.LOG_LEVEL, robust to uvicorn having
    already installed handlers.

    - basicConfig installs a handler+format IF the root has none (direct/test runs).
    - We ALSO set the root level explicitly, so our level takes effect even when
      uvicorn has already added handlers (in which case basicConfig is a no-op).
    """
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(level)


# Configure immediately, before the app submodules are imported below.
_configure_logging()

# Now import the rest (their module-level loggers will use the config above).
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from app import backends, images, terminology  # noqa: E402
from app.pipeline import process_request  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown wiring. Startup order is deliberate:
      1) HTTP client  — needed by any backend call.
      2) Acronyms CSV — loaded into memory for terminology matching.
      3) Image cache  — SQLite table ensured; shared, persistent.
    Each step is logged so a boot failure is obvious in container logs.
    """
    logger.info("Middleware starting up...")

    # 1) HTTP client (shared, pooled).
    backends.init_http_client()

    # 2) Acronym terminology CSV (optional; degrade gracefully on failure).
    try:
        count = terminology.load_acronyms(settings.ACRONYMS_CSV_PATH)
        logger.info("Terminology: %d acronyms loaded.", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load acronyms (terminology disabled): %s", exc)

    # 3) Image description cache (SQLite). Logged loudly on failure; not fatal.
    try:
        images.init_image_cache(settings.IMAGE_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize image cache: %s", exc)

    logger.info(
        "Middleware ready. Llama=%s Gemma=%s ceiling=%d (max_model_len=%d, "
        "reserved=%d).",
        settings.LLAMA_URL,
        settings.GEMMA_URL,
        settings.LLAMA_CEILING,
        settings.LLAMA_MAX_MODEL_LEN,
        settings.LLAMA_RESERVED_GENERATION,
    )

    try:
        yield
    finally:
        # Shutdown: close the HTTP client. GUARDED so a teardown error never
        # masks the original boot exception (which propagates from within `yield`
        # or from a startup step above).
        logger.info("Middleware shutting down...")
        try:
            await backends.close_http_client()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close HTTP client cleanly: %s", exc)


app = FastAPI(lifespan=lifespan, title="LLM Wargame Middleware")


@app.get("/health")
async def health():
    """
    Per-subsystem readiness for a single-curl deployment check. Never raises.
    """
    try:
        backends.get_client()
        client_ready = True
    except RuntimeError:
        client_ready = False

    try:
        cached_images = images.get_cache().count()
    except Exception:  # noqa: BLE001
        cached_images = None

    return {
        "status": "ok",
        "http_client": client_ready,
        "acronyms": terminology.acronym_count(),
        "definitions": terminology.definition_count(),
        "cached_images": cached_images,
        "llama_ceiling": settings.LLAMA_CEILING,
    }


@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible model list. OWUI calls this to populate its model
    selector and Task Model dropdown; the ids here are the middleware's
    routing contract (config-driven), not real backend model names.
    """
    created = 0  # static; clients only key on id
    ids = [
        settings.MODEL_ID_LLAMA,
        settings.MODEL_ID_GEMMA,
        settings.MODEL_ID_AUTO,
        settings.MODEL_ID_TASKS,
    ]
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": created, "owned_by": "middleware"}
            for mid in ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint (Path A). Reads the body,
    delegates all logic to the pipeline, and streams the SSE response back to
    OWUI. Malformed bodies return a clean 400 (no traceback leakage).
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body.", "type": "bad_request"}},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be a JSON object.",
                    "type": "bad_request",
                }
            },
        )

    result = await process_request(body, request)

    if isinstance(result, dict):
        return JSONResponse(result)

    return StreamingResponse(
        result,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )