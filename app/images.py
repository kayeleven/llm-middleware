"""
images.py — Image handling and persistent shared description cache (README Step 3).

Async note:
  All Gemma image description is injected as an ASYNC callable and awaited, so
  this module runs on the single event loop (no asyncio.run, no sync bridge).
  Blocking SQLite is offloaded via asyncio.to_thread so it never stalls the loop.

Two responsibilities plus a proactive warm-up:

  A) FRESH image in the CURRENT user turn:
       Route the turn to Gemma (native image answers it). Separately and
       silently, fire a background warm-up that DB-first-checks each image and,
       on a MISS, describes it via Gemma and stores it — so a later Llama turn
       has NO lag. The warm-up NEVER alters the answering payload. Because images
       are SHARED across ~50 participants, the first time THIS chat sees an image
       may not be the first time the SYSTEM has; we always check the DB first.

  B) Images only in HISTORY (current turn text-only):
       Hash each image's bytes, look up the shared cache, describe on a miss
       (almost always a hit due to the warm-up), and REPLACE the image part with
       "[Image description: <text>]" so history is 100% text-only for Llama.

Guardrails:
  - MAX_IMAGE_BYTES: images larger than the configured limit are skipped with a
    placeholder BEFORE any hashing/base64 encoding (memory guardrail for IL5
    containers; base64 inflates ~33% and multiple copies can spike memory).
  - Failed describes are NOT cached, so a later healthy attempt can succeed.
  - Undecodable images degrade to a clear placeholder rather than crashing.

Persistence & concurrency:
  - SQLite (WAL), keyed by image-byte SHA-256; SHARED across participants and
    persistent across restarts (Caveat #3). Connections are per-operation.
  - The async warm-up offloads all SQLite via asyncio.to_thread and uses a
    per-image-hash asyncio.Lock (double-checked) so simultaneous submissions of
    the SAME new image trigger only ONE Gemma describe.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import re
import sqlite3
from typing import Any, Iterable, Optional, Protocol

from app.config import settings

logger = logging.getLogger(__name__)

# An OpenAI-style message dict, e.g. {"role": "user", "content": <str | list>}.
Message = dict[str, Any]


# ==================================================================== #
# Injected capability: describe an image using Gemma (ASYNC, single form).
# ==================================================================== #
class AsyncImageDescriber(Protocol):
    """
    Async describer used by BOTH the warm-up and the history-rewrite path. Given
    raw image bytes and optional MIME type, returns a plain-text description.
    Implemented by backends.describe_image_async. Must NOT raise for normal
    failures; return "" on failure (turned into a safe placeholder / cold cache).
    """
    async def __call__(
        self, image_bytes: bytes, mime_type: Optional[str] = None
    ) -> str: ...


# ==================================================================== #
# Parsing OWUI / OpenAI multimodal content.
# ==================================================================== #
# RFC 2397 allows optional ;name=value params before ;base64. Confirm against a
# real OWUI image request fixture; this is the standard OpenAI encoding.
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[\w./+-]+)"      # MIME type, e.g. image/png
    r"(?:;[\w.+-]+=[^;,]*)*"          # zero or more optional ;name=value params
    r";base64,"                       # required base64 marker
    r"(?P<payload>.+)$",              # the base64 payload
    re.DOTALL,
)


def _is_image_part(part: Any) -> bool:
    """True if a content element is an image part ({"type":"image_url","image_url":{"url":...}})."""
    return (
        isinstance(part, dict)
        and part.get("type") == "image_url"
        and isinstance(part.get("image_url"), dict)
        and "url" in part["image_url"]
    )


def _decode_image_part(part: dict) -> Optional[tuple[bytes, Optional[str]]]:
    """
    Extract (image_bytes, mime_type) from an image_url part.

    Handles the base64 data-URI case. A plain http(s) URL (not fetchable in an
    air-gapped setup) yields None. Returns None if bytes cannot be decoded OR if
    the decoded image exceeds settings.MAX_IMAGE_BYTES (memory guardrail: we
    reject BEFORE any further processing).
    """
    url = part["image_url"]["url"]
    if not isinstance(url, str):
        return None

    m = _DATA_URI_RE.match(url.strip())
    if not m:
        logger.warning("Image part url is not a base64 data URI; cannot hash.")
        return None

    mime = m.group("mime")
    payload = m.group("payload")
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Failed to base64-decode image payload: %s", exc)
        return None

    # Memory guardrail: reject oversized images before hashing/encoding.
    if len(image_bytes) > settings.MAX_IMAGE_BYTES:
        logger.warning(
            "Image (%d bytes) exceeds MAX_IMAGE_BYTES (%d); skipping.",
            len(image_bytes),
            settings.MAX_IMAGE_BYTES,
        )
        return None

    return image_bytes, mime


def hash_image_bytes(image_bytes: bytes) -> str:
    """SHA-256 hex of the raw image BYTES. Stable across resends and identical across shared uploads."""
    return hashlib.sha256(image_bytes).hexdigest()


# ==================================================================== #
# Image detection helpers.
# ==================================================================== #
def current_turn_has_image(messages: list[Message]) -> bool:
    """True if the last user message contains a fresh image (forces Gemma, README Step 3)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                return any(_is_image_part(p) for p in content)
            return False
    return False


def history_has_image(messages: list[Message]) -> bool:
    """True if ANY message contains an image part (cheap pre-check to skip the rewrite path)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(_is_image_part(p) for p in content):
            return True
    return False


def count_images(messages: list[Message]) -> int:
    """Total number of image parts across all messages (chat-log divergence-event helper)."""
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            count += sum(1 for p in content if _is_image_part(p))
    return count


def _iter_current_turn_images(
    messages: list[Message],
) -> Iterable[tuple[bytes, Optional[str]]]:
    """Yield (image_bytes, mime) for every DECODABLE, in-limit image in the last user turn."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if _is_image_part(part):
                        decoded = _decode_image_part(part)
                        if decoded is not None:
                            yield decoded
            return
    return


# ==================================================================== #
# Persistent cache (the only local I/O in this module).
# ==================================================================== #
class ImageCache:
    """
    Persistent map: image SHA-256 hex -> description text. SQLite (WAL), SHARED
    across participants and persistent across restarts (Caveat #3). Connections
    are per-operation (sharing one across threads/tasks is unsafe).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init(self) -> None:
        """Create the table if absent. Called once from main.py's lifespan."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_descriptions (
                    hash        TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    mime_type   TEXT,
                    created_at  INTEGER DEFAULT (strftime('%s','now'))
                );
                """
            )
        logger.info("Image description cache ready at %s", self._db_path)

    def get(self, image_hash: str) -> Optional[str]:
        """Return cached description or None. A read failure is treated as a miss."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT description FROM image_descriptions WHERE hash = ?",
                    (image_hash,),
                ).fetchone()
                return row[0] if row else None
        except sqlite3.Error as exc:
            logger.error("Image cache read failed for %s: %s", image_hash, exc)
            return None

    def put(self, image_hash: str, description: str, mime_type: Optional[str]) -> None:
        """Store/replace a description (idempotent). A write failure is non-fatal."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO image_descriptions
                        (hash, description, mime_type)
                    VALUES (?, ?, ?);
                    """,
                    (image_hash, description, mime_type),
                )
        except sqlite3.Error as exc:
            logger.error("Image cache write failed for %s: %s", image_hash, exc)

    def count(self) -> int:
        """Number of cached descriptions (for /health)."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM image_descriptions"
                ).fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0


# ==================================================================== #
# Module-level cache singleton, initialized at startup.
# ==================================================================== #
_cache: Optional[ImageCache] = None


def init_image_cache(db_path: str) -> ImageCache:
    """Create and initialize the module-level ImageCache singleton (from main.py)."""
    global _cache
    _cache = ImageCache(db_path)
    _cache.init()
    return _cache


def get_cache() -> ImageCache:
    """Return the initialized cache, or raise if init was skipped (a bug)."""
    if _cache is None:
        raise RuntimeError(
            "Image cache not initialized. Call init_image_cache() at startup."
        )
    return _cache


# ==================================================================== #
# Shared async describe-and-cache (DB-first; used by both branches).
# ==================================================================== #
# Per-image-hash in-flight locks prevent multiple concurrent describes of the
# SAME never-before-seen image. The registry is guarded by a module lock so two
# tasks cannot create two different Lock objects for one hash.
_inflight_locks: dict[str, asyncio.Lock] = {}
_inflight_registry_lock = asyncio.Lock()


async def _get_inflight_lock(image_hash: str) -> asyncio.Lock:
    async with _inflight_registry_lock:
        lock = _inflight_locks.get(image_hash)
        if lock is None:
            lock = asyncio.Lock()
            _inflight_locks[image_hash] = lock
        return lock


async def _cleanup_inflight_lock(image_hash: str) -> None:
    """Drop a per-hash lock once warmed, so the registry does not grow unbounded."""
    async with _inflight_registry_lock:
        existing = _inflight_locks.get(image_hash)
        if existing is not None and not existing.locked():
            _inflight_locks.pop(image_hash, None)


async def _describe_and_cache(
    image_bytes: bytes,
    mime_type: Optional[str],
    describe: AsyncImageDescriber,
) -> str:
    """
    Return a description for an image, DB-FIRST, deduplicated, on the event loop.

    Flow:
      hash -> cache.get (offloaded) -> hit? return it
           -> miss -> acquire per-hash lock -> re-check cache -> describe -> put
    On a describe failure (empty), return a placeholder and DO NOT cache it, so a
    later healthy attempt can succeed.
    """
    image_hash = hash_image_bytes(image_bytes)
    cache = get_cache()

    cached = await asyncio.to_thread(cache.get, image_hash)
    if cached is not None:
        return cached

    lock = await _get_inflight_lock(image_hash)
    try:
        async with lock:
            # Double-check inside the lock (another task may have just warmed it).
            cached = await asyncio.to_thread(cache.get, image_hash)
            if cached is not None:
                return cached

            description = (await describe(image_bytes, mime_type) or "").strip()
            if not description:
                logger.warning(
                    "Image description empty for hash %s; using placeholder.",
                    image_hash,
                )
                return "[Image could not be described]"

            await asyncio.to_thread(cache.put, image_hash, description, mime_type)
            return description
    finally:
        await _cleanup_inflight_lock(image_hash)


# ==================================================================== #
# Branch B: async history rewrite (image -> text description).
# ==================================================================== #
async def _rewrite_message_content(
    content: list, describe: AsyncImageDescriber
) -> list:
    """
    Return a NEW content list where every image part is replaced by a text part:
        {"type": "text", "text": "[Image description: <desc>]"}
    Non-image parts pass through unchanged.
    """
    new_parts: list = []
    for part in content:
        if _is_image_part(part):
            decoded = _decode_image_part(part)
            if decoded is None:
                new_parts.append(
                    {
                        "type": "text",
                        "text": "[Image present but could not be processed]",
                    }
                )
                continue
            image_bytes, mime = decoded
            description = await _describe_and_cache(image_bytes, mime, describe)
            new_parts.append(
                {"type": "text", "text": f"[Image description: {description}]"}
            )
        else:
            new_parts.append(part)
    return new_parts


async def replace_history_images_with_descriptions(
    messages: list[Message],
    describe: AsyncImageDescriber,
) -> list[Message]:
    """
    Return a NEW message list in which every image anywhere in `messages` is
    replaced by its cached text description (branch B). Awaited by the pipeline
    on the event loop; internal SQLite is offloaded inside _describe_and_cache.

    Use ONLY when the current turn has NO fresh image. Guarantees text-only
    history for Llama; DB-first shared cache; input not mutated. If there are no
    images, the original list is returned unchanged (no work).
    """
    if not history_has_image(messages):
        return messages

    rewritten: list[Message] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(_is_image_part(p) for p in content):
            new_msg = dict(msg)  # shallow copy; replace only `content`
            new_msg["content"] = await _rewrite_message_content(content, describe)
            rewritten.append(new_msg)
        else:
            rewritten.append(msg)
    return rewritten


# ==================================================================== #
# Branch A: fire-and-forget proactive cache warm-up (async).
# ==================================================================== #
async def _warm_one_image(
    image_bytes: bytes,
    mime_type: Optional[str],
    describe: AsyncImageDescriber,
) -> None:
    """
    Ensure a description for ONE image exists in the cache, DB-FIRST and
    deduplicated. Reuses _describe_and_cache (which does the get/lock/describe/put
    and returns a value we discard here — we only care about the side effect).
    Never raises: failures are logged and leave the cache cold, so branch B's
    describe-on-demand handles it later.
    """
    try:
        await _describe_and_cache(image_bytes, mime_type, describe)
    except Exception as exc:  # noqa: BLE001 - warm-up must never break anything
        logger.error("Image cache warm-up failed: %s", exc)


async def warm_current_turn_images(
    messages: list[Message],
    describe: AsyncImageDescriber,
) -> None:
    """
    PROACTIVE, FIRE-AND-FORGET cache warming (README Step 3).

    For each fresh image in the current user turn, ensure a description is cached
    (DB-first; skipped if already present), so the next turn has NO lag if the
    chat moves to Llama.

    CRITICAL GUARANTEES:
      - No effect on the payload used to ANSWER this turn (Gemma gets the native
        image); this only writes to the shared cache as a side effect.
      - DB-first + per-hash lock: a conversation that stays on Gemma pays at most
        ONE describe per UNIQUE image system-wide (never per turn).
      - Never raises.

    Intended usage in pipeline.py (fire-and-forget so the answer is not delayed):
        if current_turn_has_image(messages):
            task = asyncio.create_task(
                warm_current_turn_images(messages, describe_image_async)
            )
            # pipeline keeps a reference + a done-callback to surface exceptions.

    Multiple images warm in parallel via asyncio.gather.
    """
    images = list(_iter_current_turn_images(messages))
    if not images:
        return
    await asyncio.gather(
        *(_warm_one_image(image_bytes, mime, describe) for (image_bytes, mime) in images)
    )