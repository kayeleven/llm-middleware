"""
chatlog.py — Fire-and-forget chat event logging for post-event analysis.

Two thin events per real turn (never the full OWUI-resent payload):
  - REQUEST: routing decision + the latest user turn, verbatim, captured
    BEFORE any transformation (summarization/terminology/image-rewrite).
  - RESPONSE: the fully assembled answer, captured by reparsing the SSE byte
    stream the same way streaming.collect_response already does — NOT by
    instrumenting streaming.py's forwarding paths. One of those paths (the
    "SAFE COMMON PATH" used for every Gemma-routed response) is a byte-
    faithful raw passthrough that deliberately never decodes content, so
    there is no accumulator to hook there without changing that module's
    correctness-critical behavior. Reparsing the final byte stream after the
    fact works uniformly regardless of which internal path produced it.

Plus two evaluation-only divergence events (summarization, image handling) —
relevant only to response->move evaluation, never to the topic corpus. These
are observed from pipeline.py via before/after state comparisons, so the
pure logic modules (summarization.py, terminology.py, images.py) are not
touched.

Fail-soft, non-blocking:
  - log_event() never raises; a logging failure must never break or delay a
    participant's chat. File I/O is offloaded via asyncio.to_thread.
  - Callers use pipeline.py's existing fire-and-forget task helper
    (_spawn_background) for events that must not delay the response; the
    streaming tee awaits its own log write directly, but only AFTER every
    byte has already been yielded to the client, so it adds no participant-
    visible latency.

Storage: append-only JSONL, one file per LOCAL calendar day under
CHATLOG_DIR. The filename is a pure function of the current date, computed
fresh at each write — no held-open file handle, no rotation trigger to get
wrong.

Pseudonymization: user_id is HMAC-SHA256'd with a configured salt the moment
it is read from the request body (see pipeline.py), so the raw id never
reaches this module, the log file, or any in-process map.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from app import terminology
from app.backends import parse_sse_data_line
from app.config import settings
from app.routing import Reason, Target

logger = logging.getLogger(__name__)


# ==================================================================== #
# Pseudonymization.
# ==================================================================== #
def pseudonymize(user_id: Any) -> Optional[str]:
    """
    HMAC-SHA256(salt, user_id) hex digest, or None if user_id is absent.

    A keyed HMAC needs no stored id->pseudonym map: it is deterministic
    (same user always maps to the same pseudonym, so turns join correctly)
    without ever holding a reversible mapping table to protect or discard.
    """
    if not user_id:
        return None
    return hmac.new(
        settings.CHATLOG_PSEUDONYM_SALT.encode("utf-8"),
        str(user_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# ==================================================================== #
# Per-request identity (gathered once in pipeline.py, reused across the
# routing branches) and the per-dispatch log context built from it.
# ==================================================================== #
@dataclass(frozen=True)
class LogContext:
    """Everything the REQUEST/RESPONSE events need for one dispatch."""
    correlation_id: str
    conv_key: str
    pseudo_user: Optional[str]
    group_ids: list[str]
    model_requested: str
    target: Target
    reason: Reason
    scan_text: str
    matched_acronyms: list[str]


@dataclass(frozen=True)
class RequestIdentity:
    """
    Fields constant for the whole request, gathered once at the top of
    pipeline.process_request and reused across whichever routing branch
    ultimately dispatches (avoids threading five separate params through
    _divert_to_gemma and every call site).
    """
    correlation_id: str
    conv_key: str
    pseudo_user: Optional[str]
    group_ids: list[str]
    model_requested: str

    def log_context(self, target: Target, reason: Reason, scan_text: str) -> LogContext:
        return LogContext(
            correlation_id=self.correlation_id,
            conv_key=self.conv_key,
            pseudo_user=self.pseudo_user,
            group_ids=self.group_ids,
            model_requested=self.model_requested,
            target=target,
            reason=reason,
            scan_text=scan_text,
            matched_acronyms=list(terminology.find_matches(scan_text).keys()),
        )


# ==================================================================== #
# Storage: append-only JSONL, one file per LOCAL calendar day.
# ==================================================================== #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path() -> Path:
    """Today's (LOCAL calendar day) log file path. Pure function of the clock."""
    return Path(settings.CHATLOG_DIR) / f"chat_log-{date.today().isoformat()}.jsonl"


def _write_line_sync(event: dict) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


async def log_event(event: dict) -> None:
    """
    Append one JSON line to today's log file. NEVER raises: a logging
    failure must never break or delay a participant's chat (fail-soft).
    Offloaded via asyncio.to_thread so file I/O never blocks the event loop.
    """
    if not settings.CHATLOG_ENABLED:
        return
    try:
        await asyncio.to_thread(_write_line_sync, event)
    except Exception as exc:  # noqa: BLE001 - logging must never break a chat
        logger.error("Chat log write failed: %s", exc)


# ==================================================================== #
# REQUEST event.
# ==================================================================== #
async def log_request(ctx: LogContext) -> None:
    await log_event({
        "type": "request",
        "correlation_id": ctx.correlation_id,
        "ts": _now_iso(),
        "chat_id": ctx.conv_key,
        "pseudo_user": ctx.pseudo_user,
        "group_ids": ctx.group_ids,
        "model_requested": ctx.model_requested,
        "model_served": ctx.target.value,
        "reason": ctx.reason.value,
        "matched_acronyms": ctx.matched_acronyms,
        "user_text": ctx.scan_text,
    })


# ==================================================================== #
# RESPONSE event: reparse the assembled SSE bytes (mirrors
# streaming.collect_response's own parsing exactly; duplicated here rather
# than imported since those helpers are private to streaming.py and that
# module's forwarding logic is correctness-critical / deliberately untouched).
# ==================================================================== #
_SSE_SEP = "\n\n"


def _extract_content(frame_obj: dict) -> str:
    try:
        delta = frame_obj["choices"][0].get("delta", {})
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_finish_reason(frame_obj: dict) -> Optional[str]:
    try:
        return frame_obj["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _parse_sse_bytes(raw: bytes) -> tuple[str, Optional[str]]:
    """Reparse a complete SSE byte stream into (full_text, finish_reason)."""
    text = raw.decode("utf-8", errors="replace")
    parts: list[str] = []
    finish_reason: Optional[str] = None
    for frame in text.split(_SSE_SEP):
        stripped = frame.strip()
        if not stripped or stripped == "data: [DONE]":
            continue
        obj = parse_sse_data_line(stripped)
        if obj is None:
            continue
        parts.append(_extract_content(obj))
        fr = _extract_finish_reason(obj)
        if fr:
            finish_reason = fr
    return "".join(parts), finish_reason


_NOTICE_FIELDS = (
    "NOTICE_LENGTH_HANDOFF",
    "NOTICE_OVERSIZE_DIVERT",
    "NOTICE_IMAGE_DIVERT",
    "NOTICE_LLAMA_UNAVAILABLE",
    "NOTICE_BACKEND_UNAVAILABLE",
)


def _notices_shown(answer_text: str) -> list[str]:
    """Which known notice constants appear verbatim in the assembled answer."""
    shown = []
    for name in _NOTICE_FIELDS:
        text = getattr(settings, name).strip()
        if text and text in answer_text:
            shown.append(name)
    return shown


async def log_response(
    ctx: LogContext, target: Target, answer_text: str, finish_reason: Optional[str]
) -> None:
    await log_event({
        "type": "response",
        "correlation_id": ctx.correlation_id,
        "ts": _now_iso(),
        "model_served": target.value,
        "answer_text": answer_text,
        "finish_reason": finish_reason,
        "length_handoff": settings.NOTICE_LENGTH_HANDOFF.strip() in answer_text,
        "notices_shown": _notices_shown(answer_text),
    })


async def log_response_from_result(ctx: LogContext, target: Target, result: dict) -> None:
    """RESPONSE event for the non-streaming path: collect_response already
    assembled everything needed into `result`."""
    choice = result["choices"][0]
    await log_response(
        ctx, target, choice["message"]["content"], choice.get("finish_reason")
    )


async def tee_and_log_stream(
    stream: AsyncIterator[bytes], ctx: LogContext, target: Target
) -> AsyncIterator[bytes]:
    """
    Forward every chunk to the caller UNCHANGED — zero added latency, zero
    risk to byte fidelity — while accumulating a copy for a post-hoc
    reparse. Logs once the stream ends, however it ends (normal completion,
    client disconnect, or an exception): a partial answer is still worth a
    RESPONSE event with whatever finish_reason (if any) was captured.

    The log write happens AFTER every byte has already been yielded to the
    caller, so awaiting it directly here (rather than spawning yet another
    background task) adds no participant-visible delay.
    """
    buffer = bytearray()
    try:
        async for chunk in stream:
            yield chunk
            buffer += chunk
    finally:
        answer_text, finish_reason = _parse_sse_bytes(bytes(buffer))
        await log_response(ctx, target, answer_text, finish_reason)


# ==================================================================== #
# Evaluation-only divergence events (objective 3). Never part of the topic
# corpus (that's `user_text` on the REQUEST event, captured pre-transform).
# ==================================================================== #
async def log_summarization_event(
    ctx: LogContext, cutoff_before: int, cutoff_after: int, summary_text: str
) -> None:
    """
    Summarization folded more turns into the summary this request. Logs the
    FULL summary text: the summarizer call is non-deterministic, so this log
    line is the only surviving record of what the model actually saw at this
    turn — it cannot be regenerated later from history + the algorithm.
    """
    await log_event({
        "type": "eval_summarization",
        "correlation_id": ctx.correlation_id,
        "ts": _now_iso(),
        "cutoff_before": cutoff_before,
        "cutoff_after": cutoff_after,
        "summary_text": summary_text,
    })


async def log_image_event(ctx: LogContext, mode: str, image_count: int) -> None:
    """mode: "native_gemma" (fresh image, answered natively) or
    "history_rewrite" (history images replaced with text descriptions)."""
    await log_event({
        "type": "eval_image",
        "correlation_id": ctx.correlation_id,
        "ts": _now_iso(),
        "mode": mode,
        "image_count": image_count,
    })
