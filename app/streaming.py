"""
streaming.py — SSE streaming, disconnect handling, and length-handoff (Step 7).

Forwarding strategy (correctness-critical):
  - SAFE COMMON PATHS (Gemma-primary; Llama-primary with handoff disabled):
      pure RAW passthrough — fast, byte-faithful, nothing to edit. Because we do
      not decode on this path, multi-byte UTF-8 characters pass through intact.
  - HANDOFF-CAPABLE PATH (Llama-primary, handoff enabled) and the Gemma
    CONTINUATION: FRAME-LEVEL forwarding — assemble complete SSE frames and
    re-emit content as our own chunks, so we can suppress [DONE] and inject a
    notice + continuation. These paths DECODE bytes to parse JSON and therefore
    use a STATEFUL INCREMENTAL UTF-8 DECODER so a multi-byte character split
    across two network chunks is buffered, not corrupted.

Sequencing on a Llama length cutoff:
  Llama partial (streamed) -> notice (streamed) -> Gemma continuation (streamed).
"""

from __future__ import annotations

import codecs
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import Request

from app.backends import open_chat_stream, parse_sse_data_line
from app.config import settings
from app.routing import Target

logger = logging.getLogger(__name__)

Message = dict[str, Any]

# SSE frame separator. vLLM emits "data: {...}\n\n". Confirm against a fixture.
_SSE_SEP = "\n\n"
_DONE = b"data: [DONE]\n\n"
# Frame-anchored [DONE] marker for the passthrough best-effort signal.
_DONE_MARKER = b"data: [DONE]"


# ==================================================================== #
# Synthetic SSE chunk construction.
# ==================================================================== #
def _sse_content_chunk(text: str, model: str, chunk_id: str, created: int) -> bytes:
    obj = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _sse_final_chunk(model: str, chunk_id: str, created: int, finish_reason: str = "stop") -> bytes:
    obj = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


# ==================================================================== #
# Parsing helpers over assembled SSE frames.
# ==================================================================== #
def _extract_delta_content(frame_obj: dict) -> str:
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


def _new_utf8_decoder() -> codecs.IncrementalDecoder:
    """
    Fresh stateful incremental UTF-8 decoder. Buffers incomplete trailing byte
    sequences across chunks so a multi-byte character split across network chunk
    boundaries is decoded correctly rather than mangled into replacement chars.
    errors='replace' still applies to GENUINELY malformed bytes (not boundary
    splits), so real corruption degrades gracefully instead of raising.
    """
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def _process_residual_buffer(buffer: str) -> list[str]:
    """
    Parse any trailing text left in `buffer` after a stream ends WITHOUT a final
    _SSE_SEP (abrupt upstream termination). Returns the list of content deltas
    recovered from a trailing complete-but-unterminated frame, or [] if none.
    [DONE] residue is ignored.
    """
    stripped = buffer.strip()
    if not stripped or stripped == "data: [DONE]":
        return []
    obj = parse_sse_data_line(stripped)
    if obj is None:
        return []
    content = _extract_delta_content(obj)
    return [content] if content else []


# ==================================================================== #
# The main streaming entrypoint.
# ==================================================================== #
async def stream_response(
    request: Request,
    target: Target,
    messages: list[Message],
    *,
    max_tokens: Optional[int],
    original_messages_for_continuation: list[Message],
    user_notice: Optional[str] = None,
    extra: Optional[dict] = None,
    allow_length_handoff: bool = True,
) -> AsyncIterator[bytes]:
    """Stream a completion to OWUI; handle disconnect and (for Llama) length handoff."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())  # single timestamp for the whole response
    primary_model = (
        settings.LLAMA_SERVED_MODEL_NAME
        if target == Target.LLAMA
        else settings.GEMMA_SERVED_MODEL_NAME
    )

    handoff_capable = target == Target.LLAMA and allow_length_handoff

    # 1) Optional pre-answer notice.
    if user_notice:
        yield _sse_content_chunk(user_notice, primary_model, chunk_id, created)

    # 2) Stream the primary response.
    partial_content: list[str] = []
    hit_length_limit = False
    saw_done = False
    backend_error: Optional[Exception] = None

    stream = open_chat_stream(
        target, messages, max_tokens=max_tokens, extra=extra
    )
    try:
        if handoff_capable:
            # FRAME-LEVEL forwarding with incremental UTF-8 decoding.
            decoder = _new_utf8_decoder()
            buffer = ""
            try:
                async for raw in stream:
                    if await request.is_disconnected():
                        logger.info("Client disconnected; aborting Llama stream.")
                        return
                    buffer += decoder.decode(raw, final=False)
                    emit: list[str] = []
                    while _SSE_SEP in buffer:
                        frame, buffer = buffer.split(_SSE_SEP, 1)
                        stripped = frame.strip()
                        if not stripped:
                            continue
                        if stripped == "data: [DONE]":
                            saw_done = True
                            continue  # never forward; we finalize ourselves
                        obj = parse_sse_data_line(stripped)
                        if obj is None:
                            continue
                        content = _extract_delta_content(obj)
                        if content:
                            partial_content.append(content)
                            emit.append(content)
                        if _extract_finish_reason(obj) == "length":
                            hit_length_limit = True
                    if emit:
                        yield _sse_content_chunk("".join(emit), primary_model, chunk_id, created)

                # Flush decoder + process any trailing unterminated frame.
                buffer += decoder.decode(b"", final=True)
                residual = _process_residual_buffer(buffer)
                if residual:
                    partial_content.extend(residual)
                    yield _sse_content_chunk("".join(residual), primary_model, chunk_id, created)
            except httpx.HTTPError as exc:
                # Connection never opened, dropped mid-stream, or upstream
                # returned a non-2xx (raise_for_status). Anything already
                # yielded above stays in the response; we still owe the
                # participant a visible notice + a clean finalize below rather
                # than letting the exception truncate the SSE body silently.
                backend_error = exc
        else:
            # SAFE COMMON PATH: pure raw passthrough (byte-faithful, no decode).
            try:
                async for raw in stream:
                    if await request.is_disconnected():
                        logger.info("Client disconnected; aborting stream.")
                        return
                    if raw:
                        yield raw
                        if _DONE_MARKER in raw:  # frame-anchored, not bare [DONE]
                            saw_done = True
            except httpx.HTTPError as exc:
                backend_error = exc
    finally:
        await stream.aclose()  # always free the upstream connection / GPU

    # 3) Stop if the client is gone.
    if await request.is_disconnected():
        return

    # 3b) Backend failure (connect error, dropped connection, non-2xx status).
    # Headers are already committed by StreamingResponse at this point, so a
    # clean HTTP error status is not possible; a visible notice + graceful
    # finalize is the best available repair (never leave a silently truncated
    # response — README design philosophy: "repair, don't fail").
    if backend_error is not None:
        logger.error("Backend stream to %s failed: %s", target.value, backend_error)
        yield _sse_content_chunk(
            settings.NOTICE_BACKEND_UNAVAILABLE, primary_model, chunk_id, created
        )
        yield _sse_final_chunk(primary_model, chunk_id, created, finish_reason="stop")
        yield _DONE
        return

    # 4) Length handoff (Llama only).
    if handoff_capable and hit_length_limit:
        async for out in _perform_length_handoff(
            request,
            chunk_id=chunk_id,
            created=created,
            llama_partial="".join(partial_content),
            original_messages=original_messages_for_continuation,
            extra=extra,
        ):
            yield out
        return

    # 5) Finalize.
    if handoff_capable:
        yield _sse_final_chunk(primary_model, chunk_id, created, finish_reason="stop")
        yield _DONE
    elif not saw_done:
        yield _sse_final_chunk(primary_model, chunk_id, created, finish_reason="stop")
        yield _DONE


# ==================================================================== #
# Non-streaming sibling of stream_response (README: honor `stream: false`).
# ==================================================================== #
async def collect_response(
    request: Request,
    target: Target,
    messages: list[Message],
    *,
    max_tokens: Optional[int],
    original_messages_for_continuation: list[Message],
    user_notice: Optional[str] = None,
    extra: Optional[dict] = None,
    allow_length_handoff: bool = True,
) -> dict:
    """
    Non-streaming path: internally run stream_response and assemble its SSE
    output into a single chat.completion object. Joining ALL bytes before
    frame-splitting sidesteps chunk-boundary decoding entirely. Repair notices
    (diverts, length handoff) become part of the message content — intended.

    Deliberately omits `usage`: computing it correctly for every routing path
    would add a /tokenize round-trip per request for a cosmetic field. Most
    clients tolerate its absence. Revisit only if a client demands it.
    """
    primary_model = (
        settings.LLAMA_SERVED_MODEL_NAME
        if target == Target.LLAMA
        else settings.GEMMA_SERVED_MODEL_NAME
    )
    chunks = [
        c
        async for c in stream_response(
            request,
            target,
            messages,
            max_tokens=max_tokens,
            original_messages_for_continuation=original_messages_for_continuation,
            user_notice=user_notice,
            extra=extra,
            allow_length_handoff=allow_length_handoff,
        )
    ]
    text = b"".join(chunks).decode("utf-8", errors="replace")
    content_parts: list[str] = []
    finish_reason = "stop"
    for frame in text.split(_SSE_SEP):
        stripped = frame.strip()
        if not stripped or stripped == "data: [DONE]":
            continue
        obj = parse_sse_data_line(stripped)
        if obj is None:
            continue
        content_parts.append(_extract_delta_content(obj))
        fr = _extract_finish_reason(obj)
        if fr:
            finish_reason = fr
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": primary_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }
        ],
    }


# ==================================================================== #
# The length handoff: notice + Gemma continuation (frame-level, boundary-safe).
# ==================================================================== #
async def _perform_length_handoff(
    request: Request,
    *,
    chunk_id: str,
    created: int,
    llama_partial: str,
    original_messages: list[Message],
    extra: Optional[dict] = None,
) -> AsyncIterator[bytes]:
    """Stream (a) the length notice, (b) a Gemma continuation, (c) a final chunk + [DONE]."""
    gemma_model = settings.GEMMA_SERVED_MODEL_NAME

    # (a) Notice.
    yield _sse_content_chunk(settings.NOTICE_LENGTH_HANDOFF, gemma_model, chunk_id, created)

    # (b) Gemma continuation (frame-level + incremental UTF-8 decoding).
    continuation_messages = _build_continuation_messages(original_messages, llama_partial)
    decoder = _new_utf8_decoder()
    buffer = ""
    backend_error: Optional[Exception] = None
    stream = open_chat_stream(
        Target.GEMMA, continuation_messages, max_tokens=None, extra=extra
    )
    try:
        try:
            async for raw in stream:
                if await request.is_disconnected():
                    logger.info("Client disconnected during handoff; aborting.")
                    return
                buffer += decoder.decode(raw, final=False)
                emit: list[str] = []
                while _SSE_SEP in buffer:
                    frame, buffer = buffer.split(_SSE_SEP, 1)
                    stripped = frame.strip()
                    if not stripped:
                        continue
                    if stripped == "data: [DONE]":
                        continue  # suppress; we finalize ourselves
                    obj = parse_sse_data_line(stripped)
                    if obj is None:
                        continue
                    content = _extract_delta_content(obj)
                    if content:
                        emit.append(content)
                if emit:
                    yield _sse_content_chunk("".join(emit), gemma_model, chunk_id, created)

            # Flush decoder + process any trailing unterminated frame.
            buffer += decoder.decode(b"", final=True)
            residual = _process_residual_buffer(buffer)
            if residual:
                yield _sse_content_chunk("".join(residual), gemma_model, chunk_id, created)
        except httpx.HTTPError as exc:
            backend_error = exc
    finally:
        await stream.aclose()

    if await request.is_disconnected():
        return

    # (c) Finalize the combined message.
    if backend_error is not None:
        logger.error("Gemma continuation stream failed: %s", backend_error)
        yield _sse_content_chunk(
            settings.NOTICE_BACKEND_UNAVAILABLE, gemma_model, chunk_id, created
        )
    yield _sse_final_chunk(gemma_model, chunk_id, created, finish_reason="stop")
    yield _DONE


def _build_continuation_messages(
    original_messages: list[Message], llama_partial: str
) -> list[Message]:
    """Message list for Gemma to CONTINUE Llama's cut-off answer (no repeat, no preamble)."""
    messages = list(original_messages)
    messages.append({"role": "assistant", "content": llama_partial})
    messages.append(
        {
            "role": "system",
            "content": (
                "The previous assistant reply above was cut off before it "
                "finished. Continue the reply seamlessly from exactly where it "
                "stopped. Do not repeat any earlier text, do not restate the "
                "question, and do not add a preamble — simply continue the "
                "answer to completion."
            ),
        }
    )
    return messages