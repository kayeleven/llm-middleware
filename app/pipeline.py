"""
pipeline.py — Request orchestration (README Steps 1–7).

Locking (async): state locks are asyncio.Lock, acquired with `async with`,
because the pipeline holds a per-conversation lock across `await
prepare_for_llama(...)`. A threading.Lock held across await would block/deadlock
the event loop; asyncio.Lock yields correctly.

History model (REPLACE, not accumulate): OWUI resends the ENTIRE conversation
each request, so the incoming `messages` array is the authoritative, complete
history. Each turn we set  state.history = <text-only version of this request's
messages>  (replace), while  summarized_up_to / summary_text  PERSIST across
turns. OWUI's stable ordering keeps the summary indices aligned; replacing
self-heals any drift from OWUI's authoritative version.

Continuation basis (standardized, race-free): the length-handoff continuation is
based on `text_messages` — the request-local, text-only, image-rewritten FULL
history. It is immutable for the life of the request (no shared-state race) and
identical across all routing paths. Gemma has headroom, so no truncation needed.

System prompt safety: terminology context is MERGED into any existing leading
system message rather than inserted as a second system message, to avoid strict
chat-template / vLLM validation failures on multiple leading system prompts.

This module owns the tricky wiring: fire-and-forget warm-up (retained task +
done-callback), replace-under-lock, the DoesNotFitInLlama divert, the fresh-image
native-answer path, and the max_tokens clamp. All network I/O is delegated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import Request

from app import backends, images, terminology
from app.routing import (
    Reason,
    Target,
    decide_target,
    llama_unavailable_divert_decision,
    oversize_divert_decision,
)
from app.state import ConversationState, store
from app.summarization import (
    DoesNotFitInLlama,
    build_gemma_tail,
    prepare_for_llama,
)
from app.streaming import collect_response, stream_response

logger = logging.getLogger(__name__)

Message = dict[str, Any]


# ==================================================================== #
# Fire-and-forget task registry (prevents GC of background warm-ups and
# surfaces their exceptions to the log).
# ==================================================================== #
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    """
    Launch a fire-and-forget coroutine safely:
      - retain a reference so the task is not garbage-collected mid-flight,
      - remove the reference when done,
      - log any exception (a background failure must never surface to the user).
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        exc = t.exception() if not t.cancelled() else None
        if exc is not None:
            logger.error("Background task failed: %s", exc)

    task.add_done_callback(_done)


# ==================================================================== #
# Request parsing helpers.
# ==================================================================== #
def _extract_messages(body: dict) -> list[Message]:
    """Return the messages array (empty list if absent/malformed)."""
    msgs = body.get("messages")
    return msgs if isinstance(msgs, list) else []


# OpenAI sampling fields forwarded verbatim to the backend. Confirmed via OWUI
# 0.8.5 fixture capture: per-model Advanced Params arrive as top-level body
# fields. Deliberately EXCLUDES model/messages/max_tokens/stream — those have
# dedicated handling and must never be overridden by a generic passthrough.
_PASSTHROUGH_PARAMS = (
    "temperature", "top_p", "presence_penalty", "frequency_penalty", "stop", "seed",
)


def _passthrough_params(body: dict) -> dict:
    return {k: body[k] for k in _PASSTHROUGH_PARAMS if k in body}


def _conversation_key(body: dict, messages: list[Message]) -> str:
    """
    Stable per-conversation key: prefer OWUI's chat_id; else hash the first user
    message content. Defensive against non-serializable content.
    """
    chat_id = body.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        return chat_id
    seed = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            try:
                seed = json.dumps(content, sort_keys=True)[:512]
            except (TypeError, ValueError):
                seed = str(content)[:512]
            break
    return "anon-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _latest_user_text(messages: list[Message]) -> str:
    """
    Plain text of the LATEST user message for terminology scanning (Step 5).

    Intended behavior (per README Step 5: "scan the latest user message"): we
    scan only the current turn's user text, not the whole history — terminology
    is context for THIS turn's question, and re-scanning history would bloat the
    system block with stale definitions. Do not "fix" this to scan more.
    Handles string content and multimodal list content (joining text parts).
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return ""
    return ""


def _with_terminology(
    messages: list[Message],
    scan_text: str,
    precomputed_block: Optional[str] = None,
) -> list[Message]:
    """
    Add the terminology block (Step 5) as system context WITHOUT creating a
    second leading system message.

    - If no acronyms matched, return `messages` unchanged.
    - If the first message is already a system message (OWUI's own system prompt,
      OR a summary system message from summarization), MERGE the terminology
      block into that message's content. Two leading system messages can break
      strict chat templates / vLLM validation.
    - Otherwise, prepend a new system message.

    `precomputed_block`, when not None, is used as-is instead of recomputing via
    build_terminology_block (Branch C computes it once, up front, to size the
    fit check — see prepare_for_llama's reserved_overhead). Safe because
    build_terminology_block never returns "" (only None or a non-empty string).

    Returns a new list; does not mutate the input.
    """
    block = (
        precomputed_block
        if precomputed_block is not None
        else terminology.build_terminology_block(scan_text)
    )
    if not block:
        return messages

    if messages and messages[0].get("role") == "system":
        first = messages[0]
        existing = first.get("content", "")
        if isinstance(existing, list):
            existing = " ".join(
                p.get("text", "")
                for p in existing
                if isinstance(p, dict) and p.get("type") == "text"
            )
        merged = (str(existing).rstrip() + "\n\n" + block).strip()
        new_first = dict(first)
        new_first["content"] = merged
        return [new_first] + messages[1:]

    return [{"role": "system", "content": block}] + messages


async def _finish(
    request: Request,
    target: Target,
    messages: list[Message],
    *,
    stream_requested: bool,
    **kwargs,
):
    """
    Dispatch to the streaming or non-streaming completion path.

    Why this works: stream_response is an async GENERATOR function — calling
    it returns the generator object immediately without executing any of its
    body, so returning that object (unawaited) from this async def is correct.
    collect_response is a plain coroutine and IS awaited here. Callers
    uniformly write `return await _finish(...)` and receive either the SSE
    async generator or the assembled chat.completion dict.
    """
    if stream_requested:
        return stream_response(request, target, messages, **kwargs)
    return await collect_response(request, target, messages, **kwargs)


async def _divert_to_gemma(
    request: Request,
    state: ConversationState,
    text_messages: list[Message],
    scan_text: str,
    user_notice: Optional[str],
    stream_requested: bool,
    extra: Optional[dict],
) -> AsyncIterator[bytes]:
    """
    Shared Gemma-fallback path for a Llama-bound request that cannot proceed on
    Llama, used both for genuine oversize (DoesNotFitInLlama) and for Llama-side
    sizing failures (tokenize/backend errors). We never advance the summary
    cutoff on this path (build_gemma_tail only reads state).
    """
    async with state.lock:
        gemma_tail = build_gemma_tail(state)
    gemma_payload = _with_terminology(gemma_tail, scan_text)

    return await _finish(
        request,
        Target.GEMMA,
        gemma_payload,
        stream_requested=stream_requested,
        max_tokens=None,
        original_messages_for_continuation=text_messages,  # standardized
        user_notice=user_notice,
        extra=extra,
        allow_length_handoff=False,
    )


# ==================================================================== #
# Main entry: process a request and return an SSE generator.
# ==================================================================== #
async def process_request(body: dict, request: Request) -> AsyncIterator[bytes] | dict:
    """
    Run the full pipeline and return either an async SSE generator (streamed
    request) or an assembled chat.completion dict (non-streamed request), per
    `_finish`'s dispatch on `stream_requested`. main.py wraps the former in a
    StreamingResponse and the latter in a JSONResponse.
    """
    messages = _extract_messages(body)
    conv_key = _conversation_key(body, messages)

    # OpenAI spec: stream defaults to FALSE when omitted. OWUI sends it
    # explicitly on every captured fixture, so this default only affects
    # direct API callers — who per the spec expect a JSON object. (This is a
    # deliberate behavior change from the pre-fix middleware, which always
    # returned SSE.)
    stream_requested = bool(body.get("stream", False))
    extra = _passthrough_params(body)

    # ---- Step 3 (detect): fresh image in the current turn? ----
    fresh_image = images.current_turn_has_image(messages)

    # ---- Step 6 (preliminary target): routing rule tree. ----
    decision = decide_target(body, current_turn_has_image=fresh_image)

    # =============================================================== #
    # Branch A: fresh image -> Gemma answers NATIVELY + fire warm-up.
    # =============================================================== #
    if fresh_image:
        # Fire-and-forget warm-up (never blocks; never alters the answering payload).
        _spawn_background(
            images.warm_current_turn_images(messages, backends.describe_image_async)
        )

        scan_text = _latest_user_text(messages)
        gemma_messages = _with_terminology(messages, scan_text)

        # REPLACE history with the native (image-bearing) current messages. On
        # the NEXT turn, Step 3B rewrites that image to its now-warmed
        # description before summarization ever sees it.
        state = await store.get_or_create(conv_key)
        async with state.lock:
            state.history = list(messages)

        return await _finish(
            request,
            Target.GEMMA,
            gemma_messages,
            stream_requested=stream_requested,
            max_tokens=None,                             # Gemma has headroom
            original_messages_for_continuation=messages,  # moot (handoff off)
            user_notice=decision.user_notice,             # image-divert notice (if any)
            extra=extra,
            allow_length_handoff=False,
        )

    # =============================================================== #
    # Non-fresh-image path.
    # =============================================================== #

    # ---- Step 3B: rewrite HISTORY images to text (DB-first, cached). ----
    text_messages = await images.replace_history_images_with_descriptions(
        messages, backends.describe_image_async
    )

    # ---- Step 5: terminology scan text (text-only now). ----
    scan_text = _latest_user_text(text_messages)

    # =============================================================== #
    # Branch B0: background task -> Gemma, STATELESS. OWUI-internal
    # scaffolding is never a real turn: skip state.history entirely (no read,
    # no write) so a task call can never overwrite a real conversation's
    # state, and never creates an orphan ConversationState either.
    # =============================================================== #
    if decision.reason == Reason.BACKGROUND_TASK:
        gemma_payload = _with_terminology(text_messages, scan_text)
        return await _finish(
            request,
            Target.GEMMA,
            gemma_payload,
            stream_requested=stream_requested,
            max_tokens=None,
            original_messages_for_continuation=text_messages,
            user_notice=None,
            extra=extra,
            allow_length_handoff=False,
        )

    # ---- Per-conversation state handle. ----
    state = await store.get_or_create(conv_key)

    # =============================================================== #
    # Branch B: Gemma-bound (explicit Gemma).
    # =============================================================== #
    if decision.target == Target.GEMMA:
        async with state.lock:
            state.history = list(text_messages)
            gemma_tail = build_gemma_tail(state)
        gemma_payload = _with_terminology(gemma_tail, scan_text)

        return await _finish(
            request,
            Target.GEMMA,
            gemma_payload,
            stream_requested=stream_requested,
            max_tokens=None,
            original_messages_for_continuation=text_messages,  # standardized
            user_notice=decision.user_notice,                  # None for explicit
            extra=extra,
            allow_length_handoff=False,
        )

    # =============================================================== #
    # Branch C: Llama-bound (explicit Llama or Auto). Summarize to fit; on
    # DoesNotFitInLlama, divert to Gemma with the oversize notice.
    # =============================================================== #
    try:
        term_block = terminology.build_terminology_block(scan_text)
        # chars/4 estimate is sufficient here: the exact /tokenize on the FINAL
        # payload (below) still governs the max_tokens clamp; this only makes
        # the fit check aware of the block's approximate size.
        term_overhead = (len(term_block) // 4) if term_block else 0

        async with state.lock:
            # REPLACE history, then prepare (summarization contract). Replacing
            # keeps indices aligned with OWUI's stable ordering; summary state
            # (summarized_up_to / summary_text) persists across turns.
            state.history = list(text_messages)
            llama_payload = await prepare_for_llama(
                state,
                backends.llama_token_counter_hybrid,  # injected async counter
                backends.summarize_chunk,             # injected async summarizer
                reserved_overhead=term_overhead,
            )
        llama_payload = _with_terminology(llama_payload, scan_text, precomputed_block=term_block)

        # Clamp max_tokens against the exact prompt size (avoid vLLM 400).
        prompt_tokens = await backends.tokenize_count(llama_payload, Target.LLAMA)
        clamped_max = backends.clamp_llama_max_tokens(
            prompt_tokens, body.get("max_tokens")
        )
        if backends.clamp_llama_max_tokens(prompt_tokens, None) <= 0:
            # No generation room at all — the E1/summarization fit check should
            # have prevented this (estimation error safety net). Sending
            # max_tokens=0 would make vLLM reject the request; divert instead.
            raise DoesNotFitInLlama("Prompt leaves no generation room after clamping.")

        return await _finish(
            request,
            Target.LLAMA,
            llama_payload,
            stream_requested=stream_requested,
            max_tokens=clamped_max,
            # Continuation is the FULL text-only history (Gemma has headroom);
            # request-local => race-free and consistent across all paths.
            original_messages_for_continuation=text_messages,
            user_notice=None,                # explicit/auto Llama: silent
            extra=extra,
            allow_length_handoff=True,
        )

    except DoesNotFitInLlama:
        # Oversize divert -> Gemma with the oversize notice.
        logger.info("Request does not fit in Llama; diverting to Gemma.")
        divert = oversize_divert_decision()
        return await _divert_to_gemma(
            request, state, text_messages, scan_text, divert.user_notice,
            stream_requested, extra,
        )

    except (httpx.HTTPError, ValueError) as exc:
        # A Llama-side sizing call failed outright — either the injected token
        # counter inside prepare_for_llama (backends.llama_token_counter_hybrid)
        # or the post-summarization /tokenize clamp call. Both only ever raise
        # httpx.HTTPError (network/non-2xx) or ValueError (malformed /tokenize
        # response) — see backends.tokenize_count's documented failure contract.
        # This means Llama's vLLM is unreachable/erroring while Gemma may still
        # be healthy; fail safe toward Gemma (README §6) instead of surfacing an
        # opaque 500 for what would otherwise be a normal request.
        logger.error("Llama sizing call failed (%s); failing safe to Gemma.", exc)
        divert = llama_unavailable_divert_decision()
        return await _divert_to_gemma(
            request, state, text_messages, scan_text, divert.user_notice,
            stream_requested, extra,
        )