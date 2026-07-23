"""
backends.py — Async HTTP clients for the Llama and Gemma vLLM backends.

Intentionally THIN and LITERAL: a faithful mirror of vLLM's OpenAI-compatible
server API. One of only two modules that touch the network (the other is
streaming.py). The pure-logic modules (summarization, routing, images,
terminology) receive their network needs from here as INJECTED ASYNC callables,
so the hard-to-verify I/O is confined to this file.

Async-only design (critical):
  - Every injected capability is `async` and is `await`ed by the pure modules.
  - There is NO asyncio.run and NO sync->async bridge anywhere. This eliminates
    the "event loop already running" RuntimeError class of bug and the per-call
    loop teardown that would otherwise break httpx connection pooling.

Capabilities provided (matching the injected signatures used elsewhere):
  - tokenize_count(messages, target) -> int                 (authoritative count)
  - exact_token_counter(messages, target=LLAMA) -> int      (routing side)
  - llama_token_counter_hybrid(messages) -> int             (summarization side)
  - summarize_chunk(turns) -> str                           (summarization side)
  - describe_image_async(bytes, mime) -> str                (images side)
  - chat_once(target, messages, ...) -> str                 (non-streaming gen)
  - open_chat_stream(target, messages, ...) -> AsyncIterator[bytes]  (streaming)
  - clamp_llama_max_tokens(prompt_tokens, requested) -> int (prevent vLLM 400s)
  - parse_sse_data_line(line) -> Optional[dict]             (shared with streaming)

vLLM endpoints used (per instance):
  - POST {base}/v1/chat/completions   (chat, streaming or not)
  - POST {base}/tokenize              (authoritative token count)

Timeouts:
  - Short calls (tokenize, describe, summarize, non-streaming chat) use
    HTTP_READ_TIMEOUT.
  - The STREAMING chat call uses NO read timeout (it can run long) and relies on
    the disconnect watch in streaming.py to cancel it (closing the generator
    closes the HTTP connection and frees the GPU).

Failure contracts (must match callers' expectations):
  - describe_image_async returns "" on failure (images.py -> placeholder / cold
    cache).
  - tokenize_count / exact_token_counter / llama_token_counter_hybrid RAISE on
    failure (routing/summarization fail safe -> Gemma).
  - summarize_chunk returns a best-effort string; on failure returns an honest,
    sanitized fallback (never leaks internals) so the summarization loop can
    still advance.
  - chat_once returns "" on failure (internal, non-interactive generations).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import settings
from app.routing import Target, estimate_tokens

logger = logging.getLogger(__name__)

Message = dict[str, Any]


# ==================================================================== #
# Shared async client lifecycle (created/closed in main.py lifespan).
# ==================================================================== #
_client: Optional[httpx.AsyncClient] = None


def init_http_client() -> httpx.AsyncClient:
    """
    Create the shared AsyncClient with connection pooling. Called ONCE at startup
    from main.py's lifespan. A single shared client (not one per request) is the
    correct, performant pattern.

    The default timeout here suits SHORT calls; the streaming chat call overrides
    the read timeout to None explicitly (see open_chat_stream).
    """
    global _client
    timeout = httpx.Timeout(
        connect=settings.HTTP_CONNECT_TIMEOUT,
        read=settings.HTTP_READ_TIMEOUT,
        write=settings.HTTP_READ_TIMEOUT,
        pool=settings.HTTP_CONNECT_TIMEOUT,
    )
    _client = httpx.AsyncClient(timeout=timeout)
    logger.info("HTTP client initialized.")
    return _client


async def close_http_client() -> None:
    """Close the shared client at shutdown (from main.py's lifespan)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("HTTP client closed.")


def get_client() -> httpx.AsyncClient:
    """
    Return the initialized client, or RAISE if init was skipped.

    We deliberately do NOT lazily auto-initialize: on a deploy-blind service,
    explicit failure surfaces a wiring bug loudly rather than silently creating a
    client that may be bound to the wrong event loop.
    """
    if _client is None:
        raise RuntimeError(
            "HTTP client not initialized. Call init_http_client() at startup."
        )
    return _client


# ==================================================================== #
# Target -> URLs / served model name.
# ==================================================================== #
def _chat_url(target: Target) -> str:
    return settings.LLAMA_CHAT_URL if target == Target.LLAMA else settings.GEMMA_CHAT_URL


def _tokenize_url(target: Target) -> str:
    return (
        settings.LLAMA_TOKENIZE_URL
        if target == Target.LLAMA
        else settings.GEMMA_TOKENIZE_URL
    )


def _served_model_name(target: Target) -> str:
    return (
        settings.LLAMA_SERVED_MODEL_NAME
        if target == Target.LLAMA
        else settings.GEMMA_SERVED_MODEL_NAME
    )


# ==================================================================== #
# Tokenization (authoritative count).
# ==================================================================== #
async def tokenize_count(messages: list[Message], target: Target) -> int:
    """
    Exact token count for `messages` as the TARGET tokenizes them, including
    chat-template overhead, via vLLM's /tokenize endpoint.

    We send a chat-shaped request with add_generation_prompt=True so the count
    reflects a real completion. vLLM returns {"count": N, "tokens": [...], ...}
    across versions; we prefer "count" and fall back to len("tokens").

    RAISES on any failure (network, non-200, malformed) so callers fail safe
    toward Gemma. Confirm the exact schema against a captured /tokenize fixture.
    """
    client = get_client()
    payload = {
        "model": _served_model_name(target),
        "messages": messages,
        "add_generation_prompt": True,
    }
    resp = await client.post(_tokenize_url(target), json=payload)
    resp.raise_for_status()
    data = resp.json()
    count = data.get("count")
    if count is None:
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            count = len(tokens)
    if not isinstance(count, int):
        raise ValueError(f"Unexpected /tokenize response shape: {data!r}")
    return count



async def llama_token_counter_hybrid(messages: list[Message]) -> int:
    """
    Async hybrid TokenCounter for summarization: chars/4 for the loop interior,
    exact /tokenize only near the ceiling. Awaited by summarization.py.

      - Fast estimate at/below the fast-path threshold -> return the estimate.
      - Otherwise -> exact Llama count.

    This bounds how often we hit /tokenize during the incremental summarization
    loop (only when genuinely near the ceiling).
    """
    est = estimate_tokens(messages)
    if est <= settings.LLAMA_FAST_PATH_TOKENS:
        return est
    return await tokenize_count(messages, Target.LLAMA)


# ==================================================================== #
# Non-streaming chat (summaries, and length-handoff continuation if desired).
# ==================================================================== #
async def chat_once(
    target: Target,
    messages: list[Message],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Single NON-streaming chat completion; returns the assistant text.

    Used for internal, non-interactive generations (summaries). Returns "" on
    failure (callers degrade). Errors are logged WITHOUT leaking internals to any
    user-visible path.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "model": _served_model_name(target),
        "messages": messages,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        resp = await client.post(_chat_url(target), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001 - non-interactive; degrade to ""
        logger.error("chat_once(%s) failed: %s", target.value, exc)
        return ""


# ==================================================================== #
# Summarization capability (injected into summarization.py) — async.
# ==================================================================== #
async def summarize_chunk(turns: list[Message]) -> str:
    """
    Async ChunkSummarizer: summarize `turns` via GEMMA (headroom to hold the raw
    chunk). Returns a best-effort string; on failure returns an honest, sanitized
    fallback so the summarization loop can still advance (never leaks internals).
    """
    prompt = _build_summary_prompt(turns)
    result = await chat_once(
        Target.GEMMA,
        prompt,
        max_tokens=settings.SUMMARY_MAX_TOKENS,
        temperature=0.2,  # low temp for faithful, stable summaries
    )
    if result.strip():
        return result.strip()
    logger.warning("Summarization returned empty; using fallback marker.")
    return "[Earlier conversation could not be summarized reliably.]"


def _build_summary_prompt(turns: list[Message]) -> list[Message]:
    """
    Build the Gemma request that summarizes `turns`. We render the turns as plain
    text inside one user message so Gemma summarizes them as CONTENT rather than
    trying to continue the conversation.
    """
    rendered_lines: list[str] = []
    for t in turns:
        role = t.get("role", "unknown")
        content = t.get("content", "")
        if isinstance(content, list):
            # Content should already be text here; join text parts.
            text = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = str(content)
        rendered_lines.append(f"{role.upper()}: {text}")

    conversation_text = "\n".join(rendered_lines)
    instruction = (
        "Summarize the following conversation excerpt for use as background "
        "context in a strategic wargame discussion. Preserve intent, decisions, "
        "postures, and reasoning. Omit pleasantries and exact figures that are "
        "not decision-relevant. Write a single concise paragraph.\n\n"
        "CONVERSATION EXCERPT:\n"
        f"{conversation_text}"
    )
    return [{"role": "user", "content": instruction}]


# ==================================================================== #
# Image description (injected into images.py) — async.
# ==================================================================== #
def _image_to_data_uri(image_bytes: bytes, mime_type: Optional[str]) -> str:
    """Re-encode raw image bytes to a base64 data URI for the vision request."""
    mime = mime_type or "image/png"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_describe_prompt(image_bytes: bytes, mime_type: Optional[str]) -> list[Message]:
    """
    Build the Gemma vision request asking for a thorough factual description of
    one image. The description substitutes for the image when the chat later
    moves to Llama, so we ask for detail without speculation.
    """
    data_uri = _image_to_data_uri(image_bytes, mime_type)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this image in thorough, factual detail so that "
                        "someone who cannot see it could reason about its content. "
                        "Include any visible text, labels, symbols, spatial "
                        "relationships, and notable features. Do not speculate "
                        "beyond what is visible."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]


async def describe_image_async(
    image_bytes: bytes, mime_type: Optional[str] = None
) -> str:
    """
    Async image describer (images.AsyncImageDescriber AND the history-rewrite
    describer): sends the image to Gemma, returns a text description or "" on
    failure (images.py turns "" into a placeholder / leaves the cache cold).

    NOTE: images.py enforces settings.MAX_IMAGE_BYTES BEFORE calling this, so we
    never base64-encode an oversized image here (memory guardrail for IL5
    containers). We keep this function unconditional and simple.
    """
    prompt = _build_describe_prompt(image_bytes, mime_type)
    # Descriptions need not be huge; cap them to keep latency/memory bounded.
    return await chat_once(Target.GEMMA, prompt, max_tokens=1024, temperature=0.2)


# ==================================================================== #
# max_tokens clamp (prevent vLLM 400s on Llama).
# ==================================================================== #
def clamp_llama_max_tokens(
    prompt_tokens: int, requested_max_tokens: Optional[int]
) -> int:
    """
    Compute a safe max_tokens for a Llama request so that
        prompt_tokens + max_tokens <= LLAMA_MAX_MODEL_LEN
    (vLLM rejects the request with a 400 otherwise).

    Returns min(requested, room) if a max_tokens was requested, else the full
    remaining room. Never negative; 0 means "no room" (should have diverted — a
    safety net, not an expected path).
    """
    room = settings.llama_max_gen_tokens(prompt_tokens)
    if requested_max_tokens is None:
        return room
    return max(0, min(requested_max_tokens, room))


# ==================================================================== #
# Streaming chat (consumed by streaming.py). Yields RAW response bytes.
# ==================================================================== #
async def open_chat_stream(
    target: Target,
    messages: list[Message],
    max_tokens: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> AsyncIterator[bytes]:
    """
    Open a STREAMING chat completion against `target` and yield RAW response byte
    chunks (via aiter_bytes) as they arrive — no decode/re-encode round-trip.

    streaming.py consumes these bytes to:
      (a) forward them to OWUI byte-faithfully, and
      (b) assemble SSE frames on a lightweight buffer to inspect finish_reason
          (using parse_sse_data_line) for the length-handoff logic.

    Cancellation: streaming can run long, so this uses NO read timeout. The
    disconnect watch in streaming.py cancels by closing this async generator,
    which closes the underlying HTTP connection and frees the GPU.

    `extra` allows passing through selected OpenAI fields (e.g. top_p) if the
    pipeline chooses to forward them.

    Usage:
        async for chunk in open_chat_stream(...):
            ... forward raw chunk; buffer+parse a copy for finish_reason ...
    """
    client = get_client()
    payload: dict[str, Any] = {
        "model": _served_model_name(target),
        "messages": messages,
        "stream": True,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra:
        payload.update(extra)

    # No read timeout for the stream; keep connect timeout.
    stream_timeout = httpx.Timeout(
        connect=settings.HTTP_CONNECT_TIMEOUT,
        read=None,
        write=settings.HTTP_READ_TIMEOUT,
        pool=settings.HTTP_CONNECT_TIMEOUT,
    )

    async with client.stream(
        "POST", _chat_url(target), json=payload, timeout=stream_timeout
    ) as resp:
        resp.raise_for_status()
        # Raw bytes: byte-faithful forwarding + zero re-encode overhead.
        async for chunk in resp.aiter_bytes():
            if chunk:
                yield chunk


# ==================================================================== #
# SSE parsing helper (shared with streaming.py).
# ==================================================================== #
def parse_sse_data_line(line: str) -> Optional[dict]:
    """
    Parse a single SSE line of the form 'data: {json}' into a dict, or return
    None for non-data lines, the '[DONE]' sentinel, or malformed JSON.

    streaming.py uses this on complete lines it assembles from the raw byte
    stream, to inspect finish_reason while forwarding bytes untouched.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None