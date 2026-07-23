"""
summarization.py — Incremental, lazy summarization (README Step 4).

Async note:
  The injected capabilities (token counter, chunk summarizer) are ASYNC and are
  awaited here, so this module runs entirely on the single event loop (no
  asyncio.run, no thread bridges). The ALGORITHM is otherwise unchanged from the
  reviewed synchronous version.

Goal:
  Keep participants in Llama's stronger reasoning mode as long as possible by
  summarizing OLD messages while keeping RECENT messages raw, shrinking the
  prompt to fit Llama's usable PROMPT BUDGET (settings.LLAMA_CEILING =
  max_model_len - reserved_generation).

Core invariants (verify by reading — the whole correctness story):
  I1. `summarized_up_to` ONLY moves forward. We never un-summarize.
  I2. We NEVER summarize into the protected recent tail. The furthest we may
      advance the cutoff is  len(history) - RAW_TAIL_TURNS  (the "max cutoff").
  I3. Summarization is LAZY: it runs ONLY when the Llama-bound prompt does not
      fit. Gemma-bound requests never trigger summarization.
  I4. The prompt sent to Llama is  [summary_message] + history[summarized_up_to:].
  I5. `summary_text` grows monotonically as a concatenation of chunk summaries,
      except for the rare full re-summarization reset (README Step 4).

Early-exit optimization (E1, corrected):
  Before summarizing anything, check whether the SMALLEST achievable prompt —
  the EXACT protected raw tail plus a COMPACTED SUMMARY FLOOR
  (min(current summary tokens, SUMMARY_MAX_TOKENS)) — already exceeds
  LLAMA_CEILING. If so, diverting to Gemma is inevitable, so we raise
  DoesNotFitInLlama IMMEDIATELY with zero Gemma calls and zero state mutation.
  The compacted floor accounts for the fact that a full re-summarization can
  SHRINK the summary, so E1 never falsely diverts a conversation that could
  actually fit. The raw-tail term is exact, so E1 never fails to divert a
  genuinely-too-large tail.

Budget semantics:
  settings.LLAMA_CEILING is the PROMPT budget, already net of reserved
  generation. Fitting under it guarantees room to answer. This module measures
  PROMPT size only.

Terminology:
  A "turn" == ONE message (one dict in history), NOT a user+assistant pair.
  RAW_TAIL_TURNS counts MESSAGES.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.config import settings
from app.state import ConversationState, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Injected capability signatures (ASYNC).
# ---------------------------------------------------------------------- #
class TokenCounter(Protocol):
    """Async: return the token count for a list of OpenAI-style messages."""
    async def __call__(self, messages: list[Turn]) -> int: ...


class ChunkSummarizer(Protocol):
    """Async: summarize a list of raw turns into a single condensed paragraph."""
    async def __call__(self, turns: list[Turn]) -> str: ...


# ---------------------------------------------------------------------- #
# Sentinel: "cannot fit in Llama; the pipeline must divert to Gemma".
# ---------------------------------------------------------------------- #
class DoesNotFitInLlama(Exception):
    """
    Raised when the Llama-bound prompt cannot be brought under LLAMA_CEILING,
    either because the protected raw tail (plus compacted summary floor) is too
    large (early exit E1) or because summarizing down to that tail is still
    insufficient. The pipeline catches this and reroutes to Gemma (README
    Step 6: oversize -> Gemma), attaching NOTICE_OVERSIZE_DIVERT.
    """


# ---------------------------------------------------------------------- #
# Payload assembly helpers (pure; no mutation, no I/O).
# ---------------------------------------------------------------------- #
def _summary_message(summary_text: str) -> Turn:
    """
    Wrap the accumulated summary as a single system message that precedes the raw
    tail. System role frames it as background context, not a live utterance.
    """
    return {
        "role": "system",
        "content": (
            "Summary of earlier conversation (older messages condensed for "
            "context; fine detail may be lost):\n" + summary_text
        ),
    }


def _payload_at_cutoff(state: ConversationState, cutoff: int) -> list[Turn]:
    """
    The message list that WOULD be sent to Llama if the summary cutoff were
    `cutoff`:  [summary_message?] + history[cutoff:]. Pure hypothetical used for
    both the real payload and the E1 minimal-tail estimate.
    """
    raw_tail = state.history[cutoff:]
    if state.summary_text:
        return [_summary_message(state.summary_text)] + raw_tail
    return list(raw_tail)


def build_llama_payload(state: ConversationState) -> list[Turn]:
    """The message list to send to Llama given current state (Invariant I4). Pure read."""
    return _payload_at_cutoff(state, state.summarized_up_to)


def build_gemma_tail(state: ConversationState) -> list[Turn]:
    """
    For Gemma-bound requests we do NOT summarize (Invariant I3). Gemma has the
    headroom, so we send [summary_message?] + history[summarized_up_to:]. Any
    pre-existing summary is included so a chat trimmed for Llama and then
    diverted to Gemma keeps earlier context; we never advance the cutoff.
    """
    return _payload_at_cutoff(state, state.summarized_up_to)


def _max_cutoff(history_len: int) -> int:
    """
    Furthest `summarized_up_to` may advance so the last RAW_TAIL_TURNS messages
    stay raw (Invariant I2). Never negative.
    """
    return max(0, history_len - settings.RAW_TAIL_TURNS)


# ---------------------------------------------------------------------- #
# E1 helpers (async, because they await the injected counter).
# ---------------------------------------------------------------------- #
async def _summary_floor_tokens(
    state: ConversationState, count_tokens: TokenCounter
) -> int:
    """
    Smallest token count the summary portion could occupy in any achievable
    Llama payload:
        min(current summary tokens, SUMMARY_MAX_TOKENS)
    because a full re-summarization compacts the summary to at most
    SUMMARY_MAX_TOKENS. Using this as the E1 lower bound guarantees E1 never
    over-estimates the minimal payload (so it never falsely diverts).
    """
    if not state.summary_text:
        return 0
    current = await count_tokens([_summary_message(state.summary_text)])
    if current <= settings.SUMMARY_MAX_TOKENS:
        return current
    return settings.SUMMARY_MAX_TOKENS


async def _minimal_payload_tokens(
    state: ConversationState, count_tokens: TokenCounter
) -> int:
    """
    Conservative LOWER BOUND on the smallest achievable Llama prompt while
    honoring the protected raw tail:
        tokens(history[max_cutoff:])  +  summary_floor_tokens
    The raw-tail portion is EXACT and unavoidable; the summary portion uses the
    compacted floor. If this exceeds LLAMA_CEILING, no allowed summarization
    (including full re-summarization) can help.
    """
    max_cutoff = _max_cutoff(len(state.history))
    raw_tail = state.history[max_cutoff:]
    raw_tail_tokens = await count_tokens(raw_tail) if raw_tail else 0
    return raw_tail_tokens + await _summary_floor_tokens(state, count_tokens)


# ---------------------------------------------------------------------- #
# The core algorithm (async).
# ---------------------------------------------------------------------- #
async def prepare_for_llama(
    state: ConversationState,
    count_tokens: TokenCounter,
    summarize_chunk: ChunkSummarizer,
    *,
    reserved_overhead: int = 0,
) -> list[Turn]:
    """
    Ensure the Llama-bound prompt fits under settings.LLAMA_CEILING, summarizing
    older messages incrementally as needed, and return the final prompt.

    Preconditions:
      - The pipeline has ALREADY appended the latest user message to
        state.history before calling this.
      - The caller SHOULD hold state.lock so the read/mutate/read sequence is
        atomic for this conversation.

    Mutates state.summary_text / state.summarized_up_to only when it must
    summarize more (forward-only, Invariant I1).

    Raises DoesNotFitInLlama when the prompt cannot be made to fit (E1, or after
    summarizing down to the protected tail).
    """
    # reserved_overhead: tokens the CALLER will add to the payload after this
    # function returns (e.g. the terminology system block). Subtracting it here
    # keeps the fit check honest about the final prompt size. Never negative.
    ceiling = max(0, settings.LLAMA_CEILING - reserved_overhead)
    max_cutoff = _max_cutoff(len(state.history))

    # --- Fast path (Invariant I3: lazy) ---
    current_payload = build_llama_payload(state)
    if await count_tokens(current_payload) <= ceiling:
        return current_payload

    # --- Early minimal-tail pre-check (E1, corrected) ---
    if await _minimal_payload_tokens(state, count_tokens) > ceiling:
        logger.info(
            "Minimal achievable Llama prompt (exact raw tail from index %d + "
            "compacted summary floor) exceeds prompt budget (%d); diverting to "
            "Gemma with no summarization.",
            max_cutoff,
            ceiling,
        )
        raise DoesNotFitInLlama(
            "Protected raw tail plus compacted summary floor exceeds the "
            "Llama prompt budget."
        )

    # --- Incremental summarization loop ---
    while await count_tokens(build_llama_payload(state)) > ceiling:

        # Given E1 passed, reaching max_cutoff should already fit; this guard
        # guarantees termination and defends against non-monotonic token counts.
        if state.summarized_up_to >= max_cutoff:
            logger.warning(
                "Reached max_cutoff (%d) but payload still over budget; "
                "diverting to Gemma. (Unexpected given E1 pre-check passed.)",
                max_cutoff,
            )
            raise DoesNotFitInLlama(
                "Summarized to the protected tail but still over budget."
            )

        # Fold the earliest portion of the summarizable range (default: half).
        summarizable_len = max_cutoff - state.summarized_up_to
        chunk_len = max(1, round(summarizable_len * settings.SUMMARY_CHUNK_FRACTION))
        chunk_end = min(state.summarized_up_to + chunk_len, max_cutoff)

        chunk_turns = state.history[state.summarized_up_to : chunk_end]

        new_chunk_summary = (await summarize_chunk(chunk_turns)).strip()

        if state.summary_text:
            state.summary_text = (
                state.summary_text + " " + new_chunk_summary
            ).strip()
        else:
            state.summary_text = new_chunk_summary
        state.summarized_up_to = chunk_end  # forward-only; <= max_cutoff (I2)

        logger.debug(
            "Summarized turns [%d:%d]; cutoff now %d/%d; summary chars=%d",
            chunk_end - len(chunk_turns),
            chunk_end,
            state.summarized_up_to,
            max_cutoff,
            len(state.summary_text),
        )

        # Rare guardrail (I5): compact an over-large summary via full re-summary.
        if await _summary_too_large(state, count_tokens):
            await _full_resummarize(state, summarize_chunk)

    return build_llama_payload(state)


async def _summary_too_large(
    state: ConversationState, count_tokens: TokenCounter
) -> bool:
    """True if summary_text alone (as the system message we send) exceeds SUMMARY_MAX_TOKENS."""
    if not state.summary_text:
        return False
    return (
        await count_tokens([_summary_message(state.summary_text)])
        > settings.SUMMARY_MAX_TOKENS
    )


async def _full_resummarize(
    state: ConversationState, summarize_chunk: ChunkSummarizer
) -> None:
    """
    Collapse ALL already-summarized turns (history[0:summarized_up_to]) into a
    single fresh, tighter summary, replacing the accumulated concatenation. The
    protected raw tail is untouched and `summarized_up_to` is unchanged (same
    range, tighter summary). Rare path (README Step 4, Invariant I5). We
    re-summarize the ORIGINAL turns to avoid summary-of-summary drift.
    """
    cutoff = state.summarized_up_to
    if cutoff <= 0:
        return
    original_range = state.history[0:cutoff]
    state.summary_text = (await summarize_chunk(original_range)).strip()
    logger.info(
        "Full re-summarization: compacted %d turns into %d chars.",
        len(original_range),
        len(state.summary_text),
    )