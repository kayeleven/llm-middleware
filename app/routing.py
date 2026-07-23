"""
routing.py — Routing decision + prompt sizing (README Step 6).

Responsibilities:
  1. Decide the routing TARGET for a request, applying the README's rules:
       - Hard overrides (evaluated first, can override explicit selection):
           * Fresh image in the current turn        -> Gemma
           * OWUI background task (title/tags/etc.)  -> Gemma
       - Explicit model selection honored next:
           * Explicit Gemma  -> Gemma
           * Explicit Llama  -> Llama (subject to the oversize check downstream)
       - "Auto" -> Llama by default (reasoning preference); the size-based
         diversion to Gemma is handled downstream (see the handoff note below).
  2. Provide PROMPT SIZING utilities:
       - A pure fast estimate (chars / 4).
       - A near-ceiling exact count via an INJECTED /tokenize capability.
       - A single "does this fit under the Llama prompt budget?" helper that uses
         the fast path when safe and the exact path only when near the ceiling.

Handoff / separation of concerns (READ THIS):
  - This module does NOT summarize and does NOT forward. It returns a decision.
  - The OVERSIZE -> Gemma diversion is NOT decided here in isolation, because
    whether a Llama-bound request fits depends on summarization. The flow is:
        1) routing.decide_target(...)  -> preliminary target (+ notice)
        2) if target is Llama, the pipeline calls summarization.prepare_for_llama
           which either returns a fitting payload OR raises DoesNotFitInLlama.
        3) On DoesNotFitInLlama, the pipeline re-routes to Gemma and attaches the
           oversize notice (routing.oversize_notice() provides the text).
  - This keeps the "can it fit?" question (which needs summarization) out of the
    pure decision tree, while the tree still owns the image/background/explicit
    rules and the sizing primitives.

Purity / injection:
  - NO network I/O. The exact token count is provided via an INJECTED callable:
        exact_count(messages, target) -> int
    backends.py implements it by calling the TARGET backend's /tokenize endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from app.config import settings

logger = logging.getLogger(__name__)

# Diagnostic counter: how many times a request was classified as a background
# task ONLY because chat_id was absent while the model id looked like a real
# participant selection. A burst of these is the signature of the OWUI
# chat_id-injection Filter being disabled/broken. Warn loudly for the first
# few, then go quiet (single event loop; no lock needed).
_MISSING_CHAT_ID_WARN_LIMIT = 5
_missing_chat_id_warnings = 0

Message = dict[str, Any]


# ==================================================================== #
# Routing target and decision result.
# ==================================================================== #
class Target(str, Enum):
    """Canonical routing targets. String-valued for easy logging/serialization."""
    LLAMA = "llama"
    GEMMA = "gemma"


class Reason(str, Enum):
    """Why a target was chosen — drives which (if any) user notice is shown."""
    FRESH_IMAGE = "fresh_image"            # image in current turn -> Gemma
    BACKGROUND_TASK = "background_task"     # OWUI title/tags/etc.  -> Gemma
    EXPLICIT_GEMMA = "explicit_gemma"       # user chose Gemma
    EXPLICIT_LLAMA = "explicit_llama"       # user chose Llama
    AUTO_LLAMA = "auto_llama"               # Auto default -> Llama
    OVERSIZE_DIVERT = "oversize_divert"     # set by pipeline after DoesNotFit
    LLAMA_UNAVAILABLE = "llama_unavailable"  # Llama-side sizing call failed


@dataclass(frozen=True)
class RouteDecision:
    """
    The outcome of a routing decision.

    - target: where to send the request.
    - reason: why (for logging and to select a user-facing notice).
    - user_notice: text to stream to the participant BEFORE the answer, or None.
      Only diversions that change behavior the user would notice carry a notice
      (image divert; oversize divert). Explicit/auto choices are silent.
    """
    target: Target
    reason: Reason
    user_notice: Optional[str] = None


# ==================================================================== #
# Model-selection parsing (OWUI "model" string -> intent).
# ==================================================================== #
class Selection(str, Enum):
    """The participant's model selection intent, parsed from the OWUI request."""
    LLAMA = "llama"
    GEMMA = "gemma"
    AUTO = "auto"


def parse_selection(model_id: Optional[str]) -> Selection:
    """
    Map the OWUI "model" string to a Selection intent using config's contract.

    Unknown / missing model ids default to AUTO (safest: let the middleware
    decide), which is also what OWUI background tasks and odd payloads will hit.
    Comparison is case-insensitive on the id to tolerate minor variations; the
    canonical ids come from config.
    """
    if not model_id:
        return Selection.AUTO
    mid = model_id.strip().lower()
    if mid == settings.MODEL_ID_LLAMA.lower():
        return Selection.LLAMA
    if mid == settings.MODEL_ID_GEMMA.lower():
        return Selection.GEMMA
    if mid == settings.MODEL_ID_AUTO.lower():
        return Selection.AUTO
    logger.debug("Unknown model id '%s'; defaulting to AUTO.", model_id)
    return Selection.AUTO


# ==================================================================== #
# Background-task detection (README Step 6).
# ==================================================================== #
def is_background_task(body: dict) -> bool:
    """
    True if this is an OWUI internal task call (title/tags/follow-up/tool
    selection), not a real participant turn.

    Fixture capture against real OWUI 0.8.5 traffic showed the documented
    "background_tasks" field NEVER appears in practice. Detection therefore
    uses layered signals, strongest first:

      1. PRIMARY: model == settings.MODEL_ID_TASKS. The OWUI admin pins the
         Task Model setting to this dedicated synthetic id. Config-driven and
         robust: if the pin is removed, tasks simply route by whatever model
         OWUI selects — degraded, never broken.
      2. SECONDARY (config-gated): chat_id absent. The companion OWUI Filter
         injects chat_id onto every real turn; task calls never carry it.
         Gated by TREAT_MISSING_CHAT_ID_AS_TASK because its failure mode is
         inverted — a broken Filter would misroute ALL real traffic. When this
         signal fires alone on a real-looking model id, we log a WARNING (rate
         limited) so a broken Filter is diagnosable from logs.
      3. LEGACY: the documented background_tasks dict, kept in case a future
         OWUI version reintroduces it. Never observed in fixtures.
    """
    global _missing_chat_id_warnings

    model_id = body.get("model")
    if isinstance(model_id, str) and model_id.strip().lower() == settings.MODEL_ID_TASKS.lower():
        return True

    if settings.TREAT_MISSING_CHAT_ID_AS_TASK and "chat_id" not in body:
        if _missing_chat_id_warnings < _MISSING_CHAT_ID_WARN_LIMIT:
            _missing_chat_id_warnings += 1
            logger.warning(
                "Request classified as background task via chat_id absence "
                "(model=%r). This is EXPECTED during healthy operation when "
                "OWUI's Task Model is not pinned to '%s' (signal-2-only "
                "config). If Task Model IS pinned, or if this was a real "
                "participant turn, the OWUI chat_id-injection Filter may be "
                "disabled/broken. (notice %d/%d, then suppressed)",
                model_id,
                settings.MODEL_ID_TASKS,
                _missing_chat_id_warnings,
                _MISSING_CHAT_ID_WARN_LIMIT,
            )
        return True

    bg = body.get("background_tasks")
    if isinstance(bg, dict):
        return any(bool(bg.get(flag)) for flag in settings.BACKGROUND_TASK_FLAGS)
    return False


# ==================================================================== #
# The preliminary target decision (pre-summarization).
# ==================================================================== #
def decide_target(
    body: dict,
    current_turn_has_image: bool,
) -> RouteDecision:
    """
    Decide the PRELIMINARY routing target, applying README Step 6 rules in order.
    Does NOT consider size (that requires summarization; see the handoff note in
    the module docstring). The pipeline may later override to Gemma on
    DoesNotFitInLlama, using oversize_notice().

    Args:
      body: the raw OWUI request body (for model id and background_tasks).
      current_turn_has_image: from images.current_turn_has_image(messages).

    Rule order (first match wins):
      1. Fresh image in current turn      -> Gemma (only Gemma sees it natively).
      2. Background task                  -> Gemma (protect Llama capacity).
      3. Explicit Gemma selection         -> Gemma.
      4. Explicit Llama selection         -> Llama.
      5. Auto                             -> Llama (reasoning preference).
    """
    selection = parse_selection(body.get("model"))

    # 1) Hard override: fresh image -> Gemma. Overrides explicit selection.
    if current_turn_has_image:
        # Notice only if the user did NOT already choose Gemma (they'd expect it
        # otherwise). If they were on Gemma already, the image flows natively
        # with no note (README §7).
        notice = None if selection == Selection.GEMMA else settings.NOTICE_IMAGE_DIVERT
        return RouteDecision(Target.GEMMA, Reason.FRESH_IMAGE, notice)

    # 2) Hard override: background task -> Gemma. Silent (user never sees these).
    if is_background_task(body):
        return RouteDecision(Target.GEMMA, Reason.BACKGROUND_TASK, None)

    # 3) Explicit Gemma.
    if selection == Selection.GEMMA:
        return RouteDecision(Target.GEMMA, Reason.EXPLICIT_GEMMA, None)

    # 4) Explicit Llama. (Oversize diversion handled downstream if it won't fit.)
    if selection == Selection.LLAMA:
        return RouteDecision(Target.LLAMA, Reason.EXPLICIT_LLAMA, None)

    # 5) Auto -> Llama (reasoning preference). Size diversion handled downstream.
    return RouteDecision(Target.LLAMA, Reason.AUTO_LLAMA, None)


def oversize_divert_decision() -> RouteDecision:
    """
    The decision the pipeline uses AFTER summarization raises DoesNotFitInLlama:
    reroute to Gemma with the oversize notice. Centralized here so all routing
    outcomes (and their notices) live in one module.
    """
    return RouteDecision(
        Target.GEMMA,
        Reason.OVERSIZE_DIVERT,
        settings.NOTICE_OVERSIZE_DIVERT,
    )


def llama_unavailable_divert_decision() -> RouteDecision:
    """
    The decision the pipeline uses when a Llama-side sizing call (the injected
    token counter used by summarization, or the pre-send /tokenize clamp call)
    fails outright — e.g. Llama's vLLM is unreachable. Distinct from
    oversize_divert_decision() so the participant sees an accurate notice
    (temporary unavailability, not "too large").
    """
    return RouteDecision(
        Target.GEMMA,
        Reason.LLAMA_UNAVAILABLE,
        settings.NOTICE_LLAMA_UNAVAILABLE,
    )


# ==================================================================== #
# Prompt sizing (README Step 6): fast estimate + injected exact count.
# ==================================================================== #
def estimate_tokens(messages: list[Message]) -> int:
    """
    Fast, pure token ESTIMATE via chars / 4 over all textual content.

    - Handles both string content and list (multimodal) content: for list
      content we sum the text parts and count each non-text part as a small
      fixed cost (they should already be text by the time we size a Llama
      request, but we stay robust).
    - Deliberately rough: used only to decide whether we can skip the exact
      /tokenize call. When near the ceiling we defer to the exact count.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        total_chars += len(part["text"])
                    else:
                        # Non-text part (should be rare here); nominal cost.
                        total_chars += 8
                elif isinstance(part, str):
                    total_chars += len(part)
        # role/other fields contribute chat-template overhead the exact count
        # captures; the estimate intentionally ignores them.
    return total_chars // 4
