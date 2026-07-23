"""
config.py — Centralized configuration for the LLM Wargame Middleware.

Design notes:
- PURE DECLARATION. No logic that requires running to verify.
- All tunables are environment-overridable via pydantic-settings.
- Containers reach each other by docker-compose SERVICE NAME (not localhost),
  because middleware, both vLLMs, OWUI, the embedding proxy, and the two light
  FastAPI services all run in ONE compose project assembled via `include`.

Context-budget model (IMPORTANT — read this):
- vLLM's `max_model_len` is the TOTAL budget for PROMPT + GENERATED tokens.
- Therefore the usable PROMPT budget is:  max_model_len - reserved_generation.
- We reserve a block of tokens for the answer so a prompt that "fits" still
  leaves room to actually generate a useful reply.
- Proactive defense: size the prompt against LLAMA_CEILING (the derived prompt
  budget); summarize/divert if it won't fit (routing.py / summarization.py).
- Reactive defense: even when the prompt fits, the ANSWER may exceed the
  reserved room. We watch the stream; a finish_reason == "length" from Llama
  triggers a notice + handoff to Gemma (streaming.py / pipeline.py).
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # This service (the middleware)
    # ------------------------------------------------------------------ #
    MIDDLEWARE_HOST: str = "0.0.0.0"
    MIDDLEWARE_PORT: int = 8080

    # ------------------------------------------------------------------ #
    # Backend vLLM services (reached by docker-compose SERVICE NAME)
    # ------------------------------------------------------------------ #
    LLAMA_HOST: str = "llama"
    LLAMA_PORT: int = 8002

    GEMMA_HOST: str = "gemma"
    GEMMA_PORT: int = 8001

    # ------------------------------------------------------------------ #
    # Model-name mapping (contract between OWUI and the middleware)
    # ------------------------------------------------------------------ #
    MODEL_ID_LLAMA: str = "llama"   # explicit Llama selection from OWUI
    MODEL_ID_GEMMA: str = "gemma"   # explicit Gemma selection from OWUI
    MODEL_ID_AUTO: str = "auto"     # "Auto": middleware decides

    # Dedicated OWUI "Task Model" id. Pointing OWUI's Task Model setting at this
    # id is the PRIMARY background-task signal: it is config-driven, survives
    # filter failures, and fails safe (a misconfiguration just routes tasks by
    # model id like any other request).
    MODEL_ID_TASKS: str = "tasks"

    # SECONDARY background-task signal: treat a request with no chat_id as a
    # task call. Real participant turns carry chat_id via the companion OWUI
    # Filter (inlet, global scope) that injects it from __metadata__; OWUI's
    # internal task calls never carry it (confirmed against OWUI 0.8.5
    # fixtures). GATED because its failure mode is dangerous: if the Filter is
    # disabled or broken, every real turn loses chat_id and would be silently
    # misrouted to Gemma as a "task". Set to false to disable this signal if
    # the Filter cannot be guaranteed.
    TREAT_MISSING_CHAT_ID_AS_TASK: bool = True

    # The served model name each vLLM expects in its OWN request body
    # (must match each instance's --served-model-name).
    LLAMA_SERVED_MODEL_NAME: str = "llama"
    GEMMA_SERVED_MODEL_NAME: str = "gemma"

    # ================================================================== #
    # Llama context budget (see "Context-budget model" above)
    # ================================================================== #
    # The REAL value Llama's vLLM was launched with (--max-model-len).
    # SOURCE OF TRUTH. Set this to match your actual deployment.
    LLAMA_MAX_MODEL_LEN: int = 9000

    # Tokens reserved for GENERATION (the answer). The prompt budget is
    # max_model_len - this value. Choose enough headroom that a typical answer
    # can complete; if answers frequently hit the cap, raise this (which lowers
    # the prompt budget) or rely on the reactive Gemma handoff.
    LLAMA_RESERVED_GENERATION: int = 1500

    # Fast-path safety margin (tokens) below the prompt ceiling. If the chars/4
    # estimate is comfortably under (ceiling - this margin), skip the exact
    # /tokenize call. Keeps estimation error from silently pushing us over.
    LLAMA_FAST_PATH_MARGIN: int = 500

    # Most-recent MESSAGES to ALWAYS keep raw (never summarized).
    # NOTE: a "turn" == one message here, not a user+assistant pair. If you want
    # "last 3 exchanges", set this to 6.
    RAW_TAIL_TURNS: int = 3

    # Fraction of the current raw tail to summarize per pass when over budget.
    SUMMARY_CHUNK_FRACTION: float = 0.5

    # If summary_text itself exceeds this many tokens (rare), trigger a full
    # re-summarization and reset (README Step 4).
    SUMMARY_MAX_TOKENS: int = 2000

    # Max decoded image size (bytes). base64 inflates ~33%; keep headroom for IL5.
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MiB, tune to deployment

    # ------------------------------------------------------------------ #
    # Persistence and data paths (mounted volume; survive restarts)
    # ------------------------------------------------------------------ #
    IMAGE_CACHE_PATH: str = "/data/image_cache.db"
    ACRONYMS_CSV_PATH: str = "/data/acronyms.csv"

    # ------------------------------------------------------------------ #
    # HTTP client behavior toward the vLLM backends
    # ------------------------------------------------------------------ #
    HTTP_CONNECT_TIMEOUT: float = 10.0
    HTTP_READ_TIMEOUT: float = 120.0  # non-streaming calls only

    # ------------------------------------------------------------------ #
    # OWUI background-task detection (README Step 6)
    # ------------------------------------------------------------------ #
    BACKGROUND_TASK_FLAGS: tuple[str, ...] = (
        "title_generation",
        "tags_generation",
        "follow_up_generation",
    )

    # ================================================================== #
    # User-facing notice strings (centralized so wording is easy to tune
    # without touching logic). These are streamed to the participant as
    # ordinary assistant text between model segments.
    # ================================================================== #
    # Shown when Llama's response is cut off for length and we hand off to Gemma
    # to complete it. Written for an average user (no jargon like "tokens").
    NOTICE_LENGTH_HANDOFF: str = (
        "\n\n---\n"
        "*The reasoning model reached the maximum length for a single reply and "
        "could not finish. I'm handing this off to the other model to complete "
        "the response below.*\n\n"
    )

    # Shown when an Auto/Llama request is routed to Gemma because the request is
    # too large for the reasoning model to handle at all (proactive divert).
    NOTICE_OVERSIZE_DIVERT: str = (
        "*This request was too large for the reasoning model, so it was routed "
        "to the other model.*\n\n"
    )

    # Shown when a request is routed to Gemma for image analysis (README §7).
    NOTICE_IMAGE_DIVERT: str = (
        "*Message routed to the image-capable model for image analysis.*\n\n"
    )

    # Shown when the reasoning model's sizing check (the exact /tokenize call)
    # fails before generation starts — e.g. Llama's vLLM is temporarily
    # unreachable while Gemma is healthy. We fail safe toward Gemma rather than
    # surfacing an opaque 500 for what would otherwise be a normal request.
    NOTICE_LLAMA_UNAVAILABLE: str = (
        "*The reasoning model is temporarily unavailable, so this was routed "
        "to the other model.*\n\n"
    )

    # Shown when a model backend becomes unreachable (or errors) AFTER we have
    # already started streaming a response, so the participant sees why the
    # reply stopped instead of a silently truncated answer.
    NOTICE_BACKEND_UNAVAILABLE: str = (
        "\n\n---\n"
        "*The model backend became unavailable and the response could not be "
        "completed. Please try again in a moment.*\n\n"
    )


    @field_validator("SUMMARY_CHUNK_FRACTION")
    @classmethod
    def _validate_chunk_fraction(cls, v: float) -> float:
        """
        Enforce 0.0 < SUMMARY_CHUNK_FRACTION <= 1.0. A value outside this range
        indicates configuration drift: 0 or negative would summarize one message
        at a time (many wasteful Gemma calls), and >1 would over-summarize. On an
        air-gapped deployment we prefer a clear boot-time failure over subtle
        runtime misbehavior.
        """
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"SUMMARY_CHUNK_FRACTION must be in (0.0, 1.0], got {v}."
            )
        return v

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ================================================================== #
    # Derived values (single source of truth; computed, never stored)
    # ================================================================== #
    @property
    def LLAMA_CEILING(self) -> int:
        """
        The usable PROMPT budget for Llama = max_model_len - reserved generation.
        This is the number the sizing/routing logic compares prompts against.
        A prompt at or below this leaves at least LLAMA_RESERVED_GENERATION
        tokens for the answer.
        """
        return self.LLAMA_MAX_MODEL_LEN - self.LLAMA_RESERVED_GENERATION

    @property
    def LLAMA_FAST_PATH_TOKENS(self) -> int:
        """
        Below this estimated prompt size we skip the exact /tokenize call.
        Kept a safety margin under the prompt ceiling.
        """
        return self.LLAMA_CEILING - self.LLAMA_FAST_PATH_MARGIN

    @property
    def LLAMA_URL(self) -> str:
        return f"http://{self.LLAMA_HOST}:{self.LLAMA_PORT}"

    @property
    def GEMMA_URL(self) -> str:
        return f"http://{self.GEMMA_HOST}:{self.GEMMA_PORT}"

    @property
    def LLAMA_CHAT_URL(self) -> str:
        return f"{self.LLAMA_URL}/v1/chat/completions"

    @property
    def GEMMA_CHAT_URL(self) -> str:
        return f"{self.GEMMA_URL}/v1/chat/completions"

    @property
    def LLAMA_TOKENIZE_URL(self) -> str:
        return f"{self.LLAMA_URL}/tokenize"

    @property
    def GEMMA_TOKENIZE_URL(self) -> str:
        return f"{self.GEMMA_URL}/tokenize"

    def llama_max_gen_tokens(self, prompt_tokens: int) -> int:
        """
        Compute a safe outgoing max_tokens for a Llama request given the actual
        prompt size, so prompt + max_tokens never exceeds max_model_len (which
        would cause vLLM to reject the request with a 400).

        Returns the room remaining for generation. Callers should forward
        min(requested_max_tokens, this) — or just this if the request omits
        max_tokens. Never negative; 0 means "no room" (should have diverted).
        """
        return max(0, self.LLAMA_MAX_MODEL_LEN - prompt_tokens)


settings = Settings()