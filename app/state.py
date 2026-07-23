"""
state.py — Per-conversation summarization state (README Step 4).

What this module owns:
- A single in-memory store mapping a conversation key -> ConversationState.
- The three fields the summarization algorithm needs, per the README:
    * history           : full list of turns (images already replaced by text)
    * summarized_up_to  : index; all turns BEFORE it are folded into summary_text
    * summary_text      : condensed summary of history[0 : summarized_up_to]

What this module deliberately does NOT own:
- Any summarization LOGIC (fit checks, chunk sizing, calling Gemma). That lives
  in summarization.py. Keeping this module logic-free makes it trivially
  verifiable by reading, which matters for a blind (no-test) build.

Lifetime / persistence:
- EPHEMERAL BY DESIGN. State is in memory only and is LOST on restart
  (README Caveat #4). Worst case after a restart is one extra summarization
  pass, which is acceptable for a wargame session. Do NOT add persistence here
  unless the design changes; the persistent artifact in this system is the
  IMAGE cache (images.py), not summarization state.

Concurrency model:
- The store's dict mutations (get-or-create, delete, list) are guarded by a
  lock so concurrent async requests cannot corrupt the mapping itself.
- IMPORTANT: this module does NOT serialize access to an individual
  ConversationState's fields. Two concurrent requests for the SAME conversation
  must not run the summarization mutation at the same time. In OWUI's normal
  usage a single conversation is driven by one participant sending one turn at a
  time, so same-conversation concurrency is not expected. The pipeline should
  nonetheless treat mutation of a single ConversationState as a critical section
  (see per-conversation lock helper below) if it ever needs to.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional


# A single "turn" is an OpenAI-style message dict, e.g.
#   {"role": "user", "content": "..."} or
#   {"role": "assistant", "content": "..."}.
# Images in `content` have already been replaced with text descriptions by the
# time turns are stored here (see README Step 3, images.py), so history is
# guaranteed text-only. We type it as a plain dict to avoid coupling to any
# specific schema class.
Turn = dict[str, Any]


@dataclass
class ConversationState:
    """
    Mutable state for ONE conversation.

    Fields mirror README Step 4 exactly. All mutation of these fields is the
    responsibility of summarization.py; this dataclass is a passive container.
    """

    # The full, ordered list of turns for this conversation. Turns before
    # `summarized_up_to` have been condensed into `summary_text`, but we keep
    # the raw turns here too (they are simply no longer sent to Llama once
    # summarized). Images are already replaced by descriptions before storage.
    history: list[Turn] = field(default_factory=list)

    # Index into `history`. All turns in history[0:summarized_up_to] are
    # represented by `summary_text`. This index only ever MOVES FORWARD
    # (never un-summarize), per README Step 4.
    summarized_up_to: int = 0

    # The condensed summary of history[0:summarized_up_to]. Empty string until
    # the first summarization is triggered.
    summary_text: str = ""

    # Per-conversation lock. summarization.py / pipeline.py may acquire this to
    # make the "read fields -> maybe summarize -> update fields" sequence atomic
    # for a single conversation. Not used by this module's own methods.
    # (dataclass field=exclude from repr to avoid noisy logs.)
    lock: "asyncio.Lock" = field(default_factory=lambda: asyncio.Lock(), repr=False)


class ConversationStore:
    """
    Thread/async-safe registry of ConversationState objects, keyed by
    conversation. The store lock protects ONLY the mapping (get-or-create /
    delete / list). It is an asyncio.Lock for a uniform async locking story,
    though its critical sections do not await.
    """

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, conversation_key: str) -> ConversationState:
        """Return the state for a key, creating one atomically on first use."""
        async with self._lock:
            state = self._states.get(conversation_key)
            if state is None:
                state = ConversationState()
                self._states[conversation_key] = state
            return state

    async def get(self, conversation_key: str) -> Optional["ConversationState"]:
        async with self._lock:
            return self._states.get(conversation_key)

    async def delete(self, conversation_key: str) -> None:
        async with self._lock:
            self._states.pop(conversation_key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._states.clear()

    async def count(self) -> int:
        async with self._lock:
            return len(self._states)


# ---------------------------------------------------------------------- #
# Module-level singleton store.
# ---------------------------------------------------------------------- #
# summarization.py and pipeline.py import this single instance so all requests
# share one in-memory registry. There is no startup initialization required;
# the store is empty until conversations arrive.
store = ConversationStore()