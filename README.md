# LLM Middleware — README

> A routing, repair, and enrichment layer that sits invisibly between Open WebUI and two vLLM-served language models on a single air-gapped wargame server.

---

## 1. What this is (in one paragraph)

This is a **standalone FastAPI middleware service** deployed in a Docker container. It sits between Open WebUI (the chat frontend used by ~50 virtual desktops) and two vLLM model backends running on separate GPUs. To Open WebUI it looks like a single OpenAI-compatible endpoint (configured as an **OpenAI API Connection**). Internally, it inspects every chat request, routes it to the correct model, repairs requests that would otherwise fail (e.g. images in history sent to a text-only model, or a reply that runs out of room), keeps long conversations inside the smaller model's context window through incremental summarization, and enriches requests with wargame-specific terminology. All of this is invisible to participants — they simply experience a stable chat that does not crash or surprise them.

---

## 2. The environment

| Component | Detail |
|---|---|
| **Server** | Single air-gapped machine, two NVIDIA L40S GPUs (no NVLink) |
| **Model A — Llama 3.3 70B** | GPU 0. Text only. Strong reasoning. vLLM service name **`llama`**, port **8002**. |
| **Model B — Gemma 4 26B** | GPU 1. **Text + images.** Effectively unlimited context for our purposes. vLLM service name **`gemma`**, port **8001**. |
| **Web interface** | Open WebUI on port **3000**; its OpenAI API base URL points at this middleware. |
| **This middleware** | FastAPI + uvicorn in a Docker container, exposed on **8080**. |
| **Other services on the box** | Embedding proxy on **9000**; two light FastAPI services on **3000/service1** and **3000/service2**. |
| **Orchestration** | All services (middleware, both vLLMs, OWUI, embedding proxy, the two FastAPI services) run in **one docker-compose project assembled via `include`**. |

**Key points:**
- The two models are on separate GPUs and never compete for resources. Llama is the default reasoning workhorse; Gemma is the specialist for images and very long conversations.
- Because everything runs in **one compose project**, containers reach each other by **docker-compose service name** (e.g. `http://llama:8002`), **not** `localhost`. Inside a container, `localhost` is the container itself.

---

## 3. The problems this middleware solves

1. **Llama's limited context causes long chats to fail.** Long strategic discussions exceed Llama's usable window and fail or degrade. Gemma has no practical limit.
2. **Images break Llama, and switching models mid-conversation crashes.** OWUI always resends the *entire* history. A participant starts on Gemma with an image, switches to Llama for deeper analysis, and the chat crashes because Llama chokes on the image in history.
3. **A reply can run out of room mid-answer.** Even when the prompt fits, Llama may hit its generation limit and stop before finishing (`finish_reason: "length"`).
4. **The models don't know our wargame terminology.** They lack the specialized acronyms used in the wargame, producing answers that miss key concepts.

---

## 4. Design philosophy

- **Prefer Llama for reasoning.** Keep participants in Llama's stronger reasoning mode as long as possible.
- **Summarize old turns, keep recent turns raw.** Preserves the current thread while shrinking history to fit Llama.
- **Describe images once, proactively, and share the result.** The moment an image first appears, describe it in the background and cache it (keyed by image-byte hash, shared across all participants) so later Llama turns have zero lag.
- **Route to Gemma only when forced:** explicit Gemma selection, a fresh image in the current turn, an OWUI background task, or when summarization cannot fit the request into Llama.
- **Repair, don't fail.** Clamp generation limits to avoid vLLM 400s; on a length cutoff, hand off to Gemma to finish; on oversize, divert to Gemma — always with an honest, plain-language note to the participant when behavior changes.
- **Honor explicit model choices unless a hard constraint (image, background task, or size) overrides them.**
- **Never throttle or queue ourselves** — vLLM already manages concurrency safely.
- **Persist image descriptions across restarts** — never describe the same image twice.
- **Summarization is incremental and lazy** — only summarize more when the window is exceeded; reuse prior summaries; never summarize the protected recent tail.
- **Fail soft on auxiliary work.** Terminology, image warm-up, and cache I/O degrade gracefully rather than crashing a conversation.

---

## 5. Architecture decision: true middleware via OpenAI API Connection (Path A)

This is deliberately a **standalone OpenAI-compatible proxy** that OWUI calls *out* to (configured as an OpenAI API Connection), **not** an OWUI plugin/filter and **not** the OWUI internal chat-orchestration API (which drives OWUI's own chat tree via `chat_id`/`history`/`childrenIds`). Path A keeps the middleware a clean, version-resilient proxy.

It must be a true middleware because it:
- Intercepts and transforms raw HTTP requests before they reach vLLM.
- Maintains stateful, per-conversation logic (summarization state in memory; image cache on disk).
- Dynamically routes to **two** separate vLLM endpoints based on request content.
- Persists a shared, on-disk image-description cache that survives restarts.
- Monitors client disconnects and aborts upstream inference to free the GPU.
- Inspects the response stream to detect and repair length cutoffs.
- Remains fault-isolated from OWUI (a middleware crash must not take down the frontend).

**Informational note:** the OWUI internal orchestration API (a separate, community-documented pattern) is *not* used here; it was reviewed only to confirm the shape of OWUI's outbound completion request and how background tasks are signalled.

---

## 6. The context-budget model (important)

vLLM's `max_model_len` is the **total** budget for **prompt tokens + generated tokens combined**. Therefore we separate three concepts in `config.py`:

- **`LLAMA_MAX_MODEL_LEN`** — the real value Llama's vLLM was launched with (source of truth).
- **`LLAMA_RESERVED_GENERATION`** — tokens reserved for the answer.
- **`LLAMA_CEILING`** (derived) — the usable **prompt budget** = `max_model_len − reserved_generation`. This is the number all sizing/routing logic compares prompts against.

$$ \text{LLAMA\_CEILING} = \text{LLAMA\_MAX\_MODEL\_LEN} - \text{LLAMA\_RESERVED\_GENERATION} $$

Two defenses follow from this:

- **Proactive (before sending):** size the prompt against `LLAMA_CEILING`. If it won't fit, summarize; if it still won't fit, divert to Gemma. Also **clamp outgoing `max_tokens`** so `prompt + max_tokens ≤ max_model_len` (avoiding vLLM 400s).
- **Reactive (during streaming):** even when the prompt fits, the *answer* may exceed the reserved room. We watch the stream; a `finish_reason: "length"` from Llama triggers a plain-language notice and a **handoff to Gemma to complete the response**.

**vLLM behavior confirmed:** a prompt+generation overflow is rejected up-front with a `400`; hitting the generation cap mid-answer yields `finish_reason: "length"` (a clean stop, not a crash).

---

## 7. What the middleware does — the processing pipeline

Every request passes through these stages before reaching a model. Invisible to the participant.

### Step 1 — Receive and understand the request
- Capture the participant's message and the full conversation history from OWUI.
- Note the model selection: explicit **Llama**, explicit **Gemma**, or **Auto** (system decides), parsed from the OWUI `model` field via the config contract.
- Detect **OWUI background tasks** (title/tag/follow-up generation) via the `background_tasks` flags.

### Step 2 — Disconnect watch
- If the participant clicks "Stop" or closes the tab at any point, immediately close the upstream connection, freeing the GPU rather than finishing an answer nobody reads.

### Step 3 — Handle images (proactive warm-up + shared persistent cache)
Ensures Llama never receives raw image data, describes each unique image at most once system-wide, and eliminates second-turn lag.

- **If the current user turn contains a fresh image:**
  - Route the turn to **Gemma**, which receives the **native** image to answer.
  - **Fire a background, fire-and-forget warm-up** (`asyncio.create_task`) that, for each image, **checks the shared cache first** (images are shared across participants, so it may already be described), and on a miss calls Gemma to describe it and stores it. This warm-up **never alters the payload used to answer this turn** and **never blocks** it. Result: on the next turn (if the chat moves to Llama) the description is already cached — no lag.
- **Else (current turn has no fresh image) and history contains images:**
  - For each image: SHA-256-hash its **bytes** (stable across OWUI resends and identical across participants sharing the image), look it up in the **shared SQLite cache**, describe on a miss (almost always a hit thanks to the warm-up), and **replace the image part** in history with `[Image description: <text>]`.
  - After this step, history is **100% text-only** and safe for Llama.

**Cache & concurrency properties:**
- The cache is **shared** (keyed by image-byte hash) and **persistent** (SQLite, WAL mode), surviving restarts (Caveat #3).
- The async warm-up **offloads all SQLite to a thread** (`asyncio.to_thread`) so it never blocks the event loop, and uses a **per-image-hash `asyncio.Lock` (double-checked)** so simultaneous submissions of the same new image trigger **only one** Gemma describe.
- The synchronous history-rewrite path is **blocking by contract** and must be invoked by the pipeline via `await asyncio.to_thread(...)`.
- Failed describes are **not cached**, so a later healthy attempt can succeed. Undecodable images degrade to a clear placeholder.

### Step 4 — Incremental, lazy summarization (to keep the chat in Llama)
Maintain **per-conversation state** (in memory, ephemeral):
- `history`: full list of turns (images already replaced by descriptions).
- `summarized_up_to`: index — all turns **before** it are folded into `summary_text`.
- `summary_text`: condensed summary of turns `[0, summarized_up_to)`.

**Algorithm (per turn, after image handling):**
1. The pipeline appends the new user turn to `history` **before** summarization runs.
2. **Gemma-bound requests are never summarized** — Gemma has headroom; send `[summary?] + history[summarized_up_to:]`.
3. **Llama-bound requests:**
   - **Fast path:** if the current payload already fits under `LLAMA_CEILING`, send it unchanged (lazy).
   - **Early divert pre-check (E1):** if the smallest achievable prompt — the **exact protected raw tail** plus a **compacted summary floor** (`min(current summary tokens, SUMMARY_MAX_TOKENS)`) — already exceeds `LLAMA_CEILING`, diverting is inevitable, so raise `DoesNotFitInLlama` **immediately** with **zero** Gemma calls and **zero** state mutation.
   - **Otherwise, summarize incrementally:** fold the earliest portion (default: half) of the summarizable raw range into `summary_text` via Gemma, advance `summarized_up_to`, and repeat until it fits. If the summary itself grows past `SUMMARY_MAX_TOKENS`, do a rare full re-summarization (of the original turns, to avoid summary-of-summary drift) and reset.

**Invariants (verifiable by reading):**
- `summarized_up_to` only moves **forward**.
- The last `RAW_TAIL_TURNS` messages are **never** summarized.
- Summarization is **lazy** (only when over budget).
- The loop **always terminates** (each pass advances the cutoff, or diverts).
- A "turn" means **one message** (not a user+assistant pair); `RAW_TAIL_TURNS` counts messages.

### Step 5 — Add wargame terminology
- Scan the **latest user message** (and any freshly generated image description) for known acronyms from the **3,000–4,000-entry CSV** (separate row per meaning).
- Use **exact-case, whole-word matching** (`MoE` matches precisely; not `moe`/`MOE`).
- **Multiple definitions per acronym are supported.** When an acronym has several meanings, **all** are presented with an explicit instruction for the model to choose by context.
- Matches are appended as a dedicated **system block**. Terminology degrades gracefully if the CSV is missing.

### Step 6 — Routing decision and sizing
Routing owns the **rule tree** and the **sizing primitives**; it does **not** summarize or forward. The oversize→Gemma diversion is driven by `summarization` raising `DoesNotFitInLlama`, which the pipeline catches.

- **Rule order (first match wins), producing a typed `RouteDecision` (target + reason + optional user notice):**
  1. **Fresh image** in the current turn → **Gemma** (overrides explicit selection; notice shown unless the user was already on Gemma).
  2. **Background task** → **Gemma** (silent; overrides explicit selection).
  3. **Explicit Gemma** → Gemma.
  4. **Explicit Llama** → Llama (subject to the oversize check downstream).
  5. **Auto** → **Llama** (reasoning preference).
- **Sizing:** fast `chars/4` estimate; only near the ceiling call the target's `/tokenize` for the exact count. Exact-count failure **fails safe toward Gemma**.
- **Oversize diversion:** if a Llama-bound request raises `DoesNotFitInLlama`, the pipeline re-routes to Gemma and attaches the oversize notice.

### Step 7 — Forward, stream, and repair
- Forward the final request to the correct backend (`http://llama:8002` or `http://gemma:8001`), with `max_tokens` clamped for Llama so `prompt + max_tokens ≤ max_model_len`.
- **Stream SSE back to OWUI**, **parsing each chunk** to (a) forward it to the participant and (b) detect `finish_reason`.
- **Length handoff:** if a Llama response ends with `finish_reason: "length"`, stream the participant a plain-language notice, then send a **continuation request to Gemma** (original prompt + Llama's partial reply as context, instructed to continue seamlessly) and stream Gemma's completion — all within the **same** assistant response. Sequence the participant sees: *Llama partial → notice → Gemma continuation.*
- If the disconnect watch triggers during streaming, immediately close the upstream connection to release the GPU.

---

## 8. What participants experience

- **Normal text chat:** Defaults to Llama. Long conversations are silently kept in-window by summarizing old turns while keeping the last `RAW_TAIL_TURNS` messages raw.
- **Uploading an image:**
  - On Auto/Llama → a brief note *"Message routed to the image-capable model for image analysis"*, then the answer. A background warm-up caches the description silently for later.
  - Already on Gemma → no note; the image flows natively (warm-up still runs silently).
- **Switching from an image chat to Llama:** prior images are already cached as descriptions (usually warmed on first sight), so Llama receives clean text-only history and continues without crashing or lag.
- **A reply that runs out of room:** the participant sees Llama's partial answer, then a plain note that the reasoning model reached its length limit and is handing off, then Gemma completes the answer.
- **A request too large for Llama:** a brief note that it was routed to the other model, then the answer from Gemma.
- **Terminology:** acronyms are automatically expanded; ambiguous acronyms show all meanings for the model to disambiguate.

---

## 9. Project structure

```
llm-middleware/
├── app/
│   ├── __init__.py              # Marks 'app' as a Python package
│   ├── main.py                  # FastAPI app + /v1/chat/completions route + lifespan hooks
│   ├── config.py                # Settings (ports, budgets, paths, notices) via env vars
│   ├── pipeline.py              # Orchestrates the pipeline; owns append/lock/divert wiring
│   ├── images.py                # Step 3: hashing, shared SQLite cache, warm-up, history rewrite
│   ├── summarization.py         # Step 4: incremental/lazy summarization + per-convo state math
│   ├── terminology.py           # Step 5: acronym CSV loading + multi-definition matching
│   ├── routing.py               # Step 6: rule tree + sizing primitives (RouteDecision)
│   ├── backends.py              # Async httpx clients: chat, /tokenize, describe, summarize
│   ├── streaming.py             # Step 7: SSE parse/forward + disconnect + length handoff
│   └── state.py                 # Per-conversation in-memory state store (ephemeral)
├── data/
│   ├── acronyms.csv             # Acronym definitions (one row per meaning)
│   └── image_cache.db           # SQLite cache (created at runtime; mount as volume)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml           # (or an `include` fragment in the top-level compose)
└── .env                         # Environment overrides (not committed)
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, OpenAI-compatible endpoint, startup/shutdown hooks (load CSV, init cache), `/health`. |
| `config.py` | Centralizes tunables: budgets (`max_model_len`/reserved/derived ceiling), URLs (service names), model-id contract, notices; validates config at startup. |
| `pipeline.py` | Runs the pipeline; appends turns, holds per-conversation lock, fires the image warm-up, offloads blocking work, catches `DoesNotFitInLlama`, drives the length handoff. |
| `images.py` | Byte-hashing, shared persistent SQLite cache (WAL), async fire-and-forget warm-up (thread-offloaded, per-hash locked), sync history rewrite. |
| `summarization.py` | Pure incremental-summarization algorithm (token counter and Gemma summarizer **injected**); E1 pre-check; `DoesNotFitInLlama`. |
| `terminology.py` | Loads acronym CSV once; exact-case whole-word matching; multi-definition output block. |
| `routing.py` | `RouteDecision` rule tree; `estimate_tokens`; injected exact `/tokenize` sizing; oversize-divert decision. |
| `backends.py` | Async `httpx` clients for both vLLMs; implements the injected describe/summarize/tokenize/chat calls; clamps Llama `max_tokens`. |
| `streaming.py` | Parses and forwards SSE; detects `finish_reason: "length"`; performs the Gemma continuation; handles disconnect. |
| `state.py` | Per-conversation state (`history`, `summarized_up_to`, `summary_text`) in memory; ephemeral by design; thread-safe store. |

### Architecture pattern
- **Dependency injection at every I/O boundary.** Pure-logic modules (`summarization`, `routing`, most of `images`, `terminology`) take network capabilities (tokenize, describe, summarize) as **injected callables/protocols**, so ~80% of the code is verifiable by reading. All real network I/O is confined to `backends.py` and `streaming.py`.

---

## 10. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MIDDLEWARE_HOST` / `MIDDLEWARE_PORT` | `0.0.0.0` / `8080` | Bind address/port OWUI connects to (`/v1`). |
| `LLAMA_HOST` / `LLAMA_PORT` | `llama` / `8002` | Llama vLLM (compose service name). |
| `GEMMA_HOST` / `GEMMA_PORT` | `gemma` / `8001` | Gemma vLLM (compose service name). |
| `MODEL_ID_LLAMA` / `MODEL_ID_GEMMA` / `MODEL_ID_AUTO` | `llama` / `gemma` / `auto` | Model-id contract OWUI sends. |
| `LLAMA_SERVED_MODEL_NAME` / `GEMMA_SERVED_MODEL_NAME` | `llama` / `gemma` | `--served-model-name` each vLLM expects in the forwarded body. |
| `LLAMA_MAX_MODEL_LEN` | `9000` | **Set to Llama's real `--max-model-len`.** Source of truth. |
| `LLAMA_RESERVED_GENERATION` | `1500` | Tokens reserved for the answer. |
| `LLAMA_CEILING` (derived) | — | Prompt budget = `max_model_len − reserved`. |
| `LLAMA_FAST_PATH_MARGIN` | `500` | Margin below the ceiling under which the exact `/tokenize` is skipped. |
| `RAW_TAIL_TURNS` | `3` | Most-recent **messages** always kept raw (set `6` for "3 exchanges"). |
| `SUMMARY_CHUNK_FRACTION` | `0.5` | Fraction of the raw tail summarized per pass (validated `0 < f ≤ 1`). |
| `SUMMARY_MAX_TOKENS` | `2000` | Summary size that triggers full re-summarization. |
| `IMAGE_CACHE_PATH` | `/data/image_cache.db` | Shared, persistent SQLite image cache. |
| `ACRONYMS_CSV_PATH` | `/data/acronyms.csv` | Acronym CSV (one row per meaning). |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `120` | Connect / non-streaming read timeouts (streaming relies on the disconnect watch). |
| `BACKGROUND_TASK_FLAGS` | title/tags/follow-up | OWUI `background_tasks` flags that route to Gemma. |
| `NOTICE_LENGTH_HANDOFF` / `NOTICE_OVERSIZE_DIVERT` / `NOTICE_IMAGE_DIVERT` | (see config) | Plain-language user notices. |
| `LOG_LEVEL` | `INFO` | Logging level. |

> **Docker networking:** in one compose project via `include`, use **service names** (`llama`, `gemma`), never `localhost`.

---

## 11. Build and run (air-gapped)

Because the server is offline and iterative transfer is costly, build for **first-transfer correctness**:

- **Vendor all wheels** on a connected "download" network:
  ```bash
  pip download -r requirements.txt -d ./wheels \
    --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:
  ```
- **Save the base image** as a tarball: `docker save python:3.12-slim -o python-3.12-slim.tar`.
- **Dockerfile installs offline:** `pip install --no-index --find-links=/wheels -r requirements.txt`.
- **Pin every dependency version exactly.**
- On the server: `docker load` the base image, then build within the single compose project.
- **Persist `/data` as a volume** so the shared image cache survives restarts.
- **No GPU is required in this container** — the middleware only orchestrates.
- Point OWUI's OpenAI API base URL at `http://<middleware-host>:8080/v1`.

---

## 12. Honest caveats for stakeholders

1. **Summarization loses fine detail.** Turns older than the last `RAW_TAIL_TURNS` are compressed into a strategic-intent summary. Exact prior statements may be gone. Acceptable for COA reasoning.
2. **Image description is a translation.** A text description is not a perfect visual substitute; participants needing repeated visual reasoning should stay on Gemma.
3. **The image cache is not backed up.** It survives restarts and is shared across participants, but loss means images are described again — harmless.
4. **Summarization state is in memory only.** A restart resets it; worst case is one extra summarization pass.
5. **Length handoff produces a two-model answer.** Llama's start and Gemma's finish are written by different models and may have a slight seam in tone; the notice makes this transparent.
6. **Real concurrency with participants has not been observed.** vLLM has been stress-tested with no OOM (only context limits). We rely on vLLM's admission control, not our own throttling.
7. **A conversation that starts and stays on Gemma may pay for one unused warm-up describe per unique image.** Accepted to eliminate second-turn lag in the common case; bounded to one describe per unique image system-wide.

---

## 13. Guidance for AI tools contributing code to this repo

- **Stack:** Python 3.12, FastAPI, uvicorn, `httpx` (async), `pydantic-settings`, SQLite (stdlib). Keep dependencies minimal and version-pinned.
- **Path A only:** the public endpoint is `POST /v1/chat/completions`, OpenAI-compatible in and out, including streaming SSE. Do **not** use the OWUI internal orchestration API.
- **Streaming is parse-and-forward**, not blind byte copy: forward chunks unchanged to the user **and** inspect `finish_reason` to drive the length handoff.
- **Async throughout;** never block the event loop. Offload blocking work (SQLite, sync describers) via `asyncio.to_thread`.
- **Never throttle or queue** inference. Rely on vLLM's concurrency control.
- **Dependency injection at I/O boundaries.** Pure-logic modules take injected capabilities; real network I/O lives only in `backends.py`/`streaming.py`.
- **Respect the context-budget model:** compare prompts against the **derived `LLAMA_CEILING`**; clamp Llama `max_tokens` so `prompt + max_tokens ≤ max_model_len`.
- **Respect the routing rules and their order exactly** (Step 6); oversize diversion is signalled by `DoesNotFitInLlama`, not decided in the rule tree.
- **Image handling:** DB-first (shared cache), byte-hash keys, fire-and-forget warm-up that never alters the answering payload, per-hash lock to prevent redundant describes, thread-offloaded SQLite.
- **Summarization invariants are non-negotiable** (forward-only cutoff, protected tail, laziness, termination, E1 pre-check with compacted summary floor).
- **Terminology:** exact-case whole-word matching; **all** meanings shown for ambiguous acronyms.
- **Config-driven:** ports, budgets, model ids, notices, and paths come from `config.py` — no hardcoded values in logic modules; config is validated at startup.
- **Fail soft** on auxiliary work; **fail safe toward Gemma** on sizing uncertainty.
- **Do not invent behavior** not described here. If a requirement is ambiguous, prefer the simplest option consistent with the design philosophy and flag the assumption in a code comment.
- **Fixtures to capture when backend access is available:** a real OWUI completion request (plain, image-bearing, and title-generation), and a real vLLM SSE stream (including a `finish_reason: "length"` case). Confirm the multimodal image shape, the `background_tasks` shape, the `MODEL_ID_*` strings, and each `--served-model-name`.

---

*This README is the authoritative description of the middleware's purpose and behavior. Keep it updated as the design evolves, and use it to begin AI-assisted coding sessions so contributors share the same context.*