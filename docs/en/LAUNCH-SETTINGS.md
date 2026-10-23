# Launch settings

Every field on the launch form, what it maps to, and the behavior notes. For the four-step "how to launch" walkthrough, see the [README](../../README.md#your-first-model).

## Contents

- [Launch form tools](#launch-form-tools) · [Launch settings reference](#launch-settings-reference) · [Live preset updates](#live-preset-updates) · [Concurrency & queueing](#concurrency--queueing) · [Image & PDF Input](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Per-Instance Proxy](#per-instance-proxy) · [Idle Timeout](#idle-timeout)

Behavior notes below are terse; hover the field's info-tip in the UI for the full explanation. All flags map to llama-server unless stated - see the [llama-server CLI reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for exact semantics and current defaults.

## Launch form tools

The launch form's action row carries three tool buttons on the right:

- **Export preset** - download the current form values as a JSON file (portable across nodes; does not save to storage).
- **Import preset** - load a preset JSON into the form. Populates fields only; you still hit Save Preset to persist.
- **Copy llama.cpp command** - copies the equivalent `llama-server ...` invocation for the current form values to your clipboard. Rendered server-side through the same flag-emission code the real launch uses, so it stays in sync as flags evolve.

With settings collapsed, a **Quick Launch** button starts the selected model straight from its preset.

Each instance exposes an OpenAI-compatible API on its assigned port. When a GGUF is selected, llamaMan reads its metadata to detect layer count and shows it next to **GPU Layers** (e.g. `/ 32`).

## Launch settings reference

**Model Settings**

| Setting | Default | Flag | Notes |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Prompt + response token cap per request |
| **Parallel Slots** | `1` | `--parallel` | Concurrent decoding slots inside llama-server (KV slots) |
| **Batch Size** | `2048` | `--batch-size` | Most prompt tokens processed in one step. Blank omits the flag |
| **Micro-batch Size** | `512` | `--ubatch-size` | Physical batch sent to the GPU; bigger = faster prefill, more VRAM. Embedding and encoder (BERT-style) decision models need their whole input in one micro-batch. Blank omits the flag |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`. Quantized reduces K memory; works with or without Flash Attention |
| **V Cache Type** | `f16` | `--cache-type-v` | Same values. Any quantized value **requires Flash Attention = On** - the UI greys the dropdown and auto-snaps V back to `f16` when FA leaves On |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` omits the flag (llama.cpp's own default); `on` forces it (fails if backend can't); `off` disables |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`. Controls how thinking tags are parsed out (routed to `reasoning_content` vs left inline). `auto` omits the flag |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. How model weights are loaded into memory (successor to deprecated `--mlock`/`--mmap`/`--direct-io`). `auto` = mmap unless unsupported, matches llama.cpp's default and omits the flag |

**Container & Proxy Settings**

| Setting | Default | Flag | Notes |
|---|---|---|---|
| **Port** | *(auto)* | - | Public port; must fall within `PORT_RANGE_START/END` |
| **CPU Threads** | *(auto)* | `--threads` | Also sets the container's CPU quota (`--cpus N`). Blank = let container and llama-server use all cores |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | Separate thread count for prefill/batch. Blank falls back to `--threads` in llama-server |
| **System Memory Limit** | *(none)* | - | Docker RAM cap (e.g. `32g`, `8192m`) |
| **Idle Timeout min** | `0` | - | Auto-sleep after N minutes idle. `0` = off. Live-updated from preset (no relaunch). See [Idle Timeout](#idle-timeout) |
| **Max Concurrent** | `0` | - | Max in-flight inference requests; excess queues. `0` = no gating. See [Concurrency & queueing](#concurrency--queueing) |
| **Max Queue Depth** | `200` | - | Queue cap before HTTP 429 |
| **Exclude from Max Models** | off | - | Not counted toward `LLAMAMAN_MAX_MODELS`, never evicted. See [Model Eviction](MODELS.md#model-eviction). Live-updated from preset |

**GPU Settings**

| Setting | Default | Flag | Notes |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (aliases) | `-1` = **auto** (llama.cpp estimates VRAM and picks; may leave layers on CPU). `-2` = **all** (force every layer; fails if it doesn't fit). `0` = CPU only, no GPU attached to the container. Any positive N offloads exactly N layers from the bottom |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | Only meaningful on Mixture-of-Experts models. `0` = don't emit anything. `-1` = `--cpu-moe` (all layers' routed expert weights pinned to CPU). Positive N = `--n-cpu-moe N` (experts of the first N layers on CPU). Shrinks VRAM ~4-10x on big MoEs because experts dominate the weight count but only a few fire per token. Inert on dense models: llama.cpp's regex (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`) matches nothing and the override is silently dropped. **Does not cover shared experts** (`_shexp` in DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE) - add an `-ot` override in Extra Args for those |
| **GPU Devices** | *(all)* | *(container-level)* | Comma-separated host-GPU indices to attach (e.g. `0,1`). Inside the container they're renumbered from 0. Not supported on Intel Arc |
| **Split Mode** | `layer` | `--split-mode` | `none` (single GPU, ignores Tensor Split) / `layer` (llama.cpp's default; splits whole layers) / `row` (splits tensor rows; typically only wins with fast interconnect like NVLink) |
| **Tensor Split** | *(auto)* | `--tensor-split` | Comma-separated relative weights per container-visible GPU (e.g. `24,16`). Blank auto-fills from each GPU's total VRAM at launch. Ignored when Split Mode = `none` or only one GPU visible |

**Extra Args** *(text field)* - additional flags passed directly to llama-server (e.g. `--mlock`, or n-gram speculative types not surfaced in the form).

**Proxy-Side Sampling Overrides** *(toggle)* - when on, the proxy forces sampling params on every forwarded request:

| Setting | Default | Range |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0 (`0` disables) |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 to 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0 (`0` = disabled) |

**Speculative Decoding** *(toggle)*

| Setting | Default | Flag | Notes |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`. Only `draft-mtp` accepts a blank drafter (falls back to the main model's built-in MTP heads) |
| Draft Model | *(none)* | `-md` | Path to the drafter GGUF. Required for all types except `draft-mtp` |
| Draft N Max | `2` | `--spec-draft-n-max` | Max tokens drafted per step |
| Draft N Min *(Advanced)* | *(auto)* | `--spec-draft-n-min` | Blank omits the flag |
| Draft P Split *(Advanced)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`. Blank omits the flag |
| Draft P Min *(Advanced)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`. Blank omits the flag. `0` is a real value distinct from blank |

**Image & PDF Input** *(toggle)* - see [Image & PDF Input](#image--pdf-input) below.

**Anti-Loop** — two independent sub-toggles (both off by default). See [Anti-Loop](#anti-loop) below.

## Live preset updates

Saving a preset updates already-running instances of that model in place where possible:

- **Live (no relaunch):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, all six proxy-sampling fields, all five loop-detection fields (next request picks them up)
- **Require relaunch:** everything baked into the container at launch (gpu layers, MoE CPU layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, DRY sampler, spec-decoding fields, embedding flag, mmproj/PDF fields, extra args, memory limit)

**Proxy-sampling caveat:** if the instance was launched with all of `idle_timeout=0`, `max_concurrent=0`, and `override_enabled=false`, no sidecar proxy was spawned. Live-toggling `override_enabled=true` then still applies overrides on requests routed through the app's compat endpoints, but direct hits to the public port bypass it. Relaunch to spawn the proxy in that case.

## Concurrency & queueing

When **Max Concurrent** > 0, llamaMan places a FIFO gate in front of the instance. Excess requests wait up to **Max Queue Depth** before returning HTTP 429. Gate stats (active + queued) are visible via the API.

**Parallel vs Max Concurrent:** `Parallel` controls llama-server's internal KV slots; `Max Concurrent` gates how many requests llamaMan forwards at once. Use both together (e.g. `Parallel=4`, `Max Concurrent=4`) to keep the server well-fed without overflowing its slots.

## Image & PDF Input

Enable the **Image & PDF Input** toggle to load a vision model's multimodal projector (`--mmproj`) alongside its main GGUF. The **MMPROJ Model** field takes the projector GGUF (typical filename: `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (on by default): llama.cpp keeps the projector on the GPU unless it gets `--no-mmproj-offload`, so *on* omits the flag entirely (byte-identical CLI to before this option existed) and *off* passes `--no-mmproj-offload` to run the vision encoder on CPU — useful on tight-VRAM setups, where the encoder pins a few GB.

**PDFs** are supported by rasterizing each page to a PNG **before** forwarding to llama-server (which has no native PDF support). Enable **Accept PDF uploads** (inside the Image & PDF Input group) to turn this on for an instance; the master **Image & PDF Input** toggle gates the whole group — turning it off also clears the PDF sub-toggles, since rasterized pages have no vision model to reach without the projector. Recognized in both:

- **OpenAI**: `image_url` with a `data:application/pdf;base64,...` URL, or the newer `file` block
- **Ollama**: base64 PDF bytes mixed into `images[]` - llamaMan sniffs `%PDF-` and expands PDFs; ordinary images keep their existing path

**Text-layer shortcut** (off by default): with **Try text layer first** on, llamaMan runs `pypdf.extract_text()` before rasterizing. Born-digital PDFs (Word/LaTeX/browser print) with substantive text (≥ ~50 chars/page) get inlined as one text block and skip the vision path entirely. Sparse text falls back to rasterization. Off by default because rasterization preserves layout for the vision model, and predictable behavior on every PDF beats a silent shortcut.

**Per-request caps:** `PDF DPI` (72-600, default 200) and `Max PDF Pages` (1-200, default 20). Over-cap PDFs return HTTP 400 rather than being truncated.

**Rasterization concurrency:** PDF rendering is CPU- and RAM-heavy and runs in the gunicorn worker thread before any per-instance gate. A process-wide semaphore caps concurrent rasters; tune via `LLAMAMAN_PDF_MAX_CONCURRENT` (default 4).

## Anti-Loop

Two independent controls in the **Anti-Loop** section of the launch form, both off by default. They target the same problem (models getting stuck in stable output loops) at two different points in the pipeline; use them together for the best coverage.

**DRY sampler (soft, sampling-time)**. Maps to llama-server's `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n`. Every token that would extend a recently-seen n-gram gets a probability penalty at sampling time, so the model is nudged away from repeating itself before it ever emits a duplicate span. This is the *soft* line of defense — usually enough on its own for prose. Baked in at container launch; a preset edit requires a relaunch. Multiplier = 0 disables entirely (llama.cpp's own default).

*DRY Sampler* (baked in at launch — requires relaunch):

| Setting | Default | Flag | Notes |
|---|---|---|---|
| Multiplier | `0.8` (UI value; `0` disables) | `--dry-multiplier` | Penalty strength; llama.cpp treats `0` as disabled |
| Base | `1.75` | `--dry-base` | Exponential base ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | N-grams up to this length are exempt |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Lookback window. Blank omits the flag |

**Output loop detection (hard, post-hoc)**. When DRY isn't enough — the model still gets stuck in a stable attractor — llamaman's proxy watches the assistant-visible output text (content + reasoning) as it streams to the client. Every scan cadence (default: inline every 64 chunks, plus a 10s worker fallback), the last `min_chunk_chars` of the buffer are checked for repetition. If they appear ≥ `min_repetitions` times, the turn is terminated:

- **The upstream connection to llama-server is closed** — llama-server's stop-on-client-disconnect halts generation, freeing the model for the next request.
- **A synthetic terminator is injected into the client's stream** — OpenAI clients see one final delta with `finish_reason=stop` and content `[llamaman: output loop detected, ending turn]`; Ollama clients see `done_reason: "loop_detected"` on a `done: true` line.

*Output Loop Detection* (proxy-side; live-updated from preset — next request picks up new thresholds):

| Setting | Default | Notes |
|---|---|---|
| Min chunk chars | `200` | Range 60-4096. Lower catches shorter loops but false-positives on repetitive-but-legitimate content |
| Min repetitions | `3` | Range 2-20 |
| Buffer size (chars) | `8192` | Rolling window per in-flight request. Range 512-65536 |
| Scan every N chunks | `64` | Inline scan cadence |
| Fallback scan (seconds) | `10` | Background worker for streams that emit tokens too slowly to hit the inline threshold |

**Where it watches:**
- **Compat routes on `:42069`** (Ollama `/api/chat`, `/api/generate`; OpenAI `/v1/chat/completions`, `/v1/completions`) — always, whenever the toggle is on for the target instance's preset.
- **Per-instance ports (`8000-8020`)** — only when a sidecar proxy is already present for that instance (i.e. Idle Timeout, Max Concurrent, or Proxy Sampling Overrides also on). Loop detection does not, by itself, spawn a sidecar proxy — enable one of the other three to also watch direct hits.

**Tuning:**
- `min_chunk_chars` (60-4096, default 200) — the smallest repeating period that will trigger. Lower values catch shorter loops but risk false positives on repetitive-but-legitimate content (numbered lists, poetry choruses, markdown tables). 60-100 is aggressive; 200 is the balanced default for prose; 400+ only triggers on paragraph-scale loops.
- `min_repetitions` (2-20, default 3) — how many exact copies of the chunk must appear before terminating. 2 fires on any doubled chunk (aggressive); 3-5 balances responsiveness against false positives.
- `max_buffer_chars` (512-65536, default 8192) — rolling window kept per active request. Must be at least `min_chunk_chars × min_repetitions`.
- `scan_every_n_tokens` (8-4096, default 64) — inline scan cadence. Lower = faster kill, more CPU.
- `scan_interval_s` (1-600, default 10) — how often the background worker rescans buffers that didn't hit the inline threshold.

Changes to the loop-detection thresholds take effect on the *next* request (in-flight streams keep the thresholds captured at attach time). DRY changes require a relaunch since the flags are baked into the container.

**Detection algorithm** (v1): take the last `min_chunk_chars` of the buffer as the target; count exact occurrences; flag when ≥ `min_repetitions`. Doesn't catch drifting near-repeats ("The answer is: The answer is: The answer is:" with slight punctuation drift between reps) — those get caught by DRY's sampling-time defenses. Deliberately simple and cheap.

**Defensive invariant:** every hook in the streaming path is wrapped in `try/except Exception` that logs and falls through — a detector bug can never break a real user's stream.

## Per-Instance Proxy

When **Idle Timeout**, **Max Concurrent**, or **Proxy Sampling Overrides** are on for an instance, llamaMan inserts a WSGI proxy in front of the llama-server container: the public port (e.g. 8000) is the proxy; llama-server listens on a separate internal port. The proxy does model-name validation, wake-on-request for sleeping instances, `RequestGate` concurrency limiting, and sampling injection.

**Model-name validation:** on inference endpoints, if the request body has a `"model"` field it's compared against the loaded model's filename stem (lowercased, no extension). **Prefix match is accepted** (e.g. `"qwen2.5-0.5b-instruct-q2"` matches `"qwen2.5-0.5b-instruct-q2_k"`). Mismatch returns HTTP 404; for sleeping instances, no wake. Requests with no `"model"` field are forwarded unconditionally.

## Idle Timeout

Set **Idle Timeout min** in the launch form (0 = disabled). When enabled, after N minutes with no requests the llama-server container is stopped to free VRAM; the next request wakes a new container with the same config. Clients see the same port with a cold-start delay.

For proxy-managed (OpenWebUI) instances, use `LLAMAMAN_IDLE_TIMEOUT` instead.
