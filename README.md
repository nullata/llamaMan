<p align="center">
  <img src="docs/llamaman-logo-wide.jpg" alt="llamaMan" width="1000">
</p>

# <img src="static/images/logo.svg" alt="logo" width="24"> llamaMan

Browser UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances. llamaMan is a lightweight Python app with **no dependency on llama.cpp itself** - it spawns `ghcr.io/ggml-org/llama.cpp:server-*` containers as siblings over the Docker socket. Ships an Ollama-compatible API proxy so it drops in as an Ollama replacement for [Open WebUI](https://github.com/open-webui/open-webui).

> llama-server flag semantics are the source of truth on llama.cpp's side: **[server CLI reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)**.

## Contents

- [Features](#features) · [How It Works](#how-it-works) · [Quick Start](#quick-start)
- [Authentication](#authentication) · [Models](#models) · [Launching Instances](#launching-instances) · [Launch settings reference](#launch-settings-reference)
- [Image & PDF Input](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Per-Instance Proxy](#per-instance-proxy) · [Idle Timeout](#idle-timeout) · [GPU Stats](#gpu-stats)
- [Request Recording & Stats](#request-recording--stats) · [Model Eviction](#model-eviction) · [OpenWebUI](#openwebui-integration)
- [Storage & DB Outage Mirror](#storage-backends) · [Clustering](#clustering)
- [Environment Variables](#environment-variables) · [REST API](#rest-api) · [Troubleshooting](#troubleshooting)

## Features

- **Universal GPU image** - one Dockerfile / image flow for NVIDIA, AMD (ROCm), Intel Arc, and CPU; GPU vendor auto-detected (`GPU_TYPE` / `LLAMA_IMAGE` override)
- **Flexible deployment** - runs in Docker (default) or bare-metal on the host (e.g. under WSL); auto-detects which
- **Model library + downloader** - scans for GGUF, pulls from HuggingFace with speed limits, resume, auto-retry, and republish detection with atomic re-pull
- **One-click launch + presets** - per-model launch settings with **live updates** for fields that don't need a relaunch (idle-timeout, gates, sampling overrides)
- **Speculative decoding** - all five draft-model families (`draft-simple/-mtp/-dflash/-dspark/-eagle3`) with advanced knobs (`n-max/n-min/p-split/p-min`)
- **Flash Attention + KV cache quant + reasoning format + load mode** - `--flash-attn`, `--cache-type-k/v`, `--reasoning-format`, `--load-mode` (mmap/mlock/dio) all exposed; UI enforces the quantized-V-requires-FA-On constraint
- **Anti-Loop** - two-tier defense against stuck models: DRY sampler (soft, sampling-time) plus proxy-side output loop detection that watches the streamed text and hard-kills the turn when a large chunk repeats often enough; both off by default, tuned per preset
- **Image & PDF input** - `--mmproj` for vision models; PDFs rasterized page-by-page (or inlined as text via the born-digital shortcut); OpenAI `image_url`/`file` and Ollama `images[]` both supported
- **Instance management + monitoring** - stop/restart/logs; per-GPU VRAM, container CPU% + RAM, and per-instance throughput / TTFT / latency rolled up from the request log
- **Ollama + OpenAI proxy on `:42069`** - Open WebUI drops in; auto-starts models on demand; LRU-evicts once `LLAMAMAN_MAX_MODELS` is hit
- **Auth** - user accounts, API keys (bearer tokens), toggle for whether model endpoints require auth
- **Persistent state** - JSON (default) or MariaDB/MySQL; MariaDB unlocks clustering + optional local DB-outage mirror
- **Multi-node clustering** *(opt-in)* - several llamaman deployments as one logical cluster: aggregated dashboard, cross-node launches/pulls/downloads, shared-queue load balancing
- **Request recording + logging dashboard** - opt-in per-request or per-conversation recording with retention; dedicated Logging page over the record
- **Per-model display names** - a friendly name that API clients (OpenWebUI) see and accept instead of the raw quant filename
- **Docker image management** - pull any llama.cpp image by name, delete old local images from the UI

## How It Works

llamaMan spawns each llama-server as a **sibling container** on the host via the Docker socket. GPU passthrough, port binding, and volume mounts are configured per-container via the Docker SDK.

```
Host machine
├── Docker daemon
│   ├── llamaman container         (Python only - no GPU usage - only monitoring, no llama.cpp)
│   │   └── /var/run/docker.sock   (talks to Docker daemon)
│   ├── llamaman-<id> container    (llama.cpp:server-cuda, GPU attached)
│   └── llamaman-<id> container    (llama.cpp:server-cuda, GPU attached)
└── GPU hardware
```

**Containerized vs bare-metal:** the diagram above is the default (llamaman as a container on `llamaman-net`, reaching siblings by container name). llamaman can also run bare-metal directly on the host - in that case it reaches spawned containers via `localhost` on their published ports. Mode is auto-detected; force it with `LLAMAMAN_IN_DOCKER`.

**Update llama.cpp without rebuilding llamaman:**
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
```

## Requirements

- Docker with access to `/var/run/docker.sock`
- **One** of:
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for NVIDIA GPUs
  - [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for AMD GPUs
  - Intel Arc with `/dev/dri` access
  - CPU-only

## Quick Start

Edit `docker-compose.yml` first:

- Set `HOST_MODELS_DIR` / `HOST_LOGS_DIR` to the **real host paths** that back your volume mounts (llamaMan passes them to the Docker daemon when spawning sibling containers).
- `LLAMAMAN_NODE_NAME` (default `srv1`) is **required for every install** - it's the partition key for this deployment's stored state. Pick once, keep it (changing later orphans stored state).

Pull the llama.cpp image for your GPU and start:

| GPU | `LLAMA_IMAGE` |
|---|---|
| NVIDIA | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| AMD (ROCm) | `ghcr.io/ggml-org/llama.cpp:server-rocm` |
| Intel Arc | `ghcr.io/ggml-org/llama.cpp:server-sycl` |
| CPU only | `ghcr.io/ggml-org/llama.cpp:server` |

```bash
docker pull <LLAMA_IMAGE>          # from the table above
# For non-CUDA: also set LLAMA_IMAGE=... in docker-compose.yml
docker compose up --build
```

For NVIDIA native VRAM monitoring, also uncomment `deploy.resources.reservations` in `docker-compose.yml`.

- **Management UI**: <http://localhost:5000> (create admin on `/setup`)
- **Ollama-compatible API**: <http://localhost:42069>
- **llama-server instance ports**: 8000-8020

> **Security note:** the mounted `/var/run/docker.sock` gives the container full control of Docker on the host. Standard Docker-socket-access implications apply.

### Running bare-metal

Useful for dev or hosts (e.g. WSL) where running the manager itself in Docker is awkward. llama-server still runs in containers.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

A single process serves the UI/API on `:5000` and the proxy on `:42069`. Container-vs-bare-metal is auto-detected; if detection is ever wrong for your runtime, set `LLAMAMAN_IN_DOCKER=true` or `false` explicitly.

## Authentication

Two credential types:

- **Session cookies** for the UI (created via `/setup` on first launch)
- **API keys** for external clients (`Authorization: Bearer llm-xxxxxxxxxx`), managed under **API Keys** in the UI

The **"Require authentication for all endpoints"** toggle (on by default) controls whether model-serving endpoints demand a bearer token:

| Toggle | Model endpoints (`/api/chat`, `/v1/chat/completions`, per-instance ports) | Management endpoints (`/api/instances`, etc.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (default) | Bearer token required | Bearer token or session |
| **OFF** | Open | Bearer token or session |

All three port surfaces (5000, 42069, 8000-8020) go through the same auth hook.

### OpenWebUI with authentication

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

## Models

Place models under the `/models` volume as `.gguf` files, HuggingFace repo directories (containing `config.json`), or use the **Download** button in the UI.

### Display Name

Each model can be given a friendly **Display Name** (in the Launch form). When set it's the id API clients see on `/api/tags` and `/v1/models`, so OpenWebUI shows `Qwen 2.5 14B` instead of `Qwen2.5-14B-Instruct-Q4_K_M`. Must be unique and not clash with another model's filename or a cluster queue-group name.

### Model updates

The **Check for updates** button (shown for models with a known source repo) compares the local file against the repo's published content hash:

- **Up to date** - recorded hash matches
- **Update available** - re-pulls into a staging folder and atomically swaps in only once the download completes (a failed pull leaves the current model intact). Refuses with 409 while the model is loaded.
- **Verify hash** - hashes the local file (background job with progress) when no hash has been recorded yet

An opt-in background scan (**Settings → Download Settings**) keeps these answers ready ahead of time.

## Launching Instances

1. Select a model from the sidebar
2. Configure launch settings (or use the saved preset)
3. Click **Launch** - llamaMan spawns a llama-server container and the instance appears with a status badge
4. Optionally **Save Preset** to remember settings

With settings collapsed, a **Quick Launch** button starts the selected model straight from its preset.

Each instance exposes an OpenAI-compatible API on its assigned port. When a GGUF is selected, llamaMan reads its metadata to detect layer count and shows it next to **GPU Layers** (e.g. `/ 32`).

### Launch settings reference

Behavior notes below are terse; hover the field's info-tip in the UI for the full explanation. All flags map to llama-server unless stated - see the [llama-server CLI reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for exact semantics and current defaults.

**Model Settings**

| Setting | Default | Flag | Notes |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Prompt + response token cap per request |
| **Parallel Slots** | `1` | `--parallel` | Concurrent decoding slots inside llama-server (KV slots) |
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
| **Idle Timeout min** | `0` | - | Auto-sleep after N minutes idle. `0` = off. Live-updated from preset (no relaunch) |
| **Max Concurrent** | `0` | - | Max in-flight inference requests; excess queues. `0` = no gating |
| **Max Queue Depth** | `200` | - | Queue cap before HTTP 429 |

**GPU Settings**

| Setting | Default | Flag | Notes |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (aliases) | `-1` = **auto** (llama.cpp estimates VRAM and picks; may leave layers on CPU). `-2` = **all** (force every layer; fails if it doesn't fit). `0` = CPU only, no GPU attached to the container. Any positive N offloads exactly N layers from the bottom |
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

*DRY Sampler* (baked in at launch — requires relaunch):

| Setting | Default | Flag | Notes |
|---|---|---|---|
| Multiplier | `0.8` (UI value; `0` disables) | `--dry-multiplier` | Penalty strength; llama.cpp treats `0` as disabled |
| Base | `1.75` | `--dry-base` | Exponential base ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | N-grams up to this length are exempt |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Lookback window. Blank omits the flag |

*Output Loop Detection* (proxy-side; live-updated from preset — next request picks up new thresholds):

| Setting | Default | Notes |
|---|---|---|
| Min chunk chars | `200` | Range 60-4096. Lower catches shorter loops but false-positives on repetitive-but-legitimate content |
| Min repetitions | `3` | Range 2-20 |
| Buffer size (chars) | `8192` | Rolling window per in-flight request. Range 512-65536 |
| Scan every N chunks | `64` | Inline scan cadence |
| Fallback scan (seconds) | `10` | Background worker for streams that emit tokens too slowly to hit the inline threshold |

Applies to Ollama / OpenAI compat routes (`:42069`) always, and to per-instance ports only when a sidecar proxy is already present (Idle Timeout, Max Concurrent, or Proxy Sampling Overrides also on). On detection, llamaman closes the upstream connection (which stops llama-server generating) and injects a terminator into the client stream — OpenAI clients see a `finish_reason=stop` chunk with a `[llamaman: output loop detected, ending turn]` marker; Ollama clients see `done_reason: "loop_detected"`.

### Live preset updates

Saving a preset updates already-running instances of that model in place where possible:

- **Live (no relaunch):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, all six proxy-sampling fields, all five loop-detection fields (next request picks them up)
- **Require relaunch:** everything baked into the container at launch (gpu layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, DRY sampler, spec-decoding fields, embedding flag, mmproj/PDF fields, extra args, memory limit)

**Proxy-sampling caveat:** if the instance was launched with all of `idle_timeout=0`, `max_concurrent=0`, and `override_enabled=false`, no sidecar proxy was spawned. Live-toggling `override_enabled=true` then still applies overrides on requests routed through the app's compat endpoints, but direct hits to the public port bypass it. Relaunch to spawn the proxy in that case.

### Concurrency & queueing

When **Max Concurrent** > 0, llamaMan places a FIFO gate in front of the instance. Excess requests wait up to **Max Queue Depth** before returning HTTP 429. Gate stats (active + queued) are visible via the API.

**Parallel vs Max Concurrent:** `Parallel` controls llama-server's internal KV slots; `Max Concurrent` gates how many requests llamaMan forwards at once. Use both together (e.g. `Parallel=4`, `Max Concurrent=4`) to keep the server well-fed without overflowing its slots.

## Image & PDF Input

Enable the **Image & PDF Input** toggle to load a vision model's multimodal projector (`--mmproj`) alongside its main GGUF. The **MMPROJ Model** field takes the projector GGUF (typical filename: `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (on by default): llama.cpp keeps the projector on the GPU unless it gets `--no-mmproj-offload`, so *on* omits the flag entirely (byte-identical CLI to before this option existed) and *off* passes `--no-mmproj-offload` to run the vision encoder on CPU — useful on tight-VRAM setups, where the encoder pins a few GB.

**PDFs** are supported by rasterizing each page to a PNG **before** forwarding to llama-server (which has no native PDF support). Enable **Accept PDF uploads** to turn this on for an instance. Recognized in both:

- **OpenAI**: `image_url` with a `data:application/pdf;base64,...` URL, or the newer `file` block
- **Ollama**: base64 PDF bytes mixed into `images[]` - llamaMan sniffs `%PDF-` and expands PDFs; ordinary images keep their existing path

**Text-layer shortcut** (off by default): with **Try text layer first** on, llamaMan runs `pypdf.extract_text()` before rasterizing. Born-digital PDFs (Word/LaTeX/browser print) with substantive text (≥ ~50 chars/page) get inlined as one text block and skip the vision path entirely. Sparse text falls back to rasterization. Off by default because rasterization preserves layout for the vision model, and predictable behavior on every PDF beats a silent shortcut.

**Per-request caps:** `PDF DPI` (72-600, default 200) and `Max PDF Pages` (1-200, default 20). Over-cap PDFs return HTTP 400 rather than being truncated.

**Rasterization concurrency:** PDF rendering is CPU- and RAM-heavy and runs in the gunicorn worker thread before any per-instance gate. A process-wide semaphore caps concurrent rasters; tune via `LLAMAMAN_PDF_MAX_CONCURRENT` (default 4).

## Anti-Loop

Two independent controls in the **Anti-Loop** section of the launch form, both off by default. They target the same problem (models getting stuck in stable output loops) at two different points in the pipeline; use them together for the best coverage.

**DRY sampler (soft, sampling-time)**. Maps to llama-server's `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n`. Every token that would extend a recently-seen n-gram gets a probability penalty at sampling time, so the model is nudged away from repeating itself before it ever emits a duplicate span. This is the *soft* line of defense — usually enough on its own for prose. Baked in at container launch; a preset edit requires a relaunch. Multiplier = 0 disables entirely (llama.cpp's own default).

**Output loop detection (hard, post-hoc)**. When DRY isn't enough — the model still gets stuck in a stable attractor — llamaman's proxy watches the assistant-visible output text (content + reasoning) as it streams to the client. Every scan cadence (default: inline every 64 chunks, plus a 10s worker fallback), the last `min_chunk_chars` of the buffer are checked for repetition. If they appear ≥ `min_repetitions` times, the turn is terminated:

- **The upstream connection to llama-server is closed** — llama-server's stop-on-client-disconnect halts generation, freeing the model for the next request.
- **A synthetic terminator is injected into the client's stream** — OpenAI clients see one final delta with `finish_reason=stop` and content `[llamaman: output loop detected, ending turn]`; Ollama clients see `done_reason: "loop_detected"` on a `done: true` line.

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

## GPU Stats

Per-GPU VRAM and utilization, queried natively - no running llama-server required.

| Vendor | Method | Requirement |
|---|---|---|
| NVIDIA | `pynvml` (NVML direct) | Uncomment `deploy.resources.reservations` in `docker-compose.yml` for NVIDIA toolkit `utility` capability |
| AMD | `/sys/class/drm` sysfs | `/sys/class/drm:ro` volume mount (included by default) |
| Intel Arc | `/sys/class/drm` sysfs | Same mount as AMD |

Without native access, llamaMan falls back to exec-ing `nvidia-smi` / `rocm-smi` inside a running container. Stats always reflect full host GPU state.

## Request Recording & Stats

Under **Settings → App Settings → Request recording**: **Off** (default) / **Per request** / **Per conversation** (turns grouped by a content hash of the system prompt + first user message). Each record captures the bodies + envelope fields (model, endpoint, status, duration, token counts) plus **accurate per-turn metrics** (throughput measured over the generation window; TTFT). Records live under `RECORDINGS_DIR` (JSON) or the `request_log` table (MariaDB). **Retention (days)** prunes hourly (`0` = keep forever).

Each instance card exposes a **Stats** button (request count, avg/peak throughput, avg TTFT/latency, token totals, active time span) rolled up from the request log - so it persists after the instance is stopped and shows an empty state when recording is off. The **Logging** link in the header opens a full-page dashboard (summary tiles, recent conversations, per-conversation drill-down) with a 24h / 7d / 30d / All time-window selector.

## Model Eviction

`LLAMAMAN_MAX_MODELS` caps concurrent **chat** models via the proxy. When the cap is full and a new chat model is requested, the LRU chat model is evicted:

| Launcher | Evicts | Cannot evict |
|---|---|---|
| **Admin UI** | Ollama-managed first (LRU), then admin-launched | - |
| **Ollama API** | Ollama-managed (LRU) | Admin-launched *(by default)* |
| **OpenAI API** | Nothing *(by default; returns 503 when full)* | Everything *(by default)* |

Three toggles under **Settings → App Settings** relax these defaults:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - silent LRU-evict before launch (Ollama-managed first). Off by default → UI prompts first.
- **Allow Ollama API to evict admin-launched models** - fallback when no Ollama-managed models are available. Off by default.
- **Allow OpenAI API to evict admin-launched models** - grants OpenAI API the same LRU eviction as Ollama. Off by default.

Other details:

- All running chat instances count (both admin UI and proxy-managed).
- **Sleeping instances still count** (slot claim persists across the idle pause) - but waking a sleeper for its own model is never blocked by the cap.
- **Embedding models are excluded** and never evicted.
- `LLAMAMAN_MAX_MODELS=0` disables eviction entirely.

## OpenWebUI Integration

Point OpenWebUI at the Ollama proxy (with an API key when `require_auth` is on):

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

**How it works:** OpenWebUI hits `/api/tags` → llamaMan lists available GGUFs; selecting a model → `/api/chat` arrives → llamaMan spawns a llama-server container (using saved preset or defaults), waits healthy, then proxies. When the cap is hit, the LRU Ollama-managed model is evicted (see [Model Eviction](#model-eviction)).

Supported: Ollama `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`; OpenAI `/v1/models`, `/v1/chat/completions`.

Models are listed by GGUF filename stem by default. Set a per-model **Display Name** to have OpenWebUI show/accept a friendly name. In a cluster, live shared-queue group aliases are also advertised as selectable models.

## Download Settings

Under **Settings → Download Settings**:

- **Auto-retry failed downloads** (off by default) + **Retry count per failed download** (default 3)
- **Check models for updates in the background** (off by default) - opt-in worker that asks each source repo whether a file has been republished, plus computes a checksum for any model that doesn't have one yet. **Update check interval (hours)** default 24. Hashes at most one model per pass; never runs while a download is in progress.

## Docker Image Management

**Settings → Docker Images**: pull any llama.cpp image by name, delete old local images (disabled for the active `LLAMA_IMAGE`, and returns an error if Docker refuses because a container is using it), and optionally auto-update the active image on a schedule.

## Model Backup and Restore

**Settings → App Settings**:

- **Download Stored Models JSON** - exports all scanned models + preset configs to a timestamped JSON
- **Restore from JSON** - per model: already-present → preset merged in (existing values preserved); missing but with a HuggingFace source → queued to download with preset pre-populated; missing and no known source → reported as unrestorable

## Cleanup Settings

Under **Settings → Cleanup Settings** (all runs periodically in the background; only removes/updates records - never deletes model files):

- **Auto-clean completed/failed downloads** (default 24h) - active downloads never touched
- **Auto-clean stopped instances** (default 24h) - running instances never removed
- **Auto-remove stale instance records** (default 5 min interval) - re-checks `starting`/`healthy`/`sleeping` records against Docker; ones whose container is gone are marked stopped

## Storage Backends

### JSON (default)

Zero-config. Files under `DATA_DIR` (`/data`): `state.json`, `presets.json`, `users.json`, `settings.json`, `api_keys.json`, `request_log/`. Instance and download logs go to `LOGS_DIR` (`/tmp/llama-logs`).

### MariaDB / MySQL

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
```

```yaml
environment:
  - DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

Tables auto-create on first connect. Request logs go to the `request_log` table instead of the JSON `request_log/` directory (`RECORDINGS_DIR` has no effect with MariaDB).

Per-node model metadata (source repo + content hash) lives in a node-scoped `model_files` table keyed by `(node_id, model_path)`. On upgrade, each node copies its own entries out of the legacy shared settings blob (only paths that exist on its own disk) and keeps the blob as a read fallback and rollback path. Migrations are versioned per node.

> **InnoDB note:** `(node_id, model_path)` is `(64 + 700) × 4 = 3056` bytes under utf8mb4, just under InnoDB's 3072-byte index limit, so the table is created with `ROW_FORMAT=DYNAMIC`.

### Surviving a database outage (local mirror)

*Optional, off by default.*

With `DATABASE_URL` set, the database is on the critical path of **every** request (auth checks read settings and verify API keys on each call). If it becomes unreachable the node stops serving, and a container restart mid-outage can't boot at all.

Turn on **Settings → App settings → "Keep a local mirror of the database"** (or set `LLAMAMAN_DB_MIRROR=1`) and the node keeps a write-through copy in `DATA_DIR/db_mirror/`.

| Still works while offline | Blocked while offline |
|---|---|
| All inference (both ports) | Creating the first user (`/setup`) - a brand-new install has no mirror to fall back to anyway |
| Launching / stopping / sleeping / waking models | |
| Downloads, model hashing and update checks | |
| Creating and editing **presets** (including for models you just downloaded) | |
| Creating and revoking **API keys** | |
| Changing download and app **settings** | |
| Adding and removing **Hugging Face tokens** | |

Offline changes go into an append-only journal and are replayed on reconnect (10s probe). Replay records the *edit*, not the resulting row, so a change another node made during the outage survives (a different preset field, a different settings key, a different HF token are all left intact).

**Requirements and caveats:**

- **`DATA_DIR` must be a persistent volume** (already is in the sample compose file).
- **Each node needs its own `DATA_DIR`.** The mirror is stamped with the owning node id; two nodes sharing one switches mirroring off rather than corrupting either view.
- **Secrets land on local disk.** Settings are copied verbatim; **Hugging Face tokens are mirrored in plaintext.** Password + API-key hashes are already hashed. If you chose MariaDB partly to keep secrets off individual hosts, this trade-off is why the feature is opt-in.
- **Cross-node balancing stops during an outage.** Peer liveness lives in the database; a degraded node sees only itself.
- **API key changes are local until reconnect.** A key created offline works only on this node; a key **revoked** offline stays valid on other nodes until reconnect. Plan revocations accordingly.
- **Request logs are dropped, not buffered,** while offline (the records carry full bodies; spooling them through a long outage could fill the volume).
- **Turning the mirror off mid-outage is deferred** until reconnect.
- Schema migrations never run while degraded; they run against the real database on reconnect, before any journal replay.

## Clustering

*Optional, off by default - single-node installs are completely unaffected.*

Several llamaMan deployments act as **one logical cluster**: a single dashboard aggregating every node's GPUs, instances, and downloads, with cross-node launches/pulls/downloads and multi-node shared-queue load balancing. Nodes discover each other through the shared storage backend - no pairwise key exchange.

**Requirements:**

- **Shared storage backend** - every node points at the **same** `DATABASE_URL` (MariaDB/MySQL). JSON is per-host and can't be shared.
- **Unique `LLAMAMAN_NODE_NAME` per node** - each node's cluster identity (and required for every install anyway).
- **The same `CLUSTER_SECRET` on every node** - the bearer token (sent as `X-Cluster-Secret`) for all node-to-node HTTP.
- **`CLUSTER_ADVERTISE_URL` per node** for cross-node *actions* - a hostname/IP routable from the **other** hosts (not `localhost`), e.g. `http://srv1:5000`. A node without one appears in the dashboard but is view-only.

Set on **each** node (only `LLAMAMAN_NODE_NAME` and `CLUSTER_ADVERTISE_URL` differ):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Each node heartbeats every ~5s; a node silent past `CLUSTER_NODE_ONLINE_WINDOW_S` (default 45s) shows offline. Inspect and manage under **Settings → Cluster**.

A few settings are scoped per node because they're host-specific: tracked **Docker images** and the model-cap eviction toggles. Everything else is shared cluster-wide. Live shared-queue group aliases are advertised as selectable models in `/api/tags` and `/v1/models` (deduped cluster-wide), so a client can send the alias and have it routed to the least-loaded node serving it.

> **Security:** the cluster secret lets any peer drive actions on this node. Run node-to-node traffic over a trusted network or behind TLS.

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(required)* | **Required.** Unique stable identity - partition key for stored state and cluster identity. Any string; pick once, keep it (changing later orphans stored state). |
| `MODELS_DIR` | `/models` | Directory scanned for model files (container path) |
| `DATA_DIR` | `/data` | Directory for persistent config/state (JSON files) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Directory for per-conversation request log records. JSON backend only; ignored when `DATABASE_URL` is set |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs (container path) |
| `HOST_MODELS_DIR` | *(same as `MODELS_DIR`)* | **Host-side** absolute path of the models volume. Must match the left side of `-v /host/path:/models` - passed to the Docker daemon when spawning sibling containers |
| `HOST_LOGS_DIR` | *(same as `LOGS_DIR`)* | Host-side absolute path of the logs volume. Same requirement as `HOST_MODELS_DIR` |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Public llama-server/proxy port pool |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Internal port pool used when proxy mode is enabled |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent chat models via the proxy (LRU eviction; 0 = unlimited) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle-timeout minutes for proxy-managed instances (0 = disabled) |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Max concurrent PDF rasterizations across the whole process. Each raster spawns poppler + a few hundred MB of transient RAM at high DPI |
| `SECRET_KEY` | *(auto)* | Flask session secret. Auto-derived from machine-id if unset. Set for multi-replica deployments |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Session cookie name. Namespaced so llamaman coexists with other Flask apps on the same host - cookies are scoped by host+path, not port |
| `DATABASE_URL` | *(unset)* | MariaDB/MySQL connection string. Unset = JSON files |
| `LLAMAMAN_DB_MIRROR` | *(unset)* | Force local DB mirror on (`1`) / off (`0`), overriding the per-node setting. Only meaningful with `DATABASE_URL` |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy. Raise for very large models |
| `REQUEST_TIMEOUT` | `300` | **Read** timeout in seconds for upstream requests to llama-server, cross-node forwarding, and gate acquire waits. Does **not** govern peer connect time (separate 5s bound) |

### Docker / GPU

| Variable | Default | Description |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | llama.cpp Docker image for all spawned containers. Auto-selected from detected GPU vendor; set to pin a version/backend |
| `LLAMA_NETWORK` | `llamaman-net` | Docker network for llamaman + all llama-server containers. Created if missing |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Name prefix for spawned llama-server containers |
| `LLAMAMAN_IN_DOCKER` | *(auto)* | Whether llamaman itself runs in a container. Auto-detected from marker files + cgroups. Set `true`/`false` to override |
| `LLAMA_HOST_ADDR` | `localhost` | Host address used to reach spawned containers' published ports when running bare-metal |
| `GPU_TYPE` | *(auto)* | Override GPU vendor detection: `cuda`, `rocm`, or `intel` |
| `LLAMA_GPU_DEVICES` | *(all)* | Comma-separated GPU indices visible to all spawned containers (e.g. `0,1,3`). Per-instance **GPU Devices** overrides. Not supported on Intel Arc |

### Clustering

Optional. `LLAMAMAN_NODE_NAME` (under Core) doubles as the cluster identity and is required for all installs.

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Set truthy to join this node to a cluster. Requires `CLUSTER_SECRET`; ignored with a warning if secret is empty |
| `CLUSTER_SECRET` | *(unset)* | Shared bearer token (`X-Cluster-Secret`) for all node-to-node HTTP. Identical on every node. Use a long random value over a trusted network or behind TLS |
| `CLUSTER_ADVERTISE_URL` | *(unset)* | How peers reach **this** node's UI/API - hostname/IP routable from the other hosts (not `localhost`). Without it the node is view-only |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Seconds since last heartbeat before a node shows offline. Raise if nodes flap under load or clock skew |
| `LLAMAMAN_PERF_LOG` | *(off)* | Diagnostics. Set truthy to log per-phase timings (`perf <name> <ms>`) for suspected slow paths: remote instance stop, cluster proxy forwards, reachability probes, heartbeat snapshot, auth checks, `save_state`, container stats. Off by default; negligible overhead when off |

## REST API

All endpoints return / accept JSON. Management endpoints need a session cookie (browser login) or `Authorization: Bearer <key>`. Model-serving endpoints also require a bearer when `require_auth` is on (default).

### Auth & keys

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/login` | Login page / authenticate |
| `GET` / `POST` | `/setup` | First-run setup |
| `GET` | `/logout` | End session |
| `GET` / `POST` | `/api/api-keys` | List / create API keys |
| `DELETE` | `/api/api-keys/<id>` | Revoke |

### Instances

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/instances` | List / launch |
| `GET` / `DELETE` | `/api/instances/<id>` | Get / stop and remove |
| `POST` | `/api/instances/<id>/restart` | Restart a stopped/sleeping instance |
| `DELETE` | `/api/instances/<id>/remove` | Remove a stopped-instance record |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Tail / SSE stream logs |
| `GET` | `/api/next-port` | Get next available port from the pool |

**Launch body** (`POST /api/instances`): the launch form's fields, serialized as JSON. See the [Launch settings reference](#launch-settings-reference) above for every field and its default; enum-valued fields (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) accept only the values listed there and default to the reference's default. Numeric fields left as `null` or omitted use the reference default; unknown fields are ignored.

Boundary quirks worth pinning:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp estimates VRAM), `-2` = **all** (force every layer), `0` = CPU only (no GPU attached to the container), positive N = exactly N layers.
- `threads`: when set, applies `--threads N` to llama-server **and** sets the container CPU quota to N cores.
- `threads_batch`: null / omitted → the flag is not emitted and llama-server falls back to `--threads`.
- `flash_attn`: legacy `true` / `false` values from pre-tri-state configs are folded on read (`true` → `"on"`, `false` → `"off"`), so no storage migration is needed.
- `cache_type_v` quantized values require `flash_attn: "on"` - `"auto"` is not a guarantee; llama-server refuses to start otherwise.
- `spec_draft_model` is required when `spec_enabled: true` and `spec_type` is anything other than `"draft-mtp"` - a request without it is rejected 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / omitted omits the flag. `0` is a real value distinct from null and is passed through.

### Downloads, Models, Presets

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | List / start |
| `GET` / `DELETE` | `/api/downloads/<id>` | Get / cancel |
| `DELETE` | `/api/downloads/<id>/remove` | Remove a completed/failed entry |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Tail / SSE stream |
| `GET` | `/api/models` | Discovered models (includes `repo_id` when known) |
| `POST` | `/api/models/delete` | Delete from disk (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Read layer count from GGUF metadata |
| `GET` | `/api/disk-space` | Free/used space on the models volume |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Get / save / delete a preset (`GET /api/presets` lists all) |

Download body: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Blank `filename` pulls the full repo.

### Settings, System, Request log

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Get / save global settings |
| `GET` | `/api/system-info` | CPU usage, core count, RAM usage |
| `GET` | `/api/gpu-info` | Per-GPU VRAM and utilization (native; falls back to container exec) |
| `GET` | `/health` | `{"status": "ok"}` - always open, no auth required |
| `GET` | `/api/request-log/conversations` | Recent conversations with rolled-up metadata (`limit` query, default 100, max 500) |
| `GET` | `/api/request-log/conversations/<id>` | All recorded turns for one conversation, oldest first |
| `GET` | `/api/request-log/stats` | Aggregate metrics (tokens, avg/peak tokens/s, TTFT, latency, errors, streamed counts). Optional `inst_id` + `window_hours` |

### Ollama & OpenAI compat

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate auto-start).
OpenAI: `/v1/models`, `/v1/chat/completions` (chat auto-starts).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Instance stuck on **starting** | Check logs via the Logs button. Common: OOM, model path typo, corrupt GGUF, image not pulled |
| _"Docker image not found"_ | Pull the matching image: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (or `server-rocm` / `server-sycl` / `server`) |
| _"Docker API error"_ on launch | Ensure `/var/run/docker.sock` is mounted (default in `docker-compose.yml`) |
| No GPU / CUDA error | Ensure NVIDIA Container Toolkit is installed and `docker run --gpus all` works on the host |
| No GPU / ROCm error | Ensure `/dev/kfd` and `/dev/dri` exist and your user is in `video`/`render` |
| No GPU / Intel Arc error | Ensure `/dev/dri` is accessible and your user is in `video`/`render` |
| GPU stats unavailable | NVIDIA: uncomment the `deploy.resources.reservations` block. AMD/Intel: ensure `/sys/class/drm:ro` is mounted |
| Wrong GPU vendor detected | Set `GPU_TYPE=cuda`/`rocm`/`intel` to override |
| Instance stuck on **starting** running bare-metal | The container is healthy but llamaman can't reach it. Set `LLAMAMAN_IN_DOCKER=false`/`true` explicitly if auto-detection is wrong for your runtime |
| Stats modal is empty | Enable **Settings → App Settings → Request recording** |
| Launch fails with GPU/CDI error on a host without GPU passthrough | Set **GPU Layers** to `0` for CPU-only with no GPU device attached, or fix the GPU runtime |
| Port conflict | The form auto-suggests an unused port; adjust if needed |
| Model not showing in OpenWebUI | Ensure `OLLAMA_BASE_URL=http://llamaman:42069`. Check `/api/tags` returns models |
| OpenWebUI gets 401 | `require_auth` is on (default). Create an API key and set `OPENAI_API_KEYS` in OpenWebUI |
| Containers not cleaned up after stop | llamaMan removes containers on stop. Orphans after a crash: `docker ps --filter name=llamaman-`, or restart llamaMan (orphan adoption runs on startup) |
| Client (Hermes / OpenWebUI / etc.) reports the trained context window instead of the preset cap | Upgrade to 1.1.2+. `/api/ps` now includes `context_length` set to the runtime ctx the instance was launched with, and `/api/show`'s `model_info["<arch>.context_length"]` is overridden with the effective cap |

## Credits

This work would not be possible without [ggml-org/llama.cpp](https://github.com/ggerganov/llama.cpp).

## License

llamaMan is licensed under the [Elastic License 2.0](LICENSE). You may use, copy, distribute, and modify the software, subject to:

- No providing the software to third parties as a hosted / managed service where the service gives users access to a substantial set of its features or functionality.
- No removing or obscuring licensing, copyright, or other notices of the licensor.

### Third-party licenses

- **[Font Awesome Free 7.1.0](https://fontawesome.com/)** by Fonticons, Inc. - icons (CC BY 4.0), fonts (SIL OFL 1.1), and code (MIT). Full license text ships in [`static/fontawesome-free-7.1.0-web/LICENSE.txt`](static/fontawesome-free-7.1.0-web/LICENSE.txt).
