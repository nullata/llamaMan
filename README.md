<p align="center">
  <img src="docs/llamaman-logo-wide.jpg" alt="llamaMan" width="1000">
</p>

# <img src="static/images/logo.svg" alt="logo" width="24"> llamaMan

A browser-based UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances. llamaMan runs as a lightweight Python container and spawns llama-server as sibling Docker containers using the official llama.cpp images. Includes an Ollama-compatible API proxy so it works as a drop-in replacement for Ollama with [Open WebUI](https://github.com/open-webui/open-webui).

## Features

- **Universal GPU support** - one `Dockerfile` and image flow for NVIDIA, AMD (ROCm), Intel Arc, and CPU. The GPU vendor and matching `LLAMA_IMAGE` are auto-detected at startup; `GPU_TYPE` / `LLAMA_IMAGE` override if needed.
- **Flexible deployment** - run llamaman in Docker (default) or bare-metal on the host (e.g. under WSL). It auto-detects which and reaches spawned containers accordingly.
- **Multi-node clustering** *(optional)* - run several llamaman deployments as one cluster sharing a database and a secret: aggregated dashboard, cross-node launches/pulls/downloads, and multi-node shared-queue load balancing. Off by default; single-node installs are unaffected.
- **Model library** - scans `/models` for GGUF files, shows quant type and file size
- **One-click launch** - configure GPU layers, context size, threads, multi-GPU, speculative decoding, extra args. With the Settings card collapsed, a Quick Launch button starts the selected model straight from its preset
- **Speculative decoding** - optional `--spec-type` toggle exposing all five draft-model-family values llama.cpp accepts: `draft-simple` (any smaller model with the same tokenizer), `draft-mtp` (standalone MTP drafter or the main model's built-in MTP heads), `draft-dflash`, `draft-dspark`, `draft-eagle3`. Configurable draft length (`--spec-draft-n-max`) plus an **Advanced** subsection for `--spec-draft-n-min` / `--spec-draft-p-split` / `--spec-draft-p-min`. The n-gram / lookup speculative family (no drafter model) stays reachable via Extra Args
- **Flash Attention + KV cache quantization** - independent Flash Attention select (`--flash-attn [on|off|auto]`, default `auto`) and per-side cache-type dropdowns (`--cache-type-k` / `--cache-type-v`, choose from `f16` default / `f32` / `bf16` / `q8_0` / `q5_1` / `q5_0` / `iq4_nl` / `q4_1` / `q4_0`). The UI enforces llama-server's own guard - a quantized V cache requires Flash Attention set explicitly to **On** (Auto is not enough; llama.cpp may resolve it to off on some backends), and the form auto-snaps V back to `f16` the moment Flash Attention leaves On so it can't submit a combination llama-server would reject
- **Preset configs** - save/load per-model launch settings, with live updates to running instances where possible
- **Download manager** - pull models from HuggingFace with speed throttling and auto-retry on failure
- **Model update detection & re-pull** - detects when a repo has republished a model under the same filenames (requant, fixed template) via its published content hash, verifies local files by hashing them on disk, and re-pulls through the normal download pipeline with an atomic swap. Optional background scan keeps the answer ready
- **Model backup and restore** - export all model metadata and presets to JSON, restore on any instance by re-queuing missing downloads automatically
- **Instance management** - stop, restart, remove, view live-streamed logs
- **GPU VRAM indicator** - per-GPU VRAM and utilization, queried natively (no running instance required)
- **Container resource monitoring** - live CPU%, core quota, RAM usage with thin progress bars, and GPU assignment per running instance card
- **Per-instance stats** - a Stats button on each instance card surfaces throughput (tokens/s), time-to-first-token, latency, and token totals rolled up from the request log
- **Request log dashboard** - a dedicated Logging page with summary tiles, a conversations list, and per-conversation drill-down over the recorded request log, filterable by time window
- **Request recording** - optionally record proxied requests/responses per request or per conversation, with configurable retention
- **Idle timeout** - auto-sleep instances after configurable idle period, wake on next request
- **Ollama-compatible proxy** - OpenWebUI discovers models and auto-starts servers on demand
- **Per-model display names** - give a model a friendly name that API clients (OpenWebUI) see and accept instead of the raw quant filename
- **Authentication** - user accounts with session login, API key management with bearer tokens
- **Require auth toggle** - enforce bearer token authentication on all endpoints (including model loading) or leave model endpoints open
- **Persistent state** - instance history and configs survive container restarts
- **Storage backends** - JSON files (default) or MariaDB/MySQL via SQLAlchemy
- **Database outage survival** *(optional)* - with a database backend, keep a write-through mirror on local disk so the node keeps serving inference, launching models, and saving presets/API keys/settings if the database goes away - including across a container restart, which otherwise can't boot at all. Offline changes are journalled and synced back automatically when the database returns. Off by default
- **Proxy sampling overrides** - force temperature, top-k, top-p, presence penalty, and repeat penalty on all proxied requests, configurable per model preset
- **CPU quota + memory limit** - CPU Threads also applies a Docker CPU quota; a Memory Limit field caps container RAM
- **Docker image management** - pull any llama.cpp image by name, delete old local images from the Settings UI

## How It Works

llamaMan is a lightweight Python web app with no dependency on llama.cpp itself. When you launch a model, llamaMan uses the Docker socket to spawn a `ghcr.io/ggml-org/llama.cpp:server-*` container as a sibling on the host. GPU passthrough, port binding, and volume mounts are configured per-container via the Docker SDK.

```
Host machine
├── Docker daemon
│   ├── llamaman container        (Python only - no GPU usage - only monitoring, no llama.cpp)
│   │   └── /var/run/docker.sock  (talks to Docker daemon)
│   ├── llamaman-<id> container   (llama.cpp:server-cuda, GPU attached)
│   └── llamaman-<id> container   (llama.cpp:server-cuda, GPU attached)
└── GPU hardware
```

**Containerized vs bare-metal:** the diagram above shows the default - llamaman running as a container alongside its spawned siblings on the `llamaman-net` Docker network, reaching them by container name. llamaman can also run bare-metal directly on the host (e.g. a Python process under WSL); in that case it reaches the spawned containers via `localhost` on their published ports. The mode is auto-detected (marker files + cgroup inspection) and can be forced with `LLAMAMAN_IN_DOCKER`.

**To update llama.cpp** - no llamaman rebuild needed:
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
```

## Requirements

- Docker with access to `/var/run/docker.sock`
- **One** of:
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for NVIDIA GPUs
  - [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for AMD GPUs
  - Intel Arc with `/dev/dri` access for Intel GPUs
- The matching llama.cpp server image pulled on the host (see Quick Start)

## Quick Start

Before starting, edit `docker-compose.yml` and set the two host path variables to match your volume mount sources:

```yaml
- HOST_MODELS_DIR=/absolute/host/path/to/models
- HOST_LOGS_DIR=/absolute/host/path/to/logs
```

These must be the real paths on the Docker host. llamaMan passes them to the Docker daemon when spawning sibling llama-server containers, so they must resolve on the host - not inside the llamaman container.

The bundled `docker-compose.yml` also sets **`LLAMAMAN_NODE_NAME`** (default `srv1`) - a unique, stable identity for this deployment that is **required for every install**. The default is fine for a single node; give each host a distinct value if you run more than one (see [Clustering](#clustering)). Pick it once and keep it - changing it later orphans this node's stored instances and presets.

**NVIDIA:**
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
docker compose up --build
```

For native VRAM monitoring, also uncomment the `deploy.resources.reservations` block in `docker-compose.yml`.

**AMD (ROCm):**
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-rocm
# Edit docker-compose.yml: set LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-rocm
docker compose up --build
```

**Intel Arc:**
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-sycl
# Edit docker-compose.yml: set LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-sycl
docker compose up --build
```

**CPU only:**
```bash
docker pull ghcr.io/ggml-org/llama.cpp:server
# Edit docker-compose.yml: set LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
docker compose up --build
```

- **Management UI**: http://localhost:5000
- **Llamaman proxy** (Ollama-compatible API): http://localhost:42069
- **llama-server public instance ports**: 8000-8020

On first launch, visit the UI to create an admin account via `/setup`.

> **Note:** llamaMan needs access to the Docker socket (`/var/run/docker.sock`) to spawn llama-server containers. This is already configured in `docker-compose.yml`. Be aware of the security implications - a container with Docker socket access has the ability to manage other containers on the host.

### Running bare-metal

llamaMan can also run directly on the host instead of in a container - useful for development or on hosts (e.g. WSL) where running the manager itself in Docker is awkward. It still spawns llama-server **containers** via the Docker socket, but talks to them over `localhost` on their published ports rather than the Docker network.

```bash
pip install -r requirements.txt

# Simplest (dev): starts the UI/API on :5000 and the proxy thread on :42069
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs python app.py

# Or via gunicorn (production config lives in gunicorn.conf.py; 1 worker)
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs gunicorn -c gunicorn.conf.py app:app
```

A single process serves both the management UI/API (port 5000) and the Ollama-compatible proxy (port 42069). The container/bare-metal mode is auto-detected; if detection is ever wrong for your runtime, set `LLAMAMAN_IN_DOCKER=true` or `false` explicitly. Bare-metal, `HOST_MODELS_DIR` / `HOST_LOGS_DIR` already resolve to real host paths, so they need no special handling.

## Authentication

llamaMan has a built-in auth system with two layers:

### User accounts (session-based)

On first launch, `/setup` lets you create an admin account. After that, all browser access requires login. Session cookies authenticate UI requests.

### API keys (bearer tokens)

Create API keys in the **API Keys** section of the UI. External clients (OpenWebUI, scripts, etc.) authenticate with:

```
Authorization: Bearer llm-xxxxxxxxxx
```

### Require authentication toggle

The **"Require authentication for all endpoints"** toggle (on by default) controls whether model-serving endpoints require a bearer token:

| Toggle | Model endpoints (`/api/chat`, `/v1/chat/completions`, etc.) | Management endpoints (`/api/instances`, etc.) | Per-instance proxy ports |
|--------|--------------------------------------------------------------|-----------------------------------------------|--------------------------|
| **ON** (default) | Bearer token required | Bearer token or session required | Bearer token required |
| **OFF** | Open (no auth) | Bearer token or session required | Open (no auth) |

When the toggle is **ON**, all three port surfaces are protected:
- **Port 5000** (management UI + API) - Flask `before_request` hook
- **Port 42069** (Ollama-compatible proxy) - same Flask app, same hook
- **Ports 8000-8020** (per-instance proxies) - WSGI-level auth check

### OpenWebUI with authentication

When `require_auth` is on, configure OpenWebUI to send a valid API key:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

## Models

Place models inside the `models/` volume:

- **GGUF files**: any `.gguf` file (recommended - llama.cpp native format)
- **HuggingFace repos**: directories containing `config.json`

Or use the **Download** button in the UI to pull from HuggingFace.

### Display Name

Each model can be given an optional **Display Name** in the Launch form (the narrow field on the `Model Path | Display Name | Note` row). When set, it's the id API clients see on `/api/tags` and `/v1/models` - so OpenWebUI shows `Qwen 2.5 14B` instead of `Qwen2.5-14B-Instruct-Q4_K_M` - and clients can send it as the model name too. It must be unique and must not clash with another model's filename or a cluster queue-group name. Leave it blank to use the filename. It's stored on the model's preset (alongside favorite/note); no migration needed.

### Model updates

Model authors often republish a repo under the same filenames (a requant, a fixed chat template, a rebuilt imatrix), so a model pulled months ago can be silently stale. The **Check for updates** button in the Launch form's tab bar (shown for models with a known source repo) compares the local file against the repo's published content hash - one HTTP request, no download:

- **Up to date** - the recorded hash matches the published file.
- **Update available** - the repo has republished this model. The button arms for a **re-pull**, which downloads into a staging folder nested inside the model's own directory and atomically swaps the file in only once the download completes, so a failed pull leaves the current model intact. It's an ordinary download record (global/per-model speed limits, HF tokens, progress, pause/resume/cancel, and failed-download auto-retry all apply) and refuses with **409** while an instance has the model loaded.
- **Verify hash** - shown when no hash has been recorded yet. Hashes the local file on disk (a background job with progress) instead of re-downloading gigabytes to compare, and records the result so later checks are exact and instant.

See [Download Settings](#download-settings) for the optional background scan that computes these hashes ahead of time.

## Launching Instances

1. Select a model from the sidebar
2. Configure launch settings (GPU layers, context size, idle timeout, etc.)
3. Click **Launch** - llamaMan spawns a llama-server container and the instance appears with a status badge
4. Optionally click **Save Preset** to remember settings for that model

Each instance exposes an OpenAI-compatible API on its assigned port.

### Layer autodetection

When you select a GGUF model, llamaMan reads the file's metadata to detect the total number of layers (block count). This is displayed next to the **GPU Layers** input so you can see exactly how many layers are available to offload (e.g. `/ 32`). Set GPU Layers to `-1` to offload all layers to GPU.

### Launch settings reference

| Setting | Default | Description |
|---|---|---|
| **GPU Layers** | `-1` | Number of layers to offload to GPU. `-1` = all layers, `0` = **CPU only**. Total layers are autodetected from the GGUF file. With `0`, no GPU is attached to the container at all (it's launched without any GPU device request), so it runs fully on CPU and its card shows no GPU - handy on hosts where GPU passthrough isn't available. |
| **Context Size** | `4096` | Maximum context window in tokens (`--ctx-size`). |
| **Parallel** | `1` | Number of parallel sequences the llama-server can process simultaneously (`--parallel`). Controls KV cache slot allocation inside the server itself. |
| **Idle Timeout min** | `0` | Minutes of inactivity before the server is stopped to free VRAM. `0` = disabled. See [Idle Timeout](#idle-timeout). |
| **Max Concurrent** | `0` | Maximum number of inference requests allowed in-flight at once. `0` = unlimited. When set, incoming requests are queued and gated by a semaphore. |
| **Max Queue Depth** | `200` | Maximum number of requests that can wait in the queue when `Max Concurrent` is active. Requests beyond this limit are rejected with HTTP 429. |
| **Share Queue** | off | When enabled, multiple proxy-managed instances of the **same model** share a single request queue. Incoming requests are distributed across instances as slots become available, providing simple load balancing. |
| **Embedding Model** | off | Marks the instance as an embedding model. Embedding instances are **excluded** from the `LLAMAMAN_MAX_MODELS` count and will never be evicted by the proxy's LRU policy. |
| **CPU Threads** | _(auto)_ | Sets both `--threads N` for llama-server and the container's CPU quota (`--cpus N`). Leave blank to let the container and llama-server use all available cores. |
| **Memory Limit** | _(none)_ | Hard memory cap for the llama-server container (e.g. `32g`, `8192m`). Equivalent to `deploy.resources.limits.memory` in Docker Compose. Leave blank for no limit. |
| **GPU Devices** | _(global default)_ | Comma-separated GPU indices to make visible to this container (e.g. `0,1`). Overrides `LLAMA_GPU_DEVICES` for this instance. Leave blank (or the literal `all`) to expose all GPUs. The instance card labels exactly the GPUs selected here. Not supported on Intel Arc. |
| **Split Mode** | `layer` | How llama.cpp distributes weights across the GPUs visible inside the container (`--split-mode`). `layer` splits whole transformer layers per GPU (llama.cpp's own default; low interconnect traffic, works well on plain PCIe). `row` splits tensor rows per GPU (activations cross the interconnect on every matmul, so it's typically slower than `layer` on plain PCIe and only beats it with fast interconnect like NVLink). `none` uses a single GPU only (the first visible) and ignores **Tensor Split**. Only distinguishes anything when 2+ GPUs are visible. |
| **Tensor Split** | _(auto)_ | Comma-separated relative weights (`--tensor-split`), one per container-visible GPU (e.g. `24,16` for a 24 GB + 16 GB pair). llama.cpp normalizes internally, so `24,16` == `3,2` == `0.6,0.4`. The number of values must match the container-visible GPU count. Leave blank and llamaMan auto-fills at launch time from each visible GPU's total VRAM (uses total, not free, so the value is stable across relaunches). Ignored when **Split Mode** is `none` or only one GPU is visible. |
| **Flash Attention** | `auto` | Enables the fused Flash Attention kernel (`--flash-attn [on\|off\|auto]`, default `auto`): faster prompt processing and lower activation memory during inference. **Auto** lets llama.cpp decide per backend/GPU (the flag is omitted, matching its own default); **On** forces it (fails to start if the backend can't do it); **Off** disables it. Independent of KV cache type in general, but a quantized **V Cache Type** requires this set to **On** — llama-server refuses to start with a quantized V cache and no flash-attn actually enabled, and Auto is not a guarantee (may resolve to off on some backends). |
| **K Cache Type** | `f16` | Precision for the K (keys) side of the KV cache (`--cache-type-k`). Accepts `f16` (default) / `f32` / `bf16` / `q8_0` / `q5_1` / `q5_0` / `iq4_nl` / `q4_1` / `q4_0`. Quantized types reduce the K portion of KV memory at a small quality cost — `q8_0` is near-lossless on most models, `q4_0` more aggressive. Works with or without Flash Attention. Only the flag is emitted when the value differs from `f16`, so leaving the default matches pre-feature behavior byte-for-byte. |
| **V Cache Type** | `f16` | Precision for the V (values) side of the KV cache (`--cache-type-v`). Same accepted values as **K Cache Type**. Any quantized value requires Flash Attention set to **On** — the form greys the dropdown whenever Flash Attention is Auto or Off (Auto is not enough; llama.cpp may resolve it to off on some backends) and auto-snaps a stale quantized V back to `f16` the moment Flash Attention leaves On, so the launch never submits a combination llama-server would reject. V is typically less sensitive to quantization than K. |
| **Extra Args** | _(empty)_ | Additional flags passed directly to llama-server (e.g. `--mlock`, or one of the n-gram speculative types not surfaced elsewhere). |
| **Speculative Decoding** | off | Runs the model with speculative decoding. Only the draft-model family is surfaced here; the n-gram / lookup family (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) doesn't take a drafter model and stays reachable via **Extra Args**. |
| **Draft Type** | `draft-mtp` | Which drafter format to use (`--spec-type`). `draft-simple`: classic speculative decoding with a plain smaller model (same tokenizer as the target). `draft-mtp`: Multi-Token Prediction — drafts from a separate MTP drafter, or from the main model's built-in MTP heads if **Draft Model** is blank. `draft-dflash`: Block-Diffusion Flash drafter (needs a DFlash checkpoint). `draft-dspark`: DSpark drafter (newer, same family as DFlash). `draft-eagle3`: EAGLE-3 drafter (needs an EAGLE-3-format checkpoint). |
| **Draft Model** | _(none)_ | Path to the drafter model (`-md`). **Required** for `draft-simple`, `draft-dflash`, `draft-dspark`, `draft-eagle3` — the drafter's format has to match the Draft Type. **Optional** for `draft-mtp`, where it points at a standalone MTP-head GGUF (e.g. a Gemma 4 assistant-MTP drafter) and blank falls back to the main model's built-in MTP heads. Pick one of the models on the target node or type a path under the models directory. |
| **Draft N Max** | `2` | Max tokens drafted per step (`--spec-draft-n-max`), used when speculative decoding is on. Leave blank to use llama.cpp's default. |
| **Draft N Min** *(Advanced)* | _(auto)_ | Minimum tokens drafted per step (`--spec-draft-n-min`). Blank omits the flag so llama-server uses its own default (which drifts across versions — deliberately not hard-coded). |
| **Draft P Split** *(Advanced)* | _(auto)_ | Probability threshold at which drafting splits a batch (`--spec-draft-p-split`). Range `0.0`–`1.0`. Blank omits the flag. |
| **Draft P Min** *(Advanced)* | _(auto)_ | Minimum probability for greedy acceptance of a drafted token (`--spec-draft-p-min`). Range `0.0`–`1.0`. Blank omits the flag. `0` is a real value distinct from blank and is passed through. |
| **Proxy Sampling Overrides** | off | When enabled, the proxy forces the configured sampling parameters on every request forwarded to this instance, regardless of what the client sends. |
| **Temperature** | `0.8` | Sampling temperature to enforce (range: `0.0`–`2.0`). Only active when proxy sampling overrides are enabled. |
| **Top K** | `40` | Top-k sampling value to enforce (min: `0`). Only active when proxy sampling overrides are enabled. |
| **Top P** | `0.95` | Top-p (nucleus) sampling value to enforce (range: `0.01`–`1.0`). Only active when proxy sampling overrides are enabled. |
| **Presence Penalty** | `0.0` | Presence penalty to enforce (range: `-2.0`–`2.0`). Only active when proxy sampling overrides are enabled. |
| **Repeat Penalty** | `0.0` | Repeat penalty to enforce (range: `0.0`–`2.0`). `0` = disabled (not injected). Only active when proxy sampling overrides are enabled. |

### Live preset updates

Saving a preset (**Save Preset** in the Launch tab) updates already-running instances of that model in place where possible, so most parameter tweaks don't require a relaunch:

- **Apply live (no relaunch needed):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, and all six proxy-sampling fields (`proxy_sampling_override_enabled`, `temperature`, `top_k`, `top_p`, `presence_penalty`, `repeat_penalty`). The reaper re-reads idle timeout each tick, the request gate is refreshed in place, and the proxy + compat routes read sampling fields from the instance config per request.
- **Require relaunch:** everything baked into the llama-server container at launch - GPU layers, context size, threads, memory limit, parallel slots, GPU devices, split mode, tensor split, flash attention, K cache type, V cache type, speculative-decoding fields (spec type, draft model, spec-draft n-max / n-min / p-split / p-min), embedding flag, extra args.

**Caveat for proxy-sampling toggles:** if the instance was launched with `idle_timeout = 0`, `max_concurrent = 0`, **and** `override_enabled = false`, no sidecar proxy was spawned (see [Per-Instance Proxy](#per-instance-proxy)). Toggling `override_enabled = true` live still applies overrides on requests routed through the main app's Ollama/OpenAI compat endpoints, but direct hits to the public port go straight to llama-server and bypass the override. Relaunch the instance to spawn the proxy in that case.

### Concurrency and queueing

When **Max Concurrent** is set to a value greater than 0, llamaMan places a concurrency gate in front of the instance. Requests that exceed the limit are held in a FIFO queue (up to **Max Queue Depth**). If the queue is also full, new requests are rejected with HTTP 429.

The gate tracks active and queued request counts, which are visible in the instance list via the API.

**Parallel vs Max Concurrent:** `Parallel` controls how many sequences the llama-server processes internally (KV cache slots). `Max Concurrent` is an external gate that limits how many requests llamaMan forwards to the server at once. You can use both together - for example, `Parallel=4` with `Max Concurrent=4` ensures the server always has enough KV slots for the requests it receives.

## GPU Stats

llamaMan queries GPU VRAM and utilization natively - no running llama-server instance required.

| Vendor | Method | Requirement |
|---|---|---|
| NVIDIA | `pynvml` (NVML library direct) | Uncomment the `deploy.resources.reservations` block in `docker-compose.yml` to grant the llamaman container NVIDIA toolkit `utility` capability |
| AMD | `/sys/class/drm` sysfs | `/sys/class/drm:ro` volume mount (included in `docker-compose.yml` by default) |
| Intel Arc | `/sys/class/drm` sysfs | Same mount as AMD |

When native access is not configured, llamaMan falls back to exec-ing `nvidia-smi` / `rocm-smi` inside a running llama-server container (previous behavior). Stats always reflect the full host GPU state, not just a single container's usage.

## Request Recording & Stats

### Request recording

Under **Settings >> App Settings >> Request recording**, choose how proxied inference traffic is logged:

| Mode | Behaviour |
|---|---|
| **Off** (default) | Nothing is recorded. |
| **Per request** | Each turn is stored as its own record. |
| **Per conversation** | Turns are grouped by a content hash of the system prompt + first user message, so a multi-turn chat lands in one file/row. |

Each record captures the request/response bodies plus envelope fields - model, endpoint, status, duration, prompt/completion token counts, and **accurate per-turn metrics**: generation throughput (tokens/s, measured over the generation window so it excludes prompt evaluation) and time-to-first-token. Records live under `request_log/` for the JSON backend (`RECORDINGS_DIR` to relocate) or the `request_log` table for MariaDB. A **Retention (days)** setting prunes older records hourly in the background (`0` = keep forever).

### Per-instance stats

Each instance card has a **Stats** button that opens a modal summarizing that instance's recorded traffic: request count (and errors), average and peak throughput, average time-to-first-token, average latency, prompt/completion/total tokens, and the active time span. Because the numbers are rolled up from the request log, the modal shows an empty state prompting you to enable recording when it's off, and stats persist even after the instance is stopped. Throughput and TTFT use the accurate per-turn metrics captured at generation time rather than re-derived end-to-end figures.

### Request log dashboard

The **Logging** link in the header opens a full-page view of the recorded request log: summary tiles (token totals, average/peak throughput, TTFT, latency, error and streamed counts), a recent-conversations list, and a per-conversation drill-down that shows prompts and responses first with the metrics tucked into a collapsible. A time-window selector (24h / 7d / 30d / All) scopes every figure. Like the per-instance stats, it reads from the request log, so enable **Request recording** for it to populate.

## Idle Timeout

Set **Idle Timeout min** in the launch form (0 = disabled). When enabled:

- The manager proxies the instance port (transparent to clients)
- After N minutes of no requests, the llama-server container is stopped to free VRAM
- On the next request, a new container is spawned with the same config
- Client sees the same port/API with just a cold-start delay

For instances managed by the llamaman proxy (OpenWebUI), use the `LLAMAMAN_IDLE_TIMEOUT` env var instead.

## Per-Instance Proxy

When any of the following are enabled for an instance, llamaMan inserts a WSGI proxy in front of the llama-server container on that port: **Idle Timeout**, **Max Concurrent**, or **Proxy Sampling Overrides**. The public port (e.g. 8000) is handled by the proxy; the llama-server container listens internally on a separate port.

### Model name validation

The proxy enforces that requests reach the correct model. On inference endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/completion`, `/chat/completions`):

- If the request body includes a `"model"` field, the proxy compares it against the loaded model's filename stem (lowercased, without extension). A **prefix match is accepted** - e.g. `"qwen2.5-0.5b-instruct-q2"` matches `"qwen2.5-0.5b-instruct-q2_k"`. A mismatch returns HTTP 404:
  ```json
  {"error": "model 'wrong-model' is not loaded on this port"}
  ```
- If the request body has **no `"model"` field**, the request is forwarded unconditionally.

This check applies whether the instance is currently running or sleeping. For sleeping instances, a mismatched model name prevents the wake - no container is spawned.

### Wake on request

When an instance with idle timeout is sleeping and a request arrives:

1. If the request carries a `"model"` field that does not match >> HTTP 404, no wake
2. If the model matches (or no model field) >> a new container is spawned, request is held until healthy, then forwarded

## Download Settings

The UI provides download-related options under **Settings >> Download Settings**:

- **Auto-retry failed downloads** - automatically retries downloads that fail due to network errors or interruptions. Off by default.
- **Retry count per failed download** - how many times to retry before marking a download as permanently failed (default: 3, min: 1). Only active when auto-retry is enabled.
- **Check models for updates in the background** - opt-in worker that periodically asks each model's source repo whether the file has been republished, and computes a checksum for any model that doesn't have one yet, so the per-model **Check for updates** answer is exact and instant when you click it. Off by default.
- **Update check interval (hours)** - how often the background scan runs (default: 24). A checksum is a full read of the model file, so the scan hashes at most one model per pass and never runs while a download is in progress.

## Docker Image Management

**Settings >> Docker Images** lets you manage the llama.cpp server images used to spawn containers:

- **Pull image by name** - type any image name (e.g. `ghcr.io/ggml-org/llama.cpp:server-cuda`) and pull it directly without it needing to be in the tracked list first
- **Delete local image** - each tracked image has a delete button that removes it from Docker and from the tracked list. Disabled for the active `LLAMA_IMAGE`. Returns an error if Docker refuses (e.g. image in use by a running container)
- **Auto-update** - optionally pull the active image on a configurable interval

## Model Backup and Restore

**Settings >> App Settings** provides export and restore for model metadata and presets:

- **Download Stored Models JSON** - exports all scanned models with their preset configs to a timestamped JSON file. Use this to back up your configuration or migrate to a new host.
- **Restore from JSON** - upload a previously exported JSON. For each model in the file:
  - Already present on disk: preset is merged in (existing values are not overwritten)
  - Not present but has a HuggingFace source: download is queued immediately and preset is pre-populated at the expected path so it is ready when the file lands
  - Not present and no known source: reported as unrestorable

## Cleanup Settings

The UI provides automatic cleanup under **Settings >> Cleanup Settings**:

- **Auto-clean completed/failed downloads** - removes download records older than a configurable number of hours (default: 24). Only affects completed, failed, or cancelled downloads - active downloads are never touched.
- **Auto-clean stopped instances** - removes stopped instance records older than a configurable number of hours (default: 24). Only affects stopped instances - running instances are never removed.
- **Auto-remove stale instance records** - periodically checks all `starting`/`healthy`/`sleeping` instance records against their actual Docker container. Records whose backing container is no longer running are marked stopped. Configurable check interval (default: 5 minutes).

Cleanup runs periodically in the background. These settings only remove or update records in the UI/state - they do not delete model files.

## OpenWebUI Integration (llamaman proxy)

The llamaman proxy exposes an Ollama-compatible API on port **42069** (configurable). Point OpenWebUI at it:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
```

**How it works:**

1. OpenWebUI calls `/api/tags` -> llamaMan returns all available GGUF models
2. User selects a model in OpenWebUI -> `/api/chat` request arrives
3. llamaMan spawns a llama-server container (using saved preset or defaults)
4. Waits for healthy, then proxies the request with format translation
5. When `LLAMAMAN_MAX_MODELS` limit is reached, the least-recently-used **Ollama-managed** model is evicted. Admin UI launched models are never evicted by the Ollama API by default (see [Model eviction policy](#model-eviction-policy))

Supported Ollama endpoints: `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`

Also supports OpenAI-compatible endpoints with auto-start: `/v1/models`, `/v1/chat/completions`

**Model names in the listing:** by default each model is listed under its GGUF filename stem. Give a model a **Display Name** (see [Display Name](#display-name)) and OpenWebUI shows and accepts that friendly name instead. In a cluster, live shared-queue [group aliases](#clustering) are also advertised as selectable models, so a client can pick the load-balanced alias directly.

### Model eviction policy

The `LLAMAMAN_MAX_MODELS` limit controls how many **chat** models the proxy will keep loaded simultaneously. When a new model is requested and the limit is reached, the least-recently-used (LRU) chat model is evicted to make room.

#### Priority rules

Admin UI launched models have ultimate priority. The two API surfaces have different eviction rights:

| Launcher | Eviction behaviour | Cannot evict |
|----------|--------------------|--------------|
| **Admin UI** | Evicts Ollama-managed models first (LRU), then admin UI models if needed | - |
| **Ollama API** (`/api/chat`, `/api/generate`) | Evicts Ollama-managed models (LRU) | Admin UI launched models (by default) |
| **OpenAI API** (`/v1/chat/completions`) | No eviction by default - starts model only if a slot is free | Everything (by default) |

If the cap is full, requests that cannot evict return HTTP 503:
```
model limit reached (LLAMAMAN_MAX_MODELS=N); admin-launched models cannot be evicted via the API
model limit reached (LLAMAMAN_MAX_MODELS=N); the OpenAI API does not evict running models
```

#### App Settings toggles

Three toggles in **Settings >> App Settings** control eviction behaviour:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - when on, the admin UI silently evicts the LRU model (Ollama-managed first) before launching. When off (default), the UI prompts you to confirm before exceeding the cap.
- **Allow Ollama API to evict admin-launched models** - when on, the Ollama API can also evict admin UI launched models as a fallback if no Ollama-managed models are available to evict. Off by default.
- **Allow OpenAI API to evict admin-launched models** - when on, the OpenAI API gains LRU eviction (Ollama-managed first, then admin UI launched models) to make room, just like the Ollama API with its override enabled. Off by default, in which case the OpenAI API never evicts and only loads a model when a slot is free (returning 503 otherwise).

#### Other details

- **All running instances count toward the limit** - both admin UI and proxy-managed instances. If you manually launch 2 models and `LLAMAMAN_MAX_MODELS=1`, the proxy sees you are already over the limit.
- **Embedding models are excluded.** Instances marked as **Embedding Model** do not count toward the limit and are never evicted. This lets you keep an embedding model loaded permanently alongside your chat models.
- **`LLAMAMAN_MAX_MODELS=0` (default) disables eviction entirely.** The proxy will launch models on demand without ever stopping existing ones.

## Storage Backends

### JSON (default)

Zero-config. Stores data in JSON files under `DATA_DIR` (`/data`):
- `state.json` - instances and downloads
- `presets.json` - per-model launch presets
- `users.json` - user accounts
- `settings.json` - global settings
- `api_keys.json` - API key hashes
- `request_log/` - per-conversation request log records (override location with `RECORDINGS_DIR`)

Instance and download logs are written to `LOGS_DIR` (`/tmp/llama-logs`), which is separate from persistent data.

When running with the MariaDB backend (`DATABASE_URL` set), request logs are stored in the `request_log` table instead and `RECORDINGS_DIR` has no effect.

### MariaDB / MySQL

Create the database and a dedicated user:

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
FLUSH PRIVILEGES;
```

Then set `DATABASE_URL` to enable:

```yaml
environment:
  - DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

Tables are auto-created on first connection. Requires `sqlalchemy` and `pymysql` (included in requirements).

Per-node model metadata (a file's source repo and its content hash) lives in a node-scoped `model_files` table keyed by `(node_id, model_path)`, so two nodes holding different files at the same path don't collide. On upgrade, each node copies its own entries out of the legacy shared settings blob - only paths that exist on its own disk - and the blob is kept as a read fallback and rollback path. Schema migrations are versioned **per node**, so every node runs the ones it needs on its own first boot.

> **InnoDB note:** the `model_files` primary key `(node_id, model_path)` is `(64 + 700) x 4 = 3056` bytes under utf8mb4, just under InnoDB's 3072-byte index limit, so the table is created with `ROW_FORMAT=DYNAMIC`. It fits on any modern MariaDB/MySQL; widening either column past that would fail at `CREATE TABLE`.

### Surviving a database outage (local mirror)

*Optional, off by default.*

With `DATABASE_URL` set, the database is on the critical path of **every** request - auth checks read settings and verify API keys on each call - so if it becomes unreachable the node stops serving, and a container restart during the outage cannot boot at all.

Turn on **Settings → App settings → "Keep a local mirror of the database"** (or set `LLAMAMAN_DB_MIRROR=1`) and the node keeps a write-through copy of the database in `DATA_DIR/db_mirror/`. Every successful write goes to both, and a full pull once a day picks up rows written by other cluster nodes.

If the database becomes unreachable, the node **keeps working from the mirror**:

| Still works | Blocked while offline |
|---|---|
| All inference (both ports) | Creating the first user (`/setup`) - a brand-new install has no mirror to fall back to anyway |
| Launching, stopping, sleeping, waking models | |
| Downloads, model hashing and update checks | |
| Creating and editing **presets** (including for models you just downloaded) | |
| Creating and revoking **API keys** | |
| Changing download and app **settings** | |
| Adding and removing **Hugging Face tokens** | |

Changes made offline are recorded in an append-only journal and replayed in order when the database returns - a probe checks every 10s. Replay records the *edit*, not the resulting row, so a change another node made during the outage survives: a different preset, a different field of the same preset, a different settings key, or a different Hugging Face token are all left intact.

**Requirements and caveats:**

- **`DATA_DIR` must be a persistent volume** (it already is in the sample compose file). The mirror is useless on a container's ephemeral layer.
- **Each node needs its own `DATA_DIR`.** The mirror directory is stamped with the owning node id; if two nodes share one, mirroring switches itself off rather than corrupting either view.
- **Secrets are mirrored to local disk.** Settings are copied verbatim, and Hugging Face tokens are stored in the settings blob in **plaintext**. Password hashes and API key hashes are mirrored too (already hashed). If you chose MariaDB partly to keep secrets off individual hosts, this trade-off is the reason the feature is opt-in.
- **Cross-node balancing stops during an outage.** Peer liveness lives in the database, so a degraded node sees only itself and serves locally. Peers likewise stop routing to it.
- **API key changes are local until reconnect.** A key created offline works only on this node; a key **revoked** offline stays valid on other nodes until the database is back. Plan revocations accordingly.
- **Request logs are dropped, not buffered,** while offline - the records carry full request and response bodies, and spooling them through a long outage could fill the volume and take down the inference this is meant to protect.
- **Turning the mirror off while the database is offline is deferred** until it comes back. Switching to direct mode mid-outage would mean every request hits the unreachable database instead of falling back - the setting is saved and applied on recovery.
- Schema migrations never run while degraded; they are applied against the real database on reconnect, before anything is replayed.

## Clustering

*Optional, off by default - single-node installs are completely unaffected.*

Clustering lets several llamaMan deployments act as **one logical cluster**: a single dashboard that aggregates every node's GPUs, instances, and downloads, with cross-node launches/pulls/downloads and multi-node shared-queue load balancing. Nodes discover each other automatically through the shared storage backend - no pairwise key exchange.

**Requirements:**

- **A shared storage backend.** Every node must point at the **same** `DATABASE_URL` (MariaDB/MySQL) - the database doubles as the node registry and coordination store. The JSON backend is per-host and cannot be shared.
- **A unique `LLAMAMAN_NODE_NAME` per node.** This is each node's identity in the cluster (and the partition key for its own rows). Required for every install, clustered or not.
- **The same `CLUSTER_SECRET` on every node.** It's the bearer token (sent as `X-Cluster-Secret`) for all node-to-node HTTP.
- **`CLUSTER_ADVERTISE_URL` per node** if you want cross-node *actions*. It's how peers reach this node - a hostname/IP routable from the **other** hosts (not `localhost`), e.g. `http://srv1:5000`. A node without one still appears in the shared dashboard but is view-only and is skipped as an inference target.

Set on **each** node (only `LLAMAMAN_NODE_NAME` and `CLUSTER_ADVERTISE_URL` differ between them):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical on all nodes
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical on all nodes
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Each node heartbeats every ~5s; a node silent past `CLUSTER_NODE_ONLINE_WINDOW_S` (default 45s) is shown offline. Inspect and manage the cluster under **Settings >> Cluster**.

**Per-node vs shared settings:** most settings are shared cluster-wide via the database, but a few are scoped per node because they're host-specific: the tracked **Docker images** (a CUDA host and a ROCm host differ) and the model-cap eviction toggles (**Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches**, **Allow Ollama API to evict admin-launched models**, and **Allow OpenAI API to evict admin-launched models**). Existing single-node values are inherited until a node overrides them.

**Discovering shared-queue aliases:** when instances across nodes share a queue group (the cross-node, load-balanced entry point), that group's alias is advertised as a selectable model in `/api/tags` and `/v1/models`, deduped cluster-wide, so a client can send the alias and have it routed to the least-loaded node serving it. Only live groups are surfaced - clustering on, the alias actually set, and at least one non-stopped member on any node - which is exactly when a request for the alias can be routed rather than 404'd.

> **Security:** the cluster secret lets any peer drive actions on this node. Run node-to-node traffic over a trusted network or behind TLS.

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | _(required)_ | **Required - the app refuses to start without it.** Unique, stable identity for this deployment: the partition key for its instances, downloads, and per-node settings in storage, and its key in the cluster registry. Any string (`srv1`, a hostname, a uuid). Pick once and keep it - changing it later orphans this node's stored state. |
| `MODELS_DIR` | `/models` | Directory scanned for model files (container path) |
| `DATA_DIR` | `/data` | Directory for persistent config/state (JSON files) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Directory for per-conversation request log records. JSON backend only - ignored when `DATABASE_URL` is set. |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs (container path) |
| `HOST_MODELS_DIR` | _(same as `MODELS_DIR`)_ | **Host-side** absolute path of the models volume - must match the left side of `-v /host/path/models:/models`. Passed to the Docker daemon when spawning sibling llama-server containers so they can bind-mount the same directory. |
| `HOST_LOGS_DIR` | _(same as `LOGS_DIR`)_ | **Host-side** absolute path of the logs volume. Same requirement as `HOST_MODELS_DIR`. |
| `PORT_RANGE_START` | `8000` | Start of public llama-server/proxy port pool |
| `PORT_RANGE_END` | `8020` | End of public llama-server/proxy port pool |
| `INTERNAL_PORT_RANGE_START` | `9000` | Start of internal port pool used when proxy mode is enabled |
| `INTERNAL_PORT_RANGE_END` | `9020` | End of internal port pool used when proxy mode is enabled |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent **chat** models via the proxy (LRU eviction, 0 = unlimited) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle timeout in minutes for proxy-managed instances (0 = disabled) |
| `SECRET_KEY` | _(auto)_ | Flask session secret. Auto-derived from machine-id if unset. Set this for multi-replica deployments. |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Name of the session cookie. Namespaced so llamaman coexists with other Flask apps on the same host - cookies are scoped by host+path, not port, so two apps both using Flask's default `session` name would log each other's users out. Change only if another `llamaman_session` on the same host clashes. |
| `DATABASE_URL` | _(unset)_ | MariaDB/MySQL connection string. Unset = use JSON files. |
| `LLAMAMAN_DB_MIRROR` | _(unset)_ | Force the local database mirror on (`1`) or off (`0`), overriding the per-node setting. Only meaningful with `DATABASE_URL`. See [Surviving a database outage](#surviving-a-database-outage-local-mirror). |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy during launch/relaunch. Increase for very large models. |
| `REQUEST_TIMEOUT` | `300` | **Read** timeout in seconds for upstream requests to llama-server, for cross-node inference forwarding, and for gate acquire waits. On the forwarding path it covers the peer loading the model on demand plus its time to first token. It does **not** govern how long a node waits for a peer to accept the connection - that is a separate 5s connect bound - so raising this will not help against an unreachable peer. |

### Docker / GPU

| Variable | Default | Description |
|---|---|---|
| `LLAMA_IMAGE` | _(auto)_ | llama.cpp Docker image used for all spawned containers. Auto-selected from the detected GPU vendor if not set (`server-cuda` / `server-rocm` / `server-sycl` / `server`). Set explicitly to pin a specific image or version. |
| `LLAMA_NETWORK` | `llamaman-net` | Docker network that llamaMan and all llama-server containers are attached to. Created automatically if it doesn't exist. |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Name prefix for spawned llama-server containers (e.g. `llamaman-abcd1234`). |
| `LLAMAMAN_IN_DOCKER` | _(auto-detect)_ | Whether llamaman itself runs in a container. Auto-detected from runtime marker files and cgroups. In Docker it reaches spawned containers by name on the Docker network; bare-metal it uses `localhost` on their published ports. Set `true`/`false` to override detection. |
| `LLAMA_HOST_ADDR` | `localhost` | Host address used to reach spawned containers' published ports when running bare-metal. Change only if those ports are published on a non-loopback address. |
| `GPU_TYPE` | _(auto-detect)_ | Override GPU vendor detection: `cuda` (NVIDIA), `rocm` (AMD), `intel` (Intel Arc). Leave unset to let llamaMan probe the host automatically. |
| `LLAMA_GPU_DEVICES` | _(unset = all)_ | Comma-separated GPU indices visible to all spawned llama-server containers, e.g. `0,1,3`. Unset exposes all GPUs. Per-instance **GPU Devices** overrides this when set. Not supported on Intel Arc. |

### Clustering

Optional - leave unset for single-node installs. See [Clustering](#clustering). (`LLAMAMAN_NODE_NAME`, listed under **Core**, is required for all installs and is also each node's cluster identity.)

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Set `true`/`1`/`yes`/`on` to join this node to a cluster. Requires `CLUSTER_SECRET`; ignored with a warning if the secret is empty. |
| `CLUSTER_SECRET` | _(unset)_ | Shared bearer secret sent on every node-to-node call (`X-Cluster-Secret`). Must be identical on every node. Use a long random value over a trusted network or behind TLS. |
| `CLUSTER_ADVERTISE_URL` | _(unset)_ | How peers reach **this** node's UI/API - a hostname/IP routable from the other hosts (e.g. `http://srv1:5000`), not `localhost`. Needed for cross-node actions and shared-queue inference forwarding; a node without it is view-only in the dashboard and skipped as an inference target. |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Seconds since a node's last heartbeat before it's shown offline. Raise it if nodes flap offline under load or clock skew (e.g. an unsynced WSL host). |

## REST API

All endpoints return and accept JSON.

**Authentication:** Management endpoints require either a session cookie (from browser login) or an `Authorization: Bearer <key>` header. When `require_auth` is enabled (default), model-serving endpoints also require a bearer token.

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/login` | Login page |
| `POST` | `/login` | Authenticate (`username`, `password` form data) |
| `GET` | `/setup` | First-run setup page |
| `POST` | `/setup` | Create first user account |
| `GET` | `/logout` | End session |

### API Keys

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/api-keys` | List all API keys (hashes stripped) |
| `POST` | `/api/api-keys` | Create a new API key (`{"name": "..."}`) |
| `DELETE` | `/api/api-keys/<id>` | Revoke an API key |

### Instances

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/instances` | List all instances |
| `POST` | `/api/instances` | Launch a new instance |
| `GET` | `/api/instances/<id>` | Get a single instance |
| `DELETE` | `/api/instances/<id>` | Stop and remove an instance |
| `POST` | `/api/instances/<id>/restart` | Restart a stopped/sleeping instance |
| `DELETE` | `/api/instances/<id>/remove` | Remove a stopped instance from the list |
| `GET` | `/api/instances/<id>/logs` | Last N log lines |
| `GET` | `/api/instances/<id>/logs/stream` | SSE live log tail |
| `GET` | `/api/next-port` | Get next available port from the pool |

**Launch body** (`POST /api/instances`):
```json
{
  "model_path": "/models/my-model.gguf",
  "port": 8000,
  "n_gpu_layers": -1,
  "ctx_size": 4096,
  "threads": null,
  "memory_limit": null,
  "parallel": null,
  "extra_args": "--mlock",
  "gpu_devices": "",
  "split_mode": "layer",
  "tensor_split": "",
  "flash_attn": "auto",
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "spec_enabled": false,
  "spec_type": "draft-mtp",
  "spec_draft_model": "",
  "spec_draft_n_max": null,
  "spec_draft_n_min": null,
  "spec_draft_p_split": null,
  "spec_draft_p_min": null,
  "idle_timeout_min": 0,
  "max_concurrent": 0,
  "max_queue_depth": 200,
  "share_queue": false,
  "proxy_sampling_override_enabled": false,
  "proxy_sampling_temperature": 0.8,
  "proxy_sampling_top_k": 40,
  "proxy_sampling_top_p": 0.95,
  "proxy_sampling_presence_penalty": 0.0,
  "proxy_sampling_repeat_penalty": 0.0
}
```

`gpu_devices`: comma-separated GPU indices for this instance (e.g. `"0"`, `"0,1"`). Leave empty to use `LLAMA_GPU_DEVICES` (or all GPUs if that is also unset). Not supported on Intel Arc.

`split_mode`: one of `"none"` / `"layer"` / `"row"`, mapped 1:1 to llama-server `--split-mode`. Empty (or omitted) is treated as `"layer"` at emit time, matching llama.cpp's own default. See the launch settings table above for the semantics.

`tensor_split`: comma-separated relative weights (`--tensor-split`), one per container-visible GPU (e.g. `"24,16"`). Values are normalized by llama.cpp. Number of values must match the container-visible GPU count. Empty means llamaMan will auto-fill it at launch from total VRAM when `split_mode` is `"layer"` or `"row"` and 2+ GPUs are visible; otherwise the flag is omitted and llama.cpp uses an even split (or ignores it entirely when `split_mode` is `"none"`).

`flash_attn`: one of `"on"` / `"off"` / `"auto"` (default; matches llama.cpp's own default for `--flash-attn`). `"on"` and `"off"` emit `--flash-attn on|off`; `"auto"` omits the flag entirely so llama.cpp decides per backend. Legacy `true` / `false` values from configs and presets saved before the tri-state rollout are folded on read (`true` → `"on"`, `false` → `"off"`), so no storage migration is needed. Must be `"on"` if `cache_type_v` is a quantized value (`q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`) — otherwise llama-server refuses to start with *"quantized V cache was requested, but this requires Flash Attention"*. `"auto"` is **not** a guarantee (llama.cpp may resolve it to off on some backends). We deliberately don't second-guess this on the server side, so an API caller sees the real llama-server error rather than a silently dropped flag.

`cache_type_k` / `cache_type_v`: precision for the K / V side of the KV cache (`--cache-type-k` / `--cache-type-v`). Accepts `"f16"` (default, emits no flag) / `"f32"` / `"bf16"` / `"q8_0"` / `"q5_1"` / `"q5_0"` / `"iq4_nl"` / `"q4_1"` / `"q4_0"`. Values outside the whitelist are dropped rather than shipped. See `cache_type_v` note above about the Flash Attention constraint.

`spec_enabled`: boolean gate for all `spec_*` fields; when `false`, none of them reach llama-server.

`spec_type`: one of `"draft-simple"` / `"draft-mtp"` (default) / `"draft-dflash"` / `"draft-dspark"` / `"draft-eagle3"`. Any other value is rejected with `400`. The n-gram / lookup speculative family (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) is not accepted here — those don't take a drafter model; pass them via `extra_args`.

`spec_draft_model`: path to the drafter (`-md`). Required when `spec_enabled` is `true` and `spec_type` is anything other than `draft-mtp` — a save/launch with a missing drafter for `draft-simple` / `draft-dflash` / `draft-dspark` / `draft-eagle3` is rejected with `400`. Optional for `draft-mtp`, where blank falls back to the main model's built-in MTP heads.

`spec_draft_n_max` / `spec_draft_n_min` / `spec_draft_p_split` / `spec_draft_p_min`: optional numeric knobs mapped to `--spec-draft-n-max` / `--spec-draft-n-min` / `--spec-draft-p-split` / `--spec-draft-p-min`. `null` (or omitted) means llamaMan skips the flag entirely so llama-server uses its own default. Integers ≥ 0 for the `n-*` pair; floats in `[0.0, 1.0]` for the `p-*` pair; out-of-range or non-numeric values are rejected with `400`. `0` is a real value distinct from `null` and is passed through.

`memory_limit`: Docker memory cap string, e.g. `"32g"` or `"8192m"`. Omit or `null` for no limit.

`threads`: when set, applies `--threads N` to llama-server **and** sets the container CPU quota to N cores.

### Downloads

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/downloads` | List all downloads |
| `POST` | `/api/downloads` | Start a new download |
| `GET` | `/api/downloads/<id>` | Get a single download |
| `DELETE` | `/api/downloads/<id>` | Cancel an active download |
| `DELETE` | `/api/downloads/<id>/remove` | Remove a completed/failed entry |
| `GET` | `/api/downloads/<id>/logs` | Download log output |
| `GET` | `/api/downloads/<id>/logs/stream` | SSE live log tail |

**Download body** (`POST /api/downloads`):
```json
{
  "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
  "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
  "hf_token": "hf_...",
  "speed_limit_mbps": 0
}
```

Leave `filename` blank to download the full repository.

### Models

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List discovered models in `MODELS_DIR` (includes `repo_id` when source is known) |
| `POST` | `/api/models/delete` | Delete a model from disk (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Read layer count from GGUF metadata |
| `GET` | `/api/disk-space` | Free/used space on the models volume |

### Presets

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/presets` | List all saved presets |
| `GET` | `/api/presets/<model_path>` | Get preset for a model |
| `PUT` | `/api/presets/<model_path>` | Save/update a preset |
| `DELETE` | `/api/presets/<model_path>` | Delete a preset |

### Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/settings` | Get global settings |
| `POST` | `/api/settings` | Save global settings |

**Settings body** (`POST /api/settings`):
```json
{
  "require_auth": true,
  "admin_ui_enforce_max_models": false,
  "allow_ollama_api_override_admin": false,
  "auto_retry_failed_downloads": false,
  "retry_count_per_failed_download": 3,
  "cleanup": {
    "downloads_enabled": true,
    "downloads_max_age_hours": 24,
    "downloads_last_run_at": 1710000000,
    "instances_enabled": false,
    "instances_max_age_hours": 48,
    "instances_last_run_at": 1710000000,
    "stale_records_enabled": false,
    "stale_records_interval_min": 5,
    "stale_records_last_run_at": null
  }
}
```

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system-info` | CPU usage, core count, RAM usage |
| `GET` | `/api/gpu-info` | Per-GPU VRAM and utilization (native query; falls back to container exec if native access is not configured) |
| `GET` | `/health` | Health check (`{"status": "ok"}`) - always open, no auth required |

### Request Log

Available when request recording is enabled (see [Request Recording & Stats](#request-recording--stats)).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/request-log/conversations` | Recent conversations with rolled-up metadata (`limit` query, default 100, max 500) |
| `GET` | `/api/request-log/conversations/<id>` | All recorded turns for one conversation, oldest first |
| `GET` | `/api/request-log/stats` | Aggregate metrics (token totals, avg/peak tokens/s, avg TTFT, latency, error/streamed counts). Optional `inst_id` to scope to one instance and `window_hours` to limit the time range. Also returns the current `recording` mode. |

### Ollama-compatible (llamaman)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/tags` | List available models (Ollama format) |
| `GET` | `/api/version` | Version info |
| `POST` | `/api/show` | Model metadata |
| `GET` | `/api/ps` | Running models |
| `POST` | `/api/chat` | Chat completion (auto-starts model) |
| `POST` | `/api/generate` | Text generation (auto-starts model) |
| `GET` | `/v1/models` | List models (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI format, auto-starts model) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Instance stuck on **starting** | Check logs via the Logs button. Common causes: OOM, model path typo, corrupt GGUF, image not pulled. |
| _"Docker image not found"_ | Pull the matching image: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (NVIDIA), `server-rocm` (AMD), `server-sycl` (Intel Arc), or `server` (CPU). |
| _"Docker API error"_ on launch | Ensure `/var/run/docker.sock` is mounted into the llamaMan container (it is by default in `docker-compose.yml`). |
| No GPU / CUDA error | Ensure the NVIDIA Container Toolkit is installed and `docker run --gpus all` works on the host. |
| No GPU / ROCm error | Ensure `/dev/kfd` and `/dev/dri` exist on the host and your user is in the `video`/`render` groups. |
| No GPU / Intel Arc error | Ensure `/dev/dri` is accessible and your user is in the `video`/`render` groups. |
| GPU stats show unavailable | For NVIDIA: uncomment the `deploy.resources.reservations` block in `docker-compose.yml`. For AMD/Intel: ensure `/sys/class/drm:ro` is mounted (default in `docker-compose.yml`). |
| Wrong GPU vendor detected | Set `GPU_TYPE=cuda`, `GPU_TYPE=rocm`, or `GPU_TYPE=intel` in the environment to override auto-detection. |
| Instance stuck on **starting** when running bare-metal | The container is healthy but llamaman can't reach it. Deployment mode is auto-detected, but if it's wrong for your runtime, set `LLAMAMAN_IN_DOCKER=false` (bare-metal) or `true` (in Docker) explicitly. |
| Stats modal is empty | Per-instance stats are rolled up from the request log. Enable **Settings >> App Settings >> Request recording** (per request or per conversation). |
| Launch fails with GPU/CDI error on a host without GPU passthrough | Set **GPU Layers** to `0` to launch CPU-only with no GPU device attached, or fix the GPU runtime (e.g. install the NVIDIA Container Toolkit). |
| Port conflict | The form auto-suggests an unused port; adjust if needed. |
| Model not showing in OpenWebUI | Ensure `OLLAMA_BASE_URL` points to `http://llamaman:42069`. Check `/api/tags` returns models. |
| OpenWebUI gets 401 errors | `require_auth` is on (default). Create an API key in the UI and set `OPENAI_API_KEYS` in OpenWebUI's environment. |
| _"API key required"_ on all requests | Either create an API key, or turn off the "Require authentication" toggle in the API Keys section. |
| Containers not cleaned up after stop | llamaMan stops and removes containers when instances are stopped. If containers are orphaned after a crash, run `docker ps --filter name=llamaman-` to find and remove them manually, or restart llamaMan (orphan adoption runs on startup). |
| Client (Hermes / OpenWebUI / etc.) reports the trained context window instead of the preset cap | Upgrade to 1.1.2+. `/api/ps` now includes a `context_length` field set to the runtime ctx the instance was launched with, and `/api/show`'s `model_info["<arch>.context_length"]` is overridden with the *effective* cap (running instance > preset > GGUF default). Clients reading either will see the preset value (e.g. 64K) instead of the GGUF's trained max (e.g. 256K). |

## Credits

This work would not be possible without the work of [ggml-org/llama.cpp](https://github.com/ggerganov/llama.cpp)

## License

llamaMan is licensed under the [Elastic License 2.0](LICENSE). You may use, copy, distribute, and modify the software, subject to the following limitations:

- You may not provide the software to third parties as a hosted or managed service where the service gives users access to a substantial set of its features or functionality.
- You may not remove or obscure any licensing, copyright, or other notices of the licensor.

### Third-party licenses

llamaMan bundles the following third-party assets, each under their own license:

- **[Font Awesome Free 7.1.0](https://fontawesome.com/)** by Fonticons, Inc. - icons (CC BY 4.0), fonts (SIL OFL 1.1), and code (MIT). The full license text ships in [`static/fontawesome-free-7.1.0-web/LICENSE.txt`](static/fontawesome-free-7.1.0-web/LICENSE.txt).
