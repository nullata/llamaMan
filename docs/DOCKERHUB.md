# LlamaMan

![LlamaMan](https://raw.githubusercontent.com/nullata/llamaman/main/docs/llamaman-1.0.0.jpg)

A browser-based UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances from inside a Docker container. Includes an Ollama-compatible API proxy so it works as a drop-in replacement for Ollama with [Open WebUI](https://github.com/open-webui/open-webui).

## Features

- **Universal GPU support** - single image for NVIDIA, AMD (ROCm), Intel Arc, and CPU. The GPU vendor and matching `LLAMA_IMAGE` are auto-detected at startup; `GPU_TYPE` / `LLAMA_IMAGE` override if needed.
- **Model library** - scans `/models` for GGUF files, shows quant type and file size
- **One-click launch** - configure GPU layers, context size, threads, multi-GPU, speculative decoding, extra args
- **Speculative decoding (MTP)** - optional `--spec-type draft-mtp` toggle with a configurable draft length, for models with MTP heads
- **Preset configs** - save/load per-model launch settings, with live updates to running instances where possible
- **Download manager** - pull models from HuggingFace with speed throttling and auto-retry on failure
- **Model backup and restore** - export model metadata and presets to JSON, restore on any instance with downloads queued automatically for missing models
- **Instance management** - stop, restart, remove, view live-streamed logs
- **GPU VRAM indicator** - per-GPU VRAM and utilization, queried natively (no running instance required)
- **Container resource monitoring** - live CPU%, core quota, RAM usage with thin progress bars, and GPU assignment per running instance card
- **Per-instance stats** - a Stats button on each instance card surfaces throughput (tokens/s), time-to-first-token, latency, and token totals rolled up from the request log
- **Request log dashboard** - a dedicated Logging page with summary tiles, a conversations list, and per-conversation drill-down over the recorded request log, filterable by time window
- **Request recording** - optionally record proxied requests/responses per request or per conversation, with configurable retention
- **Idle timeout** - auto-sleep instances after configurable idle period, wake on next request
- **Ollama-compatible proxy** - OpenWebUI discovers models and auto-starts servers on demand
- **Authentication** - user accounts with session login, API key management with bearer tokens
- **Require auth toggle** - enforce bearer token authentication on all endpoints (including model loading) or leave model endpoints open
- **Persistent state** - instance history and configs survive container restarts
- **Storage backends** - JSON files (default) or MariaDB/MySQL via SQLAlchemy
- **Multi-node clustering** *(optional)* - run several instances as one cluster sharing a database and a secret: aggregated dashboard, cross-node launches/pulls/downloads, and multi-node shared-queue load balancing. Off by default; single-node installs are unaffected.
- **Proxy sampling overrides** - force temperature, top-k, top-p, presence penalty, and repeat penalty on all proxied requests, configurable per model preset
- **CPU quota + memory limit** - CPU Threads also applies a Docker CPU quota; a Memory Limit field caps container RAM
- **Docker image management** - pull any llama.cpp image by name, delete old local images from the UI

## Tags

- `latest`, `<version>` - Universal image, auto-detects GPU vendor (NVIDIA / AMD / Intel Arc / CPU)

## Quick Start

Pull the llama.cpp image for your GPU first, then run LlamaMan.

`HOST_MODELS_DIR` and `HOST_LOGS_DIR` must be the **absolute paths on the Docker host** that match your volume mounts. LlamaMan passes these to the Docker daemon when spawning sibling llama-server containers.

### NVIDIA

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  -e LLAMAMAN_NODE_NAME=srv1 \
  --restart unless-stopped \
  nullata/llamaman:latest
```

For native GPU monitoring (pynvml), add `--gpus` with utility capability:
```bash
  --gpus '"driver=nvidia,capabilities=utility"' \
```

### AMD (ROCm)

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-rocm

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-rocm \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  -e LLAMAMAN_NODE_NAME=srv1 \
  --restart unless-stopped \
  nullata/llamaman:latest
```

### Intel Arc

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-sycl

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-sycl \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  -e LLAMAMAN_NODE_NAME=srv1 \
  --restart unless-stopped \
  nullata/llamaman:latest
```

### Docker Compose

```yaml
services:
  llamaman:
    image: nullata/llamaman:latest
    ports:
      - "5000:5000"
      - "42069:42069"
      - "8000-8020:9000-9020"
    volumes:
      - /path/to/models:/models
      - /path/to/data:/data
      - /path/to/logs:/tmp/llama-logs
      - /var/run/docker.sock:/var/run/docker.sock
      - /sys/class/drm:/sys/class/drm:ro
    environment:
      - LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
      # Required - unique, stable identity for this deployment. Any string; pick
      # once and keep it. The container refuses to start without it.
      - LLAMAMAN_NODE_NAME=srv1
      # Must be the absolute host-side paths matching the volume mounts above.
      - HOST_MODELS_DIR=/path/to/models
      - HOST_LOGS_DIR=/path/to/logs
    # NVIDIA native GPU monitoring (pynvml) - uncomment on NVIDIA hosts.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           capabilities: [utility]
    networks:
      - llamaman-net
    restart: unless-stopped

networks:
  llamaman-net:
    driver: bridge
    name: llamaman-net
```

## Ports

| Port | Description |
|---|---|
| `5000` | Management UI and REST API |
| `42069` | Ollama-compatible API proxy |
| `8000-8020` | Individual llama-server instances |

## Volumes

| Path | Description |
|---|---|
| `/models` | GGUF model files. Place your models here or use the built-in download manager. |
| `/data` | Persistent state: instance configs, presets, user accounts, settings, API keys, and recorded request logs. |
| `/tmp/llama-logs` | Instance and download logs. Optional - mount to preserve logs across restarts. |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(required)* | **Required - the container refuses to start without it.** Unique, stable identity for this deployment: the partition key for its instances, downloads, and per-node settings in storage, and its key in the cluster registry. Any string (`srv1`, a hostname, a uuid). Pick once and keep it - changing it later orphans this node's stored state. |
| `LLAMA_IMAGE` | *(auto)* | llama.cpp server image for spawned containers. Auto-selected from detected GPU vendor if not set. Set explicitly to pin a version or backend (`server-cuda`, `server-rocm`, `server-sycl`, `server`). |
| `GPU_TYPE` | *(auto-detect)* | Override GPU vendor detection: `cuda`, `rocm`, or `intel`. Leave unset to auto-detect. |
| `LLAMA_GPU_DEVICES` | *(all)* | Comma-separated GPU indices visible to spawned containers, e.g. `0,1`. Not supported on Intel Arc. |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent **chat** models via the proxy. Uses LRU eviction when the limit is reached. `0` = unlimited. |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle timeout in minutes for proxy-managed instances. Stopped instances auto-restart on next request. `0` = disabled. |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy. |
| `MODELS_DIR` | `/models` | Directory scanned for model files (container path). |
| `DATA_DIR` | `/data` | Directory for persistent config/state. |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Directory for recorded request-log records (JSON backend only; ignored when `DATABASE_URL` is set). |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs (container path). |
| `HOST_MODELS_DIR` | *(same as `MODELS_DIR`)* | **Host-side** absolute path of the models volume. Must match the left side of `-v /host/path/models:/models`. LlamaMan passes this to the Docker daemon when spawning sibling containers. |
| `HOST_LOGS_DIR` | *(same as `LOGS_DIR`)* | **Host-side** absolute path of the logs volume. Same requirement as `HOST_MODELS_DIR`. |
| `PORT_RANGE_START` | `8000` | Start of public llama-server/proxy port pool. |
| `PORT_RANGE_END` | `8020` | End of public llama-server/proxy port pool. |
| `INTERNAL_PORT_RANGE_START` | `9000` | Start of internal llama-server port pool used for proxied instances. |
| `INTERNAL_PORT_RANGE_END` | `9020` | End of internal llama-server port pool used for proxied instances. |
| `SECRET_KEY` | *(auto)* | Flask session secret. Auto-derived from machine-id if unset. |
| `DATABASE_URL` | *(unset)* | MariaDB/MySQL connection string (e.g. `mysql+pymysql://user:pass@host/db`). Unset = JSON file storage. |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks. |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy during launch/relaunch. Increase for very large models. |
| `REQUEST_TIMEOUT` | `300` | Timeout in seconds for upstream requests to llama-server and gate acquire waits. |
| `CLUSTER_ENABLED` | `false` | Set `true`/`1`/`yes`/`on` to join this node to a cluster. Requires `CLUSTER_SECRET` and a shared `DATABASE_URL`. See [Clustering](#clustering). |
| `CLUSTER_SECRET` | *(unset)* | Shared bearer secret sent on every node-to-node call (`X-Cluster-Secret`). Must be identical on every node. Use a long random value over a trusted network or behind TLS. |
| `CLUSTER_ADVERTISE_URL` | *(unset)* | How peers reach **this** node's UI/API - a hostname/IP routable from the other hosts (e.g. `http://srv1:5000`), not `localhost`. Needed for cross-node actions and shared-queue inference forwarding; a node without it is view-only and skipped as an inference target. |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Seconds since a node's last heartbeat before it's shown offline. Raise it if nodes flap offline under load or clock skew (e.g. an unsynced WSL host). |

## First Launch

1. Start the container
2. Open **http://localhost:5000** in your browser
3. Create an admin account on the `/setup` page
4. Place GGUF model files in the `models/` volume, or download from HuggingFace via the UI

## Cleanup Settings

The UI provides automatic cleanup under **Settings >> Cleanup Settings**:

- **Auto-clean completed/failed downloads** - removes download records older than a configurable number of hours (default: 24). Only affects completed, failed, or cancelled downloads - active downloads are never touched.
- **Auto-clean stopped instances** - removes stopped instance records older than a configurable number of hours (default: 24). Only affects stopped instances - running instances are never removed.
- **Auto-remove stale instance records** - periodically checks all `starting`/`healthy`/`sleeping` instance records against their backing Docker container. Records whose container is no longer running are marked stopped. Configurable check interval (default: 5 minutes).

Cleanup runs periodically in the background. These settings only remove or update records in the UI/state - they do not delete model files.

## Request Recording & Stats

Under **Settings >> App Settings >> Request recording**, choose how proxied inference traffic is logged: **Off** (default), **Per request**, or **Per conversation** (turns grouped by a hash of the system prompt + first user message). Each record captures the request/response bodies plus envelope fields and accurate per-turn metrics - generation throughput (tokens/s, measured over the generation window) and time-to-first-token. Records are stored under `RECORDINGS_DIR` (inside `/data`) for the JSON backend or the `request_log` table for MariaDB; a **Retention (days)** setting prunes older records hourly (`0` = keep forever).

Each instance card then exposes a **Stats** button that opens a modal summarizing that instance's recorded traffic - request count (with errors), average and peak throughput, average time-to-first-token, average latency, prompt/completion/total tokens, and the active time span. Stats are rolled up from the request log, so they persist after an instance is stopped and the modal prompts you to enable recording when it's off.

The **Logging** link in the header opens a full-page dashboard over the same request log - summary tiles, a recent-conversations list, and a per-conversation drill-down (prompts and responses with metrics in a collapsible), all scoped by a 24h / 7d / 30d / All time-window selector.

## OpenWebUI Integration

Point OpenWebUI at the Ollama-compatible proxy:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
```

LlamaMan auto-launches models on demand:

1. OpenWebUI calls `/api/tags` and gets the available models.
2. A request to `/api/chat` or `/api/generate` starts the selected model automatically using saved presets or defaults.
3. When `LLAMAMAN_MAX_MODELS` is reached, the proxy evicts the least-recently-used **Ollama-managed** chat model first.

Supported Ollama endpoints: `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`

Also supports OpenAI-compatible auto-start endpoints: `/v1/models`, `/v1/chat/completions`

### With authentication enabled (default)

Create an API key in the LlamaMan UI, then configure OpenWebUI:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

### Model eviction policy

The `LLAMAMAN_MAX_MODELS` limit controls how many **chat** models the proxy keeps loaded simultaneously.

| Launcher | Eviction behavior | Cannot evict |
|---|---|---|
| **Admin UI** | Evicts Ollama-managed models first (LRU), then admin-launched models if needed | - |
| **Ollama API** (`/api/chat`, `/api/generate`) | Evicts Ollama-managed models (LRU) | Admin-launched models by default |
| **OpenAI API** (`/v1/chat/completions`) | Does not evict; only starts a model if a slot is free | Everything |

Two settings under **Settings >> App Settings** control this behavior:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - when on, the admin UI evicts the LRU model before launching. When off (default), the UI prompts before exceeding the cap.
- **Allow Ollama API to evict admin-launched models** - when on, the Ollama API may evict admin-launched models as a fallback. Off by default. This does not affect the OpenAI API, which never evicts.

Other details:

- All running chat instances count toward the limit, including admin-launched and proxy-managed instances.
- Embedding models are excluded from the limit and are never evicted.
- `LLAMAMAN_MAX_MODELS=0` disables eviction entirely.

## MariaDB / MySQL Setup

By default LlamaMan uses JSON files. To use MariaDB/MySQL, create a database and dedicated user:

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
FLUSH PRIVILEGES;
```

Then set `DATABASE_URL` in your container environment:

```
DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

Tables are auto-created on first connection.

## Clustering

*Optional, off by default - single-node installs are completely unaffected.*

Clustering lets several LlamaMan deployments act as **one logical cluster**: a single dashboard that aggregates every node's GPUs, instances, and downloads, with cross-node launches/pulls/downloads and multi-node shared-queue load balancing. Nodes discover each other automatically through the shared storage backend.

**Requirements:**

- **A shared storage backend** - every node must point at the **same** `DATABASE_URL` (MariaDB/MySQL). The JSON backend is per-host and cannot be shared.
- **A unique `LLAMAMAN_NODE_NAME` per node** - each node's identity in the cluster (required for every install regardless).
- **The same `CLUSTER_SECRET` on every node** - the bearer token for all node-to-node HTTP.
- **`CLUSTER_ADVERTISE_URL` per node** for cross-node *actions* - how peers reach this node (a hostname/IP routable from the other hosts, e.g. `http://srv1:5000`). A node without one appears in the dashboard but is view-only and skipped as an inference target.

Set on **each** node (only `LLAMAMAN_NODE_NAME` and `CLUSTER_ADVERTISE_URL` differ between them):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical on all nodes
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical on all nodes
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Each node heartbeats every ~5s; a node silent past `CLUSTER_NODE_ONLINE_WINDOW_S` (default 45s) is shown offline. Inspect and manage the cluster under **Settings >> Cluster**. A few settings are scoped per node because they're host-specific (tracked Docker images and the two model-cap eviction toggles); everything else is shared cluster-wide via the database.

> **Security:** the cluster secret lets any peer drive actions on this node. Run node-to-node traffic over a trusted network or behind TLS.

## Per-Instance Proxy

When **Idle Timeout**, **Max Concurrent**, or **Proxy Sampling Overrides** are enabled for an instance, LlamaMan places a proxy in front of that instance's port. The proxy handles auth, concurrency gating, wake-on-request, and model name validation.

Saving a preset propagates idle-timeout, queue, and proxy-sampling fields to running instances live without a relaunch. If the instance was launched with all three of the above off, no proxy was spawned, so toggling **Proxy Sampling Overrides** on live applies only to requests routed through the main app's Ollama/OpenAI compat endpoints; direct hits to the public port require a relaunch to take effect.

On inference endpoints, if the request body includes a `"model"` field, the proxy validates it against the loaded model's filename stem. A prefix match is accepted (e.g. `"qwen2.5-0.5b-instruct-q2"` matches `"qwen2.5-0.5b-instruct-q2_k"`). A mismatch returns HTTP 404. Requests without a `"model"` field are forwarded unconditionally.

For sleeping instances, a mismatched model name returns 404 without waking the instance.

## Requirements

- Docker with access to `/var/run/docker.sock`
- GPU support (one of):
  - **NVIDIA**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (`docker run --gpus all` must work)
  - **AMD**: [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  - **Intel Arc**: `/dev/dri` accessible, user in `video`/`render` groups
  - **CPU only**: no GPU required

## Links

- **Source**: [GitHub](https://github.com/nullata/llamaman)

## License

LlamaMan is licensed under the [Elastic License 2.0](https://github.com/nullata/llamaman/blob/main/LICENSE). You may use, copy, distribute, and modify the software, subject to the following limitations:

- You may not provide the software to third parties as a hosted or managed service where the service gives users access to a substantial set of its features or functionality.
- You may not remove or obscure any licensing, copyright, or other notices of the licensor.
