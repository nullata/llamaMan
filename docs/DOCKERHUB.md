<p align="center">
  <img src="https://raw.githubusercontent.com/nullata/llamaman/main/docs/llamaman-logo-wide.jpg" alt="llamaMan" width="1000">
</p>

# <img src="https://raw.githubusercontent.com/nullata/llamaMan/4d17202d108547537c0dbc13083794274b491fd3/static/images/logo.svg" alt="logo" width="24"> llamaMan

Browser UI for launching and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances from a single Docker container. Ships an Ollama-compatible API proxy so it drops in as an Ollama replacement for [Open WebUI](https://github.com/open-webui/open-webui).

Full docs and source: **[github.com/nullata/llamaman](https://github.com/nullata/llamaman)**

## Features

- **Universal GPU image** - one image, auto-detects NVIDIA / AMD (ROCm) / Intel Arc / CPU
- **Model library + downloader** - scans `/models` for GGUF, pulls from HuggingFace with speed limits, resume, auto-retry, and republish detection with atomic re-pull
- **One-click launch + presets** - per-model launch settings with live updates for the fields that don't need a relaunch (idle-timeout, gates, sampling overrides)
- **Instance management** - stop / restart / logs / stats; per-GPU VRAM, container CPU% + RAM, and per-instance throughput / TTFT / latency rolled up from the request log
- **Ollama + OpenAI proxy on `:42069`** - Open WebUI drops in, auto-starts models on demand, LRU-evicts once `LLAMAMAN_MAX_MODELS` is hit
- **Flash Attention + KV cache quant + reasoning format** - `--flash-attn`, `--cache-type-k/v`, `--reasoning-format` all exposed; UI enforces the quantized-V-requires-FA-On constraint
- **Anti-Loop** - DRY sampler (soft, sampling-time) + proxy-side output loop detection that watches the streamed text and hard-kills the turn when a large chunk repeats often enough. Both off by default, tuned per preset
- **Speculative decoding** - all five draft-model families (`draft-simple/-mtp/-dflash/-dspark/-eagle3`) with advanced knobs (`n-min`, `p-split`, `p-min`)
- **Image & PDF input** - `--mmproj` for vision models, PDFs rasterized page-by-page (or inlined as text via the born-digital shortcut). Works on OpenAI `image_url`/`file` and Ollama `images[]`
- **Auth** - user accounts, API keys (bearer tokens), toggle for whether model endpoints require auth
- **Persistent state, JSON or MariaDB** - MariaDB unlocks multi-node clustering (shared dashboard, cross-node launch/download, shared-queue load balancing) and an optional local write-through mirror that keeps the node serving through a DB outage
- **Request recording + logging dashboard** - opt-in per-request or per-conversation recording with retention, plus a dedicated logging page

## Tags

- `latest`, `<version>` - universal image, auto-detects GPU vendor

## Quick Start

Pull the llama.cpp server image for your GPU, then run llamaMan. `HOST_MODELS_DIR` / `HOST_LOGS_DIR` MUST be the absolute paths on the Docker host - llamaMan passes them to the daemon when spawning sibling llama-server containers.

| GPU | `LLAMA_IMAGE` |
|---|---|
| NVIDIA | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| AMD (ROCm) | `ghcr.io/ggml-org/llama.cpp:server-rocm` |
| Intel Arc | `ghcr.io/ggml-org/llama.cpp:server-sycl` |
| CPU only | `ghcr.io/ggml-org/llama.cpp:server` |

```bash
docker pull <LLAMA_IMAGE>              # from the table above
docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 -p 42069:42069 -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=<LLAMA_IMAGE> \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  -e LLAMAMAN_NODE_NAME=srv1 \
  --restart unless-stopped \
  nullata/llamaman:latest
```

`LLAMAMAN_NODE_NAME` is **required** - the container refuses to start without it. Pick it once and keep it (changing later orphans stored state). See the full env-var list below.

For native NVIDIA VRAM monitoring (pynvml), also add:
```bash
  --gpus '"driver=nvidia,capabilities=utility"' \
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
      - LLAMAMAN_NODE_NAME=srv1
      - HOST_MODELS_DIR=/path/to/models
      - HOST_LOGS_DIR=/path/to/logs
    # NVIDIA native GPU monitoring - uncomment on NVIDIA hosts.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           capabilities: [utility]
    networks: [llamaman-net]
    restart: unless-stopped

networks:
  llamaman-net:
    driver: bridge
    name: llamaman-net
```

## First Launch

1. Open <http://localhost:5000>
2. Create an admin account on `/setup`
3. Drop GGUF files into `/models` (or download via the UI)

## Ports & Volumes

| Port | Purpose |
|---|---|
| `5000` | Management UI + REST API |
| `42069` | Ollama-compatible API proxy |
| `8000-8020` | Individual llama-server instances |

| Path | Purpose |
|---|---|
| `/models` | GGUF model files |
| `/data` | Persistent state: instances, presets, users, API keys, settings, request logs |
| `/tmp/llama-logs` | Instance and download logs (optional; mount to preserve across restarts) |

## Environment Variables

Only the ones you'll typically touch. See the [full reference on GitHub](https://github.com/nullata/llamaman#environment-variables) for every knob.

| Variable | Default | Description |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(required)* | **Required.** Unique stable identity for this deployment - partition key for its stored state and its cluster identity. Pick once, keep it. |
| `LLAMA_IMAGE` | *(auto)* | llama.cpp server image for spawned containers. Auto-picked from detected GPU vendor; set to pin a version / backend. |
| `HOST_MODELS_DIR` | *(same as `MODELS_DIR`)* | **Host-side** absolute path of the models volume. Must match the left side of `-v /host/path:/models`. |
| `HOST_LOGS_DIR` | *(same as `LOGS_DIR`)* | Same requirement as `HOST_MODELS_DIR`. |
| `GPU_TYPE` | *(auto)* | Override GPU vendor detection: `cuda` / `rocm` / `intel`. |
| `LLAMA_GPU_DEVICES` | *(all)* | Comma-separated GPU indices visible to spawned containers (e.g. `0,1`). Not supported on Intel Arc. |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent chat models via the proxy (LRU eviction). `0` = unlimited. |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle-timeout minutes for proxy-managed instances (auto-restarts on next request). `0` = disabled. |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy. |
| `DATABASE_URL` | *(unset)* | MariaDB/MySQL connection string (`mysql+pymysql://user:pass@host/db`). Unset = JSON files. Required for clustering. |
| `LLAMAMAN_DB_MIRROR` | *(unset)* | Force local DB mirror on (`1`) / off (`0`), overriding the per-node setting. Only meaningful with `DATABASE_URL`. |
| `CLUSTER_ENABLED` | `false` | Set truthy to join this node to a cluster. Requires `CLUSTER_SECRET` + shared `DATABASE_URL`. |
| `CLUSTER_SECRET` | *(unset)* | Shared bearer token for node-to-node calls (`X-Cluster-Secret`). Identical on every node. |
| `CLUSTER_ADVERTISE_URL` | *(unset)* | How peers reach this node's UI/API - hostname/IP routable from other hosts (e.g. `http://srv1:5000`). |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy. Raise for very large models. |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Max concurrent PDF rasterizations process-wide (each raster spawns poppler + transient RAM). |

## OpenWebUI Integration

Point OpenWebUI at the Ollama proxy (with an API key when `require_auth` is on, which it is by default):

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

Models are listed by GGUF filename stem; set a per-model **Display Name** to have OpenWebUI show/accept a friendly name instead. In a cluster, live shared-queue group aliases are advertised as selectable models too, so a client can send the alias and have it routed to the least-loaded node.

## Persistent State

**JSON (default):** zero-config, stored in `/data`. Fine for single-node.

**MariaDB / MySQL:** set `DATABASE_URL`. Tables auto-created on first connect.

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
```

MariaDB unlocks two optional features:

- **Local DB mirror** - write-through copy in `DATA_DIR/db_mirror/` keeps the node serving through a DB outage (inference, launches, presets, API keys, settings; only first-user `/setup` is blocked). Offline writes are journalled and replayed on reconnect. Off by default; needs a persistent `DATA_DIR` per node. **Hugging Face tokens land on local disk in plaintext** if mirroring is on.
- **Multi-node clustering** - multiple llamaMan deployments share the DB and act as one logical cluster: aggregated dashboard, cross-node launch/pull/download, shared-queue load balancing. Requires `CLUSTER_ENABLED=true`, the same `CLUSTER_SECRET` on every node, and a per-node `CLUSTER_ADVERTISE_URL` for cross-node actions (a node without one is view-only).

## Per-Instance Proxy

When **Idle Timeout**, **Max Concurrent**, or **Proxy Sampling Overrides** are on for an instance, llamaMan puts a proxy in front of that instance's port. It handles auth, concurrency gating, wake-on-request, and model-name validation - requests with a `"model"` field are checked against the loaded model's filename stem (prefix match accepted; mismatch → 404, no wake for sleeping instances).

## Model Eviction

`LLAMAMAN_MAX_MODELS` caps concurrent **chat** models via the proxy. Embedding-flagged instances are excluded and never evicted; sleeping instances still count.

| Launcher | Evicts | Cannot evict |
|---|---|---|
| **Admin UI** | Ollama-managed first, then admin-launched | - |
| **Ollama API** | Ollama-managed (LRU) | Admin-launched (by default) |
| **OpenAI API** | Nothing (by default; 503 when full) | Everything (by default) |

Three toggles under **Settings → App Settings** relax these defaults ("Enforce cap for admin UI launches", "Allow Ollama API to evict admin-launched", "Allow OpenAI API to evict admin-launched").

## Requirements

- Docker with access to `/var/run/docker.sock`
- One of: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), a [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), Intel Arc with `/dev/dri` access, or CPU-only

## Links

- **Source & full docs**: [github.com/nullata/llamaman](https://github.com/nullata/llamaman)
- **llama.cpp server flags**: [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

## License

[Elastic License 2.0](https://github.com/nullata/llamaman/blob/main/LICENSE). No hosting as a managed service; no removing license notices.
