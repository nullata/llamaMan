# Configuration & storage

Bare-metal installs, the auth model in detail, GPU stats, request recording, storage backends, and the full environment-variable reference.

## Contents

- [Running bare-metal](#running-bare-metal) · [Authentication detail](#authentication-detail) · [GPU Stats](#gpu-stats) · [Request Recording & Stats](#request-recording--stats) · [Storage Backends](#storage-backends) · [Surviving a database outage (local mirror)](#surviving-a-database-outage-local-mirror) · [Environment Variables](#environment-variables)

## Running bare-metal

Useful for dev or hosts (e.g. WSL) where running the manager itself in Docker is awkward. llama-server still runs in containers.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

A single process under gunicorn serves the UI/API on `:5000` and the proxy on `:42069` (`python app.py` without gunicorn runs the Flask dev server on `:5000` only). Container-vs-bare-metal is auto-detected; if detection is ever wrong for your runtime, set `LLAMAMAN_IN_DOCKER=true` or `false` explicitly.

## Authentication detail

Two credential types:

- **Session cookies** for the UI (created via `/setup` on first launch)
- **API keys** for external clients (`Authorization: Bearer llm-xxxxxxxxxx`), managed under **API Keys** in the UI

The **"Require authentication for all endpoints"** toggle (on by default) controls whether model-serving endpoints demand a bearer token:

| Toggle | Model endpoints (`/api/chat`, `/v1/chat/completions`, per-instance ports) | Management endpoints (`/api/instances`, etc.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (default) | Bearer token required | Bearer token or session |
| **OFF** | Open | Bearer token or session |

All three port surfaces (5000, 42069, 8000-8020) go through the same auth hook.

## GPU Stats

Per-GPU VRAM and utilization, queried natively - no running llama-server required.

| Vendor | Method | Requirement |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML direct) | Uncomment `deploy.resources.reservations` in `docker-compose.yml` for NVIDIA toolkit `utility` capability |
| AMD | `/sys/class/drm` sysfs | `/sys/class/drm:ro` volume mount (included by default) |
| Intel Arc | `/sys/class/drm` sysfs | Same mount as AMD |

Without native access, llamaMan falls back to exec-ing `nvidia-smi` / `rocm-smi` inside a running container. Stats always reflect full host GPU state.

## Request Recording & Stats

Under **Settings → App Settings → Request recording**: **Off** (default) / **Per request** / **Per conversation** (turns grouped by a content hash of the system prompt + first user message). Each record captures the bodies + envelope fields (model, endpoint, status, duration, token counts) plus **accurate per-turn metrics** (throughput measured over the generation window; TTFT). Records live under `RECORDINGS_DIR` (JSON) or the `request_log` table (MariaDB). **Retention (days)** prunes hourly (`0` = keep forever).

Each instance card exposes a **Stats** button (request count, avg/peak throughput, avg TTFT/latency, token totals, active time span) rolled up from the request log - so it persists after the instance is stopped and shows an empty state when recording is off. The **Logging** link in the header opens a full-page dashboard (summary tiles, recent conversations, per-conversation drill-down) with a 24h / 7d / 30d / All time-window selector plus per-model and per-day filters over the conversation list.

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

MariaDB is also the prerequisite for [Clustering](CLUSTERING.md) and the [Knowledge Base](KNOWLEDGE-BASE.md).

## Surviving a database outage (local mirror)

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
| `ARCHIVE_DIR` | *(unset)* | Container path where an archive volume is mounted (e.g. `/archive`). Setting it enables [Model Archive](MODELS.md#model-archive); unset hides it |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Public llama-server/proxy port pool |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Internal port pool used when proxy mode is enabled |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent chat models via the proxy (LRU eviction; 0 = unlimited). Rules: [Model Eviction](MODELS.md#model-eviction) |
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
| `GPU_TYPE` | *(auto)* | Override GPU vendor detection: `cuda`, `rocm`, `intel`, or `vulkan` (`vulkan` is opt-in only; picks `/dev/dri` + host render/video GIDs and the `server-vulkan` image) |
| `LLAMA_GPU_DEVICES` | *(all)* | Comma-separated GPU indices visible to all spawned containers (e.g. `0,1,3`). Per-instance **GPU Devices** overrides. Not supported on Intel Arc |

### Strata

Optional; see [Strata](STRATA.md).

| Variable | Default | Description |
|---|---|---|
| `STRATA_ENABLED` | `false` | Offer Strata models on this node. Also needs an NVIDIA GPU (`GPU_TYPE=cuda` or detected) |
| `STRATA_IMAGE` | `strata:latest` | Preferred Strata image for launches. When it isn't on the node, the built image, then a pulled one, is used |
| `STRATA_BUILD_IMAGE` | `strata:latest` | Tag *Pull repo & build latest* builds |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | Named Docker volume mounted at `/data` in every Strata container (model files, packs, MTP layer, per-model setup; 70-120 GB per model). Created by Docker on first use |
| `HOST_STRATA_DATA_DIR` | *(unset)* | Host path to mount at `/data` instead of the named volume. Wins over `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | Seconds a request (or relaunch) waits for a Strata instance to become ready. High because a first start downloads and prepares the model before the port opens |
| `STRATA_REPO` | `Niko1221/Strata` | GitHub repository (owner/name) *Pull repo & build latest* downloads |
| `STRATA_REPO_REF` | `main` | Branch, tag or commit of `STRATA_REPO` to build |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | Where the downloaded repository is kept (the build context). Its commit is recorded next to it in `Strata.source.json` |
| `STRATA_WEB_HOSTS` | *(unset)* | Extra host names / IPs (comma-separated) you open a Strata instance's own web app by (`http://<host>:<instance port>/`). Without an API key Strata refuses chat requests from pages on other names; the host of `CLUSTER_ADVERTISE_URL` is always allowed. Applies to instances started afterwards |

### Clustering

Optional. `LLAMAMAN_NODE_NAME` (under Core) doubles as the cluster identity and is required for all installs.

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Set truthy to join this node to a cluster. Requires `CLUSTER_SECRET`; ignored with a warning if secret is empty |
| `CLUSTER_SECRET` | *(unset)* | Shared bearer token (`X-Cluster-Secret`) for all node-to-node HTTP. Identical on every node. Use a long random value over a trusted network or behind TLS |
| `CLUSTER_ADVERTISE_URL` | *(unset)* | How peers reach **this** node's UI/API - hostname/IP routable from the other hosts (not `localhost`). Without it the node is view-only |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Seconds since last heartbeat before a node shows offline. Raise if nodes flap under load or clock skew |
| `LLAMAMAN_PERF_LOG` | *(off)* | Diagnostics. Set truthy to log per-phase timings (`perf <name> <ms>`) for suspected slow paths: remote instance stop, cluster proxy forwards, reachability probes, heartbeat snapshot, auth checks, `save_state`, container stats. Off by default; negligible overhead when off |
