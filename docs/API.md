# REST API

All endpoints return / accept JSON. Management endpoints need a session cookie (browser login) or `Authorization: Bearer <key>`. Model-serving endpoints also require a bearer when `require_auth` is on (default). Auth details: [Authentication](CONFIGURATION.md#authentication-detail).

## Contents

- [Auth & keys](#auth--keys) · [Instances](#instances) · [Downloads, Models, Presets](#downloads-models-presets) · [Settings, System, Request log](#settings-system-request-log) · [Ollama & OpenAI compat](#ollama--openai-compat)

## Auth & keys

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/login` | Login page / authenticate |
| `GET` / `POST` | `/setup` | First-run setup |
| `GET` | `/logout` | End session |
| `GET` / `POST` | `/api/api-keys` | List / create API keys |
| `DELETE` | `/api/api-keys/<id>` | Revoke |

## Instances

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/instances` | List / launch |
| `GET` / `DELETE` | `/api/instances/<id>` | Get / stop and remove |
| `POST` | `/api/instances/<id>/restart` | Restart a stopped/sleeping instance |
| `DELETE` | `/api/instances/<id>/remove` | Remove a stopped-instance record |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Tail / SSE stream logs |
| `GET` | `/api/next-port` | Get next available port from the pool |
| `POST` | `/api/preview-command` | Render the equivalent `llama-server ...` command for a launch body (used by the UI's "Copy llama.cpp command" tool; does not launch anything) |

**Launch body** (`POST /api/instances`): the launch form's fields, serialized as JSON. See the [Launch settings reference](LAUNCH-SETTINGS.md#launch-settings-reference) for every field and its default; enum-valued fields (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) accept only the values listed there and default to the reference's default. Numeric fields left as `null` or omitted use the reference default; unknown fields are ignored.

Boundary quirks worth pinning:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp estimates VRAM), `-2` = **all** (force every layer), `0` = CPU only (no GPU attached to the container), positive N = exactly N layers.
- `n_cpu_moe_layers`: `0` (default) omits both MoE flags. `-1` emits `--cpu-moe` (all routed experts on CPU). Positive N emits `--n-cpu-moe N` (experts of the first N layers on CPU). Inert on dense models.
- `threads`: when set, applies `--threads N` to llama-server **and** sets the container CPU quota to N cores.
- `threads_batch`: null / omitted → the flag is not emitted and llama-server falls back to `--threads`.
- `flash_attn`: legacy `true` / `false` values from pre-tri-state configs are folded on read (`true` → `"on"`, `false` → `"off"`), so no storage migration is needed.
- `cache_type_v` quantized values require `flash_attn: "on"` - `"auto"` is not a guarantee; llama-server refuses to start otherwise.
- `spec_draft_model` is required when `spec_enabled: true` and `spec_type` is anything other than `"draft-mtp"` - a request without it is rejected 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / omitted omits the flag. `0` is a real value distinct from null and is passed through.

## Downloads, Models, Presets

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | List / start |
| `GET` / `DELETE` | `/api/downloads/<id>` | Get / cancel |
| `DELETE` | `/api/downloads/<id>/remove` | Remove a completed/failed entry |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Tail / SSE stream |
| `GET` | `/api/models` | Discovered models (includes `repo_id` when known; archived models flagged `archived` with `restore_path`) |
| `POST` | `/api/models/delete` | Delete from disk (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Read layer count from GGUF metadata |
| `GET` | `/api/disk-space` | Free/used space on the models volume |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Get / save / delete a preset (`GET /api/presets` lists all) |
| `GET` | `/api/engines` | This node's inference engines: availability (with reason), capabilities, launch fields, Strata's model catalogue |
| `GET` | `/api/archive` | Archive status (`enabled`, `available`, `reason`, `free_bytes`) and move jobs |
| `POST` | `/api/archive` | Archive a model: `{"path": "/models/..."}` (moves its whole folder). 409 when in use |
| `POST` | `/api/archive/restore` | Restore an archived model: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Cancel a queued/running move (source kept), or remove a finished one |
| `POST` | `/api/engines/strata/download` | Pre-download a Strata model's shards (`{"model": "strata/qwen-IQ2_XS"}`, optional `hf_token_id` / `speed_limit_mbps`) |

Download body: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Blank `filename` pulls the full repo.

Instances and presets take an optional `engine` (`llamacpp` default, or `strata`; inferred from a `/strata/...` model path) plus, for Strata, `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) and `strata_low_ram` (`auto`/`on`/`off`). Strata model paths are virtual: `/strata/<family>-<SIZE>`, e.g. `PUT /api/presets/strata/qwen-IQ2_XS`. See [Strata](STRATA.md).

## Settings, System, Request log

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Get / save global settings |
| `GET` | `/api/system-info` | CPU usage, core count, RAM usage |
| `GET` | `/api/gpu-info` | Per-GPU VRAM and utilization (native; falls back to container exec) |
| `GET` | `/health` | `{"status": "ok"}` - always open, no auth required |
| `GET` | `/api/request-log/conversations` | Recent conversations with rolled-up metadata (`limit` query, default 100, max 500) |
| `GET` | `/api/request-log/conversations/<id>` | All recorded turns for one conversation, oldest first |
| `GET` | `/api/request-log/stats` | Aggregate metrics (tokens, avg/peak tokens/s, TTFT, latency, errors, streamed counts). Optional `inst_id` + `window_hours` |

## Ollama & OpenAI compat

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate auto-start).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat auto-starts).
Decision models: `/v1/systemone` (auto-starts) - see [Decision Models](MODELS.md#decision-models).
Knowledge base: `/api/kb/*` (admin-only) and the MCP endpoint `/mcp/knowledge` - see [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Cluster node-to-node endpoints use `X-Cluster-Secret` - see [Clustering](CLUSTERING.md).
