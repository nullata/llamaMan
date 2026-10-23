# Troubleshooting

The full symptom → fix table. The [README](../../README.md#troubleshooting) keeps the nine most common first-run rows; everything else is here.

| Symptom | Fix |
|---|---|
| Instance stuck on **starting** | Check logs via the Logs button. Common: OOM, model path typo, corrupt GGUF, image not pulled |
| _"Docker image not found"_ | Pull the matching image: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (or `server-rocm` / `server-sycl` / `server`) |
| _"Docker API error"_ on launch | Ensure `/var/run/docker.sock` is mounted (default in `docker-compose.yml`) |
| No GPU / CUDA error | Ensure NVIDIA Container Toolkit is installed and `docker run --gpus all` works on the host |
| No GPU / ROCm error | Ensure `/dev/kfd` and `/dev/dri` exist and your user is in `video`/`render` |
| No GPU / Intel Arc error | Ensure `/dev/dri` is accessible and your user is in `video`/`render` |
| Vulkan image fails with _"Unable to find group render"_ | Set `GPU_TYPE=vulkan` so llamaMan attaches the host render/video GIDs numerically (name-based group_add fails on images whose /etc/group doesn't ship a `render` entry) |
| GPU stats unavailable | NVIDIA: uncomment the `deploy.resources.reservations` block. AMD/Intel: ensure `/sys/class/drm:ro` is mounted |
| Wrong GPU vendor detected | Set `GPU_TYPE=cuda`/`rocm`/`intel` to override |
| Instance stuck on **starting** running bare-metal | The container is healthy but llamaman can't reach it. Set `LLAMAMAN_IN_DOCKER=false`/`true` explicitly if auto-detection is wrong for your runtime |
| Strata: _"Docker image 'strata:latest' not found"_ | Pull or build a Strata image in Settings → Docker Images (or `docker build -t strata .` in a clone of the Strata repo). See [Strata](STRATA.md) |
| Strata models not listed | Set `STRATA_ENABLED=true`; the node must be NVIDIA (`GPU_TYPE=cuda` if detection fails). `GET /api/engines` shows the reason |
| Strata instance stuck on **starting** | Normal for minutes on a first start (download + pack). The card shows the stage; the log shows Strata's setup. "setup failed" means its setup stopped - read the log's `[X]` line |
| Stats modal is empty | Enable **Settings → App Settings → Request recording** (see [Request Recording & Stats](CONFIGURATION.md#request-recording--stats)) |
| Launch fails with GPU/CDI error on a host without GPU passthrough | Set **GPU Layers** to `0` for CPU-only with no GPU device attached, or fix the GPU runtime |
| Port conflict | The form auto-suggests an unused port; adjust if needed |
| Model not showing in OpenWebUI | Ensure `OLLAMA_BASE_URL=http://llamaman:42069`. Check `/api/tags` returns models |
| OpenWebUI gets 401 | `require_auth` is on (default). Create an API key and set `OPENAI_API_KEYS` in OpenWebUI |
| OpenAI client gets 503 when models are full | `LLAMAMAN_MAX_MODELS` cap reached and the OpenAI API can't evict by default. Raise the cap or enable the eviction toggles: [Model Eviction](MODELS.md#model-eviction) |
| Containers not cleaned up after stop | llamaMan removes containers on stop. Orphans after a crash: `docker ps --filter name=llamaman-`, or restart llamaMan (orphan adoption runs on startup) |
| Client (Hermes / OpenWebUI / etc.) reports the trained context window instead of the preset cap | Upgrade to 1.1.2+. `/api/ps` now includes `context_length` set to the runtime ctx the instance was launched with, and `/api/show`'s `model_info["<arch>.context_length"]` is overridden with the effective cap |
| Client asking about a queue-group name gets its own default ctx (e.g. 128k) instead of the real deployment ctx | `/api/show` and `/v1/models` now answer for a queue group on any clustered node — even one with no local member — using the peer's runtime ctx from shared state. Multi-member groups report the min across live members so the advertised value is safe for any dispatch target. See [Clustering](CLUSTERING.md) |
