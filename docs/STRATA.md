# Strata engine - design notes and verification

[Strata](https://github.com/Niko1221/Strata) is a second inference engine in llamaMan, next to llama.cpp. This page
covers how the integration works, what it relies on upstream, its limits, and how to verify it on a real NVIDIA
host. For setup and day-to-day use, see the README's "Strata Engine" section.

Upstream references point at Strata commit `99f3dbd` (2026-10-03).

## How it fits

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- An instance's engine is `config["engine"]`. A missing value means llama.cpp, so every existing instance row,
  preset and container label keeps meaning what it did. llama.cpp configs are never stamped with the key, because it
  would change the container's `llamaman.config` label. `tests/fixtures/llamacpp_container_spec.json` holds the
  pre-refactor `containers.run` kwargs for 9 configs × 5 GPU vendors, and `tests/test_engines.py` asserts that they
  are unchanged.
- No storage migration is needed. Presets and instances are JSON blobs in both backends (`presets.json` /
  `state.json`, and the MariaDB `data` TEXT columns), so `engine` and `strata_*` are additive keys.
- Each Strata model is a virtual entry, `/strata/<family>-<SIZE>` (id `strata/<family>-<SIZE>`), and is its own
  instance and container. Switching models, eviction, sleep and wake work as they do for GGUF files.
- `_run_container` takes the engine's `container_spec()` and adds the GPU devices the engine asks for. A Strata
  container gets:

  | | |
  |---|---|
  | image | `STRATA_IMAGE` |
  | command | none (the entrypoint reads env vars only) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | data volume → `/data`; logs; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<tag>` (rw), when llamaMan holds complete shards |
  | mem_limit | from System Memory Limit (forces `LOW_RAM=on`) |
  | network / ports / labels | as for llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

## Answers to the open questions

1. **Readiness.** `/health` answers `{"status": "ok", ..., "service": "strata"}` before the API-key gate
   (`serve/server.py:2172`). The HTTP port only opens after the model has loaded: `serve()` runs at
   `server.py:3008`, after the engine starts at `:2925`. While Strata is downloading or loading, connections are
   refused, so llamaMan's existing health check (`status == "ok"`) needs no special case. `/v1/models` returns an empty
   list while the model is unloaded in lazy mode (`server.py:2190`), so it is not a good readiness signal.
   `/v1/status` needs the API key when one is set.
2. **Settings.** Everything is passed as an env var read by `docker-entrypoint.sh`: `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Command arguments are ignored,
   because the entrypoint rebuilds `$@` (`:42`, `:65`). GPU pinning uses container numbering: the NVIDIA runtime
   renumbers the host GPUs llamaMan attaches, so host `1,3` becomes `GPUS=0,1`. llamaMan doesn't set `API_KEY`, which
   matches how it runs llama-server (no key); its own proxy and auth sit in front.
3. **Reusing GGUFs on disk.** `setup.py` has `--gguf-dir` (`setup.py:2954`), but the entrypoint never forwards it.
   Setup does, however, accept whole shards that are already at `/data/models/<tag>/<original name>`: it checks each
   against its own tensor directory and marks it done (`setup.py:3389-3405`, "#173"). llamaMan therefore bind-mounts
   its copy there (read-write, for the `.done` marks), and only once its own download of those shards has completed.
   Shards are fetched from the Hugging Face revisions Strata pins (`setup.py:65-70`, mirrored in
   `core/engines/strata.py`). The MTP layer (~5 GB) and the pack are still prepared inside the container.
4. **Memory limit / memlock.** Setup reads RAM from `/proc/meminfo`, which is the host's total, so a capped container
   must ask for `LOW_RAM=on` (Dockerfile comment at `:31`, `docs/INSTALL.md:116-118`). Without it, setup picks the
   in-RAM mode and the container is likely to be OOM-killed. llamaMan forces `LOW_RAM=on` whenever a memory limit is
   set, and the UI warns below 64 GB. `memlock=-1` matters because the engine page-locks part of its arena for GPU DMA,
   and experts it cannot pin go through slower host copies (`src/prefill/prefill.cpp:110`, `:207-214`). From the
   source, a missing memlock limit looks like a slowdown rather than a failure, but **I couldn't verify that without a
   GPU**. llamaMan always sets it.

Also from the source: Strata serves one request at a time (`server.py:1323`). The Dockerfile sets `STRATA_EXECV=1`,
so the server is PID 1 and `docker stop` reaches it. A container stopped mid-download resumes from its `.part` files
on the next start.

## Design decisions

- **Concurrency is forced to 1** at launch, on preset merge, on live preset edits and in saved presets. The gate
  forces the sidecar proxy, so queueing happens in llamaMan (`RequestGate`), not inside Strata.
- **`REINSTALL` by fingerprint.** Context, vision, KV, Low-RAM and GPU pinning are recorded in Strata's per-model
  config on the volume. `DATA_DIR/strata_setups.json` remembers the settings each model last became healthy with,
  keyed by data volume and setup tag, and `REINSTALL=1` is sent only when they differ. An unknown fingerprint (first
  launch through llamaMan, or a volume set up by hand) also reinstalls. That costs one setup pass that reuses the files.
- **One live instance per Strata model per node.** Two instances of one model would run setup over the same files.
- **A launch is refused while llamaMan is still downloading that model**, so the container doesn't download it a
  second time into its volume.
- **Pre-downloaded shards skip update-check provenance.** A pinned file must not be "updated" to the repo's `main`.
- **Virtual models are listed only where they can run** (`STRATA_ENABLED` plus NVIDIA): in `/api/models`,
  `/api/tags`, `/v1/models` and name resolution. Update scans and the cluster model snapshot stay files-only. Peers
  learn engine availability from `snapshot.system.engines` (nested under `system`, so older peers ignore it).
- **Load stage** comes from the last 64 KB of the container log: setup's `=== Step N ===` headers, the download's
  `\r` progress line, and the server's "loading the model". It's best-effort and degrades to no stage line.
- **vLLM (#53)** would be another `Engine`: its own image, command, port (`internal_port`), health semantics and
  launch fields. The places that assume both engines use port 8080 (`resolve_llama_endpoint` callers) are the ones
  to parameterise then.

## Not verified without a GPU

- An actual Strata launch, first-run download, pack build, load and chat. Everything above was checked against the
  upstream source and with unit tests that mock Docker; no Strata container has run.
- The exact log format on a real first start. The stage parser follows `setup.py`'s `step()` / `download()` output
  and the server's startup prints.
- Behaviour without `memlock=-1`, and the RAM actually used under `LOW_RAM=on` with a container limit.
- Nested bind mount (`/data/models/<tag>` over a named volume at `/data`) on the Docker versions in use. Docker
  supports nested mounts, but this hasn't been run here.

## Known limitations

- NVIDIA only. Strata's AMD (HIP) path is an experimental manual build upstream and isn't offered.
- `LAYER_SPLIT` (multi-GPU layer placement) isn't exposed. Pinning several GPUs uses Strata's automatic split.
- The vision encoder (mmproj) is always downloaded by Strata itself, into the volume.
- Strata's own web UI (`/`, `/settings`, `/load`, `/unload`) is reachable through the instance port, but llamaMan
  doesn't use or manage it. Its idle-unload feature is unused; llamaMan's sleep stops the container instead.
- Changing a setting that is recorded in the setup re-runs Strata's setup pass on the next start (the files are reused).
- A first start can take longer than an HTTP client's own timeout (Open WebUI and others). llamaMan keeps waiting up
  to `STRATA_LOAD_TIMEOUT`, but the client may give up first. Pre-download, or launch once from the UI before serving.
- The catalogue (families, sizes, pinned revisions, file names) is a mirror of Strata's `setup.py`. A new upstream
  family or size needs a matching edit in `core/engines/strata.py`.

## Manual verification checklist (NVIDIA host)

Prerequisites: an NVIDIA driver 580+, a working `docker run --gpus all`, 64 GB RAM recommended, and about 150 GB
free disk.

1. **Build:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Optionally
   add `--build-arg CUDA_ARCHITECTURES=<your arch>`.
2. **Enable:** set `STRATA_ENABLED=true` and restart llamaMan.
   - `GET /api/engines` shows strata `available: true`, with 8 models.
   - Settings → Docker Images → *Built locally* shows `strata:latest` as **local**.
3. **Gating:** on a non-NVIDIA node (or with `GPU_TYPE=rocm`), Strata models aren't listed, and
   `POST /api/instances` for `/strata/qwen-IQ2_XS` returns 400 with the NVIDIA reason.
4. **Launch from the UI:** select `strata/qwen-IQ2_XS`.
   - The llama.cpp-only fields are hidden, the Strata section is shown, and Max Concurrent is locked at 1.
   - Set a 32g memory limit and check the below-64 GB warning appears; clear it again.
   - Launch.
5. **First run (container download):**
   - The instance card shows *setting up → downloading NN% → preparing pack → loading*.
   - The Logs button streams Strata's setup output.
   - The card turns **healthy**, and `docker inspect` shows `Ulimits: memlock -1`, the env vars and the `/data`
     volume.
6. **Chat:**
   - Through the Ollama proxy: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Through `/v1/chat/completions` on `:42069`, and directly on the instance port.
   - Also try `model: "qwen3.8-flash-next-iq2_xs"`.
   - Send two requests at once: the second queues (the card shows 1/1 active · 1 queued).
7. **Idle sleep and wake:**
   - Set Idle Timeout to 1 min (the cold-start warning shows). After a minute the instance is **sleeping** and its
     container is gone.
   - The next request wakes it (minutes) and is answered.
   - No setup pass in the log (no `=== Step` lines; `REINSTALL` absent from `docker inspect`).
8. **Settings change:**
   - Change the context to 65536 and restart.
   - `docker inspect` shows `REINSTALL=1`, and the log shows a setup pass without a download.
   - On the next restart, `REINSTALL` is absent again.
9. **Pre-download path:**
   - Pick another model (e.g. `strata/qwen-Q2_0`) and click *Pre-download with llamaMan*.
   - The Downloads tab shows the pinned-revision download.
   - Launching during the download is refused.
   - After it completes, launch: `docker inspect` shows `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/q2_0`, and the
     log shows no shard download (only the MTP layer and pack).
10. **Stop / remove:** stop the instance. The container is removed, and Remove clears the card.
11. **Restart llamaMan with a Strata instance running:**
    - It is re-attached (state restore), or adopted if the state was lost: delete it from `state.json` and restart.
    - The card shows the Strata badge and `strata/qwen-IQ2_XS`.
    - Requests still work.
12. **Bare-metal mode:**
    - Run llamaMan with `python app.py` (`LLAMAMAN_IN_DOCKER=false`) and repeat 4-6.
    - llamaMan reaches the published port on `LLAMA_HOST_ADDR`.
13. **Cluster:**
    - Add a non-NVIDIA peer. Strata models appear for the NVIDIA node only when it's the Target node.
    - Launching on the NVIDIA peer from the other node's UI works.
14. **llama.cpp unaffected:** launch a GGUF model as before.
    - The command line (`docker inspect` Args) matches one from before the upgrade.
    - Presets saved before the upgrade load and launch unchanged.
