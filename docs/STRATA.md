# Strata engine

*(opt-in, NVIDIA only)* [Strata](https://github.com/Niko1221/Strata) runs the Qwen3.8-Flash-Next mixture-of-experts family (and its Coder, Swift 1.5 and Unsloth variants) on consumer NVIDIA GPUs plus system RAM. llamaMan can launch, monitor, proxy, sleep/wake and stop Strata instances next to llama.cpp ones. llama.cpp behaviour does not change: the engine layer (`core/engines/`) keeps the llama.cpp launch byte-for-byte identical (snapshot-tested).

## Contents

- [Getting started](#getting-started) · [Limits and behaviour](#limits-and-behaviour) · [Design notes and verification](#design-notes-and-verification)

## Getting started

**1. Get an image** (upstream publishes none; the host needs NVIDIA driver **580+**, i.e. CUDA 13, and the NVIDIA Container Toolkit). After enabling Strata (step 2), either pull one you trust by name in **Settings → Docker Images** (engine *Strata* in the pull row), or use **Manage Strata images → Pull repo & build latest**, which downloads the repository into `STRATA_SRC_DIR` and builds `STRATA_BUILD_IMAGE` through the Docker socket for the node's GPU generations (read from the driver; all supported ones if that fails), or build it by hand:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. Enable it** on the llamaMan node: `STRATA_ENABLED=true` (see [Strata variables](CONFIGURATION.md#strata)). Settings → Docker Images then shows *Manage Strata images*: the pulled Strata images, the built one (downloaded and built commits, the build button) and their auto-update. The launch form offers Strata once one of them is on the node.

**3. Launch.** The model library gains one virtual entry per family × size: `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(experimental)*. Each is its own instance and container, so switching models works like llama.cpp: launch or wake another instance, with the usual [eviction](MODELS.md#model-eviction) rules (a Strata instance is one chat model). The entries also appear in `/api/tags` and `/v1/models`, and requests may use the id (`strata/qwen-IQ2_XS`, any case, `:tag` ignored) or Strata's own served name (`qwen3.8-flash-next-iq2_xs`).

The launch form hides the llama.cpp-only fields (GPU layers, MoE, threads, parallel, KV types, flash attention, spec decoding, mmproj, DRY, extra args) and shows **Image Input** (no / GPU / CPU), **KV Cache** (Strata default int8, `int8`, `q4_0`, `k8v4`) and **Low-RAM Mode** (auto / on / off). The API rejects llama.cpp-only fields set for a Strata model.

**First start.** Strata downloads ~58-111 GB, prepares its "pack" and fetches its MTP layer before the server opens its port; later starts go straight to loading (1-3 minutes). The instance card shows *setting up → downloading (with %) → preparing pack → loading* parsed from the container log, and the Logs button streams the log. Two ways to download:

- **Let Strata download** into its `/data` volume (default).
- **Pre-download with llamaMan** (button in the Strata section, or `POST /api/engines/strata/download`): an ordinary entry in the Downloads tab (progress, pause/resume, retry, HF tokens, speed limits) that fetches the shards from the Hugging Face revision Strata pins into `MODELS_DIR/strata/<tag>/`. Once complete they are bind-mounted into the container and Strata's setup uses them instead of downloading. A launch is refused while that download is still running. The MTP layer (~5 GB) and the pack are still prepared by Strata on the first start.

## Limits and behaviour

- **One request at a time.** Max Concurrent is forced to 1; further requests queue in llamaMan's gate (Max Queue Depth, then 429).
- **RAM.** Strata loads 32-62 GB into RAM (`--ulimit memlock=-1` is always set). A **System Memory Limit** forces Low-RAM mode on, because Strata's setup reads the host's RAM and cannot see a container limit; the UI warns below ~64 GB.
- **Settings are baked into Strata's setup.** Context / image input / KV / Low-RAM / GPU pinning are recorded per model on the data volume. llamaMan remembers the settings each model last came up healthy with and passes `REINSTALL=1` only when they change (the files are reused).
- **Idle timeout** works (sleep stops the container; the next request wakes it), but a cold start takes minutes and the waking request waits for it (`STRATA_LOAD_TIMEOUT`).
- **One instance per Strata model per node** (instances of one model share its setup on the volume).
- **NVIDIA only.** The option is unavailable, with the reason, on ROCm / Intel / Vulkan / CPU nodes and when `STRATA_ENABLED` is off. In a cluster each node advertises its engines, so only capable nodes offer Strata models.
- Bare-metal llamaMan reaches Strata the same way as llama-server (published port on `LLAMA_HOST_ADDR`).

---

# Design notes and verification

This page covers how the integration works, what it relies on upstream, its limits, and how to verify it on a real NVIDIA
host.

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
  | image | `STRATA_IMAGE` (else the built, then a pulled Strata image) |
  | entrypoint | `sh -c ENTRY_SCRIPT` (`core/engines/strata.py`), under Docker's init; not the image's own (see 2.) |
  | command | none (the script reads env vars only) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `GGUF_DIR=/models`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `SERVE_PORT=8081`, `RELAY_SCRIPT`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | data volume → `/data`; logs; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/models` (rw, for setup's `.done` marks), when llamaMan holds complete shards; renamed copies are mounted file by file under the published names |
  | mem_limit | from System Memory Limit (forces `LOW_RAM=on`) |
  | network / ports / labels | as for llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

## Answers to the open questions

1. **Readiness.** `/health` answers `{"status": "ok", ..., "service": "strata"}` before the API-key gate
   (`serve/server.py:2172`). The HTTP port only opens after the model has loaded: `serve()` runs at
   `server.py:3008`, after the engine starts at `:2925`. While Strata is downloading or loading, connections are
   refused, so llamaMan's existing health check (`status == "ok"`) needs no special case. `/v1/models` returns an empty
   list while the model is unloaded in lazy mode (`server.py:2190`), so it is not a good readiness signal.
   `/v1/status` needs the API key when one is set.
2. **Settings.** llamaMan passes the 0.1.x env contract (`HOST`, `PORT`, `GPU` / `GPUS` / `LAYER_SPLIT`,
   `LOW_RAM`, `KV`, `CONTEXT`, `VISION`, `REINSTALL`, `GGUF_DIR`) to its own entrypoint script. From 0.2.x the image's
   `docker/entrypoint.sh` runs `strata launch` instead: it ignores those variables, requires an API key for anything
   but loopback (`strata_cli/cli.py` `_launch_plan`), generates one if none is given, downloads whatever it can't
   find, and ends the container from a status loop. The installer and server are still in the image, so the script
   runs `/opt/strata/terminal/setup.py --setup --yes --no-start ...` in Control mode (`STRATA_CONTROL_INSTALL=1`,
   `STRATA_INSTALL_DIR=/data/install`: the engine is the image's component) when the run config
   `/data/install/strata-<tag>.json` is missing or `REINSTALL=1`, then execs `serve/server.py --engine strata
   --config <that file>` with `STRATA_NO_ACQUIRE=1`. The server refuses a non-loopback bind without an API key
   (`serve/server.py` `main()`: "a non-loopback bind requires an inference API key"), and llamaMan sends backends no
   key, so it binds `127.0.0.1:8081` and a small TCP relay (`RELAY_SCRIPT`) carries `0.0.0.0:8080` to it. Its Host
   check still applies through `STRATA_ALLOWED_HOSTS`. `LAYER_SPLIT` is written into the config's `layer_split` at every
   start. On a 0.1.x image (no `/opt/strata/terminal`) it runs that image's `docker-entrypoint.sh`, which reads the
   same variables. GPU pinning uses container numbering: the NVIDIA runtime
   renumbers the host GPUs llamaMan attaches, so host `1,3` becomes `GPUS=0,1`. llamaMan doesn't set `API_KEY`, which
   matches how it runs llama-server (no key); its own proxy and auth sit in front.
3. **Reusing GGUFs on disk.** llamaMan mounts its completed download of the shards at `/models` and sets
   `GGUF_DIR=/models`, which the script passes as `setup.py --gguf-dir` (0.1.x's entrypoint forwards `GGUF_DIR` too).
   With `--gguf-dir` setup never downloads the model; it checks each shard against its own tensor directory. 0.2.x no
   longer reads the 0.1.x `/data/models/<tag>` folder, which is why a 0.2.x image started through its own entrypoint
   downloads the model again.
   Shards are fetched from the Hugging Face revisions Strata pins (`setup.py:65-70`, mirrored in
   `core/engines/strata.py`). The MTP layer (~5 GB) and the pack are still prepared inside the container.
4. **Memory limit / memlock.** Setup reads RAM from `/proc/meminfo`, which is the host's total, so a capped container
   must ask for `LOW_RAM=on` (Dockerfile comment at `:31`, `docs/INSTALL.md:116-118`). Without it, setup picks the
   in-RAM mode and the container is likely to be OOM-killed. llamaMan forces `LOW_RAM=on` whenever a memory limit is
   set, and the UI warns below 64 GB. `memlock=-1` matters because the engine page-locks part of its arena for GPU DMA,
   and experts it cannot pin go through slower host copies (`src/prefill/prefill.cpp:110`, `:207-214`). From the
   source, a missing memlock limit looks like a slowdown rather than a failure, but **I couldn't verify that without a
   GPU**. llamaMan always sets it.

Also from the source: Strata serves one request at a time (`server.py:1323`). The container runs with Docker's init,
and the script execs the server, so `docker stop` reaches it. A container stopped mid-download resumes from its `.part` files
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
- The entrypoint script against a real 0.2.x image (setup in Control mode, then `serve/server.py`). It was checked
  against the upstream source and `sh -n`, and the catalogue dump against the real 0.2.x `setup.py` tables.

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
   add `--build-arg CUDA_ARCHITECTURES=<your arch>`. Or, after step 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** It builds for this node's GPU generations only.
2. **Enable:** set `STRATA_ENABLED=true` and restart llamaMan.
   - `GET /api/engines` shows strata `available: true`, with its models (9 in the built-in list).
   - Settings → Docker Images → *Manage Strata images* shows `strata:latest` as **local**.
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
   - After it completes, launch: `docker inspect` shows `MODELS_DIR/strata/q2_0/Q2_0 → /models` and
     `GGUF_DIR=/models`, and the log shows no shard download (only the MTP layer and pack).
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
