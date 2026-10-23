# Models, downloads & eviction

Everything about the model library: adding models, keeping them current, how the proxy picks and evicts them, and the housekeeping settings.

## Contents

- [Models](#models) · [Decision Models](#decision-models) · [Model Eviction](#model-eviction) · [Download Settings](#download-settings) · [Docker Image Management](#docker-image-management) · [Model Archive](#model-archive) · [Model Backup and Restore](#model-backup-and-restore) · [Cleanup Settings](#cleanup-settings)

## Models

Place models under the `/models` volume as `.gguf` files, HuggingFace repo directories (containing `config.json`), or use the **Download** button in the UI.

### Display Name

Models are listed by GGUF filename stem by default. Each model can be given a friendly **Display Name** (in the Launch form). When set it's the id API clients see on `/api/tags` and `/v1/models`, so OpenWebUI shows `Qwen 2.5 14B` instead of `Qwen2.5-14B-Instruct-Q4_K_M`. Must be unique and not clash with another model's filename or a cluster queue-group name.

### Model updates

The **Check for updates** button (shown for models with a known source repo) compares the local file against the repo's published content hash:

- **Up to date** - recorded hash matches
- **Update available** - re-pulls into a staging folder and atomically swaps in only once the download completes (a failed pull leaves the current model intact). Refuses with 409 while the model is loaded.
- **Verify hash** - hashes the local file (background job with progress) when no hash has been recorded yet

An opt-in background scan (**Settings → Download Settings**, see [Download Settings](#download-settings)) keeps these answers ready ahead of time.

## Decision Models

Decision models answer typed questions about an input with probabilities instead of generating text: the System One API introduced by TypeSafe's Jev, served by llama.cpp (images built after 2026-10-05) on `POST /v1/systemone`. Open GGUFs: Julia-1, Laya, Kev-4B, lev, Clef-Flash, Clef, OpenJev (OpenJev and Clef also take images, with their `--mmproj`). llama-server recognises them from `<arch>.decision.type` in the GGUF; no flag is needed.

- **Detection.** Picking one in the launch form says so under Model Path. Without a preset the form is filled for it: **Exclude from Max Models** on, **Web UI** off (its chat page can't drive the model), **Parallel Slots** `8` (each question of a request takes a slot, and they share the input), **Embedding Model** off (llama.cpp turns embedding mode on itself where the model needs it; forcing it is not needed). An encoder (BERT-style) model also gets Context, Batch and Micro-batch Size set to its context length: its whole input must fit in one micro-batch. **Clef** (and Clef-Flash) asks all the questions in one prompt and reads them from one pass, so it gets **Parallel Slots** `1` and Batch = Micro-batch = Context (16384, its trained default, when Context is blank): the whole request (input, every question and option) must fit in one micro-batch, or llama-server answers "increase the physical batch size".
- **Routing.** `POST :42069/v1/systemone` picks the model by `"model"` like `/v1/chat/completions` (auto-start, eviction rules, cluster forwarding, queue). The request is forwarded unchanged (no sampling overrides). On the instance's own port it goes through the per-instance proxy's queue when that proxy runs.
- **Chat endpoints refuse them** (HTTP 422) - they don't generate text.
- **Request log:** the answers, and `input_tokens` as the prompt tokens (`output_tokens` is always 0).
- **Errors:** a question whose options don't fit in one batch is rejected by llama-server ("increase the physical batch size") - raise Batch Size / Micro-batch Size.

```bash
curl http://localhost:42069/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "Clef-Flash-Q4_K_M",
  "state": "Customer: I was charged twice for my order last week.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "shipping": null}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["can wait", "today", "right now"]}
  }
}'
```

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
- Instances with **Exclude from Max Models** on (launch form) are not counted and never evicted: embedding models, decision models, anything small that should stay up. Turning **Embedding Model** on turns it on too, and presets saved before the toggle existed get it from their Embedding Model setting.
- `LLAMAMAN_MAX_MODELS=0` disables eviction entirely.

## Download Settings

Under **Settings → Download Settings**:

- **Global speed limit (Mbps)** (0 = unlimited) - caps every download on this node and applies to running downloads within a second. Per node: in a cluster each node has its own limit (a value saved before it became per-node applies to every node until that node saves its own)
- **Auto-retry failed downloads** (off by default) + **Retry count per failed download** (default 3)
- **Check models for updates in the background** (off by default) - opt-in worker that asks each source repo whether a file has been republished, plus computes a checksum for any model that doesn't have one yet. **Update check interval (hours)** default 24. Hashes at most one model per pass; never runs while a download is in progress.

## Docker Image Management

**Settings → Docker Images**: pull any llama.cpp image by name, delete old local images (disabled for the active `LLAMA_IMAGE`, and returns an error if Docker refuses because a container is using it), and optionally auto-update the node's llama.cpp images (`LLAMA_IMAGE` and every image pulled there) on a schedule. The pull row has an engine select: with `STRATA_ENABLED`, an image pulled as *Strata* goes to *Manage Strata images* (there is no default Strata source; type the name). **Pull repo & build latest** downloads the newest commit of the Strata repository (a GitHub tarball, into `STRATA_SRC_DIR`) and builds `STRATA_BUILD_IMAGE` with BuildKit through the Docker socket. Strata's own auto-update re-pulls the pulled Strata images and, when the repository was downloaded, fetches its newest commit and rebuilds when it changed (or the image is missing); it never downloads the repository on its own. **Delete image & repo** removes the built image (refused while a running instance uses it) and the downloaded repository, for a node that runs a pulled image. Strata launches use `STRATA_IMAGE` when it is on the node, else the built image, else a pulled one. See [Strata](STRATA.md).

## Model Archive

*(opt-in)* Move models off fast storage to a second volume and back from the UI, instead of shuffling files by hand on the host.

**Enable it** by mounting the archive volume into the llamaMan container and pointing `ARCHIVE_DIR` at it:

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

Unset, the feature is hidden. llamaMan moves the files itself, so no `HOST_` path is needed.

**Use it.**

- Hover a model in the library and click the archive icon.
- Archived models stay in the list, dimmed, with an *archived* badge and a **Restore** button. They can't be selected for launch and never appear in `/api/tags` / `/v1/models`.
- Moves show in the **Downloads** panel with progress, speed and time left, and can be cancelled.
- Restore puts a model back at its original path, so presets, favorites, notes and display names come back with it.

**What moves.** A model's whole top-level folder under `/models`: a download's folder, with every shard of a split GGUF. A `.gguf` lying directly in `/models` moves together with its split siblings. A Strata model's shard folder (`strata/<tag>`) moves on its own.

**Safety.**

- **Same disk:** the move is an instant rename.
- **Across disks:** llamaMan copies into a hidden `.llamaman-partial-*` folder on the destination, fsyncs, checks every file's size, renames it into place, and only then deletes the source.
- **Failures:** a cancel, a failed check or too little free space leave the source untouched. Partial copies left by a restart mid-move are removed at startup.
- **One move at a time.**
- **Refused** while an instance uses the model (including a sleeping one, or as a draft model / mmproj, or a Strata instance using the shards), while a download writes into it, or when the destination name already exists. Launching or deleting a model that is being moved is refused too.

**Limits.** Move jobs live in memory: after a llamaMan restart the list is empty, and a move that was interrupted must be started again (nothing is lost). In cluster mode each node archives to its own volume, and the buttons show for the node you are browsing.

## Model Backup and Restore

**Settings → App Settings**:

- **Download Stored Models JSON** - exports all scanned models + preset configs to a timestamped JSON
- **Restore from JSON** - per model: already-present → preset merged in (existing values preserved); missing but with a HuggingFace source → queued to download with preset pre-populated; missing and no known source → reported as unrestorable

## Cleanup Settings

Under **Settings → Cleanup Settings** (all runs periodically in the background; only removes/updates records - never deletes model files):

- **Auto-clean completed/failed downloads** (default 24h) - active downloads never touched
- **Auto-clean stopped instances** (default 24h) - running instances never removed
- **Auto-remove stale instance records** (default 5 min interval) - re-checks `starting`/`healthy`/`sleeping` records against Docker; ones whose container is gone are marked stopped
