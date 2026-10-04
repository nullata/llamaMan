# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata - https://github.com/Niko1221/Strata - runs the Qwen3.8-Flash-Next
mixture-of-experts family on one or more consumer NVIDIA GPUs plus system RAM.

What llamaman relies on (upstream file references are to the Strata repo):

  * The image is built locally (`docker build -t strata .`), never pulled.
  * Its entrypoint (docker-entrypoint.sh) takes every choice as an env var -
    FAMILY, MODEL, CONTEXT, VISION, KV, HOST, PORT, GPU / GPUS, LOW_RAM,
    REINSTALL - and ignores command arguments, so command is empty here.
  * Each family x size is set up once into the /data volume (download ~58-111
    GB, build the "pack", fetch the MTP layer) and recorded as
    /data/config/strata-<tag>.json. Later starts serve straight from it.
    Context / vision / KV / LOW_RAM are baked into that config: changing one
    needs REINSTALL=1 (docker-entrypoint.sh:44), which this engine sends only
    when the settings differ from the ones it last saw come up healthy.
  * The HTTP port opens only after the model is loaded (serve/server.py
    main(): serve() runs after the engine starts), and /health answers
    {"status": "ok", ...} before the API-key gate - the same readiness
    contract as llama-server, so the health poller needs no special case.
  * One request at a time (serve/server.py v1_status: concurrency.serving=1).
  * It loads 32-62 GB into RAM and page-locks part of it for GPU DMA, so the
    container runs with memlock unlimited. Its setup reads RAM from
    /proc/meminfo (the host's, not a cgroup limit), so a memory-capped
    container must ask for LOW_RAM=on itself (Dockerfile comment, INSTALL.md).
  * Shards already on disk are used instead of downloading when they sit at
    /data/models/<fam tag><SIZE>/<original name>, case kept (setup.py "#173: a whole file copied in
    by hand has no finish mark" - it checks them against their own tensor
    directory), so a model downloaded through llamaman's downloader into
    MODELS_DIR/strata/<tag>/ is bind-mounted there.

Each family x size is a virtual model with the path /strata/<family>-<SIZE>
(id strata/<family>-<SIZE>) and is its own llamaman instance.
"""

import copy
import json
import os
import re
import threading

from core.engines.base import Engine

# Virtual model paths. The leading slash keeps them valid preset keys
# (api/presets._normalize_model_path) - nothing is ever read at this path.
PATH_PREFIX = "/strata/"
ID_PREFIX = "strata/"

# Hugging Face revisions Strata's setup pins its downloads to (setup.py
# HF_REVISIONS). Mirrored so a pre-download through llamaman fetches the same
# bytes; if upstream re-pins, setup re-checks every shard's tensor directory
# and fails loudly on a mismatch rather than serving a wrong file.
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",
}

# setup.py FAMILIES. `served` is the model name Strata's own /v1/models
# reports (setup.py: model_name = f"{fam['name']}-{model.lower()}").
# `subdir`: the repo keeps each size in a folder named after it.
FAMILIES = {
    "qwen": {
        "title": "Qwen3.8-Flash-Next", "served": "qwen3.8-flash-next",
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "subdir": True,
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "shards": 2,
        "vision": True, "experimental": False,
    },
    "swift": {
        "title": "Swift 1.5", "served": "swift-1.5",
        "repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "subdir": False,
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "shards": 2,
        "vision": True, "experimental": False,
    },
    "coder": {
        "title": "Qwen3.8-Flash-Next Coder", "served": "qwen3.8-flash-next-coder",
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "subdir": True,
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "shards": 2,
        "vision": True, "experimental": False,
    },
    "unsloth": {
        "title": "Qwen3.8-Flash-Next (Unsloth)", "served": "qwen3.8-flash-next-unsloth",
        "repo": "unsloth/Qwen3.8-Flash-Next-GGUF", "subdir": True,
        "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4,
        "vision": False, "experimental": True,
    },
}

# setup.py MODELS: which families ship each size, and what it costs.
SIZES = {
    "Q2_0": {"families": ("qwen",), "download_gb": 66.4, "ram_gb": 48},
    "IQ2_XS": {"families": ("qwen", "swift"), "download_gb": 68.0, "ram_gb": 48},
    "IQ3_XXS": {"families": ("qwen", "swift"), "download_gb": 75.8, "ram_gb": 60},
    "IQ3_S": {"families": ("qwen",), "download_gb": 83.6, "ram_gb": 62},
    "IQ1_M": {"families": ("coder",), "download_gb": 58.4, "ram_gb": 32},
    "UD-Q4_K_XL": {"families": ("unsloth",), "download_gb": 111.3, "ram_gb": 48},
}

# docker-entrypoint.sh's own default, and setup.py's CONTEXTS menu.
# ---------------------------------------------------------------------------
# The catalogue from the installed Strata image
# ---------------------------------------------------------------------------
# FAMILIES / SIZES / HF_REVISIONS above are a copy of setup.py's tables at
# the pinned commit. The image llamaman runs carries its own setup.py, so the
# background poller reads the tables out of it (refresh_catalogue_from_image)
# whenever the image changes and replaces these dicts in place; everything
# here reads them through those same names. Any failure keeps the built-in
# copy (logged once per image).

_BUILTIN_CATALOGUE = (copy.deepcopy(FAMILIES), copy.deepcopy(SIZES), dict(HF_REVISIONS))
_CATALOGUE_LOCK = threading.Lock()
_catalogue_state = {"image_id": None, "source": "built-in", "error": None}

# Run inside the image (network off): print setup.py's tables as JSON.
# setup.py's main() is guarded and it imports only the standard library.
_DUMP_SCRIPT = r"""
import json, sys
sys.path.insert(0, "/opt/strata")
sys.argv = ["setup.py"]
import setup as s
fam = {k: {"title": f.get("title"), "name": f.get("name"), "hf": f.get("hf"), "file": f.get("file"),
           "shards": f.get("shards", 2), "vision": f.get("vision", True),
           "experimental": f.get("experimental", False)} for k, f in s.FAMILIES.items()}
mod = {k: {"download_gb": m.get("download_gb"), "ram_gb": m.get("ram_gb"),
           "families": list(m.get("families", ("qwen", "swift")))} for k, m in s.MODELS.items()}
print(json.dumps({"families": fam, "models": mod, "revisions": s.HF_REVISIONS}))
"""


def catalogue_from_dump(dump: dict) -> tuple[dict, dict, dict]:
    """setup.py's tables (as _DUMP_SCRIPT prints them) -> (FAMILIES, SIZES,
    HF_REVISIONS) in this module's shape. ValueError when they don't fit."""
    try:
        revisions = {str(k): str(v) for k, v in dump["revisions"].items()}
        families = {}
        for key, f in dump["families"].items():
            hf = str(f["hf"] or "")
            repo = next((r for r in revisions if r in hf), None)
            if not repo:
                raise ValueError(f"family {key}: no pinned revision for {hf!r}")
            file = str(f["file"] or "")
            if "{q}" not in file or "{i}" not in file:
                raise ValueError(f"family {key}: unexpected file pattern {file!r}")
            shards = int(f["shards"])
            if shards < 1:
                raise ValueError(f"family {key}: {shards} shards")
            families[str(key)] = {
                "title": str(f["title"] or key), "served": str(f["name"] or key),
                "repo": repo, "subdir": hf.endswith("{q}/"), "file": file, "shards": shards,
                "vision": bool(f["vision"]), "experimental": bool(f["experimental"]),
            }
        sizes = {}
        for key, m in dump["models"].items():
            fams = tuple(str(x) for x in m["families"])
            unknown = [x for x in fams if x not in families]
            if not fams or unknown:
                raise ValueError(f"size {key}: unknown families {unknown or fams}")
            sizes[str(key)] = {"families": fams, "download_gb": float(m["download_gb"]),
                               "ram_gb": int(m["ram_gb"])}
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"unexpected table shape ({type(e).__name__}: {e})") from e
    if not families or not sizes:
        raise ValueError("empty tables")
    return families, sizes, revisions


def _apply_catalogue(families: dict, sizes: dict, revisions: dict) -> None:
    with _CATALOGUE_LOCK:
        FAMILIES.clear()
        FAMILIES.update(families)
        SIZES.clear()
        SIZES.update(sizes)
        HF_REVISIONS.clear()
        HF_REVISIONS.update(revisions)


def _catalogue_cache_file() -> str:
    from config import DATA_DIR
    return os.path.join(DATA_DIR, "strata_catalogue.json")


def catalogue_source() -> dict:
    """Where the current tables came from, for /api/engines."""
    return dict(_catalogue_state)


def refresh_catalogue_from_image() -> None:
    """Read the tables from the Strata image when its id changed (cheap when
    it didn't: one image lookup). Called by the background poller."""
    from config import STRATA_ENABLED, STRATA_IMAGE
    from config import logger
    if not STRATA_ENABLED:
        return
    import docker
    from core.helpers import get_docker_client
    try:
        client = get_docker_client()
        image_id = client.images.get(STRATA_IMAGE).id
    except docker.errors.ImageNotFound:
        image_id, why = None, f"image {STRATA_IMAGE} is not built"
    except Exception as e:
        image_id, why = None, f"Docker unavailable ({type(e).__name__})"
    if image_id is None:
        if _catalogue_state["source"] != "built-in" or _catalogue_state["error"] != why:
            _apply_catalogue(*copy.deepcopy(_BUILTIN_CATALOGUE))
            _catalogue_state.update(image_id=None, source="built-in", error=why)
        return
    if image_id == _catalogue_state["image_id"]:
        return

    try:
        with open(_catalogue_cache_file()) as f:
            cache = json.load(f)
        cache = cache if isinstance(cache, dict) else {}
    except (OSError, ValueError):
        cache = {}
    try:
        dump = cache.get(image_id)
        if dump is None:
            out = client.containers.run(
                STRATA_IMAGE, remove=True, network_disabled=True, stdout=True, stderr=False,
                entrypoint=["sh", "-c", 'cd /opt/strata && { .venv/bin/python -c "$LLAMAMAN_DUMP" 2>/dev/null '
                                        '|| python3 -c "$LLAMAMAN_DUMP"; }'],
                environment={"LLAMAMAN_DUMP": _DUMP_SCRIPT},
            )
            dump = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
        tables = catalogue_from_dump(dump)
    except Exception as e:
        why = f"could not read the model list from {STRATA_IMAGE}: {e}"
        logger.warning("strata: %s - using the built-in list", why)
        _apply_catalogue(*copy.deepcopy(_BUILTIN_CATALOGUE))
        _catalogue_state.update(image_id=image_id, source="built-in", error=why)
        return
    _apply_catalogue(*tables)
    _catalogue_state.update(image_id=image_id, source="image", error=None)
    if image_id not in cache:
        cache = {image_id: dump}                     # only the current image's
        try:
            tmp = _catalogue_cache_file() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, _catalogue_cache_file())
        except OSError:
            pass
    logger.info("strata: model list read from %s (%s): %d models",
                STRATA_IMAGE, image_id[:19], sum(len(m["families"]) for m in tables[1].values()))


DEFAULT_CONTEXT = 32768
CONTEXT_CHOICES = (8192, 32768, 65536, 131072, 262144, 393216, 524288)
VISION_CHOICES = ("no", "yes", "cpu")
KV_CHOICES = ("", "int8", "q4_0", "k8v4")
LOW_RAM_CHOICES = ("auto", "on", "off")

# Below this a memory limit is very likely to starve even LOW_RAM=on (the UI
# warns; the API does not refuse - the operator may know their model).
MEMORY_WARN_GB = 64

_SETUPS_LOCK = threading.Lock()


_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$", re.IGNORECASE)
# The quantization at the end of a shard's stem (setup.py GGUF_QUANT):
# "...-GSQ-RCO-IQ3_S", "...-UD-Q4_K_XL".
_QUANT_RE = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?I?Q\d+(?:_[A-Za-z0-9]+)*)$", re.IGNORECASE)


def model_for_file(path: str | None) -> tuple[str, str] | None:
    """A shard of a model Strata runs -> (family, size); else None.

    The published file names match exactly. Any other name matches when it
    is split like the published files and ends in a quant Strata runs (a
    renamed copy or another upload of the same files): setup.py reads the
    shard count from the name and runs only its own quants
    (SUPPORTED_GGUFS), so a different split or quant is not offered. Sizes
    two families share (IQ2_XS, IQ3_XXS) are Swift's when the name says so."""
    if not isinstance(path, str) or not path.lower().endswith(".gguf"):
        return None
    name = os.path.basename(path)
    for family, fam in FAMILIES.items():
        for size, meta in SIZES.items():
            if family in meta["families"] and name in shard_files(family, size):
                return family, size
    m = _SHARD_RE.match(name)
    q = _QUANT_RE.search(m.group("stem")) if m else None
    size = next((s for s in SIZES if q and s.lower() == q.group(1).lower()), None)
    if not size:
        return None
    families = SIZES[size]["families"]
    lowered = name.lower()
    family = next((f for f in families if f != "qwen" and f in lowered), None) \
        or ("qwen" if "qwen" in families else (families[0] if len(families) == 1 else None))
    if not family or int(m.group("n")) != FAMILIES[family]["shards"]:
        return None
    return family, size


def _shard_pairs(path: str, parsed: tuple[str, str]) -> list[tuple[str, str]]:
    """(the sibling shard's path, its published name) for every shard of the
    model `path` belongs to, whether or not they exist."""
    published = shard_files(*parsed)
    name = os.path.basename(path)
    if name in published:
        own = published
    else:
        stem = _SHARD_RE.match(name).group("stem")
        own = ["%s-%05d-of-%05d.gguf" % (stem, i, len(published)) for i in range(1, len(published) + 1)]
    d = os.path.dirname(path)
    return [(os.path.join(d, n), pub) for n, pub in zip(own, published)]


def shard_mounts_for_file(path: str | None) -> list[tuple[str, str]] | None:
    """Every shard of the model `path` is a shard of, as (path, the
    published name setup.py looks for), when all are next to it; else None."""
    parsed = model_for_file(path)
    if not parsed:
        return None
    pairs = _shard_pairs(path, parsed)
    return pairs if all(os.path.isfile(f) for f, _ in pairs) else None


def shards_dir_for_file(path: str | None) -> str | None:
    """The folder of a downloaded shard when every shard of its model is
    there, else None."""
    mounts = shard_mounts_for_file(path)
    return os.path.dirname(mounts[0][0]) if mounts else None


def parse_model_path(model_path: str | None) -> tuple[str, str] | None:
    """'/strata/qwen-IQ2_XS' (or the id 'strata/qwen-IQ2_XS', or a downloaded
    shard file of it) -> ('qwen', 'IQ2_XS'). None when it isn't a Strata
    model in the catalogue. Family names have no '-', sizes may
    ('UD-Q4_K_XL'), so split once."""
    if not isinstance(model_path, str):
        return None
    p = model_path.strip()
    if p.startswith(PATH_PREFIX):
        rest = p[len(PATH_PREFIX):]
    elif p.lower().startswith(ID_PREFIX):
        rest = p[len(ID_PREFIX):]
    else:
        return model_for_file(p)
    family, sep, size = rest.partition("-")
    if not sep:
        return None
    family = family.lower()
    size_key = next((s for s in SIZES if s.lower() == size.lower()), None)
    if family not in FAMILIES or size_key is None:
        return None
    if family not in SIZES[size_key]["families"]:
        return None
    return family, size_key


def model_path_for(family: str, size: str) -> str:
    return f"{PATH_PREFIX}{family}-{size}"


def model_id_for(family: str, size: str) -> str:
    return f"{ID_PREFIX}{family}-{size}"


def setup_tag(family: str, size: str) -> str:
    """docker-entrypoint.sh's tag: qwen has an empty family prefix."""
    prefix = "" if family == "qwen" else f"{family}-"
    return f"{prefix}{size.lower()}"


def shards_dir_tag(family: str, size: str) -> str:
    """The folder under /data/models/ setup.py reads this model's shards from:
    its own tag, fam["tag"] + model with the size's ORIGINAL case
    (setup.py:3264, used at :3389) - e.g. "IQ3_S", "coder-IQ1_M". Only the
    entrypoint's config name and the pack folder are lowercased."""
    prefix = "" if family == "qwen" else f"{family}-"
    return f"{prefix}{size}"


def shard_files(family: str, size: str) -> list[str]:
    fam = FAMILIES[family]
    return [fam["file"].format(q=size, i=i) for i in range(1, fam["shards"] + 1)]


def repo_files(family: str, size: str) -> list[str]:
    """The shards' paths inside their Hugging Face repo."""
    prefix = f"{size}/" if FAMILIES[family]["subdir"] else ""
    return [prefix + f for f in shard_files(family, size)]


def catalogue() -> list[dict]:
    """Every family x size Strata can set up, as virtual model entries."""
    out = []
    for family, fam in FAMILIES.items():
        for size, meta in SIZES.items():
            if family not in meta["families"]:
                continue
            out.append({
                "id": model_id_for(family, size),
                "path": model_path_for(family, size),
                "family": family,
                "size": size,
                "title": f"{fam['title']} {size}",
                "served_name": f"{fam['served']}-{size.lower()}",
                "tag": setup_tag(family, size),
                "download_gb": meta["download_gb"],
                "ram_gb": meta["ram_gb"],
                "vision": fam["vision"],
                "experimental": fam["experimental"],
            })
    return out


def shards_base_dir(family: str, size: str) -> str:
    """Where llamaman keeps this model's shards: MODELS_DIR/strata/<tag>."""
    from config import MODELS_DIR
    return os.path.join(MODELS_DIR, "strata", setup_tag(family, size))


def model_download(family: str, size: str) -> dict | None:
    """The newest llamaman download record for this model's shards."""
    from core.state import downloads, downloads_lock
    model_id = model_id_for(family, size)
    with downloads_lock:
        mine = [dict(d) for d in downloads.values() if d.get("engine_model") == model_id]
    if not mine:
        return None
    return max(mine, key=lambda d: d.get("started_at") or 0)


def local_shard_dir(family: str, size: str) -> str | None:
    """The directory (as llamaman sees it) holding every shard of this model
    under MODELS_DIR/strata/<tag>/, or None. Looks in that folder and one
    level below it, because llamaman's downloader keeps a repo's size folder
    (e.g. .../iq2_xs/IQ2_XS/<shards>).

    A llamaman download of the shards that hasn't completed means the files
    may be partial (the downloader writes under the final names), so they
    are not offered to Strata until it finishes."""
    base = shards_base_dir(family, size)
    dl = model_download(family, size)
    if dl and dl.get("status") != "completed":
        return None
    names = shard_files(family, size)
    candidates = [base]
    try:
        candidates += sorted(os.path.join(base, d) for d in os.listdir(base)
                             if os.path.isdir(os.path.join(base, d)))
    except OSError:
        return None
    for d in candidates:
        if all(os.path.isfile(os.path.join(d, n)) for n in names):
            return d
    return None


def _host_path_for_models_subdir(path: str) -> str:
    """Translate a path under MODELS_DIR into the same path under
    HOST_MODELS_DIR (the Docker daemon's view)."""
    from config import HOST_MODELS_DIR, MODELS_DIR
    rel = os.path.relpath(path, MODELS_DIR)
    return os.path.join(HOST_MODELS_DIR, rel)


# ---------------------------------------------------------------------------
# Load-stage detection from the container log
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"=== Step (\d+): (.+?) ===")
_PROGRESS_RE = re.compile(r"([\w.\-]+\.gguf|[\w .\-]+): +([\d.]+) / ([\d.]+) GB \((\d+)%\)")
_STAGE_BY_STEP = {1: "setting up", 2: "setting up", 3: "setting up", 4: "setting up",
                  5: "downloading", 6: "preparing pack", 7: "loading"}


def load_stage_from_log(text: str) -> dict | None:
    """Best-effort 'where is the first start' from the tail of the container
    log: setup's '=== Step N: ... ===' headers (setup.py step()), download()'s
    carriage-return progress line, and the server's 'loading the model'.

    Returns {"stage": ..., "detail": ..., "percent": int|None} or None when
    nothing recognizable has been printed yet."""
    if not text:
        return None
    if "Setup stopped" in text[-4000:]:
        return {"stage": "setup failed", "detail": "see the log", "percent": None}
    stage, detail, percent = None, "", None
    step_pos = -1
    for m in _STEP_RE.finditer(text):
        step_pos = m.end()
        stage = _STAGE_BY_STEP.get(int(m.group(1)), "setting up")
        detail = m.group(2)
    load_pos = max(text.rfind("loading the model"), text.rfind("loading the vision encoder"))
    if load_pos > step_pos:
        stage, detail = "loading", "loading the model into RAM and VRAM"
    if stage == "downloading":
        # download() redraws one line with '\r'; the newest segment wins.
        tail = text[step_pos:].replace("\r", "\n").splitlines()
        for line in reversed(tail):
            m = _PROGRESS_RE.search(line)
            if m:
                percent = int(m.group(4))
                detail = f"{m.group(1).strip()}: {m.group(2)} / {m.group(3)} GB"
                break
    if stage is None:
        if "Setting up " in text:
            stage, detail = "setting up", ""
        else:
            return None
    return {"stage": stage, "detail": detail, "percent": percent}


def read_log_tail(path: str, max_bytes: int = 64 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class StrataEngine(Engine):
    name = "strata"
    label = "Strata"
    # docker-entrypoint.sh PORT; kept equal to llama-server's so
    # resolve_llama_endpoint's in-Docker port applies unchanged.
    internal_port = 8080
    default_ctx_size = DEFAULT_CONTEXT
    capabilities = {
        "max_concurrency": 1,
        "supports_gpu_layers": False,
        "supports_spec_decoding": False,   # it runs its own MTP drafting
        "supports_embeddings": False,
        "virtual_models": True,
        # Instances of one model share /data/models/<tag> and its setup
        # config; two at once would run setup over the same files.
        "single_instance_per_model": True,
        "nvidia_only": True,
    }
    option_keys = ("strata_vision", "strata_kv", "strata_low_ram", "strata_layer_split")
    launch_fields = frozenset({
        "ctx_size", "memory_limit", "gpu_devices", "image",
        "idle_timeout_min", "max_concurrent", "max_queue_depth",
        "share_queue", "share_queue_group", "share_queue_fallback",
        "auto_restart_on_crash",
        "proxy_sampling_override_enabled", "proxy_sampling_temperature",
        "proxy_sampling_top_k", "proxy_sampling_top_p",
        "proxy_sampling_presence_penalty", "proxy_sampling_repeat_penalty",
        "loop_detect_enabled", "loop_detect_min_chunk_chars",
        "loop_detect_min_repetitions", "loop_detect_max_buffer_chars",
        "loop_detect_scan_interval_s", "loop_detect_scan_every_n_tokens",
        "pdf_input_enabled", "pdf_extract_text_first", "pdf_dpi", "pdf_max_pages",
        "strata_vision", "strata_kv", "strata_low_ram", "strata_layer_split",
    })

    # ------------------------------------------------------------- paths/ids
    def owns_model_path(self, model_path: str | None) -> bool:
        # Only the ids: a downloaded shard file is a plain file (llama.cpp by
        # default) that Strata can run when the engine is picked for it.
        return parse_model_path(model_path) is not None and str(model_path).startswith(PATH_PREFIX)

    def file_model_id(self, path: str) -> str | None:
        parsed = model_for_file(path)
        return model_id_for(*parsed) if parsed else None

    def display_name(self, model_path: str) -> str:
        parsed = parse_model_path(model_path)
        return model_id_for(*parsed) if parsed else super().display_name(model_path)

    def virtual_models(self) -> list[dict]:
        out = []
        for m in catalogue():
            size_bytes = int(m["download_gb"] * 1e9)
            out.append({
                "name": m["id"],
                "path": m["path"],
                "type": "strata",
                "engine": self.name,
                "quant": m["size"],
                "size_bytes": size_bytes,
                "size_display": f"~{m['download_gb']:.0f} GB",
                "title": m["title"],
                "family": m["family"],
                "served_name": m["served_name"],
                "ram_gb": m["ram_gb"],
                "vision": m["vision"],
                "experimental": m["experimental"],
                "local_shards": local_shard_dir(m["family"], m["size"]) is not None,
                "download": self._download_summary(m["family"], m["size"]),
            })
        return out

    @staticmethod
    def _download_summary(family: str, size: str) -> dict | None:
        dl = model_download(family, size)
        return {"id": dl["id"], "status": dl.get("status")} if dl else None

    def canonical_model_path(self, name: str) -> str | None:
        parsed = parse_model_path(name)
        return model_path_for(*parsed) if parsed else None

    def download_plan(self, model_path: str) -> dict | None:
        parsed = parse_model_path(model_path)
        if not parsed:
            return None
        family, size = parsed
        repo = FAMILIES[family]["repo"]
        return {
            "repo_id": repo,
            # Shard 1: the downloader expands a split GGUF to all its shards.
            "filename": repo_files(family, size)[0],
            "revision": HF_REVISIONS[repo],
            "dest_path": shards_base_dir(family, size),
            "model_id": model_id_for(family, size),
            "download_gb": SIZES[size]["download_gb"],
        }

    def launch_blocker(self, model_path: str) -> str | None:
        """Strata runs only models llamaman has downloaded: never let the
        container fetch ~70 GB into its own volume. `model_path` is the
        file it is launched from, or a model id."""
        parsed = parse_model_path(model_path)
        if not parsed:
            return None
        dl = model_download(*parsed)
        if dl and dl.get("status") in ("downloading", "paused"):
            return (f"{model_id_for(*parsed)} is still being downloaded "
                    f"(download {dl['id'][:8]}, {dl['status']}); launch it when the download finishes")
        if model_for_file(model_path):
            missing = [os.path.basename(f) for f, _ in _shard_pairs(model_path, parsed)
                       if not os.path.isfile(f)]
            if missing:
                return (f"not every shard of {model_id_for(*parsed)} is next to "
                        f"{os.path.basename(model_path)} (missing {', '.join(missing)})")
        elif not local_shard_dir(*parsed):
            return (f"{model_id_for(*parsed)} is not downloaded: download it from Recommended "
                    f"models in the Strata settings, then launch it from the model library")
        return None

    def model_metadata(self, model_path: str) -> dict:
        parsed = parse_model_path(model_path)
        if not parsed:
            return {}
        family, size = parsed
        arch = FAMILIES[family]["served"]
        # context_length: what a launch without a preset gets
        # (docker-entrypoint.sh CONTEXT default); _effective_ctx_for_model
        # prefers a running instance's or the preset's ctx over this.
        return {
            "general.architecture": arch,
            "general.name": f"{FAMILIES[family]['title']} {size}",
            f"{arch}.context_length": DEFAULT_CONTEXT,
        }

    def served_model_names(self, model_path: str) -> list[str]:
        parsed = parse_model_path(model_path)
        if not parsed:
            return []
        family, size = parsed
        return [model_id_for(family, size).lower(),
                f"{FAMILIES[family]['served']}-{size.lower()}"]

    # ----------------------------------------------------------------- image
    def default_image(self) -> str:
        from config import STRATA_IMAGE
        return STRATA_IMAGE

    def image_missing_message(self, image_name: str) -> str:
        return (f"Docker image '{image_name}' not found. Strata is not published to a registry: "
                f"build it from https://github.com/Niko1221/Strata with "
                f"`docker build -t {image_name} .` (NVIDIA driver 580+ needed to run it)")

    # ------------------------------------------------------------- options
    def parse_options(self, body: dict, model_path: str) -> tuple[dict, str | None]:
        """Validate the strata_* launch fields. Returns (options, error)."""
        parsed = parse_model_path(model_path)
        if not parsed:
            return {}, f"'{model_path}' is not a Strata model (expected strata/<family>-<size>)"
        family, _ = parsed
        vision = str(body.get("strata_vision") or "no").strip().lower()
        if vision not in VISION_CHOICES:
            return {}, f"strata_vision must be one of {', '.join(VISION_CHOICES)}"
        if vision != "no" and not FAMILIES[family]["vision"]:
            return {}, f"{FAMILIES[family]['title']} has no image encoder; strata_vision must be 'no'"
        kv = str(body.get("strata_kv") or "").strip().lower()
        if kv not in KV_CHOICES:
            return {}, "strata_kv must be one of int8, q4_0, k8v4 (or empty for Strata's default)"
        low_ram = str(body.get("strata_low_ram") or "auto").strip().lower()
        if low_ram not in LOW_RAM_CHOICES:
            return {}, f"strata_low_ram must be one of {', '.join(LOW_RAM_CHOICES)}"
        split = str(body.get("strata_layer_split") or "").strip().lower().replace(" ", "")
        if split == "auto":
            split = ""
        if split and not re.fullmatch(r"\d+(,\d+)*", split):
            return {}, ("strata_layer_split must be empty (auto) or the layer each later GPU "
                        "starts at, e.g. 18 or 16,32")
        return {"strata_vision": vision, "strata_kv": kv, "strata_low_ram": low_ram,
                "strata_layer_split": split}, None

    def image_input_enabled(self, body: dict) -> bool | None:
        return str(body.get("strata_vision") or "no").strip().lower() != "no"

    # ------------------------------------------------------------ container
    def command(self, model_path: str, config: dict) -> list[str]:
        return []  # the entrypoint reads env vars only

    def _gpu_env(self, config: dict) -> dict:
        """GPU / GPUS for setup.py, in the CONTAINER's numbering.

        device_requests already limits the container to the chosen host GPUs,
        and the NVIDIA runtime renumbers them from 0 inside it - so a pin to
        host GPUs "1,3" is GPUS=0,1 here. Pinning even a single card keeps a
        volume set up on another host from being 'offered to the pair'
        (docker-entrypoint.sh comment on offer_together)."""
        from config import LLAMA_GPU_DEVICES
        effective = (config.get("gpu_devices") or LLAMA_GPU_DEVICES or "").strip()
        if not effective or effective.lower() == "all":
            return {}
        n = len([d for d in effective.split(",") if d.strip()])
        if n == 1:
            return {"GPU": "0"}
        return {"GPUS": ",".join(str(i) for i in range(n))}

    def _low_ram(self, config: dict) -> str:
        # setup.py reads /proc/meminfo, i.e. the HOST's RAM, so it cannot see
        # a cgroup memory limit: a capped container must ask for low-RAM mode.
        if config.get("memory_limit"):
            return "on"
        v = str(config.get("strata_low_ram") or "auto").lower()
        return v if v in LOW_RAM_CHOICES else "auto"

    def setup_settings(self, model_path: str, config: dict) -> dict:
        """The settings Strata bakes into /data/config/strata-<tag>.json."""
        family, size = parse_model_path(model_path)
        try:
            context = int(config.get("ctx_size") or DEFAULT_CONTEXT)
        except (TypeError, ValueError):
            context = DEFAULT_CONTEXT
        return {
            "family": family,
            "model": size,
            "context": context,
            "vision": config.get("strata_vision") or "no",
            "kv": config.get("strata_kv") or "",
            "low_ram": self._low_ram(config),
            "gpus": self._gpu_env(config),
        }

    def environment(self, model_path: str, config: dict) -> dict:
        s = self.setup_settings(model_path, config)
        env = {
            "FAMILY": s["family"],
            "MODEL": s["model"],
            "CONTEXT": str(s["context"]),
            "VISION": s["vision"],
            "HOST": "0.0.0.0",
            "PORT": str(self.internal_port),
            "LOW_RAM": s["low_ram"],
        }
        if s["kv"]:
            env["KV"] = s["kv"]
        env.update(s["gpus"])
        # Where each later card's layers start; only means something across
        # several cards. The entrypoint passes it at every start, so it is not
        # part of the setup fingerprint (no REINSTALL to change it).
        split = (config.get("strata_layer_split") or "").strip()
        if split and "GPUS" in s["gpus"]:
            env["LAYER_SPLIT"] = split
        if self.needs_reinstall(model_path, config):
            env["REINSTALL"] = "1"
        return env

    def container_spec(self, **kw) -> dict:
        kwargs = super().container_spec(**kw)
        kwargs.setdefault("environment", {})["STRATA_ALLOWED_HOSTS"] = ",".join(
            self.allowed_hosts(kw["container_name"]))
        return kwargs

    @staticmethod
    def allowed_hosts(container_name: str) -> list[str]:
        """Names Strata (without an API key) accepts in Host and Origin
        (serve/server.py host_allowed / origin_allowed): llamaman's own route
        to it (the container name on the Docker network, LLAMA_HOST_ADDR
        bare-metal) and the names a browser opens its web app by - the host
        of CLUSTER_ADVERTISE_URL and STRATA_WEB_HOSTS; the web app's chat
        requests carry the page's Origin. Read at every server start (no
        REINSTALL). Entries Strata would reject (it refuses to start on a
        malformed one) are dropped."""
        from urllib.parse import urlsplit
        from config import CLUSTER_ADVERTISE_URL, LLAMA_HOST_ADDR, STRATA_WEB_HOSTS
        candidates = [container_name, LLAMA_HOST_ADDR]
        if CLUSTER_ADVERTISE_URL:
            try:
                candidates.append(urlsplit(CLUSTER_ADVERTISE_URL).hostname or "")
            except ValueError:
                pass
        for raw in STRATA_WEB_HOSTS:
            # Like Strata: drop a scheme, port or path ("http://a.lan:12021/" -> "a.lan").
            x = raw.split("://", 1)[-1].split("/", 1)[0]
            candidates.append(x.split(":", 1)[0] if x.count(":") == 1 else x.strip("[]"))
        out = []
        for h in candidates:
            h = (h or "").strip().lower().rstrip(".")
            ok = bool(h) and (all(c.isalnum() or c in "-._" for c in h) or ":" in h)
            if ok and h not in out:
                out.append(h)
        return out

    def data_mount_source(self) -> str:
        from config import HOST_STRATA_DATA_DIR, STRATA_DATA_VOLUME
        return HOST_STRATA_DATA_DIR or STRATA_DATA_VOLUME

    def volumes(self, model_path: str, config: dict) -> dict:
        from config import HOST_LOGS_DIR, LOGS_DIR
        vols = {
            self.data_mount_source(): {"bind": "/data", "mode": "rw"},
            HOST_LOGS_DIR: {"bind": LOGS_DIR, "mode": "rw"},
        }
        family, size = parse_model_path(model_path)
        target = f"/data/models/{shards_dir_tag(family, size)}"
        # The file it was launched from (launch_instance), else llamaman's
        # own download folder for it.
        mounts = shard_mounts_for_file(config.get("engine_source_path"))
        if mounts and any(os.path.basename(f) != pub for f, pub in mounts):
            # Renamed copies: setup.py looks for the published names, so each
            # file is mounted under its own.
            for f, pub in mounts:
                vols[_host_path_for_models_subdir(f)] = {"bind": f"{target}/{pub}", "mode": "rw"}
            return vols
        shards = os.path.dirname(mounts[0][0]) if mounts else local_shard_dir(family, size)
        if shards:
            # rw: setup writes a <shard>.done mark beside each file it accepts.
            vols[_host_path_for_models_subdir(shards)] = {"bind": target, "mode": "rw"}
        return vols

    def ulimits(self, config: dict) -> list:
        import docker
        # The engine page-locks part of its 32-62 GB arena for GPU DMA;
        # experts it cannot pin go through slower host copies.
        return [docker.types.Ulimit(name="memlock", soft=-1, hard=-1)]

    def gpu_attachment(self, config: dict, vendor: str | None) -> str:
        return "nvidia"

    # ------------------------------------------------- setup fingerprinting
    def _setups_file(self) -> str:
        from config import DATA_DIR
        return os.path.join(DATA_DIR, "strata_setups.json")

    def _setup_key(self, model_path: str) -> str:
        family, size = parse_model_path(model_path)
        return f"{self.data_mount_source()}|{setup_tag(family, size)}"

    def _read_setups(self) -> dict:
        try:
            with open(self._setups_file()) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def recorded_setup(self, model_path: str) -> dict | None:
        return self._read_setups().get(self._setup_key(model_path))

    def needs_reinstall(self, model_path: str, config: dict) -> bool:
        """REINSTALL=1 unless these exact settings were last seen healthy on
        this data volume. Unknown (first launch through llamaman, a volume
        set up by hand) also reinstalls: setup then reuses the files already
        there and only rewrites the config, and a wrong context is worse than
        one extra setup pass."""
        return self.recorded_setup(model_path) != self.setup_settings(model_path, config)

    def on_ready(self, inst: dict) -> None:
        """Called once an instance is healthy: the config on the volume now
        matches these settings."""
        model_path = inst.get("model_path", "")
        if not parse_model_path(model_path):
            return
        settings = self.setup_settings(model_path, inst.get("config") or {})
        key = self._setup_key(model_path)
        with _SETUPS_LOCK:
            data = self._read_setups()
            if data.get(key) == settings:
                return
            data[key] = settings
            tmp = self._setups_file() + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=1, sort_keys=True)
                os.replace(tmp, self._setups_file())
            except OSError:
                pass

    # ------------------------------------------------------------ readiness
    def load_timeout(self) -> int:
        from config import STRATA_LOAD_TIMEOUT
        return STRATA_LOAD_TIMEOUT

    def load_stage(self, inst: dict) -> dict | None:
        log_file = inst.get("log_file")
        return load_stage_from_log(read_log_tail(log_file)) if log_file else None

    # --------------------------------------------------------- availability
    def availability(self, vendor: str | None) -> tuple[bool, str]:
        from config import STRATA_ENABLED
        if not STRATA_ENABLED:
            return False, "Strata is disabled on this node (set STRATA_ENABLED=true)"
        if vendor != "cuda":
            found = {"rocm": "an AMD GPU", "intel": "an Intel GPU",
                     "vulkan": "a Vulkan GPU"}.get(vendor or "", "no GPU")
            return False, (f"Strata needs an NVIDIA GPU (driver 580+); this node has {found}. "
                           "Strata's AMD support is an experimental manual build upstream.")
        return True, ""

    def describe(self, vendor: str | None) -> dict:
        d = super().describe(vendor)
        from config import STRATA_IMAGE
        d["image"] = STRATA_IMAGE
        d["data"] = self.data_mount_source()
        d["build_command"] = f"docker build -t {STRATA_IMAGE} ."
        d["memory_warn_gb"] = MEMORY_WARN_GB
        d["context_choices"] = list(CONTEXT_CHOICES)
        d["catalogue"] = catalogue_source()
        return d
