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
    /data/models/<tag>/<original name> (setup.py "#173: a whole file copied in
    by hand has no finish mark" - it checks them against their own tensor
    directory), so a model downloaded through llamaman's downloader into
    MODELS_DIR/strata/<tag>/ is bind-mounted there.

Each family x size is a virtual model with the path /strata/<family>-<SIZE>
(id strata/<family>-<SIZE>) and is its own llamaman instance.
"""

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
DEFAULT_CONTEXT = 32768
CONTEXT_CHOICES = (8192, 32768, 65536, 131072, 262144, 393216, 524288)
VISION_CHOICES = ("no", "yes", "cpu")
KV_CHOICES = ("", "int8", "q4_0", "k8v4")
LOW_RAM_CHOICES = ("auto", "on", "off")

# Below this a memory limit is very likely to starve even LOW_RAM=on (the UI
# warns; the API does not refuse - the operator may know their model).
MEMORY_WARN_GB = 64

_SETUPS_LOCK = threading.Lock()


def parse_model_path(model_path: str | None) -> tuple[str, str] | None:
    """'/strata/qwen-IQ2_XS' (or the id 'strata/qwen-IQ2_XS') ->
    ('qwen', 'IQ2_XS'). None when it isn't a Strata model in the catalogue.
    Family names have no '-', sizes may ('UD-Q4_K_XL'), so split once."""
    if not isinstance(model_path, str):
        return None
    p = model_path.strip()
    if p.startswith(PATH_PREFIX):
        rest = p[len(PATH_PREFIX):]
    elif p.lower().startswith(ID_PREFIX):
        rest = p[len(ID_PREFIX):]
    else:
        return None
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


def local_shard_dir(family: str, size: str) -> str | None:
    """The directory (as llamaman sees it) holding every shard of this model
    under MODELS_DIR/strata/<tag>/, or None. Looks in that folder and one
    level below it, because llamaman's downloader keeps a repo's size folder
    (e.g. .../iq2_xs/IQ2_XS/<shards>)."""
    from config import MODELS_DIR
    base = os.path.join(MODELS_DIR, "strata", setup_tag(family, size))
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
    option_keys = ("strata_vision", "strata_kv", "strata_low_ram")
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
        "strata_vision", "strata_kv", "strata_low_ram",
    })

    # ------------------------------------------------------------- paths/ids
    def owns_model_path(self, model_path: str | None) -> bool:
        return parse_model_path(model_path) is not None and str(model_path).startswith(PATH_PREFIX)

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
            })
        return out

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
        return {"strata_vision": vision, "strata_kv": kv, "strata_low_ram": low_ram}, None

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
        if self.needs_reinstall(model_path, config):
            env["REINSTALL"] = "1"
        return env

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
        shards = local_shard_dir(family, size)
        if shards:
            # rw: setup writes a <shard>.done mark beside each file it accepts.
            vols[_host_path_for_models_subdir(shards)] = {
                "bind": f"/data/models/{setup_tag(family, size)}", "mode": "rw",
            }
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
        return d
