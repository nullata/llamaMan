# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import re
import shutil
import struct
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from api.settings import get_hf_token_secret
from config import MODELS_DIR
from core.helpers import format_size
from core.model_sources import get_model_sources, remove_model_sources_for_path, resolve_model_source_repo_id
from core.state import instances, instances_lock
from storage import get_storage

bp = Blueprint("models", __name__)

_QUANT_PATTERN = re.compile(
    r'(?i)(bf16|f16|f32|q[0-9]_[0-9]|q[0-9]+_k(?:_[sml])?|iq[0-9]+_[a-z]+|q[0-9]+)',
)


def detect_quant(name: str) -> str:
    m = _QUANT_PATTERN.search(name)
    return m.group(1).upper() if m else ""


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def discover_models(models_dir: str) -> list[dict]:
    found = []
    base = Path(models_dir)
    if not base.exists():
        return found

    for config_file in base.rglob("config.json"):
        model_dir = config_file.parent
        size = _dir_size(model_dir)
        found.append({
            "name": model_dir.name,
            "path": str(model_dir),
            "type": "hf",
            "quant": "",
            "size_bytes": size,
            "size_display": format_size(size),
        })

    for gguf_file in base.rglob("*.gguf"):
        size = gguf_file.stat().st_size
        found.append({
            "name": gguf_file.stem,
            "path": str(gguf_file),
            "type": "gguf",
            "quant": detect_quant(gguf_file.stem),
            "size_bytes": size,
            "size_display": format_size(size),
        })

    seen = set()
    unique = []
    for m in found:
        if m["path"] not in seen:
            seen.add(m["path"])
            unique.append(m)
    return unique


def attach_model_sources(models: list[dict], sources: dict[str, str]) -> list[dict]:
    enriched = []
    for model in models:
        enriched_model = dict(model)
        repo_id = resolve_model_source_repo_id(enriched_model["path"], sources)
        if repo_id:
            enriched_model["repo_id"] = repo_id
        enriched.append(enriched_model)
    return enriched


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

def _read_gguf_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    if vtype == 1: return struct.unpack("<b", f.read(1))[0]
    if vtype == 2: return struct.unpack("<H", f.read(2))[0]
    if vtype == 3: return struct.unpack("<h", f.read(2))[0]
    if vtype == 4: return struct.unpack("<I", f.read(4))[0]
    if vtype == 5: return struct.unpack("<i", f.read(4))[0]
    if vtype == 6: return struct.unpack("<f", f.read(4))[0]
    if vtype == 7: return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8: return _read_gguf_string(f)
    if vtype == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11: return struct.unpack("<q", f.read(8))[0]
    if vtype == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unknown GGUF type: {vtype}")


_GGUF_FIXED_WIDTHS = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


def _skip_gguf_value(f, vtype: int) -> None:
    """Advance past a GGUF value without materializing it. Used to step over
    huge tokenizer arrays (tokens/merges/scores) so trailing scalar keys
    (bos/eos token ids, chat_template) remain reachable."""
    width = _GGUF_FIXED_WIDTHS.get(vtype)
    if width is not None:
        f.read(width)
        return
    if vtype == 8:  # string
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(length)
        return
    if vtype == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        elem_width = _GGUF_FIXED_WIDTHS.get(elem_type)
        if elem_width is not None:
            f.read(elem_width * count)
        else:
            for _ in range(count):
                _skip_gguf_value(f, elem_type)
        return
    raise ValueError(f"Unknown GGUF type: {vtype}")


def get_gguf_full_metadata(filepath: str) -> dict:
    """Read all scalar/string GGUF metadata key-value pairs into a flat dict.

    Tokenizer array values (tokens/merges/scores/token_type) are skipped
    without being parsed into Python lists - they can be 100k+ entries and
    are not useful for our /api/ps and /api/show responses. Other tokenizer
    scalars (bos_token_id, eos_token_id, chat_template, ...) are kept.
    """
    out: dict = {}
    try:
        with open(filepath, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            struct.unpack("<I", f.read(4))   # version
            struct.unpack("<Q", f.read(8))   # tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key = _read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 9 and key.startswith("tokenizer."):
                    try:
                        _skip_gguf_value(f, vtype)
                    except Exception:
                        break
                    continue
                try:
                    out[key] = _read_gguf_value(f, vtype)
                except Exception:
                    break
    except Exception:
        pass
    return out


_gguf_cache_lock = threading.Lock()
_gguf_cache: dict[str, tuple[float, int, dict]] = {}


def get_cached_gguf_metadata(filepath: str) -> dict:
    """Cached wrapper for get_gguf_full_metadata. Cache entries auto-invalidate
    when the file's mtime or size changes."""
    try:
        st = os.stat(filepath)
    except OSError:
        return {}
    mtime, size = st.st_mtime, st.st_size
    with _gguf_cache_lock:
        cached = _gguf_cache.get(filepath)
        if cached and cached[0] == mtime and cached[1] == size:
            return cached[2]
    meta = get_gguf_full_metadata(filepath)
    with _gguf_cache_lock:
        _gguf_cache[filepath] = (mtime, size, meta)
    return meta


def get_gguf_metadata(filepath: str) -> dict:
    """Architecture metadata used by the layer-fit calculator. Backed by the
    cached full reader so /api/model-layers and /api/ps share one parse."""
    full = get_cached_gguf_metadata(filepath)
    arch = full.get("general.architecture", "") or ""

    def _intish(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "block_count": _intish(full.get(f"{arch}.block_count") if arch else None),
        "embedding_length": _intish(full.get(f"{arch}.embedding_length") if arch else None),
        "feed_forward_length": _intish(full.get(f"{arch}.feed_forward_length") if arch else None),
        "head_count": _intish(full.get(f"{arch}.attention.head_count") if arch else None),
        "head_count_kv": _intish(full.get(f"{arch}.attention.head_count_kv") if arch else None),
        "vocab_size": _intish(full.get(f"{arch}.vocab_size") if arch else None),
    }


def format_param_count(n: int) -> str:
    """Render a parameter count the way Ollama's `parameter_size` displays it."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.1f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    return f"{n}"


def estimate_model_vram(size_bytes: int, n_gpu_layers: int, block_count: int | None) -> int:
    """Approximate the VRAM footprint of a model based on how many layers were
    requested on GPU. -1 (default) means "all"; 0 means CPU-only. For partial
    offload we scale by block_count when known. Excludes KV cache - this is
    weight-only, the same convention Ollama's size_vram uses."""
    if n_gpu_layers == 0:
        return 0
    if n_gpu_layers == -1 or not block_count or n_gpu_layers >= block_count:
        return int(size_bytes)
    return int(size_bytes * n_gpu_layers / block_count)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/models")
def api_models():
    models = discover_models(MODELS_DIR)
    models = attach_model_sources(models, get_model_sources(get_storage().get_settings()))
    return jsonify(models)


def _resolve_model_arg(model_path: str) -> tuple[str, str | None, int]:
    """Validate a caller-supplied model path: absolute, inside MODELS_DIR, and
    present on disk. Same containment rule the delete endpoint enforces."""
    if not model_path:
        return "", "path is required", 400
    try:
        resolved = os.path.realpath(model_path)
        models_real = os.path.realpath(MODELS_DIR)
        if not resolved.startswith(models_real + os.sep) and resolved != models_real:
            return "", "path is outside models directory", 403
    except Exception:
        return "", "invalid path", 400
    if not os.path.isfile(resolved):
        return "", "model file does not exist", 404
    return resolved, None, 200


def _repo_id_for(model_path: str) -> str:
    return resolve_model_source_repo_id(
        model_path, get_model_sources(get_storage().get_settings()))


@bp.route("/api/models/update-check", methods=["POST"])
def api_models_update_check():
    """Report whether a model's source repo has republished the file."""
    body = request.get_json(force=True)
    resolved, err, code = _resolve_model_arg(body.get("path", "").strip())
    if err:
        return jsonify({"error": err}), code

    token = body.get("hf_token", "").strip()
    token_id = body.get("hf_token_id", "").strip()
    if token_id:
        token = get_hf_token_secret(token_id) or ""

    from core.model_updates import check_model_update
    result = check_model_update(resolved, _repo_id_for(resolved), token or None)
    result["path"] = resolved
    return jsonify(result)


@bp.route("/api/models/verify-hash", methods=["GET", "POST"])
def api_models_verify_hash():
    """Hash the local model file so it can be compared exactly to the published
    one, for models downloaded before hashes were recorded.

    POST starts the job (or rejoins one already running) and returns immediately
    - the read takes minutes on a large model, so it never blocks the request.
    GET only reports state and never starts anything, which is what polling and
    model-selection use: asking about a model must not kick off work on it.

    Jobs are keyed by model path and outlive the page, so switching models (or
    reloading) leaves a running hash alone and you can come back to its progress.
    """
    if request.method == "GET":
        raw_path = request.args.get("path", "").strip()
    else:
        raw_path = (request.get_json(force=True) or {}).get("path", "").strip()

    resolved, err, code = _resolve_model_arg(raw_path)
    if err:
        return jsonify({"error": err}), code

    from core.model_updates import local_hash_state, start_local_hash

    if request.method == "GET":
        state = local_hash_state(resolved) or {"status": "none"}
    else:
        state = start_local_hash(resolved)
    state["path"] = resolved
    return jsonify(state)


@bp.route("/api/models/update", methods=["POST"])
def api_models_update():
    """Re-pull a model whose repo republished it, replacing the local copy.

    Downloads into a staging dir nested in the model's own directory and swaps
    on success (handled by the poller), so a failed or interrupted pull never
    leaves the live model damaged.
    """
    body = request.get_json(force=True)
    resolved, err, code = _resolve_model_arg(body.get("path", "").strip())
    if err:
        return jsonify({"error": err}), code

    repo_id = _repo_id_for(resolved)
    if not repo_id:
        return jsonify({"error": "no source repository recorded for this model"}), 400

    # Replacing bytes under a loaded model would leave llama-server reading a
    # file that no longer matches what it mapped. Mirrors the delete endpoint.
    with instances_lock:
        for inst in instances.values():
            if inst["status"] in ("stopped",):
                continue
            if os.path.realpath(inst["model_path"]) == resolved:
                return jsonify({
                    "error": f"model is in use by instance on port {inst['port']} - stop it first"
                }), 409

    token = body.get("hf_token", "").strip()
    token_id = body.get("hf_token_id", "").strip()
    if token_id:
        token = get_hf_token_secret(token_id)
        if not token:
            return jsonify({"error": "Saved Hugging Face token not found"}), 400

    from core.downloader import list_repo_files, resolve_filename
    from core.model_updates import cleanup_update_temp, update_temp_dir

    try:
        repo_files = list_repo_files(repo_id, token or None)
    except Exception as e:
        return jsonify({"error": f"Could not list files in {repo_id}: {e}"}), 502
    try:
        targets = resolve_filename(os.path.basename(resolved), repo_files, rid=repo_id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    # Clear any leftovers from an abandoned attempt so a stale partial file
    # can't be swapped in as if it were this update's result.
    temp_dir = update_temp_dir(resolved)
    cleanup_update_temp(temp_dir)

    from api.downloads import start_update_download
    dl, err = start_update_download(
        repo_id=repo_id,
        filename=targets[0]["name"],
        temp_dir=temp_dir,
        model_path=resolved,
        expected_sha=targets[0].get("sha256", ""),
        token=token or "",
        token_id=token_id,
        per_model_mbps=float(body.get("speed_limit_mbps", 0) or 0),
    )
    if err:
        cleanup_update_temp(temp_dir)
        return jsonify({"error": err}), 500

    from core.helpers import public_dict
    return jsonify(public_dict(dl)), 201


@bp.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    body = request.get_json(force=True)
    model_path = body.get("path", "").strip()
    if not model_path:
        return jsonify({"error": "path is required"}), 400

    try:
        resolved = os.path.realpath(model_path)
        models_real = os.path.realpath(MODELS_DIR)
        if not resolved.startswith(models_real + os.sep) and resolved != models_real:
            return jsonify({"error": "path is outside models directory"}), 403
    except Exception:
        return jsonify({"error": "invalid path"}), 400

    if not os.path.exists(resolved):
        return jsonify({"error": "path does not exist"}), 404

    with instances_lock:
        for inst in instances.values():
            if inst["status"] in ("stopped",):
                continue
            if os.path.realpath(inst["model_path"]) == resolved or \
               resolved.startswith(os.path.realpath(inst["model_path"]) + os.sep):
                return jsonify({"error": f"model is in use by instance on port {inst['port']}"}), 409

    # Same idea as the in-use check above: deleting the file out from under a
    # running hash leaves the worker reading a file that's gone. It survives
    # that (the error is caught and reported), but the user gets a confusing
    # failure for something they caused, so refuse while it's in flight.
    from core.model_updates import local_hash_state
    hash_job = local_hash_state(resolved) or {}
    if hash_job.get("status") == "hashing":
        return jsonify({
            "error": "model is being hashed right now - wait for it to finish"
        }), 409

    try:
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
        else:
            os.remove(resolved)
            parent = os.path.dirname(resolved)
            if parent != models_real and os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    from config import logger
    remove_model_sources_for_path(resolved)
    logger.info("Deleted model: %s", resolved)
    return jsonify({"status": "deleted"})


@bp.route("/api/model-layers")
def api_model_layers():
    model_path = request.args.get("path", "").strip()
    if not model_path:
        return jsonify({"error": "path is required"}), 400
    if not model_path.lower().endswith(".gguf"):
        return jsonify({"layers": None})
    meta = get_gguf_metadata(model_path)
    meta["layers"] = meta["block_count"]  # backward-compat alias
    meta["quant"] = detect_quant(Path(model_path).stem)
    return jsonify(meta)


@bp.route("/api/disk-space")
def api_disk_space():
    try:
        usage = shutil.disk_usage(MODELS_DIR)
        return jsonify({
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
