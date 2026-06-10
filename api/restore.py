# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import time
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request

from api.downloads import _spawn_download_process
from api.settings import get_hf_token_secret
from config import MODELS_DIR, logger
from core.model_sources import record_model_source
from core.state import downloads, downloads_lock, save_state
from storage import get_storage

bp = Blueprint("restore", __name__)


def _models_by_name() -> dict[str, str]:
    """Return {name: path} for all GGUF files and HF model dirs in MODELS_DIR."""
    found = {}
    base = Path(MODELS_DIR)
    if not base.exists():
        return found

    for config_file in base.rglob("config.json"):
        model_dir = config_file.parent
        found[model_dir.name] = str(model_dir)

    for gguf_file in base.rglob("*.gguf"):
        found[gguf_file.stem] = str(gguf_file)

    return found


def _queue_download(repo_id: str, filename: str, token: str, token_id: str) -> tuple[dict | None, str | None]:
    """Start a download and register it in the downloads dict. Returns (dl, error)."""
    dest_name = Path(filename).stem if filename else repo_id.split("/")[-1]
    dest_path = os.path.join(MODELS_DIR, dest_name)
    os.makedirs(dest_path, exist_ok=True)
    model_path = os.path.join(dest_path, filename) if filename else dest_path

    dl_id = str(uuid.uuid4())
    try:
        proc, log_fh, log_file = _spawn_download_process(
            dl_id, repo_id, dest_path, filename, token, 0, log_mode="w",
        )
    except Exception as e:
        return None, str(e)

    dl = {
        "id": dl_id,
        "repo_id": repo_id,
        "filename": filename,
        "dest_path": dest_path,
        "status": "downloading",
        "pid": proc.pid,
        "log_file": log_file,
        "started_at": time.time(),
        "_hf_token": token,
        "_hf_token_id": token_id,
        "per_model_speed_limit_mbps": 0,
        "retry_attempts": 0,
        "_process": proc,
        "_log_fh": log_fh,
    }

    with downloads_lock:
        downloads[dl_id] = dl

    record_model_source(dest_path, repo_id, model_path=model_path)
    return dl, None


def _expected_model_path(entry: dict) -> str:
    """Derive the model path that a download would produce for this entry."""
    name = entry.get("name", "")
    entry_type = entry.get("type", "gguf")
    if entry_type == "hf":
        return os.path.join(MODELS_DIR, name)
    return os.path.join(MODELS_DIR, name, f"{name}.gguf")


@bp.route("/api/restore", methods=["POST"])
def api_restore():
    entries = request.get_json(force=True)
    if not isinstance(entries, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    storage = get_storage()
    results = []
    present = queued = missing = errors = 0

    existing = _models_by_name()

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        name = (entry.get("name") or "").strip()
        if not name:
            continue

        repo_id = (entry.get("repo_id") or "").strip()
        entry_type = entry.get("type", "gguf")
        preset = entry.get("preset")

        # Already on disk
        if name in existing:
            actual_path = existing[name]
            if preset:
                try:
                    existing_preset = storage.get_preset(actual_path) or {}
                    merged = {**preset, **existing_preset}  # existing values win
                    storage.save_preset(actual_path, merged)
                except Exception:
                    pass
            results.append({"name": name, "status": "present"})
            present += 1
            continue

        # Not on disk - need a source to download
        if not repo_id:
            results.append({"name": name, "status": "missing"})
            missing += 1
            continue

        # Resolve HF token
        token = ""
        token_id = (entry.get("hf_token_id") or "").strip()
        if token_id:
            token = get_hf_token_secret(token_id) or ""

        filename = f"{name}.gguf" if entry_type != "hf" else ""

        dl, err = _queue_download(repo_id, filename, token, token_id)
        if err:
            logger.warning("Restore: failed to queue %s: %s", name, err)
            results.append({"name": name, "status": "error", "error": err})
            errors += 1
            continue

        # Pre-populate preset at the expected post-download path
        if preset:
            try:
                expected_path = _expected_model_path(entry)
                storage.save_preset(expected_path, preset)
            except Exception:
                pass

        logger.info("Restore: queued download for %s from %s (dl_id=%s)", name, repo_id, dl["id"])
        results.append({"name": name, "status": "queued", "dl_id": dl["id"]})
        queued += 1

    save_state()

    return jsonify({
        "results": results,
        "summary": {
            "present": present,
            "queued": queued,
            "missing": missing,
            "errors": errors,
        },
    })
