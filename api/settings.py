# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import tempfile
import time
import uuid

from flask import Blueprint, jsonify, request

from config import DATA_DIR, logger
from storage import get_storage

_SUBPROCESS_SETTINGS_FILE = os.path.join(DATA_DIR, "subprocess_settings.json")


def snapshot_subprocess_settings() -> None:
    """Mirror settings that subprocesses (downloader, etc.) need to a small
    file that stays consistent regardless of which storage backend is canonical.

    Subprocesses poll this file every second. Keeping it backend-agnostic means
    live updates work identically on JSON and MariaDB.
    """
    try:
        settings = get_storage().get_settings()
    except Exception as e:
        logger.warning("subprocess_settings snapshot: failed to read settings: %s", e)
        return

    payload = {
        "global_speed_limit_mbps": float(settings.get("global_speed_limit_mbps", 0) or 0),
    }

    try:
        dir_name = os.path.dirname(_SUBPROCESS_SETTINGS_FILE) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, _SUBPROCESS_SETTINGS_FILE)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("subprocess_settings snapshot: failed to write: %s", e)

bp = Blueprint("settings", __name__)

DEFAULT_GLOBAL_SPEED_LIMIT_MBPS = 0.0
DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD = 3
DEFAULT_RECORDING_MODE = "off"
VALID_RECORDING_MODES = ("off", "per_request", "per_conversation")
DEFAULT_RECORDING_RETENTION_DAYS = 30
DEFAULT_CLUSTER_HIDE_OFFLINE_MONITORING = False


def _get_hf_tokens() -> list[dict]:
    settings = get_storage().get_settings()
    tokens = settings.get("huggingface_tokens", [])
    return [t for t in tokens if isinstance(t, dict) and t.get("id")]


def _save_hf_tokens(tokens: list[dict]) -> None:
    storage = get_storage()
    settings = storage.get_settings()
    settings["huggingface_tokens"] = tokens
    storage.save_settings(settings)


def _mask_hf_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 10:
        return token[:3] + "..." + token[-2:]
    return token[:6] + "..." + token[-4:]


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_non_negative_float(value, default: float = 0.0) -> float:
    try:
        coerced = float(value or 0)
    except (TypeError, ValueError):
        return default
    return max(0.0, coerced)


def _coerce_min_int(value, default: int, minimum: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, coerced)


def _coerce_recording_mode(value, default: str = DEFAULT_RECORDING_MODE) -> str:
    if isinstance(value, str) and value in VALID_RECORDING_MODES:
        return value
    return default


def _normalize_settings_patch(settings: dict) -> dict:
    normalized = dict(settings)
    if "global_speed_limit_mbps" in normalized:
        normalized["global_speed_limit_mbps"] = _coerce_non_negative_float(
            normalized.get("global_speed_limit_mbps"),
            default=DEFAULT_GLOBAL_SPEED_LIMIT_MBPS,
        )
    if "auto_retry_failed_downloads" in normalized:
        normalized["auto_retry_failed_downloads"] = _coerce_bool(
            normalized.get("auto_retry_failed_downloads"),
            default=False,
        )
    if "retry_count_per_failed_download" in normalized:
        normalized["retry_count_per_failed_download"] = _coerce_min_int(
            normalized.get("retry_count_per_failed_download"),
            default=DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD,
            minimum=1,
        )
    if "recording_mode" in normalized:
        normalized["recording_mode"] = _coerce_recording_mode(
            normalized.get("recording_mode"),
            default=DEFAULT_RECORDING_MODE,
        )
    if "recording_retention_days" in normalized:
        normalized["recording_retention_days"] = _coerce_min_int(
            normalized.get("recording_retention_days"),
            default=DEFAULT_RECORDING_RETENTION_DAYS,
            minimum=0,
        )
    if "cluster_hide_offline_monitoring" in normalized:
        normalized["cluster_hide_offline_monitoring"] = _coerce_bool(
            normalized.get("cluster_hide_offline_monitoring"),
            default=DEFAULT_CLUSTER_HIDE_OFFLINE_MONITORING,
        )
    return normalized


def _apply_settings_defaults(settings: dict) -> dict:
    normalized = dict(settings)
    normalized["global_speed_limit_mbps"] = _coerce_non_negative_float(
        normalized.get("global_speed_limit_mbps"),
        default=DEFAULT_GLOBAL_SPEED_LIMIT_MBPS,
    )
    normalized["auto_retry_failed_downloads"] = _coerce_bool(
        normalized.get("auto_retry_failed_downloads"),
        default=False,
    )
    normalized["retry_count_per_failed_download"] = _coerce_min_int(
        normalized.get("retry_count_per_failed_download"),
        default=DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD,
        minimum=1,
    )
    normalized["recording_mode"] = _coerce_recording_mode(
        normalized.get("recording_mode"),
        default=DEFAULT_RECORDING_MODE,
    )
    normalized["recording_retention_days"] = _coerce_min_int(
        normalized.get("recording_retention_days"),
        default=DEFAULT_RECORDING_RETENTION_DAYS,
        minimum=0,
    )
    normalized["cluster_hide_offline_monitoring"] = _coerce_bool(
        normalized.get("cluster_hide_offline_monitoring"),
        default=DEFAULT_CLUSTER_HIDE_OFFLINE_MONITORING,
    )
    return normalized


def serialize_hf_token(token_entry: dict) -> dict:
    return {
        "id": token_entry["id"],
        "name": token_entry.get("name", "Untitled"),
        "preview": _mask_hf_token(token_entry.get("token", "")),
        "created_at": token_entry.get("created_at", 0),
    }


def get_hf_token_secret(token_id: str) -> str | None:
    for token in _get_hf_tokens():
        if token.get("id") == token_id:
            secret = token.get("token", "").strip()
            return secret or None
    return None


def _sanitize_settings(settings: dict) -> dict:
    from core.node_settings import NODE_SCOPED_KEYS, effective_from_settings
    safe = _apply_settings_defaults(settings)
    if "huggingface_tokens" in safe:
        safe["huggingface_tokens"] = [serialize_hf_token(token) for token in _get_hf_tokens()]
    # Surface the local node's value for per-node keys the UI reads from here
    # (the eviction toggles); never leak the raw per-node namespace.
    for key in ("admin_ui_enforce_max_models", "allow_ollama_api_override_admin"):
        safe[key] = bool(effective_from_settings(settings, key, safe.get(key, False)))
    safe.pop("nodes", None)
    return safe


@bp.route("/api/settings")
def get_settings():
    return jsonify(_sanitize_settings(get_storage().get_settings()))


@bp.route("/api/settings", methods=["POST"])
def save_settings():
    from core.node_settings import NODE_SCOPED_KEYS, merge_node_settings
    data = request.get_json(silent=True) or {}
    data.pop("huggingface_tokens", None)
    # Per-node keys (eviction toggles, docker_images) go to this node's
    # namespace; everything else stays shared cluster-wide.
    node_patch = {k: data.pop(k) for k in list(data.keys()) if k in NODE_SCOPED_KEYS}
    if node_patch:
        merge_node_settings(node_patch)
    if data:
        settings = get_storage().merge_settings(_normalize_settings_patch(data))
    else:
        settings = get_storage().get_settings()
    if "recording_mode" in data:
        from core.request_log import invalidate_cache as _invalidate_recording_cache
        _invalidate_recording_cache()
    if "global_speed_limit_mbps" in data:
        snapshot_subprocess_settings()
    return jsonify({"ok": True, "settings": _sanitize_settings(settings)})


@bp.route("/api/settings/huggingface-tokens")
def list_huggingface_tokens():
    return jsonify([serialize_hf_token(token) for token in _get_hf_tokens()])


@bp.route("/api/settings/huggingface-tokens", methods=["POST"])
def create_huggingface_token():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip() or "Untitled"
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"error": "token is required"}), 400

    tokens = _get_hf_tokens()
    entry = {
        "id": uuid.uuid4().hex,
        "name": name,
        "token": token,
        "created_at": int(time.time()),
    }
    tokens.append(entry)
    _save_hf_tokens(tokens)
    return jsonify(serialize_hf_token(entry)), 201


@bp.route("/api/settings/huggingface-tokens/<token_id>", methods=["DELETE"])
def delete_huggingface_token(token_id):
    tokens = _get_hf_tokens()
    kept = [token for token in tokens if token.get("id") != token_id]
    if len(kept) == len(tokens):
        return jsonify({"error": "Not found"}), 404
    _save_hf_tokens(kept)
    return jsonify({"ok": True})
