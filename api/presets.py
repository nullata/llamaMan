# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, request

from core.proxy_sampling import parse_proxy_sampling_config
from core.state import instances, instances_lock, save_state
from storage import get_storage

bp = Blueprint("presets", __name__)

# Hardware fields that may be overridden per node (everything else in a preset
# is shared cluster-wide). A node's overrides live in preset["node_overrides"].
PRESET_HARDWARE_KEYS = ("n_gpu_layers", "threads", "memory_limit", "gpu_devices", "parallel")


def resolve_preset_for_node(preset: dict | None, node_id: str) -> dict | None:
    """Overlay a node's hardware overrides onto the shared base preset."""
    if not preset:
        return preset
    overrides = (preset.get("node_overrides") or {}).get(node_id)
    if not overrides:
        return preset
    merged = dict(preset)
    for key in PRESET_HARDWARE_KEYS:
        if key in overrides and overrides[key] is not None:
            merged[key] = overrides[key]
    return merged


def _normalize_model_path(model_path: str) -> str:
    """Ensure model_path is an absolute path (leading /).

    Flask's <path:> converter strips the leading / from the URL, so
    /api/presets/models/foo.gguf yields model_path='models/foo.gguf'
    but the storage key is '/models/foo.gguf'.
    """
    if not model_path.startswith("/"):
        model_path = "/" + model_path
    return model_path


@bp.route("/api/presets", methods=["GET"])
def api_presets_list():
    return jsonify(get_storage().get_all_presets())


@bp.route("/api/presets/<path:model_path>", methods=["GET"])
def api_preset_get(model_path):
    model_path = _normalize_model_path(model_path)
    preset = get_storage().get_preset(model_path)
    if preset is None:
        return jsonify({"error": "No preset for this model"}), 404
    return jsonify(preset)


@bp.route("/api/presets/<path:model_path>", methods=["PUT"])
def api_preset_save(model_path):
    model_path = _normalize_model_path(model_path)
    body = request.get_json(force=True)
    ctx_size = body.get("ctx_size")
    if ctx_size in (None, ""):
        return jsonify({"error": "ctx_size is required"}), 400
    try:
        ctx_size = int(ctx_size)
    except (TypeError, ValueError):
        return jsonify({"error": "ctx_size must be an integer"}), 400
    if ctx_size <= 0:
        return jsonify({"error": "ctx_size must be greater than 0"}), 400
    proxy_sampling_config, proxy_sampling_err = parse_proxy_sampling_config(body)
    if proxy_sampling_err:
        return jsonify({"error": proxy_sampling_err}), 400
    # Preserve existing meta fields (favorite, note) that aren't part of the launch form
    existing = get_storage().get_preset(model_path) or {}
    if not isinstance(existing, dict):
        existing = {}
    # Group/fallback are meaningless without share_queue; drop them so a stale
    # value can't leak into the preset and surface on the next launch.
    share_queue_on = bool(body.get("share_queue", False))
    data = {
        "n_gpu_layers": body.get("n_gpu_layers", -1),
        "ctx_size": ctx_size,
        "threads": body.get("threads"),
        "memory_limit": body.get("memory_limit", ""),
        "parallel": body.get("parallel"),
        "extra_args": body.get("extra_args", ""),
        "spec_enabled": body.get("spec_enabled", False),
        "spec_draft_n_max": body.get("spec_draft_n_max"),
        "gpu_devices": body.get("gpu_devices", ""),
        "idle_timeout_min": body.get("idle_timeout_min", 0),
        "max_concurrent": body.get("max_concurrent", 0),
        "max_queue_depth": body.get("max_queue_depth", 200),
        "share_queue": share_queue_on,
        # Cluster: alias-based group key + fallback role. Normalized at the
        # boundary so cluster matching (lowercased) stays consistent. Empty
        # group = legacy "group by filename".
        "share_queue_group": (body.get("share_queue_group") or "").strip().lower() if share_queue_on else "",
        "share_queue_fallback": bool(body.get("share_queue_fallback", False)) if share_queue_on else False,
        "embedding_model": body.get("embedding_model", False),
        "auto_restart_on_crash": body.get("auto_restart_on_crash", False),
        "favorite": body.get("favorite", existing.get("favorite", False)),
        "note": body.get("note", existing.get("note", "")),
        **proxy_sampling_config,
    }

    # Cluster: when a target node is named, the form's hardware fields are that
    # node's override; the shared base hardware is kept from the existing preset.
    node_overrides = dict(existing.get("node_overrides", {}))
    override_node_id = (body.get("override_node_id") or "").strip()
    if override_node_id:
        node_overrides[override_node_id] = {k: body.get(k) for k in PRESET_HARDWARE_KEYS}
        for key in PRESET_HARDWARE_KEYS:
            if key in existing:
                data[key] = existing[key]  # don't let an override edit move the base
    if node_overrides:
        data["node_overrides"] = node_overrides

    get_storage().save_preset(model_path, data)
    _apply_live_preset_changes(model_path, data)
    return jsonify({"status": "saved"})


_LIVE_PROXY_SAMPLING_FIELDS = (
    "proxy_sampling_override_enabled",
    "proxy_sampling_temperature",
    "proxy_sampling_top_k",
    "proxy_sampling_top_p",
    "proxy_sampling_presence_penalty",
    "proxy_sampling_repeat_penalty",
)


def _apply_live_preset_changes(model_path: str, preset: dict) -> None:
    """Update fields that take effect on a running instance without relaunch:
    the reaper re-reads idle_timeout_min each tick, refresh_gate picks up
    queue changes, and the proxy + Ollama/OpenAI compat layers read the
    proxy_sampling_* fields from inst["config"] per request. Everything else
    (gpu layers, ctx size, threads, ...) is baked into the container at launch.

    Caveat for proxy_sampling toggles: if the instance was launched with all
    of idle_timeout=0, max_concurrent=0, and override_enabled=False, no
    sidecar proxy was spawned, so direct hits to the public port bypass the
    override even after a live toggle. Compat routes still apply it. A
    relaunch is required to spawn the proxy in that case."""
    from proxy import refresh_gate

    touched = []
    with instances_lock:
        for inst in instances.values():
            if inst.get("model_path") != model_path or inst.get("status") == "stopped":
                continue
            config = inst.setdefault("config", {})
            config["idle_timeout_min"] = preset.get("idle_timeout_min", 0)
            config["max_concurrent"] = preset.get("max_concurrent", 0)
            config["max_queue_depth"] = preset.get("max_queue_depth", 200)
            config["share_queue"] = preset.get("share_queue", False)
            # share_queue_group propagates live for routing purposes, but
            # llama-server was launched with the OLD --alias (or none); direct
            # hits to the instance port still advertise the old name until
            # relaunch. Routing through the cluster/compat layer uses the
            # live value, so the inconsistency is cosmetic. share_queue_fallback
            # is pure routing policy, fully live.
            config["share_queue_group"] = (preset.get("share_queue_group") or "").strip().lower()
            config["share_queue_fallback"] = bool(preset.get("share_queue_fallback", False))
            config["auto_restart_on_crash"] = preset.get("auto_restart_on_crash", False)
            for f in _LIVE_PROXY_SAMPLING_FIELDS:
                if f in preset:
                    config[f] = preset[f]
            touched.append(inst["id"])

    for inst_id in touched:
        refresh_gate(inst_id)

    if touched:
        save_state()


@bp.route("/api/presets/<path:model_path>", methods=["PATCH"])
def api_preset_patch(model_path):
    """Partially update preset fields (e.g. favorite, note) without requiring a full preset."""
    model_path = _normalize_model_path(model_path)
    body = request.get_json(force=True)
    storage = get_storage()
    preset = storage.get_preset(model_path) or {}
    allowed = {"favorite", "note"}
    for key in allowed:
        if key in body:
            preset[key] = body[key]
    storage.save_preset(model_path, preset)
    return jsonify({"status": "saved"})


@bp.route("/api/presets/<path:model_path>", methods=["DELETE"])
def api_preset_delete(model_path):
    model_path = _normalize_model_path(model_path)
    get_storage().delete_preset(model_path)
    return jsonify({"status": "deleted"})
