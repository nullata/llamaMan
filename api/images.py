# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import threading
import time

from flask import Blueprint, jsonify, request

from config import LLAMA_IMAGE, logger
from core.node_settings import get_effective_setting, merge_node_settings
from storage import get_storage

bp = Blueprint("images", __name__)


def _read_docker_images() -> dict:
    """This node's docker_images settings (node-scoped, legacy-fallback)."""
    return dict(get_effective_setting("docker_images", {}) or {})


def _write_docker_images(docker_images: dict) -> None:
    merge_node_settings({"docker_images": docker_images})

# ---------------------------------------------------------------------------
# In-memory pull operation state
# ---------------------------------------------------------------------------

_pull_lock = threading.Lock()
_pull_state: dict = {
    "image": None,
    "status": "idle",   # idle | pulling | done | error
    "message": "",
    "started_at": None,
    "finished_at": None,
}


def _do_pull(image_name: str) -> None:
    """Background thread: pull image via Docker SDK and update stored metadata."""
    from core.helpers import get_docker_client

    logger.info("Image pull started: %s", image_name)
    try:
        client = get_docker_client()
        last_status = "Pulling..."

        for line in client.api.pull(image_name, stream=True, decode=True):
            status = line.get("status", "")
            if not status:
                continue

            error = line.get("error")
            if error:
                raise RuntimeError(error)

            # Build a human-readable progress string
            detail = line.get("progressDetail") or {}
            current = detail.get("current")
            total = detail.get("total")
            if current and total and total > 0:
                pct = round(current / total * 100)
                layer_id = line.get("id", "")
                last_status = f"{status} {layer_id} {pct}%"
            else:
                last_status = status

            with _pull_lock:
                _pull_state["message"] = last_status

        # Pull succeeded - refresh local image metadata
        now = time.time()
        digest = None
        size_bytes = None
        try:
            image = client.images.get(image_name)
            digests = image.attrs.get("RepoDigests", [])
            if digests:
                digest = digests[0].split("@")[-1]
            size_bytes = image.attrs.get("Size")
        except Exception:
            pass

        _update_image_record(image_name, pulled_at=now, checked_at=now,
                              digest=digest, size_bytes=size_bytes)

        with _pull_lock:
            _pull_state["status"] = "done"
            _pull_state["message"] = last_status
            _pull_state["finished_at"] = now
        logger.info("Image pull complete: %s (digest=%s)", image_name, digest)

    except Exception as e:
        with _pull_lock:
            _pull_state["status"] = "error"
            _pull_state["message"] = str(e)
            _pull_state["finished_at"] = time.time()
        logger.warning("Image pull failed: %s: %s", image_name, e)


def _update_image_record(image_name: str, pulled_at: float | None = None,
                          checked_at: float | None = None,
                          digest: str | None = None,
                          size_bytes: int | None = None) -> None:
    docker_images = _read_docker_images()
    images_list = docker_images.setdefault("images", [])

    record = next((r for r in images_list if r.get("name") == image_name), None)
    if record is None:
        record = {"name": image_name}
        images_list.append(record)

    if pulled_at is not None:
        record["last_pulled_at"] = pulled_at
    if checked_at is not None:
        record["last_checked_at"] = checked_at
    if digest is not None:
        record["digest"] = digest
    if size_bytes is not None:
        record["size_mb"] = round(size_bytes / (1024 * 1024))

    _write_docker_images(docker_images)


def _get_image_local_info(image_name: str) -> dict:
    """Return local Docker image metadata. Returns {present: False} on miss."""
    from core.helpers import get_docker_client
    try:
        client = get_docker_client()
        image = client.images.get(image_name)
        digests = image.attrs.get("RepoDigests", [])
        digest = digests[0].split("@")[-1] if digests else None
        size_bytes = image.attrs.get("Size")
        created = image.attrs.get("Created", "")
        return {
            "present": True,
            "digest": digest,
            "size_mb": round(size_bytes / (1024 * 1024)) if size_bytes else None,
            "created": created,
        }
    except Exception:
        return {"present": False}


def _trigger_pull(image_name: str) -> bool:
    """Start an image pull in a background thread. Returns False if already pulling."""
    with _pull_lock:
        if _pull_state["status"] == "pulling":
            return False
        _pull_state["image"] = image_name
        _pull_state["status"] = "pulling"
        _pull_state["message"] = "Starting..."
        _pull_state["started_at"] = time.time()
        _pull_state["finished_at"] = None

    thread = threading.Thread(target=_do_pull, args=(image_name,), daemon=True)
    thread.start()
    return True


def check_and_pull_if_needed(image_name: str) -> bool:
    """Trigger an auto-update pull if the configured interval has elapsed.

    Called by the background monitoring poller. Returns True if a pull was started.
    """
    with _pull_lock:
        if _pull_state["status"] == "pulling":
            return False

    docker_images = _read_docker_images()

    if not docker_images.get("auto_update_enabled"):
        return False

    interval_hours = docker_images.get("auto_update_interval_hours", 24)
    images_list = docker_images.get("images", [])

    record = next((r for r in images_list if r.get("name") == image_name), {})
    last_pulled = record.get("last_pulled_at", 0)

    if time.time() - last_pulled < interval_hours * 3600:
        return False

    return _trigger_pull(image_name)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/images", methods=["GET"])
def list_images():
    docker_images = _read_docker_images()
    images_list = list(docker_images.get("images", []))

    # Always show the currently configured LLAMA_IMAGE first
    known_names = {r["name"] for r in images_list}
    if LLAMA_IMAGE and LLAMA_IMAGE not in known_names:
        images_list.insert(0, {"name": LLAMA_IMAGE})

    result = []
    for record in images_list:
        name = record.get("name", "")
        local = _get_image_local_info(name)
        result.append({
            "name": name,
            "present": local.get("present", False),
            "digest": record.get("digest") or local.get("digest"),
            "size_mb": record.get("size_mb") or local.get("size_mb"),
            "created": local.get("created"),
            "last_pulled_at": record.get("last_pulled_at"),
        })

    return jsonify({
        "images": result,
        "auto_update_enabled": docker_images.get("auto_update_enabled", False),
        "auto_update_interval_hours": docker_images.get("auto_update_interval_hours", 24),
        "current_image": LLAMA_IMAGE,
    })


@bp.route("/api/images/pull-status", methods=["GET"])
def get_pull_status():
    with _pull_lock:
        state = dict(_pull_state)
    return jsonify(state)


@bp.route("/api/images/pull", methods=["POST"])
def pull_image():
    data = request.get_json(silent=True) or {}
    image_name = (data.get("image") or "").strip() or LLAMA_IMAGE
    if not image_name:
        return jsonify({"error": "no image specified"}), 400

    if not _trigger_pull(image_name):
        return jsonify({"error": "a pull is already in progress"}), 409

    return jsonify({"ok": True, "image": image_name})


@bp.route("/api/images", methods=["DELETE"])
def delete_image():
    data = request.get_json(silent=True) or {}
    image_name = (data.get("image") or "").strip()
    if not image_name:
        return jsonify({"error": "no image specified"}), 400

    # Any image may be deleted (including the node default) unless a running
    # instance on this node is using it.
    from core.state import instances, instances_lock
    with instances_lock:
        for inst in instances.values():
            if inst.get("status") in ("stopped",):
                continue
            inst_image = (inst.get("config", {}) or {}).get("image") or LLAMA_IMAGE
            if inst_image == image_name:
                return jsonify({"error": f"image is in use by an instance on port {inst['port']}  stop it first"}), 409

    from core.helpers import get_docker_client
    import docker

    client = get_docker_client()
    removed_from_docker = False
    try:
        client.images.remove(image_name, force=False)
        removed_from_docker = True
    except docker.errors.ImageNotFound:
        pass
    except docker.errors.APIError as e:
        return jsonify({"error": str(e)}), 409

    # Remove from tracked list (node-scoped)
    docker_images = _read_docker_images()
    images_list = docker_images.get("images", [])
    docker_images["images"] = [r for r in images_list if r.get("name") != image_name]
    _write_docker_images(docker_images)

    logger.info("Image removed: %s (docker=%s)", image_name, removed_from_docker)
    return jsonify({"ok": True, "removed_from_docker": removed_from_docker})


@bp.route("/api/images/settings", methods=["POST"])
def save_image_settings():
    data = request.get_json(silent=True) or {}

    auto_update_enabled = bool(data.get("auto_update_enabled", False))
    try:
        interval_hours = max(1, int(data.get("auto_update_interval_hours", 24)))
    except (TypeError, ValueError):
        interval_hours = 24

    docker_images = _read_docker_images()
    docker_images["auto_update_enabled"] = auto_update_enabled
    docker_images["auto_update_interval_hours"] = interval_hours
    _write_docker_images(docker_images)

    return jsonify({"ok": True})
