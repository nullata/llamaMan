# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import sys
import subprocess
import time
import uuid
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from api.settings import get_hf_token_secret
from config import DATA_DIR, LOGS_DIR, MODELS_DIR, logger
from core.downloader import list_repo_files, resolve_filename
from core.helpers import cleanup_download_dir, kill_instance_process, public_dict, read_log_file, stream_log_file
from core.model_sources import record_model_source
from core.state import downloads, downloads_lock, save_state
from storage import get_storage

bp = Blueprint("downloads", __name__)


def _download_progress_path(dl_id: str) -> str:
    return os.path.join(LOGS_DIR, f"dl-{dl_id}.progress.json")


def _build_download_env(repo_id: str, dest_path: str, filename: str, token: str,
                        per_model_mbps: float, progress_file: str = "") -> dict:
    global_mbps = float(get_storage().get_settings().get("global_speed_limit_mbps", 0) or 0)
    effective_mbps = global_mbps if global_mbps > 0 else per_model_mbps
    return {
        **os.environ,
        "HF_REPO_ID": repo_id,
        "HF_LOCAL_DIR": dest_path,
        "HF_FILENAME": filename,
        "HF_TOKEN": token,
        "HF_PROGRESS_FILE": progress_file,
        "HF_SPEED_LIMIT": str(int(effective_mbps * 1_000_000 / 8)) if effective_mbps else "0",
        "HF_PER_MODEL_SPEED_LIMIT": str(int(per_model_mbps * 1_000_000 / 8)) if per_model_mbps else "0",
        "DATA_DIR": DATA_DIR,
        "PYTHONUNBUFFERED": "1",
    }


def _spawn_download_process(dl_id: str, repo_id: str, dest_path: str, filename: str,
                            token: str, per_model_mbps: float, log_mode: str = "w"):
    log_file = os.path.join(LOGS_DIR, f"dl-{dl_id}.log")
    progress_file = _download_progress_path(dl_id)
    log_fh = open(log_file, log_mode, buffering=1)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.downloader"],
        env=_build_download_env(repo_id, dest_path, filename, token, per_model_mbps, progress_file),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    return proc, log_fh, log_file


def _resolve_download_token(dl: dict) -> tuple[str | None, str | None]:
    token = dl.get("_hf_token", "")
    token_id = dl.get("_hf_token_id", "")
    if not token and token_id:
        token = get_hf_token_secret(token_id)
        if not token:
            return None, "Saved Hugging Face token is no longer available"
    return token, None


def _restart_existing_download(dl: dict):
    token, err = _resolve_download_token(dl)
    if err:
        return None, None, None, err
    try:
        return (*_spawn_download_process(
            dl["id"],
            dl["repo_id"],
            dl["dest_path"],
            dl.get("filename", ""),
            token or "",
            float(dl.get("per_model_speed_limit_mbps", 0) or 0),
            log_mode="a",
        ), None)
    except Exception as e:
        return None, None, None, str(e)


def _activate_download_process(dl: dict, proc, log_fh, log_file: str, *, reset_started_at: bool) -> None:
    dl["status"] = "downloading"
    dl["pid"] = proc.pid
    dl["log_file"] = log_file
    dl["_process"] = proc
    dl["_log_fh"] = log_fh
    if reset_started_at:
        dl["started_at"] = time.time()


def restart_download_in_place(
    dl: dict,
    *,
    reset_started_at: bool,
    reset_retry_attempts: bool = False,
    increment_retry_attempts: bool = False,
) -> str | None:
    proc, log_fh, log_file, err = _restart_existing_download(dl)
    if err:
        return err

    _activate_download_process(dl, proc, log_fh, log_file, reset_started_at=reset_started_at)

    if reset_retry_attempts:
        dl["retry_attempts"] = 0
    elif increment_retry_attempts:
        dl["retry_attempts"] = int(dl.get("retry_attempts", 0) or 0) + 1

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/downloads", methods=["GET"])
def api_downloads_list():
    with downloads_lock:
        safe = [public_dict(dl) for dl in downloads.values()]
    return jsonify(safe)


@bp.route("/api/downloads", methods=["POST"])
def api_downloads_create():
    body = request.get_json(force=True)
    repo_id = body.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400

    filename = body.get("filename", "").strip()
    token = body.get("hf_token", "").strip()
    token_id = body.get("hf_token_id", "").strip()
    if token_id:
        token = get_hf_token_secret(token_id)
        if not token:
            return jsonify({"error": "Saved Hugging Face token not found"}), 400
    per_model_mbps = float(body.get("speed_limit_mbps", 0) or 0)

    if filename:
        try:
            repo_files = list_repo_files(repo_id, token or None)
        except Exception as e:
            return jsonify({"error": f"Could not list files in {repo_id}: {e}"}), 502
        try:
            targets = resolve_filename(filename, repo_files, rid=repo_id)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        # Canonical name: shard 1 for multipart, full repo path for nested basenames.
        filename = targets[0]["name"]

    dest_name = repo_id.split("/")[-1]
    if filename:
        dest_name = Path(filename).stem
    dest_path = os.path.join(MODELS_DIR, dest_name)
    os.makedirs(dest_path, exist_ok=True)
    model_path = os.path.join(dest_path, filename) if filename else dest_path

    dl_id = str(uuid.uuid4())

    try:
        proc, log_fh, log_file = _spawn_download_process(
            dl_id, repo_id, dest_path, filename, token, per_model_mbps, log_mode="w",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        "per_model_speed_limit_mbps": per_model_mbps,
        "retry_attempts": 0,
        "_process": proc,
        "_log_fh": log_fh,
    }

    with downloads_lock:
        downloads[dl_id] = dl

    record_model_source(dest_path, repo_id, model_path=model_path)

    logger.info("Download started: %s -> %s (pid %d)", repo_id, dest_path, proc.pid)
    save_state()
    return jsonify(public_dict(dl)), 201


@bp.route("/api/downloads/<dl_id>", methods=["GET"])
def api_downloads_get(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(public_dict(dl))


@bp.route("/api/downloads/<dl_id>/pause", methods=["POST"])
def api_downloads_pause(dl_id):
    exited_status = ""
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        if dl["status"] != "downloading":
            return jsonify({"error": "Only active downloads can be paused"}), 409
        proc = dl.get("_process")
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                kill_instance_process(dl)
                dl["status"] = "completed" if exit_code == 0 else "failed"
                dl["pid"] = 0
                exited_status = dl["status"]
            else:
                kill_instance_process(dl)
                dl["status"] = "paused"
                dl["pid"] = 0
        else:
            dl["status"] = "paused"
            dl["pid"] = 0

    save_state()
    if exited_status:
        return jsonify({"error": f"Download already {exited_status}"}), 409
    logger.info("Download paused: %s", dl_id)
    return jsonify({"status": "paused"})


@bp.route("/api/downloads/<dl_id>/resume", methods=["POST"])
def api_downloads_resume(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        if dl["status"] != "paused":
            return jsonify({"error": "Only paused downloads can be resumed"}), 409

        proc, log_fh, log_file, err = _restart_existing_download(dl)
        if err:
            code = 409 if "token" in err.lower() else 500
            return jsonify({"error": err}), code

        _activate_download_process(dl, proc, log_fh, log_file, reset_started_at=False)

    save_state()
    logger.info("Download resumed: %s (pid %d)", dl_id, proc.pid)
    return jsonify(public_dict(dl))


@bp.route("/api/downloads/<dl_id>/retry", methods=["POST"])
def api_downloads_retry(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        if dl["status"] != "failed":
            return jsonify({"error": "Only failed downloads can be retried"}), 409

        err = restart_download_in_place(
            dl,
            reset_started_at=True,
            reset_retry_attempts=True,
        )
        if err:
            code = 409 if "token" in err.lower() else 500
            return jsonify({"error": err}), code

    save_state()
    logger.info("Download retried: %s (pid %d)", dl_id, dl["pid"])
    return jsonify(public_dict(dl))


@bp.route("/api/downloads/<dl_id>", methods=["DELETE"])
def api_downloads_cancel(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        dest_path = dl.get("dest_path", "")
        kill_instance_process(dl)
        dl["status"] = "cancelled"
        dl["pid"] = 0

    if dest_path:
        cleanup_download_dir(dest_path)
    save_state()
    return jsonify({"status": "cancelled"})


@bp.route("/api/downloads/<dl_id>/remove", methods=["DELETE"])
def api_downloads_remove(dl_id):
    dest_path = ""
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        if dl["status"] in ("downloading", "paused"):
            return jsonify({"error": "Cannot remove active or paused download, cancel it first"}), 409
        if dl["status"] in ("failed", "cancelled"):
            dest_path = dl.get("dest_path", "")
        del downloads[dl_id]
    if dest_path:
        cleanup_download_dir(dest_path)
    save_state()
    return jsonify({"status": "removed"})


@bp.route("/api/downloads/<dl_id>/logs")
def api_download_logs(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
    if dl is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"lines": read_log_file(dl["log_file"], tail=200)})


@bp.route("/api/downloads/<dl_id>/progress")
def api_download_progress(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        download_status = dl["status"]
    data = {}
    try:
        with open(_download_progress_path(dl_id)) as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        data = {}
    data["download_status"] = download_status
    return jsonify(data)


@bp.route("/api/downloads/<dl_id>/logs/stream")
def api_download_logs_stream(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
    if dl is None:
        return jsonify({"error": "Not found"}), 404
    return Response(stream_log_file(dl["log_file"]),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
