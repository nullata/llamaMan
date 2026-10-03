# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os

from flask import Blueprint, jsonify, request

from config import MODELS_DIR, logger
from core import archive
from core.archive import ArchiveError
from core.state import downloads, downloads_lock, instances, instances_lock

bp = Blueprint("archive", __name__)


def _instance_paths(inst: dict) -> list[str]:
    """Files an instance needs on disk: its model, draft model and mmproj,
    and - for Strata - the shard folder mounted into its container."""
    cfg = inst.get("config") or {}
    paths = [inst.get("model_path", ""), cfg.get("spec_draft_model") or "",
             cfg.get("mmproj_path") or ""]
    if cfg.get("engine") == "strata":
        from core.engines import strata
        parsed = strata.parse_model_path(inst.get("model_path"))
        if parsed:
            paths.append(strata.shards_base_dir(*parsed))
    return [p for p in paths if p]


def _unit_in_use(base: str, unit: list[str]) -> str | None:
    """Why this unit can't be moved right now (an instance or download
    depends on it), or None."""
    with instances_lock:
        for inst in instances.values():
            # Sleeping counts: waking it would start a container on files
            # that are gone.
            if inst["status"] == "stopped":
                continue
            if any(archive.path_in_unit(p, base, unit) for p in _instance_paths(inst)):
                return f"in use by the {inst['status']} instance on port {inst['port']} - stop it first"
    with downloads_lock:
        for dl in downloads.values():
            if dl.get("status") not in ("downloading", "paused"):
                continue
            for p in (dl.get("dest_path"), dl.get("update_temp_dir"), dl.get("update_model_path")):
                if p and archive.path_in_unit(p, base, unit):
                    return "a download is writing into it - wait for it or cancel it first"
    return None


def _start(kind: str):
    st = archive.status()
    if not st["available"]:
        return jsonify({"error": f"archive is unavailable: {st['reason']}"}), 400
    body = request.get_json(force=True) or {}
    path = (body.get("path") or "").strip()
    if kind == "archive":
        src_base, dst_base = MODELS_DIR, st["dir"]
    else:
        src_base, dst_base = st["dir"], MODELS_DIR
    try:
        unit = archive.model_unit(path, src_base)
    except ArchiveError as e:
        return jsonify({"error": str(e)}), 400
    reason = _unit_in_use(src_base, unit)
    if reason:
        return jsonify({"error": f"{unit[0]} is {reason}"}), 409
    try:
        job = archive.start_job(kind, path, src_base, dst_base, unit)
    except ArchiveError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify(job), 202


@bp.route("/api/archive", methods=["GET"])
def api_archive_status():
    return jsonify({**archive.status(), "jobs": archive.list_jobs()})


@bp.route("/api/archive", methods=["POST"])
def api_archive_start():
    """Move a model from MODELS_DIR to ARCHIVE_DIR. body: {"path": "/models/..."}"""
    return _start("archive")


@bp.route("/api/archive/restore", methods=["POST"])
def api_archive_restore():
    """Move an archived model back to MODELS_DIR. body: {"path": "<ARCHIVE_DIR>/..."}"""
    return _start("restore")


@bp.route("/api/archive/jobs/<job_id>", methods=["DELETE"])
def api_archive_cancel(job_id):
    """Cancel a queued or running move (the source stays where it was), or
    remove a finished one from the list."""
    job = archive.cancel_job(job_id)
    if job is None:
        return jsonify({"error": "Not found"}), 404
    logger.info("archive: cancel/remove requested for %s (%s)", job_id, job["status"])
    return jsonify(job)


def archived_models() -> list[dict]:
    """Models on the archive volume, in /api/models shape, flagged archived
    with the path a restore puts them back at."""
    from api.models import discover_models
    st = archive.status()
    if not st["enabled"] or not os.path.isdir(st["dir"]):
        return []
    out = []
    for m in discover_models(st["dir"]):
        rel = os.path.relpath(m["path"], st["dir"])
        out.append({**m, "archived": True, "restore_path": os.path.join(MODELS_DIR, rel)})
    return out
