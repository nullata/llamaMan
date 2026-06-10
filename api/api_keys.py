# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import secrets

from flask import Blueprint, jsonify, request, session

from config import logger
from core.timeutil import now_iso
from storage import get_storage

bp = Blueprint("api_keys", __name__)


def _require_session():
    """Return an error response if no user is logged in, else None."""
    if not session.get("user"):
        return jsonify({"error": "Login required to manage API keys"}), 401


@bp.route("/api/api-keys")
def list_keys():
    """Return all API keys (without the hash, with a masked preview)."""
    err = _require_session()
    if err:
        return err
    keys = get_storage().get_api_keys()
    # Strip the hash, return only safe fields
    safe = []
    for k in keys:
        safe.append({
            "id": k["id"],
            "name": k.get("name", ""),
            "prefix": k.get("prefix", ""),
            "created_at": k.get("created_at"),
        })
    return jsonify(safe)


@bp.route("/api/api-keys", methods=["POST"])
def create_key():
    """Generate a new API key. Returns the raw key exactly once."""
    err = _require_session()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip() or "Untitled"

    raw_key = "llm-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = secrets.token_hex(8)

    entry = {
        "id": key_id,
        "name": name,
        "key_hash": key_hash,
        "prefix": raw_key[:8] + "...",
        "created_at": now_iso(),
    }
    get_storage().save_api_key(entry)
    logger.info("API key created: name=%r id=%s by user=%s", name, key_id, session.get("user"))

    return jsonify({"id": key_id, "name": name, "key": raw_key}), 201


@bp.route("/api/api-keys/<key_id>", methods=["DELETE"])
def delete_key(key_id):
    err = _require_session()
    if err:
        return err
    get_storage().delete_api_key(key_id)
    logger.info("API key deleted: id=%s by user=%s", key_id, session.get("user"))
    return jsonify({"ok": True})
