# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from datetime import timedelta

from flask import Blueprint, jsonify, request

from core.request_log import enabled as recording_enabled, get_mode
from core.timeutil import now_utc
from storage import get_storage

bp = Blueprint("request_log", __name__)


@bp.route("/api/request-log/conversations", methods=["GET"])
def list_conversations():
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 500))
    except (TypeError, ValueError):
        limit = 100
    return jsonify(get_storage().list_conversations(limit=limit))


@bp.route("/api/request-log/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id: str):
    turns = get_storage().get_conversation_turns(conversation_id)
    if not turns:
        return jsonify({"error": "not found"}), 404
    return jsonify({"conversation_id": conversation_id, "turns": turns})


@bp.route("/api/request-log/stats", methods=["GET"])
def request_log_stats():
    """Rolled-up metrics over the recorded turns.

    Query params: inst_id (filter to one instance), window_hours (only turns
    within the last N hours). `recording` reports the current mode so the UI
    can show an enable-logging prompt when there's nothing to record.
    """
    inst_id = request.args.get("inst_id") or None
    since = None
    window_hours = request.args.get("window_hours")
    if window_hours:
        try:
            hours = float(window_hours)
            if hours > 0:
                since = now_utc() - timedelta(hours=hours)
        except (TypeError, ValueError):
            since = None

    stats = get_storage().request_log_stats(inst_id=inst_id, since=since)
    stats["recording"] = recording_enabled()
    stats["recording_mode"] = get_mode()
    return jsonify(stats)
