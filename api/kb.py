# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Knowledge base REST API for the Knowledge settings tab.

Blueprint "kb" — deliberately falls through to the existing /api/ auth
(session cookie or bearer) in api/auth.py; no special branch. NOTE (intentional
per docs/kb-mcp-plan.md §5): bearer API keys therefore gain KB read/write over
REST. Accepted — MCP clients hold exactly these keys for the same data, so
this widens nothing beyond what kb_mcp_enabled already implies, and v1 keys
have no scopes anywhere in the app. If scoped keys ever land, /api/kb/* writes
are a candidate for session-only.

404 (not 403) when the backend cannot host the KB at all, so a misconfigured
client gets one clean signal.
"""

from flask import Blueprint, jsonify, request

from core.kb import KBUnavailableError, kb_feature_supported
from storage import get_storage

bp = Blueprint("kb", __name__, url_prefix="/api/kb")


@bp.before_request
def _admin_only():
    """The REST surface is the admin UI's: unscoped, and not bound by the
    MCP ingest/delete toggles. An API key reaching it would bypass both
    per-key ownership and those toggles, so bearer-authenticated requests
    are refused — keys use /mcp/knowledge. UI sessions pass; so do cluster
    peers (X-Cluster-Secret, e.g. /_peer_embed) and installs with auth not
    set up, which send no bearer."""
    from flask import session
    from api.auth import is_cluster_peer_request
    if session.get("user") or is_cluster_peer_request():
        return None
    if request.headers.get("Authorization", "").startswith("Bearer "):
        return jsonify({"error": "the knowledge base REST API is for the "
                        "admin UI; API keys use the MCP endpoint"}), 403
    return None


def _require_supported():
    ok, reason = kb_feature_supported()
    if not ok:
        return jsonify({"error": f"knowledge base unavailable ({reason})"}), 404
    return None


@bp.route("/status")
def kb_status():
    ok, reason = kb_feature_supported()
    settings = get_storage().get_settings()
    out = {
        "supported": ok,
        "reason": reason,
        "enabled": bool(settings.get("kb_enabled", False)),
        "mcp_enabled": bool(settings.get("kb_mcp_enabled", False)),
        "access": settings.get("kb_mcp_access", "global"),
        "ingest_enabled": bool(settings.get("kb_mcp_ingest", False)),
        "delete_enabled": bool(settings.get("kb_mcp_delete", False)),
        "embedding_instance": settings.get("kb_embedding_instance", "") or "",
        "embedding_model": settings.get("kb_embedding_model", "") or "",
        # Same default as api/auth.is_require_auth_enabled; drives whether the
        # copied MCP client config carries an Authorization header.
        "require_auth": settings.get("require_auth", True) is not False,
        "meta": {},
        "topics": 0, "documents": 0, "chunks": 0,
        "db_version": None,
    }
    ver = None
    try:
        ver = get_storage().kb_server_version()
    except Exception:
        pass
    if ver:
        out["db_version"] = f"{ver[0]}.{ver[1]}"
    if ok:
        try:
            out["meta"] = get_storage().kb_meta_get()
            out.update(get_storage().kb_counts())
        except Exception as e:
            out["error"] = str(e)
    return jsonify(out)


@bp.route("/settings", methods=["POST"])
def kb_settings():
    data = request.get_json(silent=True) or {}
    patch = {}
    ok, reason = kb_feature_supported()
    for key in ("kb_enabled", "kb_mcp_enabled", "kb_mcp_ingest",
                "kb_mcp_delete"):
        if key in data:
            if data[key] and not ok:
                return jsonify({"error": f"cannot enable knowledge base: {reason}"}), 400
            patch[key] = bool(data[key])
    if "kb_mcp_access" in data:
        v = data["kb_mcp_access"]
        if v not in ("per_key", "global"):
            return jsonify({"error": "kb_mcp_access must be per_key or global"}), 400
        patch["kb_mcp_access"] = v
    if "kb_embedding_instance" in data:
        inst_id = (data["kb_embedding_instance"] or "").strip()
        if inst_id:
            from core.state import instances
            inst = instances.get(inst_id)
            if not inst or not inst.get("config", {}).get("embedding_model"):
                return jsonify({"error": "kb_embedding_instance must be a "
                                "running instance with embedding_model enabled"}), 400
        patch["kb_embedding_instance"] = inst_id
    if "kb_embedding_model" in data:
        # Model-name selection is cluster-wide: no strict validation here
        # (the model may live on a peer we haven't heard from yet). Trim,
        # store as-is; runtime resolution proves it or fails cleanly.
        patch["kb_embedding_model"] = (data["kb_embedding_model"] or "").strip()
    if not patch:
        return jsonify({"error": "no kb settings provided"}), 400
    settings = get_storage().merge_settings(patch)
    keep = ("kb_enabled", "kb_mcp_enabled", "kb_mcp_access", "kb_mcp_ingest",
            "kb_mcp_delete", "kb_embedding_instance", "kb_embedding_model")
    return jsonify({"ok": True, "settings": {k: settings.get(k) for k in keep}})


# --- topics ---

@bp.route("/topics")
def list_topics():
    err = _require_supported()
    if err:
        return err
    return jsonify(get_storage().kb_list_topics())


@bp.route("/topics", methods=["POST"])
def create_topic():
    err = _require_supported()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    # Optional: create the topic as a private topic of one API key ('' or
    # absent = shared pool).
    owner = (data.get("owner_key_id") or "").strip()
    if owner and owner not in {k.get("id") for k in get_storage().get_api_keys()}:
        return jsonify({"error": "owner_key_id is not an existing API key"}), 400
    try:
        return jsonify(get_storage().kb_create_topic(
            name, (data.get("description") or "").strip(),
            owner_key_id=owner))
    except Exception as e:
        if "Duplicate" in str(e) or "1062" in str(e):
            return jsonify({"error": f"topic '{name}' already exists"}), 409
        return jsonify({"error": str(e)}), 500


@bp.route("/topics/<int:topic_id>", methods=["POST", "DELETE"])
def edit_topic(topic_id):
    err = _require_supported()
    if err:
        return err
    storage = get_storage()
    if request.method == "DELETE":
        storage.kb_delete_topic(topic_id)
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(storage.kb_update_topic(
            topic_id,
            name=(data["name"].strip() if "name" in data else None),
            description=(data["description"] if "description" in data else None)))
    except LookupError:
        return jsonify({"error": "topic not found"}), 404


# --- documents ---

@bp.route("/documents")
def list_documents():
    err = _require_supported()
    if err:
        return err
    topic_id = request.args.get("topic_id", type=int)
    return jsonify(get_storage().kb_list_documents(topic_id))


@bp.route("/documents/<int:document_id>")
def get_document(document_id):
    err = _require_supported()
    if err:
        return err
    doc = get_storage().kb_get_document(document_id)
    if doc is None:
        return jsonify({"error": "document not found"}), 404
    return jsonify(doc)


@bp.route("/documents", methods=["POST"])
def create_document():
    err = _require_supported()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    topic_id = data.get("topic_id")
    title = (data.get("title") or "").strip()
    content = data.get("content") or ""
    if not isinstance(topic_id, int) or not title or not content:
        return jsonify({"error": "topic_id, title and content are required"}), 400
    if not any(t["id"] == topic_id for t in get_storage().kb_list_topics()):
        return jsonify({"error": "topic not found"}), 404
    from core.kb import ingest_document
    try:
        return jsonify(ingest_document(topic_id, title, content,
                                       data.get("source") or ""))
    except KBUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except LookupError:
        return jsonify({"error": "topic not found"}), 404


@bp.route("/documents/<int:document_id>/delete", methods=["POST"])
def delete_document(document_id):
    err = _require_supported()
    if err:
        return err
    get_storage().kb_delete_document(document_id)
    return jsonify({"ok": True})


# --- search / reembed ---

@bp.route("/search", methods=["POST"])
def search():
    err = _require_supported()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    from core.kb import search as kb_search
    try:
        return jsonify(kb_search(data.get("query") or "",
                                 topic_id=data.get("topic_id"),
                                 limit=data.get("limit", 8)))
    except KBUnavailableError as e:
        return jsonify({"error": str(e)}), 503


@bp.route("/reembed", methods=["POST"])
def reembed():
    err = _require_supported()
    if err:
        return err
    from core.kb import reembed_all
    try:
        return jsonify(reembed_all()), 202
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except KBUnavailableError as e:
        return jsonify({"error": str(e)}), 503


@bp.route("/reembed/status")
def reembed_status():
    err = _require_supported()
    if err:
        return err
    from core.kb import reembed_status as status
    return jsonify(status())


@bp.route("/_peer_embed", methods=["POST"])
def peer_embed():
    """Node-to-node embed forwarder.

    A peer node calls this endpoint when it wants embeddings but has no
    local instance of the configured model. This handler resolves the
    embedding target locally only — never bounces to another peer — so
    a broken configuration can't cascade into an embedding loop.

    Cluster secret is required explicitly here on top of the auth
    middleware's peer bypass, so an accidentally-authenticated client can't
    consume this compute path.
    """
    from core.cluster import CLUSTER_SECRET_HEADER, verify_cluster_secret
    from core.kb import _embed_local, _resolve_embed_target
    if not verify_cluster_secret(request.headers.get(CLUSTER_SECRET_HEADER, "")):
        return jsonify({"error": "cluster secret required"}), 401
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return jsonify({"error": "texts must be a list of strings"}), 400
    try:
        target = _resolve_embed_target()
    except KBUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    if target[0] != "local":
        return jsonify({"error": "no local embedding instance for peer embed"}), 503
    _, inst, host, port = target
    try:
        vectors = _embed_local(inst, host, port, texts)
    except KBUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"embed failed: {e}"}), 500
    return jsonify({"vectors": vectors})
