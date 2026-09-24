# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""MCP Streamable-HTTP endpoint: POST /mcp/knowledge on the main app port (5000).

Blueprint name "kb-mcp" — api/auth.py's branch keys off it (docs/kb-mcp-plan.md
§6.2): when kb_mcp_enabled is off the auth branch allows through and this
handler 404s uniformly, so a disabled server leaks no route-existence signal.

GET/DELETE /mcp receive Flask's automatic 405 (only POST is registered), which
the MCP spec permits for a server without SSE server-push.

CORS: OPTIONS preflight + origin-reflected headers on this blueprint only, so
browser-based MCP clients work without any other route gaining CORS.
"""

from flask import Blueprint, Response, g, jsonify, request

from core import mcp as mcp_core
from storage import get_storage

bp = Blueprint("kb-mcp", __name__)

_CORS_HEADERS = ("Authorization", "Mcp-Session-Id",
                 mcp_core.HEADER_PROTOCOL_VERSION, "Content-Type")


@bp.after_request
def _cors(resp):
    origin = request.headers.get("Origin")
    if origin:
        # Reflect, never "*": bearer-credentialed requests forbid wildcards.
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = ", ".join(_CORS_HEADERS)
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Max-Age"] = "600"
    return resp


@bp.route("/mcp/knowledge", methods=["POST", "OPTIONS"])
def mcp_endpoint():
    if request.method == "OPTIONS":
        return Response(status=204)

    # Disabled => 404 for every request, no protocol probing possible.
    try:
        settings = get_storage().get_settings()
    except Exception:
        # DB offline: the feature cannot serve; same uniform 404.
        return jsonify({"error": "not found"}), 404
    if not settings.get("kb_mcp_enabled", False):
        return jsonify({"error": "not found"}), 404

    # global: scope to the shared pool only — keys' private topics stay
    # private whatever the mode. per_key: scope to the caller's API key
    # (api/auth.py put its id on g). No identity — e.g. a cluster peer,
    # which authenticates without a key — gets nothing rather than the KB.
    scope = SHARED_POOL
    if settings.get("kb_mcp_access", "global") == "per_key":
        scope = getattr(g, "kb_key_id", None)
        if not scope:
            return jsonify({"error": "API key required"}), 401

    # Raw body so JSON parse errors are reported as -32700 by the protocol
    # layer (request.json would raise before we get control).
    body = request.get_data()
    reply, new_session_id, status = mcp_core.handle_message(
        body,
        request.headers.get("Mcp-Session-Id"),
        request.headers.get(mcp_core.HEADER_PROTOCOL_VERSION),
        _KbService(settings, scope),
    )

    if reply is None:
        resp = Response(status=status)          # 202 for notifications
    else:
        resp = jsonify(reply)
        resp.status_code = status
        resp.headers["Content-Type"] = "application/json"
    if new_session_id:
        resp.headers["Mcp-Session-Id"] = new_session_id
    return resp


SHARED_POOL = ""  # owner_key_id of shared topics; the scope of global mode


class _KbService:
    """Adapts core.kb to the service surface core.mcp expects.

    Ownership (kb_mcp_access), via scope:
      - global: scope is SHARED_POOL. Callers see and (if the MCP toggles
        allow) edit the shared pool only; keys' private topics are hidden,
        so switching per_key -> global never exposes them.
      - per_key: scope is the caller's API key id. It reads its own topics
        plus the shared pool, writes only to its own.
      - None: unrestricted (not used by the endpoint; tests/admin tooling).
    Anything out of scope is "not found", so other owners' document ids and
    topic names don't leak."""

    def __init__(self, settings: dict, scope: str | None = None):
        self._settings = settings
        self._scope = scope

    def ingest_enabled(self) -> bool:
        return bool(self._settings.get("kb_mcp_ingest", False))

    def delete_enabled(self) -> bool:
        return bool(self._settings.get("kb_mcp_delete", False))

    # -- scope helpers --

    def _visible(self, owner: str) -> bool:
        return self._scope is None or owner in ("", self._scope)

    def _check_writable(self, owner: str, what: str) -> None:
        # Only reachable for visible rows, so in global mode (scope '') this
        # never fires: everything visible is the shared pool.
        from storage.base import KBUnavailableError
        if self._scope is not None and owner != self._scope:
            raise KBUnavailableError(
                f"{what} is in the shared pool, which is read-only for API keys")

    @staticmethod
    def _present(row: dict) -> dict:
        """Swap the raw owner id for a shared flag: callers never see key ids."""
        out = {k: v for k, v in row.items() if k != "owner_key_id"}
        out["shared"] = not row.get("owner_key_id")
        return out

    def _find_topic(self, name: str):
        from core.kb import find_topic
        return find_topic(get_storage(), name, self._scope)

    # -- tools --

    def list_topics(self):
        from core.kb import assert_kb_ready
        assert_kb_ready()
        return [self._present(t)
                for t in get_storage().kb_list_topics(visible_to=self._scope)]

    def list_documents(self, topic=None, limit=50):
        from core.kb import assert_kb_ready
        assert_kb_ready()
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise TypeError("limit must be an integer")
        limit = max(1, min(limit, 500))
        topic_id = None
        if topic:
            t = self._find_topic(topic)
            if t is None:
                return []
            topic_id = t["id"]
        docs = get_storage().kb_list_documents(topic_id=topic_id,
                                               visible_to=self._scope)
        return [self._present(d) for d in docs[:limit]]

    def search(self, query, topic=None, limit=8):
        from core.kb import search
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise TypeError("limit must be an integer")
        return search(query, topic_name=topic, limit=limit,
                      visible_to=self._scope)

    def _visible_document(self, document_id: int):
        doc = get_storage().kb_get_document(document_id)
        if doc is None or not self._visible(doc.get("owner_key_id") or ""):
            return None
        return doc

    def get_document(self, document_id: int):
        from core.kb import assert_kb_ready
        assert_kb_ready()
        doc = self._visible_document(document_id)
        return self._present(doc) if doc is not None else None

    def delete_document(self, document_id: int):
        from core.kb import assert_kb_ready
        from storage.base import KBUnavailableError
        assert_kb_ready()
        # Existence check so we can return an honest error rather than a
        # silent no-op; the cascade to kb_chunks is enforced at the FK.
        doc = self._visible_document(document_id)
        if doc is None:
            raise KBUnavailableError(f"document {document_id} not found")
        self._check_writable(doc.get("owner_key_id") or "",
                             f"document {document_id}")
        get_storage().kb_delete_document(document_id)
        return {"deleted": True, "document_id": document_id}

    def delete_topic(self, topic_name: str):
        from core.kb import assert_kb_ready
        from storage.base import KBUnavailableError
        assert_kb_ready()
        t = self._find_topic(topic_name)
        if t is None:
            raise KBUnavailableError(f"topic '{topic_name}' not found")
        self._check_writable(t.get("owner_key_id") or "",
                             f"topic '{topic_name}'")
        get_storage().kb_delete_topic(t["id"])
        return {"deleted": True, "topic": topic_name, "topic_id": t["id"]}

    def ingest(self, topic, title, content, source="", max_bytes=None):
        from core.kb import assert_kb_ready
        from storage.base import KBUnavailableError
        assert_kb_ready()
        storage = get_storage()
        # Resolve/create the topic by name (cluster-wide, idempotent). New
        # topics belong to the calling key in per_key mode, shared in global.
        t = self._find_topic(topic)
        if t is None:
            try:
                t = storage.kb_create_topic(topic,
                                            owner_key_id=self._scope or "")
            except Exception as e:
                # Lost a create race to a peer node? Re-resolve once.
                t = self._find_topic(topic)
                if t is None:
                    raise KBUnavailableError(f"kb: cannot create topic: {e}")
        self._check_writable(t.get("owner_key_id") or "", f"topic '{topic}'")
        res = ingest_document_wrapped(t["id"], title, content, source, max_bytes)
        res["topic"] = topic
        return res


def ingest_document_wrapped(topic_id, title, content, source, max_bytes):
    from core.kb import ingest_document
    return ingest_document(topic_id, title, content, source,
                           max_bytes=max_bytes)
