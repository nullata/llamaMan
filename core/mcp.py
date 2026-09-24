# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""MCP (Model Context Protocol) server — Streamable HTTP, JSON-RPC 2.0.

Pure protocol layer: no Flask, no KB SQL. The HTTP shape lives in api/mcp.py;
the KB work lives behind the `service` object handed to handle_message().
Compliance: rev 2025-06-18 / 2025-03-26 / 2024-11-05, single-response style
(application/json; SSE server-push not offered — spec-legal per the transport
section), POST-only (GET/DELETE get Flask's automatic 405, which the spec
permits).

Tool failures are MCP-level results ({isError: true} at HTTP 200), never
JSON-RPC errors: an unavailable embedding instance is a tool working
correctly and reporting failure, not a protocol fault.
"""

import secrets
import threading
import time

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = PROTOCOL_VERSIONS[0]
HEADER_PROTOCOL_VERSION = "MCP-Protocol-Version"

SESSION_TTL_S = 24 * 3600
SESSION_MAX = 512

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SESSION_NOT_FOUND = -32001

_INSTRUCTIONS = (
    "Search the llamaman knowledge base. Topics group documents; kb_search "
    "returns the most semantically relevant passages with a 0-1 relevance "
    "score (not a probability). kb_ingest_document adds or updates a "
    "document and is idempotent: resubmitting unchanged content is a no-op."
)

# --- session store -------------------------------------------------------------

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _sessions_new_id() -> str:
    return secrets.token_hex(16)


def _sessions_put(session_id: str, value: dict) -> None:
    now = time.time()
    with _sessions_lock:
        # Evict idle-forever entries first, then LRU until under the cap, so
        # a months-long process with churny clients doesn't grow unbounded.
        # An evicted id behaves like an unknown one (404) — conformant clients
        # re-initialize.
        stale = [k for k, v in _sessions.items()
                 if now - v.get("last_seen", now) > SESSION_TTL_S]
        for k in stale:
            _sessions.pop(k, None)
        while len(_sessions) >= SESSION_MAX:
            oldest = min(_sessions, key=lambda k: _sessions[k].get("last_seen", 0))
            _sessions.pop(oldest, None)
        value["last_seen"] = now
        _sessions[session_id] = value


def _sessions_touch(session_id: str) -> dict | None:
    with _sessions_lock:
        s = _sessions.get(session_id)
        if s is not None:
            s["last_seen"] = time.time()
        return s


def _sessions_reset() -> None:
    """Test hook."""
    with _sessions_lock:
        _sessions.clear()


# --- tools ---------------------------------------------------------------------
#
# inputSchema is Draft 2020-12. kb_ingest_document visibility follows the
# kb_mcp_ingest setting read per tools/list call (via service.ingest_enabled);
# capabilities advertise listChanged, but a POST-only server has no channel to
# push notifications/tools/list_changed on, so clients must re-list after
# toggling (documented behavior; the flag keeps polling clients correct).


def tool_definitions(ingest_enabled: bool, delete_enabled: bool = False) -> list[dict]:
    tools = [
        {
            "name": "kb_list_topics",
            "description": ("List knowledge-base topics with document counts. "
                            "Read-only."),
            "inputSchema": {"type": "object", "properties": {},
                            "additionalProperties": False},
        },
        {
            "name": "kb_search",
            "description": ("Semantic search over the knowledge base. Returns "
                            "passages ranked by a clamped 0-1 relevance score "
                            "(not a calibrated probability). Read-only."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "topic": {"type": "string",
                              "description": "Restrict to one topic by name."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                              "default": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "kb_get_document",
            "description": "Fetch a document's full text by id. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "kb_list_documents",
            "description": ("List knowledge-base documents (metadata only — no "
                            "content; use kb_get_document for full text). "
                            "Optionally filter by topic name. Read-only."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string",
                              "description": "Restrict to one topic by name."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500,
                              "default": 50},
                },
                "additionalProperties": False,
            },
        },
    ]
    if ingest_enabled:
        tools.append({
            "name": "kb_ingest_document",
            "description": ("Add or update a knowledge-base document (topic is "
                            "created if absent). Idempotent: resubmitting "
                            "content whose sha256 matches the stored document "
                            "is a no-op. Max 200 KB content per call."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1},
                    "source": {"type": "string"},
                },
                "required": ["topic", "title", "content"],
                "additionalProperties": False,
            },
        })
    if delete_enabled:
        tools.append({
            "name": "kb_delete_document",
            "description": ("Delete one knowledge-base document by id. Cascades "
                            "to its chunks. Irreversible."),
            "inputSchema": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
                "additionalProperties": False,
            },
        })
        tools.append({
            "name": "kb_delete_topic",
            "description": ("Delete one knowledge-base topic by name. Cascades "
                            "to its documents and chunks. Irreversible."),
            "inputSchema": {
                "type": "object",
                "properties": {"topic": {"type": "string", "minLength": 1}},
                "required": ["topic"],
                "additionalProperties": False,
            },
        })
    return tools


# MCP-side ingest single-flight: embedding a document can take tens of seconds
# and every MCP request holds a gunicorn thread — queueing more than one ingest
# at a time risks starving inference traffic (docs/kb-mcp-plan.md §7).
_ingest_semaphore = threading.Semaphore(1)


# --- dispatch --------------------------------------------------------------------


def _error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def _result(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _tool_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _tool_text(payload) -> dict:
    import json
    text = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_message(raw_body, session_id: str | None,
                   header_version: str | None, service) -> tuple:
    """Process one HTTP POST payload.

    raw_body: bytes/str of the request body (parsed here so -32700 is ours).
    session_id: value of the Mcp-Session-Id header (None if absent).
    header_version: value of the MCP-Protocol-Version header.
    service: object implementing list_topics/search/get_document/ingest/
             ingest_enabled (see api/mcp.py for the real one; tests inject a
             fake).

    Returns (reply, new_session_id, http_status):
      reply None -> empty body (202 Accepted for notifications);
      new_session_id set -> respond with an Mcp-Session-Id header.
    """
    try:
        import json
        body = json.loads(raw_body)
    except Exception:
        return _error(None, PARSE_ERROR, "Parse error"), None, 400

    if isinstance(body, list):
        # JSON-RPC batching was dropped from the MCP spec; reject cleanly.
        return _error(None, INVALID_REQUEST,
                      "Batch requests are not supported"), None, 400

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" \
            or not isinstance(body.get("method"), str):
        return _error(body.get("id") if isinstance(body, dict) else None,
                      INVALID_REQUEST,
                      "Invalid request: jsonrpc 2.0 envelope required"), \
            None, 400

    method = body["method"]
    req_id = body.get("id")
    is_request = "id" in body
    params = body.get("params") or {}

    # -- initialize ---------------------------------------------------------
    if method == "initialize":
        # A session id carried on initialize is ignored (spec: the server
        # mints it here).
        requested = params.get("protocolVersion") or LATEST_PROTOCOL_VERSION
        negotiated = requested if requested in PROTOCOL_VERSIONS \
            else LATEST_PROTOCOL_VERSION
        new_id = _sessions_new_id()
        _sessions_put(new_id, {
            "client_info": params.get("clientInfo") or {},
            "initialized": False,
            "protocol_version": negotiated,
        })
        result = {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "llamaman", "version": _server_version()},
            "instructions": _INSTRUCTIONS,
        }
        return _result(req_id, result), new_id, 200

    # -- every other method requires a live session ------------------------
    if not session_id or _sessions_touch(session_id) is None:
        return _error(req_id, SESSION_NOT_FOUND,
                      "Unknown or missing Mcp-Session-Id"), None, 404

    session = _sessions_touch(session_id)

    # MCP-Protocol-Version header: required by 2025-06-18 on every request
    # after initialize; older negotiated revisions predate the header and are
    # exempt (spec transitional rule).
    if session["protocol_version"] >= "2025-06-18":
        if not header_version:
            return _error(req_id, INVALID_REQUEST,
                          f"Missing {HEADER_PROTOCOL_VERSION} header"), None, 400
        if header_version not in PROTOCOL_VERSIONS:
            return _error(req_id, INVALID_REQUEST,
                          f"Unsupported {HEADER_PROTOCOL_VERSION}: "
                          f"{header_version}"), None, 400

    # -- notifications -------------------------------------------------------
    if not is_request:
        if method == "notifications/initialized":
            session["initialized"] = True
        # notifications/cancelled and anything else unknown: accepted and
        # ignored (v1 embeds are seconds-long; cancellation not honored).
        return None, None, 202

    # -- requests --------------------------------------------------------------
    if method == "ping":
        return _result(req_id, {}), None, 200

    if method == "tools/list":
        return _result(req_id, {"tools": tool_definitions(
            _safe_ingest_enabled(service),
            _safe_delete_enabled(service))}), None, 200

    if method == "tools/call":
        return _dispatch_tool(req_id, params, service)

    return _error(req_id, METHOD_NOT_FOUND,
                  f"Method not found: {method}"), None, 200


def _server_version() -> str:
    try:
        from config import VERSION
        return VERSION
    except Exception:
        return "0"


def _safe_ingest_enabled(service) -> bool:
    try:
        return bool(service.ingest_enabled())
    except Exception:
        return False


def _safe_delete_enabled(service) -> bool:
    try:
        return bool(service.delete_enabled())
    except Exception:
        return False


def _dispatch_tool(req_id: str | None, params: dict, service) -> tuple:
    name = params.get("name")
    args = params.get("arguments") or {}
    if name not in ("kb_list_topics", "kb_list_documents", "kb_search",
                    "kb_get_document", "kb_ingest_document",
                    "kb_delete_document", "kb_delete_topic"):
        # Protocol-level: the tool does not exist at all.
        return _error(req_id, INVALID_PARAMS,
                      f"Unknown tool: {name}"), None, 200
    if name == "kb_ingest_document" and not _safe_ingest_enabled(service):
        # Not exposed via tools/list; calling it anyway is a protocol-level
        # invalid-params case.
        return _error(req_id, INVALID_PARAMS,
                      "Tool not enabled: kb_ingest_document"), None, 200
    if name in ("kb_delete_document", "kb_delete_topic") \
            and not _safe_delete_enabled(service):
        return _error(req_id, INVALID_PARAMS,
                      f"Tool not enabled: {name}"), None, 200

    try:
        if name == "kb_list_topics":
            return _result(req_id, _tool_text(service.list_topics())), None, 200
        if name == "kb_list_documents":
            docs = service.list_documents(topic=args.get("topic"),
                                          limit=args.get("limit", 50))
            return _result(req_id, _tool_text(docs)), None, 200
        if name == "kb_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return _result(req_id, _tool_error(
                    "kb_search: 'query' is required")), None, 200
            limit = args.get("limit", 8)
            hits = service.search(query, topic=args.get("topic"), limit=limit)
            return _result(req_id, _tool_text(hits)), None, 200
        if name == "kb_get_document":
            doc_id = args.get("document_id")
            if not isinstance(doc_id, int):
                return _result(req_id, _tool_error(
                    "kb_get_document: 'document_id' (integer) is required")), \
                    None, 200
            doc = service.get_document(doc_id)
            if doc is None:
                return _result(req_id, _tool_error(
                    f"document {doc_id} not found")), None, 200
            return _result(req_id, _tool_text(doc)), None, 200
        if name == "kb_delete_document":
            doc_id = args.get("document_id")
            if not isinstance(doc_id, int):
                return _result(req_id, _tool_error(
                    "kb_delete_document: 'document_id' (integer) is required")), \
                    None, 200
            res = service.delete_document(doc_id)
            return _result(req_id, _tool_text(res)), None, 200
        if name == "kb_delete_topic":
            topic = args.get("topic")
            if not isinstance(topic, str) or not topic.strip():
                return _result(req_id, _tool_error(
                    "kb_delete_topic: 'topic' (string) is required")), None, 200
            res = service.delete_topic(topic)
            return _result(req_id, _tool_text(res)), None, 200
        # kb_ingest_document — single-flight + size cap (thread starvation
        # guard, docs/kb-mcp-plan.md §7).
        from core.kb import MCP_INGEST_MAX_BYTES
        if not _ingest_semaphore.acquire(blocking=False):
            return _result(req_id, _tool_error(
                "ingest busy, retry shortly")), None, 200
        try:
            for field in ("topic", "title", "content"):
                if not isinstance(args.get(field), str) or not args[field].strip():
                    return _result(req_id, _tool_error(
                        f"kb_ingest_document: '{field}' is required")), None, 200
            res = service.ingest(args["topic"], args["title"], args["content"],
                                 args.get("source", ""),
                                 max_bytes=MCP_INGEST_MAX_BYTES)
            return _result(req_id, _tool_text(res)), None, 200
        finally:
            _ingest_semaphore.release()
    except (TypeError, ValueError) as e:
        # Bad argument types/values reaching the service: a tool-level failure.
        return _result(req_id, _tool_error(f"invalid arguments: {e}")), None, 200
    except Exception as e:
        # Any KB-side failure (unavailable/disabled/mismatch/busy/DB) is an
        # isError result at HTTP 200, never an RPC error.
        return _result(req_id, _tool_error(str(e))), None, 200
