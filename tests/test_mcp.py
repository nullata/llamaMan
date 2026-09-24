# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""core/mcp.py protocol dispatch — backend-free (fake service injected).

Compliance rules under test (docs/kb-mcp-plan.md §5, MCP rev 2025-06-18):
  - initialize negotiates the version, returns top-level instructions +
    listChanged capability, and mints the Mcp-Session-Id;
  - post-initialize requests need a live session (else 404/-32001) and, on
    2025-06-18, the MCP-Protocol-Version header;
  - notifications get 202 and never a reply;
  - tool failures are isError results at 200, protocol faults are RPC errors;
  - tools/list gating follows kb_mcp_ingest.
"""

import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core import mcp
from core.mcp import handle_message


class FakeService:
    def __init__(self, ingest_enabled=False, delete_enabled=False):
        self._ingest = ingest_enabled
        self._delete = delete_enabled
        self.calls = []

    def ingest_enabled(self):
        return self._ingest

    def delete_enabled(self):
        return self._delete

    def delete_document(self, document_id):
        self.calls.append(("delete_document", document_id))
        return {"deleted": True, "document_id": document_id}

    def delete_topic(self, topic_name):
        self.calls.append(("delete_topic", topic_name))
        return {"deleted": True, "topic": topic_name, "topic_id": 1}

    def list_topics(self):
        self.calls.append("list_topics")
        return [{"id": 1, "name": "T", "document_count": 2}]

    def list_documents(self, topic=None, limit=50):
        self.calls.append(("list_documents", topic, limit))
        return [{"id": 7, "topic_id": 1, "title": "d",
                 "source": "", "chunk_count": 3}]

    def search(self, query, topic=None, limit=8):
        self.calls.append(("search", query, topic, limit))
        return [{"document_id": 1, "title": "d", "topic": "T", "seq": 1,
                 "text": "hit", "score": 0.8}]

    def get_document(self, document_id):
        self.calls.append(("get_document", document_id))
        return {"id": document_id, "title": "d", "content": "body"}

    def ingest(self, topic, title, content, source="", max_bytes=None):
        self.calls.append(("ingest", topic, title, len(content)))
        return {"document_id": 5, "chunks": 1, "unchanged": False,
                "topic": topic}


def initialize(service=None, version="2025-06-18"):
    service = service or FakeService()
    reply, sid, status = handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": version,
                               "clientInfo": {"name": "test"}}}),
        None, None, service)
    return reply, sid, status, service


class Initialize(unittest.TestCase):
    def setUp(self):
        mcp._sessions_reset()

    def test_shape_and_session(self):
        reply, sid, status, _ = initialize()
        self.assertEqual(status, 200)
        result = reply["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["serverInfo"]["name"], "llamaman")
        # instructions is a TOP-LEVEL sibling of serverInfo, not inside it
        self.assertIn("instructions", result)
        self.assertNotIn("instructions", result["serverInfo"])
        self.assertEqual(result["capabilities"]["tools"], {"listChanged": True})
        self.assertTrue(sid and len(sid) >= 16)

    def test_unknown_version_negotiates_latest(self):
        reply, _, _, _ = initialize(version="1999-01-01")
        self.assertEqual(reply["result"]["protocolVersion"], "2025-06-18")

    def test_older_version_echoed_and_header_exempt(self):
        reply, sid, _, service = initialize(version="2024-11-05")
        self.assertEqual(reply["result"]["protocolVersion"], "2024-11-05")
        # no MCP-Protocol-Version header required at this revision
        r2, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            sid, None, service)
        self.assertEqual(st, 200)

    def test_session_carried_on_initialize_is_ignored(self):
        mcp._sessions_put("bogus", {"client_info": {}, "initialized": False,
                                    "protocol_version": "2025-06-18"})
        _, sid, _, _ = initialize()
        self.assertNotEqual(sid, "bogus")


class SessionRules(unittest.TestCase):
    def setUp(self):
        mcp._sessions_reset()

    def test_missing_session_404(self):
        _, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            None, "2025-06-18", FakeService())
        self.assertEqual(st, 404)

    def test_unknown_session_404_with_rpc_error(self):
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            "nope", "2025-06-18", FakeService())
        self.assertEqual(st, 404)
        self.assertEqual(reply["error"]["code"], mcp.SESSION_NOT_FOUND)

    def test_protocol_version_header_required(self):
        _, sid, _, service = initialize()
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            sid, None, service)
        self.assertEqual(st, 400)
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            sid, "1999-01-01", service)
        self.assertEqual(st, 400)

    def test_session_bounded(self):
        for i in range(mcp.SESSION_MAX + 40):
            mcp._sessions_put(f"s{i}", {"client_info": {}, "initialized": True,
                                        "protocol_version": "2025-06-18"})
        self.assertLessEqual(len(mcp._sessions), mcp.SESSION_MAX)


class Envelope(unittest.TestCase):
    def setUp(self):
        mcp._sessions_reset()

    def test_parse_error(self):
        reply, _, st = handle_message("{not json", None, None, FakeService())
        self.assertEqual(st, 400)
        self.assertEqual(reply["error"]["code"], mcp.PARSE_ERROR)

    def test_bad_envelope(self):
        reply, _, _ = handle_message(
            json.dumps({"jsonrpc": "1.0", "id": 1, "method": "ping"}),
            None, None, FakeService())
        self.assertEqual(reply["error"]["code"], mcp.INVALID_REQUEST)

    def test_batch_rejected(self):
        reply, _, _ = handle_message(
            json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]),
            None, None, FakeService())
        self.assertEqual(reply["error"]["code"], mcp.INVALID_REQUEST)

    def test_unknown_method_request(self):
        _, sid, _, service = initialize()
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 5, "method": "bogus/method"}),
            sid, "2025-06-18", service)
        self.assertEqual(reply["error"]["code"], mcp.METHOD_NOT_FOUND)
        self.assertEqual(st, 200)  # RPC-level error rides a 200

    def test_unknown_notification_silently_accepted(self):
        _, sid, _, service = initialize()
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/whatever"}),
            sid, "2025-06-18", service)
        self.assertIsNone(reply)
        self.assertEqual(st, 202)

    def test_initialized_notification_no_reply(self):
        _, sid, _, service = initialize()
        reply, _, st = handle_message(
            json.dumps({"jsonrpc": "2.0",
                        "method": "notifications/initialized"}),
            sid, "2025-06-18", service)
        self.assertIsNone(reply)
        self.assertEqual(st, 202)


class Tools(unittest.TestCase):
    def setUp(self):
        mcp._sessions_reset()

    def _call(self, method, params, service, sid):
        return handle_message(
            json.dumps({"jsonrpc": "2.0", "id": 9, "method": method,
                        "params": params}), sid, "2025-06-18", service)

    def test_tools_list_gating(self):
        _, sid, _, _ = initialize(FakeService(ingest_enabled=False))
        reply, _, _ = self._call("tools/list", {}, FakeService(False), sid)
        names = {t["name"] for t in reply["result"]["tools"]}
        self.assertNotIn("kb_ingest_document", names)
        # Read-only tools always visible, ingest-gated one hidden here.
        for read_only in ("kb_list_topics", "kb_list_documents",
                          "kb_search", "kb_get_document"):
            self.assertIn(read_only, names)
        reply, _, _ = self._call("tools/list", {}, FakeService(True), sid)
        self.assertIn("kb_ingest_document",
                      {t["name"] for t in reply["result"]["tools"]})

    def test_list_documents_roundtrip(self):
        _, sid, _, service = initialize()
        reply, _, st = self._call(
            "tools/call", {"name": "kb_list_documents",
                           "arguments": {"topic": "T", "limit": 20}},
            service, sid)
        self.assertEqual(st, 200)
        self.assertFalse(reply["result"]["isError"])
        self.assertIn("chunk_count", reply["result"]["content"][0]["text"])
        self.assertEqual(service.calls[-1], ("list_documents", "T", 20))

    def test_delete_tools_hidden_when_off(self):
        _, sid, _, _ = initialize(FakeService(delete_enabled=False))
        reply, _, _ = self._call("tools/list", {},
                                 FakeService(delete_enabled=False), sid)
        names = {t["name"] for t in reply["result"]["tools"]}
        self.assertNotIn("kb_delete_document", names)
        self.assertNotIn("kb_delete_topic", names)

    def test_delete_tools_visible_when_on(self):
        _, sid, _, _ = initialize(FakeService(delete_enabled=True))
        reply, _, _ = self._call("tools/list", {},
                                 FakeService(delete_enabled=True), sid)
        names = {t["name"] for t in reply["result"]["tools"]}
        self.assertIn("kb_delete_document", names)
        self.assertIn("kb_delete_topic", names)

    def test_disabled_delete_tools_rpc_error(self):
        _, sid, _, service = initialize()
        for name in ("kb_delete_document", "kb_delete_topic"):
            reply, _, _ = self._call(
                "tools/call", {"name": name, "arguments": {}}, service, sid)
            self.assertEqual(reply["error"]["code"], mcp.INVALID_PARAMS)

    def test_delete_document_roundtrip(self):
        service = FakeService(delete_enabled=True)
        _, sid, _, _ = initialize(service)
        reply, _, st = self._call(
            "tools/call", {"name": "kb_delete_document",
                           "arguments": {"document_id": 7}}, service, sid)
        self.assertEqual(st, 200)
        self.assertFalse(reply["result"]["isError"])
        self.assertIn("deleted", reply["result"]["content"][0]["text"])
        self.assertEqual(service.calls[-1], ("delete_document", 7))

    def test_delete_topic_roundtrip(self):
        service = FakeService(delete_enabled=True)
        _, sid, _, _ = initialize(service)
        reply, _, st = self._call(
            "tools/call", {"name": "kb_delete_topic",
                           "arguments": {"topic": "T"}}, service, sid)
        self.assertEqual(st, 200)
        self.assertFalse(reply["result"]["isError"])
        self.assertEqual(service.calls[-1], ("delete_topic", "T"))

    def test_delete_document_bad_arg_is_tool_error(self):
        service = FakeService(delete_enabled=True)
        _, sid, _, _ = initialize(service)
        reply, _, st = self._call(
            "tools/call", {"name": "kb_delete_document",
                           "arguments": {"document_id": "not-int"}},
            service, sid)
        self.assertEqual(st, 200)
        self.assertTrue(reply["result"]["isError"])

    def test_tool_error_is_iserror_result_not_rpc(self):
        class Broken(FakeService):
            def list_topics(self):
                raise RuntimeError("embedding instance offline")
        broken = Broken()
        _, sid, _, _ = initialize(broken)
        reply, _, st = self._call("tools/call", {"name": "kb_list_topics",
                                                 "arguments": {}}, broken, sid)
        self.assertEqual(st, 200)
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("offline", reply["result"]["content"][0]["text"])
        self.assertNotIn("error", reply)

    def test_unknown_tool_rpc_error(self):
        _, sid, _, service = initialize()
        reply, _, _ = self._call("tools/call", {"name": "kb_exfiltrate",
                                                "arguments": {}}, service, sid)
        self.assertEqual(reply["error"]["code"], mcp.INVALID_PARAMS)

    def test_disabled_ingest_tool_rpc_error(self):
        _, sid, _, service = initialize()
        reply, _, _ = self._call("tools/call", {"name": "kb_ingest_document",
                                                "arguments": {}}, service, sid)
        self.assertEqual(reply["error"]["code"], mcp.INVALID_PARAMS)

    def test_search_roundtrip(self):
        _, sid, _, service = initialize()
        reply, _, st = self._call(
            "tools/call", {"name": "kb_search",
                           "arguments": {"query": "hello", "limit": 3}},
            service, sid)
        self.assertEqual(st, 200)
        self.assertFalse(reply["result"]["isError"])
        self.assertIn("hit", reply["result"]["content"][0]["text"])
        self.assertEqual(service.calls[-1], ("search", "hello", None, 3))

    def test_ingest_roundtrip(self):
        _, sid, _, _ = initialize(FakeService(ingest_enabled=True))
        reply, _, st = self._call(
            "tools/call", {"name": "kb_ingest_document",
                           "arguments": {"topic": "T", "title": "D",
                                         "content": "body"}},
            FakeService(True), sid)
        self.assertEqual(st, 200)
        self.assertFalse(reply["result"]["isError"])

    def test_get_document_not_found_is_tool_error(self):
        class Empty(FakeService):
            def get_document(self, document_id):
                return None
        empty = Empty()
        _, sid, _, _ = initialize(empty)
        reply, _, st = self._call(
            "tools/call", {"name": "kb_get_document",
                           "arguments": {"document_id": 99}}, empty, sid)
        self.assertEqual(st, 200)
        self.assertTrue(reply["result"]["isError"])

    def test_input_schemas_valid(self):
        for tool in mcp.tool_definitions(True):
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object")
            self.assertIsInstance(schema.get("properties"), dict)


if __name__ == "__main__":
    unittest.main()
