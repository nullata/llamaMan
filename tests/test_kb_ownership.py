# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Per-API-key knowledge base ownership (kb_mcp_access = "per_key").

Guarantees under test:
  - a key sees its own topics plus the shared pool (owner ''), never
    another key's; other keys' documents are "not found", not "forbidden";
  - the shared pool is read-only for keys: ingest/delete into it refuse;
  - topics a key creates over MCP are owned by that key;
  - raw owner key ids never reach MCP clients (a `shared` flag instead);
  - global mode (scope None) is unrestricted;
  - the /api/kb REST surface refuses bearer-key requests (admin UI only).
"""

import os
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from flask import Flask

from api.mcp import SHARED_POOL, _KbService
from core import kb
from storage.base import KBUnavailableError


class OwnedKBStorage:
    """Topics/documents with owners; honours visible_to like the backend."""

    def __init__(self):
        self.topics = [
            {"id": 1, "name": "pool", "owner_key_id": ""},
            {"id": 2, "name": "mine", "owner_key_id": "keyA"},
            {"id": 3, "name": "theirs", "owner_key_id": "keyB"},
        ]
        self.docs = {10: 1, 20: 2, 30: 3}  # doc id -> topic id
        self.deleted_docs, self.deleted_topics, self.created = [], [], []

    def _owner(self, topic_id):
        return next(t["owner_key_id"] for t in self.topics if t["id"] == topic_id)

    @staticmethod
    def _sees(owner, visible_to):
        return visible_to is None or owner in ("", visible_to)

    def kb_list_topics(self, visible_to=None):
        return [dict(t) for t in self.topics
                if self._sees(t["owner_key_id"], visible_to)]

    def kb_list_documents(self, topic_id=None, visible_to=None):
        return [{"id": d, "topic_id": t, "owner_key_id": self._owner(t)}
                for d, t in self.docs.items()
                if (topic_id is None or t == topic_id)
                and self._sees(self._owner(t), visible_to)]

    def kb_get_document(self, document_id):
        t = self.docs.get(document_id)
        if t is None:
            return None
        return {"id": document_id, "topic_id": t, "title": "d",
                "owner_key_id": self._owner(t)}

    def kb_delete_document(self, document_id):
        self.deleted_docs.append(document_id)

    def kb_delete_topic(self, topic_id):
        self.deleted_topics.append(topic_id)

    def kb_create_topic(self, name, description="", owner_key_id=""):
        t = {"id": 100 + len(self.created), "name": name,
             "owner_key_id": owner_key_id}
        self.topics.append(t)
        self.created.append(t)
        return dict(t)


class _Base(unittest.TestCase):
    def setUp(self):
        self.st = OwnedKBStorage()
        self._patches = [
            patch("api.mcp.get_storage", return_value=self.st),
            patch("storage.get_storage", return_value=self.st),
            patch("core.kb.assert_kb_ready", lambda: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def svc(self, scope):
        return _KbService({"kb_mcp_ingest": True, "kb_mcp_delete": True}, scope)


class PerKeyReads(_Base):
    def test_sees_own_and_shared_not_others(self):
        names = {t["name"] for t in self.svc("keyA").list_topics()}
        self.assertEqual(names, {"pool", "mine"})

    def test_owner_ids_never_exposed(self):
        for t in self.svc("keyA").list_topics():
            self.assertNotIn("owner_key_id", t)
        flags = {t["name"]: t["shared"] for t in self.svc("keyA").list_topics()}
        self.assertEqual(flags, {"pool": True, "mine": False})

    def test_other_keys_document_is_not_found(self):
        self.assertIsNone(self.svc("keyA").get_document(30))
        self.assertIsNotNone(self.svc("keyA").get_document(10))  # shared
        self.assertIsNotNone(self.svc("keyA").get_document(20))  # own

    def test_list_documents_scoped(self):
        ids = {d["id"] for d in self.svc("keyA").list_documents()}
        self.assertEqual(ids, {10, 20})
        self.assertEqual(self.svc("keyA").list_documents(topic="theirs"), [])

    def test_search_passes_scope(self):
        with patch("core.kb.search", return_value=[]) as s:
            self.svc("keyA").search("q")
        self.assertEqual(s.call_args.kwargs["visible_to"], "keyA")


class PerKeyWrites(_Base):
    def test_delete_own_document(self):
        self.assertTrue(self.svc("keyA").delete_document(20)["deleted"])
        self.assertEqual(self.st.deleted_docs, [20])

    def test_delete_shared_document_refused(self):
        with self.assertRaisesRegex(KBUnavailableError, "read-only"):
            self.svc("keyA").delete_document(10)
        self.assertEqual(self.st.deleted_docs, [])

    def test_delete_other_keys_document_is_not_found(self):
        with self.assertRaisesRegex(KBUnavailableError, "not found"):
            self.svc("keyA").delete_document(30)

    def test_delete_shared_topic_refused(self):
        with self.assertRaisesRegex(KBUnavailableError, "read-only"):
            self.svc("keyA").delete_topic("pool")
        self.assertEqual(self.st.deleted_topics, [])

    def test_delete_other_keys_topic_is_not_found(self):
        with self.assertRaisesRegex(KBUnavailableError, "not found"):
            self.svc("keyA").delete_topic("theirs")

    def test_ingest_new_topic_owned_by_key(self):
        with patch("api.mcp.ingest_document_wrapped",
                   return_value={"document_id": 1}) as ing:
            self.svc("keyA").ingest("fresh", "t", "body")
        self.assertEqual(self.st.created[-1]["owner_key_id"], "keyA")
        self.assertEqual(ing.call_args[0][0], self.st.created[-1]["id"])

    def test_ingest_into_shared_topic_refused(self):
        with patch("api.mcp.ingest_document_wrapped") as ing:
            with self.assertRaisesRegex(KBUnavailableError, "read-only"):
                self.svc("keyA").ingest("pool", "t", "body")
        ing.assert_not_called()

    def test_ingest_other_keys_topic_name_creates_own(self):
        # keyB's "theirs" is invisible to keyA, so keyA gets its own.
        with patch("api.mcp.ingest_document_wrapped",
                   return_value={"document_id": 1}):
            self.svc("keyA").ingest("theirs", "t", "body")
        self.assertEqual(self.st.created[-1]["owner_key_id"], "keyA")


class GlobalMode(_Base):
    """Global = the shared pool only. Keys' private topics stay hidden, so
    switching per_key -> global never exposes them."""

    def svc(self, scope=SHARED_POOL):
        return super().svc(scope)

    def test_sees_only_shared_pool(self):
        self.assertEqual({t["name"] for t in self.svc().list_topics()}, {"pool"})
        self.assertEqual({d["id"] for d in self.svc().list_documents()}, {10})

    def test_private_documents_not_found(self):
        self.assertIsNone(self.svc().get_document(20))
        self.assertIsNone(self.svc().get_document(30))
        with self.assertRaisesRegex(KBUnavailableError, "not found"):
            self.svc().delete_document(30)
        with self.assertRaisesRegex(KBUnavailableError, "not found"):
            self.svc().delete_topic("mine")

    def test_can_edit_shared(self):
        self.svc().delete_topic("pool")
        self.assertEqual(self.st.deleted_topics, [1])
        self.svc().delete_document(10)
        self.assertEqual(self.st.deleted_docs, [10])

    def test_new_topic_is_shared(self):
        with patch("api.mcp.ingest_document_wrapped",
                   return_value={"document_id": 1}):
            self.svc().ingest("fresh", "t", "body")
        self.assertEqual(self.st.created[-1]["owner_key_id"], "")

    def test_ingest_never_lands_in_a_private_topic(self):
        # "mine" exists only as keyA's private topic. A global ingest into
        # that name creates a shared "mine" instead of writing into keyA's.
        with patch("api.mcp.ingest_document_wrapped",
                   return_value={"document_id": 1}) as ing:
            self.svc().ingest("mine", "t", "body")
        created = self.st.created[-1]
        self.assertEqual(created["owner_key_id"], "")
        self.assertEqual(ing.call_args[0][0], created["id"])

    def test_search_scoped_to_shared_pool(self):
        with patch("core.kb.search", return_value=[]) as s:
            self.svc().search("q")
        self.assertEqual(s.call_args.kwargs["visible_to"], "")


class FindTopic(unittest.TestCase):
    def test_own_topic_wins_over_shared_same_name(self):
        st = OwnedKBStorage()
        st.topics = [{"id": 1, "name": "notes", "owner_key_id": ""},
                     {"id": 2, "name": "Notes", "owner_key_id": "keyA"}]
        self.assertEqual(kb.find_topic(st, "notes", "keyA")["id"], 2)
        self.assertEqual(kb.find_topic(st, "notes", "keyB")["id"], 1)
        self.assertEqual(kb.find_topic(st, "notes", None)["id"], 1)


class RestIsAdminOnly(unittest.TestCase):
    def _app(self):
        from api.kb import bp
        app = Flask(__name__)
        app.secret_key = "test"
        app.register_blueprint(bp)
        return app

    def test_bearer_without_session_refused(self):
        with patch("api.auth.is_cluster_peer_request", return_value=False):
            r = self._app().test_client().get(
                "/api/kb/topics", headers={"Authorization": "Bearer k"})
        self.assertEqual(r.status_code, 403)

    def test_session_passes_gate(self):
        app = self._app()
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
        with patch("api.kb.kb_feature_supported",
                   return_value=(False, "json_backend")):
            r = client.get("/api/kb/topics",
                           headers={"Authorization": "Bearer k"})
        self.assertNotEqual(r.status_code, 403)


class _E2EStorage(OwnedKBStorage):
    """OwnedKBStorage plus the settings/API-key surface the auth hook and the
    MCP handler read, so a request can run through the whole stack."""

    TOKENS = {"token-a": "keyA", "token-b": "keyB"}

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def get_settings(self):
        return dict(self.settings)

    def get_api_key_id(self, token):
        return self.TOKENS.get(token)

    def verify_api_key(self, token):
        return token in self.TOKENS

    def user_count(self):
        return 1


class PerKeyEndToEnd(unittest.TestCase):
    """Bearer token -> auth hook (g.kb_key_id) -> handler scope -> tool
    result, through the real Flask app. Catches wiring breaks the service-
    level tests can't see (they hand _KbService its scope directly)."""

    def setUp(self):
        from core import mcp as mcp_core
        mcp_core._sessions_reset()
        self.st = _E2EStorage({"kb_mcp_enabled": True,
                               "kb_mcp_access": "per_key",
                               "require_auth": False})
        self._patches = [
            patch("api.auth.get_storage", return_value=self.st),
            patch("api.mcp.get_storage", return_value=self.st),
            patch("storage.get_storage", return_value=self.st),
            patch("core.kb.assert_kb_ready", lambda: None),
        ]
        for p in self._patches:
            p.start()
        from api import auth
        from api.mcp import bp
        app = Flask(__name__)
        app.register_blueprint(bp)
        auth.init_auth(app)
        self.client = app.test_client()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _rpc(self, token, method, params, sid=None):
        import json
        headers = {"Authorization": f"Bearer {token}"}
        if sid:
            headers["Mcp-Session-Id"] = sid
            headers["MCP-Protocol-Version"] = "2025-06-18"
        return self.client.post("/mcp/knowledge", headers=headers, data=json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))

    def _session(self, token):
        r = self._rpc(token, "initialize", {"protocolVersion": "2025-06-18"})
        self.assertEqual(r.status_code, 200)
        return r.headers["Mcp-Session-Id"]

    def _topics(self, token, sid):
        import json
        r = self._rpc(token, "tools/call",
                      {"name": "kb_list_topics", "arguments": {}}, sid)
        self.assertEqual(r.status_code, 200)
        result = r.get_json()["result"]
        self.assertFalse(result["isError"])
        return {t["name"] for t in json.loads(result["content"][0]["text"])}

    def test_each_key_sees_own_plus_shared(self):
        self.assertEqual(self._topics("token-a", self._session("token-a")),
                         {"pool", "mine"})
        self.assertEqual(self._topics("token-b", self._session("token-b")),
                         {"pool", "theirs"})

    def test_scope_follows_the_request_key_not_the_session(self):
        # MCP sessions aren't bound to a key; every request is scoped by the
        # key it carries, so reusing keyA's session id with keyB's token
        # must still yield keyB's view.
        sid = self._session("token-a")
        self.assertEqual(self._topics("token-b", sid), {"pool", "theirs"})

    def test_switch_to_global_hides_private_topics(self):
        self.st.settings["kb_mcp_access"] = "global"
        sid = self._session("token-a")
        self.assertEqual(self._topics("token-a", sid), {"pool"})

    def test_other_keys_document_not_found_over_the_wire(self):
        sid = self._session("token-a")
        r = self._rpc("token-a", "tools/call",
                      {"name": "kb_get_document",
                       "arguments": {"document_id": 30}}, sid)
        result = r.get_json()["result"]
        self.assertTrue(result["isError"])
        self.assertIn("not found", result["content"][0]["text"])


class SearchScopeReachesStorage(unittest.TestCase):
    """core.kb.search must hand visible_to to the storage query, not just
    accept it."""

    def test_visible_to_passed_to_kb_search(self):
        class St:
            def kb_meta_get(self):
                return {"embedding_dims": "4"}

            def kb_search(self, qv, topic_id=None, limit=8, visible_to=None):
                self.visible_to = visible_to
                return []

        st = St()
        with patch("storage.get_storage", return_value=st), \
             patch("core.kb.assert_kb_ready", lambda: None), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 4]):
            kb.search("q", visible_to="keyA")
            self.assertEqual(st.visible_to, "keyA")
            kb.search("q")
            self.assertIsNone(st.visible_to)


class RestTopicOwner(unittest.TestCase):
    """POST /api/kb/topics owner_key_id must name an existing API key."""

    def _post(self, body):
        from api.kb import bp

        class St(OwnedKBStorage):
            def get_api_keys(self):
                return [{"id": "keyA", "name": "a"}]

        st = St()
        app = Flask(__name__)
        app.secret_key = "test"
        app.register_blueprint(bp)
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
        with patch("api.kb.get_storage", return_value=st), \
             patch("api.kb.kb_feature_supported", return_value=(True, "")):
            return client.post("/api/kb/topics", json=body), st

    def test_unknown_owner_rejected(self):
        r, st = self._post({"name": "x", "owner_key_id": "nope"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(st.created, [])

    def test_known_owner_creates_private_topic(self):
        r, st = self._post({"name": "x", "owner_key_id": "keyA"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(st.created[-1]["owner_key_id"], "keyA")

    def test_no_owner_creates_shared_topic(self):
        r, st = self._post({"name": "x"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(st.created[-1]["owner_key_id"], "")


if __name__ == "__main__":
    unittest.main()
