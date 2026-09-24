# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Knowledge base settings, MCP auth branch, and the no-regression contract.

  - _normalize_settings_patch coerces the five kb_* keys (generic UI saves
    tolerate them); display defaults are read-path only;
  - api/auth.py's kb-mcp branch: disabled => 404 uniformly in BOTH access
    modes (no route-existence oracle); when require_auth is ON both per_key
    AND global demand a valid bearer (global is a KB-scoping policy, not an
    auth bypass); when require_auth is OFF the endpoint is open in both
    modes;
  - with the feature off, no existing request path touches any kb_* storage
    method (strict mock raises on contact).
"""

import json
import os
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from api.settings import _apply_settings_defaults, _normalize_settings_patch
from storage.json_backend import JsonBackend

KB_KEYS = ("kb_enabled", "kb_mcp_enabled", "kb_mcp_access", "kb_mcp_ingest",
           "kb_mcp_delete", "kb_embedding_instance", "kb_embedding_model")


# --- settings patch coercion ----------------------------------------------


class SettingsCoercion(unittest.TestCase):
    def test_bool_coercion(self):
        out = _normalize_settings_patch({"kb_enabled": "true",
                                         "kb_mcp_ingest": "0",
                                         "kb_mcp_delete": "1"})
        self.assertIs(out["kb_enabled"], True)
        self.assertIs(out["kb_mcp_ingest"], False)
        self.assertIs(out["kb_mcp_delete"], True)

    def test_access_enum(self):
        self.assertEqual(_normalize_settings_patch(
            {"kb_mcp_access": "global"})["kb_mcp_access"], "global")
        self.assertEqual(_normalize_settings_patch(
            {"kb_mcp_access": "bogus"})["kb_mcp_access"], "global")

    def test_instance_str(self):
        self.assertEqual(_normalize_settings_patch(
            {"kb_embedding_instance": "i7"})["kb_embedding_instance"], "i7")
        self.assertEqual(_normalize_settings_patch(
            {"kb_embedding_instance": 5})["kb_embedding_instance"], "")

    def test_embedding_model_str(self):
        self.assertEqual(_normalize_settings_patch(
            {"kb_embedding_model": "bge-m3.gguf"})["kb_embedding_model"],
            "bge-m3.gguf")
        self.assertEqual(_normalize_settings_patch(
            {"kb_embedding_model": None})["kb_embedding_model"], "")

    def test_absent_keys_stay_absent_in_patch(self):
        out = _normalize_settings_patch({"recording_mode": "off"})
        for k in KB_KEYS:
            self.assertNotIn(k, out)

    def test_display_defaults_read_path_only(self):
        out = _apply_settings_defaults({})
        self.assertIs(out["kb_enabled"], False)
        self.assertEqual(out["kb_mcp_access"], "global")
        self.assertIs(out["kb_mcp_ingest"], False)
        self.assertIs(out["kb_mcp_delete"], False)
        self.assertEqual(out["kb_embedding_instance"], "")
        self.assertEqual(out["kb_embedding_model"], "")


# --- MCP auth branch --------------------------------------------------------


class FakeAuthStorage:
    def __init__(self, settings=None, api_keys_valid=False):
        self.settings = settings or {}
        self.api_keys_valid = api_keys_valid

    def get_settings(self):
        return dict(self.settings)

    def verify_api_key(self, token):
        return self.api_keys_valid and token == "good-key"

    def get_api_key_id(self, token):
        return "key1" if self.verify_api_key(token) else None

    def user_count(self):
        return 1


def _mcp_app():
    from api.mcp import bp
    from api import auth
    app = Flask(__name__)
    app.register_blueprint(bp)
    auth.init_auth(app)
    return app


INIT_BODY = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18"}})


class McpAuth(unittest.TestCase):
    def _post(self, storage, headers=None):
        app = _mcp_app()
        with patch("api.auth.get_storage", return_value=storage), \
             patch("api.mcp.get_storage", return_value=storage):
            return app.test_client().post("/mcp/knowledge", data=INIT_BODY,
                                          headers=headers or {})

    def test_disabled_404_uniformly_per_key(self):
        r = self._post(FakeAuthStorage({"kb_mcp_enabled": False,
                                        "kb_mcp_access": "per_key"}))
        self.assertEqual(r.status_code, 404)

    def test_disabled_404_uniformly_global(self):
        r = self._post(FakeAuthStorage({"kb_mcp_enabled": False,
                                        "kb_mcp_access": "global"}))
        self.assertEqual(r.status_code, 404)

    def test_require_auth_on_per_key_requires_bearer(self):
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "per_key"},
                            api_keys_valid=True)
        self.assertEqual(self._post(s).status_code, 401)
        r = self._post(s, {"Authorization": "Bearer good-key"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Mcp-Session-Id", r.headers)

    def test_require_auth_on_global_still_requires_bearer(self):
        # "global" is a KB-scoping policy, not an auth bypass — this is the
        # cell of the matrix that used to be open by mistake.
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "global"},
                            api_keys_valid=True)
        self.assertEqual(self._post(s).status_code, 401)
        r = self._post(s, {"Authorization": "Bearer good-key"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Mcp-Session-Id", r.headers)

    def test_per_key_requires_bearer_even_with_auth_off(self):
        # per_key scopes every tool to the caller's key, so there must be
        # one: Require Authentication off does not open the endpoint.
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "per_key",
                             "require_auth": False}, api_keys_valid=True)
        self.assertEqual(self._post(s).status_code, 401)
        r = self._post(s, {"Authorization": "Bearer bad-key"})
        self.assertEqual(r.status_code, 401)
        r = self._post(s, {"Authorization": "Bearer good-key"})
        self.assertEqual(r.status_code, 200)

    def test_per_key_cluster_peer_without_key_refused(self):
        # A peer passes auth via X-Cluster-Secret but carries no key; in
        # per_key mode there is nobody to scope to, so it gets nothing.
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "per_key"})
        with patch("api.auth.is_cluster_peer_request", return_value=True):
            self.assertEqual(self._post(s).status_code, 401)

    def test_require_auth_off_global_is_open(self):
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "global",
                             "require_auth": False})
        self.assertEqual(self._post(s).status_code, 200)

    def test_cors_preflight_no_credentials(self):
        s = FakeAuthStorage({"kb_mcp_enabled": True, "kb_mcp_access": "per_key"})
        app = _mcp_app()
        with patch("api.auth.get_storage", return_value=s), \
             patch("api.mcp.get_storage", return_value=s):
            r = app.test_client().options("/mcp/knowledge",
                                          headers={"Origin": "http://localhost:6000"})
        self.assertIn(r.status_code, (200, 204))
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"),
                         "http://localhost:6000")


# --- zero-round-trip contract --------------------------------------------------


class StrictJsonBackend(JsonBackend):
    """JSON storage that fails loudly if ANY kb_* method is touched while the
    feature is off (the ABC defaults raise KBNotSupportedError, which core
    could catch and hide — AssertionError cannot be hidden). Overridden
    explicitly: __getattr__ would never fire for methods the base class
    already defines."""


def _make_kb_trip(name):
    def trip(self, *a, **kw):
        raise AssertionError(f"KB touched while off: {name}")
    trip.__name__ = name
    return trip


for _name in ("kb_available", "kb_server_version", "ensure_kb_tables",
              "kb_create_topic", "kb_list_topics", "kb_update_topic",
              "kb_delete_topic", "kb_upsert_document", "kb_get_document",
              "kb_list_documents", "kb_delete_document", "kb_meta_get",
              "kb_meta_set", "kb_insert_chunks", "kb_search",
              "kb_clear_chunks", "kb_counts"):
    setattr(StrictJsonBackend, _name, _make_kb_trip(_name))


class ZeroTouch(unittest.TestCase):
    """Existing endpoints must not touch KB code with the feature disabled."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        cls.storage = StrictJsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"))
        # get_storage must hand out the strict backend everywhere, including
        # app.py's import-time migrations/load_state — so patch the module
        # singleton before importing app, per the mounting note in the plan.
        cls._patcher = patch("storage._backend", cls.storage)
        cls._patcher.start()
        import app as app_module
        cls.app = app_module.app

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        cls._tmp.cleanup()

    def _get(self, path, **kw):
        # Auth bypassed via the repo's test_cluster pattern: this test is
        # about KB non-contact, not about auth (covered above).
        import api.auth as auth_module
        saved = auth_module._has_users
        auth_module._has_users = False
        app = self.app
        funcs = app.before_request_funcs
        app.before_request_funcs = {}
        try:
            with patch("storage.get_storage", return_value=self.storage):
                return app.test_client().get(path, **kw)
        finally:
            app.before_request_funcs = funcs
            auth_module._has_users = saved

    def test_existing_endpoints_never_touch_kb(self):
        for path in ("/api/settings", "/api/instances", "/api/cluster/nodes"):
            r = self._get(path)
            self.assertIn(r.status_code, (200, 503), f"{path} -> {r.status_code}")

    def test_mcp_endpoint_404_when_disabled(self):
        import api.auth as auth_module
        funcs = self.app.before_request_funcs
        self.app.before_request_funcs = {}
        try:
            with patch("storage.get_storage", return_value=self.storage):
                r = self.app.test_client().post("/mcp/knowledge", data=INIT_BODY)
        finally:
            self.app.before_request_funcs = funcs
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
