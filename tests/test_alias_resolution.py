# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""#111 gap 2: the node owning a queue group's members serves its alias whatever
state they're in. An alias usually names no file, so a request for a sleeping
group used to reach _find_model_by_name and die as "model not found"; one that
substring-matched a file woke that unrelated file instead."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.cluster as cluster_api
import api.llamaman as llamaman
from core.state import instances, instances_lock

RETRY_HEADER = cluster_api.RETRY_ELSEWHERE_HEADER


def _member(inst_id, status, alias="chat-pool", path=None):
    return {
        "id": inst_id, "status": status, "port": 8000,
        "model_path": path or f"/models/{inst_id}.gguf",
        "config": {"share_queue": True, "share_queue_group": alias},
    }


def _file(stem):
    return {"path": f"/models/{stem}.gguf", "name": f"{stem}.gguf"}


class _AliasCase(unittest.TestCase):
    """The real _ensure_model_running; only the library and the wake are faked."""

    def setUp(self):
        with instances_lock:
            saved = dict(instances)
            instances.clear()

        def restore():
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.addCleanup(restore)

        self.library = []
        self.wake = Mock(side_effect=self._woke)
        for target, value in (
            ("api.llamaman._library", lambda: self.library),
            ("api.instances.relaunch_inactive_instance", self.wake),
            ("core.model_alias.resolve_to_path", Mock(return_value=None)),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _woke(self, inst_id):
        instances[inst_id]["status"] = "healthy"
        return True

    def _add(self, *members):
        for m in members:
            instances[m["id"]] = m

    def _ensure(self, name="chat-pool"):
        return llamaman._ensure_model_running(name)


class RankingTests(_AliasCase):

    def test_a_sleeping_member_is_woken_and_served(self):
        self._add(_member("a", "sleeping"))
        inst, err = self._ensure()
        self.assertIsNone(err)
        self.assertEqual(inst["id"], "a")
        self.wake.assert_called_once_with("a")

    def test_a_starting_member_is_returned_for_the_caller_to_wait_on(self):
        self._add(_member("a", "starting"))
        inst, err = self._ensure()
        self.assertIsNone(err)
        self.assertEqual(inst["id"], "a")
        self.wake.assert_not_called()

    def test_a_sleeper_listed_first_cannot_hide_a_healthy_sibling(self):
        self._add(_member("asleep", "sleeping"), _member("up", "healthy"))
        inst, _ = self._ensure()
        self.assertEqual(inst["id"], "up")
        self.wake.assert_not_called()

    def test_starting_beats_sleeping(self):
        # Waking a second copy while one is already loading wastes a cold load.
        self._add(_member("asleep", "sleeping"), _member("loading", "starting"))
        inst, _ = self._ensure()
        self.assertEqual(inst["id"], "loading")
        self.wake.assert_not_called()

    def test_a_stopping_member_does_not_hide_a_sleeper(self):
        self._add(_member("going", "stopping"), _member("asleep", "sleeping"))
        inst, _ = self._ensure()
        self.assertEqual(inst["id"], "asleep")

    def test_the_tag_and_case_of_the_request_are_ignored(self):
        self._add(_member("a", "sleeping"))
        inst, _ = self._ensure("CHAT-POOL:latest")
        self.assertEqual(inst["id"], "a")

    def test_a_pretty_name_still_beats_an_alias(self):
        self._add(_member("a", "sleeping", alias="qwen2.5-14b"))
        self.library = [_file("other")]
        with patch("core.model_alias.resolve_to_path", return_value=self.library[0]["path"]), \
                patch.object(llamaman, "_find_running_instance_for_model",
                             return_value={"id": "o", "status": "healthy"}):
            inst, _ = self._ensure("qwen")
        self.assertEqual(inst["id"], "o")
        self.wake.assert_not_called()


class ExactVersusSubstringTests(_AliasCase):

    def test_an_exact_alias_never_falls_through_to_a_substring_file(self):
        self._add(_member("a", "sleeping"))
        self.library = [_file("chat-pool-unrelated-q8")]
        with patch("api.instances.launch_instance") as launch:
            inst, _ = self._ensure()
        self.assertEqual(inst["id"], "a")
        launch.assert_not_called()

    def test_an_exact_alias_beats_an_exact_file_name(self):
        # The operator opted that instance into the name.
        self._add(_member("a", "sleeping"))
        self.library = [_file("chat-pool")]
        inst, _ = self._ensure()
        self.assertEqual(inst["id"], "a")

    def test_a_partial_name_resolves_to_a_single_alias(self):
        # Peers route by substring, so one can arrive forwarded.
        self._add(_member("a", "sleeping"))
        inst, _ = self._ensure("chat")
        self.assertEqual(inst["id"], "a")

    def test_a_partial_alias_beats_a_substring_file(self):
        self._add(_member("a", "sleeping"))
        self.library = [_file("chatterbox-q4")]
        inst, _ = self._ensure("chat")
        self.assertEqual(inst["id"], "a")

    def test_a_partial_name_gives_way_to_a_file_with_exactly_that_name(self):
        self._add(_member("a", "sleeping"))
        file = _file("chat")
        self.library = [file]
        running = {"id": "file-inst", "status": "healthy"}
        with patch.object(llamaman, "_find_running_instance_for_model", return_value=running):
            inst, _ = self._ensure("chat")
        self.assertEqual(inst["id"], "file-inst")
        self.wake.assert_not_called()

    def test_a_partial_name_fitting_two_aliases_wakes_neither(self):
        self._add(_member("a", "sleeping", alias="chat-pool"),
                  _member("b", "sleeping", alias="chat-large"))
        inst, err = self._ensure("chat")
        self.assertIsNone(inst)
        self.assertIn("not found", err)
        self.wake.assert_not_called()

    def test_a_partial_name_fitting_two_aliases_still_finds_a_healthy_member(self):
        self._add(_member("a", "sleeping", alias="chat-pool"),
                  _member("b", "healthy", alias="chat-large"))
        inst, _ = self._ensure("chat")
        self.assertEqual(inst["id"], "b")
        self.wake.assert_not_called()

    def test_an_unknown_name_is_still_not_found(self):
        self._add(_member("a", "sleeping"))
        inst, err = self._ensure("llama-3")
        self.assertIsNone(inst)
        self.assertEqual(err, "model 'llama-3' not found")
        self.assertFalse(hasattr(err, "http_status"))


class UnavailableMemberTests(_AliasCase):

    def _assert_unavailable(self, err, code):
        self.assertIsInstance(err, llamaman.ModelUnavailable)
        self.assertEqual(err.code, code)
        self.assertEqual(err.http_status, 503)
        self.assertTrue(err.retry_elsewhere)

    def test_stopping_is_typed_and_launches_nothing(self):
        self._add(_member("a", "stopping"))
        self.library = [_file("chat-pool-unrelated")]
        with patch("api.instances.launch_instance") as launch:
            inst, err = self._ensure()
        self.assertIsNone(inst)
        self._assert_unavailable(err, "model_stopping")
        launch.assert_not_called()
        self.wake.assert_not_called()

    def test_stopped_is_typed_and_wakes_nothing(self):
        self._add(_member("a", "stopped"))
        self.library = [_file("chat-pool-unrelated")]
        with patch("api.instances.launch_instance") as launch:
            inst, err = self._ensure()
        self.assertIsNone(inst)
        self._assert_unavailable(err, "group_stopped")
        launch.assert_not_called()
        self.wake.assert_not_called()

    def test_a_stopped_member_named_by_its_own_file_still_wakes(self):
        # Asking for a file by name wakes its stopped instance, alias or not.
        member = _member("a", "stopped", alias="q4", path="/models/q4.gguf")
        self._add(member)
        self.library = [_file("q4")]
        with patch.object(llamaman, "_count_running_instances", return_value=0), \
                patch("api.presets.resolve_preset_for_node", return_value={}), \
                patch.object(llamaman, "get_storage"):
            inst, err = self._ensure("q4")
        self.assertIsNone(err)
        self.assertEqual(inst["id"], "a")
        self.wake.assert_called_once_with("a")

    def test_a_failed_wake_is_typed(self):
        self._add(_member("a", "sleeping"))
        self.wake.side_effect = lambda _id: False
        inst, err = self._ensure()
        self.assertIsNone(inst)
        self._assert_unavailable(err, "wake_failed")

    def test_a_wake_that_timed_out_is_typed_apart_from_one_that_failed(self):
        self._add(_member("a", "sleeping"))

        def timed_out(inst_id):
            instances[inst_id]["status"] = "starting"
            return False

        self.wake.side_effect = timed_out
        inst, err = self._ensure()
        self.assertIsNone(inst)
        self._assert_unavailable(err, "wake_timeout")

    def test_a_member_evicted_before_the_recheck_is_typed(self):
        self._add(_member("a", "sleeping"))
        real = llamaman._find_running_instance_by_alias
        answers = [real("chat-pool"), None]
        with patch.object(llamaman, "_find_running_instance_by_alias",
                          side_effect=lambda _n: answers.pop(0)):
            inst, err = self._ensure()
        self.assertIsNone(inst)
        self._assert_unavailable(err, "group_stopped")
        self.wake.assert_not_called()

    def test_the_wake_runs_outside_the_global_lock(self):
        # It blocks until healthy, and every slow-path launch shares that lock.
        self._add(_member("a", "sleeping"))
        held = []

        def wake(inst_id):
            held.append(llamaman._llamaman_lock.locked())
            return self._woke(inst_id)

        self.wake.side_effect = wake
        self._ensure()
        self.assertEqual(held, [False])


@patch("api.cluster.dispatch_inference", return_value=None)
class SurfaceTests(unittest.TestCase):
    """The typed error's status, shape and failover header on each surface."""

    OPENAI = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    ROUTES = (
        ("/api/chat", OPENAI),
        ("/v1/chat/completions", OPENAI),
        ("/v1/completions", {"model": "m", "prompt": "hi"}),
        ("/v1/embeddings", {"model": "m", "input": "hi"}),
    )

    def _post(self, path, body, err, dispatched=False):
        app = Flask(__name__)
        app.register_blueprint(llamaman.bp)
        headers = {"X-Cluster-Dispatch": "1"} if dispatched else {}
        with patch.object(llamaman, "_ensure_model_running", return_value=(None, err)):
            return app.test_client().post(path, json=body, headers=headers)

    @staticmethod
    def _unavailable():
        return llamaman.ModelUnavailable("model_stopping", "model 'm' is stopping on this node")

    def test_typed_errors_are_503_on_every_surface(self, _):
        for path, body in self.ROUTES:
            with self.subTest(path=path):
                self.assertEqual(self._post(path, body, self._unavailable()).status_code, 503)

    def test_each_surface_keeps_its_error_shape(self, _):
        r = self._post("/api/chat", self.OPENAI, self._unavailable())
        self.assertEqual(r.get_json(), {"error": "model 'm' is stopping on this node"})
        r = self._post("/v1/chat/completions", self.OPENAI, self._unavailable())
        self.assertEqual(r.get_json(), {"error": {"message": "model 'm' is stopping on this node",
                                                  "code": "model_stopping"}})

    def test_plain_errors_keep_their_old_status(self, _):
        r = self._post("/api/chat", self.OPENAI, "model 'm' not found")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json(), {"error": "model 'm' not found"})
        r = self._post("/v1/chat/completions", self.OPENAI, "model 'm' not found")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json(), {"error": {"message": "model 'm' not found"}})

    def test_the_model_limit_is_still_503_on_ollama(self, _):
        err = llamaman.ModelUnavailable("model_limit", "model limit reached (x)", retry_elsewhere=False)
        self.assertEqual(self._post("/api/chat", self.OPENAI, err).status_code, 503)

    def test_only_a_forwarding_peer_is_told_to_retry_elsewhere(self, _):
        for path, body in self.ROUTES:
            with self.subTest(path=path):
                r = self._post(path, body, self._unavailable(), dispatched=True)
                self.assertEqual(r.headers.get(RETRY_HEADER), "1")
                r = self._post(path, body, self._unavailable())
                self.assertIsNone(r.headers.get(RETRY_HEADER))

    def test_errors_that_arent_about_this_node_dont_fail_over(self, _):
        limit = llamaman.ModelUnavailable("model_limit", "model limit reached (x)", retry_elsewhere=False)
        for err in (limit, "model 'm' not found"):
            r = self._post("/api/chat", self.OPENAI, err, dispatched=True)
            self.assertIsNone(r.headers.get(RETRY_HEADER))


class ForwardFailoverTests(unittest.TestCase):

    NODE = {"node_id": "peer", "node_name": "peer"}

    def _forward(self, status, headers):
        resp = MagicMock(status_code=status, headers=headers, content=b"{}")
        app = Flask(__name__)
        with app.test_request_context("/v1/chat/completions", method="POST"), \
                patch("core.cluster.cluster_request", return_value=resp):
            before = cluster_api._inflight_get("peer")
            out = cluster_api._forward_inference(self.NODE, "/v1/chat/completions", b"{}", None)
            return out, resp, cluster_api._inflight_get("peer") - before

    def test_a_retry_elsewhere_503_tries_the_next_member(self):
        out, resp, inflight = self._forward(503, {RETRY_HEADER: "1"})
        self.assertIsNone(out)
        resp.close.assert_called()
        self.assertEqual(inflight, 0)

    def test_a_plain_503_is_returned_as_before(self):
        # An older peer, or a failure that isn't about this node.
        out, _, _ = self._forward(503, {"Content-Type": "application/json"})
        self.assertEqual(out.status_code, 503)

    def test_the_header_only_means_something_on_a_503(self):
        out, _, _ = self._forward(200, {RETRY_HEADER: "1", "Content-Type": "application/json"})
        self.assertEqual(out.status_code, 200)


class _FakeServer(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        payload = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


class OwnerEndToEndTests(_AliasCase):
    """The reported failure on a real route: a forwarded request for a sleeping
    member, answered by a real HTTP server once the wake (faked) brings it up."""

    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        member = _member("a", "sleeping")
        member.update(_server_host="127.0.0.1", _server_port=self.server.server_address[1])
        self._add(member)
        app = Flask(__name__)
        app.register_blueprint(llamaman.bp)
        self.client = app.test_client()

    def _post(self, path):
        with patch("api.llamaman.get_gate", return_value=None):
            return self.client.post(
                path, headers={"X-Cluster-Dispatch": "1"},
                json={"model": "chat-pool", "stream": False,
                      "messages": [{"role": "user", "content": "hi"}]})

    def test_openai_chat(self):
        r = self._post("/v1/chat/completions")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertIn("hello", r.get_data(as_text=True))
        self.wake.assert_called_once_with("a")

    def test_ollama_chat(self):
        r = self._post("/api/chat")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertIn("hello", r.get_data(as_text=True))
        self.wake.assert_called_once_with("a")

    def test_a_stopping_member_tells_the_forwarder_to_try_elsewhere(self):
        instances["a"]["status"] = "stopping"
        r = self._post("/api/chat")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get(RETRY_HEADER), "1")
        self.wake.assert_not_called()


class CapabilityTests(unittest.TestCase):

    def test_the_snapshot_advertises_alias_wake(self):
        snap = cluster_api.build_local_snapshot()
        self.assertIn("alias_wake", snap["system"]["features"])


if __name__ == "__main__":
    unittest.main()
