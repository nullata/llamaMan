# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Integration tests for the loop-detection fork wired into the per-instance
sidecar proxy (proxy/__init__.py's _relay_and_close).

These sit on top of the pure-algorithm tests in test_loop_detect.py: they
prove the fork correctly ATTACHES/FEEDS/DETACHES a buffer on the sidecar
proxy's streaming iterator and INJECTS the synthetic terminator + stops
relaying on detection.

We drive make_proxy_app directly rather than through Werkzeug's HTTP server -
the WSGI contract (environ + start_response + iterator) is stable enough to
test that way without spinning a real port."""

import io
import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core import loop_detect as ld
from core.loop_detect import LOOP_TERMINATION_MESSAGE, active_buffer_count
from core.state import instances, instances_lock
from proxy import make_proxy_app


def _openai_sse_line(content: str) -> bytes:
    obj = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _wsgi_environ(path: str = "/v1/chat/completions", body: dict | None = None) -> dict:
    body_bytes = json.dumps(body or {"model": "chat"}).encode("utf-8")
    return {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_LENGTH": str(len(body_bytes)),
        "CONTENT_TYPE": "application/json",
        "wsgi.input": io.BytesIO(body_bytes),
        "HTTP_AUTHORIZATION": "",
    }


class SidecarProxyLoopFork(unittest.TestCase):
    """The sidecar fork attaches a buffer on SSE responses when the instance
    has loop_detect_enabled, injects a synthetic terminator into the client
    stream on detection, and stops relaying further chunks from llama-server."""

    def setUp(self):
        ld._reset_for_tests()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()
        # A healthy instance with loop-detect on and a low inline scan
        # threshold so the fork fires deterministically inside one request.
        with instances_lock:
            instances["inst-1"] = {
                "id": "inst-1",
                "model_name": "chat",
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "status": "healthy",
                "container_id": "c1",
                "container_name": "llamaman-c1",
                "log_file": "",
                "started_at": time.time(),
                "_last_request_at": time.time(),
                "_server_host": "localhost",
                "_server_port": 9000,
                "_internal_port": 9000,
                "config": {
                    "loop_detect_enabled": True,
                    "loop_detect_min_chunk_chars": 200,
                    "loop_detect_min_repetitions": 3,
                    "loop_detect_max_buffer_chars": 4096,
                    "loop_detect_scan_every_n_tokens": 1,  # scan on every chunk
                    "loop_detect_scan_interval_s": 60,
                },
                "stats": {},
            }

    def tearDown(self):
        ld._reset_for_tests()
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    def _fake_upstream_response(self, sse_chunks: list[bytes], status_code: int = 200):
        """Build a MagicMock that behaves like a requests.Response with SSE
        content-type and iter_content yielding the supplied chunks."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.reason = "OK"
        resp.headers = {"Content-Type": "text/event-stream"}
        resp.iter_content.return_value = iter(sse_chunks)
        resp.close = MagicMock()
        return resp

    @patch("proxy._check_proxy_auth", return_value=False)
    @patch("proxy.request_local_worker")
    def test_looping_stream_is_terminated_with_openai_terminator(
        self, mock_worker, _mock_auth,
    ):
        # A stream of 4 identical 200-char content deltas. With
        # min_chunk=200 / min_reps=3 / scan_every_n=1, detection fires by
        # the 3rd chunk.
        loop_content = "L" * 200
        chunks = [_openai_sse_line(loop_content) for _ in range(4)]
        mock_worker.return_value = self._fake_upstream_response(chunks)

        app = make_proxy_app("inst-1", internal_port=9000, proxy_port=8000)
        env = _wsgi_environ("/v1/chat/completions", {"model": "chat"})
        start_response = MagicMock()
        body_iter = app(env, start_response)

        # Drain the iterator - it should stop early with a synthetic
        # terminator instead of the full 4 chunks.
        emitted = b"".join(body_iter)

        # The terminator carries the fixed marker text.
        # LOOP_TERMINATION_MESSAGE lives INSIDE a JSON-encoded delta so its
        # newlines are escaped as \\n by json.dumps. Look for the invariant
        # substring "[llamaman: output loop detected" instead - both raw and
        # JSON-encoded forms contain it verbatim.
        self.assertIn(b"[llamaman: output loop detected", emitted)
        # And the SSE terminator sentinel.
        self.assertIn(b"data: [DONE]\n\n", emitted)
        # Upstream must have been closed (finally: block ran).
        mock_worker.return_value.close.assert_called()
        # Buffer detached.
        self.assertEqual(active_buffer_count(), 0)

    @patch("proxy._check_proxy_auth", return_value=False)
    @patch("proxy.request_local_worker")
    def test_normal_stream_is_relayed_verbatim_and_buffer_detached(
        self, mock_worker, _mock_auth,
    ):
        # Non-looping content: every delta is unique. The fork must attach,
        # feed each chunk, never fire, and detach in finally.
        chunks = [_openai_sse_line(f"unique-{i}-" + "z" * 190) for i in range(4)]
        chunks.append(b"data: [DONE]\n\n")
        mock_worker.return_value = self._fake_upstream_response(chunks)

        app = make_proxy_app("inst-1", internal_port=9000, proxy_port=8000)
        env = _wsgi_environ("/v1/chat/completions", {"model": "chat"})
        start_response = MagicMock()
        body_iter = app(env, start_response)
        emitted = b"".join(body_iter)

        # All 4 unique deltas made it through; no synthetic terminator was
        # injected (the terminator marker text is absent).
        for i in range(4):
            self.assertIn(f"unique-{i}-".encode(), emitted)
        self.assertNotIn(b"[llamaman: output loop detected", emitted)
        self.assertEqual(active_buffer_count(), 0)

    @patch("proxy._check_proxy_auth", return_value=False)
    @patch("proxy.request_local_worker")
    def test_loop_detect_disabled_no_buffer_attached(self, mock_worker, _mock_auth):
        # With the toggle off, attach() returns None, no buffer registered.
        # Sanity-check that path.
        with instances_lock:
            instances["inst-1"]["config"]["loop_detect_enabled"] = False

        chunks = [_openai_sse_line("A" * 200) for _ in range(4)]
        mock_worker.return_value = self._fake_upstream_response(chunks)

        app = make_proxy_app("inst-1", internal_port=9000, proxy_port=8000)
        env = _wsgi_environ("/v1/chat/completions", {"model": "chat"})
        start_response = MagicMock()
        body_iter = app(env, start_response)
        emitted = b"".join(body_iter)

        # No terminator, all 4 chunks pass through unchanged.
        self.assertNotIn(b"[llamaman: output loop detected", emitted)
        self.assertEqual(active_buffer_count(), 0)

    @patch("proxy._check_proxy_auth", return_value=False)
    @patch("proxy.request_local_worker")
    def test_broken_extractor_does_not_break_stream(self, mock_worker, _mock_auth):
        # If the SSE extractor / feed / scan raises for any reason, the
        # stream must keep flowing - the whole point of the try/except
        # wrappers around the fork.
        chunks = [_openai_sse_line("A" * 200) for _ in range(4)]
        mock_worker.return_value = self._fake_upstream_response(chunks)

        with patch("proxy._loop_detect_feed", side_effect=RuntimeError("boom")):
            app = make_proxy_app("inst-1", internal_port=9000, proxy_port=8000)
            env = _wsgi_environ("/v1/chat/completions", {"model": "chat"})
            start_response = MagicMock()
            body_iter = app(env, start_response)
            emitted = b"".join(body_iter)

        # All chunks should still have been relayed unchanged.
        for chunk in chunks:
            self.assertIn(chunk, emitted)
        # No terminator was injected (the fork bailed defensively).
        self.assertNotIn(b"[llamaman: output loop detected", emitted)
        # Buffer still cleaned up.
        self.assertEqual(active_buffer_count(), 0)

    @patch("proxy._check_proxy_auth", return_value=False)
    @patch("proxy.request_local_worker")
    def test_non_sse_response_never_attaches_buffer(self, mock_worker, _mock_auth):
        # A non-streaming JSON response goes through the buf/BUF_CAP branch,
        # never touches the loop-detect fork. attach() must not be called.
        resp = MagicMock()
        resp.status_code = 200
        resp.reason = "OK"
        resp.headers = {"Content-Type": "application/json"}
        resp.iter_content.return_value = iter([b'{"ok":true}'])
        resp.close = MagicMock()
        mock_worker.return_value = resp

        with patch("proxy._loop_detect_attach") as mock_attach:
            app = make_proxy_app("inst-1", internal_port=9000, proxy_port=8000)
            env = _wsgi_environ("/v1/chat/completions", {"model": "chat"})
            start_response = MagicMock()
            list(app(env, start_response))  # drain

        mock_attach.assert_not_called()
        self.assertEqual(active_buffer_count(), 0)


if __name__ == "__main__":
    unittest.main()
