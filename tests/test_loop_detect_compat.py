# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Integration tests for the loop-detection fork wired into the Ollama /
OpenAI compat streaming generators in api/llamaman.py.

We test _stream_llamaman() directly (an Ollama-native streaming generator)
because it's the simplest reachable path. The /v1/chat/completions and
/v1/completions inline _relay() closures follow the same shape - if
_stream_llamaman works, they work.

Full end-to-end HTTP tests through Flask are avoided: the Flask setup for
those routes requires model discovery, ensure_running, and other layers of
infrastructure that are irrelevant to the fork's contract."""

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
from core.loop_detect import active_buffer_count
from core.state import instances, instances_lock


def _openai_upstream_line(content: str = "", reasoning: str = "") -> str:
    delta = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    obj = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(obj)}\n"


class OllamaStreamLoopFork(unittest.TestCase):
    """The Ollama-native stream (_stream_llamaman) attaches a TurnBuffer when
    the instance's preset has loop_detect_enabled, feeds each delta's
    content + thinking into it, and on detection yields the Ollama-shaped
    loop_detected terminator + stops iterating."""

    def setUp(self):
        ld._reset_for_tests()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()
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
                    "loop_detect_scan_every_n_tokens": 1,  # fire per-token
                    "loop_detect_scan_interval_s": 60,
                },
                "stats": {},
            }

    def tearDown(self):
        ld._reset_for_tests()
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    def _fake_upstream(self, lines: list[str], status_code: int = 200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.reason = "OK"
        resp.encoding = "utf-8"
        resp.iter_lines.return_value = iter(lines)
        resp.close = MagicMock()
        return resp

    def test_looping_stream_terminates_with_ollama_done_reason(self):
        from api.llamaman import _stream_llamaman

        loop_content = "L" * 200
        upstream_lines = [_openai_upstream_line(content=loop_content) for _ in range(4)]
        resp = self._fake_upstream(upstream_lines)

        with patch("api.llamaman.request_local_worker", return_value=resp):
            gen = _stream_llamaman(
                host="localhost", port=9000,
                openai_body={"model": "chat", "stream": True},
                model_name="chat", mode="chat", inst_id="inst-1",
            )
            emitted = list(gen)

        # Look for the Ollama loop_detected terminator by decoding each
        # ndjson line - the message text is JSON-escaped inside "content".
        found_terminator = False
        for line in emitted:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done") and obj.get("done_reason") == "loop_detected":
                self.assertIn("[llamaman: output loop detected", obj["message"]["content"])
                found_terminator = True
                break
        self.assertTrue(found_terminator, "no ollama loop_detected terminator emitted")
        # Buffer cleaned up in the finally: block.
        self.assertEqual(active_buffer_count(), 0)
        # Upstream closed.
        resp.close.assert_called()

    def test_non_looping_stream_relayed_and_buffer_detached(self):
        from api.llamaman import _stream_llamaman

        # Unique deltas + a normal finish frame.
        upstream_lines = [
            _openai_upstream_line(content=f"unique-{i}-" + "z" * 190)
            for i in range(3)
        ]
        # Terminating frame with finish_reason
        upstream_lines.append(
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        )
        resp = self._fake_upstream(upstream_lines)

        with patch("api.llamaman.request_local_worker", return_value=resp):
            gen = _stream_llamaman(
                host="localhost", port=9000,
                openai_body={"model": "chat", "stream": True},
                model_name="chat", mode="chat", inst_id="inst-1",
            )
            emitted = list(gen)

        # No loop_detected done_reason should appear.
        for line in emitted:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done"):
                self.assertNotEqual(obj.get("done_reason"), "loop_detected")
        self.assertEqual(active_buffer_count(), 0)

    def test_reasoning_content_loop_also_terminated(self):
        # A model looping in the thinking phase must be caught same as
        # content-phase loops.
        from api.llamaman import _stream_llamaman

        loop_reasoning = "R" * 200
        upstream_lines = [_openai_upstream_line(reasoning=loop_reasoning) for _ in range(4)]
        resp = self._fake_upstream(upstream_lines)

        with patch("api.llamaman.request_local_worker", return_value=resp):
            gen = _stream_llamaman(
                host="localhost", port=9000,
                openai_body={"model": "chat", "stream": True},
                model_name="chat", mode="chat", inst_id="inst-1",
            )
            emitted = list(gen)

        found_terminator = False
        for line in emitted:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done") and obj.get("done_reason") == "loop_detected":
                found_terminator = True
                break
        self.assertTrue(found_terminator, "thinking-phase loop was not detected")

    def test_disabled_preset_never_attaches_buffer(self):
        from api.llamaman import _stream_llamaman

        with instances_lock:
            instances["inst-1"]["config"]["loop_detect_enabled"] = False

        upstream_lines = [_openai_upstream_line(content="A" * 200) for _ in range(4)]
        resp = self._fake_upstream(upstream_lines)

        with patch("api.llamaman.request_local_worker", return_value=resp), \
             patch("api.llamaman._loop_detect_attach", wraps=ld.attach) as mock_attach:
            gen = _stream_llamaman(
                host="localhost", port=9000,
                openai_body={"model": "chat", "stream": True},
                model_name="chat", mode="chat", inst_id="inst-1",
            )
            list(gen)

        # attach() IS called (the fork always calls it and relies on the
        # helper to return None when disabled), but the returned buf is None
        # so no buffer ever landed in the registry.
        mock_attach.assert_called_once()
        self.assertEqual(active_buffer_count(), 0)

    def test_generate_mode_terminator_uses_response_field(self):
        # mode='generate' - Ollama uses "response" instead of "message".
        from api.llamaman import _stream_llamaman

        upstream_lines = [_openai_upstream_line(content="G" * 200) for _ in range(4)]
        resp = self._fake_upstream(upstream_lines)

        with patch("api.llamaman.request_local_worker", return_value=resp):
            gen = _stream_llamaman(
                host="localhost", port=9000,
                openai_body={"model": "chat", "stream": True},
                model_name="chat", mode="generate", inst_id="inst-1",
            )
            emitted = list(gen)

        found = False
        for line in emitted:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done") and obj.get("done_reason") == "loop_detected":
                self.assertIn("response", obj)
                self.assertNotIn("message", obj)
                found = True
                break
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
