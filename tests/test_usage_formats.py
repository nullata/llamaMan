"""Token usage is recorded for every API the proxy relays, not only OpenAI
chat: Anthropic Messages, OpenAI Responses and System One (decision models)
report input_tokens / output_tokens."""

import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core.request_log import RecordingHandle, SSEAccumulator, normalize_usage
from proxy import _GATED_PATHS


def _stream(events, named=False):
    acc = SSEAccumulator()
    outputs = []
    for e in events:
        prefix = f"event: {e.get('type')}\n".encode() if named else b""
        outputs.append(acc.feed(prefix + b"data: " + json.dumps(e).encode() + b"\n\n"))
    return acc.finish(), outputs


class NormalizeUsageTests(unittest.TestCase):
    def test_openai_unchanged(self):
        u = {"prompt_tokens": 5, "completion_tokens": 2}
        self.assertEqual(normalize_usage(u), u)

    def test_system_one(self):
        u = normalize_usage({"input_tokens": 239, "output_tokens": 0})
        self.assertEqual((u["prompt_tokens"], u["completion_tokens"]), (239, 0))

    def test_anthropic_adds_cached_prompt(self):
        u = normalize_usage({"input_tokens": 10, "cache_read_input_tokens": 90,
                             "cache_creation_input_tokens": 5, "output_tokens": 7})
        self.assertEqual((u["prompt_tokens"], u["completion_tokens"]), (105, 7))

    def test_none_and_other(self):
        self.assertIsNone(normalize_usage(None))
        self.assertEqual(normalize_usage({"x": 1}), {"x": 1})


class StreamTests(unittest.TestCase):
    def test_anthropic_stream(self):
        (text, usage), outputs = _stream([
            {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ], named=True)
        self.assertEqual(text, "hi")
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"]), (10, 3))
        self.assertEqual(outputs, [False, False, True, True, False, False])

    def test_responses_stream(self):
        (text, usage), outputs = _stream([
            {"type": "response.created", "response": {"usage": None}},
            {"type": "response.reasoning_text.delta", "delta": "hm"},
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 8, "output_tokens": 2}}},
        ])
        self.assertEqual(text, "hi")
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"]), (8, 2))
        self.assertEqual(outputs, [False, True, True, False])

    def test_openai_chat_unchanged(self):
        (text, usage), _ = _stream([
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 4, "completion_tokens": 1}},
        ])
        self.assertEqual((text, usage), ("hi", {"prompt_tokens": 4, "completion_tokens": 1}))


class RecordTests(unittest.TestCase):
    def test_set_response_records_input_output(self):
        h = RecordingHandle({}, "full", 0.0)
        h.set_response(usage={"input_tokens": 239, "output_tokens": 0}, status_code=200)
        self.assertEqual((h._record["prompt_tokens"], h._record["completion_tokens"]), (239, 0))


class GatedPathTests(unittest.TestCase):
    def test_paths(self):
        for p in ("/v1/systemone", "/v1/messages", "/v1/responses"):
            self.assertIn(p, _GATED_PATHS)


if __name__ == "__main__":
    unittest.main()
