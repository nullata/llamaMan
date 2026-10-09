"""SSEAccumulator.feed reports the first generated output, not the first bytes:
Strata sends an empty role chunk and keep-alive comments while it reads the
prompt, which made the recorded TTFT ~0 and tokens/s too low."""

import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core.request_log import SSEAccumulator


def _data(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _chunk(delta, finish=None, **extra):
    return _data({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra})


class FeedReportsOutputTests(unittest.TestCase):
    def test_strata_prelude_is_not_output(self):
        acc = SSEAccumulator()
        self.assertFalse(acc.feed(_chunk({"role": "assistant", "content": ""})))
        self.assertFalse(acc.feed(b": keep-alive\n\n"))
        self.assertFalse(acc.feed(_data({**json.loads(_chunk({})[6:]), "prompt_progress": {"total": 9}})))
        self.assertTrue(acc.feed(_chunk({"content": "hi"})))

    def test_reasoning_and_tool_calls_are_output(self):
        self.assertTrue(SSEAccumulator().feed(_chunk({"reasoning_content": "hm"})))
        self.assertTrue(SSEAccumulator().feed(_chunk({"tool_calls": [{"index": 0, "id": "c"}]})))

    def test_completions_and_native_text_are_output(self):
        self.assertTrue(SSEAccumulator().feed(_data({"choices": [{"text": "x"}]})))
        self.assertTrue(SSEAccumulator().feed(_data({"content": "x"})))

    def test_final_usage_chunk_still_recorded(self):
        acc = SSEAccumulator()
        acc.feed(_chunk({"content": "hi"}))
        self.assertFalse(acc.feed(_chunk({}, "stop", usage={"prompt_tokens": 10, "completion_tokens": 2})))
        self.assertEqual(acc.finish(), ("hi", {"prompt_tokens": 10, "completion_tokens": 2}))

    def test_split_chunk(self):
        acc = SSEAccumulator()
        raw = _chunk({"content": "hi"})
        self.assertFalse(acc.feed(raw[:10]))
        self.assertTrue(acc.feed(raw[10:]))


if __name__ == "__main__":
    unittest.main()
