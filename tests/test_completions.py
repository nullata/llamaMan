# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Recording/extraction for the raw completion endpoints (/v1/completions and
/completion). The handlers need a live llama-server to exercise end to end, but
the format-handling pieces are pure and worth pinning down per shape:
OpenAI legacy (choices[].text), chat (choices[].message/delta), llama.cpp
native (top-level content)."""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core.request_log import SSEAccumulator
from api.llamaman import _extract_completion_text, _completion_usage


def _sse(*objs: str) -> bytes:
    return b"".join(f"data: {o}\n\n".encode() for o in objs)


class SSEAccumulatorFormatTests(unittest.TestCase):
    def test_chat_delta_still_works(self):
        acc = SSEAccumulator()
        acc.feed(_sse('{"choices":[{"delta":{"content":"He"}}]}',
                      '{"choices":[{"delta":{"content":"llo"}}],"usage":{"completion_tokens":2}}',
                      '[DONE]'))
        text, usage = acc.finish()
        self.assertEqual(text, "Hello")
        self.assertEqual(usage, {"completion_tokens": 2})

    def test_legacy_text_format(self):
        acc = SSEAccumulator()
        acc.feed(_sse('{"choices":[{"text":"Hel"}]}',
                      '{"choices":[{"text":"lo"}]}', '[DONE]'))
        text, usage = acc.finish()
        self.assertEqual(text, "Hello")
        self.assertIsNone(usage)

    def test_native_content_and_token_usage(self):
        acc = SSEAccumulator()
        acc.feed(_sse('{"content":"Hel","stop":false}',
                      '{"content":"lo","stop":false}',
                      '{"content":"","stop":true,"tokens_predicted":2,"tokens_evaluated":5}'))
        text, usage = acc.finish()
        self.assertEqual(text, "Hello")
        self.assertEqual(usage, {"completion_tokens": 2, "prompt_tokens": 5, "total_tokens": 7})

    def test_split_across_chunk_boundaries(self):
        acc = SSEAccumulator()
        raw = _sse('{"choices":[{"text":"abc"}]}')
        acc.feed(raw[:7])
        acc.feed(raw[7:])
        self.assertEqual(acc.finish()[0], "abc")


class CompletionExtractTests(unittest.TestCase):
    def test_extract_text_all_shapes(self):
        self.assertEqual(_extract_completion_text({"choices": [{"text": "abc"}]}), "abc")
        self.assertEqual(_extract_completion_text({"choices": [{"message": {"content": "xyz"}}]}), "xyz")
        self.assertEqual(_extract_completion_text({"content": "native"}), "native")
        self.assertEqual(_extract_completion_text({}), "")
        self.assertEqual(_extract_completion_text(None), "")

    def test_usage_openai_passthrough(self):
        u = {"completion_tokens": 3, "prompt_tokens": 1}
        self.assertEqual(_completion_usage({"usage": u}), u)

    def test_usage_native_mapping(self):
        self.assertEqual(
            _completion_usage({"tokens_predicted": 4, "tokens_evaluated": 6}),
            {"completion_tokens": 4, "prompt_tokens": 6, "total_tokens": 10},
        )

    def test_usage_absent(self):
        self.assertIsNone(_completion_usage({}))
        self.assertIsNone(_completion_usage(None))


if __name__ == "__main__":
    unittest.main()
