# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the auto model output loop detector (core/loop_detect.py).

Covers:
  - parse_loop_detect_config boundary validation
  - scan_text pure algorithm: real loops, code with for-loops, poetry
    choruses, markdown tables, numbered lists, drifting near-repeats
  - TurnBuffer rolling cap + thread-safety
  - attach/feed/detach registry lifecycle
  - Worker tick: only scans past-interval buffers, skips empty/cancelled
  - Defensive: a broken scan MUST NOT propagate an exception up the
    streaming call site
  - SSETextExtractor: split-mid-chunk boundaries, [DONE], reasoning_content,
    llama.cpp native format
  - Synthetic terminators produce valid parseable payloads
"""

import os
import threading
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core import loop_detect as ld
from core.loop_detect import (
    LOOP_DETECT_KEYS, LOOP_TERMINATION_MESSAGE, SSETextExtractor,
    TurnBuffer, active_buffer_count, attach, detach, feed,
    loop_detect_enabled, make_ollama_terminator, make_openai_sse_terminator,
    parse_loop_detect_config, scan_text, worker_tick,
)


class ParseLoopDetectConfigTests(unittest.TestCase):
    def test_empty_body_gives_disabled_defaults(self):
        cfg, err = parse_loop_detect_config({})
        self.assertIsNone(err)
        self.assertFalse(cfg["loop_detect_enabled"])
        self.assertEqual(cfg["loop_detect_min_chunk_chars"], 200)
        self.assertEqual(cfg["loop_detect_min_repetitions"], 3)
        self.assertEqual(cfg["loop_detect_max_buffer_chars"], 8192)
        self.assertEqual(cfg["loop_detect_scan_interval_s"], 10)
        self.assertEqual(cfg["loop_detect_scan_every_n_tokens"], 64)

    def test_typical_enabled_values(self):
        cfg, err = parse_loop_detect_config({
            "loop_detect_enabled": True,
            "loop_detect_min_chunk_chars": 150,
            "loop_detect_min_repetitions": 4,
        })
        self.assertIsNone(err)
        self.assertTrue(cfg["loop_detect_enabled"])
        self.assertEqual(cfg["loop_detect_min_chunk_chars"], 150)
        self.assertEqual(cfg["loop_detect_min_repetitions"], 4)

    def test_min_chunk_below_floor_rejected(self):
        # Sub-60 char chunks false-positive on real content (numbered lists,
        # short choruses). Reject at the boundary.
        cfg, err = parse_loop_detect_config({"loop_detect_min_chunk_chars": 30})
        self.assertEqual(cfg, {})
        self.assertIn("loop_detect_min_chunk_chars", err)

    def test_min_repetitions_below_2_rejected(self):
        cfg, err = parse_loop_detect_config({"loop_detect_min_repetitions": 1})
        self.assertEqual(cfg, {})
        self.assertIn("loop_detect_min_repetitions", err)

    def test_buffer_too_small_for_chunk_x_reps_rejected(self):
        # A detection window that couldn't hold the pattern even once would
        # silently do nothing - reject so the user isn't running with
        # detection effectively off.
        cfg, err = parse_loop_detect_config({
            "loop_detect_min_chunk_chars": 200,
            "loop_detect_min_repetitions": 5,
            "loop_detect_max_buffer_chars": 512,   # < 200 * 5 = 1000
        })
        self.assertEqual(cfg, {})
        self.assertIn("loop_detect_max_buffer_chars", err)

    def test_non_integer_rejected(self):
        cfg, err = parse_loop_detect_config({"loop_detect_min_chunk_chars": "big"})
        self.assertEqual(cfg, {})

    def test_scan_interval_bounds(self):
        cfg, err = parse_loop_detect_config({"loop_detect_scan_interval_s": 0})
        self.assertEqual(cfg, {})
        cfg, err = parse_loop_detect_config({"loop_detect_scan_interval_s": 10000})
        self.assertEqual(cfg, {})

    def test_scan_every_n_bounds(self):
        cfg, err = parse_loop_detect_config({"loop_detect_scan_every_n_tokens": 1})
        self.assertEqual(cfg, {})
        cfg, err = parse_loop_detect_config({"loop_detect_scan_every_n_tokens": 999999})
        self.assertEqual(cfg, {})


class LoopDetectEnabledHelperTests(unittest.TestCase):
    def test_none(self):
        self.assertFalse(loop_detect_enabled(None))

    def test_empty(self):
        self.assertFalse(loop_detect_enabled({}))

    def test_disabled(self):
        self.assertFalse(loop_detect_enabled({"loop_detect_enabled": False}))

    def test_enabled(self):
        self.assertTrue(loop_detect_enabled({"loop_detect_enabled": True}))


class ScanTextTests(unittest.TestCase):
    """The pure detection function. If this passes on realistic content
    patterns, everything downstream is just plumbing."""

    def test_empty_text_is_not_a_loop(self):
        self.assertFalse(scan_text("", 200, 3))

    def test_short_text_is_not_a_loop(self):
        self.assertFalse(scan_text("hi", 200, 3))

    def test_clear_loop_is_detected(self):
        # A 200-char chunk repeated 4 times = 800 chars of pure loop.
        chunk = "X" * 200
        text = chunk * 4
        self.assertTrue(scan_text(text, 200, 3))

    def test_two_reps_below_threshold_missed(self):
        # min_repetitions=3 must NOT flag on 2 reps - the user set that
        # threshold deliberately.
        chunk = "Y" * 200
        text = chunk * 2
        self.assertFalse(scan_text(text, 200, 3))

    def test_exact_threshold_is_a_loop(self):
        chunk = "Z" * 200
        text = chunk * 3
        self.assertTrue(scan_text(text, 200, 3))

    def test_realistic_loop_text(self):
        # A stuck model repeating a full-paragraph unit >= min_chunk.
        # The detector requires the repeating period p to be >= min_chunk;
        # a shorter period p only produces count-1 matches at min_chunk
        # (because the tail chunk straddles two reps), so with default
        # thresholds a paragraph-length loop needs each paragraph to be
        # >= 200 chars OR to need enough reps that even the straddled
        # target hits min_repetitions matches.
        loop_paragraph = (
            "I apologize for any confusion I may have caused with my previous "
            "response. Let me try to clarify the situation more carefully. "
            "Actually, upon further reflection, the correct answer is that "
            "the answer depends on the specific context of your question. "
            "Let me think about this more carefully.\n"
        )
        self.assertGreaterEqual(len(loop_paragraph), 200)
        text = loop_paragraph * 4
        self.assertTrue(scan_text(text, 200, 3))

    def test_short_period_loop_at_defaults_needs_more_reps(self):
        # A sentence shorter than min_chunk repeats fine, but the tail chunk
        # only exact-matches (min_chunk // period) + 1 times shifted, so we
        # need enough reps to still hit min_repetitions. This is a real
        # property of the exact-match algorithm - users who care about
        # short-period loops should set a smaller min_chunk (down to the
        # 60-char parse floor) or rely on DRY sampling for that regime.
        sentence = "A" * 100 + " and then some more filler text goes here. "  # ~145 chars
        # With period=145, min_chunk=200, target has ~55 chars of prev rep +
        # 145 of current. Each shifted rep boundary matches: (n_reps - 1)
        # matches for n_reps repetitions.
        text = sentence * 6  # target matches 5 times, > min_reps=3
        self.assertTrue(scan_text(text, 200, 3))

        text = sentence * 3  # only 2 boundary matches, below min_reps=3
        self.assertFalse(scan_text(text, 200, 3))

    def test_code_with_for_loops_not_flagged(self):
        # 3 near-identical for-loops separated by different bodies. Each
        # for-loop line is < 60 chars so wouldn't dominate a 200-char chunk,
        # AND the loops don't repeat verbatim.
        code = (
            "for i in range(10):\n    total += arr[i]\n\n"
            "for j in range(10):\n    total -= brr[j]\n\n"
            "for k in range(10):\n    total *= crr[k]\n\n"
        )
        # Repeat 3 times to fill the buffer
        text = (code + "some other work here.\n") * 3
        # With realistic thresholds this should NOT flag - the pattern
        # isn't a stable attractor.
        self.assertFalse(scan_text(text, 200, 3))

    def test_poetry_chorus_not_flagged_at_defaults(self):
        # A chorus that repeats but each rep is short + interleaved with
        # different verses - the last 200 chars are a mix, not a stable
        # repeating chunk.
        verse1 = "There was a rider, tall and lean, / who galloped through the meadow green.\n"
        chorus = "And on and on and on he rode, / into the fading twilight glow.\n"
        verse2 = "The moon rose high above the plain, / and dew fell soft as gentle rain.\n"
        text = verse1 + chorus + verse2 + chorus + verse1 + chorus
        # Chorus is under 100 chars so a 200-char tail includes surrounding
        # verse; the chorus alone doesn't dominate.
        self.assertFalse(scan_text(text, 200, 3))

    def test_markdown_table_header_not_flagged_at_defaults(self):
        # A 5-column markdown table with 8 rows. Rows have distinct data but
        # the "|---|" separator is common. Modest length; the last 200 chars
        # are a MIX of rows, not one repeated row.
        table = "| A | B | C | D | E |\n|---|---|---|---|---|\n"
        for i in range(8):
            table += f"| {i} | b{i} | c{i} | d{i} | e{i} |\n"
        self.assertFalse(scan_text(table, 200, 3))

    def test_numbered_list_not_flagged(self):
        # A 20-item numbered list. Each item is distinct.
        items = [f"{i}. Item number {i} with some description about it.\n"
                 for i in range(1, 21)]
        text = "".join(items)
        self.assertFalse(scan_text(text, 200, 3))

    def test_drifting_near_repeats_missed_by_v1(self):
        # v1 uses exact-match; near-repeats with punctuation drift are
        # deliberately NOT caught. Documented in the module docstring.
        base = "The answer is: yes"
        drift = "The answer is: yes."
        # Alternate between the two - each 200-char tail won't have 3 exact reps.
        text = ((base + " ") * 100 + drift) * 3
        # Whether flagged depends on the mix; the assertion here is that we
        # AT LEAST don't crash. We don't guarantee detection for drift.
        _ = scan_text(text, 200, 3)  # noqa: F841

    def test_thinking_loop_still_flagged(self):
        # A model looping in the reasoning phase produces content that looks
        # like normal text to the detector - same detection either way.
        chunk = "Wait, let me reconsider that. Actually, I need to think about this more carefully. "
        text = chunk * 10
        self.assertTrue(scan_text(text, 60, 5))


class TurnBufferTests(unittest.TestCase):
    def test_default_config_gives_defaults(self):
        buf = TurnBuffer("sid", "iid", {})
        self.assertEqual(buf.min_chunk, 200)
        self.assertEqual(buf.min_reps, 3)
        self.assertEqual(buf.max_buffer, 8192)
        self.assertEqual(buf.scan_every_n, 64)

    def test_config_overrides_apply(self):
        buf = TurnBuffer("sid", "iid", {
            "loop_detect_min_chunk_chars": 150,
            "loop_detect_min_repetitions": 4,
            "loop_detect_max_buffer_chars": 4096,
            "loop_detect_scan_every_n_tokens": 32,
            "loop_detect_scan_interval_s": 5,
        })
        self.assertEqual(buf.min_chunk, 150)
        self.assertEqual(buf.min_reps, 4)
        self.assertEqual(buf.max_buffer, 4096)
        self.assertEqual(buf.scan_every_n, 32)
        self.assertEqual(buf.scan_interval_s, 5)


class AttachFeedDetachTests(unittest.TestCase):
    def setUp(self):
        ld._reset_for_tests()

    def tearDown(self):
        ld._reset_for_tests()

    def test_attach_returns_none_when_disabled(self):
        buf = attach("iid", {"loop_detect_enabled": False})
        self.assertIsNone(buf)
        self.assertEqual(active_buffer_count(), 0)

    def test_attach_returns_none_when_config_missing(self):
        self.assertIsNone(attach("iid", None))
        self.assertIsNone(attach("iid", {}))

    def test_attach_registers_when_enabled(self):
        buf = attach("iid", {"loop_detect_enabled": True})
        self.assertIsNotNone(buf)
        self.assertEqual(active_buffer_count(), 1)
        detach(buf)
        self.assertEqual(active_buffer_count(), 0)

    def test_detach_is_idempotent(self):
        buf = attach("iid", {"loop_detect_enabled": True})
        detach(buf)
        detach(buf)   # second call is a no-op, not a crash
        detach(None)  # None is safe
        self.assertEqual(active_buffer_count(), 0)

    def test_feed_none_buffer_is_noop(self):
        # feed() must accept a None buffer (loop-detect off for the request)
        # so callers can pass unconditionally.
        self.assertFalse(feed(None, "any text"))

    def test_feed_below_inline_threshold_does_not_scan(self):
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_scan_every_n_tokens": 10,
        })
        for _ in range(9):
            self.assertFalse(feed(buf, "X" * 100))
        # 9 chunks under threshold of 10 - no scan means no detection
        # even though the text obviously repeats.
        self.assertFalse(buf.cancel_flag.is_set())
        detach(buf)

    def test_feed_triggers_scan_at_threshold(self):
        # min_chunk=200, min_reps=3, scan_every_n=3 means the 3rd chunk of a
        # repeating 200-char pattern gets scanned and detected.
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_min_chunk_chars": 200,
            "loop_detect_min_repetitions": 3,
            "loop_detect_scan_every_n_tokens": 3,
            "loop_detect_max_buffer_chars": 4096,
        })
        chunk = "L" * 200
        # Feed 3 chunks - the 3rd triggers a scan on 600 chars of pure loop.
        self.assertFalse(feed(buf, chunk))
        self.assertFalse(feed(buf, chunk))
        self.assertTrue(feed(buf, chunk))
        self.assertTrue(buf.cancel_flag.is_set())
        detach(buf)

    def test_rolling_cap_never_exceeded(self):
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_max_buffer_chars": 1024,
            "loop_detect_min_chunk_chars": 60,
            "loop_detect_min_repetitions": 3,
            "loop_detect_scan_every_n_tokens": 5,
        })
        # Feed 100 KB of unique text; buffer must stay <= 1024 chars.
        for i in range(100):
            feed(buf, chr(ord("A") + (i % 26)) * 1024)
        with buf._lock:
            total = sum(len(c) for c in buf._chars)
        self.assertLessEqual(total, 1024)
        detach(buf)

    def test_feed_exception_does_not_propagate(self):
        # A broken scan MUST not break the stream - callers must be able to
        # trust feed() to return a bool no matter what.
        buf = attach("iid", {"loop_detect_enabled": True})
        with patch("core.loop_detect.scan_text", side_effect=RuntimeError("boom")):
            # Force a scan by setting the counter high
            with buf._lock:
                buf.tokens_since_last_scan = buf.scan_every_n - 1
            result = feed(buf, "some text")
        self.assertFalse(result)
        self.assertFalse(buf.cancel_flag.is_set())
        detach(buf)

    def test_thread_safety_of_feed(self):
        # Multiple threads feeding the same buffer must not corrupt state.
        # (In real usage one buffer belongs to one stream, but if it were
        # ever concurrent - and the LOCK exists precisely for that - it has
        # to be safe.)
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_scan_every_n_tokens": 1000,  # no scans triggered
        })

        def worker():
            for _ in range(200):
                feed(buf, "X")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with buf._lock:
            total = sum(len(c) for c in buf._chars)
        # 8 workers x 200 chunks of "X" = 1600 chars. Buffer default max is
        # 8192, well above 1600, so nothing was dropped.
        self.assertEqual(total, 1600)
        detach(buf)


class WorkerTickTests(unittest.TestCase):
    def setUp(self):
        ld._reset_for_tests()

    def tearDown(self):
        ld._reset_for_tests()

    def test_empty_registry_is_cheap_noop(self):
        self.assertEqual(worker_tick(), 0)

    def test_skips_buffer_whose_interval_has_not_elapsed(self):
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_scan_interval_s": 60,
        })
        feed(buf, "a")
        # last_scan_at is very recent
        self.assertEqual(worker_tick(), 0)
        detach(buf)

    def test_scans_past_interval_buffer(self):
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_scan_interval_s": 1,
            "loop_detect_min_chunk_chars": 200,
            "loop_detect_min_repetitions": 3,
            "loop_detect_scan_every_n_tokens": 10000,  # inline scan disabled
        })
        # Feed a full loop; inline scan won't trigger because we set the
        # threshold impossibly high.
        chunk = "W" * 200
        for _ in range(3):
            feed(buf, chunk)
        self.assertFalse(buf.cancel_flag.is_set())  # no inline scan
        # Fake elapsed time
        import time as _t
        buf.last_scan_at = _t.monotonic() - 5
        detected = worker_tick()
        self.assertEqual(detected, 1)
        self.assertTrue(buf.cancel_flag.is_set())
        detach(buf)

    def test_skips_already_cancelled_buffer(self):
        buf = attach("iid", {
            "loop_detect_enabled": True,
            "loop_detect_scan_interval_s": 1,
        })
        buf.cancel_flag.set()
        import time as _t
        buf.last_scan_at = _t.monotonic() - 100
        # Even a stale + long buffer that's already cancelled must not
        # re-scan (already handled).
        self.assertEqual(worker_tick(), 0)
        detach(buf)


class SSETextExtractorTests(unittest.TestCase):
    def test_openai_chat_deltas(self):
        x = SSETextExtractor()
        chunk = (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":", world"}}]}\n\n'
        )
        text = x.extract(chunk)
        self.assertEqual(text, "Hello, world")

    def test_reasoning_content_captured(self):
        # A thinking loop is JUST as bad as a content loop - detector must
        # see reasoning_content too.
        x = SSETextExtractor()
        chunk = (
            b'data: {"choices":[{"delta":{"reasoning_content":"let me think..."}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
        )
        text = x.extract(chunk)
        self.assertIn("let me think...", text)
        self.assertIn("Answer", text)

    def test_llamacpp_native_content(self):
        # llama.cpp's /completion endpoint uses a top-level content field
        # rather than the OpenAI choices structure.
        x = SSETextExtractor()
        chunk = b'data: {"content":"partial"}\n\n'
        self.assertEqual(x.extract(chunk), "partial")

    def test_legacy_completions_text(self):
        # /v1/completions uses choices[0].text
        x = SSETextExtractor()
        chunk = b'data: {"choices":[{"text":"legacy!"}]}\n\n'
        self.assertEqual(x.extract(chunk), "legacy!")

    def test_split_mid_chunk_boundary(self):
        x = SSETextExtractor()
        line = b'data: {"choices":[{"delta":{"content":"split across chunks"}}]}\n\n'
        # Feed in weird pieces
        parts = [line[:5], line[5:20], line[20:]]
        collected = ""
        for p in parts:
            collected += x.extract(p)
        self.assertEqual(collected, "split across chunks")

    def test_done_sentinel_and_keepalive_ignored(self):
        x = SSETextExtractor()
        self.assertEqual(x.extract(b"data: [DONE]\n\n"), "")
        self.assertEqual(x.extract(b": keepalive\n\n"), "")
        self.assertEqual(x.extract(b"\n\n"), "")

    def test_malformed_json_line_dropped_silently(self):
        # A broken data line must not throw - streaming continues.
        x = SSETextExtractor()
        self.assertEqual(x.extract(b"data: {broken json\n\n"), "")


class SyntheticTerminatorTests(unittest.TestCase):
    def test_openai_sse_terminator_is_valid_sse(self):
        payload = make_openai_sse_terminator("gpt-loop-test")
        self.assertTrue(payload.startswith(b"data: "))
        self.assertTrue(payload.endswith(b"data: [DONE]\n\n"))
        # And the JSON parses
        import json as _json
        first_line = payload.decode("utf-8").split("\n\n")[0]
        self.assertTrue(first_line.startswith("data: "))
        obj = _json.loads(first_line[6:])
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")
        self.assertIn(LOOP_TERMINATION_MESSAGE, obj["choices"][0]["delta"]["content"])
        self.assertEqual(obj["model"], "gpt-loop-test")

    def test_ollama_chat_terminator_shape(self):
        line = make_ollama_terminator("my-model", mode="chat")
        self.assertTrue(line.endswith("\n"))
        import json as _json
        obj = _json.loads(line)
        self.assertTrue(obj["done"])
        self.assertEqual(obj["done_reason"], "loop_detected")
        self.assertIn("message", obj)
        self.assertIn(LOOP_TERMINATION_MESSAGE, obj["message"]["content"])
        self.assertEqual(obj["model"], "my-model")

    def test_ollama_generate_terminator_shape(self):
        line = make_ollama_terminator("my-model", mode="generate")
        import json as _json
        obj = _json.loads(line)
        self.assertIn("response", obj)
        self.assertIn(LOOP_TERMINATION_MESSAGE, obj["response"])
        self.assertNotIn("message", obj)


class LoopDetectKeysTests(unittest.TestCase):
    def test_all_expected_keys_exported(self):
        # Sanity: the tuple used by other modules (api/instances.py, presets)
        # matches what parse_loop_detect_config actually produces.
        cfg, _ = parse_loop_detect_config({})
        for key in LOOP_DETECT_KEYS:
            self.assertIn(key, cfg,
                          f"{key} in LOOP_DETECT_KEYS but not in parse output")
        # And the reverse: no accidental extra keys
        self.assertEqual(set(cfg.keys()), set(LOOP_DETECT_KEYS))


if __name__ == "__main__":
    unittest.main()
