# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Auto model output loop detection: per-turn rolling text buffers, a pure
detection function, and a shared registry the streaming paths (per-instance
sidecar proxy + Ollama/OpenAI compat routes in api/llamaman.py) hook into.

Design summary — see the plan discussion in the PR for full context:

- Detection is a POST-HOC hard kill. llama.cpp's DRY sampler (in
  core/dry_sampling.py) is the SOFT, sampling-time first line of defense;
  when it fails to break a stable loop attractor, this module is the safety
  net. Off by default per model preset.

- Two triggers, one scan function. Buffers are scanned inline every N tokens
  (default 64) as they're fed - low latency, high responsiveness. A worker
  thread ticked from core/monitoring.py catches slow streams that haven't
  hit the inline threshold within scan_interval_s (default 10). worker_tick
  is called every ~5s from the existing background poller so we don't spawn
  a new thread just for this.

- Per-request buffering. With max_concurrent > 1 an instance has N turns
  in flight, so the registry is dict[stream_id -> TurnBuffer], not
  dict[inst_id -> ...]. stream_ids are minted at attach() and never reused.

- Detection algorithm (deliberately simple for v1): take the last N chars
  of the buffer as the "target" chunk, count exact occurrences in the last
  window. If >= min_repetitions, flag. Catches every runaway loop I've
  actually seen in the wild without false-positiving on code with for-loops
  or markdown tables at the default 200-char / 3-rep thresholds.

- The kill mechanism is signal-based: scan sets buf.cancel_flag; the
  streaming generator checks the flag after every chunk (see Tranche 3/4)
  and, on hit, injects a synthetic terminator into the client stream and
  closes the upstream connection. No cross-thread socket closes - those are
  a race nightmare.

- ALL entry points are defensive. Any exception in feed()/scan()/attach()
  is caught and logged rather than propagated - a detector bug can never
  break a real user's stream. The streaming callers rewrap on top of this
  as belt-and-braces.

Configuration keys the streaming callers pass in via `config` (mirroring
PROXY_SAMPLING_OVERRIDE_KEYS in core/proxy_sampling.py):

    loop_detect_enabled           bool, default False
    loop_detect_min_chunk_chars   int,  default 200
    loop_detect_min_repetitions   int,  default 3
    loop_detect_max_buffer_chars  int,  default 8192   (rolling cap)
    loop_detect_scan_interval_s   int,  default 10     (worker fallback)
    loop_detect_scan_every_n_tokens int, default 64    (inline trigger)
"""

import json
import threading
import time
import uuid

from config import logger


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

DEFAULT_LOOP_DETECT_ENABLED = False
DEFAULT_LOOP_DETECT_MIN_CHUNK_CHARS = 200
DEFAULT_LOOP_DETECT_MIN_REPETITIONS = 3
DEFAULT_LOOP_DETECT_MAX_BUFFER_CHARS = 8192
DEFAULT_LOOP_DETECT_SCAN_INTERVAL_S = 10
DEFAULT_LOOP_DETECT_SCAN_EVERY_N_TOKENS = 64

# Sanity caps. min_chunk below ~60 chars false-positives on real content
# (numbered lists, poetry choruses, markdown tables with identical headers).
# max_buffer above 64 KB per stream, times max_concurrent per instance, times
# instance count, is a memory-pressure risk. Caps are enforced at the parse
# boundary so a corrupt preset can never blow past them.
MIN_LOOP_DETECT_CHUNK_CHARS = 60
MAX_LOOP_DETECT_CHUNK_CHARS = 4096
MIN_LOOP_DETECT_REPETITIONS = 2
MAX_LOOP_DETECT_REPETITIONS = 20
MIN_LOOP_DETECT_MAX_BUFFER_CHARS = 512
MAX_LOOP_DETECT_MAX_BUFFER_CHARS = 65536
MIN_LOOP_DETECT_SCAN_INTERVAL_S = 1
MAX_LOOP_DETECT_SCAN_INTERVAL_S = 600
MIN_LOOP_DETECT_SCAN_EVERY_N_TOKENS = 8
MAX_LOOP_DETECT_SCAN_EVERY_N_TOKENS = 4096

LOOP_DETECT_KEYS = (
    "loop_detect_enabled",
    "loop_detect_min_chunk_chars",
    "loop_detect_min_repetitions",
    "loop_detect_max_buffer_chars",
    "loop_detect_scan_interval_s",
    "loop_detect_scan_every_n_tokens",
)


def parse_loop_detect_config(body: dict) -> tuple[dict, str | None]:
    """Normalize loop-detection fields out of a launch/preset body. Same
    boundary-validation contract as parse_proxy_sampling_config /
    parse_dry_config: reject invalid values so the user gets a clean 400
    instead of a runtime surprise."""
    enabled = bool(body.get("loop_detect_enabled", DEFAULT_LOOP_DETECT_ENABLED))

    def _bounded_int(key, default, lo, hi):
        raw = body.get(key, default)
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return None, f"{key} must be an integer"
        if v < lo or v > hi:
            return None, f"{key} must be between {lo} and {hi}"
        return v, None

    min_chunk, err = _bounded_int(
        "loop_detect_min_chunk_chars",
        DEFAULT_LOOP_DETECT_MIN_CHUNK_CHARS,
        MIN_LOOP_DETECT_CHUNK_CHARS,
        MAX_LOOP_DETECT_CHUNK_CHARS,
    )
    if err:
        return {}, err
    min_reps, err = _bounded_int(
        "loop_detect_min_repetitions",
        DEFAULT_LOOP_DETECT_MIN_REPETITIONS,
        MIN_LOOP_DETECT_REPETITIONS,
        MAX_LOOP_DETECT_REPETITIONS,
    )
    if err:
        return {}, err
    max_buf, err = _bounded_int(
        "loop_detect_max_buffer_chars",
        DEFAULT_LOOP_DETECT_MAX_BUFFER_CHARS,
        MIN_LOOP_DETECT_MAX_BUFFER_CHARS,
        MAX_LOOP_DETECT_MAX_BUFFER_CHARS,
    )
    if err:
        return {}, err
    # A scan window that can't hold min_chunk * min_repetitions of text will
    # never detect anything. Reject at the boundary so the user isn't
    # silently running with detection off.
    if max_buf < min_chunk * min_reps:
        return {}, (
            "loop_detect_max_buffer_chars must be at least "
            "loop_detect_min_chunk_chars * loop_detect_min_repetitions "
            f"({min_chunk * min_reps} for the current values)"
        )
    scan_interval, err = _bounded_int(
        "loop_detect_scan_interval_s",
        DEFAULT_LOOP_DETECT_SCAN_INTERVAL_S,
        MIN_LOOP_DETECT_SCAN_INTERVAL_S,
        MAX_LOOP_DETECT_SCAN_INTERVAL_S,
    )
    if err:
        return {}, err
    scan_every_n, err = _bounded_int(
        "loop_detect_scan_every_n_tokens",
        DEFAULT_LOOP_DETECT_SCAN_EVERY_N_TOKENS,
        MIN_LOOP_DETECT_SCAN_EVERY_N_TOKENS,
        MAX_LOOP_DETECT_SCAN_EVERY_N_TOKENS,
    )
    if err:
        return {}, err

    return {
        "loop_detect_enabled": enabled,
        "loop_detect_min_chunk_chars": min_chunk,
        "loop_detect_min_repetitions": min_reps,
        "loop_detect_max_buffer_chars": max_buf,
        "loop_detect_scan_interval_s": scan_interval,
        "loop_detect_scan_every_n_tokens": scan_every_n,
    }, None


def loop_detect_enabled(config: dict | None) -> bool:
    return bool((config or {}).get("loop_detect_enabled", False))


# ---------------------------------------------------------------------------
# Pure detection
# ---------------------------------------------------------------------------

def scan_text(text: str, min_chunk: int, min_repetitions: int) -> bool:
    """Return True if the tail of `text` looks like a stable loop.

    Algorithm: take the last `min_chunk` characters as the candidate chunk;
    count how many times that exact substring appears in `text`. If >=
    min_repetitions, it's a loop. Simple, fast (linear in text length via
    str.count), and empirically catches every real-world loop I've seen
    without false-positiving on code / poetry / tables at the defaults.

    Doesn't catch:
      - Drifting near-repeats ("The answer is: The answer is:" with slight
        punctuation change between reps). Would need suffix-array or fuzzy
        matching; deferred to v2 if operators actually see these.
      - Short-period loops smaller than min_chunk. Deliberate: those get
        caught by the DRY sampler's repeat_last_n or by shortening min_chunk
        below llama.cpp's DRY window on preset.

    Kept as a top-level function (not a TurnBuffer method) so tests can
    exercise the algorithm directly on synthetic strings without the
    registry / lock machinery.
    """
    if not text or min_chunk <= 0 or min_repetitions <= 1:
        return False
    if len(text) < min_chunk * min_repetitions:
        return False
    target = text[-min_chunk:]
    # str.count is C-level and doesn't overlap-match, which is what we want
    # here: a chunk repeated back-to-back N times shows up as exactly N
    # non-overlapping matches.
    return text.count(target) >= min_repetitions


# ---------------------------------------------------------------------------
# TurnBuffer + registry
# ---------------------------------------------------------------------------

class TurnBuffer:
    """One turn's rolling text buffer + detection state.

    Owned by ONE streaming request. Registered in `_active_buffers` under
    its `stream_id` so the worker fallback can find it. Detached in the
    streaming path's finally: block (or by the worker on stale-buffer GC).
    """

    __slots__ = (
        "stream_id", "inst_id",
        "_chars", "_lock",
        "min_chunk", "min_reps", "max_buffer", "scan_every_n",
        "scan_interval_s",
        "tokens_since_last_scan", "last_scan_at", "started_at",
        "cancel_flag",
    )

    def __init__(self, stream_id: str, inst_id: str, config: dict):
        self.stream_id = stream_id
        self.inst_id = inst_id
        # A list-of-strings + rolling join is O(n) in total feed size when
        # str.count runs on the joined view. For our max_buffer sizes (8 KB
        # default, 64 KB cap) the cost per scan is trivial - and we only
        # scan once per scan_every_n tokens.
        self._chars: list[str] = []
        self._lock = threading.Lock()
        self.min_chunk = int(config.get(
            "loop_detect_min_chunk_chars", DEFAULT_LOOP_DETECT_MIN_CHUNK_CHARS))
        self.min_reps = int(config.get(
            "loop_detect_min_repetitions", DEFAULT_LOOP_DETECT_MIN_REPETITIONS))
        self.max_buffer = int(config.get(
            "loop_detect_max_buffer_chars", DEFAULT_LOOP_DETECT_MAX_BUFFER_CHARS))
        self.scan_every_n = int(config.get(
            "loop_detect_scan_every_n_tokens", DEFAULT_LOOP_DETECT_SCAN_EVERY_N_TOKENS))
        self.scan_interval_s = int(config.get(
            "loop_detect_scan_interval_s", DEFAULT_LOOP_DETECT_SCAN_INTERVAL_S))
        self.tokens_since_last_scan = 0
        self.started_at = time.monotonic()
        self.last_scan_at = self.started_at
        self.cancel_flag = threading.Event()

    def _text_snapshot(self) -> str:
        """Return the current buffer as one string. MUST be called with the
        lock held or with an atomic snapshot of _chars in hand."""
        return "".join(self._chars)


_active_buffers: dict[str, TurnBuffer] = {}
_registry_lock = threading.Lock()


def attach(inst_id: str, config: dict | None) -> TurnBuffer | None:
    """Return a fresh TurnBuffer if loop detection is enabled on this config,
    else None. Callers should always call this and branch on truthiness -
    keeps the streaming call sites uniform ("attach, feed if buf, detach in
    finally").

    Any exception is swallowed and logged; a broken attach must never break
    the stream itself. Same defensive contract for feed()/detach().
    """
    try:
        if not loop_detect_enabled(config):
            return None
        stream_id = uuid.uuid4().hex
        buf = TurnBuffer(stream_id, inst_id, config or {})
        with _registry_lock:
            _active_buffers[stream_id] = buf
        return buf
    except Exception as e:
        logger.warning("loop_detect.attach failed for inst=%s: %s", inst_id, e)
        return None


def feed(buf: TurnBuffer | None, text: str) -> bool:
    """Append `text` (a fragment of assistant-visible output) to `buf`'s
    rolling window and, if the inline threshold is hit, scan.

    Returns True if THIS call detected a loop (the caller should then
    inject a terminator and close the upstream). Returns False otherwise -
    including when the flag was already set by a previous call or the
    worker fallback (in which case the caller should still check
    buf.cancel_flag directly at the next chunk boundary).

    No-op when buf is None (loop detection off for this request).
    """
    if buf is None or not text:
        return False
    try:
        with buf._lock:
            buf._chars.append(text)
            # Rolling cap: keep only the last max_buffer chars. Collapse the
            # list to one string so future slicing is O(1) and the char
            # count is monotonic.
            total_len = sum(len(c) for c in buf._chars)
            if total_len > buf.max_buffer:
                joined = "".join(buf._chars)
                buf._chars = [joined[-buf.max_buffer:]]
            buf.tokens_since_last_scan += 1
            # llama.cpp emits many tokens as multi-char strings (bpe merges,
            # word-pieces, etc), so this counter tracks CHUNKS from the
            # upstream generator, not literal LLM tokens. Close enough for
            # scan-rate purposes and much cheaper than tokenizing.
            if buf.tokens_since_last_scan < buf.scan_every_n:
                return False
            buf.tokens_since_last_scan = 0
            buf.last_scan_at = time.monotonic()
            snapshot = "".join(buf._chars)
            min_chunk = buf.min_chunk
            min_reps = buf.min_reps
        # Scan outside the buffer lock so a slow scan can't stall other
        # streams waiting on this same buf (they won't - one buf serves one
        # stream - but this also keeps the lock scope tight for correctness).
        if scan_text(snapshot, min_chunk, min_reps):
            buf.cancel_flag.set()
            logger.info(
                "loop_detect: loop detected on stream=%s inst=%s "
                "(chunk=%d reps=%d buf=%dch elapsed=%.1fs)",
                buf.stream_id, buf.inst_id, min_chunk, min_reps,
                len(snapshot), time.monotonic() - buf.started_at,
            )
            return True
        return False
    except Exception as e:
        # A scan failure must never break the stream. Log and pretend nothing
        # happened - the client keeps seeing normal output.
        logger.warning("loop_detect.feed failed for stream=%s: %s",
                       getattr(buf, "stream_id", "?"), e)
        return False


def detach(buf: TurnBuffer | None) -> None:
    """Remove a buffer from the registry. Safe to call multiple times and
    with None. Called from streaming path's finally: block."""
    if buf is None:
        return
    try:
        with _registry_lock:
            _active_buffers.pop(buf.stream_id, None)
    except Exception as e:
        logger.warning("loop_detect.detach failed for stream=%s: %s",
                       getattr(buf, "stream_id", "?"), e)


def worker_tick(now: float | None = None) -> int:
    """Periodic fallback scan of every active buffer whose last inline scan
    is older than its scan_interval_s. Called from core/monitoring.py's
    background poller every ~5s.

    Returns the number of buffers that had a loop detected THIS tick (for
    metrics / logs). A hit here still just sets buf.cancel_flag; the
    streaming generator will notice on its next chunk boundary and inject
    the terminator.

    Safe to call with an empty registry (cheap no-op)."""
    if now is None:
        now = time.monotonic()
    detected = 0
    # Snapshot the buffer list under the registry lock so we don't hold it
    # across scan_text calls.
    with _registry_lock:
        snapshot = list(_active_buffers.values())
    for buf in snapshot:
        try:
            if buf.cancel_flag.is_set():
                continue  # inline scan already caught it, nothing to do
            if now - buf.last_scan_at < buf.scan_interval_s:
                continue
            with buf._lock:
                buf.last_scan_at = now
                text = "".join(buf._chars)
                min_chunk = buf.min_chunk
                min_reps = buf.min_reps
            if scan_text(text, min_chunk, min_reps):
                buf.cancel_flag.set()
                detected += 1
                logger.info(
                    "loop_detect (worker): loop detected on stream=%s inst=%s "
                    "(chunk=%d reps=%d buf=%dch elapsed=%.1fs)",
                    buf.stream_id, buf.inst_id, min_chunk, min_reps,
                    len(text), now - buf.started_at,
                )
        except Exception as e:
            logger.warning("loop_detect.worker_tick failed for stream=%s: %s",
                           getattr(buf, "stream_id", "?"), e)
    return detected


def active_buffer_count() -> int:
    """Number of currently-registered TurnBuffers. For tests + metrics."""
    with _registry_lock:
        return len(_active_buffers)


def _reset_for_tests() -> None:
    """Wipe the registry. Test-only - production never calls this."""
    with _registry_lock:
        _active_buffers.clear()


# ---------------------------------------------------------------------------
# SSE text extraction
# ---------------------------------------------------------------------------

class SSETextExtractor:
    """Extract assistant-visible text (content + reasoning_content) from
    OpenAI-style SSE bytes as they stream in.

    Parallel to core/request_log.py's SSEAccumulator but with two extras
    that matter here:
      1. reasoning_content is captured (thinking loops matter for detection)
      2. Text is streamed out via extract() calls, not accumulated - callers
         feed extracted deltas straight into TurnBuffer.feed() without
         holding a growing string.

    Kept separate so the request-log pipeline is untouched. Line-buffered;
    tolerates chunk boundaries mid-line.
    """

    __slots__ = ("_buf",)

    def __init__(self):
        self._buf = b""

    def extract(self, chunk: bytes) -> str:
        """Feed one raw SSE chunk. Returns any newly-visible content+reasoning
        text since the last call, joined with no separator. Empty string when
        the chunk contained no completed data: line."""
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.rstrip(b"\r").strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            choices = obj.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                c0 = choices[0]
                delta = c0.get("delta")
                if isinstance(delta, dict):
                    c = delta.get("content")
                    if isinstance(c, str) and c:
                        out_parts.append(c)
                    r = delta.get("reasoning_content")
                    if isinstance(r, str) and r:
                        out_parts.append(r)
                text = c0.get("text")  # legacy /v1/completions
                if isinstance(text, str) and text:
                    out_parts.append(text)
            else:
                # llama.cpp native /completion: top-level content
                c = obj.get("content")
                if isinstance(c, str) and c:
                    out_parts.append(c)
        return "".join(out_parts)


# ---------------------------------------------------------------------------
# Synthetic terminators
#
# When a loop is detected mid-stream, the streaming generator emits one of
# these terminator payloads to the client, then closes the upstream. Kept
# here so both the sidecar proxy and the compat routes render the same
# message and the payload format matches what the client already expected.
# ---------------------------------------------------------------------------

LOOP_TERMINATION_MESSAGE = (
    "\n\n[llamaman: output loop detected, ending turn]"
)


def make_openai_sse_terminator(model_name: str = "") -> bytes:
    """SSE bytes for a client already reading OpenAI-format SSE. Emits one
    final content delta with finish_reason=stop plus the [DONE] sentinel,
    so a well-behaved client renders the notice and closes normally."""
    ts = int(time.time())
    delta_obj = {
        "id": f"chatcmpl-loop-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": LOOP_TERMINATION_MESSAGE},
            "finish_reason": "stop",
        }],
    }
    return (
        f"data: {json.dumps(delta_obj)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def make_ollama_terminator(model_name: str = "", mode: str = "chat") -> str:
    """One Ollama NDJSON line (already terminated with \\n) with
    done=true, done_reason='loop_detected', ready to yield from a
    _stream_llamaman-style generator."""
    from datetime import datetime, timezone
    obj = {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done": True,
        "done_reason": "loop_detected",
    }
    if mode == "chat":
        obj["message"] = {"role": "assistant", "content": LOOP_TERMINATION_MESSAGE}
    else:
        obj["response"] = LOOP_TERMINATION_MESSAGE
    return json.dumps(obj, ensure_ascii=False) + "\n"
