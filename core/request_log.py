# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Request logging: mode gating, conversation id, and the record_request entry point.

Call sites use the return value as a handle:

    handle = record_request(body, ...)
    try:
        ... do the request ...
        if handle:
            handle.set_response(text=text, usage=usage, status_code=200)
    finally:
        if handle:
            handle.finalize(streamed=True)

Returning None when recording is off keeps hook sites branchless.

`finalize_async(handle, streamed=...)` runs the storage write on a daemon
thread so it never blocks the HTTP response. The storage backend serializes
log writes through a global lock and on slow filesystems (WSL2 9p, NFS,
etc.) that lock contention can stretch the LAST write of a burst long
enough that Flask doesn't deliver the response within the upstream's
read-timeout window - then the client sees a 504 even though the work
completed cleanly. Anything in a request handler's `finally` runs BEFORE
Flask sends the response to the wire, so finalize had to come off that path.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any

from core.timeutil import now_iso
from storage import get_storage

logger = logging.getLogger("llamaman")


VALID_MODES = ("off", "per_request", "per_conversation")

_CACHE_TTL_SECONDS = 2.0

_mode_cache: str | None = None
_mode_cache_expires_at: float = 0.0
_mode_lock = threading.Lock()


def get_mode() -> str:
    """Return the current recording mode.

    Cached for a short interval so every inference request doesn't hit storage.
    Setting changes propagate within _CACHE_TTL_SECONDS; callers that need an
    immediate refresh should invoke invalidate_cache().
    """
    global _mode_cache, _mode_cache_expires_at
    now = time.monotonic()
    if _mode_cache is not None and now < _mode_cache_expires_at:
        return _mode_cache
    with _mode_lock:
        now = time.monotonic()
        if _mode_cache is not None and now < _mode_cache_expires_at:
            return _mode_cache
        try:
            mode = get_storage().get_settings().get("recording_mode", "off")
        except Exception:
            mode = "off"
        if mode not in VALID_MODES:
            mode = "off"
        _mode_cache = mode
        _mode_cache_expires_at = now + _CACHE_TTL_SECONDS
        return mode


def enabled() -> bool:
    return get_mode() != "off"


def invalidate_cache() -> None:
    global _mode_cache, _mode_cache_expires_at
    with _mode_lock:
        _mode_cache = None
        _mode_cache_expires_at = 0.0


def conversation_id(body: dict | None) -> str:
    """Content-hash id grouping all turns of a conversation.

    Hashes system prompt + first user message. A conversation's Nth request
    replays turns 1..N-1, but those two fields don't change, so every turn
    hashes to the same id. Works for both OpenAI-style messages[] and Ollama's
    /api/generate (prompt + optional system).
    """
    system = ""
    first_user = ""
    msgs = body.get("messages") if isinstance(body, dict) else None
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role == "system" and not system:
                system = content
            elif role == "user" and not first_user:
                first_user = content
                break
    if not first_user and isinstance(body, dict):
        if not system:
            system = body.get("system") or ""
        first_user = body.get("prompt") or ""
    digest = hashlib.sha256(f"{system}\x00{first_user}".encode("utf-8")).hexdigest()
    return digest[:32]


class RecordingHandle:
    """Collects envelope fields during one request's lifecycle and flushes once
    on finalize.

    Single-threaded per handle: callers drive set_response / set_error /
    finalize from the same thread, or serialize themselves. finalize is
    idempotent (repeat calls are ignored) so it's safe to put in a finally
    block that may run twice on some control paths.
    """

    __slots__ = ("_record", "_mode", "_finalized", "_t_start", "_t_first_token")

    def __init__(self, record: dict, mode: str, t_start: float):
        self._record = record
        self._mode = mode
        self._finalized = False
        self._t_start = t_start
        self._t_first_token = None

    def set_response(self, *, text: str | None = None,
                     usage: dict | None = None,
                     status_code: int | None = None) -> None:
        if text is not None:
            self._record["response_body"] = text
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int):
                self._record["prompt_tokens"] = pt
            if isinstance(ct, int):
                self._record["completion_tokens"] = ct
        if status_code is not None:
            self._record["status_code"] = int(status_code)

    def set_metrics(self, *, tokens_per_sec: float | None = None,
                    ttft_ms: float | None = None) -> None:
        """Record generation-throughput metrics computed by the caller.

        Use this on paths that already measure tokens/s and time-to-first-token
        from real generation timing (more precise than re-deriving from total
        duration, which folds in prompt evaluation). Streaming relays that don't
        compute these themselves can instead call mark_first_token() and let
        finalize() derive them.
        """
        if tokens_per_sec is not None:
            self._record["tokens_per_sec"] = round(float(tokens_per_sec), 2)
        if ttft_ms is not None:
            self._record["ttft_ms"] = round(float(ttft_ms), 1)

    def mark_first_token(self) -> None:
        """Stamp the moment the first response token/byte arrived.

        finalize() uses this to derive ttft_ms and a generation-only tokens/s
        when the caller didn't set them explicitly via set_metrics(). Only the
        first call takes effect.
        """
        if self._t_first_token is None:
            self._t_first_token = time.monotonic()

    def set_error(self, status_code: int, error: str) -> None:
        self._record["status_code"] = int(status_code)
        self._record["response_body"] = error

    def finalize(self, duration_ms: int | None = None,
                 streamed: bool = False) -> None:
        if self._finalized:
            return
        self._finalized = True
        now = time.monotonic()
        if duration_ms is None:
            duration_ms = int((now - self._t_start) * 1000)
        self._record["duration_ms"] = int(duration_ms)
        self._record["streamed"] = bool(streamed)
        # Derive accurate generation metrics from the first-token mark unless the
        # caller already supplied them. ttft = first token - request start;
        # tokens/s = completion tokens over the generation window only.
        if self._t_first_token is not None:
            if "ttft_ms" not in self._record:
                self._record["ttft_ms"] = round(
                    (self._t_first_token - self._t_start) * 1000, 1)
            if "tokens_per_sec" not in self._record:
                ct = self._record.get("completion_tokens")
                gen_s = now - self._t_first_token
                if isinstance(ct, int) and ct > 0 and gen_s > 0:
                    self._record["tokens_per_sec"] = round(ct / gen_s, 2)
        try:
            get_storage().append_request_log(self._record, self._mode)
        except Exception as e:
            logger.warning("request_log flush failed: %s", e)


def finalize_async(handle: "RecordingHandle | None", streamed: bool = False) -> None:
    """Run handle.finalize() on a daemon thread so the HTTP response isn't
    blocked by storage I/O. No-op when handle is None.

    duration_ms is snapshotted on THIS thread before the handoff so the value
    reflects when the work actually ended, not when the background flush ran.
    """
    if handle is None:
        return
    duration_ms = int((time.monotonic() - handle._t_start) * 1000)
    t = threading.Thread(
        target=handle.finalize,
        kwargs={"duration_ms": duration_ms, "streamed": streamed},
        daemon=True,
    )
    t.start()


def _safe_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return json.dumps(obj, separators=(",", ":"),
                              ensure_ascii=False, default=str)
        except Exception:
            return ""


class SSEAccumulator:
    """Parse OpenAI-style SSE bytes incrementally, collecting assistant
    content and final usage counts for recording.

    Bytes may arrive split across arbitrary boundaries; we buffer until
    newline-terminated lines are available. Only `data:` lines are parsed;
    keep-alives and comments are ignored.
    """

    __slots__ = ("_buf", "_content", "_usage")

    def __init__(self):
        self._buf = b""
        self._content: list[str] = []
        self._usage: dict | None = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buf += chunk
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
            usage = obj.get("usage")
            if isinstance(usage, dict):
                self._usage = usage
            choices = obj.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                c0 = choices[0]
                delta = c0.get("delta")
                if isinstance(delta, dict):                 # chat: choices[].delta.content
                    c = delta.get("content")
                    if isinstance(c, str) and c:
                        self._content.append(c)
                text = c0.get("text")                       # legacy completions: choices[].text
                if isinstance(text, str) and text:
                    self._content.append(text)
            else:                                           # llama.cpp native: top-level content
                c = obj.get("content")
                if isinstance(c, str) and c:
                    self._content.append(c)
                if obj.get("stop") and self._usage is None:
                    tp, te = obj.get("tokens_predicted"), obj.get("tokens_evaluated")
                    if isinstance(tp, int) or isinstance(te, int):
                        self._usage = {
                            "completion_tokens": tp or 0,
                            "prompt_tokens": te or 0,
                            "total_tokens": (tp or 0) + (te or 0),
                        }

    def finish(self) -> tuple[str, dict | None]:
        return "".join(self._content), self._usage


def record_request(
    body: dict | None,
    raw_body: bytes | None = None,
    *,
    endpoint: str,
    path: str,
    inst_id: str | None,
    model: str,
) -> "RecordingHandle | None":
    """Create a handle to record this turn, or None if recording is off.

    `body` is the parsed request (preferred). `raw_body` is a fallback for
    the transparent proxy path; if `body` is None, it's best-effort decoded
    as UTF-8 text. Binary bodies are represented as a byte-count placeholder.
    """
    try:
        mode = get_mode()
        if mode == "off":
            return None

        conv_id = conversation_id(body) if body is not None else conversation_id({})

        if body is not None:
            req_str = _safe_dumps(body)
        elif raw_body:
            try:
                req_str = raw_body.decode("utf-8")
            except UnicodeDecodeError:
                req_str = f"<binary {len(raw_body)}B>"
        else:
            req_str = ""

        record = {
            "conversation_id": conv_id,
            "inst_id": inst_id,
            "model": model or "",
            "endpoint": endpoint,
            "path": path,
            "created_at": now_iso(),
            "request_body": req_str,
        }
        return RecordingHandle(record, mode, time.monotonic())
    except Exception as e:
        logger.warning("record_request failed to create handle: %s", e)
        return None
