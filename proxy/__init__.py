# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import io
import json
import threading
import time

import requests
from werkzeug.serving import make_server
from werkzeug.wrappers import Request as WerkzeugRequest

from config import REQUEST_TIMEOUT, logger
from core.helpers import model_name_from_path
from core.proxy_sampling import PROXY_SAMPLING_PATHS, apply_proxy_sampling_overrides
from core.request_log import record_request, SSEAccumulator
from core.state import instances, instances_lock


# ---------------------------------------------------------------------------
# Per-instance request gate (concurrency limiter)
# ---------------------------------------------------------------------------

_instance_gates: dict[str, "RequestGate"] = {}
_shared_queue_gates: dict[str, "RequestGate"] = {}  # model_path -> gate
_instance_gate_configs: dict[str, dict] = {}
_gates_lock = threading.Lock()


class RequestGate:
    """Concurrency limiter with a bounded waiting queue.

    - max_concurrent: how many requests may be in-flight at once.
    - max_queue_depth: how many additional requests may wait.
    Requests beyond the queue depth are rejected immediately (429).
    """

    def __init__(self, max_concurrent: int, max_queue_depth: int = 200):
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self._waiting = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self.active = 0
        self.queued = 0

    def acquire(self, timeout: float = 300) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._closed:
                return False
            if self._waiting >= self.max_queue_depth:
                return False
            self._waiting += 1
            try:
                while not self._closed and self.active >= self.max_concurrent:
                    self.queued = self._waiting
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)

                if self._closed:
                    return False

                self.active += 1
                return True
            finally:
                self._waiting -= 1
                self.queued = self._waiting

    def release(self):
        with self._condition:
            self.active = max(0, self.active - 1)
            self._condition.notify()

    def cancel(self):
        """Wake queued waiters and prevent new acquires on this gate."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def get_gate(inst_id: str) -> "RequestGate | None":
    with _gates_lock:
        return _instance_gates.get(inst_id)


def create_gate(inst_id: str, max_concurrent: int, max_queue_depth: int,
                model_path: str | None = None, share_queue: bool = False):
    if max_concurrent <= 0:
        return
    with _gates_lock:
        if share_queue and model_path:
            if model_path in _shared_queue_gates:
                gate = _shared_queue_gates[model_path]
            else:
                gate = RequestGate(max_concurrent, max_queue_depth)
                _shared_queue_gates[model_path] = gate
            _instance_gates[inst_id] = gate
        else:
            gate = RequestGate(max_concurrent, max_queue_depth)
            _instance_gates[inst_id] = gate
        _instance_gate_configs[inst_id] = {
            "max_concurrent": max_concurrent,
            "max_queue_depth": max_queue_depth,
            "model_path": model_path,
            "share_queue": share_queue,
        }


def remove_gate(inst_id: str):
    with _gates_lock:
        gate = _instance_gates.pop(inst_id, None)
        _instance_gate_configs.pop(inst_id, None)
        if gate is None:
            return
        still_used = any(other_gate is gate for other_gate in _instance_gates.values())
        if not still_used:
            stale_models = [
                model_path for model_path, shared_gate in _shared_queue_gates.items()
                if shared_gate is gate
            ]
            for model_path in stale_models:
                _shared_queue_gates.pop(model_path, None)
            gate.cancel()


def refresh_gate(inst_id: str):
    """Replace a gate with a fresh object after a stop/crash so stale queued
    state does not leak into the next launch on that instance/model."""
    with _gates_lock:
        gate = _instance_gates.get(inst_id)
        config = _instance_gate_configs.get(inst_id)
        if gate is None or not config:
            return

        related_ids = [iid for iid, other_gate in _instance_gates.items() if other_gate is gate]
        active_peer_exists = False
        with instances_lock:
            for iid in related_ids:
                if iid == inst_id:
                    continue
                peer = instances.get(iid)
                if peer and peer.get("status") not in ("stopped", "sleeping"):
                    active_peer_exists = True
                    break
        if active_peer_exists:
            return

        fresh_gate = RequestGate(
            config["max_concurrent"],
            config["max_queue_depth"],
        )
        for iid in related_ids:
            _instance_gates[iid] = fresh_gate

        if config.get("share_queue") and config.get("model_path"):
            _shared_queue_gates[config["model_path"]] = fresh_gate

        gate.cancel()


# ---------------------------------------------------------------------------
# Idle proxy servers: { inst_id: { server, thread, proxy_port, internal_port } }
# ---------------------------------------------------------------------------

idle_proxies: dict[str, dict] = {}
idle_proxies_lock = threading.Lock()

_GATED_PATHS = frozenset({
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
    "/completion", "/chat/completions",
})


def _is_inference_request(environ: dict) -> bool:
    method = environ.get("REQUEST_METHOD", "GET").upper()
    if method != "POST":
        return False
    path = environ.get("PATH_INFO", "")
    return path in _GATED_PATHS


def _apply_proxy_sampling_request_overrides(req_data: bytes, path: str, config: dict | None) -> bytes:
    if path not in PROXY_SAMPLING_PATHS:
        return req_data

    try:
        body = json.loads(req_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return req_data

    updated_body = apply_proxy_sampling_overrides(body, config)
    if updated_body == body:
        return req_data
    return json.dumps(updated_body).encode("utf-8")


def _check_proxy_auth(environ, start_response):
    """Enforce bearer token auth on proxy ports when require_auth is enabled.

    Returns True if the request should be blocked (response already sent),
    False if the request is allowed to proceed.
    """
    from api.auth import is_require_auth_enabled, verify_bearer_token

    if not is_require_auth_enabled():
        return False

    error = verify_bearer_token(environ.get("HTTP_AUTHORIZATION", ""), strict=True)
    if not error:
        return False

    start_response("401 Unauthorized",
                   [("Content-Type", "application/json")])
    return True


def _peek_inference_body(environ: dict) -> dict | None:
    """Parse the request body for inference endpoints, rewinding wsgi.input
    so the downstream proxy read still sees the bytes. Returns None for
    non-inference or unparseable requests.
    """
    if environ.get("REQUEST_METHOD", "GET").upper() != "POST":
        return None
    path = environ.get("PATH_INFO", "")
    if path not in _GATED_PATHS:
        return None
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(content_length)
        environ["wsgi.input"] = io.BytesIO(body)
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _extract_model_from_request(environ: dict) -> str | None:
    data = _peek_inference_body(environ)
    if not data:
        return None
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, str):
        return model.strip() or None
    return None


def _model_matches(inst_model_path: str, requested_model: str) -> bool:
    """Check if a requested model name matches an instance's model path."""
    inst_model = model_name_from_path(inst_model_path)
    req_model = requested_model.split(":")[0].lower()
    return req_model == inst_model or req_model in inst_model


def _find_sleeping_instance_for_port(model_name: str, proxy_port: int) -> str | None:
    """Find a sleeping instance on this port whose model matches the request."""
    with instances_lock:
        for iid, inst in instances.items():
            if inst["status"] != "sleeping":
                continue
            if inst["port"] != proxy_port:
                continue
            if _model_matches(inst["model_path"], model_name):
                return iid
    return None


def make_proxy_app(inst_id: str, internal_port: int, proxy_port: int):
    """Create a WSGI app that proxies requests to the llama-server,
    waking it from sleep if necessary."""
    def proxy_app(environ, start_response):
        # Lazy import to break circular dependency
        from api.instances import relaunch_inactive_instance

        # Auth check - blocks the request if token is missing/invalid
        if _check_proxy_auth(environ, start_response):
            return [json.dumps({"error": "API key required"}).encode()]

        # Parse body once for inference endpoints; used for both model
        # extraction and (optionally) the recording pipeline.
        parsed_body = _peek_inference_body(environ)
        if isinstance(parsed_body, dict):
            _m = parsed_body.get("model")
            requested_model = _m.strip() if isinstance(_m, str) and _m.strip() else None
        else:
            requested_model = None

        with instances_lock:
            inst = instances.get(inst_id)
            if inst:
                inst["_last_request_at"] = time.time()
                stats = inst.setdefault("stats", {})
                stats["total_requests"] = stats.get("total_requests", 0) + 1

        with instances_lock:
            inst = instances.get(inst_id)
            status = inst["status"] if inst else None

        wake_id = inst_id

        if status in ("sleeping", "stopped"):
            # If the request specifies a model, verify it matches before waking
            if requested_model and inst:
                if not _model_matches(inst["model_path"], requested_model):
                    start_response("404 Not Found",
                                   [("Content-Type", "application/json")])
                    return [json.dumps({"error": f"model '{requested_model}' is not loaded on this port"}).encode()]
            if not relaunch_inactive_instance(wake_id):
                start_response("503 Service Unavailable",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "failed to wake model"}).encode()]
        elif status is None:
            # Instance record gone - try to find a sleeping instance by model + port
            if requested_model:
                found_id = _find_sleeping_instance_for_port(requested_model, proxy_port)
                if found_id:
                    wake_id = found_id
                    if not relaunch_inactive_instance(wake_id):
                        start_response("503 Service Unavailable",
                                       [("Content-Type", "application/json")])
                        return [json.dumps({"error": "failed to wake model"}).encode()]
                else:
                    start_response("503 Service Unavailable",
                                   [("Content-Type", "application/json")])
                    return [json.dumps({"error": "no matching sleeping model found on this port"}).encode()]
            else:
                start_response("503 Service Unavailable",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "instance no longer exists"}).encode()]
        elif status == "starting":
            from api.instances import wait_for_healthy
            from config import MODEL_LOAD_TIMEOUT
            with instances_lock:
                _inst = instances.get(inst_id)
                _host = _inst.get("_server_host", "localhost") if _inst else "localhost"
                _port = (_inst.get("_server_port") or _inst.get("_internal_port") or internal_port) if _inst else internal_port
            if not wait_for_healthy(_host, _port, timeout=MODEL_LOAD_TIMEOUT):
                start_response("503 Service Unavailable",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "model is loading but did not become healthy in time"}).encode()]

        # Resolve the effective instance and port (may differ if wake_id changed)
        effective_id = wake_id
        with instances_lock:
            effective_inst = instances.get(effective_id)
            if effective_inst:
                effective_host = effective_inst.get("_server_host", "localhost")
                effective_port = (effective_inst.get("_server_port")
                                  or effective_inst.get("_internal_port")
                                  or internal_port)
            else:
                effective_host = "localhost"
                effective_port = internal_port

        # Validate model name for all statuses (sleeping already validated above,
        # but healthy/starting instances need the same check for consistency)
        if requested_model and effective_inst:
            if not _model_matches(effective_inst["model_path"], requested_model):
                start_response("404 Not Found",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": f"model '{requested_model}' is not loaded on this port"}).encode()]

        gate = get_gate(effective_id) if _is_inference_request(environ) else None
        if gate:
            if not gate.acquire(timeout=REQUEST_TIMEOUT):
                start_response("429 Too Many Requests",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "request queue full"}).encode()]

        handle = record_request(
            parsed_body if isinstance(parsed_body, dict) else None,
            endpoint="proxy",
            path=environ.get("PATH_INFO", ""),
            inst_id=effective_id,
            model=requested_model or "",
        ) if _is_inference_request(environ) else None

        try:
            req = WerkzeugRequest(environ)
            target = f"http://{effective_host}:{effective_port}{req.path}"
            if req.query_string:
                target += f"?{req.query_string.decode()}"

            headers = {
                k: v for k, v in req.headers
                if k.lower() not in ("host", "content-length")
            }

            req_data = req.get_data()
            req_data = _apply_proxy_sampling_request_overrides(
                req_data,
                req.path,
                effective_inst.get("config", {}) if effective_inst else {},
            )
            last_err = None
            resp = None
            for _attempt in range(3):
                try:
                    resp = requests.request(
                        method=req.method,
                        url=target,
                        headers=headers,
                        data=req_data,
                        stream=True,
                        timeout=REQUEST_TIMEOUT,
                    )
                    break
                except (requests.ConnectionError, requests.Timeout,
                        ConnectionRefusedError) as e:
                    last_err = e
                    time.sleep(2)
                except Exception as e:
                    last_err = e
                    break
            if resp is None:
                if gate:
                    gate.release()
                    gate = None  # prevent double-release in except block
                if handle:
                    handle.set_error(502, str(last_err))
                    handle.finalize()
                start_response("502 Bad Gateway",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": str(last_err)}).encode()]

            resp_headers = [
                (k, v) for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "connection")
            ]
            start_response(f"{resp.status_code} {resp.reason}", resp_headers)

            content_type = (resp.headers.get("Content-Type") or "").lower()
            is_sse = "text/event-stream" in content_type

            def _relay_and_close():
                acc = SSEAccumulator() if (handle and is_sse) else None
                # Non-SSE capture: buffer up to a cap so we don't bloat memory
                # on large/binary responses. 256KB is enough for a typical
                # non-streaming chat/embeddings reply.
                buf: bytearray | None = bytearray() if (handle and not is_sse) else None
                BUF_CAP = 256 * 1024
                try:
                    for chunk in resp.iter_content(chunk_size=None):
                        if acc is not None:
                            acc.feed(chunk)
                            if handle and chunk:
                                handle.mark_first_token()
                        elif buf is not None and len(buf) < BUF_CAP:
                            buf.extend(chunk[: BUF_CAP - len(buf)])
                        yield chunk
                finally:
                    resp.close()
                    if gate:
                        gate.release()
                    if handle:
                        text = ""
                        usage = None
                        if acc is not None:
                            text, usage = acc.finish()
                        elif buf is not None:
                            try:
                                text = bytes(buf).decode("utf-8")
                            except UnicodeDecodeError:
                                text = f"<binary {len(buf)}B>"
                            # Best-effort usage extraction from JSON responses
                            try:
                                parsed = json.loads(text) if text else None
                                if isinstance(parsed, dict):
                                    u = parsed.get("usage")
                                    if isinstance(u, dict):
                                        usage = u
                            except (json.JSONDecodeError, ValueError):
                                pass
                        handle.set_response(text=text, usage=usage,
                                            status_code=resp.status_code)
                        handle.finalize(streamed=is_sse)
            return _relay_and_close()
        except Exception as e:
            if gate:
                gate.release()
            if handle:
                handle.set_error(500, str(e))
                handle.finalize()
            raise

    return proxy_app


def start_idle_proxy(inst_id: str, proxy_port: int, internal_port: int):
    proxy_app = make_proxy_app(inst_id, internal_port, proxy_port)
    server = make_server("0.0.0.0", proxy_port, proxy_app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with idle_proxies_lock:
        idle_proxies[inst_id] = {
            "server": server,
            "thread": thread,
            "proxy_port": proxy_port,
            "internal_port": internal_port,
        }
    logger.info("Idle proxy: port %d -> internal %d (instance %s)",
                proxy_port, internal_port, inst_id)


def stop_idle_proxy(inst_id: str):
    with idle_proxies_lock:
        proxy = idle_proxies.pop(inst_id, None)
    if proxy:
        try:
            proxy["server"].shutdown()
        except Exception:
            pass
        logger.info("Idle proxy stopped for instance %s", inst_id)


def cleanup_orphan_idle_proxies(valid_instance_ids: set[str]) -> int:
    """Stop proxy listeners whose instance records no longer exist."""
    with idle_proxies_lock:
        stale_ids = [inst_id for inst_id in idle_proxies if inst_id not in valid_instance_ids]

    for inst_id in stale_ids:
        stop_idle_proxy(inst_id)

    return len(stale_ids)
