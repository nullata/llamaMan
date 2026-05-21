# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from config import (
    HEALTH_CHECK_TIMEOUT,
    MODELS_DIR,
    LLAMAMAN_MAX_MODELS,
    MODEL_LOAD_TIMEOUT,
    REQUEST_TIMEOUT,
    VERSION,
    logger,
)
from core.helpers import (
    find_available_port,
    is_container_running,
    model_name_from_path,
)
from core.proxy_sampling import apply_proxy_sampling_overrides
from core.request_log import record_request, SSEAccumulator
from api.models import (
    detect_quant,
    discover_models,
    estimate_model_vram,
    format_param_count,
    get_cached_gguf_metadata,
)
from storage import get_storage
from core.state import instances, instances_lock, update_instance_stats
from proxy import get_gate

bp = Blueprint("llamaman", __name__)

# Serialize model launch/evict so one request can't evict a model that
# another request is currently launching or waiting on.
_llamaman_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_model_by_name(name: str) -> dict | None:
    name_lower = name.split(":")[0].lower()
    models = discover_models(MODELS_DIR)
    for m in models:
        if model_name_from_path(m["path"]) == name_lower:
            return m
    for m in models:
        if name_lower in model_name_from_path(m["path"]):
            return m
    return None


def _find_running_instance_for_model(model_path: str) -> dict | None:
    with instances_lock:
        for inst in instances.values():
            if inst["model_path"] == model_path and inst["status"] not in ("stopped",):
                return inst
    return None


def _find_any_instance_for_model(model_path: str) -> dict | None:
    with instances_lock:
        for inst in instances.values():
            if inst["model_path"] == model_path:
                return inst
    return None


def _count_running_instances() -> int:
    """Count running non-embedding instances (manual + managed)."""
    with instances_lock:
        return sum(
            1 for inst in instances.values()
            if inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        )


def _get_llamaman_managed_instances() -> list[dict]:
    """Return llamaman-managed instances sorted by LRU (eviction candidates)."""
    with instances_lock:
        managed = [
            inst for inst in instances.values()
            if inst.get("_llamaman_managed")
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    managed.sort(key=lambda i: i.get("_last_request_at", i["started_at"]))
    return managed


def _get_all_evictable_instances() -> list[dict]:
    """Return ALL non-embedding running instances sorted by LRU."""
    with instances_lock:
        all_insts = [
            inst for inst in instances.values()
            if inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    all_insts.sort(key=lambda i: i.get("_last_request_at", i["started_at"]))
    return all_insts


def _ollama_can_evict_admin_instances() -> bool:
    return bool(get_storage().get_settings().get("allow_ollama_api_override_admin", False))


def _evict_llamaman_instances_if_needed(incoming_embedding_model: bool = False) -> bool:
    """Evict oldest llamaman-managed instances to stay within limits.

    Returns True if there is room for a new instance after eviction (or if no
    limit is set). Returns False if the cap is still exceeded because admin-UI
    instances are blocking and the override setting is disabled.

    The limit is checked against ALL running instances (manual + managed),
    but only llamaman-managed instances are evicted by default.  Manually
    launched instances are never touched unless allow_ollama_api_override_admin
    is enabled.
    """
    from api.instances import stop_instance_by_id

    if LLAMAMAN_MAX_MODELS <= 0:
        return True  # 0 = no limit, never evict
    if incoming_embedding_model:
        return True  # embedding launches never count toward the chat-model cap

    total = _count_running_instances()
    if total < LLAMAMAN_MAX_MODELS:
        return True  # still under the limit

    # First pass: evict only Ollama-managed instances (LRU order).
    managed = _get_llamaman_managed_instances()
    to_free = total - LLAMAMAN_MAX_MODELS + 1
    freed = 0
    while managed and freed < to_free:
        victim = managed.pop(0)
        logger.info(
            "llamaman: evicting %s (port %d) to make room (%d/%d total, max %d)",
            victim["model_name"], victim["port"], total - freed, total, LLAMAMAN_MAX_MODELS,
        )
        stop_instance_by_id(victim["id"])
        freed += 1

    # Check if first pass freed enough slots.
    if _count_running_instances() < LLAMAMAN_MAX_MODELS:
        return True

    # Still over limit - only proceed if the override setting allows evicting
    # admin-UI launched instances as well.
    if not _ollama_can_evict_admin_instances():
        return False

    # Second pass: also evict admin-UI instances (LRU order).
    remaining = _get_all_evictable_instances()
    while remaining and _count_running_instances() >= LLAMAMAN_MAX_MODELS:
        victim = remaining.pop(0)
        logger.info(
            "llamaman: evicting admin-launched %s (port %d) - override enabled",
            victim["model_name"], victim["port"],
        )
        stop_instance_by_id(victim["id"])

    return _count_running_instances() < LLAMAMAN_MAX_MODELS


def _wait_for_model_ready(host: str, port: int, timeout: float) -> bool:
    """Poll the llama-server /health endpoint until it reports ready."""
    from api.instances import wait_for_healthy
    return wait_for_healthy(host, port, timeout=timeout)


def _ensure_model_running(
    model_name: str,
    allow_eviction: bool = True,
) -> tuple[dict | None, str | None]:
    """Ensure a model instance exists and is at least launched.

    Returns the instance as soon as it is launched (status may still be
    ``"starting"``).  Callers are expected to use ``_wait_for_model_ready``
    on the server port before forwarding the actual request.

    allow_eviction controls whether LRU eviction may be used to free a slot.
    The Ollama API sets this True; the OpenAI API sets it False so it never
    displaces a running model - it either finds a free slot or returns 503.
    """
    from api.instances import (
        launch_instance, relaunch_inactive_instance, wait_for_healthy,
    )

    model = _find_model_by_name(model_name)
    if model is None:
        return None, f"model '{model_name}' not found"

    # Fast path: model is already healthy - no lock needed
    inst = _find_running_instance_for_model(model["path"])
    if inst and inst["status"] == "healthy":
        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["_last_request_at"] = time.time()
        return inst, None

    # Slow path: need to launch/relaunch/wait - serialize so concurrent
    # requests for different models don't evict each other's instances.
    with _llamaman_lock:
        # Re-check after acquiring lock (another thread may have launched it)
        inst = _find_running_instance_for_model(model["path"])
        if inst and inst["status"] in ("healthy", "starting"):
            with instances_lock:
                if inst["id"] in instances:
                    instances[inst["id"]]["_last_request_at"] = time.time()
            return inst, None

        existing = inst or _find_any_instance_for_model(model["path"])
        preset = get_storage().get_preset(model["path"]) or {}
        incoming_embedding_model = preset.get("embedding_model", False)

        if allow_eviction:
            # Evict LRU Ollama-managed instances (and admin-UI ones if the
            # override toggle is on) to stay within LLAMAMAN_MAX_MODELS.
            room = _evict_llamaman_instances_if_needed(
                incoming_embedding_model=incoming_embedding_model,
            )
            if not room:
                return None, (
                    f"model limit reached (LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}); "
                    "admin-launched models cannot be evicted via the API"
                )
        else:
            # OpenAI API: never evict - only proceed if there is already room.
            if not incoming_embedding_model and LLAMAMAN_MAX_MODELS > 0:
                if _count_running_instances() >= LLAMAMAN_MAX_MODELS:
                    return None, (
                        f"model limit reached (LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}); "
                        "the OpenAI API does not evict running models"
                    )

        if existing and existing["status"] in ("sleeping", "stopped"):
            # relaunch_inactive_instance blocks until healthy; if it
            # succeeds the instance is ready for requests immediately.
            if relaunch_inactive_instance(existing["id"]):
                return existing, None
            return None, "failed to wake model"

        port = find_available_port()
        if port is None:
            return None, "no ports available"

        inst, err = launch_instance(
            model_path=model["path"],
            port=port,
            n_gpu_layers=preset.get("n_gpu_layers", -1),
            ctx_size=preset.get("ctx_size", 4096),
            threads=preset.get("threads"),
            memory_limit=preset.get("memory_limit") or None,
            parallel=preset.get("parallel"),
            extra_args=preset.get("extra_args", ""),
            spec_enabled=preset.get("spec_enabled", False),
            spec_draft_n_max=preset.get("spec_draft_n_max"),
            gpu_devices=preset.get("gpu_devices") or None,
            idle_timeout_min=preset.get("idle_timeout_min", 0),
            max_concurrent=preset.get("max_concurrent", 0),
            max_queue_depth=preset.get("max_queue_depth", 200),
            share_queue=preset.get("share_queue", False),
            embedding_model=preset.get("embedding_model", False),
            proxy_sampling_override_enabled=bool(preset.get("proxy_sampling_override_enabled", False)),
            proxy_sampling_temperature=float(preset.get("proxy_sampling_temperature", 0.8)),
            proxy_sampling_top_k=int(preset.get("proxy_sampling_top_k", 40)),
            proxy_sampling_top_p=float(preset.get("proxy_sampling_top_p", 0.95)),
            proxy_sampling_presence_penalty=float(preset.get("proxy_sampling_presence_penalty", 0.0)),
            proxy_sampling_repeat_penalty=float(preset.get("proxy_sampling_repeat_penalty", 0.0)),
        )
        if err:
            return None, err

        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["_llamaman_managed"] = True
                instances[inst["id"]]["_last_request_at"] = time.time()

        logger.info("llamaman: auto-launched %s on port %d", model_name, port)

    # Return immediately - the model is launched but may still be loading.
    # The caller will poll for readiness before forwarding the request.
    return inst, None


def _gguf_meta_for(model_path: str, model_type: str | None) -> dict:
    if model_type and model_type != "gguf":
        return {}
    return get_cached_gguf_metadata(model_path)


def _details_from_gguf(model_path: str, model_type: str | None,
                       fallback_quant: str, gguf_meta: dict | None = None) -> dict:
    """Build the Ollama-style `details` block, sourcing family/parameter_size
    from GGUF metadata when available and falling back to filename heuristics."""
    name = model_name_from_path(model_path)
    if gguf_meta is None:
        gguf_meta = _gguf_meta_for(model_path, model_type)
    arch = (gguf_meta.get("general.architecture") or "").strip()
    if arch:
        family = arch
        families = [arch]
    else:
        family = name.split("-")[0] if "-" in name else name
        families = [family]
    size_label = (gguf_meta.get("general.size_label") or "").strip()
    if not size_label:
        size_label = format_param_count(gguf_meta.get("general.parameter_count"))
    return {
        "parent_model": "",
        "format": model_type or "gguf",
        "family": family,
        "families": families,
        "parameter_size": size_label,
        "quantization_level": fallback_quant or "",
    }


def _llamaman_model_entry(m: dict) -> dict:
    name = model_name_from_path(m["path"])
    mtime = datetime.fromtimestamp(
        Path(m["path"]).stat().st_mtime if Path(m["path"]).exists() else 0,
        tz=timezone.utc,
    ).isoformat()
    return {
        "name": name,
        "model": name,
        "modified_at": mtime,
        "size": m["size_bytes"],
        "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_URL, m['path']).hex}",
        "details": _details_from_gguf(m["path"], m.get("type"), m.get("quant", "")),
    }


def _instance_container_alive(inst: dict) -> bool:
    container_id = inst.get("container_id")
    if not container_id:
        return False
    return is_container_running(container_id)


def _probe_server_ready(host: str, port: int) -> bool:
    try:
        resp = http_requests.get(
            f"http://{host}:{port}/health",
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        return resp.json().get("status") == "ok"
    except Exception:
        return False


_NEVER_EXPIRES_TS = 4102444800  # 2100-01-01, sentinel for instances with idle_timeout=0


def _llamaman_ps_entry(model_path: str, model_meta: dict | None = None,
                       started_at: float | None = None,
                       inst_config: dict | None = None,
                       last_request_at: float | None = None) -> dict:
    model_meta = model_meta or {}
    inst_config = inst_config or {}
    model_name = model_name_from_path(model_path)
    size_bytes = model_meta.get("size_bytes")
    if size_bytes is None:
        try:
            size_bytes = os.path.getsize(model_path)
        except OSError:
            size_bytes = 0

    gguf_meta = _gguf_meta_for(model_path, model_meta.get("type"))
    arch = (gguf_meta.get("general.architecture") or "").strip()

    # Active runtime context for this instance, baked at launch via --ctx-size.
    # Falls back to the model's trained max if config is somehow missing.
    ctx_size = inst_config.get("ctx_size")
    if not ctx_size and arch:
        ctx_size = gguf_meta.get(f"{arch}.context_length")
    try:
        context_length = int(ctx_size) if ctx_size else 0
    except (TypeError, ValueError):
        context_length = 0

    # Approximate VRAM from layer offload + GGUF block_count when partial.
    block_count = gguf_meta.get(f"{arch}.block_count") if arch else None
    try:
        block_count = int(block_count) if block_count else None
    except (TypeError, ValueError):
        block_count = None
    n_gpu_layers = inst_config.get("n_gpu_layers", -1)
    size_vram = estimate_model_vram(size_bytes, n_gpu_layers, block_count)

    # Honor the configured idle timeout for the unload deadline; if it's 0
    # (never reaped) emit a far-future sentinel so clients don't think the
    # model is about to disappear.
    idle_min = int(inst_config.get("idle_timeout_min") or 0)
    base_ts = last_request_at or started_at or time.time()
    expires_ts = base_ts + idle_min * 60 if idle_min > 0 else _NEVER_EXPIRES_TS

    details = _details_from_gguf(
        model_path,
        model_meta.get("type"),
        model_meta.get("quant", detect_quant(Path(model_path).stem)),
        gguf_meta=gguf_meta,
    )

    return {
        "name": model_name,
        "model": model_name,
        "size": size_bytes,
        "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_URL, model_path).hex}",
        "details": details,
        "expires_at": datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat(),
        "size_vram": size_vram,
        "context_length": context_length,
    }


def _list_loaded_models() -> list[dict]:
    model_index = {
        os.path.realpath(m["path"]): m
        for m in discover_models(MODELS_DIR)
    }
    live_by_path: dict[str, dict] = {}

    with instances_lock:
        tracked_instances = [dict(inst) for inst in instances.values()]

    for inst in tracked_instances:
        if not _instance_container_alive(inst):
            continue

        model_path = inst["model_path"]
        server_host = inst.get("_server_host", "localhost")
        server_port = inst.get("_server_port") or inst.get("_internal_port") or inst["port"]
        ready = _probe_server_ready(server_host, server_port)
        key = os.path.realpath(model_path)
        existing = live_by_path.get(key)

        if existing and existing.get("ready") and not ready:
            continue

        live_by_path[key] = {
            "model_path": model_path,
            "started_at": inst.get("started_at"),
            "ready": ready,
            "config": inst.get("config") or {},
            "last_request_at": inst.get("_last_request_at"),
        }

    loaded = [
        _llamaman_ps_entry(
            entry["model_path"],
            model_meta=model_index.get(os.path.realpath(entry["model_path"])),
            started_at=entry.get("started_at"),
            inst_config=entry.get("config"),
            last_request_at=entry.get("last_request_at"),
        )
        for entry in live_by_path.values()
    ]
    loaded.sort(key=lambda item: item["name"])
    return loaded


# ---------------------------------------------------------------------------
# Ollama >> OpenAI translation
# ---------------------------------------------------------------------------

def _translate_to_openai(body: dict) -> dict:
    openai_body = {
        "model": body.get("model", ""),
        "stream": body.get("stream", True),
    }

    if "messages" in body:
        openai_body["messages"] = body["messages"]

    if "prompt" in body and "messages" not in body:
        msgs = []
        if body.get("system"):
            msgs.append({"role": "system", "content": body["system"]})
        msgs.append({"role": "user", "content": body["prompt"]})
        openai_body["messages"] = msgs

    opts = body.get("options", {})
    if "temperature" in opts:
        openai_body["temperature"] = opts["temperature"]
    if "top_k" in opts:
        openai_body["top_k"] = opts["top_k"]
    if "top_p" in opts:
        openai_body["top_p"] = opts["top_p"]
    if "presence_penalty" in opts:
        openai_body["presence_penalty"] = opts["presence_penalty"]
    if "seed" in opts:
        openai_body["seed"] = opts["seed"]
    if "stop" in opts:
        openai_body["stop"] = opts["stop"]
    if "num_predict" in opts:
        openai_body["max_tokens"] = opts["num_predict"]

    for key in ("temperature", "top_k", "top_p", "presence_penalty", "seed", "stop", "max_tokens"):
        if key in body and key not in openai_body:
            openai_body[key] = body[key]

    return openai_body


def _stream_llamaman(host: str, port: int, openai_body: dict, model_name: str,
                     mode: str = "chat", inst_id: str | None = None,
                     handle=None):
    t_start = time.monotonic()
    t_first_token = None
    prompt_tokens = 0
    completion_tokens = 0
    accumulated: list[str] = []
    final_usage: dict | None = None
    final_status = 200

    def _content_field(token: str, thinking: str = ""):
        if mode == "chat":
            msg = {"role": "assistant", "content": token}
            if thinking:
                msg["thinking"] = thinking
            return {"message": msg}
        return {"response": (thinking + token) if thinking else token}

    def _done_obj(finish_reason: str = "stop", usage: dict | None = None):
        usage = usage or {}
        elapsed_ns = int((time.monotonic() - t_start) * 1e9)
        p_tokens = usage.get("prompt_tokens", prompt_tokens)
        c_tokens = usage.get("completion_tokens", completion_tokens)
        prompt_dur = int((t_first_token - t_start) * 1e9) if t_first_token else 0
        eval_dur = elapsed_ns - prompt_dur if prompt_dur else elapsed_ns
        return {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **_content_field(""),
            "done": True,
            "done_reason": finish_reason,
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": p_tokens,
            "prompt_eval_duration": prompt_dur,
            "eval_count": c_tokens,
            "eval_duration": eval_dur,
        }

    resp = None
    try:
        # Retry on connection errors (model may have just finished loading)
        last_err = None
        for _attempt in range(3):
            try:
                resp = http_requests.post(
                    f"http://{host}:{port}/v1/chat/completions",
                    json=openai_body,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except (http_requests.ConnectionError, http_requests.Timeout,
                    ConnectionRefusedError) as e:
                last_err = e
                time.sleep(2)
        if resp is None:
            raise last_err or ConnectionError("failed to connect to model server")
        if resp.status_code >= 400:
            error_text = resp.text[:500] if resp.text else f"HTTP {resp.status_code}"
            final_status = resp.status_code
            accumulated.append(f"Error: {error_text}")
            error_obj = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **_content_field(f"Error: {error_text}"),
                "done": True,
                "done_reason": "stop",
            }
            yield json.dumps(error_obj, ensure_ascii=False) + "\n"
            return
        resp.encoding = "utf-8"

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield json.dumps(_done_obj(), ensure_ascii=False) + "\n"
                    return

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                thinking = delta.get("reasoning_content", "")
                finish = choices[0].get("finish_reason")

                if token or thinking:
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                    completion_tokens += 1
                    if token:
                        accumulated.append(token)

                chunk_obj = {
                    "model": model_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **_content_field(token, thinking),
                    "done": False,
                }
                yield json.dumps(chunk_obj, ensure_ascii=False) + "\n"

                if finish:
                    final_usage = chunk.get("usage", {}) or None
                    yield json.dumps(_done_obj(finish, chunk.get("usage", {})), ensure_ascii=False) + "\n"
                    return

    except Exception as e:
        final_status = 500
        accumulated.append(f"Error: {e}")
        error_obj = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **_content_field(f"Error: {e}"),
            "done": True,
            "done_reason": "stop",
        }
        yield json.dumps(error_obj, ensure_ascii=False) + "\n"
    finally:
        if resp is not None:
            resp.close()
        if inst_id and completion_tokens > 0:
            elapsed = time.monotonic() - t_start
            tps = completion_tokens / elapsed if elapsed > 0 else None
            ttft = ((t_first_token - t_start) * 1000) if t_first_token else None
            update_instance_stats(inst_id, tokens_per_sec=tps, ttft_ms=ttft)
        if handle is not None:
            usage = final_usage or {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            handle.set_response(
                text="".join(accumulated),
                usage=usage,
                status_code=final_status,
            )


def _proxy_non_streaming(host: str, port: int, openai_body: dict, model_name: str,
                         mode: str = "chat", inst_id: str | None = None,
                         handle=None):
    t_start = time.monotonic()
    openai_body["stream"] = False
    # Retry on connection errors (model may have just finished loading)
    last_err = None
    resp = None
    for _attempt in range(3):
        try:
            resp = http_requests.post(
                f"http://{host}:{port}/v1/chat/completions",
                json=openai_body,
                timeout=REQUEST_TIMEOUT,
            )
            break
        except (http_requests.ConnectionError, http_requests.Timeout,
                ConnectionRefusedError) as e:
            last_err = e
            time.sleep(2)
    if resp is None:
        raise last_err or ConnectionError("failed to connect to model server")
    resp.raise_for_status()
    elapsed_ns = int((time.monotonic() - t_start) * 1e9)
    data = resp.json()
    choices = data.get("choices", [])
    usage = data.get("usage", {})

    result = {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done": True,
        "done_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": elapsed_ns,
    }

    if mode == "chat":
        msg = choices[0]["message"] if choices else {"role": "assistant", "content": ""}
        reasoning = msg.pop("reasoning_content", "")
        if reasoning:
            msg["thinking"] = reasoning
        result["message"] = msg
    else:
        result["response"] = choices[0]["message"]["content"] if choices else ""

    if inst_id:
        elapsed = (time.monotonic() - t_start)
        c_tokens = usage.get("completion_tokens", 0)
        tps = c_tokens / elapsed if elapsed > 0 and c_tokens else None
        update_instance_stats(inst_id, tokens_per_sec=tps)

    if handle is not None:
        if mode == "chat":
            msg = result.get("message") or {}
            resp_text = msg.get("content") or ""
        else:
            resp_text = result.get("response") or ""
        handle.set_response(text=resp_text, usage=usage, status_code=200)

    return result


def _handle_request(mode: str = "chat"):
    body = request.get_json(force=True)
    model_name = body.get("model", "").strip()
    if not model_name:
        return jsonify({"error": "model is required"}), 400

    if mode == "generate" and body.get("keep_alive") == 0 and not body.get("prompt"):
        return jsonify({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": "",
            "done": True,
        })

    inst, err = _ensure_model_running(model_name)
    if err:
        code = 503 if "model limit reached" in err else 500
        return jsonify({"error": err}), code

    if inst.get("config", {}).get("embedding_model"):
        return jsonify({"error": f"model '{model_name}' is embedding-only and cannot handle chat completions"}), 422

    server_host = inst.get("_server_host", "localhost")
    server_port = inst.get("_server_port") or inst.get("_internal_port") or inst["port"]

    # If the model was just launched it may still be loading.  Wait for it
    # to become healthy before forwarding the request so the prompt is not
    # lost to a connection-refused error.
    if inst.get("status") != "healthy":
        if not _wait_for_model_ready(server_host, server_port, MODEL_LOAD_TIMEOUT):
            return jsonify({"error": "model launched but did not become healthy in time"}), 500
        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["status"] = "healthy"
                started = instances[inst["id"]].get("started_at", 0)
                if started:
                    stats = instances[inst["id"]].setdefault("stats", {})
                    stats["model_load_time_s"] = round(time.time() - started, 1)

    handle = record_request(
        body,
        endpoint=f"ollama_{mode}",
        path=request.path,
        inst_id=inst["id"],
        model=model_name,
    )

    gate = get_gate(inst["id"])
    if gate:
        if not gate.acquire(timeout=REQUEST_TIMEOUT):
            if handle:
                handle.set_error(429, "request queue full")
                handle.finalize()
            return jsonify({"error": "request queue full"}), 429

    openai_body = _translate_to_openai(body)
    openai_body = apply_proxy_sampling_overrides(openai_body, inst.get("config", {}))
    stream_qp = request.args.get("stream", "").lower()
    if stream_qp in ("false", "0", "no"):
        stream = False
    else:
        stream = body.get("stream", True)

    stream_returned = False
    try:
        if stream:
            def _gated_stream():
                try:
                    yield from _stream_llamaman(server_host, server_port, openai_body,
                                                model_name, mode, inst_id=inst["id"],
                                                handle=handle)
                finally:
                    if gate:
                        gate.release()
                    if handle:
                        handle.finalize(streamed=True)
            stream_returned = True
            return Response(
                _gated_stream(),
                mimetype="application/x-ndjson",
            )

        result = _proxy_non_streaming(server_host, server_port, openai_body, model_name,
                                      mode, inst_id=inst["id"], handle=handle)
        return jsonify(result)
    except Exception as e:
        if handle:
            handle.set_error(500, str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        if gate and not stream_returned:
            gate.release()
        if handle and not stream_returned:
            handle.finalize(streamed=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/tags", methods=["GET"])
def llamaman_tags():
    models = discover_models(MODELS_DIR)
    return jsonify({"models": [_llamaman_model_entry(m) for m in models]})


@bp.route("/api/version", methods=["GET"])
def llamaman_version():
    return jsonify({"version": VERSION})


def _effective_ctx_for_model(model_path: str, gguf_meta: dict) -> int:
    """Resolve the context size a client would actually get for this model:
    a running instance's runtime ctx wins, then the saved preset, then the
    model's trained context_length from GGUF metadata."""
    with instances_lock:
        for inst in instances.values():
            if inst.get("model_path") != model_path:
                continue
            if inst.get("status") in ("stopped",):
                continue
            ctx = (inst.get("config") or {}).get("ctx_size")
            if ctx:
                try:
                    return int(ctx)
                except (TypeError, ValueError):
                    pass
    preset = get_storage().get_preset(model_path) or {}
    ctx = preset.get("ctx_size")
    if ctx:
        try:
            return int(ctx)
        except (TypeError, ValueError):
            pass
    arch = (gguf_meta.get("general.architecture") or "").strip()
    if arch:
        try:
            return int(gguf_meta.get(f"{arch}.context_length") or 0)
        except (TypeError, ValueError):
            pass
    return 0


@bp.route("/api/show", methods=["POST"])
def llamaman_show():
    body = request.get_json(force=True)
    model_name = body.get("model", body.get("name", "")).strip()
    model = _find_model_by_name(model_name)
    if model is None:
        return jsonify({"error": f"model '{model_name}' not found"}), 404

    entry = _llamaman_model_entry(model)
    gguf_meta = _gguf_meta_for(model["path"], model.get("type"))
    arch = (gguf_meta.get("general.architecture") or "").strip()

    # model_info mirrors the GGUF header, but we override <arch>.context_length
    # to the size a client would actually get (preset cap or running instance
    # ctx) so callers like hermes don't read the trained max and over-allocate.
    model_info = dict(gguf_meta)
    effective_ctx = _effective_ctx_for_model(model["path"], gguf_meta)
    if arch and effective_ctx:
        model_info[f"{arch}.context_length"] = effective_ctx
    if "general.architecture" not in model_info:
        model_info["general.architecture"] = entry["details"]["family"]

    template = gguf_meta.get("tokenizer.chat_template", "") or ""

    return jsonify({
        "modelfile": f"FROM {model['path']}",
        "parameters": "",
        "template": template,
        "details": entry["details"],
        "model_info": model_info,
    })


@bp.route("/api/ps", methods=["GET"])
def llamaman_ps():
    return jsonify({"models": _list_loaded_models()})


@bp.route("/api/chat", methods=["POST"])
def llamaman_chat():
    return _handle_request(mode="chat")


@bp.route("/api/generate", methods=["POST"])
def llamaman_generate():
    return _handle_request(mode="generate")


@bp.route("/v1/models", methods=["GET"])
def llamaman_v1_models():
    models = discover_models(MODELS_DIR)
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_name_from_path(m["path"]),
                "object": "model",
                "created": int(Path(m["path"]).stat().st_mtime) if Path(m["path"]).exists() else 0,
                "owned_by": "local",
            }
            for m in models
        ],
    })


@bp.route("/v1/chat/completions", methods=["POST"])
def llamaman_v1_chat():
    body = request.get_json(force=True)
    model_name = body.get("model", "").strip()
    if not model_name:
        return jsonify({"error": {"message": "model is required"}}), 400

    inst, err = _ensure_model_running(model_name, allow_eviction=False)
    if err:
        return jsonify({"error": {"message": err}}), 503

    if inst.get("config", {}).get("embedding_model"):
        return jsonify({"error": {"message": f"model '{model_name}' is embedding-only and cannot handle chat completions"}}), 422

    body = apply_proxy_sampling_overrides(body, inst.get("config", {}))

    server_host = inst.get("_server_host", "localhost")
    server_port = inst.get("_server_port") or inst.get("_internal_port") or inst["port"]
    inst_id = inst["id"]

    # Wait for the model to finish loading before forwarding
    if inst.get("status") != "healthy":
        if not _wait_for_model_ready(server_host, server_port, MODEL_LOAD_TIMEOUT):
            return jsonify({"error": {"message": "model launched but did not become healthy in time"}}), 500
        with instances_lock:
            if inst_id in instances:
                instances[inst_id]["status"] = "healthy"
                started = instances[inst_id].get("started_at", 0)
                if started:
                    stats = instances[inst_id].setdefault("stats", {})
                    stats["model_load_time_s"] = round(time.time() - started, 1)

    handle = record_request(
        body,
        endpoint="openai_chat",
        path=request.path,
        inst_id=inst_id,
        model=model_name,
    )

    gate = get_gate(inst_id)
    if gate:
        if not gate.acquire(timeout=REQUEST_TIMEOUT):
            if handle:
                handle.set_error(429, "request queue full")
                handle.finalize()
            return jsonify({"error": {"message": "request queue full"}}), 429

    stream = body.get("stream", False)
    stream_returned = False
    t_start = time.monotonic()
    try:
        # Retry on connection errors (transient failures right after load)
        last_err = None
        resp = None
        for _attempt in range(3):
            try:
                resp = http_requests.post(
                    f"http://{server_host}:{server_port}/v1/chat/completions",
                    json=body,
                    stream=stream,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except (http_requests.ConnectionError, http_requests.Timeout,
                    ConnectionRefusedError) as e:
                last_err = e
                time.sleep(2)
        if resp is None:
            raise last_err or ConnectionError("failed to connect to model server")
        if stream:
            def _relay():
                acc = SSEAccumulator() if handle else None
                try:
                    for chunk in resp.iter_content(chunk_size=None):
                        if acc is not None:
                            acc.feed(chunk)
                        yield chunk
                finally:
                    resp.close()
                    if gate:
                        gate.release()
                    # Can't extract token counts from a consumed stream,
                    # but still update timestamp and request count.
                    _touch_instance(inst_id)
                    update_instance_stats(inst_id)
                    if handle:
                        text, usage = acc.finish() if acc else ("", None)
                        handle.set_response(text=text, usage=usage,
                                            status_code=resp.status_code)
                        handle.finalize(streamed=True)
            stream_returned = True
            return Response(
                _relay(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        with resp:
            data = resp.json()
            _touch_instance(inst_id)
            # Extract stats from the non-streaming response
            usage = data.get("usage", {})
            c_tokens = usage.get("completion_tokens", 0)
            elapsed = time.monotonic() - t_start
            tps = c_tokens / elapsed if c_tokens and elapsed > 0 else None
            update_instance_stats(inst_id, tokens_per_sec=tps)
            if handle:
                choices = data.get("choices") or []
                msg = (choices[0].get("message") if choices else {}) or {}
                handle.set_response(text=msg.get("content") or "",
                                    usage=usage,
                                    status_code=resp.status_code)
            return jsonify(data), resp.status_code
    except Exception as e:
        if handle:
            handle.set_error(500, str(e))
        return jsonify({"error": {"message": str(e)}}), 500
    finally:
        if gate and not stream_returned:
            gate.release()
        if handle and not stream_returned:
            handle.finalize(streamed=False)


@bp.route("/v1/embeddings", methods=["POST"])
def llamaman_v1_embeddings():
    body = request.get_json(force=True)
    model_name = body.get("model", "").strip()
    if not model_name:
        return jsonify({"error": {"message": "model is required"}}), 400

    inst, err = _ensure_model_running(model_name, allow_eviction=False)
    if err:
        return jsonify({"error": {"message": err}}), 503

    if not inst.get("config", {}).get("embedding_model"):
        return jsonify({"error": {"message": f"model '{model_name}' is not configured as an embedding model"}}), 422

    server_host = inst.get("_server_host", "localhost")
    server_port = inst.get("_server_port") or inst.get("_internal_port") or inst["port"]
    inst_id = inst["id"]

    if inst.get("status") != "healthy":
        if not _wait_for_model_ready(server_host, server_port, MODEL_LOAD_TIMEOUT):
            return jsonify({"error": {"message": "model launched but did not become healthy in time"}}), 500
        with instances_lock:
            if inst_id in instances:
                instances[inst_id]["status"] = "healthy"

    handle = record_request(
        body,
        endpoint="openai_embed",
        path=request.path,
        inst_id=inst_id,
        model=model_name,
    )

    last_err = None
    resp = None
    try:
        for _attempt in range(3):
            try:
                resp = http_requests.post(
                    f"http://{server_host}:{server_port}/v1/embeddings",
                    json=body,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except (http_requests.ConnectionError, http_requests.Timeout,
                    ConnectionRefusedError) as e:
                last_err = e
                time.sleep(2)
        if resp is None:
            if handle:
                handle.set_error(502, str(last_err))
            return jsonify({"error": {"message": str(last_err)}}), 502

        _touch_instance(inst_id)
        data = resp.json()
        if handle:
            handle.set_response(text=json.dumps(data, separators=(",", ":")),
                                usage=data.get("usage"),
                                status_code=resp.status_code)
        return jsonify(data), resp.status_code
    finally:
        if handle:
            handle.finalize(streamed=False)


def _touch_instance(inst_id: str):
    """Update the last-request timestamp so idle timeout doesn't kill active models."""
    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["_last_request_at"] = time.time()
