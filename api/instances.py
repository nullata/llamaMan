# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import threading
import time
import uuid
from pathlib import Path

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from config import (
    HEALTH_CHECK_TIMEOUT,
    HOST_LOGS_DIR,
    HOST_MODELS_DIR,
    INTERNAL_PORT_RANGE_END,
    INTERNAL_PORT_RANGE_START,
    LLAMA_CONTAINER_PORT,
    LLAMA_CONTAINER_PREFIX,
    LLAMA_GPU_DEVICES,
    LLAMA_IMAGE,
    LLAMA_NETWORK,
    LLAMAMAN_MAX_MODELS,
    LOGS_DIR,
    MODELS_DIR,
    MODEL_LOAD_TIMEOUT,
    PORT_RANGE_END,
    PORT_RANGE_START,
    logger,
)
from core.gpu import get_vendor
from core.helpers import (
    build_llama_cmd, ensure_docker_network, find_available_port,
    get_docker_client, is_container_running, is_port_available,
    kill_instance_process, public_dict, read_log_file, resolve_llama_endpoint,
    stop_container, stream_log_file,
)
from core.proxy_sampling import parse_proxy_sampling_config
from core.state import (
    instances, instances_lock, save_state,
)
from proxy import (
    create_gate, get_gate, remove_gate, refresh_gate,
    start_idle_proxy, stop_idle_proxy,
)
from storage import get_storage

bp = Blueprint("instances", __name__)

# Fixed port llama-server listens on inside every container.
LLAMA_CONTAINER_PORT = 8080


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_instance(inst: dict) -> dict:
    d = public_dict(inst)
    d["last_request_at"] = inst.get("_last_request_at")
    if inst.get("_internal_port") is not None:
        d["internal_port"] = inst.get("_internal_port")
    gate = get_gate(inst["id"])
    if gate:
        d["queue"] = {
            "active": gate.active,
            "queued": gate.queued,
            "max_concurrent": gate.max_concurrent,
            "max_queue_depth": gate.max_queue_depth,
        }
    return d


def _merge_preset_into_config(model_path: str, config: dict) -> dict:
    """Overlay the latest saved preset onto an instance config."""
    from storage import get_storage

    merged = dict(config)
    from api.presets import resolve_preset_for_node
    from core.cluster import get_node_id
    preset = resolve_preset_for_node(get_storage().get_preset(model_path) or {}, get_node_id())
    if preset:
        for key in (
            "n_gpu_layers",
            "ctx_size",
            "threads",
            "memory_limit",
            "parallel",
            "extra_args",
            "spec_enabled",
            "spec_draft_n_max",
            "gpu_devices",
            "idle_timeout_min",
            "max_concurrent",
            "max_queue_depth",
            "share_queue",
            "share_queue_group",
            "share_queue_fallback",
            "embedding_model",
            "auto_restart_on_crash",
            "proxy_sampling_override_enabled",
            "proxy_sampling_temperature",
            "proxy_sampling_top_k",
            "proxy_sampling_top_p",
            "proxy_sampling_presence_penalty",
            "proxy_sampling_repeat_penalty",
        ):
            if key in preset:
                merged[key] = preset[key]
    return merged


def _parse_required_positive_int(body: dict, field_name: str) -> tuple[int | None, str | None]:
    raw = body.get(field_name)
    if raw in (None, ""):
        return None, f"{field_name} is required"
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, f"{field_name} must be an integer"
    if value <= 0:
        return None, f"{field_name} must be greater than 0"
    return value, None


def _admin_ui_enforces_eviction() -> bool:
    from core.node_settings import effective_from_settings
    settings = get_storage().get_settings()
    return bool(effective_from_settings(settings, "admin_ui_enforce_max_models", False))


def _count_running_chat_instances(exclude_instance_id: str | None = None) -> int:
    with instances_lock:
        return sum(
            1 for inst in instances.values()
            if inst["id"] != exclude_instance_id
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        )


def _would_ui_launch_exceed_limit(
    incoming_embedding_model: bool = False,
    exclude_instance_id: str | None = None,
) -> bool:
    if LLAMAMAN_MAX_MODELS <= 0 or incoming_embedding_model:
        return False
    return _count_running_chat_instances(exclude_instance_id=exclude_instance_id) >= LLAMAMAN_MAX_MODELS


def _get_lru_chat_instances(
    exclude_instance_id: str | None = None,
    ollama_managed_first: bool = False,
) -> list[dict]:
    with instances_lock:
        candidates = [
            inst for inst in instances.values()
            if inst["id"] != exclude_instance_id
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    if ollama_managed_first:
        candidates.sort(key=lambda inst: (
            not inst.get("_llamaman_managed", False),
            inst.get("_last_request_at", inst.get("started_at", 0)),
        ))
    else:
        candidates.sort(key=lambda inst: inst.get("_last_request_at", inst.get("started_at", 0)))
    return candidates


def _evict_instances_for_ui_launch_if_needed(
    incoming_embedding_model: bool = False,
    exclude_instance_id: str | None = None,
) -> None:
    if LLAMAMAN_MAX_MODELS <= 0 or incoming_embedding_model:
        return

    total = _count_running_chat_instances(exclude_instance_id=exclude_instance_id)
    if total < LLAMAMAN_MAX_MODELS:
        return

    to_free = total - LLAMAMAN_MAX_MODELS + 1
    freed = 0
    for victim in _get_lru_chat_instances(
        exclude_instance_id=exclude_instance_id,
        ollama_managed_first=True,
    ):
        if freed >= to_free:
            break
        logger.info(
            "ui: evicting %s (port %d) to make room (%d/%d total, max %d)",
            victim["model_name"], victim["port"], total - freed, total, LLAMAMAN_MAX_MODELS,
        )
        stop_instance_by_id(victim["id"])
        freed += 1


def wait_for_healthy(server_host: str, port: int, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = http_requests.get(
                f"http://{server_host}:{port}/health",
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            if resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_log_relay(container, log_file: str) -> threading.Thread:
    """Start a daemon thread that streams container logs to a file (Option B)."""
    def _relay():
        try:
            with open(log_file, "a") as fh:
                for chunk in container.logs(stream=True, follow=True):
                    try:
                        fh.write(chunk.decode("utf-8", errors="replace"))
                        fh.flush()
                    except Exception:
                        break
        except Exception:
            pass

    t = threading.Thread(target=_relay, daemon=True)
    t.start()
    return t


def _resolve_gpu_devices(per_instance: str | None) -> str:
    """Resolve effective GPU device string.

    Priority: per-instance > global LLAMA_GPU_DEVICES > empty (all).
    Returns a comma-separated string of device indices, or "" for all.

    The literal "all" (the UI placeholder text, which a user might type) is
    normalized to "" so it means all GPUs everywhere - both for device
    attachment and the instance card's GPU label.
    """
    effective = (per_instance or LLAMA_GPU_DEVICES or "").strip()
    if effective.lower() == "all":
        return ""
    return effective


def _make_device_requests(gpu_devices: str | None):
    """Return Docker device_requests for CUDA GPU passthrough."""
    import docker
    effective = _resolve_gpu_devices(gpu_devices)
    if effective:
        device_ids = [d.strip() for d in effective.split(",") if d.strip()]
        return [docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])]
    return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]


def _make_rocm_devices() -> list[str]:
    return ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]


def _run_container(
    inst_id: str,
    container_name: str,
    model_path: str,
    server_port: int,
    config: dict,
    log_file: str,
) -> tuple:
    """Start a llama-server Docker container. Returns (container, error_str)."""
    import docker

    cmd = build_llama_cmd(model_path, LLAMA_CONTAINER_PORT, config)
    gpu_devices = config.get("gpu_devices") or None
    image_name = config.get("image") or LLAMA_IMAGE

    ensure_docker_network()

    # Bind mounts for the sibling container.
    # SOURCE must be a path on the Docker HOST (the daemon's filesystem).
    # When llamaman itself runs in Docker, HOST_MODELS_DIR / HOST_LOGS_DIR are
    # the real host paths; they default to MODELS_DIR / LOGS_DIR for bare-metal.
    volumes = {
        HOST_MODELS_DIR: {"bind": MODELS_DIR, "mode": "ro"},
        HOST_LOGS_DIR: {"bind": LOGS_DIR, "mode": "rw"},
    }

    # Publish container port → host port so the Werkzeug proxy and direct
    # clients can reach it via localhost/host network.
    port_bindings = {LLAMA_CONTAINER_PORT: server_port}

    kwargs = dict(
        image=image_name,
        command=cmd,
        name=container_name,
        network=LLAMA_NETWORK,
        volumes=volumes,
        ports=port_bindings,
        detach=True,
        labels={
            "llamaman.instance_id": inst_id,
            "llamaman.model_path": model_path,
            "llamaman.port": str(server_port),
            "llamaman.config": json.dumps(config),
        },
    )

    threads = config.get("threads")
    if threads:
        kwargs["nano_cpus"] = int(float(threads) * 1e9)

    memory_limit = config.get("memory_limit")
    if memory_limit:
        kwargs["mem_limit"] = memory_limit

    try:
        n_gpu_layers = int(config.get("n_gpu_layers", -1))
    except (TypeError, ValueError):
        n_gpu_layers = -1

    vendor = get_vendor()
    if n_gpu_layers == 0:
        # CPU-only: attach no GPU devices at all. Besides honoring the user's
        # intent, this avoids Docker's CDI GPU discovery, which errors on hosts
        # without a configured GPU runtime (e.g. WSL without the NVIDIA
        # container toolkit).
        pass
    elif vendor == "rocm":
        kwargs["devices"] = _make_rocm_devices()
        kwargs["group_add"] = ["video", "render"]
        effective_gpus = _resolve_gpu_devices(gpu_devices)
        if effective_gpus:
            kwargs.setdefault("environment", {})["ROCR_VISIBLE_DEVICES"] = effective_gpus
    elif vendor == "intel":
        # Intel Arc: /dev/dri access only (no /dev/kfd). Per-instance GPU
        # selection is not supported for Intel (no SYCL_VISIBLE_DEVICES equivalent).
        kwargs["devices"] = ["/dev/dri:/dev/dri"]
        kwargs["group_add"] = ["video", "render"]
    else:
        # NVIDIA (cuda) or unknown/CPU - use Docker device_requests
        kwargs["device_requests"] = _make_device_requests(gpu_devices)

    try:
        client = get_docker_client()
        container = client.containers.run(**kwargs)
        _start_log_relay(container, log_file)
        return container, None
    except docker.errors.ImageNotFound:
        return None, f"Docker image '{image_name}' not found. Pull it in the Docker Images tab, or run: docker pull {image_name}"
    except docker.errors.APIError as e:
        return None, f"Docker API error: {e}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Launch / Stop / Sleep
# ---------------------------------------------------------------------------

def relaunch_inactive_instance(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None or inst["status"] not in ("sleeping", "stopped"):
            return inst is not None and inst["status"] in ("healthy", "starting")

        prior_status = inst["status"]
        config = _merge_preset_into_config(inst["model_path"], inst["config"])
        model_path = inst["model_path"]
        internal_port = inst.get("_internal_port", inst["port"])
        container_name = inst.get("container_name", f"{LLAMA_CONTAINER_PREFIX}{inst_id[:8]}")

    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["config"] = config

    # Reconcile the gate with the merged config. refresh_gate alone is a no-op
    # when no gate exists (stopped instances drop theirs via remove_gate, and
    # instances originally launched with max_concurrent=0 never had one). Cover
    # all four transitions: create/refresh/remove/none.
    merged_mc = int(config.get("max_concurrent", 0) or 0)
    if merged_mc > 0:
        if get_gate(inst_id) is None:
            create_gate(
                inst_id,
                merged_mc,
                int(config.get("max_queue_depth", 200) or 200),
                model_path=model_path,
                share_queue=bool(config.get("share_queue", False)),
            )
        else:
            refresh_gate(inst_id)
    else:
        remove_gate(inst_id)

    log_file = os.path.join(LOGS_DIR, f"{inst_id}.log")

    if not is_port_available(internal_port):
        logger.warning(
            "Cannot relaunch inactive instance %s: port %d is already occupied",
            inst_id, internal_port,
        )
        with instances_lock:
            inst = instances.get(inst_id)
            if inst:
                inst["status"] = "stopped"
        save_state()
        return False

    logger.info(
        "Relaunching %s instance %s on server port %d",
        prior_status, inst_id, internal_port,
    )

    try:
        with open(log_file, "a") as fh:
            fh.write(f"\n--- Relaunched at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except Exception:
        pass

    container, err = _run_container(inst_id, container_name, model_path, internal_port, config, log_file)
    if err:
        logger.error("Failed to relaunch %s: %s", inst_id, err)
        return False

    server_host, health_port = resolve_llama_endpoint(container_name, internal_port)

    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["status"] = "starting"
            inst["container_id"] = container.id
            inst["container_name"] = container_name
            inst["_server_host"] = server_host
            inst["_server_port"] = health_port
            inst["started_at"] = time.time()
            inst["_last_request_at"] = time.time()

    save_state()

    if not wait_for_healthy(server_host, health_port, timeout=MODEL_LOAD_TIMEOUT):
        logger.warning("Relaunched %s but it did not become healthy", inst_id)
        return False

    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["status"] = "healthy"
    save_state()
    return True


def relaunch_sleeping_instance(inst_id: str) -> bool:
    return relaunch_inactive_instance(inst_id)


def launch_instance(model_path, port, n_gpu_layers=-1, ctx_size=4096,
                    threads=None, memory_limit=None, parallel=None, extra_args="",
                    spec_enabled=False, spec_draft_n_max=None,
                    gpu_devices=None, idle_timeout_min=0,
                    max_concurrent=0, max_queue_depth=200,
                    share_queue=False, share_queue_group="",
                    share_queue_fallback=False,
                    embedding_model=False,
                    auto_restart_on_crash=False,
                    image=None,
                    proxy_sampling_override_enabled=False,
                    proxy_sampling_temperature=0.8,
                    proxy_sampling_top_k=40,
                    proxy_sampling_top_p=0.95,
                    proxy_sampling_presence_penalty=0.0,
                    proxy_sampling_repeat_penalty=0.0):
    with instances_lock:
        used_ports = {i["port"] for i in instances.values() if i["status"] not in ("stopped",)}
    if port in used_ports:
        return None, f"Port {port} is already in use"

    needs_proxy = idle_timeout_min > 0 or max_concurrent > 0 or proxy_sampling_override_enabled
    if needs_proxy:
        internal_port = find_available_port(
            exclude={port},
            range_start=INTERNAL_PORT_RANGE_START,
            range_end=INTERNAL_PORT_RANGE_END,
        )
        if internal_port is None:
            return None, "no internal ports available for proxy"
        server_port = internal_port
    else:
        server_port = port
        internal_port = None

    # Group/fallback are meaningless without share_queue; force-empty them so a
    # value typed before the toggle was flipped off can't leak into the config
    # and confuse cluster routing. The UI also gates the inputs, so this is
    # defense in depth for API callers that bypass the form.
    if not share_queue:
        share_queue_group = ""
        share_queue_fallback = False

    config = {
        "n_gpu_layers": n_gpu_layers,
        "ctx_size": ctx_size,
        "threads": threads,
        "memory_limit": memory_limit,
        "parallel": parallel,
        "extra_args": extra_args,
        "spec_enabled": spec_enabled,
        "spec_draft_n_max": spec_draft_n_max,
        "gpu_devices": gpu_devices,
        "idle_timeout_min": idle_timeout_min,
        "max_concurrent": max_concurrent,
        "max_queue_depth": max_queue_depth,
        "share_queue": share_queue,
        # Optional alias-based cross-node grouping (cluster) - empty string
        # means "group by filename" (legacy behavior). Normalized at the
        # boundary so the cluster matcher (which lowercases) stays consistent.
        "share_queue_group": (share_queue_group or "").strip().lower(),
        "share_queue_fallback": bool(share_queue_fallback),
        "embedding_model": embedding_model,
        "auto_restart_on_crash": auto_restart_on_crash,
        "image": (image or "").strip() or LLAMA_IMAGE,
        "proxy_sampling_override_enabled": proxy_sampling_override_enabled,
        "proxy_sampling_temperature": proxy_sampling_temperature,
        "proxy_sampling_top_k": proxy_sampling_top_k,
        "proxy_sampling_top_p": proxy_sampling_top_p,
        "proxy_sampling_presence_penalty": proxy_sampling_presence_penalty,
        "proxy_sampling_repeat_penalty": proxy_sampling_repeat_penalty,
    }

    inst_id = str(uuid.uuid4())
    container_name = f"{LLAMA_CONTAINER_PREFIX}{inst_id[:8]}"
    log_file = os.path.join(LOGS_DIR, f"{inst_id}.log")
    model_name = Path(model_path).name

    if not is_port_available(port):
        return None, f"Port {port} is already occupied by another process"
    if internal_port and not is_port_available(internal_port):
        return None, f"Internal port {internal_port} is already occupied by another process"

    logger.info(
        "Launching: %s model=%s port=%d (CUDA_VISIBLE_DEVICES=%s)",
        container_name, model_name, server_port, gpu_devices or "all",
    )

    container, err = _run_container(inst_id, container_name, model_path, server_port, config, log_file)
    if err:
        return None, err

    server_host, health_port = resolve_llama_endpoint(container_name, server_port)

    instance = {
        "id": inst_id,
        "model_name": model_name,
        "model_path": model_path,
        "port": port,
        "status": "starting",
        "container_id": container.id,
        "container_name": container_name,
        "log_file": log_file,
        "config": config,
        "started_at": time.time(),
        "_server_host": server_host,
        "_server_port": health_port,
        "_last_request_at": time.time(),
        "stats": {
            "model_load_time_s": None,
            "last_tokens_per_sec": None,
            "last_ttft_ms": None,
            "total_requests": 0,
            "crash_count": 0,
        },
    }

    if internal_port:
        instance["_internal_port"] = internal_port

    with instances_lock:
        instances[inst_id] = instance

    if needs_proxy and internal_port:
        start_idle_proxy(inst_id, port, internal_port)

    if max_concurrent > 0:
        create_gate(inst_id, max_concurrent, max_queue_depth,
                    model_path=model_path, share_queue=share_queue)

    if share_queue:
        # Publish this model's sampling/spec as the cluster group default
        # (last writer wins). No-op when clustering is disabled.
        try:
            from api.cluster import record_group_overrides
            record_group_overrides(model_path, config)
        except Exception as e:
            logger.warning("record_group_overrides failed: %s", e)

    save_state()
    return instance, None


def stop_instance_by_id(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return False
        container_id = inst.get("container_id")
        inst["status"] = "stopped"
        inst["container_id"] = None
    if container_id:
        stop_container(container_id)
    release_instance_reservations(inst_id)
    save_state()
    return True


def sleep_instance_by_id(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return False
        container_id = inst.get("container_id")
        inst["status"] = "sleeping"
        inst["container_id"] = None
    if container_id:
        stop_container(container_id)
    refresh_gate(inst_id)
    save_state()
    logger.info("Instance %s put to sleep (idle timeout)", inst_id)
    return True


def _restore_restarted_instance(old: dict) -> None:
    """Restore a removed instance record if restart launch fails."""
    with instances_lock:
        instances[old["id"]] = old

    if old["status"] == "sleeping":
        internal_port = old.get("_internal_port")
        if internal_port:
            start_idle_proxy(old["id"], old["port"], internal_port)
        max_concurrent = old.get("config", {}).get("max_concurrent", 0)
        if max_concurrent > 0:
            create_gate(
                old["id"],
                max_concurrent,
                old.get("config", {}).get("max_queue_depth", 200),
                model_path=old["model_path"],
                share_queue=old.get("config", {}).get("share_queue", False),
            )


def release_instance_reservations(inst_id: str) -> None:
    """Release proxy/gate resources tied to a public instance port."""
    stop_idle_proxy(inst_id)
    remove_gate(inst_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/instances/container-stats", methods=["GET"])
def api_container_stats():
    """Return CPU% and memory usage for all healthy/starting containers in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import docker as docker_sdk

    with instances_lock:
        targets = {
            inst_id: inst["container_id"]
            for inst_id, inst in instances.items()
            if inst.get("status") in ("healthy", "starting")
            and inst.get("container_id")
        }

    if not targets:
        return jsonify({})

    # GPU info - query once, map per instance
    from core.gpu import get_vendor, query_gpus
    vendor = get_vendor()
    all_gpus = query_gpus() or []  # [{index, name, ...}]
    gpu_map = {g["index"]: g["name"] for g in all_gpus}

    def _gpu_labels(inst_id: str) -> list[str]:
        with instances_lock:
            inst = instances.get(inst_id)
            if not inst:
                return []
        # CPU-only launches (GPU Layers = 0) get no GPU attached, so don't
        # label them with one. Mirrors the device gating in _run_container.
        try:
            if int(inst.get("config", {}).get("n_gpu_layers", -1)) == 0:
                return []
        except (TypeError, ValueError):
            pass
        if vendor == "intel":
            name = gpu_map.get(0, "Intel Arc")
            return [name]
        if vendor not in ("cuda", "rocm") or not gpu_map:
            return []
        effective = _resolve_gpu_devices(inst.get("config", {}).get("gpu_devices"))
        if effective:
            indices = [int(x.strip()) for x in effective.split(",") if x.strip().isdigit()]
        else:
            indices = sorted(gpu_map.keys())
        return [f"{gpu_map[i]} [{i}]" for i in indices if i in gpu_map]

    client = get_docker_client()

    def _fetch(inst_id: str, container_id: str):
        try:
            c = client.containers.get(container_id)
            raw = c.stats(stream=False)

            # CPU %
            cpu = raw.get("cpu_stats", {})
            precpu = raw.get("precpu_stats", {})
            cpu_usage = cpu.get("cpu_usage", {}).get("total_usage", 0)
            precpu_usage = precpu.get("cpu_usage", {}).get("total_usage", 0)
            sys_usage = cpu.get("system_cpu_usage", 0)
            presys_usage = precpu.get("system_cpu_usage", 0)
            num_cpus = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage") or []) or 1
            cpu_delta = cpu_usage - precpu_usage
            sys_delta = sys_usage - presys_usage
            cpu_pct = round((cpu_delta / sys_delta) * num_cpus * 100, 1) if sys_delta > 0 else 0.0

            # Memory
            mem = raw.get("memory_stats", {})
            mem_used = mem.get("usage", 0)
            cache = mem.get("stats", {}).get("cache", 0)
            mem_used = max(0, mem_used - cache)
            mem_limit = mem.get("limit", 0)

            return inst_id, {
                "cpu_pct": cpu_pct,
                "num_cpus": num_cpus,
                "mem_used_mb": round(mem_used / (1024 * 1024)),
                "mem_limit_mb": round(mem_limit / (1024 * 1024)),
            }
        except Exception:
            return inst_id, None

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as ex:
        futures = {ex.submit(_fetch, iid, cid): iid for iid, cid in targets.items()}
        for f in as_completed(futures):
            inst_id, stat = f.result()
            if stat is not None:
                results[inst_id] = stat

    # Attach GPU labels and CPU quota (derived from config, no container inspection needed)
    for inst_id in targets:
        labels = _gpu_labels(inst_id)
        with instances_lock:
            inst = instances.get(inst_id)
            threads = inst.get("config", {}).get("threads") if inst else None
        cpu_quota = int(threads) if threads else None
        entry = results.setdefault(inst_id, {})
        entry["gpus"] = labels
        entry["cpu_quota"] = cpu_quota

    return jsonify(results)


@bp.route("/api/next-port")
def api_next_port():
    port = find_available_port()
    if port is None:
        return jsonify({"error": "No ports available", "port": None}), 409
    return jsonify({"port": port})


@bp.route("/api/instances", methods=["GET"])
def api_instances_list():
    with instances_lock:
        safe = [_public_instance(inst) for inst in instances.values()]
    return jsonify(safe)


@bp.route("/api/instances", methods=["POST"])
def api_instances_create():
    body = request.get_json(force=True)
    model_path = body.get("model_path", "").strip()
    if not model_path:
        return jsonify({"error": "model_path is required"}), 400

    ctx_size, ctx_err = _parse_required_positive_int(body, "ctx_size")
    if ctx_err:
        return jsonify({"error": ctx_err}), 400
    proxy_sampling_config, proxy_sampling_err = parse_proxy_sampling_config(body)
    if proxy_sampling_err:
        return jsonify({"error": proxy_sampling_err}), 400

    incoming_embedding_model = bool(body.get("embedding_model", False))
    confirm_overcommit = bool(body.get("confirm_overcommit", False))
    if _admin_ui_enforces_eviction():
        _evict_instances_for_ui_launch_if_needed(
            incoming_embedding_model=incoming_embedding_model,
        )
    elif _would_ui_launch_exceed_limit(incoming_embedding_model=incoming_embedding_model) and not confirm_overcommit:
        return jsonify({
            "error": f"You're about to launch an instance beyond LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}. Do you want to proceed?",
            "confirm_required": True,
        }), 409

    inst, err = launch_instance(
        model_path=model_path,
        port=int(body.get("port", 8000)),
        n_gpu_layers=int(body.get("n_gpu_layers", -1)),
        ctx_size=ctx_size,
        threads=body.get("threads"),
        memory_limit=body.get("memory_limit", "").strip() or None,
        parallel=body.get("parallel"),
        extra_args=body.get("extra_args", "").strip(),
        spec_enabled=bool(body.get("spec_enabled", False)),
        spec_draft_n_max=body.get("spec_draft_n_max"),
        gpu_devices=body.get("gpu_devices", "").strip() or None,
        idle_timeout_min=int(body.get("idle_timeout_min", 0)),
        max_concurrent=int(body.get("max_concurrent", 0)),
        max_queue_depth=int(body.get("max_queue_depth", 200)),
        share_queue=bool(body.get("share_queue", False)),
        share_queue_group=body.get("share_queue_group", ""),
        share_queue_fallback=bool(body.get("share_queue_fallback", False)),
        embedding_model=bool(body.get("embedding_model", False)),
        auto_restart_on_crash=bool(body.get("auto_restart_on_crash", False)),
        image=body.get("image", "").strip() or None,
        **proxy_sampling_config,
    )
    if err:
        code = 409 if "already in use" in err else 500
        return jsonify({"error": err}), code
    return jsonify(_public_instance(inst)), 201


@bp.route("/api/instances/<inst_id>", methods=["DELETE"])
def api_instances_delete(inst_id):
    if not stop_instance_by_id(inst_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": "stopped"})


@bp.route("/api/instances/<inst_id>/restart", methods=["POST"])
def api_instances_restart(inst_id):
    body = request.get_json(silent=True) or {}
    with instances_lock:
        old = instances.get(inst_id)
        if old is None:
            return jsonify({"error": "Not found"}), 404
        if old["status"] not in ("stopped", "sleeping"):
            return jsonify({"error": "Instance must be stopped or sleeping before restarting"}), 409
        old = {
            **old,
            "config": _merge_preset_into_config(old["model_path"], old.get("config", {})),
            "stats": dict(old.get("stats", {})),
        }
        model_path = old["model_path"]
        config = old["config"]
        preferred_port = old["port"]

    incoming_embedding_model = bool(config.get("embedding_model", False))
    confirm_overcommit = bool(body.get("confirm_overcommit", False))
    if _admin_ui_enforces_eviction():
        _evict_instances_for_ui_launch_if_needed(
            incoming_embedding_model=incoming_embedding_model,
            exclude_instance_id=inst_id,
        )
    elif _would_ui_launch_exceed_limit(
        incoming_embedding_model=incoming_embedding_model,
        exclude_instance_id=inst_id,
    ) and not confirm_overcommit:
        return jsonify({
            "error": f"You're about to launch an instance beyond LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}. Do you want to proceed?",
            "confirm_required": True,
        }), 409

    release_instance_reservations(inst_id)
    with instances_lock:
        instances.pop(inst_id, None)

    port = preferred_port if is_port_available(preferred_port) else None
    if port is None:
        for p in range(PORT_RANGE_START, PORT_RANGE_END + 1):
            if is_port_available(p):
                port = p
                break
    if port is None:
        _restore_restarted_instance(old)
        save_state()
        return jsonify({"error": "No ports available"}), 409

    inst, err = launch_instance(
        model_path=model_path,
        port=port,
        n_gpu_layers=config.get("n_gpu_layers", -1),
        ctx_size=config.get("ctx_size", 4096),
        threads=config.get("threads"),
        memory_limit=config.get("memory_limit") or None,
        parallel=config.get("parallel"),
        extra_args=config.get("extra_args", ""),
        spec_enabled=config.get("spec_enabled", False),
        spec_draft_n_max=config.get("spec_draft_n_max"),
        gpu_devices=config.get("gpu_devices"),
        idle_timeout_min=config.get("idle_timeout_min", 0),
        max_concurrent=config.get("max_concurrent", 0),
        max_queue_depth=config.get("max_queue_depth", 200),
        share_queue=config.get("share_queue", False),
        share_queue_group=config.get("share_queue_group", ""),
        share_queue_fallback=config.get("share_queue_fallback", False),
        embedding_model=config.get("embedding_model", False),
        image=config.get("image"),
        proxy_sampling_override_enabled=bool(config.get("proxy_sampling_override_enabled", False)),
        proxy_sampling_temperature=float(config.get("proxy_sampling_temperature", 0.8)),
        proxy_sampling_top_k=int(config.get("proxy_sampling_top_k", 40)),
        proxy_sampling_top_p=float(config.get("proxy_sampling_top_p", 0.95)),
        proxy_sampling_presence_penalty=float(config.get("proxy_sampling_presence_penalty", 0.0)),
        proxy_sampling_repeat_penalty=float(config.get("proxy_sampling_repeat_penalty", 0.0)),
    )
    if err:
        _restore_restarted_instance(old)
        save_state()
        code = 409 if "already in use" in err else 500
        return jsonify({"error": err}), code

    return jsonify(_public_instance(inst)), 201


@bp.route("/api/instances/<inst_id>", methods=["GET"])
def api_instances_get(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return jsonify({"error": "Not found"}), 404
        d = _public_instance(inst)
    return jsonify(d)


@bp.route("/api/instances/<inst_id>/remove", methods=["DELETE"])
def api_instances_remove(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return jsonify({"error": "Not found"}), 404
        if inst["status"] not in ("stopped",):
            return jsonify({"error": "Instance must be stopped before removing"}), 409
        container_id = inst.get("container_id")
    if container_id:
        stop_container(container_id)
    release_instance_reservations(inst_id)
    with instances_lock:
        instances.pop(inst_id, None)
    save_state()
    return jsonify({"status": "removed"})


@bp.route("/api/instances/<inst_id>/logs")
def api_instance_logs(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
    if inst is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"lines": read_log_file(inst["log_file"])})


@bp.route("/api/instances/<inst_id>/logs/stream")
def api_instance_logs_stream(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
    if inst is None:
        return jsonify({"error": "Not found"}), 404
    return Response(stream_log_file(inst["log_file"]),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
