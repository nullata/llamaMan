# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import threading
import time
import uuid
from pathlib import Path

from config import LLAMA_CONTAINER_PREFIX, logger
from core.helpers import (
    get_docker_client, is_container_running, resolve_llama_endpoint, stop_container,
)


instances: dict[str, dict] = {}
instances_lock = threading.Lock()

downloads: dict[str, dict] = {}
downloads_lock = threading.Lock()

# Serializes the entire snapshot+write cycle in save_state so a slow
# writer can't overwrite a newer snapshot from another thread.
_save_lock = threading.Lock()


def update_instance_stats(inst_id: str, tokens_per_sec: float | None = None,
                          ttft_ms: float | None = None):
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return
        stats = inst.setdefault("stats", {})
        stats["total_requests"] = stats.get("total_requests", 0) + 1
        if tokens_per_sec is not None:
            stats["last_tokens_per_sec"] = round(tokens_per_sec, 2)
        if ttft_ms is not None:
            stats["last_ttft_ms"] = round(ttft_ms, 1)


def save_state():
    from storage import get_storage
    storage = get_storage()

    # Phase-timed (LLAMAMAN_PERF_LOG=1): "wait" is time blocked behind another
    # in-flight save, "write" is the backend write itself (a full delete+insert
    # of every row on MariaDB). One line for the whole save at the end.
    from core.perf import phase
    with phase("save_state.total"):
      t_wait0 = time.monotonic()
      with _save_lock:
        t_wait = time.monotonic() - t_wait0
        inst_list = []
        with instances_lock:
            for inst in instances.values():
                inst_list.append({
                    "id": inst["id"],
                    "model_name": inst["model_name"],
                    "model_path": inst["model_path"],
                    "port": inst["port"],
                    "container_id": inst.get("container_id", ""),
                    "container_name": inst.get("container_name", ""),
                    "status": inst["status"],
                    "log_file": inst.get("log_file", ""),
                    "config": inst.get("config", {}),
                    "started_at": inst.get("started_at", 0),
                    "llamaman_managed": inst.get("_llamaman_managed", False),
                    "internal_port": inst.get("_internal_port"),
                    "stats": inst.get("stats", {}),
                })

        dl_list = []
        with downloads_lock:
            for dl in downloads.values():
                dl_list.append({
                    "id": dl["id"],
                    "repo_id": dl["repo_id"],
                    "filename": dl.get("filename", ""),
                    "dest_path": dl.get("dest_path", ""),
                    "status": dl["status"],
                    "log_file": dl.get("log_file", ""),
                    "started_at": dl.get("started_at", 0),
                    "hf_token_id": dl.get("_hf_token_id", ""),
                    "per_model_speed_limit_mbps": dl.get("per_model_speed_limit_mbps", 0),
                    "retry_attempts": dl.get("retry_attempts", 0),
                    # Set only for model-update re-pulls. Persisted (not
                    # underscore-prefixed) so a restart mid-update still knows
                    # to swap the staged files in when the retry finishes.
                    "update_model_path": dl.get("update_model_path", ""),
                    "update_temp_dir": dl.get("update_temp_dir", ""),
                    "update_sha256": dl.get("update_sha256", ""),
                })

        t_write0 = time.monotonic()
        try:
            from core.cluster import get_node_id
            storage.save_state(inst_list, dl_list, node_id=get_node_id())
        except Exception as e:
            logger.warning("Failed to save state: %s", e)
        finally:
            from config import PERF_LOG
            if PERF_LOG:
                logger.info(
                    "perf save_state.write %.0fms  wait=%.0fms  insts=%d dls=%d",
                    (time.monotonic() - t_write0) * 1000.0, t_wait * 1000.0,
                    len(inst_list), len(dl_list))


def adopt_orphans() -> int:
    """Find running llama-server containers not tracked by the manager and adopt them.

    Scans Docker containers with the llamaman label. Returns the count adopted.
    """
    from core.helpers import list_llama_containers

    containers = list_llama_containers()
    if not containers:
        return 0

    adopted = 0
    from storage import get_storage
    storage = get_storage()

    with instances_lock:
        tracked_ids = {inst.get("container_id") for inst in instances.values() if inst.get("container_id")}
        active_ports = {inst["port"] for inst in instances.values() if inst["status"] not in ("stopped",)}

    for container in containers:
        cid = container.id
        if cid in tracked_ids:
            continue

        labels = container.labels or {}
        inst_id = labels.get("llamaman.instance_id")
        model_path = labels.get("llamaman.model_path")
        port_str = labels.get("llamaman.port")
        config_str = labels.get("llamaman.config", "{}")

        if not inst_id or not model_path or not port_str:
            continue

        try:
            port = int(port_str)
        except ValueError:
            continue

        if port in active_ports:
            logger.warning(
                "Orphan container %s on port %d conflicts with a tracked instance - stopping",
                cid[:12], port,
            )
            stop_container(cid)
            continue

        try:
            config = json.loads(config_str)
        except Exception:
            config = {}

        preset = storage.get_preset(model_path) or {}
        orphan_config = {
            **config,
            "embedding_model": preset.get("embedding_model", config.get("embedding_model", False)),
            "proxy_sampling_override_enabled": preset.get("proxy_sampling_override_enabled", False),
            "proxy_sampling_temperature": preset.get("proxy_sampling_temperature", 0.8),
            "proxy_sampling_top_k": preset.get("proxy_sampling_top_k", 40),
            "proxy_sampling_top_p": preset.get("proxy_sampling_top_p", 0.95),
            "proxy_sampling_presence_penalty": preset.get("proxy_sampling_presence_penalty", 0.0),
        }

        container_name = container.name
        adopt_host, adopt_port = resolve_llama_endpoint(container_name, port)
        inst = {
            "id": inst_id,
            "model_name": Path(model_path).name,
            "model_path": model_path,
            "port": port,
            "status": "starting",  # poller will verify and flip to healthy
            "container_id": cid,
            "container_name": container_name,
            "log_file": "",
            "config": orphan_config,
            "started_at": time.time(),
            "_server_host": adopt_host,
            "_server_port": adopt_port,
            "_last_request_at": time.time(),
            "stats": {
                "model_load_time_s": None,
                "last_tokens_per_sec": None,
                "last_ttft_ms": None,
                "total_requests": 0,
                "crash_count": 0,
            },
        }
        with instances_lock:
            instances[inst_id] = inst
            tracked_ids.add(cid)
            active_ports.add(port)

        logger.info(
            "Adopted orphan container %s port %d model %s%s",
            cid[:12], port, inst["model_name"],
            " [embedding]" if orphan_config.get("embedding_model") else "",
        )
        adopted += 1

    if adopted > 0:
        save_state()

    return adopted


def load_state():
    """Restore instance and download history from disk on startup.

    For instances that were running when we last saved:
    - If the container is still running, reattach to it.
    - If the container is gone, mark the instance as stopped (or sleeping if it has a proxy).

    Returns a list of (inst_id, proxy_port, internal_port) tuples for
    instances that need their idle proxies restarted.
    """
    from proxy import create_gate
    from storage import get_storage
    from core.cluster import get_node_id
    storage = get_storage()

    node_id = get_node_id()
    saved_instances = storage.load_instances(node_id)
    saved_downloads = storage.load_downloads(node_id)

    restore_proxies = []

    for entry in saved_instances:
        config = entry.get("config", {})
        idle_timeout = config.get("idle_timeout_min", 0)
        max_concurrent = config.get("max_concurrent", 0)
        internal_port = entry.get("internal_port")
        saved_status = entry.get("status", "stopped")
        saved_container_id = entry.get("container_id", "")
        container_name = entry.get("container_name", "")
        has_proxy = internal_port and (
            idle_timeout > 0
            or max_concurrent > 0
            or config.get("proxy_sampling_override_enabled", False)
        )

        if saved_status in ("stopped", "stopping"):
            # User explicitly stopped it (or crashed mid async-stop grace). If a
            # container somehow still exists, kill it - "stopping" persisted
            # means the docker stop never got to complete, so treat exactly like
            # "stopped" on restore and let the orphan reaper mop up whatever the
            # SIGTERM grace didn't reach.
            if saved_container_id and is_container_running(saved_container_id):
                logger.info("Stopping orphaned %s instance container %s",
                            saved_status, saved_container_id[:12])
                stop_container(saved_container_id)
            restored_status = "stopped"
            saved_container_id = ""
        elif saved_status in ("healthy", "starting"):
            if saved_container_id and is_container_running(saved_container_id):
                restored_status = "starting"  # poller will flip to healthy
                logger.info("Reattaching to orphaned instance %s (container %s)",
                            entry.get("model_name", "?"), saved_container_id[:12])
            elif has_proxy:
                restored_status = "sleeping"
                saved_container_id = ""
            else:
                restored_status = "stopped"
                saved_container_id = ""
        elif saved_status == "sleeping":
            if has_proxy:
                restored_status = "sleeping"
            else:
                restored_status = "stopped"
            saved_container_id = ""
        else:
            restored_status = "stopped"
            saved_container_id = ""

        if container_name:
            restore_host, restore_port = resolve_llama_endpoint(
                container_name, internal_port or entry.get("port", 0))
        else:
            restore_host, restore_port = "localhost", None

        inst = {
            "id": entry["id"],
            "model_name": entry["model_name"],
            "model_path": entry["model_path"],
            "port": entry.get("port", 0),
            "status": restored_status,
            "container_id": saved_container_id,
            "container_name": container_name,
            "log_file": entry.get("log_file", ""),
            "config": config,
            "started_at": entry.get("started_at", 0),
            "_server_host": restore_host,
            "_server_port": restore_port,
            "_llamaman_managed": entry.get("llamaman_managed", False),
            "_last_request_at": time.time(),
            "stats": entry.get("stats", {
                "model_load_time_s": None,
                "last_tokens_per_sec": None,
                "last_ttft_ms": None,
                "total_requests": 0,
            }),
        }
        if internal_port:
            inst["_internal_port"] = internal_port
        instances[inst["id"]] = inst

        # Rebuild in-memory-only pieces that die with the previous llamaman
        # process. Both apply to reattached-running ("starting") and restored
        # sleeping instances - previously this block only handled "sleeping",
        # so a healthy instance whose container survived a llamaman restart
        # would come back with its sidecar port unbound (the werkzeug server
        # for the public 8000-8020 lived in the old process and died with it)
        # and without its RequestGate - which meant `_public_instance` dropped
        # the `queue` field, the "Queue N/M active · K queued" indicator
        # disappeared from the UI, and the compat proxy at :42069 stopped
        # enforcing max_concurrent for hits routed to this instance.
        #  - restore_proxies drives app.py's post-boot start_idle_proxy loop,
        #    which rebinds the public port and installs the wake-on-request /
        #    forward-to-internal-port WSGI app. `has_proxy` gates it: only
        #    instances that were launched with a sidecar (idle_timeout,
        #    max_concurrent, or proxy sampling overrides) had one to restore.
        #  - create_gate rebuilds the RequestGate. Gated on max_concurrent > 0
        #    to match the launch path in api/instances.py; create_gate itself
        #    also short-circuits at max_concurrent <= 0 so this is defense in
        #    depth.
        if restored_status in ("starting", "sleeping") and has_proxy:
            restore_proxies.append((inst["id"], inst["port"], internal_port))
        if restored_status in ("starting", "sleeping") and max_concurrent > 0:
            create_gate(inst["id"], max_concurrent,
                        config.get("max_queue_depth", 200),
                        model_path=inst["model_path"],
                        share_queue=config.get("share_queue", False))

    for entry in saved_downloads:
        status = entry.get("status", "failed")
        if status == "downloading":
            status = "failed"
        dl = {
            "id": entry["id"],
            "repo_id": entry["repo_id"],
            "filename": entry.get("filename", ""),
            "dest_path": entry.get("dest_path", ""),
            "status": status,
            "pid": 0,
            "log_file": entry.get("log_file", ""),
            "started_at": entry.get("started_at", 0),
            "_hf_token_id": entry.get("hf_token_id", ""),
            "per_model_speed_limit_mbps": entry.get("per_model_speed_limit_mbps", 0),
            "retry_attempts": entry.get("retry_attempts", 0),
            "update_model_path": entry.get("update_model_path", ""),
            "update_temp_dir": entry.get("update_temp_dir", ""),
            "update_sha256": entry.get("update_sha256", ""),
            "_process": None,
            "_log_fh": None,
        }
        downloads[dl["id"]] = dl

    logger.info("Restored state: %d instances, %d downloads",
                len(saved_instances), len(saved_downloads))

    n = adopt_orphans()
    if n:
        logger.info("Startup orphan scan: adopted %d untracked llama-server container(s)", n)

    return restore_proxies
