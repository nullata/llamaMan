# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from config import LLAMAMAN_IDLE_TIMEOUT, HEALTH_CHECK_TIMEOUT, logger
from core.helpers import cleanup_download_dir, is_container_running, kill_instance_process, stop_container
from core.state import (
    instances, instances_lock,
    downloads, downloads_lock,
    save_state,
)
from proxy import idle_proxies, idle_proxies_lock, cleanup_orphan_idle_proxies, drain_gate

_last_cleanup_at: float = 0.0
_CLEANUP_INTERVAL = 3600  # run at most once per hour

_last_orphan_scan_at: float = 0.0
_ORPHAN_SCAN_INTERVAL = 60  # seconds

_last_stale_cleanup_at: float = 0.0
_DEFAULT_FAILED_DOWNLOAD_RETRY_COUNT = 3

_last_image_check_at: float = 0.0
_IMAGE_CHECK_INTERVAL = 3600  # check hourly whether auto-update is due

_last_request_log_prune_at: float = 0.0
_REQUEST_LOG_PRUNE_INTERVAL = 3600  # prune request_log once per hour


def _run_cleanup() -> None:
    from storage import get_storage
    from api.instances import release_instance_reservations
    storage = get_storage()
    cleanup = storage.get_settings().get("cleanup", {})
    changed = False
    cleanup_patch = {}
    now = time.time()

    if cleanup.get("downloads_enabled"):
        cleanup_patch["downloads_last_run_at"] = now
        max_age_s = (cleanup.get("downloads_max_age_hours") or 24) * 3600
        cutoff = now - max_age_s
        with downloads_lock:
            to_remove = [
                did for did, dl in downloads.items()
                if dl["status"] in ("completed", "failed", "cancelled")
                and dl.get("started_at", 0) < cutoff
            ]
            for did in to_remove:
                del downloads[did]
                logger.info("Cleanup: removed download %s", did)
        if to_remove:
            changed = True

    if cleanup.get("instances_enabled"):
        cleanup_patch["instances_last_run_at"] = now
        max_age_s = (cleanup.get("instances_max_age_hours") or 24) * 3600
        cutoff = now - max_age_s
        container_ids_to_stop = []
        with instances_lock:
            to_remove = [
                iid for iid, inst in instances.items()
                if inst["status"] == "stopped"
                and inst.get("started_at", 0) < cutoff
            ]
            for iid in to_remove:
                cid = instances[iid].get("container_id")
                if cid:
                    container_ids_to_stop.append(cid)
                release_instance_reservations(iid)
                del instances[iid]
                logger.info("Cleanup: removed instance %s", iid)
        for cid in container_ids_to_stop:
            stop_container(cid)
        if to_remove:
            changed = True

    if cleanup.get("orphan_containers_enabled"):
        cleanup_patch["orphan_containers_last_run_at"] = now
        try:
            from core.helpers import list_llama_containers
            with instances_lock:
                tracked_cids = {
                    inst.get("container_id") for inst in instances.values() if inst.get("container_id")
                }
            for container in list_llama_containers(all=True):
                if container.id in tracked_cids:
                    continue
                if container.status not in ("exited", "dead", "created"):
                    continue  # running/paused ones are left for the orphan-adoption scan
                stop_container(container.id)
                logger.info(
                    "Cleanup: removed orphaned container %s (status=%s)",
                    container.id[:12], container.status,
                )
        except Exception as e:
            logger.warning("Orphan container cleanup error: %s", e)

    if changed:
        save_state()
    if cleanup_patch:
        storage.merge_settings({"cleanup": cleanup_patch})


def _is_container_dead(inst: dict) -> bool:
    """Return True if the instance's backing container is no longer running."""
    container_id = inst.get("container_id")
    if not container_id:
        return True
    return not is_container_running(container_id)


def _run_stale_record_cleanup() -> None:
    """Remove instance records whose backing container is no longer running."""
    from storage import get_storage
    storage = get_storage()
    cleanup = storage.get_settings().get("cleanup", {})
    if not cleanup.get("stale_records_enabled"):
        return

    changed = False
    now = time.time()

    with idle_proxies_lock:
        live_proxies = set(idle_proxies.keys())

    with instances_lock:
        inst_ids = list(instances.keys())

    for inst_id in inst_ids:
        with instances_lock:
            inst = instances.get(inst_id)
            if inst is None:
                continue
            status = inst["status"]

        if status in ("starting", "healthy"):
            if _is_container_dead(inst):
                container_id = inst.get("container_id")
                crashed = False
                auto_restart = False
                with instances_lock:
                    inst = instances.get(inst_id)
                    if inst and inst["status"] not in ("stopped", "sleeping"):
                        if container_id:
                            stop_container(container_id)
                        inst["container_id"] = None
                        inst["status"] = "stopped"
                        stats = inst.setdefault("stats", {})
                        stats["crash_count"] = stats.get("crash_count", 0) + 1
                        logger.info(
                            "Stale record cleanup: marked %s stopped (container dead, crashes: %d)",
                            inst_id, stats["crash_count"],
                        )
                        changed = True
                        crashed = True
                        auto_restart = bool((inst.get("config") or {}).get("auto_restart_on_crash", False))
                # Backstop for the 5s poller: redistribute queued work to peers,
                # and honor opt-in crash recovery (loop-guarded, off-thread).
                if crashed:
                    drain_gate(inst_id)
                    if auto_restart:
                        _maybe_auto_restart(inst_id)

        elif status == "sleeping":
            has_proxy = inst_id in live_proxies
            if not has_proxy:
                with instances_lock:
                    inst = instances.get(inst_id)
                    if inst and inst["status"] == "sleeping":
                        inst["status"] = "stopped"
                        logger.info(
                            "Stale record cleanup: marked sleeping %s stopped (no proxy)",
                            inst_id,
                        )
                        changed = True

    if changed:
        save_state()
    storage.merge_settings({"cleanup": {"stale_records_last_run_at": now}})


def _get_failed_download_retry_settings() -> tuple[bool, int]:
    from storage import get_storage

    settings = get_storage().get_settings()
    retry_enabled = bool(settings.get("auto_retry_failed_downloads", False))
    try:
        retry_limit = int(settings.get("retry_count_per_failed_download", _DEFAULT_FAILED_DOWNLOAD_RETRY_COUNT))
    except (TypeError, ValueError):
        retry_limit = _DEFAULT_FAILED_DOWNLOAD_RETRY_COUNT
    return retry_enabled, max(1, retry_limit)


def _handle_download_exit(dl_id: str, exit_code: int) -> None:
    retry_enabled, retry_limit = _get_failed_download_retry_settings()
    should_save = False

    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None or dl["status"] != "downloading":
            return

        kill_instance_process(dl)
        dl["pid"] = 0

        if exit_code == 0:
            dl["status"] = "completed"
            should_save = True
        else:
            retry_attempts = int(dl.get("retry_attempts", 0) or 0)
            if retry_enabled and retry_attempts < retry_limit:
                from api.downloads import restart_download_in_place

                err = restart_download_in_place(
                    dl,
                    reset_started_at=True,
                    increment_retry_attempts=True,
                )
                if err is None:
                    logger.info(
                        "Download auto-retried: %s (%d/%d)",
                        dl_id,
                        dl["retry_attempts"],
                        retry_limit,
                    )
                    should_save = True
                else:
                    logger.warning("Download auto-retry failed for %s: %s", dl_id, err)
                    dl["status"] = "failed"
                    should_save = True
            else:
                dl["status"] = "failed"
                should_save = True

    if should_save:
        save_state()


def _prune_request_log():
    from storage import get_storage
    storage = get_storage()
    days = storage.get_settings().get("recording_retention_days", 30)
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 30
    if days <= 0:
        return  # 0 = keep forever
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    pruned = storage.prune_request_log(cutoff)
    if pruned:
        logger.info("request_log: pruned %d records older than %d days", pruned, days)


# Auto-restart-on-crash (opt-in per instance via config["auto_restart_on_crash"]).
# Bounded so a model that crashes on every launch (bad GGUF, OOM) can't hot-loop
# Docker forever: at most _AUTO_RESTART_MAX relaunches per instance within a
# rolling _AUTO_RESTART_WINDOW_S window. Old timestamps age out, so an instance
# that runs fine for a while and crashes much later restarts again from scratch.
_auto_restart_lock = threading.Lock()
_auto_restart_log: dict[str, list[float]] = {}
_AUTO_RESTART_MAX = 3
_AUTO_RESTART_WINDOW_S = 600


def _maybe_auto_restart(inst_id: str) -> None:
    """Relaunch a crashed instance that opted into auto-restart, unless it has
    crash-looped recently. The relaunch blocks until the model is healthy, so it
    runs on a daemon thread to keep the poller responsive. Only the crash path
    calls this - user stops and idle-sleep never reach it."""
    now = time.time()
    with _auto_restart_lock:
        recent = [t for t in _auto_restart_log.get(inst_id, ()) if now - t < _AUTO_RESTART_WINDOW_S]
        if len(recent) >= _AUTO_RESTART_MAX:
            _auto_restart_log[inst_id] = recent
            logger.warning(
                "Instance %s auto-restart suppressed: %d crashes within %ds (crash loop)",
                inst_id, len(recent), _AUTO_RESTART_WINDOW_S,
            )
            return
        recent.append(now)
        _auto_restart_log[inst_id] = recent

    def _run():
        from api.instances import relaunch_inactive_instance
        try:
            if relaunch_inactive_instance(inst_id):
                logger.info("Instance %s auto-restarted after crash", inst_id)
            else:
                logger.warning("Instance %s auto-restart failed", inst_id)
        except Exception as e:
            logger.warning("Instance %s auto-restart error: %s", inst_id, e)

    threading.Thread(target=_run, daemon=True, name=f"auto-restart-{inst_id[:8]}").start()


def _background_poller():
    global _last_cleanup_at, _last_orphan_scan_at, _last_stale_cleanup_at, _last_image_check_at
    global _last_request_log_prune_at
    while True:
        time.sleep(5)

        now = time.time()

        # --- Periodic cleanup ---
        if now - _last_cleanup_at >= _CLEANUP_INTERVAL:
            _last_cleanup_at = now
            try:
                _run_cleanup()
            except Exception as e:
                logger.warning("Cleanup error: %s", e)

        # --- Stale record cleanup ---
        try:
            from storage import get_storage
            stale_interval_min = (
                get_storage().get_settings()
                .get("cleanup", {})
                .get("stale_records_interval_min", 5)
            ) or 5
            stale_interval_s = stale_interval_min * 60
        except Exception:
            stale_interval_s = 300
        if now - _last_stale_cleanup_at >= stale_interval_s:
            _last_stale_cleanup_at = now
            try:
                _run_stale_record_cleanup()
            except Exception as e:
                logger.warning("Stale record cleanup error: %s", e)

        # --- Request log retention ---
        if now - _last_request_log_prune_at >= _REQUEST_LOG_PRUNE_INTERVAL:
            _last_request_log_prune_at = now
            try:
                _prune_request_log()
            except Exception as e:
                logger.warning("request_log prune error: %s", e)

        # --- Docker image auto-update ---
        if now - _last_image_check_at >= _IMAGE_CHECK_INTERVAL:
            _last_image_check_at = now
            try:
                from api.images import check_and_pull_if_needed
                from config import LLAMA_IMAGE
                if LLAMA_IMAGE:
                    triggered = check_and_pull_if_needed(LLAMA_IMAGE)
                    if triggered:
                        logger.info("Image auto-update triggered for %s", LLAMA_IMAGE)
            except Exception as e:
                logger.warning("Image auto-update check error: %s", e)

        # --- Periodic orphan scan ---
        if now - _last_orphan_scan_at >= _ORPHAN_SCAN_INTERVAL:
            _last_orphan_scan_at = now
            try:
                from core.state import adopt_orphans
                n = adopt_orphans()
                if n:
                    logger.info("Orphan scan: adopted %d untracked llama-server container(s)", n)
            except Exception as e:
                logger.warning("Orphan scan error: %s", e)

        # --- Instance health ---
        with instances_lock:
            inst_ids = list(instances.keys())

        removed_orphans = cleanup_orphan_idle_proxies(set(inst_ids))
        if removed_orphans:
            logger.info("Proxy cleanup: removed %d orphan idle proxy listener(s)", removed_orphans)

        for inst_id in inst_ids:
            with instances_lock:
                inst = instances.get(inst_id)
                if inst is None or inst["status"] in ("stopped", "sleeping"):
                    continue
                container_id = inst.get("container_id")
                server_host = inst.get("_server_host", "localhost")
                port = inst.get("_server_port") or inst.get("_internal_port") or inst["port"]

            # Detect container death.
            container_dead = not container_id or not is_container_running(container_id)

            if container_dead:
                auto_restart = False
                with instances_lock:
                    inst = instances.get(inst_id)
                    if inst and inst["status"] not in ("stopped", "sleeping"):
                        if container_id:
                            stop_container(container_id)
                        inst["container_id"] = None
                        inst["status"] = "stopped"
                        stats = inst.setdefault("stats", {})
                        stats["crash_count"] = stats.get("crash_count", 0) + 1
                        logger.info("Instance %s auto-stopped (container died, crashes: %d)",
                                    inst_id, stats["crash_count"])
                        auto_restart = bool((inst.get("config") or {}).get("auto_restart_on_crash", False))
                # Redistribute the dead worker's QUEUED requests to peers instead
                # of rejecting them (drain, not cancel). Called outside instances_lock
                # because drain_gate takes it internally.
                drain_gate(inst_id)
                save_state()
                # Opt-in crash recovery (loop-guarded, off-thread). Only crashes
                # reach here - user stops and idle-sleep set status before this.
                if auto_restart:
                    _maybe_auto_restart(inst_id)
                continue

            try:
                resp = requests.get(
                    f"http://{server_host}:{port}/health",
                    timeout=HEALTH_CHECK_TIMEOUT,
                )
                new_status = "healthy" if resp.json().get("status") == "ok" else "starting"
            except Exception:
                new_status = "starting"

            status_changed = False
            with instances_lock:
                if inst_id in instances and instances[inst_id]["status"] not in ("stopped", "sleeping"):
                    old_status = instances[inst_id]["status"]
                    if new_status != old_status:
                        instances[inst_id]["status"] = new_status
                        status_changed = True
                    if new_status == "healthy" and old_status == "starting":
                        started = instances[inst_id].get("started_at", 0)
                        if started:
                            stats = instances[inst_id].setdefault("stats", {})
                            stats["model_load_time_s"] = round(time.time() - started, 1)
                        logger.info("Instance %s is now healthy (was %s)", inst_id, old_status)

            if status_changed:
                save_state()

        # --- Idle timeout reaper ---
        for inst_id in inst_ids:
            with instances_lock:
                inst = instances.get(inst_id)
                if inst is None or inst["status"] not in ("healthy",):
                    continue

                config = inst.get("config", {})
                timeout_min = config.get("idle_timeout_min", 0)
                if inst.get("_llamaman_managed") and LLAMAMAN_IDLE_TIMEOUT > 0:
                    timeout_min = timeout_min or LLAMAMAN_IDLE_TIMEOUT

                if timeout_min <= 0:
                    continue

                last_req = inst.get("_last_request_at", inst.get("started_at", now))
                idle_secs = now - last_req

                if idle_secs < timeout_min * 60:
                    continue

                with idle_proxies_lock:
                    has_proxy = inst_id in idle_proxies
                is_llamaman = inst.get("_llamaman_managed", False)

            from api.instances import stop_instance_by_id, sleep_instance_by_id

            if has_proxy or is_llamaman:
                sleep_instance_by_id(inst_id)
                logger.info("Idle reaper: %s slept after %d min idle",
                            inst_id, int(idle_secs / 60))
            else:
                stop_instance_by_id(inst_id)
                logger.info("Idle reaper: %s stopped after %d min idle",
                            inst_id, int(idle_secs / 60))

        # --- Download process monitoring ---
        with downloads_lock:
            dl_ids = list(downloads.keys())

        for dl_id in dl_ids:
            with downloads_lock:
                dl = downloads.get(dl_id)
                if dl is None or dl["status"] in ("completed", "failed", "cancelled", "paused"):
                    continue
                proc = dl.get("_process")

            if proc is None:
                continue

            exit_code = proc.poll()
            if exit_code is None:
                continue

            _handle_download_exit(dl_id, exit_code)


# The cluster heartbeat runs on its OWN thread, NOT inside the main poller tick.
# The poller does Docker-heavy work (orphan scans, health checks, container
# stops) that can stretch a tick well past the online window when a peer drives
# Docker on this node from its UI. If the heartbeat shared that tick, this node
# would flap "offline" to its peers mid-interaction, and their dispatchers would
# skip it (breaking cross-node routing and surfacing 500s). A dedicated thread
# keeps the heartbeat on a strict cadence regardless of how busy the poller is.
CLUSTER_HEARTBEAT_INTERVAL_S = 5


def _cluster_heartbeat_loop():
    from api.cluster import publish_cluster_heartbeat
    while True:
        try:
            publish_cluster_heartbeat()
        except Exception as e:
            logger.warning("cluster heartbeat error: %s", e)
        time.sleep(CLUSTER_HEARTBEAT_INTERVAL_S)


def start_background_poller():
    logger.info(
        "Background poller started (cleanup=%ds, orphan-scan=%ds, poll=5s, heartbeat=%ds)",
        _CLEANUP_INTERVAL, _ORPHAN_SCAN_INTERVAL, CLUSTER_HEARTBEAT_INTERVAL_S,
    )
    threading.Thread(target=_cluster_heartbeat_loop, daemon=True).start()
    thread = threading.Thread(target=_background_poller, daemon=True)
    thread.start()
    return thread
