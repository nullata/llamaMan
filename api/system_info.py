# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import psutil
from flask import Blueprint, jsonify

bp = Blueprint("system_info", __name__)


def _read_cgroup_value(path: str) -> int | None:
    try:
        with open(path) as f:
            val = int(f.read().strip())
            return val
    except (FileNotFoundError, ValueError, OSError):
        return None


def _read_cgroup_stats(path: str) -> dict[str, int] | None:
    try:
        stats = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                key, value = parts
                stats[key] = int(value)
        return stats
    except (FileNotFoundError, ValueError, OSError):
        return None


def _get_container_cpu_limit() -> float | None:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] == "max":
                return None
            quota, period = int(parts[0]), int(parts[1])
            return quota / period
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    quota = _read_cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and quota > 0:
        return quota / period
    return None


def _get_container_memory_limit() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val == "max":
                return None
            return int(val)
    except (FileNotFoundError, OSError, ValueError):
        pass
    limit = _read_cgroup_value("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit and limit < (1 << 62):
        return limit
    return None


def _get_container_memory_usage() -> int | None:
    usage = _read_cgroup_value("/sys/fs/cgroup/memory.current")
    if usage is not None:
        return usage
    return _read_cgroup_value("/sys/fs/cgroup/memory/memory.usage_in_bytes")


def _get_container_inactive_file_bytes() -> int:
    stats = _read_cgroup_stats("/sys/fs/cgroup/memory.stat")
    if stats:
        return max(
            stats.get("inactive_file", stats.get("total_inactive_file", 0)),
            0,
        )

    legacy_stats = _read_cgroup_stats("/sys/fs/cgroup/memory/memory.stat")
    if legacy_stats:
        return max(legacy_stats.get("total_inactive_file", 0), 0)

    return 0


@bp.route("/api/system-info")
def api_system_info():
    try:
        cpu_limit = _get_container_cpu_limit()
        cpu_cores = cpu_limit if cpu_limit else (psutil.cpu_count(logical=True) or 1)
        cpu_percent = psutil.cpu_percent(interval=0.3)

        mem_limit = _get_container_memory_limit()
        mem_usage = _get_container_memory_usage()

        if mem_limit and mem_usage is not None:
            # cgroup usage includes page cache for recently-read GGUF files,
            # which can stay charged after a model exits. Subtract inactive
            # file cache so the UI tracks the active working set instead.
            inactive_file = min(_get_container_inactive_file_bytes(), mem_usage)
            working_set = max(mem_usage - inactive_file, 0)
            ram_total_mb = round(mem_limit / (1024 * 1024))
            ram_used_mb = round(working_set / (1024 * 1024))
            ram_free_mb = ram_total_mb - ram_used_mb
            ram_percent = round(working_set / mem_limit * 100, 1) if mem_limit > 0 else 0
        else:
            vm = psutil.virtual_memory()
            ram_total_mb = round(vm.total / (1024 * 1024))
            ram_free_mb = round(vm.available / (1024 * 1024))
            ram_used_mb = max(ram_total_mb - ram_free_mb, 0)
            ram_percent = round(ram_used_mb / ram_total_mb * 100, 1) if ram_total_mb > 0 else 0

        return jsonify({
            "cpu_percent": cpu_percent,
            "cpu_cores": round(cpu_cores, 1),
            "ram_total_mb": ram_total_mb,
            "ram_used_mb": ram_used_mb,
            "ram_free_mb": ram_free_mb,
            "ram_percent": ram_percent,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_running_llama_container():
    """Return the first running llama-server container, or None."""
    from core.helpers import list_llama_containers
    try:
        containers = list_llama_containers()
        for c in containers:
            if c.status == "running":
                return c
    except Exception:
        pass
    return None


def _exec_in_container(container, cmd: str) -> str | None:
    """Run a command inside a container and return stdout, or None on failure."""
    try:
        result = container.exec_run(cmd, demux=False)
        if result.exit_code != 0:
            return None
        return result.output.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _query_via_container_exec() -> list[dict] | None:
    """Fallback: query GPUs by exec-ing smi tools inside a running llama-server container."""
    container = _get_running_llama_container()
    if container is None:
        return None

    # Try nvidia-smi
    output = _exec_in_container(
        container,
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu"
        " --format=csv,noheader,nounits",
    )
    if output:
        gpus = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                try:
                    temp = None
                    if len(parts) >= 7 and parts[6] not in ("", "[N/A]", "N/A"):
                        try:
                            temp = int(parts[6])
                        except ValueError:
                            pass
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_pct": int(parts[5]),
                        "temperature_c": temp,
                    })
                except (ValueError, IndexError):
                    continue
        if gpus:
            return gpus

    # Try rocm-smi
    output = _exec_in_container(
        container,
        "rocm-smi --showmeminfo vram --showuse --showproductname --showtemp --csv",
    )
    if output:
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if len(lines) >= 2:
            header = [h.strip().lower() for h in lines[0].split(",")]
            gpus = []
            # Find a temperature column - rocm-smi names it "temperature (sensor edge) (c)"
            # but exposes per-sensor variants too; prefer "edge", fall back to first match.
            temp_keys = [h for h in header if "temperature" in h and "(c)" in h]
            edge_keys = [h for h in temp_keys if "edge" in h]
            temp_key = (edge_keys or temp_keys)[0] if temp_keys else None
            for line in lines[1:]:
                cols = [c.strip() for c in line.split(",")]
                if len(cols) < len(header):
                    continue
                row = dict(zip(header, cols))
                idx = row.get("device", row.get("gpu", str(len(gpus))))
                name = row.get("card series", row.get("card model", row.get("name", "AMD GPU")))
                vram_used = row.get("vram used", row.get("used vram (b)", "0"))
                vram_total = row.get("vram total", row.get("total vram (b)", "0"))
                gpu_use = row.get("gpu use (%)", row.get("gpu busy", "0"))
                try:
                    used = int(vram_used)
                    total = int(vram_total)
                except ValueError:
                    continue
                if total > 1_000_000:
                    used = used // (1024 * 1024)
                    total = total // (1024 * 1024)
                temp = None
                if temp_key:
                    try:
                        temp = int(float(row.get(temp_key, "")))
                    except ValueError:
                        temp = None
                gpus.append({
                    "index": int(idx) if str(idx).isdigit() else len(gpus),
                    "name": name,
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                    "memory_free_mb": total - used,
                    "utilization_pct": int(gpu_use.replace("%", "")) if gpu_use.replace("%", "").isdigit() else 0,
                    "temperature_c": temp,
                })
            if gpus:
                return gpus

    return None


@bp.route("/api/gpu-info")
def api_gpu_info():
    try:
        from core.gpu import query_gpus
        gpus = query_gpus()
        if gpus is not None:
            return jsonify({"gpus": gpus})

        # Fallback: exec into a running llama-server container
        gpus = _query_via_container_exec()
        if gpus is not None:
            return jsonify({"gpus": gpus})

        return jsonify({"gpus": [], "error": "No GPU data available - mount /sys/class/drm:ro (AMD/Intel) or enable NVIDIA toolkit utility capability"}), 200
    except Exception as e:
        return jsonify({"error": str(e), "gpus": []}), 500
