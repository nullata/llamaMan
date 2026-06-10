# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""GPU vendor auto-detection and native VRAM/load query.

Detection order (when GPU_TYPE env var is not set):
  1. NVIDIA  - pynvml (requires NVIDIA Container Toolkit utility capability)
  2. AMD     - /sys/class/drm sysfs (requires /sys/class/drm:ro volume mount)
  3. Intel   - same sysfs path, vendor ID 0x8086
  4. None    - no GPU detected / no monitoring access

GPU_TYPE env var always overrides auto-detection when set.
"""

import glob
import os
import threading

_lock = threading.Lock()
_detected: str | None = "__unset__"  # sentinel so None means "probed, found nothing"


def detect_gpu_vendor() -> str | None:
    """Probe the host for a GPU vendor. Returns 'cuda', 'rocm', 'intel', or None."""
    # 1. NVIDIA via pynvml (available when toolkit utility capability is present)
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            return "cuda"
    except Exception:
        pass

    # 2. AMD / Intel via DRM sysfs (available with /sys/class/drm:ro mount)
    for vendor_file in sorted(glob.glob("/sys/class/drm/card*/device/vendor")):
        try:
            with open(vendor_file) as f:
                vendor_id = f.read().strip().lower()
            if vendor_id == "0x1002":
                return "rocm"
            if vendor_id == "0x8086":
                return "intel"
        except Exception:
            continue

    return None


def get_vendor() -> str | None:
    """Return the effective GPU vendor.

    Checks GPU_TYPE env var first (manual override), then auto-detects.
    Result is cached after first call.
    """
    global _detected

    override = os.environ.get("GPU_TYPE", "").strip().lower()
    if override:
        return override

    if _detected != "__unset__":
        return _detected

    with _lock:
        if _detected == "__unset__":
            _detected = detect_gpu_vendor()

    return _detected


# ---------------------------------------------------------------------------
# Native query functions
# ---------------------------------------------------------------------------

def query_nvidia_pynvml() -> list[dict] | None:
    """Query NVIDIA GPUs via pynvml. Returns None if pynvml is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_pct = util.gpu
            except Exception:
                utilization_pct = 0
            try:
                temperature_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature_c = None
            gpus.append({
                "index": i,
                "name": name,
                "memory_used_mb": mem.used // (1024 * 1024),
                "memory_total_mb": mem.total // (1024 * 1024),
                "memory_free_mb": mem.free // (1024 * 1024),
                "utilization_pct": utilization_pct,
                "temperature_c": temperature_c,
            })
        return gpus if gpus else None
    except Exception:
        return None


def _read_hwmon_temp_c(device_dir: str) -> int | None:
    """Read GPU edge temperature in C from a DRM device's hwmon sensor.

    The kernel exposes temps under /sys/class/drm/card*/device/hwmon/hwmon*/
    with files like `temp1_input` (millidegrees C) and `temp1_label` ("edge",
    "junction", "mem"). We prefer the `edge` sensor because it's the one
    vendor tools display by default."""
    candidates = []
    for hwmon_dir in glob.glob(os.path.join(device_dir, "hwmon", "hwmon*")):
        for input_file in sorted(glob.glob(os.path.join(hwmon_dir, "temp*_input"))):
            label_file = input_file.replace("_input", "_label")
            label = ""
            try:
                with open(label_file) as f:
                    label = f.read().strip().lower()
            except OSError:
                pass
            try:
                with open(input_file) as f:
                    millideg = int(f.read().strip())
            except (OSError, ValueError):
                continue
            candidates.append((label, millideg // 1000))
    if not candidates:
        return None
    for preferred in ("edge", ""):
        for label, temp in candidates:
            if label == preferred:
                return temp
    return candidates[0][1]


def _read_drm_sysfs(vendor_filter: str) -> list[dict] | None:
    """Read GPU info from /sys/class/drm for AMD (0x1002) or Intel (0x8086)."""
    cards = sorted(glob.glob("/sys/class/drm/card*/device/vendor"))
    gpus = []
    for vendor_file in cards:
        try:
            with open(vendor_file) as f:
                vid = f.read().strip().lower()
            if vid != vendor_filter:
                continue
        except Exception:
            continue

        device_dir = os.path.dirname(vendor_file)
        index = len(gpus)

        def _read(filename: str, default="") -> str:
            try:
                with open(os.path.join(device_dir, filename)) as f:
                    return f.read().strip()
            except Exception:
                return default

        # Product name - AMD uses "product_name", Intel uses "device" (PCI ID fallback)
        name = _read("product_name") or _read("label") or (
            "AMD GPU" if vendor_filter == "0x1002" else "Intel GPU"
        )

        try:
            vram_used = int(_read("mem_info_vram_used", "0"))
            vram_total = int(_read("mem_info_vram_total", "0"))
        except ValueError:
            continue

        vram_used_mb = vram_used // (1024 * 1024)
        vram_total_mb = vram_total // (1024 * 1024)

        try:
            utilization_pct = int(_read("gpu_busy_percent", "0"))
        except ValueError:
            utilization_pct = 0

        gpus.append({
            "index": index,
            "name": name,
            "memory_used_mb": vram_used_mb,
            "memory_total_mb": vram_total_mb,
            "memory_free_mb": max(vram_total_mb - vram_used_mb, 0),
            "utilization_pct": utilization_pct,
            "temperature_c": _read_hwmon_temp_c(device_dir),
        })

    return gpus if gpus else None


def query_amd_sysfs() -> list[dict] | None:
    """Query AMD GPUs via /sys/class/drm sysfs. Returns None if unavailable."""
    return _read_drm_sysfs("0x1002")


def query_intel_sysfs() -> list[dict] | None:
    """Query Intel Arc GPUs via /sys/class/drm sysfs. Returns None if unavailable."""
    return _read_drm_sysfs("0x8086")


def query_gpus() -> list[dict] | None:
    """Query GPUs natively using the detected vendor. Returns None if unavailable."""
    vendor = get_vendor()
    if vendor == "cuda":
        return query_nvidia_pynvml()
    if vendor == "rocm":
        return query_amd_sysfs()
    if vendor == "intel":
        return query_intel_sysfs()
    return None
