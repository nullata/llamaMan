# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import sys
import logging
from pathlib import Path

VERSION = (Path(__file__).parent / "VERSION").read_text().strip()

# Every llamaman deployment needs a stable identity. It is the partition key for
# instance/download rows on the (possibly shared) storage backend, the namespace
# for per-node settings, and the registry key when clustering is enabled. It is
# REQUIRED even for single-node installs so a later transition to clustering
# does not silently change the row scope. Operators pick the value (any string -
# "srv1", a uuid, whatever); changing it after the first boot orphans this
# node's existing rows, so the rule is "pick once, keep forever".
LLAMAMAN_NODE_NAME = os.environ.get("LLAMAMAN_NODE_NAME", "").strip()
if not LLAMAMAN_NODE_NAME:
    _banner = (
        "\n"
        "============================================================\n"
        "  LLAMAMAN_NODE_NAME is required and was not set.\n"
        "------------------------------------------------------------\n"
        "  Set it to any unique string for this deployment, e.g.:\n"
        "      LLAMAMAN_NODE_NAME=srv1\n"
        "      LLAMAMAN_NODE_NAME=1234\n"
        "      LLAMAMAN_NODE_NAME=my-llamaman-host\n"
        "\n"
        "  Every instance, download and per-node setting in storage\n"
        "  is scoped to this name. Pick it ONCE and keep it - changing\n"
        "  it later orphans this node's existing state.\n"
        "\n"
        "  Add it to your docker-compose.yml (or environment) and\n"
        "  restart.\n"
        "============================================================\n"
    )
    sys.stderr.write(_banner)
    sys.stderr.flush()
    sys.exit(78)  # EX_CONFIG

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
LOGS_DIR = os.environ.get("LOGS_DIR", "/tmp/llama-logs")
PORT_RANGE_START = int(os.environ.get("PORT_RANGE_START", 8000))
PORT_RANGE_END = int(os.environ.get("PORT_RANGE_END", 8020))
INTERNAL_PORT_RANGE_START = int(os.environ.get("INTERNAL_PORT_RANGE_START", 9000))
INTERNAL_PORT_RANGE_END = int(os.environ.get("INTERNAL_PORT_RANGE_END", 9020))

PRESETS_FILE = os.path.join(DATA_DIR, "presets.json")
LLAMAMAN_MAX_MODELS = int(os.environ.get("LLAMAMAN_MAX_MODELS", 0))
LLAMAMAN_PROXY_PORT = int(os.environ.get("LLAMAMAN_PROXY_PORT", 42069))
LLAMAMAN_IDLE_TIMEOUT = int(os.environ.get("LLAMAMAN_IDLE_TIMEOUT", 0))  # minutes, 0=disabled
HEALTH_CHECK_TIMEOUT = int(os.environ.get("HEALTH_CHECK_TIMEOUT", 3))
MODEL_LOAD_TIMEOUT = int(os.environ.get("MODEL_LOAD_TIMEOUT", 300))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 300))

STATE_FILE = os.path.join(DATA_DIR, "state.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", os.path.join(DATA_DIR, "request_log"))
SECRET_KEY = os.environ.get("SECRET_KEY", "")

# Clustering - off by default; single-node installs are entirely unaffected.
# When enabled, several llamaman deployments form one logical cluster sharing a
# coordination store (the storage backend) and trusting one shared secret. The
# node's identity comes from LLAMAMAN_NODE_NAME above (required for all installs).
# CLUSTER_SECRET is the bearer token used for all node-to-node HTTP.
# CLUSTER_ADVERTISE_URL is how peers reach THIS node (e.g. http://srv1:5000).
CLUSTER_ENABLED = os.environ.get("CLUSTER_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
CLUSTER_SECRET = os.environ.get("CLUSTER_SECRET", "").strip()
CLUSTER_ADVERTISE_URL = os.environ.get("CLUSTER_ADVERTISE_URL", "").strip().rstrip("/")

# Docker-in-Docker settings
# Fixed port llama-server listens on inside every spawned container.
LLAMA_CONTAINER_PORT = 8080
LLAMA_NETWORK = os.environ.get("LLAMA_NETWORK", "llamaman-net")
LLAMA_CONTAINER_PREFIX = os.environ.get("LLAMA_CONTAINER_PREFIX", "llamaman-")


def _detect_in_docker() -> bool:
    """Whether llamaman itself runs inside a container.

    Containerized, it shares LLAMA_NETWORK with the sibling llama-server
    containers and reaches them by name on the in-container port. Bare-metal
    (e.g. running under WSL), it must reach them via localhost on the host-
    published port instead.

    Auto-detected; set LLAMAMAN_IN_DOCKER=true/false to force either mode.
    Detection order: explicit override, runtime marker files (Docker's
    /.dockerenv, Podman's /run/.containerenv), then cgroup inspection which
    catches many containerd / Kubernetes / LXC setups.
    """
    override = os.environ.get("LLAMAMAN_IN_DOCKER", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    for cgroup_path in ("/proc/self/cgroup", "/proc/1/cgroup"):
        try:
            with open(cgroup_path, "r", encoding="utf-8") as f:
                data = f.read()
        except OSError:
            continue
        if any(m in data for m in ("docker", "containerd", "kubepods", "/lxc/", "libpod")):
            return True
    return False


IN_DOCKER = _detect_in_docker()
# Host llamaman uses to reach bare-metal-published llama-server ports.
LLAMA_HOST_ADDR = os.environ.get("LLAMA_HOST_ADDR", "localhost").strip() or "localhost"
# GPU_TYPE: set to override auto-detection ("cuda", "rocm", "intel").
# Leave unset to let llamaman probe the host automatically.
GPU_TYPE = os.environ.get("GPU_TYPE", "").strip().lower()
# Comma-separated GPU indices visible to all llama-server containers, e.g. "0,1,3".
# Empty (default) means all GPUs. Per-instance gpu_devices overrides this when set.
LLAMA_GPU_DEVICES = os.environ.get("LLAMA_GPU_DEVICES", "").strip()

# LLAMA_IMAGE: which llama.cpp server image to use for spawned containers.
# If not set, auto-selected based on detected GPU vendor.
_LLAMA_IMAGE_ENV = os.environ.get("LLAMA_IMAGE", "").strip()
_VENDOR_IMAGE_DEFAULTS = {
    "cuda": "ghcr.io/ggml-org/llama.cpp:server-cuda",
    "rocm": "ghcr.io/ggml-org/llama.cpp:server-rocm",
    "intel": "ghcr.io/ggml-org/llama.cpp:server-sycl",
}


def _resolve_llama_image() -> str:
    if _LLAMA_IMAGE_ENV:
        return _LLAMA_IMAGE_ENV
    from core.gpu import get_vendor
    vendor = get_vendor()
    return _VENDOR_IMAGE_DEFAULTS.get(vendor or "", "ghcr.io/ggml-org/llama.cpp:server")


# Resolved once at startup - all modules import this name directly.
LLAMA_IMAGE = _resolve_llama_image()

# When llamaman runs inside Docker, the Docker daemon (on the host) needs the
# HOST-side paths to bind-mount into sibling llama-server containers.
# Set these to the real host paths that are mounted as MODELS_DIR / LOGS_DIR
# inside the llamaman container.  If llamaman runs bare-metal, leave unset.
HOST_MODELS_DIR = os.environ.get("HOST_MODELS_DIR", MODELS_DIR)
HOST_LOGS_DIR = os.environ.get("HOST_LOGS_DIR", LOGS_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llamaman")

# Log detected GPU vendor and resolved image at startup.
_detected_vendor = GPU_TYPE or __import__("core.gpu", fromlist=["get_vendor"]).get_vendor()
logger.info(
    "GPU vendor: %s | llama image: %s",
    _detected_vendor or "none (CPU)",
    LLAMA_IMAGE,
)
