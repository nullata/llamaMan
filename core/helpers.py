# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path


def model_name_from_path(path: str) -> str:
    """Derive a lowercase model name from a file path (stem only)."""
    return Path(path).stem.lower()


def normalize_flash_attn(value) -> str:
    """Coerce any accepted flash_attn input to llama.cpp's tri-state string.

    llama.cpp switched --flash-attn from a bare boolean flag to a valued flag
    ('on'|'off'|'auto', default 'auto'). We accept:
      - the new strings ('on'/'off'/'auto'), case- and whitespace-tolerant
      - legacy bools from configs/presets saved before the tri-state rollout,
        mapped True -> 'on' and False -> 'off' so behavior is preserved (the
        old code emitted a bare --flash-attn only when True, which under the
        old semantics meant "force on"; False meant "force off")
      - anything else (None, empty, garbage) -> 'auto', which mirrors the
        server's own default when the flag is omitted.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("on", "off", "auto"):
            return s
    return "auto"


_REASONING_FORMAT_VALUES = ("auto", "none", "deepseek", "deepseek-legacy")


def normalize_reasoning_format(value) -> str:
    """Coerce any --reasoning-format input to the exact string llama-server
    accepts (none|auto|deepseek|deepseek-legacy).

    Anything unknown / missing / non-string falls back to 'auto' - that's
    llama.cpp's own default when the flag is absent, so a corrupt or legacy
    preset value cannot make the server refuse to start. Case- and
    whitespace-tolerant, same contract as normalize_flash_attn.
    """
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _REASONING_FORMAT_VALUES:
            return s
    return "auto"


def request_local_worker(url, *, method="POST", json=None, data=None,
                         headers=None, stream=False):
    """Send an HTTP request to a local llama-server with bounded connect
    retries. Returns the requests.Response on success; raises the last
    connection error after 3 failed attempts.

    Read timeouts are NOT retried. A ReadTimeout means llama-server accepted
    the request and is generating - it keeps grinding on the original even
    after we close the socket, so a retry just enqueues a second copy of the
    same prompt while the first burns compute uselessly in the background.
    ConnectionError / ConnectionRefused are the only failures that mean
    "server isn't accepting yet, try again" - which is the brief window
    right after a model starts up.
    """
    import requests as _requests
    from config import REQUEST_TIMEOUT
    last_err = None
    for _ in range(3):
        try:
            return _requests.request(
                method=method, url=url,
                json=json, data=data, headers=headers,
                stream=stream, timeout=REQUEST_TIMEOUT,
            )
        except (_requests.ConnectionError, ConnectionRefusedError) as e:
            last_err = e
            time.sleep(2)
    raise last_err


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def public_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def resolve_llama_endpoint(container_name: str, published_port: int) -> tuple[str, int]:
    """Return the (host, port) llamaman uses to reach a spawned llama-server.

    Containerized llamaman shares the Docker network with the sibling
    container and reaches it by name on the fixed in-container port. Bare-metal
    llamaman reaches it via the host where the container's port is published.
    """
    from config import IN_DOCKER, LLAMA_CONTAINER_PORT, LLAMA_HOST_ADDR
    if IN_DOCKER:
        return container_name, LLAMA_CONTAINER_PORT
    return LLAMA_HOST_ADDR, published_port


def cleanup_download_dir(dest_path: str) -> None:
    """Delete a partial/failed download directory, guarded to stay inside MODELS_DIR."""
    from config import MODELS_DIR
    try:
        real_dest = os.path.realpath(dest_path)
        real_models = os.path.realpath(MODELS_DIR)
        if not real_dest.startswith(real_models + os.sep) and real_dest != real_models:
            return  # never delete outside models dir
        if os.path.isdir(real_dest):
            shutil.rmtree(real_dest)
        elif os.path.isfile(real_dest):
            os.remove(real_dest)
    except Exception:
        pass


def build_llama_cmd(model_path: str, port: int, config: dict) -> list[str]:
    """Build the argument list passed to the llama-server container.

    The container already has llama-server as its entrypoint, so we only
    supply the flags (no binary path prefix).
    """
    cmd = [
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n-gpu-layers", str(config.get("n_gpu_layers", -1)),
        "--ctx-size", str(config.get("ctx_size", 4096)),
    ]
    if config.get("threads"):
        cmd += ["--threads", str(int(config["threads"]))]
    # --threads-batch controls the CPU threads used during batch/prompt
    # processing. Defaults to the value of --threads when absent, so we only
    # emit when the user set it explicitly - leaving blank falls back to
    # llama-server's own default (== threads) rather than us second-guessing.
    if config.get("threads_batch"):
        cmd += ["--threads-batch", str(int(config["threads_batch"]))]
    if config.get("parallel"):
        cmd += ["--parallel", str(int(config["parallel"]))]
    if config.get("embedding_model"):
        cmd += ["--embeddings"]
    # Multi-GPU placement. All three llama.cpp modes are exposed and emitted
    # literally: none (single GPU only, ignores --tensor-split), layer (splits
    # whole layers - llama.cpp's own default when no flag is passed), row
    # (splits tensor rows, needs fast interconnect like NVLink). tensor_split
    # is passed through as-is because llama.cpp normalizes it internally, so
    # "24,16" == "3,2" == "0.6,0.4". An empty split_mode (from a preset saved
    # before this feature existed) is treated as "layer" so behavior matches
    # llama.cpp's pre-flag default.
    split_mode = (config.get("split_mode") or "").strip().lower() or "layer"
    if split_mode in ("none", "layer", "row"):
        cmd += ["--split-mode", split_mode]
    tensor_split = (config.get("tensor_split") or "").strip()
    if tensor_split:
        cmd += ["--tensor-split", tensor_split]
    # Flash Attention + KV cache quantization. As of the tri-state llama.cpp
    # rollout --flash-attn takes a value ('on'|'off'|'auto', default 'auto'),
    # so a bare --flash-attn now swallows the next arg instead of enabling
    # anything. Emit the value explicitly for 'on'/'off' and omit the flag
    # entirely for 'auto' (which is llama.cpp's own default when the flag is
    # absent) - that keeps the command line quiet for the common case.
    #
    # Cache types map to --cache-type-k / --cache-type-v. Only emit cache-type
    # flags when the user picked a non-default value (llama.cpp's own default
    # is f16 for both), and only from a whitelist of the types llama-server
    # actually accepts - a corrupt/hand-crafted value would otherwise make the
    # server refuse to start with an opaque error. NB: llama-server itself
    # enforces "quantized V cache requires --flash-attn" and will throw on
    # startup if that combination is passed; the UI grey-out prevents it, but
    # we deliberately don't second-guess here so API callers see the real
    # error message rather than a silently-dropped flag.
    flash_attn = normalize_flash_attn(config.get("flash_attn"))
    if flash_attn in ("on", "off"):
        cmd += ["--flash-attn", flash_attn]
    # Reasoning format. Maps to llama-server --reasoning-format
    # (none|auto|deepseek|deepseek-legacy, default auto). 'auto' matches
    # llama.cpp's own default when the flag is absent, so omit it for the
    # common case to keep the command line quiet. Only emit values from the
    # whitelist llama-server actually accepts - a hand-crafted / corrupt
    # value would make the server refuse to start with an opaque error.
    _ALLOWED_REASONING_FORMATS = {"none", "deepseek", "deepseek-legacy"}
    reasoning_format = (config.get("reasoning_format") or "").strip().lower()
    if reasoning_format in _ALLOWED_REASONING_FORMATS:
        cmd += ["--reasoning-format", reasoning_format]
    # DRY (Don't Repeat Yourself) sampler. Only emit when the user's toggle is
    # on AND multiplier > 0 - llama.cpp treats multiplier=0 as disabled, so
    # emitting the flags with a zero multiplier would be noisy no-ops.
    # Companion flags (--dry-base / --dry-allowed-length / --dry-penalty-last-n)
    # only ride along when the main flag is emitted; skipping them individually
    # falls back to llama.cpp's own defaults so we don't hard-code numbers that
    # may drift across versions. See core/dry_sampling.py for the value contract.
    from core.dry_sampling import dry_enabled
    if dry_enabled(config):
        try:
            cmd += ["--dry-multiplier", f"{float(config['dry_multiplier']):g}"]
        except (TypeError, ValueError, KeyError):
            pass
        base = config.get("dry_base")
        if base not in (None, ""):
            try:
                cmd += ["--dry-base", f"{float(base):g}"]
            except (TypeError, ValueError):
                pass
        allowed_length = config.get("dry_allowed_length")
        if allowed_length not in (None, ""):
            try:
                cmd += ["--dry-allowed-length", str(int(allowed_length))]
            except (TypeError, ValueError):
                pass
        penalty_last_n = config.get("dry_penalty_last_n")
        if penalty_last_n not in (None, ""):
            try:
                v = int(penalty_last_n)
                # Guard against negative slipping through from a corrupt
                # preset - llama.cpp's CLI throws on those, so the container
                # would die at startup with an opaque error otherwise.
                if v >= 0:
                    cmd += ["--dry-penalty-last-n", str(v)]
            except (TypeError, ValueError):
                pass
    _ALLOWED_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1",
                            "iq4_nl", "q5_0", "q5_1"}
    cache_type_k = (config.get("cache_type_k") or "").strip().lower()
    if cache_type_k and cache_type_k != "f16" and cache_type_k in _ALLOWED_CACHE_TYPES:
        cmd += ["--cache-type-k", cache_type_k]
    cache_type_v = (config.get("cache_type_v") or "").strip().lower()
    if cache_type_v and cache_type_v != "f16" and cache_type_v in _ALLOWED_CACHE_TYPES:
        cmd += ["--cache-type-v", cache_type_v]
    # When this instance opts into a queue-group alias, also tell llama-server
    # to advertise itself under that name (--alias). Two effects: clients
    # hitting THIS instance's port directly see the group name in /v1/models
    # and chat responses; and the alias plumbing stays honest end-to-end
    # rather than being a cluster-router-only fiction.
    alias = (config.get("share_queue_group") or "").strip()
    if alias:
        cmd += ["--alias", alias]
    if config.get("spec_enabled"):
        from core.spec_decoding import DEFAULT_SPEC_TYPE
        spec_type = (config.get("spec_type") or DEFAULT_SPEC_TYPE).strip() or DEFAULT_SPEC_TYPE
        draft_model = (config.get("spec_draft_model") or "").strip()
        # All spec types draft from -md when one is set. draft-mtp is the only
        # one where it's optional (falls back to the main model's MTP heads
        # when absent); draft-simple / draft-dflash / draft-dspark / draft-eagle3
        # all need a drafter checkpoint of the appropriate format.
        if draft_model:
            cmd += ["--model-draft", draft_model]
        cmd += ["--spec-type", spec_type]
        try:
            n_max = int(config.get("spec_draft_n_max") or 0)
        except (TypeError, ValueError):
            n_max = 0
        if n_max > 0:
            cmd += ["--spec-draft-n-max", str(n_max)]
        # Advanced spec-decoding knobs. Emit only when a value is set so an
        # empty field falls through to llama-server's own defaults (which
        # drift across versions - hard-coding them here would silently
        # override future improvements). parse_spec_config validates ranges
        # at the boundary, so we trust the config values here.
        n_min = config.get("spec_draft_n_min")
        if n_min not in (None, ""):
            try:
                cmd += ["--spec-draft-n-min", str(int(n_min))]
            except (TypeError, ValueError):
                pass
        p_split = config.get("spec_draft_p_split")
        if p_split not in (None, ""):
            try:
                cmd += ["--spec-draft-p-split", str(float(p_split))]
            except (TypeError, ValueError):
                pass
        p_min = config.get("spec_draft_p_min")
        if p_min not in (None, ""):
            try:
                cmd += ["--spec-draft-p-min", str(float(p_min))]
            except (TypeError, ValueError):
                pass
    if config.get("mmproj_enabled"):
        mmproj_path = (config.get("mmproj_path") or "").strip()
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
    if config.get("extra_args"):
        cmd += shlex.split(config["extra_args"])
    return cmd


def kill_instance_process(inst: dict):
    """Stop a subprocess-based instance (used for downloads, not llama-server containers)."""
    proc = inst.get("_process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif proc is None and inst.get("pid", 0) > 0:
        _kill_pid(inst["pid"])
    fh = inst.get("_log_fh")
    if fh:
        try:
            fh.close()
        except Exception:
            pass
    inst["_process"] = None
    inst["_log_fh"] = None


def _kill_pid(pid: int) -> None:
    """Best-effort kill of a process by PID. SIGTERM first, SIGKILL after 10s."""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def read_log_file(log_path: str, tail: int = 100) -> list[str]:
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        return lines[-tail:]
    except FileNotFoundError:
        return []


def stream_log_file(log_file):
    """Generator that tails a log file and yields SSE events."""
    try:
        with open(log_file, "r", errors="replace") as f:
            content = f.read()
            if content:
                yield f"data: {json.dumps({'lines': content.splitlines(keepends=True)})}\n\n"
            while True:
                line = f.readline()
                if line:
                    yield f"data: {json.dumps({'lines': [line]})}\n\n"
                else:
                    time.sleep(0.5)
                    yield ": keepalive\n\n"
    except GeneratorExit:
        return
    except Exception:
        return


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

_docker_client = None
_docker_client_lock = __import__("threading").Lock()


def get_docker_client():
    """Return a singleton docker.DockerClient connected via the local socket."""
    global _docker_client
    if _docker_client is not None:
        return _docker_client
    with _docker_client_lock:
        if _docker_client is None:
            import docker
            _docker_client = docker.from_env()
    return _docker_client


def stop_container(container_id: str, timeout: int = 10) -> None:
    """Stop and remove a container by ID. Best-effort; ignores not-found errors.

    Phase-timed (LLAMAMAN_PERF_LOG=1): `get` / `stop` / `remove` are three
    separate Docker SDK round-trips on the caller's thread; `stop` waits out
    the SIGTERM grace period, so it dominates whenever the daemon or the
    process inside is slow to die.
    """
    import docker

    from core.perf import phase
    with phase("docker.stop_container", cid=container_id[:12]):
        try:
            with phase("docker.get", cid=container_id[:12]):
                c = get_docker_client().containers.get(container_id)
        except docker.errors.NotFound:
            return
        except Exception:
            return
        try:
            with phase("docker.stop", cid=container_id[:12], grace=timeout):
                c.stop(timeout=timeout)
        except Exception:
            pass
        try:
            with phase("docker.remove", cid=container_id[:12]):
                c.remove(force=True)
        except docker.errors.NotFound:
            pass
        except Exception:
            pass


def is_container_running(container_id: str) -> bool:
    """Return True if the container exists and has status 'running'."""
    import docker
    try:
        c = get_docker_client().containers.get(container_id)
        c.reload()
        return c.status == "running"
    except docker.errors.NotFound:
        return False
    except Exception:
        return False


def list_llama_containers(all: bool = False) -> list:
    """Return llamaman-labeled containers. Running only unless all=True."""
    from config import LLAMA_CONTAINER_PREFIX
    try:
        return get_docker_client().containers.list(
            all=all,
            filters={"name": LLAMA_CONTAINER_PREFIX, "label": "llamaman.instance_id"},
        )
    except Exception:
        return []


def ensure_docker_network() -> None:
    """Create the llamaman Docker network if it doesn't already exist."""
    import docker
    from config import LLAMA_NETWORK
    client = get_docker_client()
    try:
        client.networks.get(LLAMA_NETWORK)
    except docker.errors.NotFound:
        client.networks.create(LLAMA_NETWORK, driver="bridge")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------

def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the TCP port can be bound locally."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_available_port(
    exclude: set[int] | None = None,
    range_start: int | None = None,
    range_end: int | None = None,
) -> int | None:
    from core.state import instances, instances_lock
    from proxy import idle_proxies, idle_proxies_lock
    from config import PORT_RANGE_START, PORT_RANGE_END

    exclude = exclude or set()
    range_start = PORT_RANGE_START if range_start is None else range_start
    range_end = PORT_RANGE_END if range_end is None else range_end
    with instances_lock:
        used = set()
        for i in instances.values():
            if i["status"] not in ("stopped",):
                used.add(i["port"])
                if i.get("_internal_port"):
                    used.add(i["_internal_port"])
    with idle_proxies_lock:
        for p in idle_proxies.values():
            used.add(p["internal_port"])
    used |= exclude
    for p in range(range_start, range_end + 1):
        if p not in used and is_port_available(p):
            return p
    return None
