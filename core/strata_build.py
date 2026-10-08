# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata image from its repository: Settings -> Docker Images -> Manage
Strata images.

Strata publishes no image, so this node downloads the repository (a GitHub
tarball of STRATA_REPO at STRATA_REPO_REF, no git needed) into STRATA_SRC_DIR
and builds STRATA_BUILD_IMAGE from it through the Docker socket. The image lands in
the host's Docker, where Strata instances are launched from.

The build goes through the Engine API with BuildKit (``/build?version=2``):
Strata's Dockerfile uses RUN heredocs, which the classic builder (all that
docker-py's ``images.build`` speaks) rejects.

Only the UI button downloads the repository the first time. The periodic
auto-update (check_and_update_if_needed, from the monitoring poller) re-pulls
the Strata images pulled by name and, on an existing download only, fetches
the newer commit and rebuilds; it never downloads the repository itself.
"""

import base64
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time

from config import logger

_lock = threading.Lock()
_state: dict = {
    "status": "idle",   # idle | fetching | building | done | error
    "message": "",
    "started_at": None,
    "finished_at": None,
    "trigger": None,    # manual | auto
}

# BuildKit progress arrives as base64 protobuf in "moby.buildkit.trace" aux
# messages; the step names ("[3/6] RUN ...") are readable inside them.
_STEP_RE = re.compile(rb"\[ *\d+/\d+\] [\x20-\x7e]{1,160}")


def _source_meta_path() -> str:
    from config import STRATA_SRC_DIR
    return STRATA_SRC_DIR.rstrip("/") + ".source.json"


def source_info() -> dict:
    """Whether the repository is downloaded here, and which commit."""
    from config import STRATA_REPO, STRATA_REPO_REF, STRATA_SRC_DIR
    info = {"repo": STRATA_REPO, "ref": STRATA_REPO_REF, "dir": STRATA_SRC_DIR,
            "url": f"https://github.com/{STRATA_REPO}",
            "present": os.path.isfile(os.path.join(STRATA_SRC_DIR, "Dockerfile"))}
    try:
        with open(_source_meta_path()) as f:
            meta = json.load(f)
        info["sha"] = meta.get("sha")
        info["fetched_at"] = meta.get("fetched_at")
    except (OSError, ValueError):
        pass
    return info


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def remote_sha() -> str:
    import requests
    from config import STRATA_REPO, STRATA_REPO_REF
    r = requests.get(f"https://api.github.com/repos/{STRATA_REPO}/commits/{STRATA_REPO_REF}",
                     headers={"Accept": "application/vnd.github.sha"}, timeout=30)
    r.raise_for_status()
    sha = r.text.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"unexpected commit id from GitHub: {sha[:60]!r}")
    return sha


def download_source(sha: str) -> None:
    """Replace STRATA_SRC_DIR with the repository at `sha` (a new copy is
    unpacked next to it and swapped in, so a failed download keeps the old)."""
    import requests
    from config import STRATA_REPO, STRATA_SRC_DIR
    parent = os.path.dirname(STRATA_SRC_DIR.rstrip("/"))
    os.makedirs(parent, exist_ok=True)
    new_dir = tempfile.mkdtemp(prefix=".strata-new-", dir=parent)
    old_dir = None
    try:
        with requests.get(f"https://codeload.github.com/{STRATA_REPO}/tar.gz/{sha}",
                          stream=True, timeout=60) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for member in tf:
                    # GitHub tarballs hold everything under one "<repo>-<sha>/" folder.
                    parts = member.name.split("/", 1)
                    if len(parts) < 2 or not parts[1]:
                        continue
                    member.name = parts[1]
                    tf.extract(member, new_dir, filter="data")
        if not os.path.isfile(os.path.join(new_dir, "Dockerfile")):
            raise RuntimeError("the download has no Dockerfile")
        if os.path.exists(STRATA_SRC_DIR):
            old_dir = tempfile.mkdtemp(prefix=".strata-old-", dir=parent)
            os.rename(STRATA_SRC_DIR, os.path.join(old_dir, "src"))
        os.rename(new_dir, STRATA_SRC_DIR)
        new_dir = None
        with open(_source_meta_path(), "w") as f:
            json.dump({"repo": STRATA_REPO, "sha": sha, "fetched_at": time.time()}, f)
    finally:
        for d in (new_dir, old_dir):
            if d:
                shutil.rmtree(d, ignore_errors=True)


# The Dockerfile llamaman uploads in place of the repository's (see
# _build_context); kept out of the image by the .dockerignore it rewrites.
CONTEXT_DOCKERFILE = ".llamaman.Dockerfile"
_SYNTAX_RE = re.compile(r"^#\s*syntax\s*=.*$", re.IGNORECASE)


def _strip_syntax_directive(text: str) -> str:
    """Drop the "# syntax=docker/dockerfile:1" parser directive. Fetching that
    frontend image needs a client session (registry auth), which a plain
    Engine API build has not got ("no active sessions"); the daemon's built-in
    Dockerfile frontend handles the heredocs on its own."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            break               # directives only appear before the first instruction
        if _SYNTAX_RE.match(stripped):
            del lines[i]
            break
    return "".join(lines)


def _build_context(path: str):
    from docker import utils
    exclude = None
    ignore = os.path.join(path, ".dockerignore")
    if os.path.exists(ignore):
        with open(ignore) as f:
            exclude = [ln.strip() for ln in f.read().splitlines()
                       if ln.strip() and not ln.strip().startswith("#")]
    with open(os.path.join(path, "Dockerfile")) as f:
        dockerfile = _strip_syntax_directive(f.read())
    # Gzipped: BuildKit decides "archive or a bare Dockerfile" from the
    # upload's first 1 KB. A plain tar whose first entry has a PAX header puts
    # the file header past that, so the whole tar was parsed as the Dockerfile.
    return utils.tar(path, exclude=exclude, gzip=True,
                     dockerfile=(CONTEXT_DOCKERFILE, dockerfile))


_FROM_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", re.IGNORECASE | re.MULTILINE)


def base_images(dockerfile: str) -> list[str]:
    """The registry images the Dockerfile's FROM lines name (not "scratch",
    earlier stages or ARG-templated names)."""
    stages, out = set(), []
    for image, alias in _FROM_RE.findall(dockerfile):
        if image.lower() != "scratch" and image.lower() not in stages and "$" not in image \
                and image not in out:
            out.append(image)
        if alias:
            stages.add(alias.lower())
    return out


def pull_base_images(path: str) -> None:
    """Pull the FROM images with the plain pull API first. BuildKit resolving
    a registry image itself needs a client session for auth, which only the
    docker CLI sets up ("no active sessions"); an image already on the host is
    used as is."""
    from core.helpers import get_docker_client
    with open(os.path.join(path, "Dockerfile")) as f:
        images = base_images(f.read())
    api = get_docker_client().api
    for image in images:
        repo, _, tag = image.rpartition(":") if ":" in image.rsplit("/", 1)[-1] else (image, "", "latest")
        _set(message=f"Pulling {image}")
        for line in api.pull(repo, tag=tag, stream=True, decode=True):
            if line.get("error"):
                raise RuntimeError(f"pulling {image}: {line['error']}")
            detail = line.get("progressDetail") or {}
            if detail.get("total"):
                pct = round(detail.get("current", 0) / detail["total"] * 100)
                _set(message=f"Pulling {image}: {line.get('status', '')} {line.get('id', '')} {pct}%")


def cuda_architectures() -> str:
    """This node's NVIDIA GPU generations as CUDA_ARCHITECTURES ("86;89"),
    so the build compiles for them only. Empty when they can't be read: the
    Dockerfile's default set (every supported generation, much slower)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        archs = set()
        for i in range(pynvml.nvmlDeviceGetCount()):
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(
                pynvml.nvmlDeviceGetHandleByIndex(i))
            if major * 10 + minor >= 75:          # Strata's CMakeLists refuses older
                archs.add(major * 10 + minor)
        return ";".join(str(a) for a in sorted(archs))
    except Exception:
        return ""


def build_image(archs: str = "") -> str:
    """Build STRATA_BUILD_IMAGE from STRATA_SRC_DIR with BuildKit, for the CUDA
    architectures `archs` (empty: the Dockerfile's default). Returns the image id."""
    from config import STRATA_BUILD_IMAGE, STRATA_SRC_DIR
    from core.helpers import get_docker_client

    api = get_docker_client().api
    params = {"t": STRATA_BUILD_IMAGE, "version": "2", "rm": "1", "forcerm": "1",
              "dockerfile": CONTEXT_DOCKERFILE}
    if archs:
        params["buildargs"] = json.dumps({"CUDA_ARCHITECTURES": archs})
    image_id = None
    pull_base_images(STRATA_SRC_DIR)
    context = _build_context(STRATA_SRC_DIR)
    try:
        resp = api._post(api._url("/build"), data=context, params=params, stream=True,
                         headers={"Content-Type": "application/x-tar"}, timeout=None)
        api._raise_for_status(resp)
        for chunk in api._stream_helper(resp, decode=True):
            if chunk.get("error") or chunk.get("errorDetail"):
                err = chunk.get("error") or (chunk.get("errorDetail") or {}).get("message")
                raise RuntimeError(err or "build failed")
            aux_id = chunk.get("id")
            aux = chunk.get("aux")
            if aux_id == "moby.image.id" and isinstance(aux, dict):
                image_id = aux.get("ID") or image_id
            elif aux_id == "moby.buildkit.trace" and isinstance(aux, str):
                try:
                    steps = _STEP_RE.findall(base64.b64decode(aux))
                except ValueError:
                    steps = []
                if steps:
                    _set(message=steps[-1].decode("ascii", "replace").strip())
            elif chunk.get("stream"):
                line = chunk["stream"].strip()
                if line:
                    _set(message=line[:200])
    finally:
        context.close()
    if not image_id:
        raise RuntimeError("the build finished without an image")
    return image_id


def _record_build(sha: str | None, image_id: str, archs: str) -> None:
    from api.images import _read_docker_images, _write_docker_images
    di = _read_docker_images()
    rec = dict(di.get("strata") or {})
    rec.update(built_sha=sha, built_at=time.time(), image_id=image_id, built_archs=archs)
    di["strata"] = rec
    _write_docker_images(di)


def _image_present() -> bool:
    from api.images import _get_image_local_info
    from config import STRATA_BUILD_IMAGE
    return bool(_get_image_local_info(STRATA_BUILD_IMAGE).get("present"))


def _run(trigger: str) -> None:
    from config import STRATA_BUILD_IMAGE
    try:
        src = source_info()
        if trigger == "auto" and not src["present"]:
            _set(status="idle", message="", finished_at=time.time())
            return
        sha = src.get("sha")
        try:
            latest = remote_sha()
        except Exception as e:
            if not src["present"]:
                raise RuntimeError(f"could not reach GitHub: {e}")
            # Offline: a manual build still works from the copy already here.
            logger.warning("Strata: could not check %s for updates: %s", src["repo"], e)
            if trigger == "auto":
                raise RuntimeError(f"could not check for updates: {e}")
            latest = sha
        fetched = False
        if latest and (latest != sha or not src["present"]):
            _set(status="fetching", message=f"Downloading {src['repo']} @ {latest[:7]}")
            download_source(latest)
            sha, fetched = latest, True
        if trigger == "auto" and not fetched and _image_present():
            from api.images import _read_docker_images
            if (_read_docker_images().get("strata") or {}).get("built_sha") == sha:
                _set(status="done", message="Up to date", finished_at=time.time())
                return
        _set(status="building", message=f"Building {STRATA_BUILD_IMAGE}")
        logger.info("Strata: building %s from %s (%s)", STRATA_BUILD_IMAGE, src["repo"], (sha or "?")[:7])
        archs = cuda_architectures()
        image_id = build_image(archs)
        _record_build(sha, image_id, archs)
        # Strata shows in the launch form once its image exists: re-read now,
        # not on the poller's next pass.
        try:
            from core.engines.strata import refresh_catalogue_from_image
            refresh_catalogue_from_image()
        except Exception as e:
            logger.warning("Strata: could not read the new image's model list: %s", e)
        _set(status="done", message=f"Built {STRATA_BUILD_IMAGE}", finished_at=time.time())
        logger.info("Strata: built %s (%s)", STRATA_BUILD_IMAGE, image_id[:19])
    except Exception as e:
        _set(status="error", message=str(e), finished_at=time.time())
        logger.warning("Strata image update failed: %s", e)


def start(trigger: str = "manual") -> bool:
    """Download the latest source (if newer) and build, in the background.
    False when a run is already going."""
    with _lock:
        if _state["status"] in ("fetching", "building"):
            return False
        _state.update(status="fetching", message="Checking for updates",
                      started_at=time.time(), finished_at=None, trigger=trigger)
    threading.Thread(target=_run, args=(trigger,), daemon=True,
                     name="strata-build").start()
    return True


def check_and_update_if_needed() -> bool:
    """Monitoring poller: when Strata's auto-update is on and its interval
    has passed, re-pull every Strata image pulled on this node (a newer
    version replaces it) and, when the repository was downloaded, fetch its
    newer commit and rebuild. True if anything started."""
    from config import STRATA_ENABLED
    if not STRATA_ENABLED:
        return False
    from api.images import _read_docker_images, _trigger_pulls, _write_docker_images, tracked_images
    di = _read_docker_images()
    rec = dict(di.get("strata") or {})
    if not rec.get("auto_update_enabled"):
        return False
    interval = rec.get("auto_update_interval_hours", 24)
    if time.time() - (rec.get("last_auto_check_at") or 0) < interval * 3600:
        return False
    pulled = tracked_images("strata")
    started = bool(pulled) and _trigger_pulls(pulled, "strata")
    if source_info()["present"]:
        started = start("auto") or started
    if not started:
        return False
    di = _read_docker_images()                 # the pull may have written meanwhile
    rec = dict(di.get("strata") or {})
    rec["last_auto_check_at"] = time.time()
    di["strata"] = rec
    _write_docker_images(di)
    return True
