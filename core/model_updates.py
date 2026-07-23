# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Detecting and applying in-place updates to downloaded models.

Model authors routinely republish a repo with the same filenames - a requant, a
fixed chat template, a rebuilt imatrix - so a file that llamaman downloaded
months ago can be silently stale. HuggingFace exposes the content hash of every
LFS-tracked file, which is every GGUF, so detecting that costs one HTTP request
and no download:

    HEAD /{repo}/resolve/main/{file}  ->  x-linked-etag: "<sha256 of the blob>"

We compare it against the hash stamped at download time (core.model_sources),
never by re-reading the model off disk. Models that predate that stamp fall back
to a size comparison, which catches most requants but cannot prove a match - so
they report "unverified" rather than claiming to be up to date.

Applying an update reuses the normal downloader wholesale (speed limits, HF
tokens, pause/resume, auto-retry) by pointing it at a temp directory nested
inside the model's own directory, then swapping the finished files into place.
Nesting the temp dir there - rather than using a system temp - keeps the bytes
on the same filesystem, which is what makes the final swap an atomic
os.replace() per file instead of a copy that could tear.
"""

import hashlib
import os
import shutil
import threading

import requests

from config import logger

HF_API = "https://huggingface.co"

# Nested inside the model's directory so the swap is a same-filesystem rename.
# Dot-prefixed so discover_models' rglob doesn't surface a half-downloaded GGUF
# as a real model while the update is in flight.
UPDATE_TEMP_DIRNAME = ".llamaman-update"

STATUS_UP_TO_DATE = "up_to_date"
STATUS_UPDATE_AVAILABLE = "update_available"
STATUS_UNVERIFIED = "unverified"
STATUS_UNKNOWN = "unknown"
STATUS_NO_REPO = "no_repo"


def _headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def remote_file_info(repo_id: str, filename: str, token: str | None = None) -> dict:
    """{"sha256", "size", "commit"} for a repo file, via one HEAD request.

    Redirects are deliberately NOT followed. For an LFS file this URL answers
    302 to a CDN, and x-linked-etag / x-linked-size / x-repo-commit live on that
    302 - follow it and you land on the CDN response, which carries none of them
    (its own `etag` is a different 64-hex value entirely, so trusting that would
    report every model as changed on every check). Not following is also cheaper:
    we never touch the CDN at all.
    """
    url = f"{HF_API}/{repo_id}/resolve/main/{filename}"
    resp = requests.head(url, headers=_headers(token), timeout=30, allow_redirects=False)
    if resp.status_code in (401, 403):
        raise RuntimeError(f"Authentication failed ({resp.status_code}). Check your HF token.")
    if resp.status_code == 404:
        raise RuntimeError(f"File not found in repo: {repo_id}/{filename}")
    if resp.status_code >= 400:
        resp.raise_for_status()

    sha = (resp.headers.get("x-linked-etag") or "").strip().strip('"').lower()
    raw_size = resp.headers.get("x-linked-size") or resp.headers.get("content-length") or ""
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        size = 0
    return {
        "sha256": sha,
        "size": size,
        "commit": (resp.headers.get("x-repo-commit") or "").strip(),
    }


def check_model_update(model_path: str, repo_id: str, token: str | None = None) -> dict:
    """Compare a local model file against its source repo.

    Returns a dict with `status` plus whatever evidence backed the verdict, so
    the UI can be honest about how confident the answer is.
    """
    from core.model_sources import get_model_sha

    if not repo_id:
        return {"status": STATUS_NO_REPO,
                "detail": "no source repository recorded for this model"}

    filename = os.path.basename(model_path)
    try:
        remote = remote_file_info(repo_id, filename, token)
    except Exception as e:
        return {"status": STATUS_UNKNOWN, "detail": str(e)}

    try:
        local_size = os.path.getsize(model_path)
    except OSError as e:
        return {"status": STATUS_UNKNOWN, "detail": str(e)}

    local_sha = get_model_sha(model_path)
    result = {
        "repo_id": repo_id,
        "filename": filename,
        "local_sha256": local_sha,
        "remote_sha256": remote["sha256"],
        "local_size": local_size,
        "remote_size": remote["size"],
        "remote_commit": remote["commit"],
    }

    # Exact: we know what we downloaded and what's there now. The size still has
    # to agree - the hash is stamped when a download *starts*, so a pull that
    # died partway would otherwise look current on the strength of that stamp.
    if local_sha and remote["sha256"]:
        if local_sha == remote["sha256"] and (not remote["size"] or local_size == remote["size"]):
            result["status"] = STATUS_UP_TO_DATE
        else:
            result["status"] = STATUS_UPDATE_AVAILABLE
            if local_sha == remote["sha256"]:
                result["detail"] = ("local file is incomplete - its size does not "
                                    "match the published file")
        return result

    # No stamped hash. A size difference still proves the file changed.
    if remote["size"] and local_size != remote["size"]:
        result["status"] = STATUS_UPDATE_AVAILABLE
        result["detail"] = "size differs from the published file"
        return result

    if remote["size"] and local_size == remote["size"]:
        # Same size is strong evidence but not proof - a requant can land on the
        # same byte count. Say so rather than claiming it's current.
        result["status"] = STATUS_UNVERIFIED
        result["detail"] = ("size matches the published file, but no hash was "
                            "recorded for this model - verify to compare exactly")
        return result

    result["status"] = STATUS_UNKNOWN
    result["detail"] = "the repo published no size or hash for this file"
    return result


# ---------------------------------------------------------------------------
# Hashing the local file
# ---------------------------------------------------------------------------
# Models that predate hash stamping have nothing to compare against, and
# re-downloading gigabytes just to answer "did this change?" is absurd when the
# bytes are already on disk. Hashing locally settles it with no network at all.
# It is not instant though - tens of GB of disk reads - so it runs on a worker
# thread with progress, and the result is stamped so it is a one-time cost.

_HASH_CHUNK = 8 * 1024 * 1024

_hash_lock = threading.Lock()
_hash_jobs: dict[str, dict] = {}   # model_path -> job state


def local_hash_state(model_path: str) -> dict | None:
    """Current hashing job for `model_path`, or None if none was ever started."""
    with _hash_lock:
        job = _hash_jobs.get(model_path)
        return dict(job) if job else None


def _hash_worker(model_path: str) -> None:
    digest = hashlib.sha256()
    read = 0
    try:
        with open(model_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                read += len(chunk)
                with _hash_lock:
                    job = _hash_jobs.get(model_path)
                    if job is None or job.get("cancelled"):
                        return
                    job["hashed_bytes"] = read
        sha = digest.hexdigest()
    except Exception as e:
        logger.warning("Could not hash %s: %s", model_path, e)
        with _hash_lock:
            job = _hash_jobs.get(model_path)
            if job is not None:
                job.update(status="error", error=str(e))
        return

    # Stamp it: from here on this model compares exactly and instantly, and the
    # answer survives a restart.
    try:
        from core.model_sources import record_model_sha
        record_model_sha(model_path, sha)
    except Exception as e:
        logger.warning("Hashed %s but could not record it: %s", model_path, e)

    with _hash_lock:
        job = _hash_jobs.get(model_path)
        if job is not None:
            job.update(status="done", sha256=sha, hashed_bytes=read)
    logger.info("Hashed local model %s -> %s", os.path.basename(model_path), sha[:16])


def start_local_hash(model_path: str) -> dict:
    """Start (or rejoin) a background hash of `model_path`. Returns job state.

    Idempotent: asking again while one is running rejoins the existing job
    rather than reading the same 30GB twice.
    """
    try:
        total = os.path.getsize(model_path)
    except OSError as e:
        return {"status": "error", "error": str(e)}

    with _hash_lock:
        job = _hash_jobs.get(model_path)
        if job and job.get("status") == "hashing":
            return dict(job)
        job = {"status": "hashing", "hashed_bytes": 0, "total_bytes": total,
               "sha256": "", "error": ""}
        _hash_jobs[model_path] = job
        snapshot = dict(job)

    threading.Thread(target=_hash_worker, args=(model_path,),
                     name=f"hash-{os.path.basename(model_path)[:20]}",
                     daemon=True).start()
    return snapshot


# ---------------------------------------------------------------------------
# Background scan (opt-in, Settings -> Downloads)
# ---------------------------------------------------------------------------
# Fills in what the manual per-model button would otherwise make you wait for:
# hashes for models that have none, and a refreshed update verdict for the rest.
# Off by default. The check half is cheap (one HEAD per model); the hash half is
# a full disk read, so it does at most one model per pass and never runs while a
# download is active - otherwise the two compete for the same disk and both
# crawl.

AUTO_SCAN_ENABLED_KEY = "auto_update_scan_enabled"
AUTO_SCAN_INTERVAL_KEY = "auto_update_scan_interval_hours"
AUTO_SCAN_DEFAULT_INTERVAL_HOURS = 24

_scan_lock = threading.Lock()
_last_scan_status: dict[str, dict] = {}   # model_path -> last check result


def auto_scan_settings() -> tuple[bool, float]:
    """(enabled, interval_seconds) from settings. Disabled on any read error."""
    try:
        from storage import get_storage
        settings = get_storage().get_settings()
    except Exception:
        return False, AUTO_SCAN_DEFAULT_INTERVAL_HOURS * 3600
    enabled = settings.get(AUTO_SCAN_ENABLED_KEY, False) is True
    try:
        hours = float(settings.get(AUTO_SCAN_INTERVAL_KEY,
                                   AUTO_SCAN_DEFAULT_INTERVAL_HOURS) or 0)
    except (TypeError, ValueError):
        hours = AUTO_SCAN_DEFAULT_INTERVAL_HOURS
    if hours <= 0:
        hours = AUTO_SCAN_DEFAULT_INTERVAL_HOURS
    return enabled, hours * 3600


def last_scan_status() -> dict[str, dict]:
    """Latest per-model verdicts from the background scan.

    In-memory only: a restart just means the next pass recomputes them, and the
    expensive part (the hash) is persisted, not recomputed.
    """
    with _scan_lock:
        return {path: dict(v) for path, v in _last_scan_status.items()}


def _downloads_active() -> bool:
    try:
        from core.state import downloads, downloads_lock
        with downloads_lock:
            return any(d.get("status") == "downloading" for d in downloads.values())
    except Exception:
        return False


def run_auto_update_scan() -> dict:
    """One pass. Returns a small summary for the log.

    Order matters: check everything first (cheap, and it's what the user
    actually asked for), then spend disk on at most one hash.
    """
    from core.model_sources import get_model_sha, get_model_sources, resolve_model_source_repo_id

    if _downloads_active():
        return {"skipped": "download in progress"}

    try:
        from api.models import discover_models
        from config import MODELS_DIR
        models = discover_models(MODELS_DIR)
        sources = get_model_sources()
    except Exception as e:
        logger.warning("Auto update scan: could not list models: %s", e)
        return {"error": str(e)}

    checked = updates_found = 0
    needs_hash = []
    for m in models:
        path = m.get("path", "")
        repo_id = resolve_model_source_repo_id(path, sources)
        if not repo_id:
            continue  # nothing to compare against
        if not get_model_sha(path):
            needs_hash.append(path)
        result = check_model_update(path, repo_id)
        checked += 1
        if result.get("status") == STATUS_UPDATE_AVAILABLE:
            updates_found += 1
        with _scan_lock:
            _last_scan_status[path] = {"status": result.get("status", STATUS_UNKNOWN),
                                       "detail": result.get("detail", ""),
                                       "repo_id": repo_id}

    hashed = ""
    for path in needs_hash:
        state = local_hash_state(path)
        if state and state.get("status") == "hashing":
            break  # one at a time; a hash is already running
        start_local_hash(path)
        hashed = path
        break

    summary = {"checked": checked, "updates": updates_found,
               "hashing": os.path.basename(hashed) if hashed else "",
               "awaiting_hash": len(needs_hash)}
    logger.info("Auto update scan: %d checked, %d with updates, %d awaiting hash%s",
                checked, updates_found, len(needs_hash),
                f", hashing {summary['hashing']}" if hashed else "")
    return summary


def update_temp_dir(model_path: str) -> str:
    """Staging directory for an update to `model_path`."""
    return os.path.join(os.path.dirname(model_path), UPDATE_TEMP_DIRNAME)


def cleanup_update_temp(temp_dir: str) -> None:
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def swap_in_update(temp_dir: str, dest_dir: str) -> tuple[list[str], str | None]:
    """Move every downloaded file from `temp_dir` into `dest_dir`, replacing
    what's there. Returns (moved_relative_paths, error).

    Moves rather than copies, and both directories are on one filesystem by
    construction, so each file lands via an atomic os.replace(). Multi-shard
    models therefore swap shard by shard: a failure partway leaves some shards
    new and some old, which is logged loudly because the model is then
    inconsistent and needs the update re-run.
    """
    if not os.path.isdir(temp_dir):
        return [], "no downloaded files found to swap in"

    staged: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(temp_dir):
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, temp_dir)
            staged.append((rel, src))

    if not staged:
        return [], "no downloaded files found to swap in"

    moved: list[str] = []
    for rel, src in staged:
        dst = os.path.join(dest_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            moved.append(rel)
        except OSError as e:
            if moved:
                logger.error(
                    "Model update swap failed partway for %s: %d of %d files "
                    "replaced (%s). The model may be inconsistent - re-run the "
                    "update.", dest_dir, len(moved), len(staged), e,
                )
            return moved, f"could not replace {rel}: {e}"

    cleanup_update_temp(temp_dir)
    return moved, None
