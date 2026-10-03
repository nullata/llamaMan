# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Model archive: move models between MODELS_DIR and a second storage volume
(ARCHIVE_DIR) and back, from the UI.

A model is moved as a *unit*: its top-level folder under the base directory
(what a download creates, e.g. models/Qwen-...-00001-of-00004/ with every
shard inside), or - for a .gguf lying directly in the base - the file plus its
multipart siblings. Strata's shard folders (strata/<tag>/) are a unit each.

Moves never lose data:
  * same filesystem -> os.rename (instant, atomic per entry)
  * otherwise -> copy into <dest>/.llamaman-partial-<job>/, fsync, verify every
    file's size, rename into place, and only then delete the source.
A crash or cancel mid-copy leaves the source untouched; stale partial folders
are removed at startup (cleanup_partials).

Jobs run one at a time on a daemon worker thread and live in memory only (a
restart mid-move drops the job; the source is still where it was).
"""

import os
import re
import shutil
import threading
import time
import uuid

from config import logger

PARTIAL_PREFIX = ".llamaman-partial-"
_MULTIPART_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_CHUNK = 8 * 1024 * 1024

ACTIVE_STATUSES = ("queued", "moving")

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
_queue_cv = threading.Condition(jobs_lock)
_worker_started = False


class ArchiveError(Exception):
    """A refusal or failure with a user-facing message."""


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Configuration / status
# ---------------------------------------------------------------------------

def archive_dir() -> str:
    from config import ARCHIVE_DIR
    return ARCHIVE_DIR


def is_enabled() -> bool:
    return bool(archive_dir())


def status() -> dict:
    """Whether archiving is configured and usable on this node."""
    d = archive_dir()
    if not d:
        return {"enabled": False, "available": False, "dir": "",
                "reason": "ARCHIVE_DIR is not set"}
    if not os.path.isdir(d):
        return {"enabled": True, "available": False, "dir": d,
                "reason": f"{d} does not exist - mount the archive volume there"}
    if not os.access(d, os.W_OK):
        return {"enabled": True, "available": False, "dir": d,
                "reason": f"{d} is not writable"}
    try:
        free = shutil.disk_usage(d).free
    except OSError:
        free = None
    return {"enabled": True, "available": True, "dir": d, "reason": "", "free_bytes": free}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def _inside(path: str, base: str) -> bool:
    real, real_base = os.path.realpath(path), os.path.realpath(base)
    return real != real_base and real.startswith(real_base + os.sep)


def model_unit(model_path: str, base: str) -> list[str]:
    """The entries (paths relative to `base`) that make up the model at
    `model_path`. Raises ArchiveError when the path isn't a model inside base."""
    if not model_path or not _inside(model_path, base):
        raise ArchiveError(f"'{model_path}' is not inside {base}")
    if not os.path.exists(model_path):
        raise ArchiveError(f"'{model_path}' does not exist")
    rel = os.path.relpath(os.path.realpath(model_path), os.path.realpath(base))
    parts = rel.split(os.sep)
    if any(p.startswith(PARTIAL_PREFIX) for p in parts):
        raise ArchiveError("that file is part of a move in progress")
    if len(parts) > 1:
        # Strata keeps one folder per model under strata/ (core/engines/strata.py).
        if parts[0] == "strata" and len(parts) > 2:
            return [os.path.join(parts[0], parts[1])]
        return [parts[0]]
    name = parts[0]
    m = _MULTIPART_RE.match(name)
    if not m:
        return [name]
    stem, total = m.group(1), m.group(3)
    siblings = sorted(
        f for f in os.listdir(os.path.realpath(base))
        if (sm := _MULTIPART_RE.match(f)) and sm.group(1) == stem and sm.group(3) == total
    )
    return siblings or [name]


def unit_paths(base: str, unit: list[str]) -> list[str]:
    return [os.path.join(base, rel) for rel in unit]


def path_in_unit(path: str, base: str, unit: list[str]) -> bool:
    if not path:
        return False
    real = os.path.realpath(path)
    for p in unit_paths(base, unit):
        rp = os.path.realpath(p)
        if real == rp or real.startswith(rp + os.sep):
            return True
    return False


def _files(base: str, unit: list[str]) -> list[str]:
    """Every regular file in the unit, relative to base."""
    out = []
    for rel in unit:
        p = os.path.join(base, rel)
        if os.path.isdir(p) and not os.path.islink(p):
            for root, _dirs, names in os.walk(p):
                for n in names:
                    out.append(os.path.relpath(os.path.join(root, n), base))
        else:
            out.append(rel)
    return out


def unit_size(base: str, unit: list[str]) -> int:
    total = 0
    for rel in _files(base, unit):
        try:
            total += os.path.getsize(os.path.join(base, rel))
        except OSError:
            pass
    return total


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def active_job_for(path: str) -> dict | None:
    """The queued/running job whose unit contains `path` (either side)."""
    with jobs_lock:
        for job in jobs.values():
            if job["status"] not in ACTIVE_STATUSES:
                continue
            if path_in_unit(path, job["src_base"], job["unit"]) or \
               path_in_unit(path, job["dst_base"], job["unit"]):
                return dict(job)
    return None


def busy_reason(model_path: str) -> str | None:
    """Why a model can't be used right now because it is being moved."""
    job = active_job_for(model_path)
    if not job:
        return None
    verb = "archived" if job["kind"] == "archive" else "restored"
    return f"{os.path.basename(model_path)} is being {verb} ({job['status']}); try again when the move finishes"


def public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def list_jobs() -> list[dict]:
    with jobs_lock:
        return [public_job(j) for j in sorted(jobs.values(), key=lambda j: j["created_at"])]


def start_job(kind: str, model_path: str, src_base: str, dst_base: str,
              unit: list[str]) -> dict:
    """Queue a move. Preflight checks that need only the filesystem happen
    here so the caller gets an immediate error; the worker re-checks."""
    for rel in unit:
        if os.path.lexists(os.path.join(dst_base, rel)):
            raise ArchiveError(f"'{rel}' already exists in {dst_base}")
    size = unit_size(src_base, unit)
    with jobs_lock:
        for job in jobs.values():
            if job["status"] in ACTIVE_STATUSES and set(job["unit"]) & set(unit) \
                    and job["src_base"] in (src_base, dst_base):
                raise ArchiveError(f"'{unit[0]}' already has a {job['kind']} in progress")
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,                     # archive | restore
            "model_path": model_path,
            "name": unit[0] if len(unit) == 1 else f"{unit[0]} (+{len(unit) - 1})",
            "unit": list(unit),
            "src_base": src_base,
            "dst_base": dst_base,
            "status": "queued",               # queued | moving | completed | failed | cancelled
            "bytes_total": size,
            "bytes_done": 0,
            "speed": 0.0,
            "method": None,                   # rename | copy
            "error": None,
            "warning": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "_cancel": threading.Event(),
        }
        jobs[job["id"]] = job
        _ensure_worker()
        _queue_cv.notify_all()
    logger.info("archive: queued %s of %s (%s -> %s)", kind, job["name"], src_base, dst_base)
    return public_job(job)


def cancel_job(job_id: str) -> dict | None:
    """Cancel a queued/running move, or forget a finished one. Returns the
    job (as it now stands) or None when unknown."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
        elif job["status"] == "moving":
            job["_cancel"].set()
        else:
            jobs.pop(job_id, None)
        return public_job(job)


def _ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    threading.Thread(target=_worker, name="archive-mover", daemon=True).start()


def _next_job() -> dict | None:
    for job in sorted(jobs.values(), key=lambda j: j["created_at"]):
        if job["status"] == "queued":
            return job
    return None


def _worker() -> None:
    while True:
        with jobs_lock:
            job = _next_job()
            while job is None:
                _queue_cv.wait()
                job = _next_job()
            job["status"] = "moving"
            job["started_at"] = time.time()
        try:
            run_move(job)
            with jobs_lock:
                job["status"] = "completed"
            logger.info("archive: %s of %s completed (%s)", job["kind"], job["name"], job["method"])
        except _Cancelled:
            with jobs_lock:
                job["status"] = "cancelled"
            logger.info("archive: %s of %s cancelled", job["kind"], job["name"])
        except Exception as e:
            with jobs_lock:
                job["status"] = "failed"
                job["error"] = str(e)
            logger.warning("archive: %s of %s failed: %s", job["kind"], job["name"], e)
        finally:
            with jobs_lock:
                job["finished_at"] = time.time()
                job["speed"] = 0.0


# ---------------------------------------------------------------------------
# The move itself
# ---------------------------------------------------------------------------

def _same_device(a: str, b: str) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def run_move(job: dict) -> None:
    """Move job['unit'] from src_base to dst_base. Raises on failure;
    _Cancelled when cancelled before the source was touched."""
    src_base, dst_base, unit = job["src_base"], job["dst_base"], job["unit"]
    for rel in unit:
        if not os.path.lexists(os.path.join(src_base, rel)):
            raise ArchiveError(f"'{rel}' no longer exists in {src_base}")
        if os.path.lexists(os.path.join(dst_base, rel)):
            raise ArchiveError(f"'{rel}' already exists in {dst_base}")

    if _same_device(src_base, dst_base):
        renamed = []
        try:
            for rel in unit:
                os.makedirs(os.path.dirname(os.path.join(dst_base, rel)), exist_ok=True)
                os.rename(os.path.join(src_base, rel), os.path.join(dst_base, rel))
                renamed.append(rel)
            job["method"] = "rename"
            job["bytes_done"] = job["bytes_total"]
            return
        except OSError as e:
            if renamed:
                # Each rename is atomic, so nothing is lost - but the unit is
                # now split between the two places.
                raise ArchiveError(f"moved {', '.join(renamed)} but not the rest ({e}); "
                                   "move the remaining files by hand") from e
            # EXDEV on separate bind mounts of one device, or a union
            # filesystem refusing cross-branch renames: copy instead.
            logger.info("archive: rename failed (%s), copying instead", e)

    job["method"] = "copy"
    total = unit_size(src_base, unit)
    job["bytes_total"] = total
    try:
        free = shutil.disk_usage(dst_base).free
    except OSError:
        free = None
    if free is not None and free < total:
        raise ArchiveError(f"not enough free space in {dst_base}: need {total / 1e9:.1f} GB, "
                           f"{free / 1e9:.1f} GB free")

    tmp = os.path.join(dst_base, PARTIAL_PREFIX + job["id"])
    os.makedirs(tmp, exist_ok=True)
    try:
        _copy_unit(job, src_base, tmp, unit)
        for rel in _files(src_base, unit):
            a, b = os.path.join(src_base, rel), os.path.join(tmp, rel)
            if os.path.getsize(a) != os.path.getsize(b):
                raise ArchiveError(f"verification failed for {rel}: sizes differ")
        if job["_cancel"].is_set():
            raise _Cancelled()
        for rel in unit:
            final = os.path.join(dst_base, rel)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            os.rename(os.path.join(tmp, rel), final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(tmp, ignore_errors=True)

    # Everything is safely at the destination; remove the source.
    problems = []
    for rel in unit:
        p = os.path.join(src_base, rel)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except OSError as e:
            problems.append(f"{rel}: {e}")
    _prune_empty_parents(src_base, unit)
    if problems:
        job["warning"] = "copied, but could not remove the source: " + "; ".join(problems)


def _prune_empty_parents(base: str, unit: list[str]) -> None:
    """Remove now-empty intermediate folders (e.g. strata/ after its last
    model moved), never base itself."""
    for rel in unit:
        parent = os.path.dirname(rel)
        while parent:
            try:
                os.rmdir(os.path.join(base, parent))
            except OSError:
                break
            parent = os.path.dirname(parent)


def _copy_unit(job: dict, src_base: str, tmp: str, unit: list[str]) -> None:
    last_t, last_b = time.monotonic(), 0
    for rel in _files(src_base, unit):
        src, dst = os.path.join(src_base, rel), os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
            continue
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            while True:
                if job["_cancel"].is_set():
                    raise _Cancelled()
                buf = fi.read(_CHUNK)
                if not buf:
                    break
                fo.write(buf)
                job["bytes_done"] += len(buf)
                now = time.monotonic()
                if now - last_t >= 1.0:
                    job["speed"] = (job["bytes_done"] - last_b) / (now - last_t)
                    last_t, last_b = now, job["bytes_done"]
            fo.flush()
            os.fsync(fo.fileno())
        try:
            shutil.copystat(src, dst, follow_symlinks=False)
        except OSError:
            # Timestamps / mode are a nicety; some filesystems (NTFS via
            # drvfs, some FUSE/SMB/NFS shares) refuse utime or chmod.
            pass


def cleanup_partials() -> int:
    """Remove partial copies left by a move interrupted by a restart."""
    from config import MODELS_DIR
    removed = 0
    for base in (MODELS_DIR, archive_dir()):
        if not base or not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.startswith(PARTIAL_PREFIX):
                shutil.rmtree(os.path.join(base, name), ignore_errors=True)
                removed += 1
    if removed:
        logger.info("archive: removed %d partial copy folder(s) from an interrupted move", removed)
    return removed
