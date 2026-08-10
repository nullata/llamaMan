# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Local mirror + degraded-mode operation for the MariaDB backend.

Without this, MariaDB is a hard dependency of every request path, not just of
persistence: `is_require_auth_enabled()` and `verify_api_key()` in api/auth.py
hit storage on every single request, and `MariaDBBackend.__init__` runs
`create_all()` at import time - so a database outage turns into 500s on every
route, and an outage that spans a container restart means the worker never boots
at all.

`ResilientBackend` wraps the real backend and a local `JsonBackend` mirror:

  - **Write-through.** Every successful primary write is replayed into the
    mirror, so the fallback data is seconds old rather than hours old. A mirror
    failure never fails the primary operation.
  - **Degraded reads.** Once the breaker opens, reads are served from the mirror
    and the primary is not touched at all - otherwise every request would pay a
    TCP connect timeout, which is the difference between an app that feels
    degraded and one that feels hung.
  - **Degraded writes.** Node-scoped writes (state, model_files, the cluster
    heartbeat) go to the mirror and are reconciled wholesale on reconnect, since
    this node is their only writer. Presets, API keys, settings patches and
    settings list edits are recorded in an append-only journal and replayed in
    order - as the *edit* rather than the resulting row, so a peer's concurrent
    change to another row/field/key survives. Only `save_settings` (a full-blob
    overwrite with no callers left in the app) and `save_user` stay blocked.
  - **Recovery.** A probe thread runs `SELECT 1` every `probe_interval` seconds
    while degraded, replays the journal, and only then closes the breaker.

The breaker is deliberately all-or-nothing: below the failure threshold the
wrapper behaves *exactly* as the bare backend does today (errors propagate to
the caller), and the switch to mirror/journal mode happens atomically at the
trip. Falling back on the first failure while still letting later calls reach a
flapping primary would interleave journalled and applied writes, and replaying
an older journalled delta on top of a newer applied one would silently revert
it.
"""

import json
import logging
import os
import threading
import time
from copy import deepcopy

from core.timeutil import now_iso
from storage.base import StorageBackend
from storage.json_backend import JsonBackend

logger = logging.getLogger("llamaman")

DEFAULT_PROBE_INTERVAL_S = 10.0
DEFAULT_FAIL_THRESHOLD = 3

MIRROR_DIRNAME = "db_mirror"
JOURNAL_FILENAME = "journal.jsonl"
OWNER_FILENAME = "owner.json"

# Node-scoped setting that turns mirroring on. Lives under settings["nodes"][id]
# because whether a host has usable local disk is a per-host fact.
MIRROR_SETTING_KEY = "db_mirror_enabled"
MIRROR_ENV_OVERRIDE = "LLAMAMAN_DB_MIRROR"


class StorageDegradedError(RuntimeError):
    """Raised when an operation requires the primary database but it is offline.

    Surfaces to clients as a 503 via the errorhandler registered in create_app().
    """


class _PrimaryDown(Exception):
    """Internal signal: the breaker is open, use the fallback path."""


def _is_connection_error(exc: BaseException) -> bool:
    """True only for errors that mean 'the database is unreachable'.

    IntegrityError / ProgrammingError are bugs, not outages, and must keep
    propagating unchanged - MariaDBBackend.merge_settings depends on catching
    IntegrityError itself to resolve its insert race.
    """
    try:
        from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
    except Exception:  # sqlalchemy absent (tests with a fake primary)
        return False
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True
    return False


def _preset_delta(baseline: dict, new: dict) -> tuple[dict, list[str]]:
    """Field-level diff of a preset row.

    Recording the delta rather than the resulting blob is what lets a peer's
    concurrent edit to a *different field of the same model* survive replay.
    api/presets.py rebuilds the whole preset dict from the launch form, so the
    blob alone would carry every field and clobber all of them.
    """
    changed = {k: v for k, v in new.items() if k not in baseline or baseline[k] != v}
    removed = [k for k in baseline if k not in new]
    return changed, removed


def _apply_preset_delta(current: dict, changed: dict, removed: list) -> dict:
    merged = dict(current)
    merged.update(changed)
    for key in removed or ():
        merged.pop(key, None)
    return merged


class _Journal:
    """Append-only record of writes made while degraded.

    Written *before* the mirror, so a journal failure rejects the operation
    outright instead of accepting a change that would silently never reach the
    database. Every op is idempotent, which is what makes replay-from-the-top
    safe after a crash: we only truncate once the whole file has been applied.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        record = {**record, "at": now_iso()}
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
        except OSError as e:
            # Refuse the operation rather than let the caller believe a change
            # was accepted that can never reach the database. Raised as the
            # degraded error so it surfaces as a 503 like every other rejection.
            raise StorageDegradedError(
                f"could not record offline change ({e}) - refusing the write"
            ) from e

    def load(self) -> list[dict]:
        with self._lock:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except (FileNotFoundError, OSError):
                return []
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                logger.warning("db mirror: skipping unparseable journal line")
        return out

    def truncate(self) -> None:
        with self._lock:
            try:
                os.unlink(self._path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("db mirror: could not truncate journal: %s", e)

    def pending_count(self) -> int:
        return len(self.load())


def build_mirror(mirror_dir: str) -> JsonBackend:
    """A JsonBackend rooted at its own directory.

    Deliberately a subdirectory of DATA_DIR rather than DATA_DIR itself: a node
    that previously ran in JSON mode still has state.json / settings.json /
    presets.json sitting there, and the mirror must never read or overwrite
    them. It also keeps the mirror's .migration.lock separate.
    """
    return JsonBackend(
        state_file=os.path.join(mirror_dir, "state.json"),
        presets_file=os.path.join(mirror_dir, "presets.json"),
        users_file=os.path.join(mirror_dir, "users.json"),
        settings_file=os.path.join(mirror_dir, "settings.json"),
        api_keys_file=os.path.join(mirror_dir, "api_keys.json"),
        recordings_dir=os.path.join(mirror_dir, "request_log"),
    )


def claim_mirror_dir(mirror_dir: str, node_id: str) -> bool:
    """Stamp the mirror directory with the owning node id.

    Two nodes bind-mounting the same host directory would otherwise interleave
    their node-scoped rows into one mirror and fight over it. Returns False if
    the directory already belongs to someone else, in which case mirroring stays
    off rather than corrupting either node's view.
    """
    marker = os.path.join(mirror_dir, OWNER_FILENAME)
    try:
        os.makedirs(mirror_dir, exist_ok=True)
        try:
            with open(marker, "r", encoding="utf-8") as f:
                owner = (json.load(f) or {}).get("node_id")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            owner = None
        if owner and owner != node_id:
            logger.error(
                "db mirror: %s belongs to node %r, not %r - mirroring disabled. "
                "Each node needs its own DATA_DIR.", mirror_dir, owner, node_id,
            )
            return False
        if owner != node_id:
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"node_id": node_id}, f)
        return True
    except OSError as e:
        logger.error("db mirror: cannot use %s (%s) - mirroring disabled", mirror_dir, e)
        return False


class ResilientBackend(StorageBackend):
    """Wraps a primary backend with a local mirror and degraded-mode fallback.

    With `enabled=False` this is a pure pass-through: no mirror I/O, no journal,
    no breaker, and every call behaves byte-identically to the bare primary.
    """

    def __init__(self, primary, mirror_dir: str, *, node_id: str,
                 builder=None, enabled: bool = False,
                 probe_interval: float = DEFAULT_PROBE_INTERVAL_S,
                 fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
                 mirror=None):
        self._primary = primary
        self._builder = builder
        self._node_id = node_id
        self._mirror_dir = mirror_dir
        self._mirror = mirror if mirror is not None else build_mirror(mirror_dir)
        self._journal = _Journal(os.path.join(mirror_dir, JOURNAL_FILENAME))

        self._enabled = bool(enabled)
        self._probe_interval = probe_interval
        self._fail_threshold = max(1, int(fail_threshold))

        self._lock = threading.RLock()
        self._degraded = primary is None
        self._degraded_since = time.time() if primary is None else None
        self._consecutive_failures = 0
        self._last_sync_at: float | None = None
        self._probe_thread: threading.Thread | None = None

        if self._degraded and self._enabled:
            self._start_probe()

    # -- status ------------------------------------------------------------

    def is_degraded(self) -> bool:
        with self._lock:
            return self._degraded

    def mirror_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @staticmethod
    def env_forced() -> bool:
        """True when LLAMAMAN_DB_MIRROR pins the toggle, so the UI can show the
        effective state and stop offering a control that would not take effect."""
        return os.environ.get(MIRROR_ENV_OVERRIDE, "").strip().lower() in (
            "1", "true", "yes", "on", "0", "false", "no", "off")

    def status(self) -> dict:
        with self._lock:
            snapshot = {
                "backend": "mariadb",
                "mirror_enabled": self._enabled,
                "env_forced": self.env_forced(),
                "degraded": self._degraded,
                "degraded_since": self._degraded_since,
                "last_sync_at": self._last_sync_at,
            }
        # Reads the journal file, so deliberately outside the lock - this is
        # polled by the dashboard and must not stall storage calls.
        snapshot["pending_ops"] = (
            self._journal.pending_count() if snapshot["mirror_enabled"] else 0)
        return snapshot

    # -- breaker -----------------------------------------------------------

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def _record_failure(self, exc: BaseException) -> bool:
        """Count a connection failure. Returns True if this one opened the breaker."""
        with self._lock:
            self._consecutive_failures += 1
            if self._degraded or self._consecutive_failures < self._fail_threshold:
                return False
            self._degraded = True
            self._degraded_since = time.time()
        logger.error(
            "db mirror: database unreachable after %d consecutive failures (%s) - "
            "serving from local mirror", self._fail_threshold, exc,
        )
        self._start_probe()
        return True

    def _start_probe(self) -> None:
        with self._lock:
            if self._probe_thread is not None and self._probe_thread.is_alive():
                return
            self._probe_thread = threading.Thread(
                target=self._probe_loop, name="db-mirror-probe", daemon=True)
            thread = self._probe_thread
        thread.start()

    def _probe_loop(self) -> None:
        while True:
            time.sleep(self._probe_interval)
            with self._lock:
                if not self._degraded:
                    return
            try:
                if self._try_recover():
                    return
            except Exception as e:  # never let the probe thread die
                logger.warning("db mirror: recovery attempt failed: %s", e)

    def _try_recover(self) -> bool:
        """Rebuild/ping the primary, replay, and close the breaker. True on success."""
        primary = self._primary
        if primary is None:
            if self._builder is None:
                return False
            # The backend never finished constructing (create_all() runs in
            # __init__), so recovery has to build it from scratch - pinging an
            # engine that does not exist is not enough.
            primary = self._builder()
            self._primary = primary

        if not self._ping(primary):
            return False

        # Schema and journal are replayed BEFORE the breaker closes, so no
        # request can observe a half-replayed state.
        self._run_reconnect(primary)

        with self._lock:
            self._degraded = False
            self._degraded_since = None
            self._consecutive_failures = 0
            self._last_sync_at = time.time()
        logger.info("db mirror: database reachable again - resumed normal operation")

        # Apply any toggle change that was deferred during the outage (see
        # resolve_enabled), now that the real settings are readable again.
        try:
            self.resolve_enabled(primary.get_settings())
        except Exception as e:
            logger.warning("db mirror: could not re-read the mirror toggle: %s", e)

        # Deliberately AFTER the breaker closes: this goes through
        # core.state.save_state(), which re-enters this wrapper, and would be
        # swallowed into the mirror if we were still marked degraded. Safe to
        # run late because save_state() is serialized by its own lock and always
        # writes the full current snapshot, so racing a concurrent save is a
        # no-op rather than a conflict.
        self._reconcile_node_rows(primary)
        return True

    @staticmethod
    def _ping(primary) -> bool:
        engine = getattr(primary, "_engine", None)
        if engine is None:
            return True  # non-SQL primary (tests); construction succeeding is the ping
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    # -- reconnect ---------------------------------------------------------

    def _run_reconnect(self, primary) -> None:
        from core.migrations import run_pending_migrations

        # Migrations first, and against the PRIMARY directly. Through the
        # wrapper, get_schema_version() would read the mirror's copy of the
        # settings blob, which can be ahead of what this node has actually
        # applied to the database.
        run_pending_migrations(primary)

        self._replay_journal(primary)

    def _replay_journal(self, primary) -> None:
        records = self._journal.load()
        if not records:
            return
        logger.info("db mirror: replaying %d journalled operation(s)", len(records))
        for rec in records:
            try:
                self._replay_one(primary, rec)
            except Exception as e:
                if _is_connection_error(e):
                    # Database dropped again mid-replay. Leave the journal
                    # intact; the next probe starts over from the top (every op
                    # is idempotent).
                    raise
                logger.error("db mirror: skipping journal op %r: %s", rec.get("op"), e)
        self._journal.truncate()
        logger.info("db mirror: journal replayed and cleared")

    @staticmethod
    def _replay_one(primary, rec: dict) -> None:
        op = rec.get("op")
        if op == "preset.set":
            path = rec["path"]
            current = primary.get_preset(path) or {}
            merged = _apply_preset_delta(current, rec.get("set") or {}, rec.get("unset") or [])
            primary.save_preset(path, merged)
        elif op == "preset.delete":
            primary.delete_preset(rec["path"])
        elif op == "apikey.save":
            primary.save_api_key(rec["entry"])
        elif op == "apikey.delete":
            primary.delete_api_key(rec["id"])
        elif op == "settings.merge":
            primary.merge_settings(rec["patch"])
        elif op == "settings.list_edit":
            primary.edit_settings_list(rec["key"], add=rec.get("add") or [],
                                       remove_ids=rec.get("remove_ids") or [])
        elif op == "settings.replace_key":
            primary.replace_settings_key(rec["key"], rec.get("value"))
        else:
            logger.warning("db mirror: unknown journal op %r - skipped", op)

    def _reconcile_node_rows(self, primary) -> None:
        """Push this node's node-scoped rows back. Safe wholesale: this node is
        their only writer and save_state(node_id=...) replaces only its own rows."""
        try:
            from core.state import save_state
            save_state()
        except Exception as e:
            logger.warning("db mirror: state reconcile failed: %s", e)

        # model_files: a model deleted while degraded must not resurrect, so
        # this is a two-way diff rather than a blind upsert.
        try:
            mirror_rows = self._mirror.get_model_files(self._node_id) or {}
            db_rows = primary.get_model_files(self._node_id) or {}
            for path in db_rows:
                if path not in mirror_rows:
                    primary.delete_model_files(self._node_id, path)
            for path, meta in mirror_rows.items():
                primary.upsert_model_file(
                    self._node_id, path,
                    repo_id=(meta or {}).get("repo_id", ""),
                    sha256=(meta or {}).get("sha256", ""),
                )
        except Exception as e:
            logger.warning("db mirror: model_files reconcile failed: %s", e)

    # -- full pull ---------------------------------------------------------

    def refresh_mirror(self) -> bool:
        """Pull the primary's shared tables into the mirror.

        Write-through cannot see rows written by *other* nodes, so this runs
        periodically to pick them up and repair drift. Skipped while the journal
        is non-empty: overwriting the mirror with the database while local edits
        are still pending would silently discard them.
        """
        with self._lock:
            if not self._enabled or self._degraded or self._primary is None:
                return False
        if self._journal.pending_count():
            logger.info("db mirror: refresh skipped, journal replay still pending")
            return False

        primary = self._primary
        try:
            settings = primary.get_settings()
            self._mirror.save_settings(settings)
            self.resolve_enabled(settings)

            for path, data in (primary.get_all_presets() or {}).items():
                self._mirror.save_preset(path, data)

            self._mirror_replace_api_keys(primary.get_api_keys() or [])

            # The cluster registry is deliberately NOT pulled into the mirror.
            # JsonBackend.register_node stamps last_heartbeat_at with the local
            # clock, so copying peer rows in would republish every peer with a
            # brand-new heartbeat - and _is_online() in api/cluster.py would then
            # report unreachable peers as online and let the dispatcher forward
            # inference to nodes we have no coordination store for. Leaving them
            # out means a degraded node sees only itself (its own register_node
            # write-through) and serves locally, which is the honest answer.

            for path, meta in (primary.get_model_files(self._node_id) or {}).items():
                self._mirror.upsert_model_file(
                    self._node_id, path,
                    repo_id=(meta or {}).get("repo_id", ""),
                    sha256=(meta or {}).get("sha256", ""),
                )
        except Exception as e:
            logger.warning("db mirror: refresh failed: %s", e)
            return False

        with self._lock:
            self._last_sync_at = time.time()
        logger.info("db mirror: refreshed from database")
        return True

    def _mirror_replace_api_keys(self, keys: list[dict]) -> None:
        """Mirror the key set exactly, including removals made on other nodes."""
        try:
            existing = {k["id"] for k in (self._mirror.get_api_keys() or []) if k.get("id")}
            incoming = {k["id"] for k in keys if k.get("id")}
            for key_id in existing - incoming:
                self._mirror.delete_api_key(key_id)
            for entry in keys:
                self._mirror.save_api_key(entry)
        except Exception as e:
            logger.warning("db mirror: api key refresh failed: %s", e)

    # -- enablement --------------------------------------------------------

    def resolve_enabled(self, settings: dict) -> None:
        """Re-read the toggle from a settings blob the caller already holds.

        Deliberately takes the blob rather than calling self.get_settings(),
        which would recurse through the very dispatch this is part of.
        """
        override = os.environ.get(MIRROR_ENV_OVERRIDE, "").strip().lower()
        if override in ("1", "true", "yes", "on"):
            value = True
        elif override in ("0", "false", "no", "off"):
            value = False
        else:
            node = ((settings or {}).get("nodes") or {}).get(self._node_id, {})
            if MIRROR_SETTING_KEY not in node:
                # Absent means "never configured", not "off". Treating it as off
                # would let any write that happens to carry a partial blob
                # silently switch mirroring back off underneath a node that has
                # it running.
                return
            value = bool(node.get(MIRROR_SETTING_KEY))
        with self._lock:
            if value == self._enabled:
                return
            if not value and self._degraded:
                # Switching off right now would drop us into pass-through
                # against a database we already know is unreachable: every read
                # would raise instead of falling back, and if we booted degraded
                # there is no primary object to call at all. Defer it - the
                # recovery path re-resolves once the database is back.
                logger.warning(
                    "db mirror: ignoring request to disable the mirror while the "
                    "database is offline; it will take effect after recovery")
                return
            self._enabled = value
            degraded = self._degraded
        logger.info("db mirror: %s", "enabled" if value else "disabled")
        if not value:
            return
        if degraded:
            # Turned on while the primary is already down (including the boot
            # path, where the wrapper is constructed before the toggle is
            # known). Nothing else would ever start the probe.
            self._start_probe()
            return
        # Turned on during normal operation. Without this the mirror would only
        # hold rows this node happened to write before the next daily sync - up
        # to 24h of near-empty fallback. Off-thread because a full pull walks
        # every preset and key, and this runs inside a settings write.
        threading.Thread(target=self.refresh_mirror, name="db-mirror-initial-sync",
                         daemon=True).start()

    # -- dispatch helpers --------------------------------------------------

    def _primary_call(self, name: str, *args, **kwargs):
        """Call the primary, converting an open breaker into _PrimaryDown.

        Below the failure threshold connection errors propagate to the caller
        exactly as they do today; the switch to fallback mode happens only at
        the trip, atomically.
        """
        with self._lock:
            if self._degraded:
                raise _PrimaryDown()
            primary = self._primary
        if primary is None:
            raise _PrimaryDown()
        try:
            result = getattr(primary, name)(*args, **kwargs)
        except Exception as e:
            if not _is_connection_error(e):
                raise
            if self._record_failure(e):
                raise _PrimaryDown() from e
            raise
        self._record_success()
        return result

    def _mirror_call(self, name: str, *args, **kwargs) -> None:
        """Best-effort write-through. Never fails the primary operation."""
        try:
            getattr(self._mirror, name)(*args, **kwargs)
        except Exception as e:
            logger.warning("db mirror: mirror write %s failed: %s", name, e)

    def _read(self, name: str, *args, **kwargs):
        if not self._enabled:
            return getattr(self._primary, name)(*args, **kwargs)
        try:
            result = self._primary_call(name, *args, **kwargs)
        except _PrimaryDown:
            return getattr(self._mirror, name)(*args, **kwargs)
        return result

    def _node_write(self, name: str, *args, **kwargs) -> None:
        """Node-scoped write: mirrored always, reconciled wholesale on reconnect."""
        if not self._enabled:
            return getattr(self._primary, name)(*args, **kwargs)
        try:
            self._primary_call(name, *args, **kwargs)
        except _PrimaryDown:
            pass
        self._mirror_call(name, *args, **kwargs)

    def _blocked_write(self, name: str, *args, **kwargs):
        if not self._enabled:
            return getattr(self._primary, name)(*args, **kwargs)
        try:
            result = self._primary_call(name, *args, **kwargs)
        except _PrimaryDown:
            raise StorageDegradedError(
                f"{name}: database offline - this change cannot be made safely "
                "while disconnected"
            )
        self._mirror_call(name, *args, **kwargs)
        return result

    # -- State -------------------------------------------------------------

    def save_state(self, instances, downloads, node_id=None) -> None:
        self._node_write("save_state", instances, downloads, node_id)

    def load_instances(self, node_id=None):
        return self._read("load_instances", node_id)

    def load_downloads(self, node_id=None):
        return self._read("load_downloads", node_id)

    # -- Presets -----------------------------------------------------------

    def get_all_presets(self):
        return self._read("get_all_presets")

    def get_preset(self, model_path: str):
        return self._read("get_preset", model_path)

    def save_preset(self, model_path: str, data: dict) -> None:
        if not self._enabled:
            return self._primary.save_preset(model_path, data)
        try:
            self._primary_call("save_preset", model_path, data)
        except _PrimaryDown:
            # The baseline is whatever the caller just read, and while degraded
            # reads come from the mirror - so the mirror IS the baseline the
            # handler in api/presets.py built its dict from. No plumbing needed.
            baseline = {}
            try:
                baseline = self._mirror.get_preset(model_path) or {}
            except Exception:
                pass
            changed, removed = _preset_delta(baseline, data)
            if changed or removed:
                self._journal.append({
                    "op": "preset.set", "path": model_path,
                    "set": changed, "unset": removed,
                })
        self._mirror_call("save_preset", model_path, data)

    def delete_preset(self, model_path: str) -> None:
        if not self._enabled:
            return self._primary.delete_preset(model_path)
        try:
            self._primary_call("delete_preset", model_path)
        except _PrimaryDown:
            self._journal.append({"op": "preset.delete", "path": model_path})
        self._mirror_call("delete_preset", model_path)

    # -- Auth --------------------------------------------------------------

    def get_user(self, username: str):
        return self._read("get_user", username)

    def save_user(self, username: str, password_hash: str) -> None:
        # Only reachable from /setup on a brand-new install, which by definition
        # has no mirror to fall back to.
        self._blocked_write("save_user", username, password_hash)

    def user_count(self) -> int:
        return self._read("user_count")

    # -- Settings ----------------------------------------------------------

    def get_settings(self) -> dict:
        return self._read("get_settings")

    def save_settings(self, settings: dict) -> None:
        # Full-blob overwrite (api/settings.py's HF-token path). Replaying it
        # would stomp every key a peer changed during the outage, so it is the
        # one settings write that stays blocked.
        self._blocked_write("save_settings", settings)
        self.resolve_enabled(settings)

    def merge_settings(self, patch: dict) -> dict:
        if not self._enabled:
            merged = self._primary.merge_settings(patch)
            # The patch that switches mirroring ON necessarily arrives while we
            # are still in pass-through, so this is the one write that has to
            # re-resolve the toggle itself - otherwise enabling it persists to
            # the database but cannot take effect until the next restart, and
            # the UI (which drives the checkbox from the effective state) just
            # unchecks the box again on its next poll.
            self.resolve_enabled(merged)
            return merged
        try:
            merged = self._primary_call("merge_settings", patch)
        except _PrimaryDown:
            self._journal.append({"op": "settings.merge", "patch": patch})
            merged = self._mirror.merge_settings(patch)
            self.resolve_enabled(merged)
            return merged
        self._mirror_call("save_settings", merged)
        self.resolve_enabled(merged)
        return merged

    def edit_settings_list(self, key: str, *, add: list[dict] | None = None,
                           remove_ids: list[str] | None = None) -> list[dict]:
        if not self._enabled:
            return self._primary.edit_settings_list(key, add=add, remove_ids=remove_ids)
        try:
            result = self._primary_call("edit_settings_list", key,
                                        add=add, remove_ids=remove_ids)
        except _PrimaryDown:
            # Journal the EDIT, not the resulting list. Replaying "add this
            # entry" merges with whatever peers did during the outage; replaying
            # the list this node happened to end up with would drop their
            # additions - and unlike the online race, that window is the whole
            # length of the outage.
            self._journal.append({
                "op": "settings.list_edit", "key": key,
                "add": deepcopy(add) or [], "remove_ids": list(remove_ids or []),
            })
            return self._mirror.edit_settings_list(key, add=add, remove_ids=remove_ids)
        self._mirror_call("merge_settings", {key: result})
        return result

    def replace_settings_key(self, key: str, value) -> None:
        if not self._enabled:
            return self._primary.replace_settings_key(key, value)
        try:
            self._primary_call("replace_settings_key", key, value)
        except _PrimaryDown:
            self._journal.append({
                "op": "settings.replace_key", "key": key, "value": deepcopy(value),
            })
        self._mirror_call("replace_settings_key", key, value)

    # -- API keys ----------------------------------------------------------

    def get_api_keys(self):
        return self._read("get_api_keys")

    def save_api_key(self, key_entry: dict) -> None:
        if not self._enabled:
            return self._primary.save_api_key(key_entry)
        try:
            self._primary_call("save_api_key", key_entry)
        except _PrimaryDown:
            # The id is a random token_hex(8) minted per key, so replaying an
            # insert can never collide with anything a peer created.
            self._journal.append({"op": "apikey.save", "entry": deepcopy(key_entry)})
        self._mirror_call("save_api_key", key_entry)

    def delete_api_key(self, key_id: str) -> None:
        if not self._enabled:
            return self._primary.delete_api_key(key_id)
        try:
            self._primary_call("delete_api_key", key_id)
        except _PrimaryDown:
            self._journal.append({"op": "apikey.delete", "id": key_id})
        self._mirror_call("delete_api_key", key_id)

    def verify_api_key(self, raw_key: str) -> bool:
        return self._read("verify_api_key", raw_key)

    # -- Model files -------------------------------------------------------

    def get_model_files(self, node_id: str):
        return self._read("get_model_files", node_id)

    def upsert_model_file(self, node_id: str, model_path: str,
                          repo_id: str = "", sha256: str = "") -> None:
        self._node_write("upsert_model_file", node_id, model_path,
                         repo_id=repo_id, sha256=sha256)

    def delete_model_files(self, node_id: str, path_prefix: str) -> None:
        self._node_write("delete_model_files", node_id, path_prefix)

    # -- Cluster registry --------------------------------------------------

    def register_node(self, node: dict, snapshot=None) -> None:
        self._node_write("register_node", node, snapshot)

    def list_nodes(self):
        return self._read("list_nodes")

    def get_node(self, node_id: str):
        return self._read("get_node", node_id)

    def remove_node(self, node_id: str) -> None:
        self._node_write("remove_node", node_id)

    # -- Request log -------------------------------------------------------
    #
    # Never mirrored: records carry full request AND response bodies, so
    # buffering them through a multi-hour outage could fill the data volume and
    # take down the inference this whole feature exists to protect. Telemetry is
    # the cheap thing to lose here.

    def append_request_log(self, record: dict, mode: str) -> None:
        if not self._enabled:
            return self._primary.append_request_log(record, mode)
        try:
            self._primary_call("append_request_log", record, mode)
        except _PrimaryDown:
            logger.warning("db mirror: request_log record dropped (database offline)")

    def list_conversations(self, limit: int = 100):
        if not self._enabled:
            return self._primary.list_conversations(limit=limit)
        try:
            return self._primary_call("list_conversations", limit=limit)
        except _PrimaryDown:
            return []

    def get_conversation_turns(self, conversation_id: str):
        if not self._enabled:
            return self._primary.get_conversation_turns(conversation_id)
        try:
            return self._primary_call("get_conversation_turns", conversation_id)
        except _PrimaryDown:
            return []

    def prune_request_log(self, older_than) -> int:
        if not self._enabled:
            return self._primary.prune_request_log(older_than)
        try:
            return self._primary_call("prune_request_log", older_than)
        except _PrimaryDown:
            return 0

    def clear_request_log(self) -> int:
        if not self._enabled:
            return self._primary.clear_request_log()
        try:
            return self._primary_call("clear_request_log")
        except _PrimaryDown:
            return 0

    def request_log_stats(self, inst_id=None, since=None) -> dict:
        if not self._enabled:
            return self._primary.request_log_stats(inst_id=inst_id, since=since)
        try:
            return self._primary_call("request_log_stats", inst_id=inst_id, since=since)
        except _PrimaryDown:
            return {
                "turn_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "avg_tokens_per_sec": None, "max_tokens_per_sec": None,
                "avg_ttft_ms": None, "avg_duration_ms": None,
                "error_count": 0, "streamed_count": 0,
                "first_seen_at": None, "last_seen_at": None,
            }

    # -- Migrations --------------------------------------------------------

    def migration_lock(self):
        # Migrations never run while degraded; app.py guards on is_degraded()
        # and the reconnect path calls run_pending_migrations(primary) directly.
        if self._primary is None:
            raise StorageDegradedError("migration_lock: database offline")
        return self._primary.migration_lock()

    def get_schema_version(self) -> int:
        if self._primary is None:
            return 0
        return self._primary.get_schema_version()

    def set_schema_version(self, version: int) -> None:
        if self._primary is None:
            raise StorageDegradedError("set_schema_version: database offline")
        self._primary.set_schema_version(version)

    def apply_migration_001_timestamps(self) -> None:
        return self._primary.apply_migration_001_timestamps()

    def apply_migration_002_request_metrics(self) -> None:
        return self._primary.apply_migration_002_request_metrics()

    def apply_migration_003_node_scoped_state(self) -> None:
        return self._primary.apply_migration_003_node_scoped_state()
