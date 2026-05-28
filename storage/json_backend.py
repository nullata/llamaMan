# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone

from core.timeutil import epoch_ms_to_iso, epoch_s_to_iso, now_iso, parse_iso, to_iso
from storage.base import StorageBackend

logger = logging.getLogger("llamaman")


def _merge_dicts(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON to a file atomically (write tmp + rename).

    If the process crashes mid-write, the original file is untouched.
    """
    dir_name = os.path.dirname(path) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
    except Exception as e:
        logger.warning("Failed to write %s: %s", path, e)


class JsonBackend(StorageBackend):
    """Stores data in local JSON files. Zero dependencies, works out of the box."""

    def __init__(self, state_file: str, presets_file: str, users_file: str,
                 settings_file: str, api_keys_file: str = "",
                 recordings_dir: str = ""):
        self._state_file = state_file
        self._presets_file = presets_file
        self._users_file = users_file
        self._settings_file = settings_file
        self._api_keys_file = api_keys_file or os.path.join(
            os.path.dirname(settings_file), "api_keys.json"
        )
        self._recordings_dir = recordings_dir or os.path.join(
            os.path.dirname(settings_file), "request_log"
        )
        self._cluster_file = os.path.join(
            os.path.dirname(settings_file) or ".", "cluster.json"
        )
        self._cluster_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._request_log_lock = threading.Lock()
        self._migration_lock_path = os.path.join(
            os.path.dirname(self._settings_file) or ".", ".migration.lock"
        )

    # -- Migrations --

    @contextlib.contextmanager
    def migration_lock(self):
        """Filesystem lockfile so a second process can't run migrations
        concurrently. We don't use fcntl here because Windows doesn't have it
        and the dev experience matters; an atomic O_CREAT|O_EXCL is enough.
        """
        deadline = time.time() + 60
        while True:
            try:
                fd = os.open(self._migration_lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Could not acquire migration lock at {self._migration_lock_path} "
                        "within 60s. If no other process is running, delete the file."
                    )
                time.sleep(0.5)
        try:
            yield
        finally:
            try:
                os.unlink(self._migration_lock_path)
            except OSError:
                pass

    def apply_migration_001_timestamps(self) -> None:
        # api_keys.json: created_at int(epoch s) -> ISO string
        try:
            keys = self._read_api_keys()
        except Exception:
            keys = []
        changed = False
        for k in keys:
            c = k.get("created_at")
            if isinstance(c, (int, float)) and c > 0:
                k["created_at"] = epoch_s_to_iso(c)
                changed = True
        if changed:
            logger.info("Migration 001: rewriting api_keys.json with ISO timestamps")
            self._write_api_keys(keys)

        # request_log/*: rewrite each .jsonl record's created_at int(ms) -> ISO
        if not os.path.isdir(self._recordings_dir):
            return
        rewritten = 0
        for date in os.listdir(self._recordings_dir):
            date_dir = os.path.join(self._recordings_dir, date)
            if not os.path.isdir(date_dir):
                continue
            for fn in os.listdir(date_dir):
                if not fn.endswith(".jsonl"):
                    continue
                full = os.path.join(date_dir, fn)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except OSError:
                    continue
                file_changed = False
                out_lines = []
                for line in lines:
                    s = line.strip()
                    if not s:
                        out_lines.append(line)
                        continue
                    try:
                        rec = json.loads(s)
                    except (json.JSONDecodeError, ValueError):
                        out_lines.append(line)
                        continue
                    c = rec.get("created_at")
                    if isinstance(c, (int, float)):
                        rec["created_at"] = epoch_ms_to_iso(c)
                        out_lines.append(json.dumps(
                            rec, separators=(",", ":"), ensure_ascii=False) + "\n")
                        file_changed = True
                    else:
                        out_lines.append(line)
                if file_changed:
                    # Atomic rewrite: write to tmp + rename so a crash mid-write
                    # leaves the original file intact.
                    tmp = full + ".tmp"
                    try:
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.writelines(out_lines)
                        os.replace(tmp, full)
                        rewritten += 1
                    except OSError as e:
                        logger.warning("Migration 001: failed to rewrite %s: %s", full, e)
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
        if rewritten:
            logger.info("Migration 001: rewrote %d request_log files", rewritten)

    # -- State helpers --

    def _read_state(self) -> dict:
        try:
            with open(self._state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # -- State (atomic read/write) --

    @staticmethod
    def _owned_by(row: dict, node_id: str) -> bool:
        # Rows predating node scoping have no node_id; treat them as the local
        # node's so single-node installs upgrade without losing state.
        rid = row.get("node_id")
        return rid is None or rid == node_id

    def save_state(self, instances: list[dict], downloads: list[dict],
                   node_id: str | None = None) -> None:
        with self._state_lock:
            if node_id is None:
                state = {"instances": instances, "downloads": downloads}
                _atomic_write_json(self._state_file, state)
                return
            # Preserve rows owned by other nodes (shared-storage case); replace
            # this node's rows, stamping each with the owning node_id.
            existing = self._read_state()
            other_inst = [r for r in existing.get("instances", []) if not self._owned_by(r, node_id)]
            other_dl = [r for r in existing.get("downloads", []) if not self._owned_by(r, node_id)]
            mine_inst = [{**r, "node_id": node_id} for r in instances]
            mine_dl = [{**r, "node_id": node_id} for r in downloads]
            state = {"instances": other_inst + mine_inst, "downloads": other_dl + mine_dl}
            _atomic_write_json(self._state_file, state)

    def load_instances(self, node_id: str | None = None) -> list[dict]:
        rows = self._read_state().get("instances", [])
        if node_id is None:
            return rows
        return [r for r in rows if self._owned_by(r, node_id)]

    def load_downloads(self, node_id: str | None = None) -> list[dict]:
        rows = self._read_state().get("downloads", [])
        if node_id is None:
            return rows
        return [r for r in rows if self._owned_by(r, node_id)]

    # -- Presets --

    def get_all_presets(self) -> dict[str, dict]:
        try:
            with open(self._presets_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get_preset(self, model_path: str) -> dict | None:
        return self.get_all_presets().get(model_path)

    def save_preset(self, model_path: str, data: dict) -> None:
        presets = self.get_all_presets()
        presets[model_path] = data
        _atomic_write_json(self._presets_file, presets)

    def delete_preset(self, model_path: str) -> None:
        presets = self.get_all_presets()
        if model_path in presets:
            del presets[model_path]
            _atomic_write_json(self._presets_file, presets)

    # -- Auth --

    def _read_users(self) -> dict[str, dict]:
        try:
            with open(self._users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_users(self, users: dict[str, dict]) -> None:
        _atomic_write_json(self._users_file, users)

    def get_user(self, username: str) -> dict | None:
        return self._read_users().get(username)

    def save_user(self, username: str, password_hash: str) -> None:
        users = self._read_users()
        users[username] = {"username": username, "password_hash": password_hash}
        self._write_users(users)

    def user_count(self) -> int:
        return len(self._read_users())

    # -- Settings --

    def get_settings(self) -> dict:
        with self._settings_lock:
            try:
                with open(self._settings_file, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

    def save_settings(self, settings: dict) -> None:
        with self._settings_lock:
            _atomic_write_json(self._settings_file, settings)

    def merge_settings(self, patch: dict) -> dict:
        with self._settings_lock:
            try:
                with open(self._settings_file, "r") as f:
                    current = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                current = {}
            merged = _merge_dicts(current, patch)
            _atomic_write_json(self._settings_file, merged)
            return merged

    # -- API Keys --

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _read_api_keys(self) -> list[dict]:
        try:
            with open(self._api_keys_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_api_keys(self, keys: list[dict]) -> None:
        _atomic_write_json(self._api_keys_file, keys)

    def get_api_keys(self) -> list[dict]:
        return self._read_api_keys()

    def save_api_key(self, key_entry: dict) -> None:
        keys = self._read_api_keys()
        # Update existing or append
        for i, k in enumerate(keys):
            if k["id"] == key_entry["id"]:
                keys[i] = key_entry
                self._write_api_keys(keys)
                return
        keys.append(key_entry)
        self._write_api_keys(keys)

    def delete_api_key(self, key_id: str) -> None:
        keys = self._read_api_keys()
        keys = [k for k in keys if k["id"] != key_id]
        self._write_api_keys(keys)

    def verify_api_key(self, raw_key: str) -> bool:
        hashed = self._hash_key(raw_key)
        for k in self._read_api_keys():
            if k.get("key_hash") == hashed:
                return True
        return False

    # -- Cluster registry --
    #
    # Stored as { node_id: {identity..., last_heartbeat_at, snapshot} } in a
    # single JSON file. For a real multi-node JSON deployment this file must live
    # on shared network storage; on a single node it simply stays one entry.

    def _read_cluster(self) -> dict:
        try:
            with open(self._cluster_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def register_node(self, node: dict, snapshot: dict | None = None) -> None:
        with self._cluster_lock:
            nodes = self._read_cluster()
            entry = nodes.get(node["node_id"], {})
            entry.update({
                "node_id": node["node_id"],
                "node_name": node.get("node_name", ""),
                "advertise_url": node.get("advertise_url", ""),
                "vendor": node.get("vendor", ""),
                "llama_image": node.get("llama_image", ""),
                "last_heartbeat_at": now_iso(),
            })
            if snapshot is not None:
                entry["snapshot"] = snapshot
            entry.setdefault("snapshot", {})
            nodes[node["node_id"]] = entry
            _atomic_write_json(self._cluster_file, nodes)

    def list_nodes(self) -> list[dict]:
        return list(self._read_cluster().values())

    def get_node(self, node_id: str) -> dict | None:
        return self._read_cluster().get(node_id)

    def remove_node(self, node_id: str) -> None:
        with self._cluster_lock:
            nodes = self._read_cluster()
            if node_id in nodes:
                del nodes[node_id]
                _atomic_write_json(self._cluster_file, nodes)

    # -- Request Log --

    def append_request_log(self, record: dict, mode: str) -> None:
        conv_id = record.get("conversation_id")
        created_at = record.get("created_at")
        if not conv_id or not created_at:
            logger.warning("request_log append skipped: missing conversation_id or created_at")
            return

        # Normalize to a datetime for path derivation, then to ISO for storage.
        if isinstance(created_at, datetime):
            dt = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        elif isinstance(created_at, (int, float)):
            # Legacy: producer fed us epoch ms.
            dt = datetime.fromtimestamp(int(created_at) / 1000, tz=timezone.utc)
        else:
            dt = parse_iso(str(created_at))
        record = {**record, "created_at": to_iso(dt)}

        date = dt.strftime("%Y-%m-%d")
        date_dir = os.path.join(self._recordings_dir, date)

        if mode == "per_request":
            # Unique filename per record; sortable lexicographically inside the
            # date dir. Colons are filename-unsafe on Windows so use dashes.
            time_part = dt.strftime("%H-%M-%S-") + f"{dt.microsecond // 1000:03d}Z"
            filename = f"{time_part}_{conv_id[:8]}.jsonl"
        else:
            # per_conversation: append to the conversation's file.
            filename = f"{conv_id}.jsonl"

        path = os.path.join(date_dir, filename)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"

        with self._request_log_lock:
            try:
                os.makedirs(date_dir, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.warning("request_log append failed (%s): %s", path, e)

    @staticmethod
    def _normalize_created_at(rec: dict) -> dict:
        """Return rec with created_at as an ISO string. Legacy records may
        carry an int (epoch ms); convert in-place at the read boundary so
        callers don't need to know.
        """
        c = rec.get("created_at")
        if isinstance(c, (int, float)):
            rec["created_at"] = epoch_ms_to_iso(c)
        return rec

    def _iter_request_log_records(self):
        """Yield every record dict across all date dirs, with (date_dir, filename, line_no)."""
        if not os.path.isdir(self._recordings_dir):
            return
        for date in sorted(os.listdir(self._recordings_dir), reverse=True):
            date_dir = os.path.join(self._recordings_dir, date)
            if not os.path.isdir(date_dir):
                continue
            for fn in os.listdir(date_dir):
                if not fn.endswith(".jsonl"):
                    continue
                full = os.path.join(date_dir, fn)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            yield self._normalize_created_at(rec), full
                except OSError:
                    continue

    def list_conversations(self, limit: int = 100) -> list[dict]:
        rollup: dict[str, dict] = {}
        for rec, _ in self._iter_request_log_records():
            cid = rec.get("conversation_id")
            if not cid:
                continue
            created = rec.get("created_at") or ""
            entry = rollup.get(cid)
            if entry is None:
                entry = {
                    "conversation_id": cid,
                    "model": rec.get("model", ""),
                    "first_seen_at": created,
                    "last_seen_at": created,
                    "turn_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "title": "",
                }
                rollup[cid] = entry
            entry["turn_count"] += 1
            pt = rec.get("prompt_tokens")
            ct = rec.get("completion_tokens")
            if isinstance(pt, int):
                entry["prompt_tokens"] += pt
            if isinstance(ct, int):
                entry["completion_tokens"] += ct
            # ISO 8601 UTC strings sort lexicographically the same way they
            # sort chronologically, so plain string compare is correct here.
            if not entry["first_seen_at"] or (created and created < entry["first_seen_at"]):
                entry["first_seen_at"] = created
            if created and created > entry["last_seen_at"]:
                entry["last_seen_at"] = created
            # Title = first user message from the first (earliest) request body
            if not entry["title"]:
                try:
                    req = json.loads(rec.get("request_body") or "{}")
                    msgs = req.get("messages") or []
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "user":
                            content = m.get("content", "")
                            if isinstance(content, str) and content:
                                entry["title"] = content[:120]
                                break
                    if not entry["title"]:
                        prompt = req.get("prompt")
                        if isinstance(prompt, str):
                            entry["title"] = prompt[:120]
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        out = sorted(rollup.values(), key=lambda e: e["last_seen_at"] or "", reverse=True)
        return out[:limit]

    def get_conversation_turns(self, conversation_id: str) -> list[dict]:
        turns = []
        for rec, _ in self._iter_request_log_records():
            if rec.get("conversation_id") == conversation_id:
                turns.append(rec)
        turns.sort(key=lambda r: r.get("created_at") or "")
        return turns

    def request_log_stats(self, inst_id: str | None = None,
                          since=None) -> dict:
        since_iso = None
        if isinstance(since, datetime):
            dt = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            since_iso = to_iso(dt)
        elif isinstance(since, str) and since:
            since_iso = to_iso(parse_iso(since))

        turn_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        tps_sum = 0.0
        tps_n = 0
        tps_max = None
        ttft_sum = 0.0
        ttft_n = 0
        dur_sum = 0
        dur_n = 0
        error_count = 0
        streamed_count = 0
        first_seen = None
        last_seen = None

        for rec, _ in self._iter_request_log_records():
            if inst_id is not None and rec.get("inst_id") != inst_id:
                continue
            created = rec.get("created_at") or ""
            # ISO 8601 UTC sorts lexicographically == chronologically.
            if since_iso and created and created < since_iso:
                continue
            turn_count += 1
            pt = rec.get("prompt_tokens")
            ct = rec.get("completion_tokens")
            if isinstance(pt, int):
                prompt_tokens += pt
            if isinstance(ct, int):
                completion_tokens += ct
            tps = rec.get("tokens_per_sec")
            if isinstance(tps, (int, float)):
                tps_sum += tps
                tps_n += 1
                if tps_max is None or tps > tps_max:
                    tps_max = tps
            ttft = rec.get("ttft_ms")
            if isinstance(ttft, (int, float)):
                ttft_sum += ttft
                ttft_n += 1
            dur = rec.get("duration_ms")
            if isinstance(dur, int):
                dur_sum += dur
                dur_n += 1
            status = rec.get("status_code")
            if isinstance(status, int) and status >= 400:
                error_count += 1
            if rec.get("streamed"):
                streamed_count += 1
            if created:
                if first_seen is None or created < first_seen:
                    first_seen = created
                if last_seen is None or created > last_seen:
                    last_seen = created

        return {
            "turn_count": turn_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_tokens_per_sec": round(tps_sum / tps_n, 2) if tps_n else None,
            "max_tokens_per_sec": round(tps_max, 2) if tps_max is not None else None,
            "avg_ttft_ms": round(ttft_sum / ttft_n, 1) if ttft_n else None,
            "avg_duration_ms": round(dur_sum / dur_n, 1) if dur_n else None,
            "error_count": error_count,
            "streamed_count": streamed_count,
            "first_seen_at": first_seen,
            "last_seen_at": last_seen,
        }

    def prune_request_log(self, older_than) -> int:
        if not os.path.isdir(self._recordings_dir):
            return 0
        pruned = 0
        if isinstance(older_than, str):
            cutoff_dt = parse_iso(older_than)
        elif isinstance(older_than, datetime):
            cutoff_dt = older_than if older_than.tzinfo else older_than.replace(tzinfo=timezone.utc)
        else:
            raise TypeError(f"prune_request_log expects datetime or ISO str, got {type(older_than)}")
        cutoff_date = cutoff_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        for date in os.listdir(self._recordings_dir):
            date_dir = os.path.join(self._recordings_dir, date)
            if not os.path.isdir(date_dir):
                continue
            if date >= cutoff_date:
                continue
            # Whole day is older than cutoff - nuke it
            try:
                for fn in os.listdir(date_dir):
                    full = os.path.join(date_dir, fn)
                    # Rough count: one turn per line
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            pruned += sum(1 for _ in f)
                    except OSError:
                        pass
                    try:
                        os.unlink(full)
                    except OSError:
                        pass
                os.rmdir(date_dir)
            except OSError as e:
                logger.warning("failed to prune %s: %s", date_dir, e)
        return pruned
