# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import contextlib
import hashlib
import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, Text, func,
    BigInteger, SmallInteger, DateTime, text, case,
)
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME, MEDIUMTEXT
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from core.timeutil import now_utc, to_iso, parse_iso
from storage.base import (
    KBNotSupportedError, KBUnavailableError, StorageBackend,
)

logger = logging.getLogger("llamaman")

Base = declarative_base()


def _merge_dicts(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class InstanceRow(Base):
    __tablename__ = "instances"
    id = Column(String(64), primary_key=True)
    node_id = Column(String(64), index=True, nullable=True)
    data = Column(Text, default="{}")


class DownloadRow(Base):
    __tablename__ = "downloads"
    id = Column(String(64), primary_key=True)
    node_id = Column(String(64), index=True, nullable=True)
    data = Column(Text, default="{}")


class PresetRow(Base):
    __tablename__ = "presets"
    model_path = Column(String(768), primary_key=True)
    data = Column(Text, default="{}")


class ModelFileRow(Base):
    # New table - create_all() creates it on both fresh and existing installs,
    # on every node at its own startup, so no schema migration is needed. That
    # matters for a shared database: schema_version is a single cluster-wide
    # value, so a migration run by the first node to upgrade is skipped by every
    # node after it. Anything per-node must therefore NOT live in a migration.
    #
    # Column widths are constrained by InnoDB's 3072-byte index key limit
    # (DYNAMIC row format), and the composite primary key must fit inside it:
    # (64 + 700) chars * 4 bytes for utf8mb4 = 3056. Widening either column
    # past that fails at CREATE TABLE. PresetRow's String(768) PK already sits
    # exactly at the limit, so this deployment is necessarily DYNAMIC.
    __tablename__ = "model_files"
    node_id = Column(String(64), primary_key=True)
    model_path = Column(String(700), primary_key=True)
    repo_id = Column(String(255), default="")
    sha256 = Column(String(64), default="")


class UserRow(Base):
    __tablename__ = "users"
    username = Column(String(255), primary_key=True)
    password_hash = Column(String(255), nullable=False)


class SettingsRow(Base):
    __tablename__ = "settings"
    key = Column(String(64), primary_key=True)
    data = Column(Text, default="{}")


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    id = Column(String(32), primary_key=True)
    name = Column(String(255), default="")
    key_hash = Column(String(64), nullable=False)
    prefix = Column(String(16), default="")
    created_at = Column(DateTime, nullable=True)


class RequestLogRow(Base):
    __tablename__ = "request_log"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(String(32), index=True)
    inst_id = Column(String(64), index=True, nullable=True)
    model = Column(String(255), index=True, default="")
    endpoint = Column(String(32), default="")
    path = Column(String(128), default="")
    created_at = Column(MYSQL_DATETIME(fsp=3), index=True)
    duration_ms = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    status_code = Column(SmallInteger, nullable=True)
    streamed = Column(Boolean, default=False)
    tokens_per_sec = Column(Float, nullable=True)
    ttft_ms = Column(Float, nullable=True)
    request_body = Column(MEDIUMTEXT, default="")
    response_body = Column(MEDIUMTEXT, nullable=True)


class ClusterNodeRow(Base):
    # New table - create_all() creates it on both fresh and existing installs,
    # so no schema migration is needed for clustering's foundation.
    __tablename__ = "cluster_nodes"
    node_id = Column(String(64), primary_key=True)
    node_name = Column(String(255), default="")
    advertise_url = Column(String(512), default="")
    vendor = Column(String(32), default="")
    llama_image = Column(String(255), default="")
    last_heartbeat_at = Column(MYSQL_DATETIME(fsp=3), nullable=True)
    snapshot = Column(MEDIUMTEXT, default="{}")


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class MariaDBBackend(StorageBackend):
    """Stores data in MariaDB/MySQL via SQLAlchemy."""

    def __init__(self, database_url: str):
        # Pool sizing is load-bearing, not tuning. The single gunicorn worker
        # runs many threads (gunicorn.conf.py: threads=32) and each thread
        # holds one connection for the whole duration of its query via the
        # thread-local scoped_session; background daemons (request-log
        # finalize_async, the cluster heartbeat) borrow connections on top of
        # that. SQLAlchemy's default pool (size=5 + overflow=10 = 15 max) is
        # far below the thread count, so under concurrent load threads block on
        # checkout and eventually raise "QueuePool limit ... reached" - which
        # surfaces as 500s on /api/request-log/stats and dropped heartbeats.
        # Size the pool to cover the worker's threads plus daemon headroom.
        pool_size = int(os.environ.get("DB_POOL_SIZE", "32"))
        max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "16"))
        pool_timeout = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
        self._engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=3600,   # refresh conns before MySQL wait_timeout drops them
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = scoped_session(sessionmaker(bind=self._engine))
        # Knowledge base state (see kb_* methods at the bottom of this class).
        self._kb_ready = False           # ensure_kb_tables() succeeded
        self._kb_version = None          # cached (major, minor); None = unknown
        logger.info("MariaDB backend connected: %s (pool=%d+%d)",
                    database_url.split("@")[-1], pool_size, max_overflow)

    def _session(self):
        return self._session_factory()

    # -- Migrations --

    @contextlib.contextmanager
    def migration_lock(self):
        """Server-side advisory lock so concurrent gunicorn workers don't both
        run migrations. The lock is released automatically on connection close
        but we explicitly release as well to be safe.
        """
        conn = self._engine.connect()
        try:
            got = conn.execute(text("SELECT GET_LOCK('llamaman_migration', 60)")).scalar()
            if not got:
                raise RuntimeError("Could not acquire MariaDB migration advisory lock within 60s")
            try:
                yield
            finally:
                conn.execute(text("SELECT RELEASE_LOCK('llamaman_migration')"))
        finally:
            conn.close()

    def _column_type(self, table: str, column: str) -> str | None:
        """Return the lowercased DATA_TYPE of a column, or None if missing."""
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c"
            ), {"t": table, "c": column}).first()
        return row[0].lower() if row else None

    def apply_migration_001_timestamps(self) -> None:
        # api_keys.created_at: INT(epoch seconds) -> DATETIME
        if self._column_type("api_keys", "created_at") in ("int", "integer", "bigint"):
            logger.info("Migration 001: converting api_keys.created_at to DATETIME")
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE api_keys ADD COLUMN created_dt DATETIME NULL"))
                conn.execute(text(
                    "UPDATE api_keys SET created_dt = FROM_UNIXTIME(created_at) "
                    "WHERE created_at IS NOT NULL AND created_at > 0"
                ))
                conn.execute(text("ALTER TABLE api_keys DROP COLUMN created_at"))
                conn.execute(text(
                    "ALTER TABLE api_keys CHANGE COLUMN created_dt created_at DATETIME NULL"
                ))

        # request_log.created_at: BIGINT(epoch ms) -> DATETIME(3), batched
        col_type = self._column_type("request_log", "created_at")
        if col_type == "bigint":
            logger.info("Migration 001: converting request_log.created_at to DATETIME(3)")
            with self._engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE request_log ADD COLUMN created_dt DATETIME(3) NULL"
                ))
            # Batch the backfill so very large tables don't blow out memory
            # or hold a single huge transaction.
            batch = 10000
            min_id = 0
            total = 0
            while True:
                with self._engine.begin() as conn:
                    res = conn.execute(text(
                        "UPDATE request_log SET created_dt = FROM_UNIXTIME(created_at/1000) "
                        "WHERE id > :min_id AND id <= :max_id "
                        "AND created_at IS NOT NULL"
                    ), {"min_id": min_id, "max_id": min_id + batch})
                    n = res.rowcount or 0
                    total += n
                if n < batch:
                    # Either we hit the end, or the gap is wider than batch;
                    # bump and re-check so we don't infinite loop on sparse ids.
                    with self._engine.connect() as conn:
                        nxt = conn.execute(text(
                            "SELECT MIN(id) FROM request_log "
                            "WHERE id > :min_id AND created_dt IS NULL "
                            "AND created_at IS NOT NULL"
                        ), {"min_id": min_id + batch}).scalar()
                    if nxt is None:
                        break
                    min_id = int(nxt) - 1
                    continue
                min_id += batch
                if total % 100000 == 0:
                    logger.info("Migration 001: %d request_log rows backfilled", total)
            logger.info("Migration 001: %d request_log rows backfilled total", total)
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE request_log DROP COLUMN created_at"))
                conn.execute(text(
                    "ALTER TABLE request_log CHANGE COLUMN created_dt created_at "
                    "DATETIME(3) NULL"
                ))
                conn.execute(text(
                    "ALTER TABLE request_log ADD INDEX idx_request_log_created_at (created_at)"
                ))

    def apply_migration_002_request_metrics(self) -> None:
        # Add tokens_per_sec / ttft_ms to existing request_log tables.
        # create_all() only creates missing tables, never new columns, so older
        # deployments need this ALTER; fresh ones already have them.
        for col in ("tokens_per_sec", "ttft_ms"):
            if self._column_type("request_log", col) is None:
                logger.info("Migration 002: adding request_log.%s", col)
                with self._engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE request_log ADD COLUMN {col} FLOAT NULL"
                    ))

    def apply_migration_003_node_scoped_state(self) -> None:
        # Add node_id to instances/downloads (create_all can't add columns to
        # existing tables) and adopt pre-cluster rows under the local node id.
        from core.cluster import get_node_id
        local_node = get_node_id()
        for table in ("instances", "downloads"):
            if self._column_type(table, "node_id") is None:
                logger.info("Migration 003: adding %s.node_id", table)
                with self._engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN node_id VARCHAR(64) NULL"
                    ))
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD INDEX idx_{table}_node_id (node_id)"
                    ))
            with self._engine.begin() as conn:
                result = conn.execute(text(
                    f"UPDATE {table} SET node_id = :nid WHERE node_id IS NULL"
                ), {"nid": local_node})
                if result.rowcount:
                    logger.info("Migration 003: adopted %d existing %s row(s) under node %s",
                                result.rowcount, table, local_node[:12])

    # -- State --

    def save_state(self, instances: list[dict], downloads: list[dict],
                   node_id: str | None = None) -> None:
        session = self._session()
        try:
            if node_id is None:
                session.query(InstanceRow).delete()
                session.query(DownloadRow).delete()
            else:
                # Replace only this node's rows (plus any legacy unscoped rows,
                # which belong to the lone pre-cluster writer). Peer rows stay.
                for Row in (InstanceRow, DownloadRow):
                    session.query(Row).filter(
                        (Row.node_id == node_id) | (Row.node_id.is_(None))
                    ).delete(synchronize_session=False)
            for inst in instances:
                session.add(InstanceRow(id=inst["id"], node_id=node_id, data=json.dumps(inst)))
            for dl in downloads:
                session.add(DownloadRow(id=dl["id"], node_id=node_id, data=json.dumps(dl)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def load_instances(self, node_id: str | None = None) -> list[dict]:
        session = self._session()
        try:
            q = session.query(InstanceRow)
            if node_id is not None:
                q = q.filter((InstanceRow.node_id == node_id) | (InstanceRow.node_id.is_(None)))
            return [json.loads(row.data) for row in q.all()]
        finally:
            self._session_factory.remove()

    def load_downloads(self, node_id: str | None = None) -> list[dict]:
        session = self._session()
        try:
            q = session.query(DownloadRow)
            if node_id is not None:
                q = q.filter((DownloadRow.node_id == node_id) | (DownloadRow.node_id.is_(None)))
            return [json.loads(row.data) for row in q.all()]
        finally:
            self._session_factory.remove()

    # -- Presets --

    def get_all_presets(self) -> dict[str, dict]:
        session = self._session()
        try:
            return {row.model_path: json.loads(row.data)
                    for row in session.query(PresetRow).all()}
        finally:
            self._session_factory.remove()

    def get_preset(self, model_path: str) -> dict | None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            return json.loads(row.data) if row else None
        finally:
            self._session_factory.remove()

    def save_preset(self, model_path: str, data: dict) -> None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            if row:
                row.data = json.dumps(data)
            else:
                session.add(PresetRow(model_path=model_path, data=json.dumps(data)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def delete_preset(self, model_path: str) -> None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            if row:
                session.delete(row)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    # -- Auth --

    def get_user(self, username: str) -> dict | None:
        session = self._session()
        try:
            row = session.get(UserRow, username)
            if not row:
                return None
            return {"username": row.username, "password_hash": row.password_hash}
        finally:
            self._session_factory.remove()

    def save_user(self, username: str, password_hash: str) -> None:
        session = self._session()
        try:
            row = session.get(UserRow, username)
            if row:
                row.password_hash = password_hash
            else:
                session.add(UserRow(username=username, password_hash=password_hash))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def user_count(self) -> int:
        session = self._session()
        try:
            return session.query(func.count(UserRow.username)).scalar()
        finally:
            self._session_factory.remove()

    # -- Settings --

    def get_settings(self) -> dict:
        session = self._session()
        try:
            row = session.get(SettingsRow, "global")
            return json.loads(row.data) if row else {}
        finally:
            self._session_factory.remove()

    def save_settings(self, settings: dict) -> None:
        session = self._session()
        try:
            # Locked for the same reason as merge_settings: a wholesale
            # overwrite must not interleave with a concurrent merge.
            row = session.get(SettingsRow, "global", with_for_update=True)
            if row:
                row.data = json.dumps(settings)
            else:
                session.add(SettingsRow(key="global", data=json.dumps(settings)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def merge_settings(self, patch: dict) -> dict:
        """Read-modify-write of the single shared settings row, serialized by a
        row lock.

        Without SELECT ... FOR UPDATE two writers both read the pre-merge value
        and the second commit silently discards the first one's keys. That is
        reachable from one node (32 gunicorn threads) and much more so from
        several nodes sharing one database. The lock is held only for the few
        statements below, so contention is negligible.
        """
        for _attempt in range(2):
            session = self._session()
            try:
                row = session.get(SettingsRow, "global", with_for_update=True)
                if row is None:
                    # No row to lock yet. A concurrent creator makes this insert
                    # fail on the primary key; the retry then finds and locks it.
                    session.add(SettingsRow(key="global", data=json.dumps(patch)))
                    session.commit()
                    return dict(patch)
                current = json.loads(row.data) if row.data else {}
                merged = _merge_dicts(current, patch)
                row.data = json.dumps(merged)
                session.commit()
                return merged
            except IntegrityError:
                session.rollback()
            except Exception:
                session.rollback()
                raise
            finally:
                self._session_factory.remove()
        raise RuntimeError("merge_settings: lost the settings row insert race twice")

    def edit_settings_list(self, key: str, *, add: list[dict] | None = None,
                           remove_ids: list[str] | None = None) -> list[dict]:
        """Same row lock as merge_settings, for the same reason - the read, the
        list edit and the write all happen inside it, so two nodes adding
        entries concurrently can't drop each other's."""
        for _attempt in range(2):
            session = self._session()
            try:
                row = session.get(SettingsRow, "global", with_for_update=True)
                if row is None:
                    updated = self._apply_list_edit(None, add, remove_ids)
                    session.add(SettingsRow(key="global",
                                            data=json.dumps({key: updated})))
                    session.commit()
                    return updated
                current = json.loads(row.data) if row.data else {}
                updated = self._apply_list_edit(current.get(key), add, remove_ids)
                current[key] = updated
                row.data = json.dumps(current)
                session.commit()
                return updated
            except IntegrityError:
                session.rollback()
            except Exception:
                session.rollback()
                raise
            finally:
                self._session_factory.remove()
        raise RuntimeError("edit_settings_list: lost the settings row insert race twice")

    def replace_settings_key(self, key: str, value) -> None:
        for _attempt in range(2):
            session = self._session()
            try:
                row = session.get(SettingsRow, "global", with_for_update=True)
                if row is None:
                    session.add(SettingsRow(key="global",
                                            data=json.dumps({key: value})))
                    session.commit()
                    return
                current = json.loads(row.data) if row.data else {}
                current[key] = value
                row.data = json.dumps(current)
                session.commit()
                return
            except IntegrityError:
                session.rollback()
            except Exception:
                session.rollback()
                raise
            finally:
                self._session_factory.remove()
        raise RuntimeError("replace_settings_key: lost the settings row insert race twice")

    # -- API Keys --

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get_api_keys(self) -> list[dict]:
        session = self._session()
        try:
            return [
                {"id": row.id, "name": row.name, "key_hash": row.key_hash,
                 "prefix": row.prefix,
                 "created_at": to_iso(row.created_at) if row.created_at else None}
                for row in session.query(ApiKeyRow).all()
            ]
        finally:
            self._session_factory.remove()

    def save_api_key(self, key_entry: dict) -> None:
        session = self._session()
        try:
            created = key_entry.get("created_at")
            if isinstance(created, str):
                created_dt = parse_iso(created)
            elif isinstance(created, datetime):
                created_dt = created
            else:
                created_dt = None
            row = session.get(ApiKeyRow, key_entry["id"])
            if row:
                row.name = key_entry.get("name", "")
                row.key_hash = key_entry["key_hash"]
                row.prefix = key_entry.get("prefix", "")
                row.created_at = created_dt
            else:
                session.add(ApiKeyRow(
                    id=key_entry["id"],
                    name=key_entry.get("name", ""),
                    key_hash=key_entry["key_hash"],
                    prefix=key_entry.get("prefix", ""),
                    created_at=created_dt,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def delete_api_key(self, key_id: str) -> None:
        session = self._session()
        try:
            row = session.get(ApiKeyRow, key_id)
            if row:
                session.delete(row)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def verify_api_key(self, raw_key: str) -> bool:
        return self.get_api_key_id(raw_key) is not None

    def get_api_key_id(self, raw_key: str) -> str | None:
        hashed = self._hash_key(raw_key)
        session = self._session()
        try:
            row = session.query(ApiKeyRow).filter_by(key_hash=hashed).first()
            return row.id if row is not None else None
        finally:
            self._session_factory.remove()

    # -- Cluster registry --

    @staticmethod
    def _node_to_dict(row: "ClusterNodeRow", age_us: int | None = None) -> dict:
        d = {
            "node_id": row.node_id,
            "node_name": row.node_name or "",
            "advertise_url": row.advertise_url or "",
            "vendor": row.vendor or "",
            "llama_image": row.llama_image or "",
            "last_heartbeat_at": to_iso(row.last_heartbeat_at) if row.last_heartbeat_at else None,
            "snapshot": json.loads(row.snapshot) if row.snapshot else {},
        }
        # Heartbeat age measured entirely on the DB clock (write and read both use
        # NOW(3)), so it is immune to per-node clock skew - an unsynced peer (e.g.
        # WSL drift) won't be wrongly judged offline by another node's clock.
        if age_us is not None:
            d["heartbeat_age_s"] = age_us / 1_000_000.0
        return d

    # -- Per-node model file metadata --

    def get_model_files(self, node_id: str) -> dict[str, dict]:
        session = self._session()
        try:
            rows = session.query(ModelFileRow).filter(
                ModelFileRow.node_id == node_id).all()
            return {
                r.model_path: {"repo_id": r.repo_id or "", "sha256": r.sha256 or ""}
                for r in rows
            }
        finally:
            self._session_factory.remove()

    def upsert_model_file(self, node_id: str, model_path: str,
                          repo_id: str = "", sha256: str = "") -> None:
        session = self._session()
        try:
            row = session.get(ModelFileRow, (node_id, model_path))
            if row is None:
                row = ModelFileRow(node_id=node_id, model_path=model_path,
                                   repo_id=repo_id or "", sha256=sha256 or "")
                session.add(row)
            else:
                # Blank means "not supplied", so a hash stamp can't wipe repo_id.
                if repo_id:
                    row.repo_id = repo_id
                if sha256:
                    row.sha256 = sha256
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def delete_model_files(self, node_id: str, path_prefix: str) -> None:
        session = self._session()
        try:
            session.query(ModelFileRow).filter(
                ModelFileRow.node_id == node_id,
                (ModelFileRow.model_path == path_prefix)
                | (ModelFileRow.model_path.like(path_prefix.rstrip("/") + "/%")),
            ).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def register_node(self, node: dict, snapshot: dict | None = None) -> None:
        session = self._session()
        try:
            row = session.get(ClusterNodeRow, node["node_id"])
            if row is None:
                row = ClusterNodeRow(node_id=node["node_id"])
                session.add(row)
            row.node_name = node.get("node_name", "")
            row.advertise_url = node.get("advertise_url", "")
            row.vendor = node.get("vendor", "")
            row.llama_image = node.get("llama_image", "")
            # Stamp with the DB server's clock, not this node's - so liveness is
            # judged against a single shared clock and node-to-node skew can't
            # make a healthy node flap offline. Paired with the DB-clock age in
            # list_nodes(). text() keeps NOW(3) a literal (the fsp arg can't be a
            # bound parameter).
            row.last_heartbeat_at = text("NOW(3)")
            if snapshot is not None:
                row.snapshot = json.dumps(snapshot)
            elif row.snapshot is None:
                row.snapshot = "{}"
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def list_nodes(self) -> list[dict]:
        session = self._session()
        try:
            # Compute heartbeat age on the DB clock (TIMESTAMPDIFF vs NOW(3)) so
            # callers get a skew-proof liveness signal.
            age = func.timestampdiff(
                text("MICROSECOND"), ClusterNodeRow.last_heartbeat_at, text("NOW(3)"))
            rows = session.query(ClusterNodeRow, age.label("age_us")).all()
            return [self._node_to_dict(row, age_us) for row, age_us in rows]
        finally:
            self._session_factory.remove()

    def get_node(self, node_id: str) -> dict | None:
        session = self._session()
        try:
            row = session.get(ClusterNodeRow, node_id)
            return self._node_to_dict(row) if row else None
        finally:
            self._session_factory.remove()

    def remove_node(self, node_id: str) -> None:
        session = self._session()
        try:
            row = session.get(ClusterNodeRow, node_id)
            if row:
                session.delete(row)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    # -- Request Log --

    def append_request_log(self, record: dict, mode: str) -> None:
        # mode is ignored: the DB layout is a single table regardless of grouping;
        # the reader does per-conversation rollups via GROUP BY.
        session = self._session()
        try:
            created = record.get("created_at")
            if isinstance(created, str):
                created_dt = parse_iso(created)
            elif isinstance(created, datetime):
                created_dt = created
            else:
                created_dt = None
            row = RequestLogRow(
                conversation_id=record.get("conversation_id"),
                inst_id=record.get("inst_id"),
                model=record.get("model", "") or "",
                endpoint=record.get("endpoint", "") or "",
                path=record.get("path", "") or "",
                created_at=created_dt,
                duration_ms=record.get("duration_ms"),
                prompt_tokens=record.get("prompt_tokens"),
                completion_tokens=record.get("completion_tokens"),
                status_code=record.get("status_code"),
                streamed=bool(record.get("streamed", False)),
                tokens_per_sec=record.get("tokens_per_sec"),
                ttft_ms=record.get("ttft_ms"),
                request_body=record.get("request_body") or "",
                response_body=record.get("response_body"),
            )
            session.add(row)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning("request_log append failed: %s", e)
        finally:
            self._session_factory.remove()

    @staticmethod
    def _extract_title(request_body: str) -> str:
        try:
            req = json.loads(request_body or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            return ""
        if not isinstance(req, dict):
            return ""
        msgs = req.get("messages") or []
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, str) and c:
                        return c[:120]
        prompt = req.get("prompt")
        if isinstance(prompt, str):
            return prompt[:120]
        return ""

    def _conversation_titles(self, cids: list[str]) -> dict[str, str]:
        """Map each conversation id to its title (first user message of the
        earliest turn), resolved in a single query.

        Runs on the caller's thread-local session, so it reuses the already
        checked-out connection rather than taking another from the pool.
        """
        cids = [c for c in cids if c]
        if not cids:
            return {}
        R = RequestLogRow
        session = self._session()
        # Earliest created_at per conversation, then join back to pull that
        # row's request_body. Exact-timestamp ties yield >1 row per id; the
        # first one wins (title is best-effort either way).
        earliest = (
            session.query(R.conversation_id, func.min(R.created_at).label("mc"))
            .filter(R.conversation_id.in_(cids))
            .group_by(R.conversation_id)
            .subquery()
        )
        body_rows = (
            session.query(R.conversation_id, R.request_body)
            .join(earliest, (R.conversation_id == earliest.c.conversation_id)
                  & (R.created_at == earliest.c.mc))
            .all()
        )
        titles: dict[str, str] = {}
        for cid, body in body_rows:
            if cid not in titles:
                titles[cid] = self._extract_title(body or "")
        return titles

    def list_conversations(self, limit: int = 100) -> list[dict]:
        session = self._session()
        try:
            rows = (
                session.query(
                    RequestLogRow.conversation_id,
                    func.min(RequestLogRow.created_at).label("first_seen_at"),
                    func.max(RequestLogRow.created_at).label("last_seen_at"),
                    func.count(RequestLogRow.id).label("turn_count"),
                    func.min(RequestLogRow.model).label("model"),
                    func.sum(RequestLogRow.prompt_tokens).label("prompt_tokens"),
                    func.sum(RequestLogRow.completion_tokens).label("completion_tokens"),
                )
                .group_by(RequestLogRow.conversation_id)
                .order_by(func.max(RequestLogRow.created_at).desc())
                .limit(limit)
                .all()
            )
            # Titles: fetch the earliest row's request_body for every
            # conversation on this page in ONE query instead of a per-row
            # subquery. The old loop issued `limit` extra round trips (up to
            # 200), each holding the pooled connection long enough to starve
            # the pool - the actual cause of the slow/never-loading page.
            titles = self._conversation_titles([r[0] for r in rows])

            out = []
            for cid, first, last, count, model, ptok, ctok in rows:
                out.append({
                    "conversation_id": cid,
                    "model": model or "",
                    "first_seen_at": to_iso(first) if first else None,
                    "last_seen_at": to_iso(last) if last else None,
                    "turn_count": int(count),
                    "prompt_tokens": int(ptok or 0),
                    "completion_tokens": int(ctok or 0),
                    "title": titles.get(cid, ""),
                })
            return out
        finally:
            self._session_factory.remove()

    def get_conversation_turns(self, conversation_id: str) -> list[dict]:
        session = self._session()
        try:
            rows = (
                session.query(RequestLogRow)
                .filter(RequestLogRow.conversation_id == conversation_id)
                .order_by(RequestLogRow.created_at.asc())
                .all()
            )
            return [{
                "id": r.id,
                "conversation_id": r.conversation_id,
                "inst_id": r.inst_id,
                "model": r.model,
                "endpoint": r.endpoint,
                "path": r.path,
                "created_at": to_iso(r.created_at) if r.created_at else None,
                "duration_ms": r.duration_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "status_code": r.status_code,
                "streamed": bool(r.streamed),
                "tokens_per_sec": r.tokens_per_sec,
                "ttft_ms": r.ttft_ms,
                "request_body": r.request_body,
                "response_body": r.response_body,
            } for r in rows]
        finally:
            self._session_factory.remove()

    def request_log_stats(self, inst_id: str | None = None,
                          since=None) -> dict:
        session = self._session()
        try:
            R = RequestLogRow
            q = session.query(
                func.count(R.id),
                func.coalesce(func.sum(R.prompt_tokens), 0),
                func.coalesce(func.sum(R.completion_tokens), 0),
                func.avg(R.tokens_per_sec),
                func.max(R.tokens_per_sec),
                func.avg(R.ttft_ms),
                func.avg(R.duration_ms),
                func.coalesce(func.sum(case((R.status_code >= 400, 1), else_=0)), 0),
                func.coalesce(func.sum(case((R.streamed.is_(True), 1), else_=0)), 0),
                func.min(R.created_at),
                func.max(R.created_at),
            )
            if inst_id is not None:
                q = q.filter(R.inst_id == inst_id)
            if since is not None:
                if isinstance(since, str):
                    cutoff = parse_iso(since).replace(tzinfo=None)
                elif isinstance(since, datetime):
                    cutoff = since.astimezone(timezone.utc).replace(tzinfo=None) \
                        if since.tzinfo else since
                else:
                    cutoff = None
                if cutoff is not None:
                    q = q.filter(R.created_at >= cutoff)
            (count, pt, ct, avg_tps, max_tps, avg_ttft, avg_dur,
             errors, streamed, first, last) = q.one()
            return {
                "turn_count": int(count or 0),
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "avg_tokens_per_sec": round(float(avg_tps), 2) if avg_tps is not None else None,
                "max_tokens_per_sec": round(float(max_tps), 2) if max_tps is not None else None,
                "avg_ttft_ms": round(float(avg_ttft), 1) if avg_ttft is not None else None,
                "avg_duration_ms": round(float(avg_dur), 1) if avg_dur is not None else None,
                "error_count": int(errors or 0),
                "streamed_count": int(streamed or 0),
                "first_seen_at": to_iso(first) if first else None,
                "last_seen_at": to_iso(last) if last else None,
            }
        finally:
            self._session_factory.remove()

    def prune_request_log(self, older_than) -> int:
        if isinstance(older_than, str):
            cutoff = parse_iso(older_than).replace(tzinfo=None)
        elif isinstance(older_than, datetime):
            cutoff = older_than.astimezone(timezone.utc).replace(tzinfo=None) \
                if older_than.tzinfo else older_than
        else:
            raise TypeError(f"prune_request_log expects datetime or ISO str, got {type(older_than)}")
        session = self._session()
        try:
            count = (
                session.query(RequestLogRow)
                .filter(RequestLogRow.created_at < cutoff)
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(count or 0)
        except Exception as e:
            session.rollback()
            logger.warning("request_log prune failed: %s", e)
            return 0
        finally:
            self._session_factory.remove()

    def clear_request_log(self) -> int:
        session = self._session()
        try:
            count = session.query(RequestLogRow).delete(synchronize_session=False)
            session.commit()
            return int(count or 0)
        except Exception as e:
            session.rollback()
            logger.warning("request_log clear failed: %s", e)
            return 0
        finally:
            self._session_factory.remove()

    # ------------------------------------------------------------------
    # Knowledge base (MCP feature). Cluster-wide tables, deliberately NOT
    # node-scoped (like the cluster registry). Raw DDL only: these tables
    # must NEVER be SQLAlchemy models on Base, or the create_all() above
    # would create them on every install regardless of server version or
    # feature enablement (docs/kb-mcp-plan.md §4).
    # ------------------------------------------------------------------

    KB_MIN_SERVER_VERSION = (11, 8)

    _KB_BASE_DDL = (
        """
        CREATE TABLE IF NOT EXISTS kb_topics (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          name VARCHAR(190) NOT NULL,
          owner_key_id VARCHAR(32) NOT NULL DEFAULT '',
          description VARCHAR(500) NOT NULL DEFAULT '',
          created_at DATETIME(3) NOT NULL DEFAULT NOW(3),
          updated_at DATETIME(3) NOT NULL DEFAULT NOW(3) ON UPDATE NOW(3),
          UNIQUE KEY kb_topics_owner_name (owner_key_id, name)
        )
        """,
        # Tables created before per-key ownership: add the owner column and
        # move uniqueness from name alone to (owner, name) so two keys can
        # each have a private topic with the same name. Idempotent.
        "ALTER TABLE kb_topics ADD COLUMN IF NOT EXISTS "
        "owner_key_id VARCHAR(32) NOT NULL DEFAULT '' AFTER name",
        "ALTER TABLE kb_topics DROP INDEX IF EXISTS name",
        "ALTER TABLE kb_topics ADD UNIQUE INDEX IF NOT EXISTS "
        "kb_topics_owner_name (owner_key_id, name)",
        """
        CREATE TABLE IF NOT EXISTS kb_documents (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          topic_id BIGINT NOT NULL,
          title VARCHAR(300) NOT NULL,
          content LONGTEXT NOT NULL,
          sha256 CHAR(64) NOT NULL,
          source VARCHAR(300) NOT NULL DEFAULT '',
          created_at DATETIME(3) NOT NULL DEFAULT NOW(3),
          updated_at DATETIME(3) NOT NULL DEFAULT NOW(3) ON UPDATE NOW(3),
          INDEX kb_documents_topic (topic_id),
          CONSTRAINT kb_documents_topic_fk FOREIGN KEY (topic_id)
            REFERENCES kb_topics(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kb_meta (
          `key` VARCHAR(64) PRIMARY KEY,
          `value` TEXT NOT NULL
        )
        """,
    )

    _KB_MISSING_TABLE_CODES = (1146, 1051)  # table doesn't exist / bad table

    def kb_server_version(self) -> tuple[int, int] | None:
        """(major, minor) of the backing MariaDB, cached. Never raises.

        dialect.server_version_info is only populated after a first connect,
        so force one before reading it.
        """
        if self._kb_version is not None:
            return self._kb_version
        try:
            with self._engine.connect() as conn:
                pass  # forces the dialect handshake
            info = self._engine.dialect.server_version_info
            if info:
                self._kb_version = (int(info[0]), int(info[1]))
        except Exception as e:
            logger.warning("kb: server version probe failed: %s", e)
        return self._kb_version

    def kb_available(self) -> bool:
        ver = self.kb_server_version()
        return ver is not None and ver >= self.KB_MIN_SERVER_VERSION

    def _kb_gate(self) -> None:
        if not self.kb_available():
            ver = ".".join(str(p) for p in (self.kb_server_version() or (0, 0)))
            raise KBUnavailableError(
                f"knowledge base requires MariaDB "
                f"{'.'.join(map(str, self.KB_MIN_SERVER_VERSION))}+ "
                f"(current: {ver})")

    def ensure_kb_tables(self) -> None:
        """Idempotent CREATE TABLE IF NOT EXISTS for the non-dims tables.

        Guarded by _kb_ready so steady-state reads pay no DDL round-trip;
        invalidated only on a caught table-missing error, so an in-place
        MariaDB upgrade self-heals on the next KB action.
        """
        if self._kb_ready:
            return
        self._kb_gate()
        with self._engine.begin() as conn:
            for ddl in self._KB_BASE_DDL:
                conn.execute(text(ddl))
        self._kb_ready = True

    def _kb_reset_ready_on_missing(self, exc: Exception) -> bool:
        """If exc is a missing-table error, drop the ready flag (self-heal)
        and return True so the caller can retry once."""
        orig = getattr(exc, "orig", None)
        code = getattr(orig, "args", (None,))[0] if orig is not None else None
        if code in self._KB_MISSING_TABLE_CODES:
            self._kb_ready = False
            return True
        return False

    def _kb_exec(self, sql, params=None, retry=True):
        """Run one raw KB statement, self-healing once on a missing table."""
        try:
            self.ensure_kb_tables()
            with self._engine.begin() as conn:
                return conn.execute(text(sql), params or {})
        except DBAPIError as e:
            if retry and self._kb_reset_ready_on_missing(e):
                return self._kb_exec(sql, params, retry=False)
            raise

    @staticmethod
    def _vec_text(vec: list[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"

    # -- topics --

    def kb_create_topic(self, name: str, description: str = "",
                        owner_key_id: str = "") -> dict:
        res = self._kb_exec(
            "INSERT INTO kb_topics (name, owner_key_id, description) "
            "VALUES (:n, :o, :d)",
            {"n": name, "o": owner_key_id or "", "d": description})
        return self._kb_topic_row(res.lastrowid)

    def _kb_topic_row(self, topic_id: int) -> dict | None:
        row = self._kb_exec(
            "SELECT id, name, owner_key_id, description, created_at, updated_at "
            "FROM kb_topics WHERE id = :i", {"i": topic_id}).mappings().first()
        if row is None:
            return None
        d = dict(row)
        d["created_at"] = to_iso(d["created_at"])
        d["updated_at"] = to_iso(d["updated_at"])
        return d

    def kb_list_topics(self, visible_to: str | None = None) -> list[dict]:
        sql = ("SELECT t.id, t.name, t.owner_key_id, t.description, "
               "  t.created_at, t.updated_at, "
               "  (SELECT COUNT(*) FROM kb_documents d WHERE d.topic_id = t.id) "
               "    AS document_count "
               "FROM kb_topics t")
        params: dict = {}
        if visible_to is not None:
            # A key sees its own topics plus the shared ('' owner) pool.
            sql += " WHERE t.owner_key_id IN ('', :o)"
            params["o"] = visible_to
        rows = self._kb_exec(sql + " ORDER BY t.name", params).mappings().all()
        out = []
        for r in rows:
            d = dict(r)
            d["created_at"] = to_iso(d["created_at"])
            d["updated_at"] = to_iso(d["updated_at"])
            out.append(d)
        return out

    def kb_update_topic(self, topic_id: int, name: str | None = None,
                        description: str | None = None) -> dict:
        sets, params = [], {"i": topic_id}
        if name is not None:
            sets.append("name = :n"); params["n"] = name
        if description is not None:
            sets.append("description = :d"); params["d"] = description
        if sets:
            self._kb_exec(
                f"UPDATE kb_topics SET {', '.join(sets)} WHERE id = :i", params)
        row = self._kb_topic_row(topic_id)
        if row is None:
            raise LookupError(f"topic {topic_id} not found")
        return row

    def kb_delete_topic(self, topic_id: int) -> None:
        # Cascades to documents and (via their FK) chunks.
        self._kb_exec("DELETE FROM kb_topics WHERE id = :i", {"i": topic_id})

    # -- documents --

    def kb_upsert_document(self, topic_id: int, title: str, content: str,
                           source: str = "") -> dict:
        import hashlib
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.ensure_kb_tables()
        existing = self._kb_exec(
            "SELECT id, sha256 FROM kb_documents WHERE topic_id = :t AND title = :ti",
            {"t": topic_id, "ti": title}).mappings().first()
        # Same content is only "unchanged" if it was actually embedded. The
        # sha is written before the embed runs, so a failed embed (model
        # down, dims mismatch, timeout) leaves the new sha with zero chunks
        # — without this check every retry would be skipped as a no-op.
        if existing is not None and existing["sha256"] == sha \
                and not self._kb_doc_has_chunks(existing["id"]):
            existing = dict(existing, sha256="")
        with self._engine.begin() as conn:
            if existing is None:
                res = conn.execute(text(
                    "INSERT INTO kb_documents (topic_id, title, content, sha256, source) "
                    "VALUES (:t, :ti, :c, :s, :src)"),
                    {"t": topic_id, "ti": title, "c": content, "s": sha,
                     "src": source})
                doc_id = res.lastrowid
                changed = True
            elif existing["sha256"] == sha:
                doc_id = existing["id"]
                changed = False
            else:
                conn.execute(text(
                    "UPDATE kb_documents SET content = :c, sha256 = :s, source = :src "
                    "WHERE id = :i"),
                    {"c": content, "s": sha, "src": source, "i": existing["id"]})
                doc_id = existing["id"]
                changed = True
        if changed:
            # Content changed (or new doc): old chunks are stale. Replace
            # semantics live here so callers can't forget them. Missing
            # kb_chunks simply means no chunks to remove.
            try:
                with self._engine.begin() as conn:
                    conn.execute(text(
                        "DELETE FROM kb_chunks WHERE document_id = :i"),
                        {"i": doc_id})
            except DBAPIError as e:
                if not self._kb_reset_ready_on_missing(e):
                    raise
        return {"id": doc_id, "unchanged": not changed}

    def _kb_doc_has_chunks(self, document_id: int) -> bool:
        try:
            row = self._kb_exec(
                "SELECT 1 FROM kb_chunks WHERE document_id = :i LIMIT 1",
                {"i": document_id}).first()
        except DBAPIError as e:
            # kb_chunks absent = nothing embedded yet.
            if not self._kb_reset_ready_on_missing(e):
                raise
            return False
        return row is not None

    def kb_get_document(self, document_id: int) -> dict | None:
        row = self._kb_exec(
            "SELECT d.id, d.topic_id, d.title, d.content, d.sha256, d.source, "
            "  d.created_at, d.updated_at, t.name AS topic, t.owner_key_id "
            "FROM kb_documents d JOIN kb_topics t ON t.id = d.topic_id "
            "WHERE d.id = :i", {"i": document_id}).mappings().first()
        if row is None:
            return None
        d = dict(row)
        d["created_at"] = to_iso(d["created_at"])
        d["updated_at"] = to_iso(d["updated_at"])
        return d

    def kb_list_documents(self, topic_id: int | None = None,
                          visible_to: str | None = None) -> list[dict]:
        sql = ("SELECT d.id, d.topic_id, d.title, d.sha256, d.source, "
               "  d.created_at, d.updated_at, t.name AS topic, "
               "  (SELECT COUNT(*) FROM kb_chunks c WHERE c.document_id = d.id) "
               "    AS chunk_count, CHAR_LENGTH(d.content) AS content_length, "
               "  t.owner_key_id "
               "FROM kb_documents d JOIN kb_topics t ON t.id = d.topic_id")
        params: dict = {}
        where = []
        if topic_id is not None:
            where.append("d.topic_id = :t")
            params["t"] = topic_id
        if visible_to is not None:
            where.append("t.owner_key_id IN ('', :o)")
            params["o"] = visible_to
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY d.updated_at DESC"
        try:
            rows = self._kb_exec(sql, params).mappings().all()
        except DBAPIError as e:
            # kb_chunks absent just means nothing embedded yet; the count
            # subquery failed against a missing table. Retry without it.
            if not self._kb_reset_ready_on_missing(e):
                raise
            sql = sql.replace(
                "  (SELECT COUNT(*) FROM kb_chunks c WHERE c.document_id = d.id) "
                "    AS chunk_count,", "  0 AS chunk_count,")
            rows = self._kb_exec(sql, params).mappings().all()
        out = []
        for r in rows:
            d = dict(r)
            d["created_at"] = to_iso(d["created_at"])
            d["updated_at"] = to_iso(d["updated_at"])
            out.append(d)
        return out

    def kb_delete_document(self, document_id: int) -> None:
        self._kb_exec("DELETE FROM kb_documents WHERE id = :i", {"i": document_id})

    # -- meta --

    def kb_meta_get(self) -> dict:
        rows = self._kb_exec("SELECT `key`, `value` FROM kb_meta").mappings().all()
        return {r["key"]: r["value"] for r in rows}

    def kb_meta_set(self, **kv) -> None:
        for k, v in kv.items():
            self._kb_exec(
                "INSERT INTO kb_meta (`key`, `value`) VALUES (:k, :v) "
                "ON DUPLICATE KEY UPDATE `value` = :v2",
                {"k": k, "v": str(v), "v2": str(v)})

    # -- chunks (dims-dependent table, created on first embed) --

    def _kb_chunks_column_type(self) -> str | None:
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'kb_chunks' "
                "AND COLUMN_NAME = 'embedding'")).first()
        return row[0].lower() if row else None

    def _kb_ensure_chunks_table(self, dims: int) -> None:
        """Create kb_chunks with VECTOR(dims) if absent; drop/recreate if the
        locked dims changed. The drop/recreate path is guarded by a cluster
        advisory lock: two nodes re-embedding concurrently must not race the
        recreate (docs/kb-mcp-plan.md §4)."""
        current = self._kb_chunks_column_type()
        want = f"vector({dims})"
        if current == want:
            return
        with self._engine.connect() as conn:
            got = conn.execute(text(
                "SELECT GET_LOCK('llamaman_kb_reembed', 60)")).scalar()
            try:
                if not got:
                    raise KBUnavailableError(
                        "kb: another node is rebuilding the embedding table")
                current = self._kb_chunks_column_type()
                if current is not None and current != want:
                    conn.execute(text("DROP TABLE kb_chunks"))
                    current = None
                if current is None:
                    conn.execute(text(f"""
                        CREATE TABLE kb_chunks (
                          id BIGINT AUTO_INCREMENT PRIMARY KEY,
                          document_id BIGINT NOT NULL,
                          seq INT NOT NULL,
                          chunk_text TEXT NOT NULL,
                          embedding VECTOR({int(dims)}) NOT NULL,
                          INDEX kb_chunks_doc (document_id),
                          CONSTRAINT kb_chunks_doc_fk FOREIGN KEY (document_id)
                            REFERENCES kb_documents(id) ON DELETE CASCADE,
                          VECTOR INDEX (embedding) DISTANCE=cosine
                        )"""))
            finally:
                if got:
                    conn.execute(text("SELECT RELEASE_LOCK('llamaman_kb_reembed')"))

    def kb_insert_chunks(self, document_id: int,
                         chunks: list[tuple[int, str, list[float]]]) -> int:
        if not chunks:
            return 0
        self.ensure_kb_tables()
        dims = self.kb_meta_get().get("embedding_dims")
        if not dims:
            raise KBUnavailableError(
                "kb: no embedding model locked yet (kb_meta.embedding_dims)")
        self._kb_ensure_chunks_table(int(dims))
        rows = [{"document_id": document_id, "seq": int(seq), "txt": text,
                 "vec": self._vec_text(vec)} for seq, text, vec in chunks]
        with self._engine.begin() as conn:
            # Replace semantics: stale chunks for this document go first.
            conn.execute(text("DELETE FROM kb_chunks WHERE document_id = :di"),
                         {"di": document_id})
            conn.execute(text(
                "INSERT INTO kb_chunks (document_id, seq, chunk_text, embedding) "
                "VALUES (:document_id, :seq, :txt, VEC_FROMTEXT(:vec))"), rows)
        return len(rows)

    def kb_search(self, query_vec: list[float], topic_id: int | None = None,
                  limit: int = 8, visible_to: str | None = None) -> list[dict]:
        self.ensure_kb_tables()
        if self._kb_chunks_column_type() is None:
            return []  # nothing embedded yet — an honest empty result
        sql = ("SELECT c.id, c.document_id, c.seq, c.chunk_text, "
               "  d.title, t.name AS topic, "
               "  VEC_DISTANCE_COSINE(c.embedding, VEC_FROMTEXT(:qv)) AS dist "
               "FROM kb_chunks c "
               "  JOIN kb_documents d ON d.id = c.document_id "
               "  JOIN kb_topics t ON t.id = d.topic_id")
        params: dict = {"qv": self._vec_text(query_vec), "lim": int(limit)}
        where = []
        if topic_id is not None:
            where.append("d.topic_id = :t")
            params["t"] = topic_id
        if visible_to is not None:
            where.append("t.owner_key_id IN ('', :o)")
            params["o"] = visible_to
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY dist ASC LIMIT :lim"
        rows = self._kb_exec(sql, params).mappings().all()
        return [{"chunk_id": r["id"], "document_id": r["document_id"],
                 "seq": r["seq"], "text": r["chunk_text"], "title": r["title"],
                 "topic": r["topic"], "distance": float(r["dist"])} for r in rows]

    def kb_clear_chunks(self) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(text("DELETE FROM kb_chunks"))
        except DBAPIError as e:
            if not self._kb_reset_ready_on_missing(e):
                raise

    def kb_counts(self) -> dict:
        self.ensure_kb_tables()
        counts = {"topics": 0, "documents": 0, "chunks": 0}
        with self._engine.connect() as conn:
            counts["topics"] = conn.execute(
                text("SELECT COUNT(*) FROM kb_topics")).scalar() or 0
            counts["documents"] = conn.execute(
                text("SELECT COUNT(*) FROM kb_documents")).scalar() or 0
            if self._kb_chunks_column_type() is not None:
                counts["chunks"] = conn.execute(
                    text("SELECT COUNT(*) FROM kb_chunks")).scalar() or 0
        return counts
