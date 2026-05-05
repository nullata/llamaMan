# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import contextlib
import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, Text, func,
    BigInteger, SmallInteger, DateTime, text,
)
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME, MEDIUMTEXT
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from core.timeutil import to_iso, parse_iso
from storage.base import StorageBackend

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
    data = Column(Text, default="{}")


class DownloadRow(Base):
    __tablename__ = "downloads"
    id = Column(String(64), primary_key=True)
    data = Column(Text, default="{}")


class PresetRow(Base):
    __tablename__ = "presets"
    model_path = Column(String(768), primary_key=True)
    data = Column(Text, default="{}")


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
    request_body = Column(MEDIUMTEXT, default="")
    response_body = Column(MEDIUMTEXT, nullable=True)


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class MariaDBBackend(StorageBackend):
    """Stores data in MariaDB/MySQL via SQLAlchemy."""

    def __init__(self, database_url: str):
        self._engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = scoped_session(sessionmaker(bind=self._engine))
        logger.info("MariaDB backend connected: %s", database_url.split("@")[-1])

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

    # -- State --

    def save_state(self, instances: list[dict], downloads: list[dict]) -> None:
        session = self._session()
        try:
            session.query(InstanceRow).delete()
            session.query(DownloadRow).delete()
            for inst in instances:
                session.add(InstanceRow(id=inst["id"], data=json.dumps(inst)))
            for dl in downloads:
                session.add(DownloadRow(id=dl["id"], data=json.dumps(dl)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def load_instances(self) -> list[dict]:
        session = self._session()
        try:
            return [json.loads(row.data) for row in session.query(InstanceRow).all()]
        finally:
            self._session_factory.remove()

    def load_downloads(self) -> list[dict]:
        session = self._session()
        try:
            return [json.loads(row.data) for row in session.query(DownloadRow).all()]
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
            row = session.get(SettingsRow, "global")
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
        session = self._session()
        try:
            row = session.get(SettingsRow, "global")
            current = json.loads(row.data) if row else {}
            merged = _merge_dicts(current, patch)
            if row:
                row.data = json.dumps(merged)
            else:
                session.add(SettingsRow(key="global", data=json.dumps(merged)))
            session.commit()
            return merged
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

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
        hashed = self._hash_key(raw_key)
        session = self._session()
        try:
            return session.query(ApiKeyRow).filter_by(key_hash=hashed).first() is not None
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
                )
                .group_by(RequestLogRow.conversation_id)
                .order_by(func.max(RequestLogRow.created_at).desc())
                .limit(limit)
                .all()
            )
            out = []
            for cid, first, last, count, model in rows:
                # Title: pull the earliest row's request_body for this conversation
                first_row = (
                    session.query(RequestLogRow.request_body)
                    .filter(RequestLogRow.conversation_id == cid)
                    .order_by(RequestLogRow.created_at.asc())
                    .limit(1)
                    .scalar()
                )
                out.append({
                    "conversation_id": cid,
                    "model": model or "",
                    "first_seen_at": to_iso(first) if first else None,
                    "last_seen_at": to_iso(last) if last else None,
                    "turn_count": int(count),
                    "title": self._extract_title(first_row or ""),
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
                "request_body": r.request_body,
                "response_body": r.response_body,
            } for r in rows]
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
