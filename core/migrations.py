# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Versioned schema migrations.

Each migration mutates persistent state (SQL columns, on-disk file formats,
etc.) in an idempotent way and bumps the recorded schema_version when done.
Migrations run on every app startup; already-applied ones are skipped via
the version check inside an advisory lock.

To add a migration:
    1. Implement the work as a method on each storage backend.
    2. Add a wrapper here that dispatches to the active backend.
    3. Append it to MIGRATIONS keyed by its version number.
    4. Bump CURRENT_SCHEMA_VERSION.

Migration code never reverts. If you need to roll back, write migration N+1
that does the reversal.
"""

import logging

logger = logging.getLogger("llamaman")


CURRENT_SCHEMA_VERSION = 3


def _migrate_001_timestamps(storage) -> None:
    """Convert legacy epoch-int timestamps to native datetime / ISO strings."""
    storage.apply_migration_001_timestamps()


def _migrate_002_request_metrics(storage) -> None:
    """Add request_log tokens_per_sec / ttft_ms columns (fixed-schema backends)."""
    storage.apply_migration_002_request_metrics()


def _migrate_003_node_scoped_state(storage) -> None:
    """Add node_id to instances/downloads and adopt existing rows under the
    local node id, so a shared database is safe for multi-node clustering."""
    storage.apply_migration_003_node_scoped_state()


MIGRATIONS = {
    1: _migrate_001_timestamps,
    2: _migrate_002_request_metrics,
    3: _migrate_003_node_scoped_state,
}


def run_pending_migrations(storage) -> None:
    """Run any unapplied migrations under an advisory lock. Aborts startup if
    a migration raises.

    Called once at app boot before any code that reads timestamp-affected
    tables. Multiple gunicorn workers race here, but the storage-level lock
    ensures only one runs the actual migration; the rest re-read the version
    inside the lock and skip.
    """
    current = storage.get_schema_version()
    if current >= CURRENT_SCHEMA_VERSION:
        return

    logger.info(
        "schema_version=%d, target=%d - running pending migrations",
        current, CURRENT_SCHEMA_VERSION,
    )
    with storage.migration_lock():
        # Re-read inside the lock: another worker may have just finished.
        current = storage.get_schema_version()
        for v in sorted(MIGRATIONS):
            if v <= current:
                continue
            logger.info("Migration %d: starting", v)
            MIGRATIONS[v](storage)
            storage.set_schema_version(v)
            logger.info("Migration %d: done", v)
    logger.info("All migrations applied.")
