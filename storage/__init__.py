# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os

from config import STATE_FILE, PRESETS_FILE, USERS_FILE, SETTINGS_FILE, RECORDINGS_DIR, DATA_DIR, logger

_backend = None


def _build_resilient(database_url: str):
    """Wrap MariaDBBackend with a local mirror when the node opts in.

    The wrapper is constructed unconditionally (when DATABASE_URL is set) but
    starts in pass-through mode: with the toggle off it does no mirror I/O, no
    journalling and no fallback, so behaviour is identical to the bare backend.
    That sidesteps a bootstrap ordering problem - the toggle lives in settings,
    which can only be read through storage.
    """
    from core.cluster import get_node_id
    from storage.mariadb_backend import MariaDBBackend
    from storage.resilient import (
        MIRROR_DIRNAME, ResilientBackend, claim_mirror_dir,
    )

    node_id = get_node_id()
    mirror_dir = os.path.join(DATA_DIR, MIRROR_DIRNAME)

    def _builder():
        return MariaDBBackend(database_url)

    primary = None
    build_error = None
    try:
        primary = _builder()
    except Exception as e:
        # create_all() runs in __init__, so an unreachable database means the
        # backend never finishes constructing. Historically that aborted startup
        # outright; with a usable mirror we can boot degraded instead.
        build_error = e

    enabled = False
    if claim_mirror_dir(mirror_dir, node_id):
        from storage.resilient import build_mirror
        source = None
        if primary is not None:
            try:
                source = primary.get_settings()
            except Exception:
                source = None
        if source is None:
            # Primary unreachable: fall back to whatever the mirror last saw.
            try:
                source = build_mirror(mirror_dir).get_settings()
            except Exception:
                source = {}
        backend = ResilientBackend(primary, mirror_dir, node_id=node_id,
                                   builder=_builder, enabled=False)
        backend.resolve_enabled(source or {})
        enabled = backend.mirror_enabled()
        if primary is None and not enabled:
            # No mirror to boot from and no primary - preserve the historical
            # hard failure rather than serving an empty database.
            raise build_error
        if primary is None:
            logger.warning(
                "Database unreachable at startup (%s) - booting from local mirror "
                "at %s. Cluster peers are not visible until it returns.",
                build_error, mirror_dir,
            )
        return backend

    if primary is None:
        raise build_error
    return primary


def get_storage():
    """Return the singleton storage backend.

    Uses MariaDBBackend if DATABASE_URL is set, otherwise JsonBackend.
    """
    global _backend
    if _backend is not None:
        return _backend

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        _backend = _build_resilient(database_url)
    else:
        from storage.json_backend import JsonBackend
        _backend = JsonBackend(
            STATE_FILE, PRESETS_FILE, USERS_FILE, SETTINGS_FILE,
            recordings_dir=RECORDINGS_DIR,
        )

    return _backend
