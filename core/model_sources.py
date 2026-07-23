# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Where a model file came from, and what its bytes are.

Two storage locations, deliberately:

* The node-scoped `model_files` table is authoritative. These facts describe a
  physical file on one node's disk - two nodes can hold different files at the
  same path, and a content hash is meaningless without knowing whose disk.
* The legacy `model_sources` settings key is read-only legacy. Migration 004
  copies this node's entries out of it; nothing is written back to it.

The blob is still *read* as a fallback, and that is not vestigial. Migration 004
can only adopt paths that exist on disk when it runs, so a node that booted
before its models volume was mounted would adopt nothing - the fallback is what
keeps that node working instead of silently losing every repo_id. It cannot
reintroduce cross-node contamination, because a path this node doesn't have
never appears in its model list to be resolved in the first place.

The blob is deliberately never deleted: it is the rollback path to the previous
build. Dropping it is a separate, later release - and per the per-node schema
version (StorageBackend.SCHEMA_VERSION_BY_NODE_KEY), any cleanup must be
something every node does for itself, not a one-shot migration.
"""

import os

from storage import get_storage

MODEL_SOURCES_SETTINGS_KEY = "model_sources"


def normalize_model_source_path(path: str) -> str:
    if not path:
        return ""
    try:
        return os.path.realpath(path)
    except Exception:
        return path


def _local_node_id() -> str:
    # Lazy: core.cluster imports storage, which imports config - hoisting this
    # to module scope reintroduces a startup cycle.
    try:
        from core.cluster import get_node_id
        return get_node_id()
    except Exception:
        return ""


def _node_model_files() -> dict[str, dict]:
    """This node's rows, keyed by normalized path. Empty on any storage error,
    so a lookup degrades to the legacy blob rather than failing a request."""
    node_id = _local_node_id()
    if not node_id:
        return {}
    try:
        rows = get_storage().get_model_files(node_id) or {}
    except Exception:
        return {}
    return {normalize_model_source_path(p): meta for p, meta in rows.items()}


def _legacy_sources(settings: dict | None = None) -> dict[str, dict]:
    if settings is None:
        settings = get_storage().get_settings()
    raw_sources = settings.get(MODEL_SOURCES_SETTINGS_KEY, {})
    if not isinstance(raw_sources, dict):
        return {}

    out: dict[str, dict] = {}
    for raw_path, raw_meta in raw_sources.items():
        path = normalize_model_source_path(raw_path)
        if not path:
            continue
        if isinstance(raw_meta, dict):
            out[path] = dict(raw_meta)
        elif isinstance(raw_meta, str):
            out[path] = {"repo_id": raw_meta.strip()}
    return out


def adopt_legacy_model_sources() -> int:
    """Copy this node's entries out of the legacy settings blob into its own
    `model_files` rows. Returns how many were adopted. Run by migration 004.

    A node claims ONLY paths that exist on its own disk. The blob is shared and
    keyed by path with no node attribution, so there is no other way to tell
    whose entry is whose - and without this rule the first node to migrate would
    absorb every other node's records as its own.

    Nothing is deleted from the blob. It stays as inert bytes so a rollback to
    the previous build still finds its data, and so a node that has not upgraded
    yet keeps working exactly as before.
    """
    node_id = _local_node_id()
    if not node_id:
        return 0

    storage = get_storage()
    try:
        legacy = _legacy_sources(storage.get_settings())
    except Exception:
        return 0
    if not legacy:
        return 0

    existing = {}
    try:
        existing = storage.get_model_files(node_id) or {}
    except Exception:
        pass

    adopted = 0
    for path, meta in legacy.items():
        if path in existing:
            continue  # already ours; never overwrite a live row with stale data
        try:
            if not os.path.exists(path):
                continue  # another node's file, or long since deleted
        except Exception:
            continue
        repo_id = str(meta.get("repo_id", "") or "").strip()
        sha256 = str(meta.get("sha256", "") or "").strip().lower()
        if not repo_id and not sha256:
            continue
        try:
            storage.upsert_model_file(node_id, path, repo_id=repo_id, sha256=sha256)
            adopted += 1
        except Exception:
            continue
    return adopted


def get_model_sources(settings: dict | None = None) -> dict[str, str]:
    """path -> repo_id for this node. Legacy blob first, node rows override."""
    merged = _legacy_sources(settings)
    for path, meta in _node_model_files().items():
        if meta.get("repo_id"):
            merged.setdefault(path, {})
            merged[path] = {**merged.get(path, {}), "repo_id": meta["repo_id"]}

    sources: dict[str, str] = {}
    for path, meta in merged.items():
        repo_id = str(meta.get("repo_id", "") or "").strip()
        if repo_id:
            sources[path] = repo_id
    return sources


def record_model_source(download_root_path: str, repo_id: str, model_path: str = "") -> None:
    """Record where a downloaded model came from, on this node.

    Writes only the node-scoped store. The legacy settings blob is no longer
    written: migration 004 copied this node's entries out of it, and continuing
    to write a single shared row per download is what made two nodes able to
    disagree about the same path in the first place.

    Consequence, accepted deliberately: a peer still running a pre-004 build
    stops seeing models downloaded here after this point, because it only reads
    the blob. Its ghost-download list goes stale until it upgrades. Everything
    recorded before the upgrade is still there - nothing was removed.
    """
    repo_id = repo_id.strip()
    node_id = _local_node_id()
    if not repo_id or not node_id:
        return

    paths = {
        normalize_model_source_path(download_root_path),
        normalize_model_source_path(model_path),
    }
    for path in paths:
        if path:
            get_storage().upsert_model_file(node_id, path, repo_id=repo_id)


def get_model_source_meta(settings: dict | None = None) -> dict[str, dict]:
    """Full per-path metadata (repo_id, sha256, ...) for this node, unlike
    get_model_sources() which flattens to path -> repo_id."""
    meta = _legacy_sources(settings)
    for path, row in _node_model_files().items():
        merged = dict(meta.get(path, {}))
        for field in ("repo_id", "sha256"):
            if row.get(field):
                merged[field] = row[field]
        meta[path] = merged
    return meta


def record_model_sha(model_path: str, sha256: str) -> None:
    """Stamp the content hash a model file currently has on THIS node.

    Recorded so a later update check is an exact hash comparison instead of a
    size heuristic, without ever having to re-read the model off disk.

    Node-scoped only - never written to the shared settings blob. A hash
    describes bytes on one disk, and the blob is a single row updated by
    read-modify-write, so concurrent stamps from several nodes could lose
    writes. Old nodes don't read hashes at all, so nothing needs it there.
    """
    sha256 = (sha256 or "").strip().lower()
    normalized = normalize_model_source_path(model_path)
    if not sha256 or not normalized:
        return

    node_id = _local_node_id()
    if not node_id:
        return
    get_storage().upsert_model_file(node_id, normalized, sha256=sha256)


def get_model_sha(model_path: str, settings: dict | None = None) -> str:
    """The stored content hash for `model_path` on this node, or "" if none
    (models that predate this, or were placed on disk by hand)."""
    normalized = normalize_model_source_path(model_path)
    if not normalized:
        return ""
    row = _node_model_files().get(normalized) or {}
    sha = str(row.get("sha256", "") or "").strip().lower()
    if sha:
        return sha
    # Fallback: hashes written by this feature's first iteration landed in the
    # settings blob. Read them so an upgrade doesn't force a re-hash.
    entry = _legacy_sources(settings).get(normalized) or {}
    return str(entry.get("sha256", "") or "").strip().lower()


def resolve_model_source_repo_id(model_path: str, sources: dict[str, str]) -> str:
    normalized_path = normalize_model_source_path(model_path)
    if not normalized_path:
        return ""

    best_match = ""
    for source_path, repo_id in sources.items():
        if normalized_path == source_path or normalized_path.startswith(source_path + os.sep):
            if len(source_path) > len(best_match):
                best_match = source_path

    return sources.get(best_match, "")


def remove_model_sources_for_path(model_path: str) -> None:
    normalized_path = normalize_model_source_path(model_path)
    if not normalized_path:
        return

    storage = get_storage()

    # Drop this node's rows first. Only ours: the same path on another node is a
    # different file and its owner deletes its own row when that file goes.
    node_id = _local_node_id()
    if node_id:
        try:
            storage.delete_model_files(node_id, normalized_path)
        except Exception:
            pass  # legacy cleanup below still runs

    settings = storage.get_settings()
    raw_sources = settings.get(MODEL_SOURCES_SETTINGS_KEY, {})
    if not isinstance(raw_sources, dict):
        return

    kept_sources = {}
    changed = False
    for raw_path, meta in raw_sources.items():
        source_path = normalize_model_source_path(raw_path)
        if source_path == normalized_path or source_path.startswith(normalized_path + os.sep):
            changed = True
            continue
        kept_sources[raw_path] = meta

    if not changed:
        return

    settings = dict(settings)
    settings[MODEL_SOURCES_SETTINGS_KEY] = kept_sources
    storage.save_settings(settings)
