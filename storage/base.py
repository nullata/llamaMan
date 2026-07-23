# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract interface for persistent storage.

    Two implementations exist:
      - JsonBackend (default): stores data in JSON files, zero dependencies
      - MariaDBBackend (optional): stores data in MariaDB, enabled via DATABASE_URL
    """

    # -- Instances & Downloads (state) --

    @abstractmethod
    def save_state(self, instances: list[dict], downloads: list[dict],
                   node_id: str | None = None) -> None:
        """Atomically persist both instances and downloads.

        When `node_id` is given, only that node's rows are replaced; rows owned
        by other cluster nodes are left untouched. This keeps a shared database
        (or shared state file) safe for multi-node use - each node manages only
        its own instances/downloads. `node_id=None` replaces everything (legacy
        single-writer behavior).
        """
        ...

    @abstractmethod
    def load_instances(self, node_id: str | None = None) -> list[dict]:
        """Return persisted instances. Scoped to `node_id` when given; legacy
        rows that predate node scoping (no node_id) are treated as belonging to
        the caller so existing single-node installs upgrade seamlessly."""
        ...

    @abstractmethod
    def load_downloads(self, node_id: str | None = None) -> list[dict]:
        ...

    # -- Presets --

    @abstractmethod
    def get_all_presets(self) -> dict[str, dict]:
        ...

    @abstractmethod
    def get_preset(self, model_path: str) -> dict | None:
        ...

    @abstractmethod
    def save_preset(self, model_path: str, data: dict) -> None:
        ...

    @abstractmethod
    def delete_preset(self, model_path: str) -> None:
        ...

    # -- Auth --

    @abstractmethod
    def get_user(self, username: str) -> dict | None:
        """Return user dict with keys: username, password_hash. None if not found."""
        ...

    @abstractmethod
    def save_user(self, username: str, password_hash: str) -> None:
        ...

    @abstractmethod
    def user_count(self) -> int:
        """Return total number of users. Used to detect first-run."""
        ...

    # -- Settings --

    @abstractmethod
    def get_settings(self) -> dict:
        """Return the global settings dict. Returns {} if not set."""
        ...

    @abstractmethod
    def save_settings(self, settings: dict) -> None:
        """Persist the global settings dict."""
        ...

    @abstractmethod
    def merge_settings(self, patch: dict) -> dict:
        """Recursively merge a partial settings patch and return the updated settings."""
        ...

    # -- API Keys --

    @abstractmethod
    def get_api_keys(self) -> list[dict]:
        """Return all API keys. Each dict has: id, name, key_hash, created_at."""
        ...

    @abstractmethod
    def save_api_key(self, key_entry: dict) -> None:
        """Add or update an API key entry."""
        ...

    @abstractmethod
    def delete_api_key(self, key_id: str) -> None:
        """Delete an API key by id."""
        ...

    @abstractmethod
    def verify_api_key(self, raw_key: str) -> bool:
        """Check if a raw bearer token matches any stored key hash."""
        ...

    # -- Per-node model file metadata --
    #
    # Facts about the bytes of a model file on ONE node's disk: which repo it
    # came from and the content hash it was last known to have. Node-scoped
    # because two nodes can hold different files at the same path, and a hash
    # describes a physical file, not a logical model. (The logical, cluster-wide
    # metadata - presets, display names - stays in the shared presets table.)
    #
    # Deliberately a table rather than a key in the settings blob: settings are
    # one shared row updated by read-modify-write, so concurrent stamps from
    # several nodes can lose writes. These are per-row upserts.

    @abstractmethod
    def get_model_files(self, node_id: str) -> dict[str, dict]:
        """model_path -> {"repo_id", "sha256"} for one node. Empty if none."""
        ...

    @abstractmethod
    def upsert_model_file(self, node_id: str, model_path: str,
                          repo_id: str = "", sha256: str = "") -> None:
        """Insert or update one node's record for a model file.

        Only non-empty fields are written, so stamping a hash never clears a
        previously recorded repo_id and vice versa.
        """
        ...

    @abstractmethod
    def delete_model_files(self, node_id: str, path_prefix: str) -> None:
        """Drop a node's records for `path_prefix` and anything beneath it,
        mirroring how deleting a model removes a directory and its contents."""
        ...

    # -- Cluster registry --

    @abstractmethod
    def register_node(self, node: dict, snapshot: dict | None = None) -> None:
        """Insert or update a node in the shared cluster registry and stamp a
        fresh heartbeat.

        `node` carries: node_id, node_name, advertise_url, vendor, llama_image.
        `snapshot` (when given) replaces the node's published metadata blob
        (system info, gpus, instances, downloads, images). Identity-only calls
        (e.g. join) may omit it.
        """
        ...

    @abstractmethod
    def list_nodes(self) -> list[dict]:
        """Return all registered nodes. Each dict has the identity fields plus
        `last_heartbeat_at` (ISO string or None) and `snapshot` (dict)."""
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> dict | None:
        """Return one node dict (as in list_nodes) or None if not registered."""
        ...

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the registry (used on graceful leave)."""
        ...

    # -- Request Log --

    @abstractmethod
    def append_request_log(self, record: dict, mode: str) -> None:
        """Persist one inference turn.

        `record` must contain at minimum `conversation_id` (32-char hex) and
        `created_at` (epoch milliseconds). Other envelope fields (inst_id,
        model, endpoint, path, duration_ms, prompt_tokens, completion_tokens,
        status_code, streamed, request_body, response_body) are optional.

        `mode` is the active recording mode: 'per_request' or 'per_conversation'.
        Backends may use it to shape storage layout; callers pass the setting
        value through so backends remain the sole source of layout knowledge.
        Must never be called with mode 'off'.
        """
        ...

    @abstractmethod
    def list_conversations(self, limit: int = 100) -> list[dict]:
        """Return the most recent conversations with rolled-up metadata.

        Each dict contains: conversation_id, model, first_seen_at (epoch ms),
        last_seen_at (epoch ms), turn_count, title (truncated first user msg).
        Ordered by last_seen_at descending.
        """
        ...

    @abstractmethod
    def get_conversation_turns(self, conversation_id: str) -> list[dict]:
        """Return all recorded turns for a conversation, ordered by created_at.
        Each dict is the full envelope + bodies. Empty list if not found."""
        ...

    @abstractmethod
    def prune_request_log(self, older_than) -> int:
        """Delete records with created_at < older_than (a datetime or ISO string).
        Returns count pruned."""
        ...

    @abstractmethod
    def clear_request_log(self) -> int:
        """Delete every recorded turn. Returns count deleted."""
        ...

    @abstractmethod
    def request_log_stats(self, inst_id: str | None = None,
                          since=None) -> dict:
        """Aggregate recorded turns into rollup metrics for the stats UI.

        Optionally filtered to one `inst_id` and to turns at/after `since`
        (a datetime or ISO string). Returns a dict with: turn_count,
        prompt_tokens, completion_tokens (sums), avg_tokens_per_sec,
        max_tokens_per_sec, avg_ttft_ms, avg_duration_ms, error_count
        (status_code >= 400), streamed_count, first_seen_at, last_seen_at
        (ISO or None). Token-rate fields are None when no turn carried them.
        """
        ...

    # -- Schema migrations --

    # Pre-cluster installs recorded ONE version for the whole database. On a
    # shared database that is wrong: the first node to upgrade runs the
    # migrations and bumps it, so every node upgrading afterwards sees the new
    # number and skips them. Anything a migration does per node was therefore
    # silently never done on nodes 2..N.
    SCHEMA_VERSION_KEY = "_schema_version"                  # legacy, global
    SCHEMA_VERSION_BY_NODE_KEY = "_schema_version_by_node"  # {node_id: version}

    # Migrations up to and including this one predate per-node versioning and
    # only ever did global, shared-database work, so a node joining later can
    # safely inherit the old global number for them. Everything from the next
    # version onward must be run by every node individually - which is exactly
    # why the seed is capped here rather than taking the global value as-is.
    LEGACY_GLOBAL_MAX_VERSION = 3

    def _node_key(self) -> str:
        try:
            from core.cluster import get_node_id
            return get_node_id() or "local"
        except Exception:
            return "local"

    def get_schema_version(self) -> int:
        """Applied schema version for THIS node (0 if it has never migrated)."""
        try:
            settings = self.get_settings()
        except Exception:
            return 0

        by_node = settings.get(self.SCHEMA_VERSION_BY_NODE_KEY, {})
        if isinstance(by_node, dict) and self._node_key() in by_node:
            try:
                return int(by_node[self._node_key()] or 0)
            except (TypeError, ValueError):
                return 0

        # First run under per-node versioning. Inherit the old global value so an
        # existing install doesn't replay migrations it has already applied, but
        # cap it so per-node migrations still run here.
        try:
            legacy = int(settings.get(self.SCHEMA_VERSION_KEY, 0) or 0)
        except (TypeError, ValueError):
            legacy = 0
        return min(legacy, self.LEGACY_GLOBAL_MAX_VERSION)

    def set_schema_version(self, version: int) -> None:
        version = int(version)
        patch = {self.SCHEMA_VERSION_BY_NODE_KEY: {self._node_key(): version}}
        # Keep the legacy global key moving forward too, so a node still on an
        # older build reads a sane number and doesn't replay old migrations.
        if version <= self.LEGACY_GLOBAL_MAX_VERSION:
            patch[self.SCHEMA_VERSION_KEY] = version
        self.merge_settings(patch)

    @abstractmethod
    def migration_lock(self):
        """Context manager preventing concurrent migration runs.

        MariaDB uses a server-side advisory lock so multi-worker setups
        serialize cleanly; the JSON backend uses a lockfile.
        """
        ...

    def apply_migration_001_timestamps(self) -> None:
        """Backend-specific implementation of migration 001.

        Converts legacy epoch timestamps in the request_log and api_keys
        tables (or their JSON-backend equivalents) to native datetime / ISO
        strings. Default is a no-op so backends that don't need it can omit.
        """
        return None

    def apply_migration_002_request_metrics(self) -> None:
        """Backend-specific implementation of migration 002.

        Adds the request_log tokens_per_sec / ttft_ms metric columns where the
        backend uses a fixed schema. The JSON backend stores schema-less
        records so it needs nothing; default is a no-op.
        """
        return None

    def apply_migration_003_node_scoped_state(self) -> None:
        """Backend-specific implementation of migration 003.

        Adds a node_id column to the instances/downloads tables and backfills
        existing rows with the local node id (the only writer before clustering
        existed). The JSON backend infers ownership at the read boundary
        (missing node_id => local), so it needs nothing; default is a no-op.
        """
        return None
