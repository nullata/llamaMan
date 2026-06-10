# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Per-node settings.

Some settings must not be shared across a cluster even though the storage
backend (a shared MariaDB) is: Docker images differ per host (CUDA vs ROCm),
and model-cap eviction policy is a per-node capacity decision. These live under
``settings["nodes"][node_id]`` instead of at the top level.

Reads fall back to the legacy top-level value when a node-scoped value is
absent, so existing single-node installs keep working with zero migration: the
old top-level value is used until the node writes its own.
"""

from core.cluster import get_node_id
from storage import get_storage

# Settings that are scoped per node rather than shared cluster-wide.
NODE_SCOPED_KEYS = (
    "docker_images",
    "admin_ui_enforce_max_models",
    "allow_ollama_api_override_admin",
)


def get_node_settings(node_id: str | None = None) -> dict:
    node_id = node_id or get_node_id()
    return dict((get_storage().get_settings().get("nodes") or {}).get(node_id, {}))


def effective_from_settings(settings: dict, key, default=None):
    """Resolve a possibly-node-scoped setting from an already-loaded settings
    dict. Callers that hold their own (possibly test-patched) storage use this
    so the lookup honors their storage instead of the module's."""
    node = (settings.get("nodes") or {}).get(get_node_id(), {})
    if key in node:
        return node[key]
    if key in settings:
        return settings[key]
    return default


def get_effective_setting(key, default=None):
    """Node-scoped value if present, else the legacy top-level value, else default."""
    return effective_from_settings(get_storage().get_settings(), key, default)


def merge_node_settings(patch: dict, node_id: str | None = None) -> dict:
    """Deep-merge a patch into this node's namespace and return the full settings."""
    node_id = node_id or get_node_id()
    return get_storage().merge_settings({"nodes": {node_id: patch}})
