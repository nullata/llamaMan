# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Pretty (display) names for models.

A pretty name is a user-chosen label stored on the model's preset, alongside
the existing `favorite` / `note` metadata. It is advertised as the model id on
the Ollama/OpenAI compat surfaces so clients like Open WebUI show something
readable instead of a 60-character quant filename, and it is accepted as an
inbound model name.

It is deliberately NOT a backend identity. The model file path stays the true
identifier and every downstream consumer - cluster group keys, RequestGate,
work-stealing - only ever sees canonical lowercase file stems. Inbound names
are translated to a stem exactly once, at the request boundary, by
resolve_to_stem(). That separation is what keeps pretty names from colliding
with `share_queue_group`: the cluster alias is deliberately *many-to-one* (Q4
on one node + Q8 on another pooling under one key) and matches by fuzzy
substring, so letting a cosmetic label into that keyspace would either match
nothing (silently disabling cross-node routing) or substring-match the wrong
group. Translating at the edge means cluster code never sees a pretty name.

Lookups here are exact and case-insensitive - never substring - so a pretty
name can only ever resolve to the one file it was set on.
"""

import os
import threading
import time

from storage import get_storage

PRETTY_NAME_KEY = "pretty_name"

_CACHE_TTL_SECONDS = 2.0

_index_cache: tuple[dict[str, str], dict[str, str]] | None = None
_index_cache_expires_at: float = 0.0
_index_lock = threading.Lock()


def _normalize(name: str) -> str:
    """Canonical form for comparing names. Mirrors the `name.split(":")[0]`
    tag-stripping the rest of the stack applies to inbound model names, so an
    Open WebUI request for "My Model:latest" still finds "My Model"."""
    return (name or "").split(":")[0].strip().lower()


def _build_index() -> tuple[dict[str, str], dict[str, str]]:
    """(alias_lower -> model_path, model_path -> pretty_name).

    Fail-safe: any storage problem yields an empty index, which makes every
    lookup a miss and leaves behavior exactly as it was before pretty names
    existed. Name resolution must never be the reason a request 500s.
    """
    by_alias: dict[str, str] = {}
    by_path: dict[str, str] = {}
    try:
        presets = get_storage().get_all_presets() or {}
    except Exception:
        return by_alias, by_path

    for path, preset in presets.items():
        if not isinstance(preset, dict):
            continue
        pretty = str(preset.get(PRETTY_NAME_KEY, "") or "").strip()
        if not pretty:
            continue
        by_path[path] = pretty
        key = _normalize(pretty)
        # First writer wins, so a duplicate created by hand-editing storage
        # degrades to "the other one is unreachable by alias" rather than
        # flip-flopping between two models between requests.
        if key and key not in by_alias:
            by_alias[key] = path
    return by_alias, by_path


def _get_index() -> tuple[dict[str, str], dict[str, str]]:
    """Cached alias index. Presets are read on every inbound inference request,
    so this is deliberately short-lived rather than per-request - same pattern
    core.request_log.get_mode() uses. Writers call invalidate() for immediacy."""
    global _index_cache, _index_cache_expires_at
    now = time.monotonic()
    if _index_cache is not None and now < _index_cache_expires_at:
        return _index_cache
    with _index_lock:
        now = time.monotonic()
        if _index_cache is not None and now < _index_cache_expires_at:
            return _index_cache
        _index_cache = _build_index()
        _index_cache_expires_at = now + _CACHE_TTL_SECONDS
        return _index_cache


def invalidate() -> None:
    """Drop the cached index (call after any preset write)."""
    global _index_cache, _index_cache_expires_at
    with _index_lock:
        _index_cache = None
        _index_cache_expires_at = 0.0


def resolve_to_path(name: str) -> str:
    """Model path for an exact pretty-name match, or "" if `name` isn't one."""
    key = _normalize(name)
    if not key:
        return ""
    by_alias, _ = _get_index()
    return by_alias.get(key, "")


def resolve_to_stem(name: str) -> str:
    """Canonical lowercase file stem for an exact pretty-name match, else "".

    This is the boundary translation: callers hand the result to matching logic
    that only understands stems.
    """
    path = resolve_to_path(name)
    if not path:
        return ""
    from core.helpers import model_name_from_path
    return model_name_from_path(path)


def canonical_name(name: str) -> str:
    """`name` translated to a file stem if it's a pretty name, else unchanged.

    Safe to call on any inbound model name, including ones that are already
    stems or `share_queue_group` cluster aliases - those aren't in the alias
    index, so they pass through untouched.
    """
    return resolve_to_stem(name) or name


def pretty_name_for_path(model_path: str) -> str:
    """The pretty name set on `model_path`, or "" if none."""
    if not model_path:
        return ""
    _, by_path = _get_index()
    if model_path in by_path:
        return by_path[model_path]
    # Presets are keyed by the path as supplied at save time; fall back to a
    # realpath comparison so a symlinked models dir still matches.
    try:
        target = os.path.realpath(model_path)
    except Exception:
        return ""
    for path, pretty in by_path.items():
        try:
            if os.path.realpath(path) == target:
                return pretty
        except Exception:
            continue
    return ""


def existing_aliases(exclude_path: str = "") -> dict[str, str]:
    """alias_lower -> model_path for every model that still exists on disk.

    Used for uniqueness validation. Aliases whose file is gone are skipped so a
    preset left behind by a deleted model can't permanently reserve a name.
    """
    by_alias, _ = _get_index()
    live: dict[str, str] = {}
    for key, path in by_alias.items():
        if exclude_path and path == exclude_path:
            continue
        try:
            if os.path.exists(path):
                live[key] = path
        except Exception:
            continue
    return live
