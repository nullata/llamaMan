# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

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


def get_model_sources(settings: dict | None = None) -> dict[str, str]:
    if settings is None:
        settings = get_storage().get_settings()

    raw_sources = settings.get(MODEL_SOURCES_SETTINGS_KEY, {})
    if not isinstance(raw_sources, dict):
        return {}

    sources: dict[str, str] = {}
    for raw_path, raw_meta in raw_sources.items():
        path = normalize_model_source_path(raw_path)
        if not path:
            continue

        repo_id = ""
        if isinstance(raw_meta, dict):
            repo_id = str(raw_meta.get("repo_id", "")).strip()
        elif isinstance(raw_meta, str):
            repo_id = raw_meta.strip()

        if repo_id:
            sources[path] = repo_id

    return sources


def record_model_source(download_root_path: str, repo_id: str, model_path: str = "") -> None:
    repo_id = repo_id.strip()
    if not repo_id:
        return

    patch: dict[str, dict[str, str]] = {
        MODEL_SOURCES_SETTINGS_KEY: {
            normalize_model_source_path(download_root_path): {"repo_id": repo_id},
        }
    }

    normalized_model_path = normalize_model_source_path(model_path)
    if normalized_model_path:
        patch[MODEL_SOURCES_SETTINGS_KEY][normalized_model_path] = {"repo_id": repo_id}

    get_storage().merge_settings(patch)


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
