# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata's prepared data for models that are gone.

A Strata setup leaves its own files on the Strata data volume (/data in every
Strata container, HOST_STRATA_DATA_DIR or STRATA_DATA_VOLUME), next to the
GGUF files llamaMan keeps in MODELS_DIR:

- Strata-data/packs/<tag>/          the prepared pack (experts.bin and dense.bin:
                                    tens of GB for a Flash-Next model)
- install/strata-<tag>.json (.bak, .log), run-<tag>.sh   its run config
- files a model family shares: the vision encoder (models/mmproj-...), the MTP
  draft layer (mtp/) and the DFlash2 drafter (dflash2/)

Strata never removes any of them, and deleting the GGUF in llamaMan used not
to either. The run config is the record: its args name the pack, the GGUFs
(at /models/..., llamaMan's bind mount) and the shared files it reads.

Two ways in:
- cleanup_after_delete: after a model is deleted in the library, the setups of
  the models it held (when no other copy of them is left) are removed.
- run(orphans=True): every setup whose model files llamaMan mounted and no
  longer has, and packs with no run config (Settings -> Docker Images).

A shared file goes only when no remaining run config names it; the files a
family shares that no config names (the drafters setup fetches itself) go
with that family's last setup. A setup Strata downloaded the model for itself
(its GGUFs on /data, not /models) is never an orphan.

The volume is not mounted in llamaMan, so the work is done in a throwaway
container of the Strata image (CLEANUP_SCRIPT, stdlib only, with the image's
Python) with the volume at /data, as Strata's own containers see it.
"""

from __future__ import annotations

import json
import os
import threading

# Run inside the Strata image with the data volume at /data. Input (env
# LLAMAMAN_CLEANUP): {"tags": [...], "present": [...], "orphans": bool,
# "apply": bool}. Prints one JSON line: the plan (and what failed, applied).
CLEANUP_SCRIPT = r'''
import fnmatch, json, os, re, shutil

ROOT = os.environ.get("LLAMAMAN_STRATA_ROOT", "/data").rstrip("/") or "/"
MOUNT = "/data"
REQ = json.loads(os.environ.get("LLAMAMAN_CLEANUP") or "{}")
DATA = os.path.join(ROOT, "Strata-data")
INSTALL = os.path.join(ROOT, "install")
PACKS = os.path.join(DATA, "packs")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Files a model family shares that no run config names (setup fetches the
# drafters by itself): they go with the family's last setup. Keyed by the
# start of the configs' model_name (setup.py FAMILIES "name").
FAMILY_ASSETS = {
    "qwen3.8-flash-next": ["mtp", "dflash2/qwen3.8-flash-next-*", "dflash2/draft_vocab*.bin",
                           "models/mmproj-Qwen3.8-Flash-Next-*"],
}
MARKS = (".done", ".ok")


def local(p):
    """A config's /data/... path -> the same path under ROOT; else None."""
    if p == MOUNT or p.startswith(MOUNT + "/"):
        return os.path.normpath(ROOT + p[len(MOUNT):])
    return None


def strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif isinstance(x, str):
        yield x


def listdir(d):
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


def read_configs():
    out = {}
    for n in listdir(INSTALL):
        if not (n.startswith("strata-") and n.endswith(".json")):
            continue
        tag = n[len("strata-"):-len(".json")]
        try:
            with open(os.path.join(INSTALL, n)) as f:
                c = json.load(f)
            ok = isinstance(c, dict)
        except (OSError, ValueError):
            ok = False
        c = c if ok else {}
        refs = set()
        for s in strings(c):
            p = local(s)
            if p and p.startswith(DATA + "/"):
                refs.add(p)
        out[tag] = {"ok": ok, "refs": refs, "name": str(c.get("model_name") or "").lower(),
                    "mounted": any(s.startswith("/models/") for s in strings(c))}
    return out


def inside(p):
    p = os.path.normpath(p)
    return any(p.startswith(base + "/") for base in (DATA, INSTALL))


def size(p):
    try:
        if os.path.islink(p) or not os.path.isdir(p):
            return os.lstat(p).st_size
    except OSError:
        return 0
    total = 0
    for d, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return total


def main():
    configs = read_configs()
    present = {t for t in REQ.get("present") or [] if isinstance(t, str)}
    tags = {t for t in REQ.get("tags") or [] if isinstance(t, str) and TAG_RE.match(t)}
    orphans = bool(REQ.get("orphans"))
    if orphans:
        tags |= {t for t, c in configs.items() if c["mounted"] and t not in present}
    tags -= present
    tags = {t for t in tags if TAG_RE.match(t)}

    items = []

    def add(p):
        p = os.path.normpath(p)
        if inside(p) and p not in items and (os.path.lexists(p)):
            items.append(p)

    def leftovers(t):
        add(os.path.join(PACKS, t))
        for n in listdir(INSTALL):
            if n.startswith("strata-%s." % t) or n.startswith("run-%s." % t):
                add(os.path.join(INSTALL, n))

    for t in sorted(tags):
        leftovers(t)
    if orphans:
        # packs and install files whose run config is gone
        stray = {n for n in listdir(PACKS) if os.path.isdir(os.path.join(PACKS, n))}
        for n in listdir(INSTALL):
            m = re.match(r"^(?:strata|run)-(.+?)\.(?:json\.bak|log|sh|bat)$", n)
            if m:
                stray.add(m.group(1))
        for t in sorted(stray - set(configs) - present):
            if TAG_RE.match(t):
                tags.add(t)
                leftovers(t)

    kept = {t: c for t, c in configs.items() if t not in tags}
    # An unreadable remaining config may name anything: shared files stay.
    if all(c["ok"] for c in kept.values()):
        kept_refs = set().union(*(c["refs"] for c in kept.values())) if kept else set()

        def used(p):
            return any(k == p or k.startswith(p + "/") or p.startswith(k + "/") for k in kept_refs)

        def shared(p):
            # never a top folder (models/, mtp/ itself is a family asset below) or a pack
            if p.startswith(PACKS + "/") or "/" not in os.path.relpath(p, DATA) or used(p):
                return
            add(p)
            for m in MARKS:
                add(p + m)

        for t in sorted(tags):
            for r in sorted(configs.get(t, {}).get("refs", ())):
                shared(r)
        gone = {configs[t]["name"] for t in tags if t in configs}
        left = {c["name"] for c in kept.values()}
        for prefix, patterns in FAMILY_ASSETS.items():
            if not any(n.startswith(prefix) for n in gone) or any(n.startswith(prefix) for n in left):
                continue
            for pat in patterns:
                folder, _, base = pat.rpartition("/")
                parent = os.path.join(DATA, folder) if folder else DATA
                for n in listdir(parent):
                    p = os.path.join(parent, n)
                    if fnmatch.fnmatch(n, base) and not used(p):
                        add(p)

    plan = [{"path": MOUNT + p[len(ROOT):], "bytes": size(p)} for p in items]
    errors = []
    if REQ.get("apply"):
        for p in items:
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except OSError as e:
                errors.append("%s: %s" % (MOUNT + p[len(ROOT):], e.strerror or e))
    print(json.dumps({"items": plan, "bytes": sum(i["bytes"] for i in plan), "tags": sorted(tags),
                      "applied": bool(REQ.get("apply")), "errors": errors}))


main()
'''

_RUN_LOCK = threading.Lock()

_ENTRYPOINT = ['sh', '-c', 'if [ -x /opt/strata/terminal/python/bin/python3 ]; then '
               'exec /opt/strata/terminal/python/bin/python3 -I -c "$LLAMAMAN_SCRIPT"; fi; '
               'exec python3 -I -c "$LLAMAMAN_SCRIPT"']


def model_files_under(path: str) -> list[str]:
    """The model files a library path holds: itself, or the files in a folder."""
    if os.path.isfile(path):
        return [path]
    out = []
    for d, _, files in os.walk(path):
        out += [os.path.join(d, f) for f in files if f.lower().endswith((".gguf", ".strata"))]
    return out


def tags_for_files(paths) -> set[str]:
    """The Strata setup tags of the models these files are (of) - read before
    they are deleted (an unsplit GGUF may be told by its architecture)."""
    from core.engines.strata import model_for_file, setup_tag
    out = set()
    for p in paths:
        try:
            parsed = model_for_file(p)
        except Exception:
            parsed = None
        if parsed:
            out.add(setup_tag(*parsed))
    return out


def present_tags() -> set[str]:
    """Setups still in use: a model in the library is one of them, or an
    instance (any state but stopped) runs one."""
    from config import MODELS_DIR
    from core.engines.strata import parse_model_path, setup_tag
    from core.state import instances, instances_lock
    files = []
    for d, _, names in os.walk(MODELS_DIR):
        if "/.llamaman-partial-" in d + "/":
            continue
        files += [os.path.join(d, n) for n in names if n.lower().endswith((".gguf", ".strata"))]
    tags = tags_for_files(files)
    with instances_lock:
        paths = [inst.get("model_path") for inst in instances.values() if inst.get("status") != "stopped"]
    for p in paths:
        parsed = parse_model_path(p)
        if parsed:
            tags.add(setup_tag(*parsed))
    return tags


def _image() -> str:
    import docker
    from core.engines.strata import image_candidates
    from core.helpers import get_docker_client
    client = get_docker_client()
    for name in image_candidates():
        try:
            client.images.get(name)
            return name
        except docker.errors.ImageNotFound:
            continue
    raise RuntimeError("no Strata image on this node (build or pull one in Settings -> Docker Images)")


def run(tags=(), orphans: bool = False, apply: bool = False) -> dict:
    """Find (and with apply, remove) the Strata data of `tags`, and with
    orphans of every setup whose model is gone. Returns the plan:
    {"items": [{"path", "bytes"}], "bytes", "tags", "applied", "errors"}."""
    from config import STRATA_ENABLED, logger
    from core.engines import ENGINES
    from core.helpers import get_docker_client
    if not STRATA_ENABLED:
        raise RuntimeError("Strata is disabled on this node")
    eng = ENGINES["strata"]
    with _RUN_LOCK:
        present = present_tags()
        request = {"tags": sorted(set(tags) - present), "present": sorted(present),
                   "orphans": bool(orphans), "apply": bool(apply)}
        if not request["tags"] and not orphans:
            return {"items": [], "bytes": 0, "tags": [], "applied": bool(apply), "errors": []}
        out = get_docker_client().containers.run(
            _image(), remove=True, network_disabled=True, stdout=True, stderr=False,
            entrypoint=_ENTRYPOINT,
            environment={"LLAMAMAN_SCRIPT": CLEANUP_SCRIPT, "LLAMAMAN_CLEANUP": json.dumps(request)},
            volumes={eng.data_mount_source(): {"bind": "/data", "mode": "rw"}},
        )
        result = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
        if apply and result.get("tags"):
            eng.forget_setups(result["tags"])
    if apply and result.get("items"):
        logger.info("strata: removed the data of %s (%.1f GB): %s", ", ".join(result["tags"]),
                    result["bytes"] / 1e9, ", ".join(i["path"] for i in result["items"]))
    for e in result.get("errors") or []:
        logger.warning("strata cleanup: could not remove %s", e)
    return result


def cleanup_after_delete(tags: set[str]) -> bool:
    """After a model was deleted from the library: remove the setups of the
    models it held, in the background (a container start takes a moment).
    Tags another copy in the library (or an instance) still uses are kept.
    Returns whether a cleanup was started."""
    from config import STRATA_ENABLED, logger
    if not STRATA_ENABLED or not tags:
        return False

    def work():
        try:
            run(tags, apply=True)
        except Exception as e:
            logger.warning("strata: could not remove the data of %s: %s", ", ".join(sorted(tags)), e)

    threading.Thread(target=work, daemon=True, name="strata-cleanup").start()
    return True

