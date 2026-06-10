# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Cluster coordination API.

Read-only aggregation for Phase 1: each node publishes a metadata snapshot
(system stats, GPUs, instances, downloads) into the shared registry on a
heartbeat, and any node can list the whole cluster. Control actions and
inference dispatch land in later phases.

All endpoints are inert (return enabled=false / empty) when clustering is not
configured, so the dashboard behaves exactly as before on single-node installs.
"""

import os
import threading

from flask import Blueprint, Response, jsonify, request

from config import REQUEST_TIMEOUT, logger
from core import cluster as cl
from core.helpers import model_name_from_path
from core.timeutil import now_iso, now_utc, parse_iso
from storage import get_storage

bp = Blueprint("cluster", __name__)

# A node whose last heartbeat is older than this is shown offline. Heartbeats are
# published every ~5s from a dedicated thread, so this tolerates several missed
# beats PLUS modest clock skew between nodes - each node writes last_heartbeat_at
# with its OWN clock, so an unsynced (e.g. WSL) peer can look a few seconds stale
# even when healthy. A tight 30s window made nodes flap offline under load / mild
# skew, which made peers skip them and surface 500s; default raised and made
# tunable. If your clocks are well-synced you can lower it again.
NODE_ONLINE_WINDOW_S = int(os.environ.get("CLUSTER_NODE_ONLINE_WINDOW_S", "45"))


import time as _time

_LOCAL_MODELS_TTL = 30.0
_local_models_cache = {"at": 0.0, "data": []}


def _local_models_snapshot() -> list[dict]:
    """Compact local model list for cross-node library rendering, cached briefly
    so the 5s heartbeat doesn't rescan disk every tick."""
    now = _time.monotonic()
    if _local_models_cache["data"] and now - _local_models_cache["at"] < _LOCAL_MODELS_TTL:
        return _local_models_cache["data"]
    try:
        from api.models import discover_models, attach_model_sources
        from core.model_sources import get_model_sources
        from config import MODELS_DIR
        models = discover_models(MODELS_DIR)
        models = attach_model_sources(models, get_model_sources(get_storage().get_settings()))
        compact = [{
            "name": m["name"], "path": m["path"], "type": m.get("type"),
            "quant": m.get("quant", ""), "size_bytes": m.get("size_bytes"),
            "size_display": m.get("size_display", ""), "repo_id": m.get("repo_id"),
        } for m in models]
    except Exception:
        compact = []
    _local_models_cache["at"] = now
    _local_models_cache["data"] = compact
    return compact


def build_local_snapshot() -> dict:
    """Assemble this node's published metadata. Reuses the same collectors the
    single-node dashboard endpoints use, so remote rendering matches local."""
    from api.system_info import collect_system_info, collect_gpu_info
    from api.instances import _public_instance
    from core.helpers import public_dict
    from core.state import instances, instances_lock, downloads, downloads_lock

    try:
        system = collect_system_info()
    except Exception:
        system = {}
    try:
        gpus = collect_gpu_info().get("gpus", [])
    except Exception:
        gpus = []

    with instances_lock:
        inst_list = [_public_instance(i) for i in instances.values()]
    with downloads_lock:
        dl_list = [public_dict(d) for d in downloads.values()]

    return {
        "system": system,
        "gpus": gpus,
        "instances": inst_list,
        "downloads": dl_list,
        "models": _local_models_snapshot(),
        "updated_at": now_iso(),
    }


def publish_cluster_heartbeat() -> None:
    """Upsert this node into the registry with a fresh snapshot. No-op when
    clustering is disabled. Called every poller tick and on join."""
    if not cl.is_cluster_enabled():
        return
    snapshot = None
    try:
        snapshot = build_local_snapshot()
    except Exception as e:
        logger.warning("cluster snapshot build failed: %s", e)
    try:
        get_storage().register_node(cl.local_node_descriptor(), snapshot=snapshot)
    except Exception as e:
        logger.warning("cluster heartbeat failed: %s", e)


def _is_online(node) -> bool:
    """True if `node` heartbeated within the online window.

    Accepts a node dict (preferred) or a raw last_heartbeat_at value (legacy /
    tests). When the backend supplies `heartbeat_age_s` (MariaDB computes it on
    the DB's own clock), we use that - it's immune to clock skew between nodes,
    which is what made an unsynced peer flap offline against another node's
    clock. Otherwise we fall back to comparing against THIS node's clock (fine
    for the single-machine JSON backend, where there is no skew)."""
    if isinstance(node, dict):
        age = node.get("heartbeat_age_s")
        if age is not None:
            # Small negative grace for sub-ms write/read ordering on the DB clock.
            return -2 <= age <= NODE_ONLINE_WINDOW_S
        hb = node.get("last_heartbeat_at")
    else:
        hb = node
    if not hb:
        return False
    try:
        age = (now_utc() - parse_iso(hb)).total_seconds()
    except (ValueError, TypeError):
        return False
    return 0 <= age <= NODE_ONLINE_WINDOW_S


_reach_cache_lock = threading.Lock()
_reach_cache: dict = {}   # node_id -> (monotonic_ts, ok, error)
_REACH_TTL = 15.0


def probe_peer_reachable(node: dict) -> tuple[bool, str]:
    """Can THIS node actually reach `node` over its advertise_url right now?

    This is the direction that cross-node dispatch and work-stealing need, and
    it is NOT what the DB heartbeat proves: a peer can heartbeat into the shared
    DB (so its card renders) while being completely unreachable over HTTP from
    here (e.g. a WSL node advertising a Windows host IP with no port-forward).
    Cached briefly so the 5s dashboard poll doesn't hammer peers."""
    nid = node.get("node_id")
    if not (node.get("advertise_url") or "").strip():
        return (False, "no advertise_url")
    now = _time.monotonic()
    with _reach_cache_lock:
        c = _reach_cache.get(nid)
        if c and now - c[0] < _REACH_TTL:
            return (c[1], c[2])
    ok, err = False, ""
    try:
        resp = cl.cluster_request(node, "GET", "/api/cluster/local-load", timeout=2)
        ok = resp.status_code == 200
        if not ok:
            err = f"HTTP {resp.status_code}"
    except Exception as e:
        err = str(e)
    with _reach_cache_lock:
        _reach_cache[nid] = (now, ok, err)
    return (ok, err)


@bp.route("/api/cluster/identity")
def api_cluster_identity():
    desc = cl.local_node_descriptor()
    desc["enabled"] = cl.is_cluster_enabled()
    return jsonify(desc)


@bp.route("/api/cluster/nodes")
def api_cluster_nodes():
    self_id = cl.get_node_id()
    if not cl.is_cluster_enabled():
        return jsonify({"enabled": False, "self_id": self_id, "nodes": []})

    storage = get_storage()
    nodes = storage.list_nodes()
    # Ensure self appears immediately, even before the first poller tick.
    if not any(n.get("node_id") == self_id for n in nodes):
        publish_cluster_heartbeat()
        nodes = storage.list_nodes()

    for n in nodes:
        n["online"] = _is_online(n)
        n["is_self"] = n.get("node_id") == self_id
        if n["is_self"]:
            n["reachable"], n["reach_error"] = True, ""
        else:
            # Probe the HTTP path we'd actually use to dispatch/steal. A peer can
            # be "online" (recent DB heartbeat) yet unreachable over HTTP - that
            # gap is exactly what silently breaks cross-node balancing.
            n["reachable"], n["reach_error"] = probe_peer_reachable(n)
    nodes.sort(key=lambda n: (not n["is_self"], (n.get("node_name") or "").lower()))
    return jsonify({"enabled": True, "self_id": self_id, "nodes": nodes})


@bp.route("/api/cluster/join", methods=["POST"])
def api_cluster_join():
    """Force an immediate heartbeat so this node registers without waiting for
    the poller. Useful right after enabling clustering."""
    publish_cluster_heartbeat()
    return jsonify({"ok": True, "enabled": cl.is_cluster_enabled()})


@bp.route("/api/cluster/leave", methods=["POST"])
def api_cluster_leave():
    """Remove this node from the registry. The poller re-adds it next tick if
    clustering is still enabled, so this is mainly for graceful shutdown."""
    cl.leave_cluster()
    return jsonify({"ok": True})


_PROXY_HOP_BY_HOP = {"content-encoding", "transfer-encoding", "connection", "content-length"}


@bp.route("/api/cluster/nodes/<node_id>/proxy/<path:subpath>",
          methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def api_cluster_proxy(node_id, subpath):
    """Forward a control request to a peer node's own REST API.

    The browser is already authenticated to THIS node (session); the forwarded
    call carries the cluster secret so the peer accepts it. Used so the UI can
    drive launches / image pulls / downloads on any node. Small JSON responses
    only - log streaming stays on the owning node.
    """
    if not cl.is_cluster_enabled():
        return jsonify({"error": "clustering is not enabled"}), 400
    node = get_storage().get_node(node_id)
    if not node:
        return jsonify({"error": "unknown cluster node"}), 404

    target_path = "/" + subpath
    if request.query_string:
        target_path += "?" + request.query_string.decode()

    headers = {}
    content_type = request.headers.get("Content-Type")
    if content_type:
        headers["Content-Type"] = content_type

    try:
        resp = cl.cluster_request(
            node, request.method, target_path,
            data=request.get_data(), headers=headers, timeout=60,
        )
    except Exception as e:
        return jsonify({"error": f"peer node '{node.get('node_name') or node_id}' unreachable: {e}"}), 502

    out_headers = [(k, v) for k, v in resp.headers.items()
                   if k.lower() not in _PROXY_HOP_BY_HOP]
    return Response(resp.content, status=resp.status_code, headers=out_headers)


# ---------------------------------------------------------------------------
# Shared inference queue: cross-node least-load dispatch
# ---------------------------------------------------------------------------
# When a model runs with "Share queue with same model" on across several nodes,
# those nodes form a group. An inference request for that model is routed to the
# group node with the fewest in-flight requests; the chosen node runs its own
# gate / sampling / recording. A loop guard header stops re-dispatch on the hop.

_DISPATCH_HEADER = "X-Cluster-Dispatch"   # value = hop count (1, 2, ...)
MAX_HOPS = 3                              # entry pick + up to 2 migrations
_OVERFLOW_POLL_S = 0.5                    # how often a queued request re-checks peers


def _request_hops() -> int:
    try:
        return int(request.headers.get(_DISPATCH_HEADER) or 0)
    except (TypeError, ValueError):
        return 0


# Sampling/spec fields shared across a group (last writer wins). Spec decoding is
# launch-time only, so it is stored for the next launch but not applied per call.
_GROUP_OVERRIDE_KEYS = (
    "proxy_sampling_override_enabled", "proxy_sampling_temperature",
    "proxy_sampling_top_k", "proxy_sampling_top_p",
    "proxy_sampling_presence_penalty", "proxy_sampling_repeat_penalty",
    "spec_enabled", "spec_draft_n_max",
)
_GROUP_SAMPLING_KEYS = _GROUP_OVERRIDE_KEYS[:6]


def effective_group_key(model_path: str, config: dict | None) -> str:
    """The cluster-group identity for an instance.

    Defaults to the lowercase filename stem (legacy behavior); a non-empty
    `share_queue_group` on the instance config overrides it, letting same-family
    different-quant instances across nodes pool under one alias (e.g.
    qwen2.5-14b-Q4 on srv1 + qwen2.5-14b-Q8 on srv2 both grouped as
    "qwen2.5-14b"). Already-lowercased at write time, but defensive lower()
    here too because callers may hand in raw config dicts."""
    alias = ((config or {}).get("share_queue_group") or "").strip().lower()
    return alias or model_name_from_path(model_path)


def _inst_matches_request(inst: dict, requested: str) -> bool:
    """True if `requested` matches this instance's group key (alias or filename)
    under the same fuzzy/case-insensitive substring rule the rest of cluster
    matching uses."""
    key = effective_group_key(inst.get("model_path", ""), inst.get("config"))
    req = requested.split(":")[0].lower()
    return req == key or req in key


def _model_matches_name(model_path: str, requested: str) -> bool:
    # Kept as a path-only matcher (no config) for callers that don't have the
    # instance dict at hand; new code should prefer _inst_matches_request.
    inst_model = model_name_from_path(model_path)
    req = requested.split(":")[0].lower()
    return req == inst_model or req in inst_model


def _match_model_load(load_by_model: dict, requested: str):
    """Pick the per-model load entry matching `requested` from a peer's
    local-load map, or None.

    The map is keyed by model_name_from_path() - the LOWERCASE file stem - but a
    client may request mixed case (e.g. the quant suffix "Q4_K_M") or just the
    base name. A plain dict.get(requested) would miss on case alone, silently
    hiding the peer from both routing and work-stealing. Mirrors the fuzzy,
    lowercase rule _model_matches_name uses for local instances so the two sides
    agree on what "the same model" means."""
    if not isinstance(load_by_model, dict):
        return None
    req = requested.split(":")[0].lower()
    entry = load_by_model.get(req)
    if entry is not None:
        return entry
    for key, val in load_by_model.items():
        if req in key:
            return val
    return None


# --- In-flight forward tracking ---------------------------------------------
# A peer's reported load lags our just-sent forwards (network + load-cache), so
# we count our own recent forwards toward that peer to stop a concurrent burst
# all picking the same "empty"-looking peer (the funnel).
#
# CRITICAL: this count is TIME-WINDOWED, not held for the request's lifetime.
# Its only job is to bridge the ~sub-second gap until the peer's OWN reported
# load (active+queued, refreshed every _LOAD_CACHE_TTL) reflects the request.
# Holding it longer DOUBLE-COUNTS - the peer reports the request AND we still
# subtract it - so a peer with a real free slot looks full and we stop
# migrating to it. That was the stuck-queue bug: a slow node could never hand
# its backlog to an idle fast node it had already forwarded one request to.

_inflight_lock = threading.Lock()
_inflight: dict[str, list[float]] = {}   # node_id -> recent forward timestamps
_INFLIGHT_WINDOW_S = 2.0                  # > _LOAD_CACHE_TTL + a little RTT


def _inflight_get(node_id: str) -> int:
    cutoff = _time.monotonic() - _INFLIGHT_WINDOW_S
    with _inflight_lock:
        ts = [t for t in _inflight.get(node_id, ()) if t >= cutoff]
        if ts:
            _inflight[node_id] = ts
        else:
            _inflight.pop(node_id, None)
        return len(ts)


def _inflight_inc(node_id: str) -> None:
    with _inflight_lock:
        _inflight.setdefault(node_id, []).append(_time.monotonic())


def _inflight_dec(node_id: str) -> None:
    """Roll back the forward we just optimistically counted, for the failure
    paths where it never actually lands. Successful forwards are NOT decremented
    here - they age out of the window once the peer reports them itself."""
    with _inflight_lock:
        ts = _inflight.get(node_id)
        if ts:
            ts.pop()
            if not ts:
                _inflight.pop(node_id, None)


def _local_group_load(model_name: str):
    """This node's backlog (in-flight + queued) for the requested group, plus
    whether it's serving only as a fallback. Returns (load, is_fallback) or
    None if this node hosts no matching instance.

    A node is "fallback" for the group only when ALL its matching instances
    are fallback-only. A single primary instance flips the whole node to
    primary - which is correct: if you also run a primary here, the primary's
    own local share_queue path will serve traffic and the fallback sibling is
    just extra capacity below it."""
    from core.state import instances, instances_lock
    from proxy import get_gate
    best = None
    saw_primary = False
    with instances_lock:
        items = list(instances.values())
    for inst in items:
        cfg = inst.get("config") or {}
        if not cfg.get("share_queue") or inst.get("status") == "stopped":
            continue
        if not _inst_matches_request(inst, model_name):
            continue
        if not cfg.get("share_queue_fallback"):
            saw_primary = True
        gate = get_gate(inst["id"])
        load = (gate.active + gate.queued) if gate else 0
        best = load if best is None else min(best, load)
    if best is None:
        return None
    return (best, not saw_primary)


def _group_candidates(model_name: str) -> list[dict]:
    """Group members with their CURRENT backlog (active + queued) and whether
    they're serving as fallback-only. Self is read live from local gates; peers
    are queried live (briefly cached) plus our own in-flight forwards to them -
    so routing reflects reality, not a stale snapshot. This is what makes
    round-robin / least-load actually balance, and what lets fallback peers
    sit idle until everyone else is busy."""
    self_id = cl.get_node_id()
    candidates = []
    local = _local_group_load(model_name)
    if local is not None:
        load, fallback = local
        candidates.append({"node": {"node_id": self_id, "node_name": ""},
                           "load": load, "is_self": True, "fallback": fallback})
    for node in get_storage().list_nodes():
        nid = node.get("node_id")
        if nid == self_id or not _is_online(node):
            continue
        if not (node.get("advertise_url") or "").strip():
            continue  # can't forward to a node with no advertise URL
        live = _peer_live_load(node, model_name)
        if not live:
            continue
        load = (live.get("active") or 0) + (live.get("queued") or 0) + _inflight_get(nid)
        candidates.append({"node": node, "load": load, "is_self": False,
                           "fallback": bool(live.get("fallback", False))})
    return candidates


def _forward_inference(node: dict, path: str, body: bytes, content_type: str | None,
                       hops: int = 1):
    """Stream an inference request to a peer's compat endpoint. Returns a Flask
    Response on success, OR a synthetic 504 Response when the peer accepted the
    request and silently kept working past our REQUEST_TIMEOUT (read-timeout),
    OR None to signal "try elsewhere" (peer never accepted: connect refused /
    unreachable, OR returned 429). `hops` is the forward count carried to the
    peer (loop/ping-pong guard).

    The connect-vs-read distinction is critical: on a read-timeout the peer is
    almost certainly still processing OUR request, and any local fallback or
    retry would double-count and burn compute on the same prompt. So we surface
    a clean 504 to the client instead of pretending the request didn't happen.
    """
    import json as _json
    import requests as _requests
    nid = node.get("node_id")
    name = node.get("node_name") or nid
    headers = {
        _DISPATCH_HEADER: str(hops),
    }
    if content_type:
        headers["Content-Type"] = content_type
    # Forward the client's API key so the peer authenticates it as a normal client.
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    # Count the in-flight forward BEFORE the call (rolled back on failure). With
    # stream=True, cluster_request() doesn't return until the peer's first byte -
    # the model's time-to-first-token on a slow node. Incrementing only after
    # that leaves a window where the peer still looks empty, so a concurrent
    # burst funnels onto it (self load, by contrast, rises instantly in-process).
    _inflight_inc(nid)
    try:
        resp = cl.cluster_request(node, "POST", path, data=body, headers=headers,
                                  stream=True, timeout=REQUEST_TIMEOUT)
    except _requests.ReadTimeout as e:
        # Peer accepted the request and is still working on it; we just gave up
        # waiting. Falling back to local would re-process the same prompt while
        # the peer continues in the background. Surface 504 to the client.
        _inflight_dec(nid)
        logger.warning("cluster forward to %s read-timeout after %ss: %s",
                       name, REQUEST_TIMEOUT, e)
        msg = (f"peer {name} accepted the request but did not respond within "
               f"REQUEST_TIMEOUT ({REQUEST_TIMEOUT}s); it may still be processing")
        return Response(_json.dumps({"error": {"message": msg}}),
                        status=504, mimetype="application/json")
    except Exception as e:
        _inflight_dec(nid)
        logger.warning("cluster dispatch to %s failed: %s", name, e)
        return None
    if resp.status_code == 429:  # peer's gate is full - try elsewhere
        resp.close()
        _inflight_dec(nid)
        return None

    out_headers = [(k, v) for k, v in resp.headers.items()
                   if k.lower() not in _PROXY_HOP_BY_HOP]

    # Detect streaming vs single-shot responses. Chat completions with
    # stream=false return application/json; we strip the peer's Content-Length
    # in out_headers, so handing that to Flask as an iterator forces chunked
    # transfer encoding. If the iterator then dies (peer disconnect, network
    # blip, generator GC race), Flask never emits the final 0-length chunk and
    # the client hangs waiting for a terminator that never comes. Buffering
    # non-streaming bodies into a single Response with Content-Length avoids
    # the whole chunked-encoding failure mode.
    ct = (resp.headers.get("Content-Type") or "").lower()
    is_streaming = ("text/event-stream" in ct
                    or "application/x-ndjson" in ct
                    or "application/jsonlines" in ct)

    if not is_streaming:
        logger.info("forward %s: headers received status=%d, reading body",
                    name, resp.status_code)
        try:
            body_bytes = resp.content
        except Exception as e:
            _inflight_dec(nid)
            logger.warning("cluster forward to %s body read failed: %s", name, e)
            try:
                resp.close()
            except Exception:
                pass
            return None
        logger.info("forward %s: body read %d bytes, returning response",
                    name, len(body_bytes))
        resp.close()
        return Response(body_bytes, status=resp.status_code, headers=out_headers)

    def relay():
        try:
            for chunk in resp.iter_content(chunk_size=None):
                yield chunk
        except Exception as e:
            # Log the abort so the next time a client hangs we can see where
            # it happened. The connection still gets cut by Werkzeug here -
            # there's no clean way to send the chunked-encoding terminator
            # after an exception - but the client at least gets EOF, not
            # silent hang.
            logger.warning("cluster forward to %s stream relay aborted: %s", name, e)
        finally:
            resp.close()
            # NOTE: no _inflight_dec here - a successful forward ages out of the
            # time window on its own (by which point the peer reports it in its
            # own load). Decrementing on stream-end would under-count long
            # requests but, worse, it never fired for the funnel case anyway.

    return Response(relay(), status=resp.status_code, headers=out_headers)


_rr_lock = threading.Lock()
_rr_counter = 0


def _order_candidates(candidates: list[dict]) -> list[dict]:
    """Least-loaded first, but round-robin among the equally-least-loaded nodes.
    Fallback-only candidates are always sorted AFTER all primaries, regardless
    of load: a fallback exists to serve when primaries are saturated, so even
    an idle fallback must wait its turn behind a busy primary. The dispatch
    loop walks this list and 429s fall through to the next entry, so the
    fallback naturally absorbs overflow only when every primary refused.

    In-flight counts only rise after a request acquires its gate, which happens
    after this decision, so a simultaneous burst sees every node tied at the
    same count. Always preferring self there would pin the whole burst to the
    entry node; rotating the tie spreads it across the group."""
    global _rr_counter
    primaries = [c for c in candidates if not c.get("fallback")]
    fallbacks = [c for c in candidates if c.get("fallback")]

    def _sort_tier(tier: list[dict]) -> list[dict]:
        if not tier:
            return []
        min_load = min(c["load"] for c in tier)
        tied = [c for c in tier if c["load"] == min_load]
        rest = sorted((c for c in tier if c["load"] != min_load),
                      key=lambda c: c["load"])
        # RR start is shared across calls; tying within a tier alone is enough
        # to spread bursts (no need for separate counters per tier).
        global _rr_counter  # noqa: PLW0603
        with _rr_lock:
            start = _rr_counter % len(tied)
            _rr_counter += 1
        return tied[start:] + tied[:start] + rest

    return _sort_tier(primaries) + _sort_tier(fallbacks)


def dispatch_inference(model_name: str):
    """If `model_name` is part of a multi-node shared queue, route the current
    request to the least-loaded group node (round-robin on ties). Returns a Flask
    Response when forwarded to a peer, or None to handle it locally."""
    if _request_hops() >= 1:
        return None  # already routed here by a peer - the gate/overflow path takes over
    if not cl.is_cluster_enabled():
        return None

    candidates = _group_candidates(model_name)
    if not candidates or not any(not c["is_self"] for c in candidates):
        return None  # only this node (or no node) hosts it - normal local path

    ordered = _order_candidates(candidates)
    body = request.get_data()
    content_type = request.headers.get("Content-Type")
    path = request.path
    qs = request.query_string.decode()
    if qs:
        path = f"{path}?{qs}"

    for c in ordered:
        if c["is_self"]:
            return None  # this node is the chosen target - serve here
        forwarded = _forward_inference(c["node"], path, body, content_type, hops=1)
        if forwarded is not None:
            return forwarded
    return None  # every peer was busy/unreachable - fall back to local handling


# ---------------------------------------------------------------------------
# Work-stealing: a queued request migrates to a peer that has a free slot
# ---------------------------------------------------------------------------
# Snapshots are ~5s stale, which is fine for the entry's initial spread but too
# stale to decide "is a peer free right now?". So migration uses a live load
# query to peers (briefly cached) - this is what lets an idle node pull a busy
# node's backlog without the staleness funnel that broke plain dispatch.

_load_cache_lock = threading.Lock()
_load_cache: dict[str, tuple[float, dict]] = {}   # node_id -> (monotonic_ts, load)
_LOAD_CACHE_TTL = 0.3


def local_load_by_model() -> dict:
    """This node's live gate load per shared-queue group: in-flight, queued,
    free slots, and a per-group `fallback` flag. Keyed by the effective group
    key (alias if set, else filename stem) so peers asking by alias find this
    node without a filename match. Read straight from gate state - cheap, no DB.

    `fallback` is true for the group only when EVERY contributing instance is
    fallback-only; one primary flips the whole group's flag back to false on
    this node, matching the local routing rule (a primary sibling here will
    serve traffic, so this node should be treated as primary by peers too)."""
    from core.state import instances, instances_lock
    from proxy import get_gate
    with instances_lock:
        snap = [(i["id"], i.get("model_path", ""), dict(i.get("config") or {}), i.get("status"))
                for i in instances.values()]
    out: dict = {}
    for inst_id, model_path, cfg, status in snap:
        if not cfg.get("share_queue") or status == "stopped":
            continue
        gate = get_gate(inst_id)
        if gate:
            active, queued, mc = gate.active, gate.queued, gate.max_concurrent
        else:
            active, queued, mc = 0, 0, int(cfg.get("max_concurrent") or 0)
        free = max(0, mc - active) if mc else 1   # no gate => unlimited => free
        key = effective_group_key(model_path, cfg)
        cur = out.setdefault(key, {"active": 0, "queued": 0, "free": 0,
                                   "max_concurrent": 0, "fallback": True})
        cur["active"] += active
        cur["queued"] += queued
        cur["free"] += free
        cur["max_concurrent"] += mc
        if not cfg.get("share_queue_fallback"):
            cur["fallback"] = False
    return out


def _peer_live_load(node: dict, model_name: str):
    """Live load for `model_name` on a peer (briefly cached). Returns the
    per-model dict or None on miss/unreachable. The whole node map is cached so
    one query serves any model; matching is fuzzy/case-insensitive so a quant-
    suffixed request (gpt-oss-20b-Q4_K_M) still finds the lowercase-keyed entry."""
    nid = node.get("node_id")
    now = _time.monotonic()
    with _load_cache_lock:
        cached = _load_cache.get(nid)
        data = cached[1] if cached and now - cached[0] < _LOAD_CACHE_TTL else None
    if data is None:
        try:
            resp = cl.cluster_request(node, "GET", "/api/cluster/local-load", timeout=2)
            data = resp.json() if resp.status_code == 200 else {}
        except Exception:
            data = {}
        with _load_cache_lock:
            _load_cache[nid] = (now, data)
    return _match_model_load(data, model_name)


def _find_free_peer(model_name: str):
    """The peer with the most genuinely-free slots for the model right now -
    live `free` minus our own in-flight forwards already heading there - or None.

    Fallback peers are only considered when NO primary peer has a free slot,
    matching the dispatch tiering: a queued request stays put as long as a
    primary peer might open up; only when every primary is full does it steal
    onto a fallback. Without this rule, work-stealing would silently route to
    an idle fallback even when primaries had spare capacity arriving soon.

    Bouncing back to a peer that already saw this request is fine: the hop
    count (MAX_HOPS) bounds the chain, and TCP keepalive on cluster_request
    means an in-flight relay socket won't silently die during a long wait.
    The migration is also gated by "genuinely free slots", so a saturated
    peer that just handed work off won't be re-picked."""
    self_id = cl.get_node_id()
    best_primary = None   # (effective_free, node)
    best_fallback = None
    for node in get_storage().list_nodes():
        nid = node.get("node_id")
        if nid == self_id or not _is_online(node):
            continue
        if not (node.get("advertise_url") or "").strip():
            continue
        load = _peer_live_load(node, model_name)
        if not load:
            continue
        eff_free = (load.get("free", 0) or 0) - _inflight_get(nid)
        if eff_free <= 0:
            continue
        if load.get("fallback"):
            if best_fallback is None or eff_free > best_fallback[0]:
                best_fallback = (eff_free, node)
        else:
            if best_primary is None or eff_free > best_primary[0]:
                best_primary = (eff_free, node)
    chosen = best_primary or best_fallback
    return chosen[1] if chosen else None


def _find_evacuation_peer(model_name: str):
    """The least-loaded reachable peer hosting the model, REGARDLESS of free
    slots. Used only when the local worker has died (a draining gate): the local
    node can't serve, so a busy-but-alive peer is strictly better than being
    stuck on a dead node - the peer queues the request and serves it as capacity
    frees up. (A peer whose own queue is full 429s and the caller retries the
    next one.) Contrast with _find_free_peer, which requires genuine free
    capacity and is the right rule for stealing from an alive-but-saturated
    node.

    No visited-set guard here for the same reason as _find_free_peer: the hop
    count caps the chain, keepalive prevents silent socket death, and an
    evacuating local node only retries to peers that actually accepted (a 429
    falls through to the next candidate)."""
    self_id = cl.get_node_id()
    best_primary = None    # (backlog, node)
    best_fallback = None
    for node in get_storage().list_nodes():
        nid = node.get("node_id")
        if nid == self_id or not _is_online(node):
            continue
        if not (node.get("advertise_url") or "").strip():
            continue
        load = _peer_live_load(node, model_name)
        if not load:
            continue  # peer doesn't host this model (or unreachable)
        backlog = (load.get("active", 0) or 0) + (load.get("queued", 0) or 0) + _inflight_get(nid)
        # Even when evacuating, prefer alive primaries over fallbacks: the
        # operator marked the fallback as second-class for a reason (different
        # model, slower hardware), so use it only if no primary is alive.
        if load.get("fallback"):
            if best_fallback is None or backlog < best_fallback[0]:
                best_fallback = (backlog, node)
        else:
            if best_primary is None or backlog < best_primary[0]:
                best_primary = (backlog, node)
    chosen = best_primary or best_fallback
    return chosen[1] if chosen else None


def acquire_or_overflow(gate, model_name: str):
    """Get a local gate slot, or migrate the request to a peer. The request stays
    counted in the gate's queue while it waits, so the queue metric is accurate.

    While the local worker is alive but saturated, migration is work-stealing: it
    only targets a peer with a genuinely FREE slot. Once the local worker has died
    (the gate is draining), it switches to evacuation: offload to ANY alive peer
    hosting the model - even a busy one - because the local node can no longer
    serve and a stuck queue on a dead node helps nobody. Returns:
      (True,  None,     None)     - got a local slot, serve here
      (False, Response, None)     - migrated to a peer; relay this back to the client
      (False, None,     reason)   - couldn't serve. reason is one of:
            "queue_full" (admission denied -> 429),
            "deadline"   (waited and gave up -> 504),
            "closed"     (gate cancelled -> 503).
    """
    hops = _request_hops()
    if not (cl.is_cluster_enabled() and hops < MAX_HOPS):
        # No migration: one plain bounded wait. queued still accurate via the gate.
        if gate.acquire(timeout=REQUEST_TIMEOUT):
            return (True, None, None)
        # gate.acquire returns False on queue_full, closed, or deadline; the gate
        # itself doesn't distinguish in this simpler API. Surface "deadline" by
        # default - it's the most common failure mode for inference and gives
        # the more accurate 504 instead of a misleading 429.
        return (False, None, "deadline")

    body = request.get_data()
    content_type = request.headers.get("Content-Type")
    path = request.path
    qs = request.query_string.decode()
    if qs:
        path = f"{path}?{qs}"

    def find_target():
        # A dead local worker (draining gate) evacuates to any alive peer; an
        # alive-but-saturated one only steals to a peer with a real free slot.
        if getattr(gate, "_draining", False):
            return _find_evacuation_peer(model_name)
        return _find_free_peer(model_name)

    def do_forward(peer):
        return _forward_inference(peer, path, body, content_type, hops=hops + 1)

    outcome, value = gate.acquire_or_overflow(
        REQUEST_TIMEOUT, _OVERFLOW_POLL_S, find_target, do_forward)
    if outcome == "acquired":
        return (True, None, None)
    if outcome == "overflow":
        return (False, value, None)
    # outcome == "rejected"; value is the reason string ("closed"/"queue_full"/"deadline")
    return (False, None, value)


def rejection_status(reason: str) -> tuple[int, str]:
    """Map an acquire_or_overflow rejection reason to (HTTP status, message).
    deadline -> 504 (Gateway Timeout) because we waited and gave up, which is
    semantically a timeout from the client's perspective, not a "queue full".
    queue_full -> 429 (Too Many Requests). closed -> 503 (Service Unavailable).
    """
    if reason == "deadline":
        return (504, "request timed out waiting for capacity")
    if reason == "queue_full":
        return (429, "request queue full")
    return (503, "instance unavailable")


@bp.route("/api/cluster/local-load")
def api_cluster_local_load():
    """Live per-model gate load on this node - used by peers to decide migration."""
    return jsonify(local_load_by_model())


def record_group_overrides(model_path: str, config: dict) -> None:
    """Persist a share_queue model's sampling/spec config as the group default
    (last writer wins). No-op outside a shared-queue cluster launch."""
    if not cl.is_cluster_enabled() or not config.get("share_queue"):
        return
    overrides = {k: config[k] for k in _GROUP_OVERRIDE_KEYS if k in config}
    if not overrides:
        return
    model = model_name_from_path(model_path)
    try:
        get_storage().merge_settings({"cluster_group_overrides": {model: overrides}})
    except Exception as e:
        logger.warning("failed to record cluster group overrides for %s: %s", model, e)


def effective_inference_config(inst: dict) -> dict:
    """Instance config with group sampling overrides applied for share_queue
    models, so every node serving the group uses the same sampling values."""
    cfg = inst.get("config", {}) or {}
    if not cl.is_cluster_enabled() or not cfg.get("share_queue"):
        return cfg
    model = model_name_from_path(inst.get("model_path", ""))
    try:
        overrides = (get_storage().get_settings().get("cluster_group_overrides") or {}).get(model)
    except Exception:
        overrides = None
    if not overrides:
        return cfg
    merged = dict(cfg)
    for k in _GROUP_SAMPLING_KEYS:
        if k in overrides:
            merged[k] = overrides[k]
    return merged
