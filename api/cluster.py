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

from flask import Blueprint, Response, g, has_request_context, jsonify, request

from config import REQUEST_TIMEOUT, logger
from core import cluster as cl
from core.perf import phase
from core.helpers import iter_response_chunks, model_name_from_path
from core.proxy_sampling import PROXY_SAMPLING_OVERRIDE_KEYS
from core.spec_decoding import SPEC_CONFIG_KEYS
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


# "alias_wake": a group alias sent here is served even when the member isn't
# healthy yet (api.llamaman._serve_alias_member).
NODE_FEATURES = ("alias_wake",)


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

    # Phase-timed (LLAMAMAN_PERF_LOG=1): runs on the 5s heartbeat thread.
    # Models are cached 30s but system/GPU collection is not — measure whether
    # they dominate the heartbeat.
    try:
        with phase("heartbeat.collect_system_info"):
            system = collect_system_info()
    except Exception:
        system = {}
    try:
        with phase("heartbeat.collect_gpu_info"):
            gpus = collect_gpu_info().get("gpus", [])
    except Exception:
        gpus = []

    with instances_lock:
        inst_list = [_public_instance(i) for i in instances.values()]
    with downloads_lock:
        dl_list = [public_dict(d) for d in downloads.values()]

    # Knowledge-base capability advertisement (docs/kb-mcp-plan.md §6.5). One
    # settings read on the 5s heartbeat; consumers (UI + peer nodes) treat
    # unknown snapshot keys as opaque, so mixed-version clusters are safe.
    try:
        from storage import get_storage
        s = get_storage().get_settings()
        kb_info = {
            "enabled": bool(s.get("kb_enabled", False)),
            "mcp": bool(s.get("kb_mcp_enabled", False)),
            "access": s.get("kb_mcp_access", "global"),
            "embedding_instance": s.get("kb_embedding_instance", "") or "",
            "embedding_model": s.get("kb_embedding_model", "") or "",
        }
    except Exception:
        kb_info = {"enabled": False, "mcp": False}

    # Engine availability, so a peer's UI only offers e.g. Strata on nodes
    # with an NVIDIA GPU and STRATA_ENABLED. Published under "system" (it is
    # a node capability, like the GPU list) rather than as a new top-level
    # key; consumers treat unknown keys as opaque, so older peers ignore it.
    try:
        from api.engines import engines_snapshot
        system = {**system, "engines": engines_snapshot()}
    except Exception:
        pass

    # What this node's code can do, for peers deciding what to send it. Same
    # opaque-to-older-peers rule as "engines".
    system = {**system, "features": list(NODE_FEATURES)}

    return {
        "system": system,
        "gpus": gpus,
        "instances": inst_list,
        "downloads": dl_list,
        "models": _local_models_snapshot(),
        "kb": kb_info,
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
        with phase("heartbeat.register_node"):
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
        # Cache-miss path: a live probe, up to 2s on the request thread.
        with phase("cluster_nodes.probe", node=nid):
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
    # Phase-timed (LLAMAMAN_PERF_LOG=1): this endpoint is polled by every node's
    # UI every 5s and runs a (cached) reachability probe per peer inline — the
    # cache-miss path costs up to 2s per peer on the request thread.
    with phase("cluster_nodes.list", self=self_id):
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
    with phase("cluster_proxy.get_node", node=node_id):
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
        # (connect, read), NOT a scalar. A scalar 60 also means "wait 60s to
        # connect", so a peer that heartbeats into the shared DB while being
        # unreachable over HTTP held this thread - and one of the browser's ~6
        # connections per origin - for a full minute per call. The dashboard
        # polls peers every ~9s, so those stacked up until every later fetch on
        # the page queued behind them and the stats/logs modals span forever.
        # 3s is far more than a LAN/tunnel peer needs to complete a TCP
        # handshake; the 60s read budget is untouched, so slow-but-alive control
        # ops (image pulls) still get their minute.
        with phase("cluster_proxy.forward", node=node_id, method=request.method,
                   path=target_path.split("?")[0][:60]):
            resp = cl.cluster_request(
                node, request.method, target_path,
                data=request.get_data(), headers=headers, timeout=(3, 60),
            )
    except Exception as e:
        return jsonify({"error": f"peer node '{node.get('node_name') or node_id}' unreachable: {e}"}), 502

    out_headers = [(k, v) for k, v in resp.headers.items()
                   if k.lower() not in _PROXY_HOP_BY_HOP]
    return Response(resp.content, status=resp.status_code, headers=out_headers)


# ---------------------------------------------------------------------------
# Shared inference queue: cross-node least-load dispatch
# ---------------------------------------------------------------------------
# When a model runs with "Queue sharing" on across several nodes,
# those nodes form a group. An inference request for that model is routed to the
# group node with the fewest in-flight requests; the chosen node runs its own
# gate / sampling / recording. A loop guard header stops re-dispatch on the hop.

_DISPATCH_HEADER = "X-Cluster-Dispatch"   # value = hop count (1, 2, ...)
# On a forwarded request's 503: this node served nothing and a sibling might, so
# the forwarder tries its next member. A peer without it just passes the 503 on.
RETRY_ELSEWHERE_HEADER = "X-Llamaman-Retry-Elsewhere"
MAX_HOPS = 3                              # entry pick + up to 2 migrations
_OVERFLOW_POLL_S = 0.5                    # how often a queued request re-checks peers


def _request_hops() -> int:
    try:
        return int(request.headers.get(_DISPATCH_HEADER) or 0)
    except (TypeError, ValueError):
        return 0


# Sampling/spec fields shared across a group (last writer wins). Spec decoding is
# launch-time only, so it is stored for the next launch but not applied per call.
_GROUP_SAMPLING_KEYS = PROXY_SAMPLING_OVERRIDE_KEYS
_GROUP_OVERRIDE_KEYS = _GROUP_SAMPLING_KEYS + SPEC_CONFIG_KEYS


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


def peer_is_routable(node: dict) -> bool:
    """A peer inference can be forwarded to: heartbeating, with a URL to reach it
    on. Listings and routing share this, so a view-only or offline peer's groups
    aren't advertised under a name that can't be served. Member status stays each
    caller's business: a listing has only the snapshot, routing has live load."""
    return _is_online(node) and bool((node.get("advertise_url") or "").strip())


def _is_group_member(config: dict | None) -> bool:
    return bool((config or {}).get("share_queue"))


def cluster_group_models() -> list[dict]:
    """Distinct `share_queue_group` aliases currently served somewhere in the
    cluster, for advertising in the model-listing APIs.

    Each entry is {"name", "path"} where `path` is a representative model file on
    THIS node when the group runs here (so the listing can borrow its GGUF
    metadata), else None for a peer-only group.

    A group is the cross-node, load-balanced entry point - a client sends the
    alias and `dispatch_inference` routes it to the least-loaded node. The
    per-file `/api/tags` listing never mentions it, so a client has no way to
    discover the name it should send. This surfaces it, but only when it means
    something: clustering enabled, the alias actually set, and at least one
    non-stopped instance on this node or a routable peer - which is exactly when
    a request for the alias can be routed rather than 404'd.
    """
    if not cl.is_cluster_enabled():
        return []

    # (alias, status, local_model_path_or_None) from every instance we know of.
    seen: list[tuple[str, str, str | None]] = []

    from core.state import instances, instances_lock
    with instances_lock:
        for inst in instances.values():
            alias = ((inst.get("config") or {}).get("share_queue_group") or "").strip()
            if alias and _is_group_member(inst.get("config")):
                seen.append((alias, inst.get("status") or "", inst.get("model_path")))

    try:
        nodes = get_storage().list_nodes() or []
    except Exception:
        nodes = []
    self_id = cl.get_node_id()
    for node in nodes:
        if node.get("node_id") == self_id:
            continue  # this node's live instances are already counted above
        if not peer_is_routable(node):
            continue
        snapshot = node.get("snapshot") or {}
        for inst in snapshot.get("instances") or []:
            alias = ((inst.get("config") or {}).get("share_queue_group") or "").strip()
            if alias and _is_group_member(inst.get("config")):
                seen.append((alias, inst.get("status") or "", None))

    groups: dict[str, dict] = {}
    for alias, status, local_path in seen:
        # "stopping" is transient - the docker stop is already in flight, so
        # dispatch should treat it as gone (matches the proxy's 503 branch).
        if status in ("stopped", "stopping"):
            continue
        key = alias.lower()
        entry = groups.setdefault(key, {"name": alias, "path": None})
        if local_path and entry["path"] is None:
            entry["path"] = local_path
    return list(groups.values())


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


# How awake a group member is. Lower serves sooner.
_TIER_HEALTHY, _TIER_STARTING, _TIER_SLEEPING = 0, 1, 2
_STATUS_TIER = {"healthy": _TIER_HEALTHY, "starting": _TIER_STARTING, "sleeping": _TIER_SLEEPING}

# The order dispatch and evacuation try (is_fallback, tier) in: awake capacity
# beats a cold load, so even a fallback that's up comes before a sleeping primary.
_CANDIDATE_ORDER = (
    (False, _TIER_HEALTHY), (False, _TIER_STARTING),
    (True, _TIER_HEALTHY), (True, _TIER_STARTING),
    (False, _TIER_SLEEPING), (True, _TIER_SLEEPING),
)


def _member_tier(status: str | None) -> int:
    return _STATUS_TIER.get(status, _TIER_HEALTHY)


def _local_group_load(model_name: str):
    """This node's backlog (in-flight + queued) for the requested group, whether
    it's serving only as a fallback, and how awake it is. Returns (load,
    is_fallback, tier) or None if this node hosts no matching instance.

    All three describe the members this node would actually serve from: those in
    its best tier (_ensure_model_running ranks them the same way). A saturated
    healthy member is not idle because a sleeper sits beside it. Within that
    tier a node is "fallback" only when ALL its members are fallback-only: one
    primary flips it, as the primary's own local share_queue path will serve
    traffic and the fallback sibling is just extra capacity below it."""
    from core.state import instances, instances_lock
    from proxy import get_gate
    with instances_lock:
        items = list(instances.values())
    members = [inst for inst in items
               if (inst.get("config") or {}).get("share_queue")
               and inst.get("status") not in ("stopped", "stopping")
               and _inst_matches_request(inst, model_name)]
    if not members:
        return None
    tier = min(_member_tier(inst.get("status")) for inst in members)
    serving = [inst for inst in members if _member_tier(inst.get("status")) == tier]
    loads = []
    for inst in serving:
        gate = get_gate(inst["id"])
        loads.append((gate.active + gate.queued) if gate else 0)
    fallback = all((inst.get("config") or {}).get("share_queue_fallback") for inst in serving)
    return (min(loads), fallback, tier)


def _peer_tier(live: dict) -> int:
    """How awake a peer's load entry says it is. An older peer reports no counts:
    assume awake, and _peer_can_serve answers for it from its snapshot."""
    if live.get("healthy"):
        return _TIER_HEALTHY
    if live.get("starting"):
        return _TIER_STARTING
    if live.get("sleeping"):
        return _TIER_SLEEPING
    return _TIER_HEALTHY


def _peer_can_serve(node: dict, model_name: str) -> bool:
    """False for an older peer whose members for this group aren't healthy: it
    would answer an alias with a 500. Its live load can't reveal that, so this
    reads the stored snapshot (~5s old), whose feature list also identifies it.
    With no snapshot there's nothing to go on, so the live load stands."""
    snapshot = node.get("snapshot")
    if not snapshot:
        return True
    if "alias_wake" in ((snapshot.get("system") or {}).get("features") or []):
        return True
    return any(inst.get("status") == "healthy" and _inst_matches_request(inst, model_name)
               for inst in snapshot.get("instances") or [])


def _snapshot_candidate(node: dict, model_name: str) -> dict | None:
    """A candidate for a peer whose live load couldn't be read, from its snapshot.
    A failed probe says nothing about which groups it hosts, so a blip mustn't
    make a served group vanish. `unknown` ranks it behind every peer that answered:
    at an assumed load of 0 a dead peer would otherwise attract the traffic."""
    if not _peer_can_serve(node, model_name):
        return None
    members = [inst for inst in (node.get("snapshot") or {}).get("instances") or []
               if _is_group_member(inst.get("config"))
               and inst.get("status") not in ("stopped", "stopping")
               and _inst_matches_request(inst, model_name)]
    if not members:
        return None
    tier = min(_member_tier(inst.get("status")) for inst in members)
    fallback = all((inst.get("config") or {}).get("share_queue_fallback") for inst in members
                   if _member_tier(inst.get("status")) == tier)
    return {"node": node, "load": _inflight_get(node.get("node_id")), "is_self": False,
            "fallback": fallback, "tier": tier, "unknown": True}


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
        load, fallback, tier = local
        candidates.append({"node": {"node_id": self_id, "node_name": ""},
                           "load": load, "is_self": True, "fallback": fallback, "tier": tier})
    for node in get_storage().list_nodes():
        nid = node.get("node_id")
        if nid == self_id or not peer_is_routable(node):
            continue
        live = _peer_live_load(node, model_name)
        if live is PEER_UNREACHABLE:
            candidate = _snapshot_candidate(node, model_name)
            if candidate:
                candidates.append(candidate)
            continue
        if not live or not _peer_can_serve(node, model_name):
            continue
        load = (live.get("active") or 0) + (live.get("queued") or 0) + _inflight_get(nid)
        candidates.append({"node": node, "load": load, "is_self": False,
                           "fallback": bool(live.get("fallback", False)),
                           "tier": _peer_tier(live)})
    return candidates


# Why each peer turned down this request, kept on the request so that if nobody
# takes it the client hears that rather than "model not found".
_REFUSED_FULL = "queue full"
_REFUSED_UNREACHABLE = "unreachable"


def _note_refusal(node_id: str, reason: str) -> None:
    if not hasattr(g, "forward_refusals"):
        g.forward_refusals = {}
    g.forward_refusals[node_id] = reason


def group_unroutable_error(model_name: str) -> dict | None:
    """When `model_name` is a group other nodes serve but none of them can take
    this request: {code, message, http_status}, naming each node and why. None
    when it isn't one, and for a forwarded request (its entry node answers).
    429 if a peer is merely full, which a client can usefully retry."""
    if not cl.is_cluster_enabled() or not has_request_context() or _request_hops() >= 1:
        return None
    from core.engines import virtual_model_key
    from core.model_alias import canonical_name
    canonical = canonical_name(model_name)
    canonical = virtual_model_key(canonical) or canonical
    req = canonical.split(":")[0].lower()
    refusals = getattr(g, "forward_refusals", {})
    try:
        nodes = get_storage().list_nodes() or []
    except Exception:
        nodes = []
    self_id = cl.get_node_id()
    reasons = []
    full = False
    for node in nodes:
        nid = node.get("node_id")
        if nid == self_id:
            continue
        hosts = any(_is_group_member(inst.get("config"))
                    and inst.get("status") not in ("stopped", "stopping")
                    and effective_group_key(inst.get("model_path", ""), inst.get("config")) == req
                    for inst in (node.get("snapshot") or {}).get("instances") or [])
        if not hosts and nid not in refusals:
            continue
        reason = refusals.get(nid)
        if reason is None:
            if not _is_online(node):
                reason = "offline"
            elif not (node.get("advertise_url") or "").strip():
                reason = "view-only, no CLUSTER_ADVERTISE_URL"
            else:
                reason = "not ready"
        full = full or reason == _REFUSED_FULL
        reasons.append(f"{node.get('node_name') or nid} ({reason})")
    if not reasons:
        return None
    message = (f"model '{model_name}' is served by other nodes, but none could take "
               f"this request: {', '.join(sorted(reasons))}")
    if full:
        return {"code": "group_full", "message": message, "http_status": 429}
    return {"code": "group_unavailable", "message": message, "http_status": 503}


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
        # (connect, read). The read half keeps the full REQUEST_TIMEOUT: it has
        # to cover the peer loading the model on demand plus its time to first
        # token. Connect is only the TCP handshake with the peer's gunicorn,
        # which is listening regardless of model state - as a scalar it silently
        # inherited the same 300s, so an unreachable peer pinned a thread here
        # for five minutes before we gave up and tried elsewhere. A ConnectTimeout
        # is not a ReadTimeout, so it falls through to the generic handler below
        # and correctly reads as "peer never accepted it, try another node".
        # 5s (not 3) because a dropped SYN over a VPN link retransmits at ~1s
        # then ~3s, and a false negative here needlessly relocates real work.
        resp = cl.cluster_request(node, "POST", path, data=body, headers=headers,
                                  stream=True, timeout=(5, REQUEST_TIMEOUT))
    except _requests.ReadTimeout as e:
        # Peer accepted the request and is still working on it; we just gave up
        # waiting. Falling back to local would re-process the same prompt while
        # the peer continues in the background. Surface 504 to the client.
        _inflight_dec(nid)
        logger.warning("cluster forward to %s read-timeout after %ss: %s",
                       name, REQUEST_TIMEOUT, e)
        msg = (f"peer {name} accepted the request but did not respond within "
               f"REQUEST_TIMEOUT ({REQUEST_TIMEOUT}s); it may still be processing")
        # Ollama's surface carries the error as a plain string, OpenAI's nested.
        error = msg if request.path.startswith("/api/") else {"message": msg}
        return Response(_json.dumps({"error": error}),
                        status=504, mimetype="application/json")
    except Exception as e:
        _inflight_dec(nid)
        logger.warning("cluster dispatch to %s failed: %s", name, e)
        _note_refusal(nid, _REFUSED_UNREACHABLE)
        return None
    if resp.status_code == 429:  # peer's gate is full - try elsewhere
        resp.close()
        _inflight_dec(nid)
        _note_refusal(nid, _REFUSED_FULL)
        return None
    if resp.status_code == 503 and resp.headers.get(RETRY_ELSEWHERE_HEADER):
        logger.info("cluster forward to %s: peer could not serve it, trying elsewhere", name)
        resp.close()
        _inflight_dec(nid)
        _note_refusal(nid, "could not serve it")
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
            _note_refusal(nid, _REFUSED_UNREACHABLE)
            return None
        logger.info("forward %s: body read %d bytes, returning response",
                    name, len(body_bytes))
        resp.close()
        return Response(body_bytes, status=resp.status_code, headers=out_headers)

    def relay():
        try:
            for chunk in iter_response_chunks(resp):
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
    Fallback-only candidates are always sorted AFTER all awake primaries,
    regardless of load: a fallback exists to serve when primaries are saturated,
    so even an idle fallback must wait its turn behind a busy primary. The
    dispatch loop walks this list and 429s fall through to the next entry, so
    the fallback naturally absorbs overflow only when every primary refused.

    Sleeping candidates come after everything awake (_CANDIDATE_ORDER): their
    gates read empty, so on load alone they'd win every tie and any backlog,
    and each of those requests would pay a cold load. A peer whose load couldn't
    be read comes after all of them: it may well be dead.

    In-flight counts only rise after a request acquires its gate, which happens
    after this decision, so a simultaneous burst sees every node tied at the
    same count. Always preferring self there would pin the whole burst to the
    entry node; rotating the tie spreads it across the group."""
    global _rr_counter
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

    ordered = []
    for unknown in (False, True):
        for fallback, tier in _CANDIDATE_ORDER:
            ordered += _sort_tier([c for c in candidates
                                   if bool(c.get("unknown")) == unknown
                                   and bool(c.get("fallback")) == fallback
                                   and c.get("tier", 0) == tier])
    return ordered


def dispatch_inference(model_name: str):
    """If `model_name` is part of a multi-node shared queue, route the current
    request to the least-loaded group node (round-robin on ties). Returns a Flask
    Response when forwarded to a peer, or None to handle it locally."""
    if _request_hops() >= 1:
        return None  # already routed here by a peer - the gate/overflow path takes over
    if not cl.is_cluster_enabled():
        return None

    # Translate a pretty name to its canonical file stem before any group
    # matching runs. Everything below (group keys, peer load maps, work-stealing)
    # speaks stems and share_queue_group aliases only; an untranslated pretty
    # name would match no group and silently disable cross-node routing, or
    # worse, substring-match the wrong one. Non-alias names pass through as-is.
    from core.model_alias import canonical_name
    model_name = canonical_name(model_name)
    # Likewise an engine's own served name for a virtual model (Strata's
    # "qwen3.8-flash-next-iq3_s") -> the model id its group is keyed by.
    from core.engines import virtual_model_key
    model_name = virtual_model_key(model_name) or model_name

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
    free slots, how many members are healthy / starting / sleeping, and a
    per-group `fallback` flag. Keyed by the effective group key (alias if set,
    else filename stem) so peers asking by alias find this node without a
    filename match. Read straight from gate state - cheap, no DB.

    `free` counts healthy members only: a sleeper's gate reads empty, and work
    moved onto it as "free" would pay a cold load. Members sharing a gate
    (siblings of one file) count it once.

    `fallback` is true for the group only when EVERY instance in its best tier
    is fallback-only; one primary there flips the flag back to false, matching
    the local routing rule (a primary sibling here will serve traffic, so this
    node should be treated as primary by peers too)."""
    from core.state import instances, instances_lock
    from proxy import get_gate
    with instances_lock:
        snap = [(i["id"], i.get("model_path", ""), dict(i.get("config") or {}), i.get("status"))
                for i in instances.values()]
    groups: dict[str, list] = {}
    for inst_id, model_path, cfg, status in snap:
        if not cfg.get("share_queue") or status in ("stopped", "stopping"):
            continue
        groups.setdefault(effective_group_key(model_path, cfg), []).append((inst_id, cfg, status))

    out: dict = {}
    for key, members in groups.items():
        cur = {"active": 0, "queued": 0, "free": 0, "max_concurrent": 0,
               "healthy": 0, "starting": 0, "sleeping": 0}
        counted, free_counted = set(), set()
        for inst_id, cfg, status in members:
            cur[{_TIER_HEALTHY: "healthy", _TIER_STARTING: "starting",
                 _TIER_SLEEPING: "sleeping"}[_member_tier(status)]] += 1
            gate = get_gate(inst_id)
            if gate:
                active, queued, mc = gate.active, gate.queued, gate.max_concurrent
            else:
                active, queued, mc = 0, 0, int(cfg.get("max_concurrent") or 0)
            # A gateless member is its own capacity; a gate may be several members'.
            slot_key = id(gate) if gate else inst_id
            if slot_key not in counted:
                counted.add(slot_key)
                cur["active"] += active
                cur["queued"] += queued
                cur["max_concurrent"] += mc
            if _member_tier(status) == _TIER_HEALTHY and slot_key not in free_counted:
                free_counted.add(slot_key)
                cur["free"] += max(0, mc - active) if mc else 1   # no gate => unlimited => free
        best = min(_member_tier(status) for _, _, status in members)
        cur["fallback"] = all(cfg.get("share_queue_fallback") for _, cfg, status in members
                              if _member_tier(status) == best)
        out[key] = cur
    return out


class _Unreachable(dict):
    """Falsy and empty like a peer with nothing to report, so code that only wants
    real capacity needn't care, while `is PEER_UNREACHABLE` tells the two apart: a
    peer hosting nothing and one that didn't answer mean different things."""


PEER_UNREACHABLE = _Unreachable()
_unreachable_peers: set[str] = set()


def _peer_load_map(node: dict) -> dict:
    """A peer's whole local-load map (briefly cached, so one query serves any
    model), or PEER_UNREACHABLE on a timeout, a non-200 or a bad body."""
    nid = node.get("node_id")
    now = _time.monotonic()
    with _load_cache_lock:
        cached = _load_cache.get(nid)
        data = cached[1] if cached and now - cached[0] < _LOAD_CACHE_TTL else None
    if data is None:
        why = None
        try:
            resp = cl.cluster_request(node, "GET", "/api/cluster/local-load", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
            else:
                why = f"HTTP {resp.status_code}"
        except Exception as e:
            why = str(e)
        if not isinstance(data, dict):
            why = why or "unexpected response"
            data = PEER_UNREACHABLE
        with _load_cache_lock:
            _load_cache[nid] = (now, data)
            changed = (why is not None) != (nid in _unreachable_peers)
            (_unreachable_peers.add if why is not None else _unreachable_peers.discard)(nid)
        # Once per change: this runs on every routing decision.
        if changed and why is not None:
            logger.warning("cluster: cannot read live load from %s: %s", node.get("node_name") or nid, why)
        elif changed:
            logger.info("cluster: live load from %s is readable again", node.get("node_name") or nid)
    return data


def _peer_live_load(node: dict, model_name: str):
    """Live load for `model_name` on a peer (briefly cached). Returns the
    per-model dict, None when the peer reports nothing for it, or PEER_UNREACHABLE.
    Matching is fuzzy/case-insensitive so a quant-suffixed request
    (gpt-oss-20b-Q4_K_M) still finds the lowercase-keyed entry."""
    data = _peer_load_map(node)
    return data if data is PEER_UNREACHABLE else _match_model_load(data, model_name)


def group_wants_wake(group_key: str, local: dict) -> bool:
    """Whether this node should wake a sleeper for the group: every healthy
    member anywhere is full with work queued, nothing is already loading, and of
    the nodes holding a sleeper this one has the lowest id, so the others don't
    wake one too. `local` is this node's own local_load_by_model() entry.

    Reads the whole group because the node holding the sleeper usually has an
    idle gate; the saturated node is the one with nothing left to wake. An older
    peer reports no counts: taken as one healthy member, which only ever makes
    this less eager. An unreachable peer is no evidence either way and is left out."""
    if local.get("free") or local.get("starting"):
        return False   # settled without asking any peer
    entries = [(cl.get_node_id(), local)]
    if cl.is_cluster_enabled():
        for node in get_storage().list_nodes():
            nid = node.get("node_id")
            if nid == entries[0][0] or not peer_is_routable(node):
                continue
            entry = _peer_load_map(node).get(group_key)
            if entry:
                entries.append((nid, entry))
    loads = [entry for _, entry in entries]
    if any(e.get("free") or e.get("starting") for e in loads):
        return False
    if not sum(e.get("queued") or 0 for e in loads):
        return False
    if not sum(e.get("healthy", 1) or 0 for e in loads):
        return False   # nothing is up to be saturated; a request wakes one itself
    return min(nid for nid, e in entries if e.get("sleeping")) == entries[0][0]


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
        if nid == self_id or not peer_is_routable(node):
            continue
        load = _peer_live_load(node, model_name)
        if not load or not _peer_can_serve(node, model_name):
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
    best = None    # (rank in _CANDIDATE_ORDER, backlog, node)
    for node in get_storage().list_nodes():
        nid = node.get("node_id")
        if nid == self_id or not peer_is_routable(node):
            continue
        load = _peer_live_load(node, model_name)
        if not load or not _peer_can_serve(node, model_name):
            continue  # peer doesn't host this model (or unreachable, or can't serve it)
        backlog = (load.get("active", 0) or 0) + (load.get("queued", 0) or 0) + _inflight_get(nid)
        # Even when evacuating, prefer alive primaries over fallbacks: the
        # operator marked the fallback as second-class for a reason (different
        # model, slower hardware), so use it only if no primary is alive. A
        # sleeper ranks last but still counts: this node's worker is dead, so a
        # cold load beats having nowhere to go.
        rank = _CANDIDATE_ORDER.index((bool(load.get("fallback")), _peer_tier(load)))
        if best is None or (rank, backlog) < best[:2]:
            best = (rank, backlog, node)
    return best[2] if best else None


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
