# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Knowledge base business logic: chunking, embedding, ingest, search.

No Flask here. Storage access goes through get_storage()'s kb_* methods
(see storage/base.py); MariaDB >= 11.8 is the only supported backend and
that gate lives in the backend layer, surfaced here via kb_feature_supported.

Embeddings are computed by a llamaman-managed llama.cpp instance launched
with --embeddings (config.embedding_model true), selected by the shared
`kb_embedding_instance` setting. Model identity and dimensions are locked
in kb_meta on the first successful embed; a later instance with different
dims refuses to search/ingest until an explicit re-embed.
"""

import hashlib
import threading
import time

from storage.base import KBNotSupportedError, KBUnavailableError

__all__ = [
    "KBNotSupportedError", "KBUnavailableError",
    "kb_feature_supported", "assert_kb_ready",
    "embed_texts", "chunk_text", "ingest_document", "search", "find_topic",
    "reembed_all", "reembed_status", "MCP_INGEST_MAX_BYTES",
]

# --- availability -----------------------------------------------------------


def kb_feature_supported() -> tuple[bool, str]:
    """(supported, reason). reason: "" | "json_backend" | "mariadb_old:<ver>"
    | "db_offline". Drives UI tooltips and server-side enable validation.
    Never raises; safe to call from the tab status endpoint only — nothing on
    pre-existing request paths may call this."""
    from storage import get_storage
    storage = get_storage()
    try:
        if storage.kb_available():
            return True, ""
    except Exception:
        return False, "db_offline"
    ver = None
    try:
        ver = storage.kb_server_version()
    except Exception:
        pass
    if ver is None:
        # JSON backend (or unreachable): kb_server_version is None there.
        try:
            from storage.mariadb_backend import MariaDBBackend
            if not isinstance(storage, MariaDBBackend) and not _wraps_mariadb(storage):
                return False, "json_backend"
        except Exception:
            return False, "json_backend"
        return False, "db_offline"
    return False, f"mariadb_old:{ver[0]}.{ver[1]}"


def _wraps_mariadb(storage) -> bool:
    from storage.mariadb_backend import MariaDBBackend
    from storage.resilient import ResilientBackend
    if not isinstance(storage, ResilientBackend):
        return False
    primary = getattr(storage, "_primary", None)
    return isinstance(primary, MariaDBBackend)


def assert_kb_ready() -> None:
    """Raise KBUnavailableError unless the KB can serve right now and the
    master switch is on."""
    ok, reason = kb_feature_supported()
    if not ok:
        raise KBUnavailableError(_humanize(reason))
    if not _kb_enabled():
        raise KBUnavailableError("knowledge base is disabled (Knowledge settings tab)")


def _kb_enabled() -> bool:
    from storage import get_storage
    try:
        return bool(get_storage().get_settings().get("kb_enabled", False))
    except Exception:
        return False


def _humanize(reason: str) -> str:
    if reason == "json_backend":
        return "knowledge base requires the MariaDB database backend"
    if reason.startswith("mariadb_old:"):
        return ("knowledge base requires MariaDB 11.8 or newer "
                f"(current: {reason.split(':', 1)[1]})")
    if reason == "db_offline":
        return "knowledge base unavailable: database offline"
    return f"knowledge base unavailable ({reason})" if reason else \
        "knowledge base unavailable"


# --- chunking ---------------------------------------------------------------

CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200


def chunk_text(text: str) -> list[str]:
    """Deterministic, dependency-free chunker: split on paragraph boundaries,
    pack to ~CHUNK_TARGET_CHARS with ~CHUNK_OVERLAP_CHARS overlap. Identical
    input always yields identical chunks (re-ingest idempotency depends on it).

    Overlap is taken as the tail of the previous chunk snapped to a word
    boundary; a paragraph longer than the target is hard-split with the same
    overlap policy.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n\n")]
    paras = [p for p in paras if p]
    if not paras:
        return []

    # Split over-long paragraphs into target-sized pieces first.
    units: list[str] = []
    for p in paras:
        if len(p) <= CHUNK_TARGET_CHARS:
            units.append(p)
            continue
        start = 0
        while start < len(p):
            end = min(len(p), start + CHUNK_TARGET_CHARS)
            if end < len(p):
                # snap back to a word boundary so overlap snapping is sane
                cut = p.rfind(" ", start + CHUNK_TARGET_CHARS // 2, end)
                if cut > start:
                    end = cut
            units.append(p[start:end].strip())
            start = max(start + 1, end - CHUNK_OVERLAP_CHARS)
            # the next unit's head is overlap text; advance past it
            while start < len(p) and p[start] == " ":
                start += 1

    chunks: list[str] = []
    cur = ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + 2 + len(u) <= CHUNK_TARGET_CHARS:
            cur = cur + "\n\n" + u
        else:
            chunks.append(cur)
            tail = _overlap_tail(cur)
            cur = (tail + "\n\n" + u) if tail else u
    if cur:
        chunks.append(cur)
    return [c for c in (c.strip() for c in chunks) if c]


def _overlap_tail(text: str) -> str:
    tail = text[-CHUNK_OVERLAP_CHARS:]
    sp = tail.find(" ")
    if 0 <= sp < len(tail) - 1:
        tail = tail[sp + 1:]
    return tail.strip()


# --- embedding --------------------------------------------------------------

EMBED_CONNECT_TIMEOUT = 5.0
EMBED_READ_TIMEOUT = 30.0
EMBED_BATCH = 32


_HEALTHY = ("running", "ready", "healthy")
_WAKEABLE = ("sleeping", "stopped")
_STARTING = ("starting",)

# Sticky auto-launch bounded like auto-restart-on-crash (core/monitoring.py):
# a broken sticky config that fails to launch can't hot-loop forever.
_AUTOLAUNCH_MAX = 3
_AUTOLAUNCH_WINDOW_S = 600
_AUTOLAUNCH_TIMEOUT_S = 180
_autolaunch_lock = threading.Lock()
_autolaunch_log: dict[str, list[float]] = {}


def _peer_participates(node: dict) -> bool:
    """A peer only serves KB traffic (forwarded embed OR auto-launch host) if
    it has both Knowledge base and Expose over MCP turned on — that's the
    explicit opt-in to the group. Reads the snapshot's kb block."""
    snap = node.get("snapshot") or {}
    kb = snap.get("kb") or {}
    return bool(kb.get("enabled")) and bool(kb.get("mcp"))


def _stamp_sticky(inst: dict) -> None:
    """Record 'this node last ran this embedding model, with this config' so
    a later cluster-wide miss can relaunch it here. Best-effort: KB embed
    calls never fail because sticky writing failed."""
    try:
        from core.cluster import get_node_id
        from storage import get_storage
        node_id = get_node_id() or ""
    except Exception:
        return
    if not node_id:
        return
    config = {
        "model_path": inst.get("model_path"),
        "config": inst.get("config") or {},
        "display_name": inst.get("display_name") or "",
    }
    try:
        import json
        get_storage().kb_meta_set(
            sticky_embedding_node=node_id,
            sticky_embedding_config=json.dumps(config, default=str),
        )
    except Exception:
        pass


def _read_sticky() -> tuple[str, dict] | tuple[None, None]:
    from storage import get_storage
    try:
        meta = get_storage().kb_meta_get()
    except Exception:
        return None, None
    node_id = meta.get("sticky_embedding_node") or ""
    raw = meta.get("sticky_embedding_config") or ""
    if not node_id or not raw:
        return None, None
    try:
        import json
        return node_id, json.loads(raw)
    except Exception:
        return None, None


def _clear_sticky() -> None:
    from storage import get_storage
    try:
        get_storage().kb_meta_set(sticky_embedding_node="",
                                  sticky_embedding_config="")
    except Exception:
        pass


def _check_autolaunch_budget(model_name: str) -> bool:
    """Rate limit sticky auto-launch attempts per embedding model, mirroring
    the auto-restart-on-crash guard (monitoring._maybe_auto_restart)."""
    now = time.time()
    with _autolaunch_lock:
        recent = [t for t in _autolaunch_log.get(model_name, ())
                  if now - t < _AUTOLAUNCH_WINDOW_S]
        if len(recent) >= _AUTOLAUNCH_MAX:
            _autolaunch_log[model_name] = recent
            return False
        recent.append(now)
        _autolaunch_log[model_name] = recent
        return True


def _find_sticky_node(nodes: list[dict], sticky_node_id: str) -> dict | None:
    for n in nodes:
        if n.get("node_id") == sticky_node_id and n.get("advertise_url") \
                and _peer_participates(n):
            return n
    return None


def _launch_on_peer(node: dict, sticky_config: dict, model_name: str,
                    timeout_s: float = _AUTOLAUNCH_TIMEOUT_S) -> None:
    """POST /api/instances on a peer to relaunch the embedding model, then
    poll /api/instances/<id> until healthy or the timeout burns out. Raises
    KBUnavailableError on any failure (transport, launch refusal, timeout).
    Marks the launch config with `_kb_autolaunch` so an operator can tell it
    apart from user-created instances."""
    from core.cluster import cluster_request
    model_path = sticky_config.get("model_path")
    if not model_path:
        raise KBUnavailableError("kb: sticky config missing model_path")
    payload = {
        "model_path": model_path,
        "display_name": sticky_config.get("display_name") or "",
        "config": {**(sticky_config.get("config") or {}),
                   "_kb_autolaunch": True,
                   "embedding_model": True},
    }
    node_tag = node.get("node_id") or node.get("advertise_url") or "?"
    try:
        r = cluster_request(node, "POST", "/api/instances",
                            json=payload, timeout=(5, 30))
    except Exception as e:
        raise KBUnavailableError(
            f"kb: auto-launch on {node_tag} failed: {e}")
    if r.status_code not in (200, 201, 202):
        raise KBUnavailableError(
            f"kb: auto-launch on {node_tag} refused ({r.status_code}): "
            f"{r.text[:200]}")
    try:
        inst_id = (r.json() or {}).get("id") or (r.json() or {}).get("inst_id")
    except ValueError:
        inst_id = None
    if not inst_id:
        raise KBUnavailableError(
            f"kb: auto-launch on {node_tag} returned no instance id")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            g = cluster_request(node, "GET", f"/api/instances/{inst_id}",
                                timeout=(3, 10))
            if g.status_code == 200:
                status = ((g.json() or {}).get("status") or "").lower()
                if status in _HEALTHY:
                    return
                if status in ("failed", "error", "stopped"):
                    raise KBUnavailableError(
                        f"kb: auto-launch on {node_tag} entered {status}")
        except KBUnavailableError:
            raise
        except Exception:
            pass
        time.sleep(2.0)
    raise KBUnavailableError(
        f"kb: auto-launch on {node_tag} did not become healthy within "
        f"{int(timeout_s)}s")


def _resolve_embed_target():
    """Resolve where to run the next embed call.

    Returns one of:
      ("local", inst_dict, host, port) — an instance on this node
      ("peer", node_dict)               — a peer with the model running,
                                          or sleeping/stopped there (node
                                          dict carries _kb_waking=True)

    Preference:
      1. Local healthy match  → use it directly.
      2. Local sleeping/stopped match → wake in-process and reuse; a local
         match already "starting" → wait for it to become healthy.
      3. Peer that opted in (kb + mcp on) with a healthy instance → forward;
         else one with a sleeping/stopped/starting instance → forward, and
         the peer's own resolver wakes or waits for it (step 2 there).
      4. Sticky node (kb_meta) if opted in → auto-launch and retry.
      Legacy `kb_embedding_instance` (per-node UUID) matches step 1 only.

    Raises KBUnavailableError otherwise.
    """
    from core.state import instances
    from storage import get_storage
    try:
        settings = get_storage().get_settings()
    except Exception as e:
        raise KBUnavailableError(f"kb: cannot read settings ({e})")
    wanted_model = (settings.get("kb_embedding_model") or "").strip()
    legacy_inst_id = (settings.get("kb_embedding_instance") or "").strip()
    if not wanted_model and not legacy_inst_id:
        raise KBUnavailableError(
            "kb: no embedding model configured (MCP settings tab)")

    def _pick_local(statuses):
        with instances_lock_guard():
            pairs = list(instances.items())
        for inst_id, inst in pairs:
            cfg = inst.get("config") or {}
            if not cfg.get("embedding_model"):
                continue
            if inst.get("status") not in statuses:
                continue
            model_name = inst.get("model_name") or ""
            if wanted_model and model_name == wanted_model:
                return inst_id, inst
            if not wanted_model and legacy_inst_id == inst_id:
                return inst_id, inst
        return None, None

    inst_id, inst = _pick_local(_HEALTHY)
    if inst is None:
        # Local sleeping/stopped instance? Wake it in-process (blocks until
        # healthy) and re-pick. relaunch_inactive_instance is idempotent so
        # a race with the poller waking the same instance is safe.
        wake_id, wake_inst = _pick_local(_WAKEABLE)
        if wake_inst is not None:
            try:
                from api.instances import relaunch_inactive_instance
                relaunch_inactive_instance(wake_id)
            except Exception:
                pass
            inst_id, inst = _pick_local(_HEALTHY)
    if inst is None:
        # Already loading — another request (or the wake above losing a race
        # to one) is bringing it up, and relaunch_inactive_instance returns
        # immediately for a "starting" instance. Wait for it rather than fail.
        start_id, start_inst = _pick_local(_STARTING)
        if start_inst is not None:
            host = start_inst.get("_server_host") or "127.0.0.1"
            port = start_inst.get("_server_port") or start_inst.get("port")
            if port:
                from api.instances import wait_for_healthy
                from config import MODEL_LOAD_TIMEOUT
                if wait_for_healthy(host, int(port), timeout=MODEL_LOAD_TIMEOUT):
                    # /health is ok; the waker may not have flipped status to
                    # "healthy" yet, so use the instance directly.
                    inst_id, inst = start_id, start_inst

    if inst is not None:
        host = inst.get("_server_host") or "127.0.0.1"
        port = inst.get("_server_port") or inst.get("port")
        if not port:
            raise KBUnavailableError(
                f"kb: instance {inst_id} has no server port")
        return ("local", dict(inst), host, int(port))

    # Peer search — model-name mode only. Legacy per-node UUIDs never
    # resolve across nodes. A healthy peer instance wins; failing that, a
    # peer with the model sleeping/stopped/starting gets the forward and
    # wakes or waits for it in its own resolver (the local branches above) —
    # relaunching a fresh instance there via sticky would duplicate it.
    peer_node = None
    peer_waking = False
    if wanted_model:
        try:
            nodes = get_storage().list_nodes()
        except Exception:
            nodes = []
        try:
            from core.cluster import get_node_id
            self_id = get_node_id()
        except Exception:
            self_id = None
        for node in nodes:
            if self_id and node.get("node_id") == self_id:
                continue
            if not node.get("advertise_url"):
                continue
            if not _peer_participates(node):
                continue
            snap = node.get("snapshot") or {}
            for pi in snap.get("instances") or []:
                cfg = pi.get("config") or {}
                if not cfg.get("embedding_model"):
                    continue
                if (pi.get("model_name") or "") != wanted_model:
                    continue
                status = pi.get("status")
                if status in _HEALTHY:
                    peer_node, peer_waking = node, False
                    break
                if status in _WAKEABLE + _STARTING and peer_node is None:
                    peer_node, peer_waking = node, True
            if peer_node is not None and not peer_waking:
                break
    if peer_node is not None:
        target = dict(peer_node)
        if peer_waking:
            # Tells _embed_peer to allow for a model load on the far side.
            target["_kb_waking"] = True
        return ("peer", target)

    # Nothing running or sleeping on any node. Try sticky auto-launch — but only
    # in model-name mode (legacy UUID sticky doesn't make sense across a
    # restart-recreate that changes the UUID).
    if wanted_model:
        sticky_node_id, sticky_config = _read_sticky()
        if sticky_node_id and sticky_config:
            try:
                nodes  # noqa: reuse from above if defined
            except NameError:
                try:
                    nodes = get_storage().list_nodes()
                except Exception:
                    nodes = []
            target_node = _find_sticky_node(nodes, sticky_node_id)
            if target_node is not None and _check_autolaunch_budget(wanted_model):
                try:
                    _launch_on_peer(target_node, sticky_config, wanted_model)
                except KBUnavailableError:
                    # Repeated failures burn the budget; when exhausted the
                    # sticky is cleared so a later manual re-selection isn't
                    # fighting a phantom target.
                    with _autolaunch_lock:
                        used = len(_autolaunch_log.get(wanted_model, ()))
                    if used >= _AUTOLAUNCH_MAX:
                        _clear_sticky()
                    raise
                # Launch succeeded — a peer is now serving the model.
                return ("peer", dict(target_node))

    tag = wanted_model or f"instance {legacy_inst_id}"
    raise KBUnavailableError(
        f"kb: no live embedding instance for {tag} on this cluster")


def instances_lock_guard():
    from core.state import instances_lock
    return instances_lock


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed via the selected instance — locally if a matching one runs on
    this node, otherwise forwarded to a peer node that has one.

    Each local POST is booked in request_log so embedding traffic shows up
    in the same stream inference does. Peer-forwarded calls are logged at
    the peer, not here (avoids double-counting).
    """
    if not texts:
        return []
    target = _resolve_embed_target()
    if target[0] == "local":
        _, inst, host, port = target
        return _embed_local(inst, host, port, texts)
    if target[0] == "peer":
        _, node = target
        return _embed_peer(node, texts)
    raise KBUnavailableError("kb: unexpected resolver result")


def _embed_local(inst: dict, host: str, port: int,
                 texts: list[str]) -> list[list[float]]:
    """Direct call to a llama-server on this node — the OpenAI-shape batch
    endpoint first, with a per-text fallback to llama.cpp's native /embedding.
    On success stamps the sticky pointer so cluster-wide misses can relaunch
    the model here later."""
    import requests
    base = f"http://{host}:{port}"
    timeout = (EMBED_CONNECT_TIMEOUT, EMBED_READ_TIMEOUT)
    inst_id = inst.get("id") or ""
    model_label = inst.get("model_name") or ""
    try:
        vectors = _embed_openai_shape(base, texts, timeout, inst_id, model_label)
    except requests.RequestException:
        vectors = _embed_llama_native(base, texts, timeout, inst_id, model_label)
    _stamp_sticky(inst)
    return vectors


def _embed_peer(node: dict, texts: list[str]) -> list[list[float]]:
    """Forward the batch to a peer that has the model running, using the
    cluster-secret transport. Peer errors surface as KBUnavailableError so
    the tool layer can turn them into a friendly isError result rather than
    an RPC crash."""
    from core.cluster import cluster_request
    node_tag = node.get("node_id") or node.get("advertise_url") or "?"
    read_timeout = EMBED_READ_TIMEOUT * 2
    if node.get("_kb_waking"):
        # The peer wakes its sleeping instance before embedding, which
        # blocks for up to its MODEL_LOAD_TIMEOUT.
        from config import MODEL_LOAD_TIMEOUT
        read_timeout += MODEL_LOAD_TIMEOUT
    try:
        r = cluster_request(node, "POST", "/api/kb/_peer_embed",
                            json={"texts": texts},
                            timeout=(EMBED_CONNECT_TIMEOUT, read_timeout))
    except Exception as e:
        raise KBUnavailableError(
            f"kb: peer embed to {node_tag} failed: {e}")
    if r.status_code != 200:
        # Peer says the model isn't there either / breaker open / etc.
        raise KBUnavailableError(
            f"kb: peer {node_tag} returned {r.status_code} on embed")
    try:
        vectors = r.json().get("vectors")
    except ValueError:
        raise KBUnavailableError(
            f"kb: peer {node_tag} returned non-JSON on embed")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise KBUnavailableError(
            f"kb: peer {node_tag} returned wrong vector count")
    return vectors


def _embed_openai_shape(base: str, texts: list[str], timeout,
                        inst_id: str, model_label: str) -> list[list[float]]:
    import requests
    from core.request_log import record_request, finalize_async
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        body = {"input": batch}
        handle = record_request(body, endpoint="kb-embed",
                                path="/v1/embeddings",
                                inst_id=inst_id, model=model_label)
        r = None
        try:
            r = requests.post(f"{base}/v1/embeddings",
                              json=body, timeout=timeout)
            usage = None
            if handle and r.ok:
                try:
                    usage = (r.json() or {}).get("usage")
                except ValueError:
                    usage = None
            if handle:
                handle.set_response(status_code=r.status_code, usage=usage)
            r.raise_for_status()
            data = r.json().get("data", [])
            if len(data) != len(batch):
                raise RuntimeError("embedding response missing items")
            out.extend(item["embedding"] for item in
                       sorted(data, key=lambda d: d.get("index", 0)))
        except Exception as e:
            if handle:
                handle.set_error(r.status_code if r is not None else 0, str(e))
            raise
        finally:
            finalize_async(handle)
    return out


def _embed_llama_native(base: str, texts: list[str], timeout,
                        inst_id: str, model_label: str) -> list[list[float]]:
    import requests
    from core.request_log import record_request, finalize_async
    out = []
    for t in texts:
        body = {"content": t}
        handle = record_request(body, endpoint="kb-embed",
                                path="/embedding",
                                inst_id=inst_id, model=model_label)
        r = None
        try:
            r = requests.post(f"{base}/embedding",
                              json=body, timeout=timeout)
            if handle:
                handle.set_response(status_code=r.status_code)
            r.raise_for_status()
            emb = r.json().get("embedding")
            if emb is None:
                raise RuntimeError("embedding response missing 'embedding'")
            out.append(emb)
        except Exception as e:
            if handle:
                handle.set_error(r.status_code if r is not None else 0, str(e))
            raise
        finally:
            finalize_async(handle)
    return out


# --- model/dims lock ---------------------------------------------------------


def _lock_or_check_model_meta(vec_len: int, model_label: str) -> None:
    """Record embedding dims/model on first embed; refuse on mismatch so
    searches never mix vector spaces. Falsy or non-numeric locked_dims is
    treated as "never locked" so a reset-but-never-relocked meta (empty KB
    hitting Re-embed) self-heals on the next embed."""
    from storage import get_storage
    meta = get_storage().kb_meta_get()
    locked_dims = meta.get("embedding_dims")
    if not locked_dims:
        get_storage().kb_meta_set(embedding_dims=str(vec_len),
                                  embedding_model=model_label)
        return
    try:
        locked_n = int(locked_dims)
    except (TypeError, ValueError):
        get_storage().kb_meta_set(embedding_dims=str(vec_len),
                                  embedding_model=model_label)
        return
    if locked_n != int(vec_len):
        raise KBUnavailableError(
            f"kb: embedding model mismatch — stored vectors are "
            f"{locked_dims}-d, current instance produces {vec_len}-d. "
            "Run Re-embed to rebuild the index.")


# --- ingest / search ---------------------------------------------------------

MCP_INGEST_MAX_BYTES = 200 * 1024  # per-call content cap (thread starvation guard)


def ingest_document(topic_id: int, title: str, content: str,
                    source: str = "", max_bytes: int | None = None) -> dict:
    """Upsert a document (sha256 dedup — unchanged content is a no-op that
    skips embedding entirely) then chunk+embed+store. Returns
    {document_id, chunks, unchanged}."""
    assert_kb_ready()
    if max_bytes is not None and len(content.encode("utf-8")) > max_bytes:
        raise KBUnavailableError(
            f"kb: document too large ({max_bytes} bytes max per ingest)")
    from storage import get_storage
    storage = get_storage()
    result = storage.kb_upsert_document(topic_id, title, content, source)
    doc_id = result["id"]
    if result["unchanged"]:
        return {"document_id": doc_id, "chunks": 0, "unchanged": True}
    chunks = chunk_text(content)
    if not chunks:
        return {"document_id": doc_id, "chunks": 0, "unchanged": False}
    vectors = embed_texts(chunks)
    _lock_or_check_model_meta(len(vectors[0]), _current_model_label())
    inserted = storage.kb_insert_chunks(
        doc_id, [(i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors), 1)])
    return {"document_id": doc_id, "chunks": inserted, "unchanged": False}


def _current_model_label() -> str:
    """Label recorded in kb_meta on first embed. Model-name selection
    (kb_embedding_model) is the cluster-wide setting and wins; the legacy
    per-node instance UUID is only consulted when no model name is set."""
    from core.state import instances
    from storage import get_storage
    try:
        settings = get_storage().get_settings()
        model = (settings.get("kb_embedding_model") or "").strip()
        if model:
            return model
        inst_id = settings.get("kb_embedding_instance", "") or ""
        with instances_lock_guard():
            inst = instances.get(inst_id) or {}
        return inst.get("model_name") or inst_id
    except Exception:
        return ""


def find_topic(storage, name: str, visible_to: str | None = None) -> dict | None:
    """Resolve a topic by name (case-insensitive) among the topics visible to
    a key. A key's own topic wins over a shared one with the same name, so
    a private "notes" shadows the pool's "notes" for that key."""
    matches = [t for t in storage.kb_list_topics(visible_to=visible_to)
               if t["name"].lower() == name.lower()]
    if not matches:
        return None
    if visible_to is not None:
        own = [t for t in matches if t.get("owner_key_id") == visible_to]
        if own:
            return own[0]
    shared = [t for t in matches if not t.get("owner_key_id")]
    return (shared or matches)[0]


def search(query: str, topic_id: int | None = None, limit: int = 8,
           topic_name: str | None = None,
           visible_to: str | None = None) -> list[dict]:
    """Semantic search. Returns [{document_id, title, topic, seq, text,
    score}] with score clamped to [0,1] (max(0, 1 - cosine_dist/2)).
    topic_name resolves a topic by name (MCP tool ergonomics). visible_to
    limits results to one API key's own topics plus the shared pool."""
    assert_kb_ready()
    query = (query or "").strip()
    if not query:
        raise KBUnavailableError("kb: empty query")
    from storage import get_storage
    storage = get_storage()
    if topic_id is None and topic_name:
        t = find_topic(storage, topic_name, visible_to)
        if t is None:
            raise KBUnavailableError(f"kb: unknown topic '{topic_name}'")
        topic_id = t["id"]
    meta = storage.kb_meta_get()
    locked = meta.get("embedding_dims")
    if not locked:
        raise KBUnavailableError("kb: knowledge base is empty (nothing embedded yet)")
    try:
        locked_n = int(locked)
    except (TypeError, ValueError):
        raise KBUnavailableError("kb: knowledge base is empty (nothing embedded yet)")
    qv = embed_texts([query])[0]
    if locked_n != len(qv):
        raise KBUnavailableError(
            f"kb: embedding model mismatch — stored vectors are "
            f"{locked}-d, current instance produces {len(qv)}-d")
    rows = storage.kb_search(qv, topic_id=topic_id,
                             limit=max(1, min(int(limit), 50)),
                             visible_to=visible_to)
    out = []
    for r in rows:
        d = r["distance"]
        out.append({"document_id": r["document_id"], "title": r["title"],
                    "topic": r["topic"], "seq": r["seq"], "text": r["text"],
                    "score": round(max(0.0, 1.0 - d / 2.0), 4)})
    return out


# --- re-embed job -------------------------------------------------------------

_reembed_lock = threading.Lock()
_reembed_job: dict = {"running": False, "done": 0, "total": 0, "error": None,
                      "started_at": None, "finished_at": None}


def reembed_status() -> dict:
    with _reembed_lock:
        return dict(_reembed_job)


def reembed_all() -> dict:
    """Start the rebuild job (drop chunks, re-chunk+embed every document).
    Returns the job dict; raises RuntimeError if one is already running
    (single-flight in-process; the DROP/CREATE itself is additionally guarded
    by the cluster-wide GET_LOCK inside the backend)."""
    assert_kb_ready()
    with _reembed_lock:
        if _reembed_job["running"]:
            raise RuntimeError("re-embed already running")
        _reembed_job.update(running=True, done=0, total=0, error=None,
                            started_at=time.time(), finished_at=None)
    t = threading.Thread(target=_reembed_worker, name="kb-reembed", daemon=True)
    t.start()
    return reembed_status()


def _reembed_worker() -> None:
    from storage import get_storage
    storage = get_storage()
    try:
        docs = storage.kb_list_documents()
        with _reembed_lock:
            _reembed_job["total"] = len(docs)
        storage.kb_clear_chunks()
        # New vector space: reset the dims/model lock so the first embed of
        # the (possibly changed) model re-locks it. Skip when there are no
        # docs — otherwise the reset writes '' to kb_meta and the next ingest
        # crashes in _lock_or_check_model_meta trying to int('').
        if docs:
            storage.kb_meta_set(embedding_dims="", embedding_model="")
        for doc in docs:
            full = storage.kb_get_document(doc["id"])
            if full is None:
                continue
            chunks = chunk_text(full["content"])
            if chunks:
                vectors = embed_texts(chunks)
                _lock_or_check_model_meta(len(vectors[0]), _current_model_label())
                storage.kb_insert_chunks(doc["id"],
                                         [(i, c, v) for i, (c, v) in
                                          enumerate(zip(chunks, vectors), 1)])
            with _reembed_lock:
                _reembed_job["done"] += 1
    except Exception as e:
        with _reembed_lock:
            _reembed_job["error"] = str(e)
    finally:
        with _reembed_lock:
            _reembed_job["running"] = False
            _reembed_job["finished_at"] = time.time()
