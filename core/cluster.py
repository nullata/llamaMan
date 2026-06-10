# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Cluster node identity and peer-to-peer transport.

Clustering lets several llamaMan deployments act as one logical cluster. This
module owns the local node's stable identity and the authenticated HTTP used for
node-to-node calls. Everything here is inert unless CLUSTER_ENABLED is set and a
CLUSTER_SECRET is configured, so single-node installs are unaffected.

Trust model: one shared CLUSTER_SECRET carried as a bearer token on every peer
call. Discovery is via the shared node registry in the storage backend (see
StorageBackend.register_node / list_nodes), so any node added anywhere becomes
visible to all nodes - clustering is transitive without pairwise key exchange.
"""

import secrets
import socket

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.poolmanager import PoolManager

import config
from config import logger


# TCP keepalive on every node-to-node call. A non-streaming forward (the common
# case: client sent stream=false) sees ZERO bytes on the wire while the peer
# queues and generates - just an idle established socket. Anything in the path
# (Docker conntrack, NAT, a firewall, a flaky NIC) can silently drop that idle
# socket without either end noticing; the entry then waits out its REQUEST_TIMEOUT
# for bytes that will never arrive and returns 504 to the client even though the
# peer's request_log shows a clean 200. Keepalive surfaces dead sockets as a
# ConnectionError within ~60s instead, which the dispatch code handles by trying
# another peer / falling back to local.
_KEEPALIVE_IDLE_S = 30   # start probing after 30s of no traffic
_KEEPALIVE_INTVL_S = 10  # probe every 10s
_KEEPALIVE_CNT = 3       # 3 missed probes -> dead


class _KeepAliveAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        opts = list(HTTPConnection.default_socket_options) + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        ]
        if hasattr(socket, "TCP_KEEPIDLE"):   # Linux
            opts += [
                (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, _KEEPALIVE_IDLE_S),
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, _KEEPALIVE_INTVL_S),
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _KEEPALIVE_CNT),
            ]
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block,
            socket_options=opts, **kwargs,
        )


_session = requests.Session()
_session.mount("http://", _KeepAliveAdapter())
_session.mount("https://", _KeepAliveAdapter())


# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------

def get_node_id() -> str:
    """Return this node's stable identity.

    Sourced from the LLAMAMAN_NODE_NAME env var (validated at config import, so
    by the time anyone calls this it is guaranteed non-empty). The same string
    is used as both the registry key (node_id) and the display name - operators
    pick something meaningful (a hostname, a uuid, anything) and live with it,
    because changing it later orphans this node's existing rows.
    """
    return config.LLAMAMAN_NODE_NAME


def get_advertise_url() -> str:
    return (config.CLUSTER_ADVERTISE_URL or "").rstrip("/")


def is_cluster_enabled() -> bool:
    """Clustering is active only when explicitly enabled AND a secret exists.

    Without a secret, peers could not authenticate, so we treat that as off and
    warn once at the call site rather than silently half-enabling.
    """
    if not config.CLUSTER_ENABLED:
        return False
    if not config.CLUSTER_SECRET:
        logger.warning("CLUSTER_ENABLED is set but CLUSTER_SECRET is empty - clustering stays disabled")
        return False
    return True


def local_node_descriptor() -> dict:
    """Identity fields this node publishes into the cluster registry."""
    from core.gpu import get_vendor
    nid = get_node_id()
    return {
        "node_id": nid,
        "node_name": nid,
        "advertise_url": get_advertise_url(),
        "vendor": get_vendor() or "",
        "llama_image": config.LLAMA_IMAGE,
    }


# ---------------------------------------------------------------------------
# Peer authentication & transport
# ---------------------------------------------------------------------------

# Node-to-node trust travels in its own header, NOT as a Bearer token, so the
# cluster secret is never accepted as a client credential. Clients use API keys.
CLUSTER_SECRET_HEADER = "X-Cluster-Secret"


def verify_cluster_secret(token: str) -> bool:
    """Constant-time check of the X-Cluster-Secret header against the secret."""
    if not config.CLUSTER_SECRET or not token:
        return False
    return secrets.compare_digest(token, config.CLUSTER_SECRET)


def cluster_request(node: dict, method: str, path: str, *, timeout: float = 10, **kwargs):
    """Make an authenticated node-to-node HTTP call to a peer.

    `node` is a registry dict carrying `advertise_url`; `path` is an absolute
    request path. The shared cluster secret is sent in X-Cluster-Secret (node
    trust); any caller-supplied Authorization (e.g. the client's API key being
    forwarded on inference dispatch) is preserved untouched. Returns the
    requests.Response; raises on transport failure so callers handle peer-down.
    """
    base = (node.get("advertise_url") or "").rstrip("/")
    if not base:
        raise ValueError(f"node {node.get('node_id')} has no advertise_url")
    headers = dict(kwargs.pop("headers", {}))
    if config.CLUSTER_SECRET:
        headers.setdefault(CLUSTER_SECRET_HEADER, config.CLUSTER_SECRET)
    return _session.request(method, base + path, headers=headers, timeout=timeout, **kwargs)


# ---------------------------------------------------------------------------
# Registry convenience
# ---------------------------------------------------------------------------

def publish_heartbeat(snapshot: dict | None = None) -> None:
    """Upsert this node into the shared registry with a fresh heartbeat.

    No-op when clustering is disabled. Called from the background poller with the
    node's current stats/instances snapshot, and once on join without one.
    """
    if not is_cluster_enabled():
        return
    from storage import get_storage
    try:
        get_storage().register_node(local_node_descriptor(), snapshot=snapshot)
    except Exception as e:
        logger.warning("cluster heartbeat failed: %s", e)


def leave_cluster() -> None:
    if not config.CLUSTER_SECRET:
        return
    from storage import get_storage
    try:
        get_storage().remove_node(get_node_id())
    except Exception as e:
        logger.warning("cluster leave failed: %s", e)
