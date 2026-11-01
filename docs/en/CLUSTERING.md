# Clustering

*Optional, off by default - single-node installs are completely unaffected.*

Several llamaMan deployments act as **one logical cluster**: a single dashboard aggregating every node's GPUs, instances, and downloads, with cross-node launches/pulls/downloads and multi-node shared-queue load balancing. Nodes discover each other through the shared storage backend - no pairwise key exchange.

**Requirements:**

- **Shared storage backend** - every node points at the **same** `DATABASE_URL` (MariaDB/MySQL, see [Storage Backends](CONFIGURATION.md#storage-backends)). JSON is per-host and can't be shared.
- **Unique `LLAMAMAN_NODE_NAME` per node** - each node's cluster identity (and required for every install anyway).
- **The same `CLUSTER_SECRET` on every node** - the bearer token (sent as `X-Cluster-Secret`) for all node-to-node HTTP.
- **`CLUSTER_ADVERTISE_URL` per node** for cross-node *actions* - a hostname/IP routable from the **other** hosts (not `localhost`), e.g. `http://srv1:5000`. A node without one appears in the dashboard but is view-only.

Set on **each** node (only `LLAMAMAN_NODE_NAME` and `CLUSTER_ADVERTISE_URL` differ):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Each node heartbeats every ~5s; a node silent past `CLUSTER_NODE_ONLINE_WINDOW_S` (default 45s) shows offline. Inspect and manage under **Settings → Cluster**.

A few settings are scoped per node because they're host-specific: tracked **Docker images** and the model-cap eviction toggles. Everything else is shared cluster-wide. Live shared-queue group aliases are advertised as selectable models in `/api/tags` and `/v1/models` (deduped cluster-wide), so a client can send the alias and have it routed to the least-loaded node serving it.

**Group context length is cluster-wide.** `/api/show` and `/v1/models` report a queue group's context as the **min across all live members**, whichever node the client asks. A group with members at different ctx values advertises the smaller value so a prompt sized to it is safe for any dispatch target — the extra headroom on the bigger member simply goes unused. A node with no local member of the group still answers (`context_length` filled from the peer's runtime ctx in shared instance state, or from the preset row in the shared DB); a group with no live members anywhere still returns 404 / no entry. The instance card on the operator UI adds `ctx <min>, capped by <node/model>` when this member's ctx exceeds the advertised min, so you can see which peer is dragging the group down.

> **Security:** the cluster secret lets any peer drive actions on this node. Run node-to-node traffic over a trusted network or behind TLS.

Cluster-related environment variables: [Clustering variables](CONFIGURATION.md#clustering).

## Queue groups

A queue group pools instances under one **Queue group name** so a client can send that name to any node. The table shows what happens to each member from another node's point of view:

| Member state | Listed in `/api/tags` and `/v1/models` | Gets routed traffic |
|---|---|---|
| Healthy | Yes | First choice |
| Starting (loading) | Yes | After healthy members |
| Sleeping | Yes | Last, and only after every awake member, including fallbacks. The request wakes it. |
| Stopping or stopped | No | Never |
| On a node with no `CLUSTER_ADVERTISE_URL` (view-only) | No | Never |
| On an offline node (no heartbeat for 45s) | No | Never |

**`CLUSTER_ADVERTISE_URL` is required to receive routed work.** Without it a node can still send requests to its peers, but none can reach it.

**Sleep and stop mean different things here.** An idle timeout puts an instance to sleep, and a sleeping member wakes when traffic needs it. A stopped member stays stopped: an operator stop, an eviction and a crash all end there, and no request from another node restarts it. Asking the node that holds it gets a `503`, unless the name is also that member's own file, which still wakes it.

### Waking under load

Awake members queue requests rather than refuse them, so traffic alone would only reach a sleeper after a whole queue overflowed. llamaMan therefore wakes one early. When every healthy member in the group is full, requests are queued and nothing is already loading, for 3 polls in a row (about 15 seconds), one sleeping member wakes. Across the cluster, only the lowest-named node holding a sleeper does this, and at most one member per group per poll.

The wake is skipped, with one log line, if the model's weights don't fit in the node's free VRAM. The check adds up free memory across all GPUs and counts weights only, so a model that passes can still fail to load. If VRAM can't be read, the wake goes ahead.

Both settings are per node, through `POST /api/settings`:

| Setting | Default | Meaning |
|---|---|---|
| `eager_wake_enabled` | `true` | Whether this node wakes its sleepers under load |
| `eager_wake_polls` | `3` | Consecutive saturated polls (5 seconds each) before waking |

### When a group can't be served

If other nodes host the group but none can take a request, the client gets an error naming each node and why, not "model not found":

| Status | Code (OpenAI routes) | Meaning |
|---|---|---|
| `429` | `group_full` | At least one peer's queue is full. Retrying can work. |
| `503` | `group_unavailable` | The peers are offline, view-only, unreachable or not ready. |

The owning node returns these `503` codes for its own members, and an entry node tries its next member when it gets one:

| Code | Meaning |
|---|---|
| `model_stopping` | The member is shutting down |
| `group_stopped` | Every member on this node is stopped |
| `wake_failed` | The wake failed |
| `wake_timeout` | The member didn't become healthy in time |

### Troubleshooting a group that isn't visible

- **The JSON backend needs one shared `DATA_DIR`.** Nodes find each other through the registry, so separate directories mean each node sees no peers. Use MariaDB unless the directory really is shared.
- **A degraded database hides peers on purpose.** With the DB mirror serving reads, a node sees only itself and serves locally.
- **A failed load probe doesn't hide a group.** A peer that doesn't answer stays routable from its snapshot (up to 45 seconds old), but ranks behind every peer that did answer. The log records when probes to a peer start failing and when they recover.
