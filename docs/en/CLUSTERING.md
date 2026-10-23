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
