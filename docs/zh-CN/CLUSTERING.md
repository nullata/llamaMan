# 集群

*可选功能，默认关闭 —— 单节点部署完全不受影响。*

多个 llamaMan 部署可以作为**一个逻辑集群**协同工作：一个统一的仪表盘聚合每个节点的 GPU、实例和下载，支持跨节点启动/镜像拉取/下载以及多节点共享队列负载均衡。节点通过共享存储后端互相发现 —— 无需两两交换密钥。

**系统要求：**

- **共享存储后端** - 所有节点指向**同一个** `DATABASE_URL`（MariaDB/MySQL，参见 [Storage Backends](CONFIGURATION.md#storage-backends)）。JSON 存储是单机独立的，无法共享。
- **每个节点拥有唯一的 `LLAMAMAN_NODE_NAME`** - 每个节点的集群标识（而且任何安装方式本身都需要设置）。
- **每个节点使用相同的 `CLUSTER_SECRET`** - 用于所有节点间 HTTP 通信的 bearer token（作为 `X-Cluster-Secret` 发送）。
- **每个节点配置 `CLUSTER_ADVERTISE_URL`** 用于跨节点*操作* - 其他主机可路由访问的主机名/IP（不能是 `localhost`），例如 `http://srv1:5000`。未配置此项的节点会显示在仪表盘中，但处于只读状态。

在**每个**节点上配置（仅 `LLAMAMAN_NODE_NAME` 和 `CLUSTER_ADVERTISE_URL` 不同）：

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

每个节点每隔约 5 秒发送一次心跳；沉默时间超过 `CLUSTER_NODE_ONLINE_WINDOW_S`（默认 45 秒）的节点将显示为离线。可在 **Settings → Cluster** 下查看与管理。

部分设置仅作用于单个节点，因为它们与具体主机绑定：跟踪的 **Docker 镜像**以及模型上限逐出开关。其余所有内容均在整个集群内共享。活跃的共享队列分组别名（shared-queue group aliases）会在 `/api/tags` 和 `/v1/models` 中作为可选模型发布（集群范围内去重），客户端发送别名即可将其路由至服务该模型且负载最低的节点。

**分组 context 长度为集群全局生效。** 无论客户端请求哪个节点，`/api/show` 和 `/v1/models` 报告的队列分组 context 均为**所有活跃成员中的最小值（min）**。包含不同 ctx 值成员的分组会对外宣告较小的值，从而确保按该大小构建的 prompt 可以安全分发到任何目标节点 —— 较大成员上的额外空间仅保持不用。即使节点本地没有该分组的成员，它仍能响应（`context_length` 从共享实例状态中的对等节点运行时 ctx 填充，或从共享 DB 中的预设行填充）；在任何地方都没有活跃成员的分组仍返回 404 / 无条目。当该成员的 ctx 超过宣告的最小值时，操作员 UI 上的实例卡片会添加 `ctx <min>, capped by <node/model>`，方便你查看是哪个对等节点拉低了整个分组。

> **安全性：** 集群密钥允许任何对等节点在此节点上执行操作。请在受信任的网络或 TLS 保护下运行节点间流量。

集群相关环境变量：[Clustering variables](CONFIGURATION.md#clustering)。
