# 配置与存储

裸机运行安装、认证模型详解、GPU 统计信息、请求记录、存储后端以及完整的环境变量参考。

## 目录

- [裸机运行](#running-bare-metal) · [认证详情](#authentication-detail) · [GPU 统计信息](#gpu-统计信息) · [请求记录与统计](#request-recording--stats) · [存储后端](#storage-backends) · [数据库故障应对 (本地镜像)](#surviving-a-database-outage-local-mirror) · [环境变量](#environment-variables)

<a id="running-bare-metal"></a>
## 裸机运行

适用于开发环境或直接在 Docker 中运行管理器本身不太方便的宿主机环境（如 WSL）。llama-server 仍运行在容器中。

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

gunicorn 下的单个进程即可在 `:5000` 上提供 UI/API 服务，并在 `:42069` 上提供 proxy 服务（不使用 gunicorn 的 `python app.py` 仅在 `:5000` 上运行 Flask 开发服务器）。容器与裸机模式会自动检测；如果检测结果与你的运行环境不符，可显式设置 `LLAMAMAN_IN_DOCKER=true` 或 `false`。

<a id="authentication-detail"></a>
## 认证详情

两种凭据类型：

- **Session cookie** 用于 UI（首次启动时通过 `/setup` 创建）
- **API key** 用于外部客户端（`Authorization: Bearer llm-xxxxxxxxxx`），在 UI 的 **API Keys** 下管理

**"Require authentication for all endpoints"** 开关（默认开启）控制模型服务 endpoint 是否要求提供 bearer token：

| 开关状态 | 模型 endpoint（`/api/chat`、`/v1/chat/completions`、每实例端口） | 管理类 endpoint（`/api/instances` 等） |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **开启**（默认） | 必须提供 Bearer token | 必须提供 Bearer token 或 session |
| **关闭** | 公开访问 | 必须提供 Bearer token 或 session |

所有三个端口暴露面（5000、42069、8000-8020）均通过同一个认证 hook。

## GPU 统计信息

每张 GPU 的显存（VRAM）和利用率，原生查询 —— 无需运行中的 llama-server。

| 厂商 | 方法 | 前置要求 |
|---|---|---|
| NVIDIA | `nvidia-ml-py`（直接调用 NVML） | 在 `docker-compose.yml` 中取消 `deploy.resources.reservations` 的注释，以获得 NVIDIA toolkit `utility` 权能 |
| AMD | `/sys/class/drm` sysfs | 挂载 `/sys/class/drm:ro` 卷（默认已包含） |
| Intel Arc | `/sys/class/drm` sysfs | 与 AMD 相同的挂载配置 |

若无原生访问权限，llamaMan 会回退到在运行中的容器内执行 `nvidia-smi` / `rocm-smi`。统计信息始终反映宿主机 GPU 的完整状态。

<a id="request-recording--stats"></a>
## 请求记录与统计

位于 **Settings → App Settings → Request recording**：**Off**（默认关闭）/ **Per request** / **Per conversation**（按 system prompt + 第一条用户消息的内容哈希对对话轮次进行分组）。每条记录都会捕获请求体 + 信封字段（模型、endpoint、状态、耗时、token 计数）以及**精确的单轮指标**（在生成时间窗口内测量的吞吐量；TTFT）。记录存储在 `RECORDINGS_DIR`（JSON）或 `request_log` 表（MariaDB）中。**Retention (days)** 每小时执行修剪清理（`0` = 永久保留）。

每个实例卡片都提供一个 **Stats** 按钮（请求数、平均/峰值吞吐量、平均 TTFT/延迟、token 总数、活跃时间跨度），由请求日志汇总生成 —— 因此在实例停止后数据依然保留，而在关闭记录功能时显示为空状态。顶部的 **Logging** 链接可打开全页面仪表盘（概览卡片、近期对话、单对话下钻详情），支持 24h / 7d / 30d / 全部 时间窗口选择器，并在对话列表上提供按模型和按日期的筛选。

<a id="storage-backends"></a>
## 存储后端

### JSON（默认）

零配置。文件位于 `DATA_DIR`（`/data`）：`state.json`、`presets.json`、`users.json`、`settings.json`、`api_keys.json`、`request_log/`。实例和下载日志输出到 `LOGS_DIR`（`/tmp/llama-logs`）。

### MariaDB / MySQL

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
```

```yaml
environment:
  - DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

数据表在首次连接时自动创建。请求日志写入 `request_log` 表而不是 JSON 的 `request_log/` 目录（使用 MariaDB 时 `RECORDINGS_DIR` 无效）。

按节点划分的模型元数据（源仓库 + 内容哈希）保存在节点作用域的 `model_files` 表中，以 `(node_id, model_path)` 为键。在升级时，每个节点会从传统的共享设置 blob 中复制出自己的条目（仅保留其自身磁盘上存在的路径），并将该 blob 保留作为读取回退和回滚路径。数据库迁移按节点进行版本控制。

> **InnoDB 说明：** 在 utf8mb4 下，`(node_id, model_path)` 大小为 `(64 + 700) × 4 = 3056` 字节，刚好低于 InnoDB 的 3072 字节索引限制，因此建表时使用 `ROW_FORMAT=DYNAMIC`。

MariaDB 同样是启用[集群](CLUSTERING.md)和[知识库](KNOWLEDGE-BASE.md)的前提条件。

<a id="surviving-a-database-outage-local-mirror"></a>
## 数据库故障应对 (本地镜像)

*可选功能，默认关闭。*

配置了 `DATABASE_URL` 后，数据库处于**每个**请求的关键路径上（认证检查每次调用都会读取设置并验证 API key）。如果数据库不可访问，节点将停止服务，且故障期间若容器重启，将完全无法启动。

开启 **Settings → App settings → "Keep a local mirror of the database"**（或设置 `LLAMAMAN_DB_MIRROR=1`），节点就会在 `DATA_DIR/db_mirror/` 中维护一份写透（write-through）本地副本。

| 离线期间仍可工作 | 离线期间受阻 |
|---|---|
| 所有推理（两个端口） | 创建第一个用户（`/setup`）- 全新安装本来就没有镜像可回退 |
| 启动 / 停止 / 休眠 / 唤醒模型 | |
| 下载、模型哈希计算与更新检查 | |
| 创建与编辑**预设**（包括针对刚下载的模型） | |
| 创建与撤销 **API key** | |
| 修改下载和应用**设置** | |
| 添加与移除 **Hugging Face token** | |

离线期间所做的变更会进入一个仅追加的日志，并在重新连接后重放（每 10 秒探测一次）。重放时记录的是*编辑操作*而非最终的行数据，因此另一节点在故障期间所做的修改得以保留（不同的预设字段、不同的设置键、不同的 HF token 均完好保留）。

**要求与注意事项：**

- **`DATA_DIR` 必须是持久化卷**（在示例 compose 文件中已是如此）。
- **每个节点需要独立的 `DATA_DIR`。** 镜像盖有所属节点的 node id 印章；两个节点共享一个目录会直接关闭镜像功能，以避免损坏任一视图。
- **机密数据会落到本地磁盘。** 设置项被原样复制；**Hugging Face token 以明文形式镜像。** 密码与 API key 本就只以哈希形式存储。如果你选择 MariaDB 的部分原因是为了让机密数据不落在单台主机上，那么这一权衡正是该功能需要手动开启（opt-in）的原因。
- **故障期间跨节点负载均衡停止。** 对等节点的存活性数据保存在数据库中；处于降级状态的节点只能感知自己。
- **API key 的变更在重新连接前仅在本地生效。** 离线创建的密钥仅在本节点有效；离线**撤销**的密钥在重新连接前在其他节点依然有效。请据此合理规划密钥撤销。
- 离线期间**请求日志会被丢弃而不是缓冲**（记录中包含完整请求体；在长时间故障中持续缓存可能会填满磁盘卷）。
- **在故障中途关闭镜像功能将被推迟**至重新连接后执行。
- 模式迁移绝不会在降级状态下运行；它们会在重新连接后、执行任何日志重放之前对照真实数据库执行。

<a id="environment-variables"></a>
## 环境变量

### 核心变量

| 变量名 | 默认值 | 描述 |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(必填)* | **必填。** 唯一且稳定的标识符 —— 存储状态的分区键以及集群身份标识。可为任意字符串；选定后请保持不变（后续更改会导致已存储状态变成孤儿数据）。 |
| `MODELS_DIR` | `/models` | 扫描模型文件的目录（容器内路径） |
| `DATA_DIR` | `/data` | 持久化配置/状态的目录（JSON 文件） |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | 存放按对话划分的请求日志记录的目录。仅限 JSON 后端；当配置了 `DATABASE_URL` 时被忽略 |
| `LOGS_DIR` | `/tmp/llama-logs` | 存放实例和下载日志的目录（容器内路径） |
| `HOST_MODELS_DIR` | *(与 `MODELS_DIR` 相同)* | 模型卷在**宿主机端**的绝对路径。必须与 `-v /host/path:/models` 的左侧匹配 —— 在生成同级容器时传递给 Docker daemon |
| `HOST_LOGS_DIR` | *(与 `LOGS_DIR` 相同)* | 日志卷在宿主机端的绝对路径。要求与 `HOST_MODELS_DIR` 相同 |
| `ARCHIVE_DIR` | *(未设置)* | 归档卷挂载的容器路径（例如 `/archive`）。设置此项可启用[模型归档](MODELS.md#model-archive)；未设置则保持隐藏 |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | 公共 llama-server/proxy 端口池 |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | 启用 proxy 模式时使用的内部端口池 |
| `LLAMAMAN_PROXY_PORT` | `42069` | Ollama 兼容 proxy 的端口 |
| `LLAMAMAN_MAX_MODELS` | `0` | 通过 proxy 运行的最大并发 chat 模型数（LRU 逐出；0 = 不限制）。规则参见：[模型逐出](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | 由 proxy 管理的实例的空闲超时分钟数（0 = 禁用） |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | 整个进程允许的最大并发 PDF 光栅化数量。在高 DPI 下，每次光栅化都会启动 poppler 并占用数百 MB 临时 RAM |
| `SECRET_KEY` | *(自动)* | Flask session 密钥。未设置时自动从 machine-id 派生。多副本部署时请显式设置 |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Session cookie 名称。使用命名空间以确保 llamaman 能与同一主机上的其他 Flask 应用共存 —— cookie 的作用域取决于 host+path，而非端口 |
| `DATABASE_URL` | *(未设置)* | MariaDB/MySQL 连接字符串。未设置 = 使用 JSON 文件 |
| `LLAMAMAN_DB_MIRROR` | *(未设置)* | 强制开启（`1`）/ 关闭（`0`）本地 DB 镜像，覆盖按节点的设置。仅在设置了 `DATABASE_URL` 时有意义 |
| `HEALTH_CHECK_TIMEOUT` | `3` | 实例健康检查的超时秒数 |
| `MODEL_LOAD_TIMEOUT` | `300` | 等待模型变为健康状态的秒数。对于超大模型可调大此值 |
| `REQUEST_TIMEOUT` | `300` | 发往 llama-server 上游请求、跨节点转发以及 gate 获取等待的**读取**超时秒数。**不**控制对等节点连接时间（独立的 5 秒限制） |

### Docker / GPU

| 变量名 | 默认值 | 描述 |
|---|---|---|
| `LLAMA_IMAGE` | *(自动)* | 适用于所有生成容器的 llama.cpp Docker 镜像。根据检测到的 GPU 厂商自动选择；设置此项可锁定版本/后端 |
| `LLAMA_NETWORK` | `llamaman-net` | 用于 llamaman + 所有 llama-server 容器的 Docker 网络。缺失时自动创建 |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | 生成的 llama-server 容器的名称前缀 |
| `LLAMAMAN_IN_DOCKER` | *(自动)* | llamaman 本身是否运行在容器中。根据标记文件 + cgroups 自动检测。可显式设置 `true`/`false` 覆盖 |
| `LLAMA_HOST_ADDR` | `localhost` | 裸机运行时用于访问生成容器映射端口的主机地址 |
| `GPU_TYPE` | *(自动)* | 覆盖 GPU 厂商检测：`cuda`、`rocm`、`intel` 或 `vulkan`（`vulkan` 仅限手动开启；会挂载 `/dev/dri` + 宿主机 render/video GID 并选用 `server-vulkan` 镜像） |
| `LLAMA_GPU_DEVICES` | *(全部)* | 所有生成容器可见的逗号分隔 GPU 索引（例如 `0,1,3`）。每实例 **GPU Devices** 设置会覆盖此项。Intel Arc 不支持此项 |

### Strata

可选配置；参见 [Strata](STRATA.md)。

| 变量名 | 默认值 | 描述 |
|---|---|---|
| `STRATA_ENABLED` | `false` | 在此节点上提供 Strata 模型。同时需要 NVIDIA GPU（`GPU_TYPE=cuda` 或已检测到） |
| `STRATA_IMAGE` | `strata:latest` | 启动实例首选的 Strata 镜像。当节点上不存在该镜像时，使用已构建的镜像，再其次使用已拉取的镜像 |
| `STRATA_BUILD_IMAGE` | `strata:latest` | *Pull repo & build latest* 构建的目标标签 |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | 挂载在每个 Strata 容器中 `/data` 下的 Docker 具名卷（模型文件、pack、MTP 层、按模型 setup；每个模型 70-120 GB）。Docker 首次使用时自动创建 |
| `HOST_STRATA_DATA_DIR` | *(未设置)* | 替代具名卷挂载到 `/data` 的宿主机路径。优先级高于 `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | 请求（或重启）等待 Strata 实例就绪的秒数。之所以设置较高，是因为首次启动在打开服务端口前需要下载并准备模型 |
| `STRATA_REPO` | `Niko1221/Strata` | *Pull repo & build latest* 下载的 GitHub 仓库（所有者/名称） |
| `STRATA_REPO_REF` | `main` | 构建 `STRATA_REPO` 的分支、标签或 commit |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | 保存已下载仓库的目录（构建上下文）。其 commit 记录在旁边的 `Strata.source.json` 中 |
| `STRATA_WEB_HOSTS` | *(未设置)* | 用于打开 Strata 实例自身 Web 应用（`http://<host>:<instance port>/`）的额外主机名 / IP（逗号分隔）。在没有 API key 的情况下，Strata 会拒绝来自其他名称页面的 chat 请求；`CLUSTER_ADVERTISE_URL` 的主机始终被允许。对后续启动的实例生效 |

<a id="clustering"></a>
### 集群

可选配置。Core 下的 `LLAMAMAN_NODE_NAME` 兼作集群身份标识，且所有安装方式均需要设置。

| 变量名 | 默认值 | 描述 |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | 设为真值以将此节点加入集群。需要配置 `CLUSTER_SECRET`；若密钥为空则发出警告并忽略 |
| `CLUSTER_SECRET` | *(未设置)* | 用于所有节点间 HTTP 通信的共享 bearer token（`X-Cluster-Secret`）。在每个节点上必须相同。请在受信任网络或 TLS 保护下使用较长的随机值 |
| `CLUSTER_ADVERTISE_URL` | *(未设置)* | 对等节点访问**本节点** UI/API 的地址 —— 其他主机可路由访问的主机名/IP（不能是 `localhost`）。缺少此项时节点为只读状态 |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | 节点显示为离线前距离上次心跳的秒数。若节点在负载下或时钟偏差时发生抖动，请调大此值 |
| `LLAMAMAN_PERF_LOG` | *(关闭)* | 诊断日志。设为真值可针对疑似较慢的路径记录每阶段耗时（`perf <name> <ms>`）：远程实例停止、集群 proxy 转发、可达性探测、心跳快照、认证检查、`save_state`、容器统计。默认关闭；关闭时开销可忽略不计 |
