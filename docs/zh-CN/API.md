# REST API

所有 endpoint 均接收 / 返回 JSON。管理类 endpoint 需要 session cookie（浏览器登录）或 `Authorization: Bearer <key>`。当开启 `require_auth` 时（默认开启），模型服务类 endpoint 同样需要 bearer token。认证详情参见：[认证](CONFIGURATION.md#authentication-detail)。

## 目录

- [认证与密钥](#认证与密钥) · [实例](#实例) · [下载、模型、预设](#下载模型预设) · [设置、系统、请求日志](#设置系统请求日志) · [Ollama 与 OpenAI 兼容](#ollama-与-openai-兼容)

## 认证与密钥

| Method | Endpoint | 描述 |
|---|---|---|
| `GET` / `POST` | `/login` | 登录页面 / 认证 |
| `GET` / `POST` | `/setup` | 首次运行初始化设置 |
| `GET` | `/logout` | 结束会话 |
| `GET` / `POST` | `/api/api-keys` | 列出 / 创建 API key |
| `DELETE` | `/api/api-keys/<id>` | 撤销 API key |

## 实例

| Method | Endpoint | 描述 |
|---|---|---|
| `GET` / `POST` | `/api/instances` | 列出 / 启动实例 |
| `GET` / `DELETE` | `/api/instances/<id>` | 获取 / 停止并删除实例 |
| `POST` | `/api/instances/<id>/restart` | 重启已停止/休眠的实例 |
| `DELETE` | `/api/instances/<id>/remove` | 删除已停止实例的记录 |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | 查看尾部日志 / SSE 流式日志 |
| `GET` | `/api/next-port` | 从端口池中获取下一个可用端口 |
| `POST` | `/api/preview-command` | 将启动请求体渲染为等效的 `llama-server ...` 命令（UI 中“Copy llama.cpp command”工具所用；不启动任何实例） |

**启动请求体（Launch body）** (`POST /api/instances`)：启动表单中的各项字段，序列化为 JSON。有关每个字段及其默认值，请参见[启动设置参考](LAUNCH-SETTINGS.md#launch-settings-reference)；枚举值字段（`flash_attn`、`reasoning_format`、`split_mode`、`cache_type_k/v`、`spec_type`）仅接受参考文档中列出的值，并默认为参考文档中的默认值。数值字段若为 `null` 或省略则使用参考默认值；未知字段会被忽略。

值得注意的边界行为：

- `n_gpu_layers`：`-1` = **auto**（llama.cpp 自动估算 VRAM），`-2` = **all**（强制卸载所有层），`0` = 仅 CPU（容器不附加 GPU），正整数 N = 恰好卸载 N 层。
- `n_cpu_moe_layers`：`0`（默认）省略两个 MoE 标志。`-1` 生成 `--cpu-moe`（所有路由的专家置于 CPU）。正整数 N 生成 `--n-cpu-moe N`（前 N 层的专家置于 CPU）。在稠密（dense）模型上无效。
- `threads`：设置时，向 llama-server 应用 `--threads N` **并**将容器 CPU 配额限制为 N 核。
- `threads_batch`：null / 省略 → 不生成该标志，llama-server 回退至 `--threads`。
- `flash_attn`：从早期非三态配置遗留的 `true` / `false` 值在读取时会被折叠（`true` → `"on"`，`false` → `"off"`），因此无需存储迁移。
- `cache_type_v` 量化值需要 `flash_attn: "on"` —— `"auto"` 不保证可用；否则 llama-server 会拒绝启动。
- 当 `spec_enabled: true` 且 `spec_type` 为除 `"draft-mtp"` 之外的任何值时，必须提供 `spec_draft_model` —— 缺少该参数的请求会被拒绝并返回 400。
- `spec_draft_n_min` / `_p_split` / `_p_min`：null / 省略时会省略该标志。`0` 是有意义的有效值（不同于 null），会被正常透传。

## 下载、模型、预设

| Method | Endpoint | 描述 |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | 列出 / 开始下载 |
| `GET` / `DELETE` | `/api/downloads/<id>` | 获取 / 取消下载 |
| `DELETE` | `/api/downloads/<id>/remove` | 移除已完成/失败的下载记录 |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | 查看尾部日志 / SSE 流 |
| `GET` | `/api/models` | 已发现的模型（已知时包含 `repo_id`；已归档模型标记为 `archived` 并带有 `restore_path`） |
| `POST` | `/api/models/delete` | 从磁盘删除（`{"path": "/models/..."}`） |
| `GET` | `/api/model-layers?path=<path>` | 从 GGUF 元数据中读取层数 |
| `GET` | `/api/disk-space` | 模型卷的可用/已用空间 |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | 获取 / 保存 / 删除预设（`GET /api/presets` 列出所有预设） |
| `GET` | `/api/engines` | 当前节点的推理引擎：可用性（附原因）、功能特性、启动字段、Strata 的模型目录 |
| `GET` | `/api/archive` | 归档状态（`enabled`、`available`、`reason`、`free_bytes`）及迁移任务 |
| `POST` | `/api/archive` | 归档模型：`{"path": "/models/..."}`（移动其整个文件夹）。使用中时返回 409 |
| `POST` | `/api/archive/restore` | 恢复已归档模型：`{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | 取消排队中/运行中的迁移（保留源文件），或移除已完成的任务 |
| `POST` | `/api/engines/strata/download` | 预下载 Strata 模型的分片（`{"model": "strata/qwen-IQ2_XS"}`，可选 `hf_token_id` / `speed_limit_mbps`） |

下载请求体：`{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`。留空 `filename` 会拉取整个仓库。

实例和预设支持可选的 `engine`（默认 `llamacpp`，或 `strata`；可从 `/strata/...` 模型路径推断），对于 Strata 还支持 `strata_vision`（`no`/`yes`/`cpu`）、`strata_kv`（`""`/`int8`/`q4_0`/`k8v4`）和 `strata_low_ram`（`auto`/`on`/`off`）。Strata 模型路径为虚拟路径：`/strata/<family>-<SIZE>`，例如 `PUT /api/presets/strata/qwen-IQ2_XS`。参见 [Strata](STRATA.md)。

## 设置、系统、请求日志

| Method | Endpoint | 描述 |
|---|---|---|
| `GET` / `POST` | `/api/settings` | 获取 / 保存全局设置 |
| `GET` | `/api/system-info` | CPU 使用率、核心数、RAM 使用量 |
| `GET` | `/api/gpu-info` | 每张 GPU 的 VRAM 和利用率（原生读取；回退至容器内 exec 读取） |
| `GET` | `/health` | `{"status": "ok"}` - 始终公开，无需认证 |
| `GET` | `/api/request-log/conversations` | 包含汇总元数据的近期对话（`limit` 查询参数，默认 100，最大 500） |
| `GET` | `/api/request-log/conversations/<id>` | 单个对话的所有记录轮次，按时间从旧到新排列 |
| `GET` | `/api/request-log/stats` | 聚合指标（token 数、平均/峰值 token/s、TTFT、延迟、错误、流式请求数）。可选 `inst_id` + `window_hours` |

## Ollama 与 OpenAI 兼容

Ollama：`/api/tags`、`/api/version`、`/api/show`、`/api/ps`、`/api/chat`、`/api/generate`（chat/generate 会自动按需启动）。
OpenAI：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`（chat 会自动按需启动）。
决策模型：`/v1/systemone`（自动启动）- 参见[决策模型](MODELS.md#decision-models)。
知识库：`/api/kb/*`（仅限管理员）及 MCP endpoint `/mcp/knowledge` - 参见[知识库与 MCP](KNOWLEDGE-BASE.md)。集群节点间 endpoint 使用 `X-Cluster-Secret` - 参见[集群](CLUSTERING.md)。
