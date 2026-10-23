# 模型、下载与逐出

关于模型库的一切：添加模型、保持更新、proxy 如何选取与逐出模型，以及日常维护设置。

## 目录

- [模型](#模型) · [决策模型](#decision-models) · [模型逐出](#model-eviction) · [下载设置](#下载设置) · [Docker 镜像管理](#docker-image-management) · [模型归档](#model-archive) · [模型备份与恢复](#模型备份与恢复) · [清理设置](#清理设置)

## 模型

将模型以 `.gguf` 文件或包含 `config.json` 的 HuggingFace 仓库目录形式放置在 `/models` 卷下，或直接使用 UI 中的 **Download** 按钮。

### 显示名称

默认情况下，模型按 GGUF 文件名主干（stem）列出。每个模型都可以设置一个友好的 **Display Name**（在启动表单中）。设置后，它就是 API 客户端在 `/api/tags` 和 `/v1/models` 上看到的 id，因此 OpenWebUI 会显示 `Qwen 2.5 14B` 而不是 `Qwen2.5-14B-Instruct-Q4_K_M`。该名称必须唯一，且不得与另一个模型的文件名或集群队列分组名称冲突。

### 模型更新

**Check for updates** 按钮（针对已知来源仓库的模型显示）会将本地文件与仓库已发布的内容哈希进行比对：

- **Up to date** - 记录的哈希匹配
- **Update available** - 重新拉取到暂存文件夹中，仅在下载完全完成后才原子性替换（拉取失败时保留当前模型完整无损）。当模型处于加载状态时拒绝更新并返回 409。
- **Verify hash** - 当尚未记录哈希时，计算本地文件的哈希值（带进度的后台任务）

可选的后台扫描（**Settings → Download Settings**，参见[下载设置](#下载设置)）可预先准备好这些结果。

<a id="decision-models"></a>
## 决策模型

决策模型针对输入回答类型化问题并输出概率，而不是生成文本：即 TypeSafe 的 Jev 引入的 System One API，由 llama.cpp（2026-10-05 之后构建的镜像）在 `POST /v1/systemone` 上提供服务。开源 GGUF 模型包括：Julia-1、Laya、Kev-4B、lev、Clef-Flash、Clef、OpenJev（OpenJev 和 Clef 还可以通过其 `--mmproj` 接受图像）。llama-server 通过 GGUF 中的 `<arch>.decision.type` 识别它们；无需额外标志。

- **检测。** 在启动表单中选择此类模型时，Model Path 下方会显示对应提示。若没有预设，表单会自动预填：开启 **Exclude from Max Models**，关闭 **Web UI**（其 chat 页面无法驱动此类模型），**Parallel Slots** 设为 `8`（请求中的每个问题占用一个槽位，且它们共享输入），关闭 **Embedding Model**（在模型需要时 llama.cpp 会自动开启 embedding 模式；无需强制开启）。编码器（BERT 风格）模型的 Context、Batch 和 Micro-batch Size 还会被设置为其 context 长度：其整个输入必须能容纳在单个 micro-batch 中。**Clef**（以及 Clef-Flash）在单个 prompt 中提出所有问题并通过单次前向传递读取，因此其 **Parallel Slots** 为 `1`，且 Batch = Micro-batch = Context（当 Context 为空时，使用其训练默认值 16384）：整个请求（输入、每个问题及选项）必须能容纳在单个 micro-batch 中，否则 llama-server 会报错 "increase the physical batch size"。
- **路由。** `POST :42069/v1/systemone` 像 `/v1/chat/completions` 一样通过 `"model"` 选取模型（自动启动、逐出规则、集群转发、队列）。请求被原样转发（无采样覆盖）。在实例自身的端口上，当运行每实例 proxy 时，请求会通过该 proxy 的队列。
- **Chat endpoint 拒绝此类模型**（HTTP 422）- 它们不生成文本。
- **请求日志：** 记录回答内容，并将 `input_tokens` 记录为 prompt token（`output_tokens` 始终为 0）。
- **错误：** 如果某个问题的选项无法容纳在单个 batch 中，llama-server 会拒绝该请求（"increase the physical batch size"）—— 此时请调大 Batch Size / Micro-batch Size。

```bash
curl http://localhost:42069/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "Clef-Flash-Q4_K_M",
  "state": "Customer: I was charged twice for my order last week.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "shipping": null}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["can wait", "today", "right now"]}
  }
}'
```

<a id="model-eviction"></a>
## 模型逐出

`LLAMAMAN_MAX_MODELS` 限制通过 proxy 运行的并发 **chat** 模型数量。当达到上限且请求新的 chat 模型时，将逐出 LRU（最近最少使用）chat 模型：

| 启动来源 | 逐出对象 | 无法逐出 |
|---|---|---|
| **Admin UI** | 优先逐出 Ollama 管理的实例（LRU），随后逐出管理员启动的实例 | - |
| **Ollama API** | Ollama 管理的实例（LRU） | 管理员启动的实例 *(默认情况下)* |
| **OpenAI API** | 不执行逐出 *(默认情况下；满载时返回 503)* | 所有实例 *(默认情况下)* |

在 **Settings → App Settings** 下的三个开关可以放宽这些默认行为：

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - 在启动前静默执行 LRU 逐出（优先逐出 Ollama 管理的实例）。默认关闭 → UI 先弹窗提示。
- **Allow Ollama API to evict admin-launched models** - 当没有 Ollama 管理的模型可用时的回退方案。默认关闭。
- **Allow OpenAI API to evict admin-launched models** - 赋予 OpenAI API 与 Ollama 相同的 LRU 逐出权限。默认关闭。

其他细节：

- 所有运行中的 chat 实例均计入配额（包括管理 UI 和 proxy 管理的实例）。
- **休眠实例仍计入配额**（槽位占用在空闲暂停期间依然保留）—— 但为休眠实例自身的模型唤醒它绝不会受到上限阻止。
- 开启了 **Exclude from Max Models** 的实例（在启动表单中）不计入配额且永不被逐出：embedding 模型、决策模型，以及任何应当保持常驻的小型模型。开启 **Embedding Model** 会自动将其开启，而在该开关引入之前保存的预设会根据其 Embedding Model 设置自动继承。
- `LLAMAMAN_MAX_MODELS=0` 会完全禁用逐出机制。

## 下载设置

位于 **Settings → Download Settings**：

- **Global speed limit (Mbps)**（0 = 不限速）- 限制当前节点上的所有下载，并在 1 秒内对正在运行的下载生效。按节点生效：在集群中每个节点拥有各自的限制（在改为按节点生效之前保存的值会应用于每个节点，直到该节点保存自己的设置为止）
- **Auto-retry failed downloads**（默认关闭）+ **Retry count per failed download**（默认 3）
- **Check models for updates in the background**（默认关闭）- 可选的后台 worker，负责向各个源仓库查询文件是否已被重新发布，并为尚未生成校验和的模型计算校验和。**Update check interval (hours)** 默认 24 小时。每轮最多为单个模型计算哈希；下载进行期间绝不运行。

<a id="docker-image-management"></a>
## Docker 镜像管理

**Settings → Docker Images**：按名称拉取任何 llama.cpp 镜像，删除旧的本地镜像（对当前活动的 `LLAMA_IMAGE` 禁用，如果 Docker 因容器正在使用而拒绝则返回错误），并可按计划自动更新节点的 llama.cpp 镜像（`LLAMA_IMAGE` 以及在该界面拉取的每个镜像）。拉取行包含引擎选择：启用 `STRATA_ENABLED` 后，作为 *Strata* 拉取的镜像会归入 *Manage Strata images*（Strata 没有默认源；需输入名称）。**Pull repo & build latest** 会下载 Strata 仓库的最新 commit（GitHub tarball，下载至 `STRATA_SRC_DIR`），并通过 Docker socket 使用 BuildKit 构建 `STRATA_BUILD_IMAGE`。Strata 自身的自动更新会重新拉取已拉取的 Strata 镜像，且在已下载仓库的情况下获取其最新 commit 并在发生变更（或镜像缺失）时重新构建；它绝不会自动下载仓库。**Delete image & repo** 会删除构建的镜像（当运行中的实例正在使用时会被拒绝）以及已下载的仓库，适用于运行拉取镜像的节点。Strata 启动时若节点上存在 `STRATA_IMAGE` 则优先使用，否则使用构建的镜像，再否则使用拉取的镜像。参见 [Strata](STRATA.md)。

<a id="model-archive"></a>
## 模型归档

*(可选功能)* 通过 UI 将模型从高速存储移至第二存储卷并可随时移回，无需在宿主机上手动移动文件。

**启用方法：** 将归档卷挂载到 llamaMan 容器中并将 `ARCHIVE_DIR` 指向该目录：

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

未设置时该功能保持隐藏。llamaMan 自身负责移动文件，因此无需配置 `HOST_` 路径。

**使用方法。**

- 悬停在模型库中的模型上并点击归档图标。
- 已归档模型保留在列表中，呈变暗显示，带有 *archived* 徽章和 **Restore** 按钮。它们无法被选择启动，且绝不会出现在 `/api/tags` / `/v1/models` 中。
- 移动任务会显示在 **Downloads** 面板中，带有进度、速度和剩余时间，并支持取消。
- 恢复操作会将模型放回其原始路径，因此预设、收藏夹、备注和显示名称都会随之恢复。

**移动内容。** 模型在 `/models` 下的整个顶层文件夹：包含分片 GGUF 每个分片的下载文件夹。直接位于 `/models` 根目录下的 `.gguf` 会与其同属一套的分片兄弟文件一起移动。Strata 模型的分片文件夹（`strata/<tag>`）则单独移动。

**安全性。**

- **同一磁盘：** 移动操作即为瞬间重命名。
- **跨磁盘：** llamaMan 先复制到目标目录下的隐藏 `.llamaman-partial-*` 文件夹中，执行 fsync，检查每个文件的大小，重命名移动到最终位置，随后才删除源文件。
- **故障处理：** 取消操作、检查失败或可用空间不足均会保留源文件完好无损。若移动中途发生重启遗留了部分副本，系统将在启动时将其清理。
- **每次仅执行一个移动任务。**
- **操作拒绝情况：** 当有实例正在使用该模型（包括休眠实例，或作为草稿模型 / mmproj，或正在使用分片的 Strata 实例）、有下载正在写入该模型，或目标名称已存在时，归档操作会被拒绝。启动或删除正在移动中的模型同样会被拒绝。

**限制。** 移动任务保存在内存中：llamaMan 重启后任务列表变为空，中断的移动必须重新发起（不会丢失数据）。在集群模式下，每个节点归档到各自的卷，按钮针对你当前浏览的节点显示。

## 模型备份与恢复

**Settings → App Settings**：

- **Download Stored Models JSON** - 将所有已扫描模型 + 预设配置导出为带时间戳的 JSON 文件
- **Restore from JSON** - 针对每个模型：若已存在 → 合并预设（保留已有值）；若缺失但有 HuggingFace 源 → 进入下载队列并预填预设；若缺失且无已知源 → 报告为无法恢复

## 清理设置

位于 **Settings → Cleanup Settings**（均在后台定期运行；仅删除/更新记录 —— 绝不删除模型文件）：

- **Auto-clean completed/failed downloads**（默认 24h）- 绝不触碰进行中的下载
- **Auto-clean stopped instances**（默认 24h）- 绝不移除运行中的实例
- **Auto-remove stale instance records**（默认间隔 5 min）- 对照 Docker 重新检查 `starting`/`healthy`/`sleeping` 记录；容器已不存在的记录被标记为 stopped
