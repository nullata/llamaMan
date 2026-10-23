# Strata 引擎

*(可选功能，仅限 NVIDIA)* [Strata](https://github.com/Niko1221/Strata) 能够在消费级 NVIDIA GPU 配合系统 RAM 上运行 Qwen3.8-Flash-Next 混合专家（MoE）系列（及其 Coder、Swift 1.5 和 Unsloth 变体）。llamaMan 支持启动、监控、proxy 转发、休眠/唤醒以及停止 Strata 实例，与 llama.cpp 实例并存运行。llama.cpp 的运行机制不受任何影响：引擎抽象层（`core/engines/`）保证了 llama.cpp 的启动逻辑逐字节一致（已通过快照测试）。

## 目录

- [快速入门](#快速入门) · [限制与运行机制](#限制与运行机制) · [设计说明与验证](#设计说明与验证)

## 快速入门

**1. 获取镜像**（上游不发布预构建镜像；宿主机需要 NVIDIA 驱动 **580+**，即 CUDA 13，以及 NVIDIA Container Toolkit）。在启用 Strata（步骤 2）后，可以在 **Settings → Docker Images** 中按名称拉取你信任的镜像（拉取行中选择引擎 *Strata*），或者使用 **Manage Strata images → Pull repo & build latest**，该操作会将仓库下载到 `STRATA_SRC_DIR` 中并通过 Docker socket 针对当前节点的 GPU 架构代数（从驱动读取；若读取失败则包含所有受支持的架构）构建 `STRATA_BUILD_IMAGE`，亦可手动构建：

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. 启用功能** 在 llamaMan 节点上设置：`STRATA_ENABLED=true`（参见 [Strata variables](CONFIGURATION.md#strata)）。Settings → Docker Images 随后会显示 *Manage Strata images*：已拉取的 Strata 镜像、已构建的镜像（下载和构建的 commit，以及构建按钮）和它们的自动更新选项。一旦节点上存在其中一个镜像，启动表单就会提供 Strata 选项。

**3. 启动实例。** 模型库按“系列 × 尺寸”增加对应的虚拟条目：`strata/qwen-Q2_0`、`strata/qwen-IQ2_XS`、`strata/qwen-IQ3_XXS`、`strata/qwen-IQ3_S`、`strata/swift-IQ2_XS`、`strata/swift-IQ3_XXS`、`strata/coder-IQ1_M`、`strata/unsloth-UD-Q4_K_XL` *(实验性)*。每个条目都是独立的实例和容器，因此切换模型的逻辑与 llama.cpp 相同：启动或唤醒另一个实例，并遵循常规的[逐出](MODELS.md#model-eviction)规则（一个 Strata 实例就是一个 chat 模型）。这些条目也会出现在 `/api/tags` 和 `/v1/models` 中，请求可以使用其 id（`strata/qwen-IQ2_XS`，不区分大小写，忽略 `:tag`）或 Strata 自身对外服务的名称（`qwen3.8-flash-next-iq2_xs`）。

启动表单会隐藏仅适用于 llama.cpp 的字段（GPU 层数、MoE、线程、并发 slot、KV 类型、flash attention、投机解码、mmproj、DRY、额外参数），并显示 **Image Input**（no / GPU / CPU）、**KV Cache**（Strata 默认 int8、`int8`、`q4_0`、`k8v4`）以及 **Low-RAM Mode**（auto / on / off）。API 会拒绝为 Strata 模型设置仅适用于 llama.cpp 的字段。

**首次启动。** Strata 会下载约 58-111 GB 数据，准备其“pack”并在服务端口打开前拉取其 MTP 层；后续启动则直接进入加载流程（1-3 分钟）。实例卡片会显示从容器日志解析出的状态：*setting up → downloading (with %) → preparing pack → loading*，点击 Logs 按钮可流式查看日志。下载有两种方式：

- **让 Strata 自动下载** 到其 `/data` 卷中（默认）。
- **通过 llamaMan 预下载**（Strata 区域中的按钮，或调用 `POST /api/engines/strata/download`）：在 Downloads 标签页中生成一个普通任务（支持进度显示、暂停/继续、重试、HF token、限速），从 Strata 固定的 Hugging Face revision 将分片下载到 `MODELS_DIR/strata/<tag>/`。下载完成后，这些分片会被 bind-mount 挂载到容器中，Strata 的 setup 会直接使用它们而无需重复下载。在该下载仍在运行期间，启动请求会被拒绝。首次启动时，MTP 层（约 5 GB）和 pack 仍由 Strata 自行准备。

## 限制与运行机制

- **一次处理一个请求。** Max Concurrent 强制为 1；超出请求在 llamaMan 的 gate 中排队（达到 Max Queue Depth 后返回 429）。
- **RAM 需求。** Strata 会向 RAM 加载 32-62 GB 数据（始终设置 `--ulimit memlock=-1`）。配置 **System Memory Limit** 会强制开启 Low-RAM 模式，因为 Strata 的 setup 读取宿主机 RAM，无法感知容器限制；UI 会在内存低于约 64 GB 时发出警告。
- **设置已固化至 Strata setup 中。** Context / 图像输入 / KV / Low-RAM / GPU 绑定均按模型记录在数据卷上。llamaMan 会记住每个模型上次健康运行时使用的设置，仅在设置变更时才传入 `REINSTALL=1`（文件会被复用）。
- **空闲超时正常生效**（休眠会停止容器；下一个请求将其唤醒），但冷启动需要数分钟，唤醒请求需要等待其完成（`STRATA_LOAD_TIMEOUT`）。
- **每个节点每种 Strata 模型仅限一个实例**（同模型的实例共享卷上的 setup 产物）。
- **仅限 NVIDIA。** 在 ROCm / Intel / Vulkan / CPU 节点上以及当 `STRATA_ENABLED` 关闭时，该选项不可用并会显示原因。在集群中，每个节点都会宣告其支持的引擎，因此仅具备能力的节点才会提供 Strata 模型。
- 裸机运行的 llamaMan 访问 Strata 的方式与 llama-server 相同（通过 `LLAMA_HOST_ADDR` 上的映射端口）。

---

# 设计说明与验证

本节说明集成的运作方式、对上游的依赖、已知局限，以及如何在真实的 NVIDIA 宿主机上进行验证。

上游参考指向 Strata commit `99f3dbd` (2026-10-03)。

## 架构集成方式

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- 实例的引擎为 `config["engine"]`。缺少该值表示 llama.cpp，因此所有现有的实例记录、预设和容器标签保持原有含义。llama.cpp 配置绝不会被盖上该键，否则会改变容器的 `llamaman.config` 标签。`tests/fixtures/llamacpp_container_spec.json` 保存了重构前针对 9 种配置 × 5 类 GPU 厂商的 `containers.run` kwargs，`tests/test_engines.py` 会断言它们未发生变化。
- 无需数据存储迁移。预设和实例在两个后端中均为 JSON blob（`presets.json` / `state.json`，以及 MariaDB 的 `data` TEXT 列），因此 `engine` 和 `strata_*` 均为增量键。
- 每个 Strata 模型都是一个虚拟条目 `/strata/<family>-<SIZE>`（id 为 `strata/<family>-<SIZE>`），并且是各自独立的实例和容器。切换模型、逐出、休眠和唤醒的工作逻辑与 GGUF 文件完全一致。
- `_run_container` 获取引擎的 `container_spec()` 并附加引擎所要求的 GPU 设备。Strata 容器配置如下：

  | | |
  |---|---|
  | image | `STRATA_IMAGE`（否则为已构建的镜像，再否则为已拉取的 Strata 镜像） |
  | command | 无（entrypoint 仅读取环境变量） |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`?（挂起看门狗 / 慢速存储宽限期）, `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests`（`gpu_devices` / `LLAMA_GPU_DEVICES`） |
  | mounts | 数据卷 → `/data`；日志；当 llamaMan 持有完整分片时，`MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>`（非 qwen 为 `<family>-<SIZE>`；setup.py 区分大小写的分片文件夹）(rw) |
  | mem_limit | 来自 System Memory Limit（强制 `LOW_RAM=on`） |
  | network / ports / labels | 与 llama-server 相同（`llamaman-net`，`8080 → host port`，`llamaman.*`） |

## 待定问题解答

1. **就绪状态。** 在经过 API key 门禁前，`/health` 返回 `{"status": "ok", ..., "service": "strata"}`（`serve/server.py:2172`）。HTTP 端口仅在模型加载完成后才会打开：`serve()` 运行在 `server.py:3008`，位于引擎启动代码 `:2925` 之后。在 Strata 下载或加载期间，连接会被拒绝，因此 llamaMan 现有的健康检查（`status == "ok"`）无需特殊处理。在 lazy 模式下模型未加载时 `/v1/models` 返回空列表（`server.py:2190`），因此不是良好的就绪信号。当设置了 API key 时，`/v1/status` 需要提供该 key。
2. **设置。** 所有内容均作为环境变量传递并由 `docker-entrypoint.sh` 读取：`HOST`, `PORT`, `API_KEY`, `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION`（`:14-24`）。命令行参数会被忽略，因为 entrypoint 会重新构建 `$@`（`:42`, `:65`）。GPU 绑定使用容器编号：NVIDIA 运行时会对 llamaMan 挂载的宿主机 GPU 进行重新编号，因此宿主机的 `1,3` 变为 `GPUS=0,1`。llamaMan 不设置 `API_KEY`，这与运行 llama-server（无 key）一致；自身的 proxy 和认证机制前置生效。
3. **复用磁盘上的 GGUF。** `setup.py` 具有 `--gguf-dir` 参数（`setup.py:2954`），但 entrypoint 从未透传该参数。不过，Setup 确实接受已经位于 `/data/models/<fam tag><SIZE>/<original name>`（例如 `IQ3_S`, `coder-IQ1_M`；区分大小写）的完整分片：它会对照自身的张量目录进行检查并标记为已完成（`setup.py:3389-3405`, "#173"）。因此，llamaMan 会将本地副本挂载到此处（读写模式，以便写入 `.done` 标记），且仅在这些分片完全下载完成后挂载。分片是从 Strata 固定的 Hugging Face revision 获取的（`setup.py:65-70`，镜像在 `core/engines/strata.py` 中）。MTP 层（约 5 GB）和 pack 仍会在容器内部生成。
4. **内存限制 / memlock。** Setup 从 `/proc/meminfo` 读取 RAM，这是宿主机的总内存，因此受限容器必须要求 `LOW_RAM=on`（Dockerfile 注释在 `:31`，`docs/INSTALL.md:116-118`）。若未设置，setup 会选择内存驻留模式，容器极易因 OOM 被杀。每当设置内存限制时，llamaMan 都会强制开启 `LOW_RAM=on`，且 UI 会在低于 64 GB 时发出警告。`memlock=-1` 至关重要，因为引擎会对其内存池的一部分进行页锁定以用于 GPU DMA，无法锁定的专家将通过较慢的主机内存拷贝传输（`src/prefill/prefill.cpp:110`, `:207-214`）。从源码来看，缺少 memlock 限制更像是变慢而非报错，但**在没有 GPU 的情况下无法对此进行验证**。llamaMan 始终对其进行配置。

同样来自源码：Strata 一次只处理一个请求（`server.py:1323`）。Dockerfile 设置了 `STRATA_EXECV=1`，因此服务端进程为 PID 1，`docker stop` 可以正常发送信号。中途停止下载的容器在下次启动时会从其 `.part` 文件断点续传。

## 设计决策

- **并发强制为 1**，无论在启动时、预设合并时、实时预设编辑还是保存的预设中均是如此。Gate 会强制启用 sidecar proxy，因此排队发生在 llamaMan（`RequestGate`）中，而不是在 Strata 内部。
- **基于指纹的 `REINSTALL`。** Context、vision、KV、Low-RAM 和 GPU 绑定会记录在卷上 Strata 的按模型配置中。`DATA_DIR/strata_setups.json` 记录每个模型上次健康上线所使用的设置，以数据卷和 setup tag 为键，仅在设置不同时才发送 `REINSTALL=1`。未知的指纹（首次通过 llamaMan 启动，或手动搭建的数据卷）同样会执行重新安装。这只需要耗费一次复用文件的 setup 过程。
- **每个节点每种 Strata 模型仅限一个活跃实例。** 同一模型的两个实例会在相同的文件上并发运行 setup。
- **当 llamaMan 仍在下载该模型时拒绝启动**，避免容器在其卷中重复下载一次。
- **预下载的分片跳过更新检查溯源。** 固定的文件绝不能“更新”为仓库的 `main` 分支。
- **虚拟模型仅在能够运行的环境中列出**（`STRATA_ENABLED` 加上 NVIDIA）：在 `/api/models`、`/api/tags`、`/v1/models` 以及名称解析中。更新扫描和集群模型快照仅针对实体文件。对等节点从 `snapshot.system.engines`（嵌套在 `system` 下，因此较旧的对等节点会忽略它）获取引擎可用性信息。
- **加载阶段** 从容器日志的最后 64 KB 中提取：setup 的 `=== Step N ===` 标头、下载的 `\r` 进度行，以及服务端的 "loading the model"。该解析为尽力而为（best-effort），解析不到时优雅退化为无阶段显示。
- **vLLM (#53)** 将作为另一个 `Engine`：拥有独立的镜像、命令、端口（`internal_port`）、健康检查机制和启动字段。届时只需对假定两个引擎均使用 8080 端口的地方（`resolve_llama_endpoint` 调用方）进行参数化改造即可。

## 无 GPU 环境下未验证项

- 真实的 Strata 启动、首次运行下载、pack 构建、加载和对话流程。上述所有内容均对照上游源码以及使用 Mock Docker 的单元测试进行了核对；尚未在实体 GPU 上运行过 Strata 容器。
- 真实首次启动时的精确日志格式。阶段解析器依据的是 `setup.py` 的 `step()` / `download()` 输出以及服务端的启动打印。
- 未配置 `memlock=-1` 时的实际表现，以及在配置了容器限制的情况下 `LOW_RAM=on` 实际占用的 RAM。
- 所用 Docker 版本上的嵌套 bind mount（在名为 `/data` 的具名卷上挂载 `/data/models/<tag>`）。Docker 本身支持嵌套挂载，但此处尚未进行实机验证。

## 已知局限

- 仅限 NVIDIA。Strata 的 AMD (HIP) 路径在上游属于实验性手动构建，暂不提供支持。
- 未暴露 `LAYER_SPLIT`（多 GPU 层分配）。绑定多个 GPU 时使用 Strata 的自动分配。
- 视觉编码器（mmproj）始终由 Strata 自身下载至数据卷中。
- Strata 自身的 Web UI（`/`, `/settings`, `/load`, `/unload`）可以通过实例端口访问，但 llamaMan 不使用也不管理它。其空闲卸载功能未被使用；llamaMan 的休眠机制通过停止容器来替代。
- 修改 setup 中记录的设置会在下次启动时重新触发 Strata 的 setup 流程（文件会被复用）。
- 首次启动耗时可能超过 HTTP 客户端自身的超时限制（Open WebUI 等）。llamaMan 会持续等待最长达 `STRATA_LOAD_TIMEOUT`，但客户端可能提前放弃。建议预先下载，或在对外提供服务前先通过 UI 启动一次。
- 模型目录（系列、尺寸、固定 revisions、文件名）是 Strata `setup.py` 的镜像映射。上游新增系列或尺寸需要对 `core/engines/strata.py` 进行对应修改。

## 手动验证清单（NVIDIA 宿主机）

前置条件：NVIDIA 驱动 580+，可用的 `docker run --gpus all`，建议 64 GB RAM，以及约 150 GB 空闲磁盘空间。

1. **构建：** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`。可选择添加 `--build-arg CUDA_ARCHITECTURES=<your arch>`。或者在完成步骤 2 之后，前往 Settings → Docker Images → *Manage Strata images* → **Pull repo & build latest**。该操作仅针对当前节点的 GPU 架构代数进行构建。
2. **启用：** 设置 `STRATA_ENABLED=true` 并重启 llamaMan。
   - `GET /api/engines` 显示 strata 的 `available: true`，并列出其模型（内置列表中有 9 个模型）。
   - Settings → Docker Images → *Manage Strata images* 显示 `strata:latest` 为 **local**。
3. **门禁校验：** 在非 NVIDIA 节点上（或设置了 `GPU_TYPE=rocm`），不会列出 Strata 模型，针对 `/strata/qwen-IQ2_XS` 调用 `POST /api/instances` 会返回 400 并提示 NVIDIA 相关原因。
4. **从 UI 启动：** 选择 `strata/qwen-IQ2_XS`。
   - 隐藏仅限 llama.cpp 的字段，显示 Strata 区域，且 Max Concurrent 锁定为 1。
   - 设置 32g 内存限制并确认出现低于 64 GB 警告；随后重新清除限制。
   - 启动。
5. **首次运行（容器内下载）：**
   - 实例卡片显示 *setting up → downloading NN% → preparing pack → loading*。
   - Logs 按钮流式输出 Strata 的 setup 内容。
   - 卡片状态变为 **healthy**，`docker inspect` 显示 `Ulimits: memlock -1`、环境变量以及 `/data` 卷。
6. **对话测试：**
   - 通过 Ollama proxy：`curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - 通过 `:42069` 上的 `/v1/chat/completions`，以及直接通过实例端口请求。
   - 尝试使用 `model: "qwen3.8-flash-next-iq2_xs"`。
   - 同时发送两个请求：第二个请求会排队（卡片显示 1/1 active · 1 queued）。
7. **空闲休眠与唤醒：**
   - 将 Idle Timeout 设置为 1 分钟（会显示冷启动警告）。1 分钟后实例状态变为 **sleeping** 且容器被移除。
   - 下一个请求将其唤醒（耗时数分钟）并成功返回响应。
   - 日志中没有 setup 流程（无 `=== Step` 行；`docker inspect` 中无 `REINSTALL`）。
8. **设置变更：**
   - 将 context 修改为 65536 并重启。
   - `docker inspect` 显示 `REINSTALL=1`，日志显示无需下载的 setup 流程。
   - 下次重启时，`REINSTALL` 再次消失。
9. **预下载路径：**
   - 选择另一个模型（例如 `strata/qwen-Q2_0`）并点击 *Pre-download with llamaMan*。
   - Downloads 标签页显示对应固定 revision 的下载任务。
   - 下载期间尝试启动会被拒绝。
   - 下载完成后启动：`docker inspect` 显示 `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`，日志中没有分片下载（仅有 MTP 层和 pack）。
10. **停止 / 移除：** 停止实例。容器被删除，点击 Remove 清除卡片。
11. **在有 Strata 实例运行的情况下重启 llamaMan：**
    - 实例被重新附加（状态恢复），或在状态丢失时被认领：从 `state.json` 中删除它并重启。
    - 卡片显示 Strata 徽章和 `strata/qwen-IQ2_XS`。
    - 请求依然正常工作。
12. **裸机模式：**
    - 使用 `python app.py` 运行 llamaMan（`LLAMAMAN_IN_DOCKER=false`）并重复步骤 4-6。
    - llamaMan 通过 `LLAMA_HOST_ADDR` 上的映射端口访问服务。
13. **集群：**
    - 添加非 NVIDIA 对等节点。仅当选择 NVIDIA 节点作为 Target node 时才会显示 Strata 模型。
    - 从另一个节点的 UI 在 NVIDIA 对等节点上启动实例可正常工作。
14. **llama.cpp 不受影响：** 如往常一样启动 GGUF 模型。
    - 命令行（`docker inspect` Args）与升级前一致。
    - 升级前保存的预设加载和启动均不受影响。
