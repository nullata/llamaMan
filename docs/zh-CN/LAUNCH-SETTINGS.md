# 启动设置

启动表单中的每个字段、其映射参数以及行为说明。有关四步“如何启动”的分步指引，请参见 [README](../../README.md#your-first-model)。

## 目录

- [启动表单工具](#启动表单工具) · [启动设置参考](#launch-settings-reference) · [实时预设更新](#实时预设更新) · [并发与排队](#并发与排队) · [图像与 PDF 输入](#image--pdf-input) · [防循环](#anti-loop) · [每实例 Proxy](#每实例-proxy) · [空闲超时](#空闲超时)

下文的行为说明较为精炼；在 UI 中悬停查看字段的提示信息可获得完整说明。除特殊说明外，所有标志均映射至 llama-server —— 精确语义和当前默认值请参见 [llama-server CLI reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

## 启动表单工具

启动表单的操作行右侧包含三个工具按钮：

- **Export preset** - 将当前表单配置下载为 JSON 文件（可在不同节点间迁移；不保存至存储）。
- **Import preset** - 将预设 JSON 加载到表单中。仅填充字段；仍需点击 Save Preset 进行持久化保存。
- **Copy llama.cpp command** - 将当前表单配置等效的 `llama-server ...` 调用命令复制到剪贴板。由服务端通过与真实启动相同的标志生成代码进行渲染，从而确保与标志的演进保持同步。

折叠高级设置时，**Quick Launch** 按钮可直接根据预设启动所选模型。

每个实例在其分配的端口上暴露 OpenAI 兼容的 API。当选中某个 GGUF 模型时，llamaMan 会读取其元数据检测层数，并在 **GPU Layers** 旁显示（例如 `/ 32`）。

<a id="launch-settings-reference"></a>
## 启动设置参考

**模型设置 (Model Settings)**

| 设置项 | 默认值 | 命令行标志 | 说明 |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | 每次请求的 prompt + response token 上限 |
| **Parallel Slots** | `1` | `--parallel` | llama-server 内部的并发解码槽位（KV slots） |
| **Batch Size** | `2048` | `--batch-size` | 单步处理的最多 prompt token 数。留空则省略该标志 |
| **Micro-batch Size** | `512` | `--ubatch-size` | 发送给 GPU 的物理 batch 大小；数值越大 = 预填充越快，占用更多 VRAM。Embedding 和编码器（BERT 风格）决策模型需要将整个输入置于单个 micro-batch 中。留空则省略该标志 |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`。量化可减少 K 缓存显存占用；支持在开启或关闭 Flash Attention 时使用 |
| **V Cache Type** | `f16` | `--cache-type-v` | 可选值相同。任何量化值**均要求 Flash Attention = On** —— 当 FA 离开 On 时，UI 会置灰下拉菜单并自动将 V 缓存切回 `f16` |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` 省略该标志（llama.cpp 自身的默认行为）；`on` 强制开启（若后端不支持则启动失败）；`off` 禁用 |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`。控制如何解析思考标签（路由至 `reasoning_content` 还是保留在正文内）。`auto` 省略该标志 |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`。模型权重加载到内存的方式（替代已弃用的 `--mlock`/`--mmap`/`--direct-io`）。`auto` = 除非不支持否则使用 mmap，与 llama.cpp 的默认行为一致并省略标志 |

**容器与 Proxy 设置 (Container & Proxy Settings)**

| 设置项 | 默认值 | 命令行标志 | 说明 |
|---|---|---|---|
| **Port** | *(auto)* | - | 公开端口；必须落在 `PORT_RANGE_START/END` 范围内 |
| **CPU Threads** | *(auto)* | `--threads` | 同时设置容器的 CPU 配额（`--cpus N`）。留空 = 允许容器和 llama-server 使用所有核心 |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | 预填充/批处理阶段的独立线程数。留空在 llama-server 中回退至 `--threads` |
| **System Memory Limit** | *(none)* | - | Docker RAM 配额上限（例如 `32g`、`8192m`） |
| **Idle Timeout min** | `0` | - | 空闲 N 分钟后自动休眠。`0` = 关闭。支持从预设实时更新（无需重启）。参见[空闲超时](#空闲超时) |
| **Max Concurrent** | `0` | - | 最大在处理推理请求数；超出的请求排队。`0` = 不限流。参见[并发与排队](#并发与排队) |
| **Max Queue Depth** | `200` | - | 返回 HTTP 429 前的最大排队深度上限 |
| **Exclude from Max Models** | off | - | 不计入 `LLAMAMAN_MAX_MODELS` 配额，永不被逐出。参见[模型逐出](MODELS.md#model-eviction)。支持从预设实时更新 |

**GPU 设置 (GPU Settings)**

| 设置项 | 默认值 | 命令行标志 | 说明 |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl`（别名） | `-1` = **auto**（llama.cpp 自动估算 VRAM 并选择；可能将部分层留在 CPU）。`-2` = **all**（强制卸载所有层；装不下则失败）。`0` = 仅 CPU，容器不附加 GPU。任何正整数 N 表示从底部起精确卸载 N 层 |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | 仅对混合专家（MoE）模型有意义。`0` = 不生成任何标志。`-1` = `--cpu-moe`（所有层的路由专家权重常驻 CPU）。正整数 N = `--n-cpu-moe N`（前 N 层的专家置于 CPU）。在大型 MoE 上可将 VRAM 缩减约 4-10 倍，因为专家占了权重的大部分但每个 token 仅激活少数几个。在稠密模型上无效：llama.cpp 的正则（`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`）匹配不到任何内容，覆盖设置会被静默忽略。**不涵盖共享专家**（DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE 中的 `_shexp`）—— 如需覆盖请在 Extra Args 中添加 `-ot` 选项 |
| **GPU Devices** | *(all)* | *(容器级别)* | 逗号分隔的附加宿主机 GPU 索引（例如 `0,1`）。容器内部会从 0 重新编号。Intel Arc 不支持此项 |
| **Split Mode** | `layer` | `--split-mode` | `none`（单 GPU，忽略 Tensor Split）/ `layer`（llama.cpp 默认值；按整层切分）/ `row`（切分张量行；通常仅在具备 NVLink 等高速互联时才有优势） |
| **Tensor Split** | *(auto)* | `--tensor-split` | 容器可见的每个 GPU 逗号分隔的相对权重（例如 `24,16`）。留空会在启动时根据每个 GPU 的总显存自动填充。当 Split Mode = `none` 或仅单张 GPU 可见时被忽略 |

**Extra Args** *(文本输入框)* - 直接透传给 llama-server 的额外标志（例如 `--mlock`，或表单中未展示的 n-gram 投机解码类型）。

**Proxy 侧采样覆盖 (Proxy-Side Sampling Overrides)** *(开关)* - 开启后，proxy 会对每个转发的请求强制应用采样参数：

| 设置项 | 默认值 | 取值范围 |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0（`0` 表示禁用） |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 到 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0（`0` = 禁用） |

**投机解码 (Speculative Decoding)** *(开关)*

| 设置项 | 默认值 | 命令行标志 | 说明 |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`。仅 `draft-mtp` 允许留空草稿模型（回退至主模型内置的 MTP 头） |
| Draft Model | *(none)* | `-md` | 草稿模型 GGUF 路径。除 `draft-mtp` 之外的所有类型必填 |
| Draft N Max | `2` | `--spec-draft-n-max` | 每步最多起草的 token 数量 |
| Draft N Min *(高级)* | *(auto)* | `--spec-draft-n-min` | 留空则省略该标志 |
| Draft P Split *(高级)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`。留空则省略该标志 |
| Draft P Min *(高级)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`。留空则省略该标志。`0` 是有意义的有效值（不同于留空） |

**图像与 PDF 输入 (Image & PDF Input)** *(开关)* - 参见下文[图像与 PDF 输入](#image--pdf-input)。

**防循环 (Anti-Loop)** — 两个独立的子开关（默认均关闭）。参见下文[防循环](#anti-loop)。

## 实时预设更新

保存预设会在可行的情况下就地更新该模型已在运行的实例：

- **实时生效（无需重启）：** `idle_timeout_min`、`max_concurrent`、`max_queue_depth`、`share_queue`、全部 6 个 proxy 采样字段、全部 5 个循环检测字段（下一个请求即可应用）
- **需要重启生效：** 启动时已固化到容器中的所有项（GPU 层数、MoE CPU 层数、ctx 大小、线程数、批处理线程数、缓存类型、flash-attn、reasoning 格式、加载模式、DRY 采样器、投机解码字段、embedding 开关、mmproj/PDF 字段、额外参数、内存限制）

**Proxy 采样注意事项：** 如果实例启动时同时满足 `idle_timeout=0`、`max_concurrent=0` 且 `override_enabled=false`，则不会生成 sidecar proxy。此时实时开启 `override_enabled=true` 仍会对经由应用兼容 endpoint 路由的请求生效，但直接请求公共端口将绕过采样覆盖。在这种情况下需要重启实例以生成 proxy。

## 并发与排队

当 **Max Concurrent** > 0 时，llamaMan 会在实例前放置一个 FIFO gate（先进先出门禁）。超出上限的请求最多排队至 **Max Queue Depth**，超过后返回 HTTP 429。Gate 统计信息（活动数 + 排队数）可通过 API 查看。

**Parallel vs Max Concurrent：** `Parallel` 控制 llama-server 内部的 KV 槽位数；`Max Concurrent` 控制 llamaMan 一次转发的请求数。两者结合使用（例如 `Parallel=4`, `Max Concurrent=4`）可以在不溢出槽位的前提下让服务端保持高效运行。

<a id="image--pdf-input"></a>
## 图像与 PDF 输入

开启 **Image & PDF Input** 开关可在主 GGUF 旁加载视觉模型的多模态投影器（`--mmproj`）。**MMPROJ Model** 字段指定投影器 GGUF（典型文件名：`mmproj-<model>-<precision>.gguf`）。

**将 mmproj 卸载至 GPU (Offload mmproj to GPU)**（默认开启）：除非接收到 `--no-mmproj-offload`，否则 llama.cpp 会将投影器保留在 GPU 上，因此*开启*会完全省略该标志（命令行与该选项引入前逐字节一致），而*关闭*会传入 `--no-mmproj-offload` 以便在 CPU 上运行视觉编码器 —— 这在显存紧张且视觉编码器会占用数 GB 的环境中非常实用。

**PDF** 通过在转发给 llama-server（其本身无原生 PDF 支持）**之前**将每页光栅化渲染为 PNG 来实现支持。开启 **Accept PDF uploads**（位于 Image & PDF Input 分组内）可为实例启用此功能；主开关 **Image & PDF Input** 控制整个分组 —— 关闭主开关也会同时清除 PDF 子开关，因为在没有投影器的情况下，光栅化页面没有视觉模型可以接收。支持以下两种格式：

- **OpenAI**：带有 `data:application/pdf;base64,...` URL 的 `image_url`，或更新的 `file` 块
- **Ollama**：混合在 `images[]` 中的 base64 PDF 字节 —— llamaMan 会嗅探 `%PDF-` 标头并展开 PDF；普通图像保持原有处理路径

**文本层快捷路径 (Text-layer shortcut)**（默认关闭）：开启 **Try text layer first** 后，llamaMan 会在光栅化之前先执行 `pypdf.extract_text()`。具有实质性文本（每页 ≥ 约 50 字符）的原生电子版 PDF（Word/LaTeX/浏览器打印）会被直接内联为单个文本块，完全跳过视觉通道。稀疏文本则回退到光栅化。默认关闭是因为光栅化能为视觉模型保留排版布局，且对所有 PDF 保持可预期的统一行为优于静默的快捷路径。

**单次请求上限：** `PDF DPI`（72-600，默认 200）和 `Max PDF Pages`（1-200，默认 20）。超出上限的 PDF 会返回 HTTP 400，而不是被截断。

**光栅化并发：** PDF 渲染非常消耗 CPU 和 RAM，并且在任何每实例 gate 之前于 gunicorn worker 线程中运行。进程级信号量会限制并发光栅化任务数；可通过 `LLAMAMAN_PDF_MAX_CONCURRENT`（默认 4）进行调整。

<a id="anti-loop"></a>
## 防循环

启动表单的 **Anti-Loop** 区域中包含两个独立的控制项，默认均关闭。它们在处理流水线的两个不同阶段针对同一个问题（模型陷入稳定的输出死循环）；配合使用可获得最佳效果。

**DRY 采样器（软性控制，采样阶段）**。映射至 llama-server 的 `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n`。在采样阶段，凡是会延长近期已出现 n-gram 的 token 都会受到概率惩罚，从而在模型输出重复片段之前就引导其避免自我重复。这是*软性*防线 —— 对于散文通常单独使用就已足够。在容器启动时固化；修改预设需要重启实例。Multiplier = 0 完全禁用（llama.cpp 自身的默认行为）。

*DRY 采样器*（启动时固化至容器 —— 需要重启）：

| 设置项 | 默认值 | 命令行标志 | 说明 |
|---|---|---|---|
| Multiplier | `0.8`（UI 显示值；`0` 表示禁用） | `--dry-multiplier` | 惩罚强度；llama.cpp 将 `0` 视作禁用 |
| Base | `1.75` | `--dry-base` | 指数底数 ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | 最多此长度的 n-gram 免除惩罚 |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | 回溯窗口。留空则省略该标志 |

**输出循环检测（硬性控制，事后检测）**。当 DRY 不足以解决问题 —— 模型仍然陷入稳定的吸引子时 —— llamaman 的 proxy 会在流式传输给客户端时监控助手的可见输出文本（正文 + 思考过程）。在每个扫描周期（默认：每 64 个 chunk 内联扫描一次，辅以 10 秒 worker 回退机制），系统会检查缓冲区最后 `min_chunk_chars` 个字符是否存在重复。如果出现次数 ≥ `min_repetitions`，则终止当前轮次：

- **关闭与 llama-server 的上游连接** —— llama-server 的客户端断开即停机制会终止生成，为下一个请求释放模型。
- **向客户端流中注入人工合成的终止符** —— OpenAI 客户端会收到最后一个带有 `finish_reason=stop` 且内容为 `[llamaman: output loop detected, ending turn]` 的增量 delta；Ollama 客户端会在 `done: true` 行收到 `done_reason: "loop_detected"`。

*输出循环检测*（Proxy 侧；从预设实时更新 —— 下一个请求即可应用新阈值）：

| 设置项 | 默认值 | 说明 |
|---|---|---|
| Min chunk chars | `200` | 取值范围 60-4096。较小的值能捕获较短的循环，但可能对合理但重复的内容产生误报 |
| Min repetitions | `3` | 取值范围 2-20 |
| Buffer size (chars) | `8192` | 每个进行中请求的滑动窗口大小。取值范围 512-65536 |
| Scan every N chunks | `64` | 内联扫描周期 |
| Fallback scan (seconds) | `10` | 针对 token 生成过慢而未达到内联阈值的流的后台 worker 扫描间隔 |

**监控范围：**
- **`:42069` 上的兼容路由**（Ollama `/api/chat`, `/api/generate`；OpenAI `/v1/chat/completions`, `/v1/completions`）—— 只要目标实例预设开启了该开关，始终进行监控。
- **每实例端口（`8000-8020`）** —— 仅当该实例已存在 sidecar proxy 时有效（即 Idle Timeout、Max Concurrent 或 Proxy Sampling Overrides 也处于开启状态）。循环检测本身不会派生 sidecar proxy —— 需同时开启其余三项之一才能对直接请求进行监控。

**参数调优：**
- `min_chunk_chars`（60-4096，默认 200）—— 触发检测的最小重复片段周期。较小的值能捕获更短的死循环，但可能对合理但重复的内容产生误报（编号列表、诗歌副歌、Markdown 表格）。60-100 较为激进；200 是适用于文本散文的均衡默认值；400+ 仅对段落级循环触发。
- `min_repetitions`（2-20，默认 3）—— 在终止生成前片段必须精确出现的完整副本次数。2 表示只要出现双重重复即触发（较为激进）；3-5 在响应灵敏度与防误报之间取得了良好平衡。
- `max_buffer_chars`（512-65536，默认 8192）—— 每个活跃请求保留的滑动窗口大小。必须至少为 `min_chunk_chars × min_repetitions`。
- `scan_every_n_tokens`（8-4096，默认 64）—— 内联扫描周期。越小 = 终止越快，消耗更多 CPU。
- `scan_interval_s`（1-600，默认 10）—— 后台 worker 重新扫描未达到内联阈值的缓冲区的频率。

循环检测阈值的修改在*下一个*请求时生效（正在处理中的流保留附加时捕获的阈值）。DRY 的修改需要重启实例，因为标志已固化到容器中。

**检测算法**（v1）：以缓冲区最后的 `min_chunk_chars` 个字符作为目标；统计其精确出现次数；当 ≥ `min_repetitions` 时标记。无法捕获漂移的近似重复（例如 "The answer is: The answer is: The answer is:"，各次重复之间存在标点微小漂移）—— 这些由 DRY 在采样阶段的防线捕获。算法特意保持简明低开销。

**防御性不变量：** 流处理路径中的每个 hook 均包装在记录日志并跳过的 `try/except Exception` 中 —— 检测器自身的 bug 绝不会中断真实用户的流式传输。

## 每实例 Proxy

当为实例启用了 **Idle Timeout**、**Max Concurrent** 或 **Proxy Sampling Overrides** 时，llamaMan 会在 llama-server 容器前插入一个 WSGI proxy：公开端口（例如 8000）即为该 proxy；llama-server 监听独立的内部端口。Proxy 负责模型名称校验、休眠实例按需唤醒、`RequestGate` 并发限流以及采样参数注入。

**模型名称校验：** 在推理 endpoint 上，若请求体包含 `"model"` 字段，会将其与已加载模型的文件名主干进行比较（小写，无扩展名）。**支持前缀匹配**（例如 `"qwen2.5-0.5b-instruct-q2"` 可匹配 `"qwen2.5-0.5b-instruct-q2_k"`）。不匹配时返回 HTTP 404；对于休眠实例不会触发唤醒。无 `"model"` 字段的请求将被无条件转发。

## 空闲超时

在启动表单中设置 **Idle Timeout min**（0 = 禁用）。启用后，若在 N 分钟内没有请求，llama-server 容器将被停止以释放显存；下一个请求会使用相同配置唤醒新容器。客户端看到的是相同的端口，仅伴随一次冷启动延迟。

对于由 proxy 管理（OpenWebUI）的实例，请改用 `LLAMAMAN_IDLE_TIMEOUT`。
