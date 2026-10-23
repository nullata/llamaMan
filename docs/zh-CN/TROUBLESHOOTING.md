# 故障排除

完整的“故障现象 → 解决方法”速查表。[README](../../README.md#troubleshooting) 中保留了最常见的 9 个首次运行问题；其余所有内容均在此处。

| 故障现象 | 解决方法 |
|---|---|
| 实例卡在 **starting** 状态 | 点击 Logs 按钮查看日志。常见原因：OOM（内存不足）、模型路径拼写错误、GGUF 文件损坏、镜像尚未拉取 |
| _"Docker image not found"_ | 拉取对应的镜像：`docker pull ghcr.io/ggml-org/llama.cpp:server-cuda`（或 `server-rocm` / `server-sycl` / `server`） |
| 启动时出现 _"Docker API error"_ | 确保已挂载 `/var/run/docker.sock`（`docker-compose.yml` 中的默认配置） |
| 无 GPU / CUDA 错误 | 确保已安装 NVIDIA Container Toolkit，并且在宿主机上运行 `docker run --gpus all` 正常 |
| 无 GPU / ROCm 错误 | 确保 `/dev/kfd` 和 `/dev/dri` 存在，且你的用户属于 `video`/`render` 用户组 |
| 无 GPU / Intel Arc 错误 | 确保 `/dev/dri` 可访问，且你的用户属于 `video`/`render` 用户组 |
| Vulkan 镜像报错 _"Unable to find group render"_ | 设置 `GPU_TYPE=vulkan`，以便 llamaMan 按数字形式附加宿主机的 render/video GID（对于 /etc/group 中未提供 `render` 条目的镜像，基于名称的 group_add 会失败） |
| GPU 状态统计不可用 | NVIDIA：取消 `deploy.resources.reservations` 块的注释。AMD/Intel：确保已挂载 `/sys/class/drm:ro` |
| 检测到错误的 GPU 厂商 | 设置 `GPU_TYPE=cuda`/`rocm`/`intel` 进行强制覆盖 |
| 裸机运行时实例卡在 **starting** 状态 | 容器本身健康，但 llamaman 无法连接到该容器。如果自动检测对你的运行环境有误，请显式设置 `LLAMAMAN_IN_DOCKER=false`/`true` |
| Strata：_"Docker image 'strata:latest' not found"_ | 在 Settings → Docker Images 中拉取或构建 Strata 镜像（或在克隆的 Strata 仓库中执行 `docker build -t strata .`）。参见 [Strata](STRATA.md) |
| 未列出 Strata 模型 | 设置 `STRATA_ENABLED=true`；节点必须为 NVIDIA（若检测失败请设置 `GPU_TYPE=cuda`）。`GET /api/engines` 会显示原因 |
| Strata 实例卡在 **starting** 状态 | 首次启动时需要数分钟属于正常现象（下载 + 打包）。卡片会显示阶段；日志会显示 Strata 的安装设置过程。“setup failed”表示其安装设置中断 —— 查看日志中的 `[X]` 行 |
| Stats 弹窗为空 | 启用 **Settings → App Settings → Request recording**（参见 [Request Recording & Stats](CONFIGURATION.md#request-recording--stats)） |
| 在没有 GPU 直通的主机上启动因 GPU/CDI 错误而失败 | 若没有连接 GPU 设备且仅使用 CPU，请将 **GPU Layers** 设置为 `0`，或修复 GPU 运行时配置 |
| 端口冲突 | 表单会自动建议未使用的端口；如有需要可手动调整 |
| 模型未在 OpenWebUI 中显示 | 确保 `OLLAMA_BASE_URL=http://llamaman:42069`。检查 `/api/tags` 是否能返回模型 |
| OpenWebUI 收到 401 | `require_auth` 处于开启状态（默认）。创建一个 API key 并在 OpenWebUI 中设置 `OPENAI_API_KEYS` |
| 模型满载时 OpenAI 客户端收到 503 | 已达到 `LLAMAMAN_MAX_MODELS` 上限，且 OpenAI API 默认不执行逐出。提高上限或启用逐出开关：[Model Eviction](MODELS.md#model-eviction) |
| 停止后容器未被清理 | llamaMan 在停止时会删除容器。若崩溃后残留孤儿容器：执行 `docker ps --filter name=llamaman-`，或重启 llamaMan（启动时会自动执行孤儿容器认领机制） |
| 客户端（Hermes / OpenWebUI 等）报告模型训练的 context 窗口而非预设上限 | 升级到 1.1.2+。`/api/ps` 现已包含设置为实例启动时运行时 ctx 的 `context_length`，并且 `/api/show` 的 `model_info["<arch>.context_length"]` 会被实际有效上限覆盖 |
| 客户端查询队列分组名称时获取到自身默认 ctx（例如 128k）而非实际部署的 ctx | `/api/show` 和 `/v1/models` 现在可以在任何集群节点上对队列分组进行响应 —— 即使本地没有该成员 —— 使用来自共享状态的对等节点运行时 ctx。多成员分组报告所有活跃成员中的最小值，确保宣告的值对任何调度目标都是安全的。参见 [Clustering](CLUSTERING.md) |
