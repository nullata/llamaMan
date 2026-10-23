# 知识库与 MCP

将文档按**主题（topic）**分组存储，llamaman 会对文档进行分块（chunk，约 1200 字符，按段落对齐，200 字符重叠），计算 embedding，并将向量存储在你的 MariaDB 数据库中（`VECTOR` 列 + 余弦 ANN 索引）。任何 MCP 客户端（Claude Desktop、Claude Code、智能体等）均可通过语义进行搜索。所有内容都在你已有的硬件上运行：embedding 来自你启动并启用了 **Embedding model** 的 llama.cpp 实例，向量保存在数据库中，且 llamaman 自身提供 MCP endpoint 服务。

**系统要求：** 需要运行在 **11.8+** 版本上的 MariaDB 后端（`DATABASE_URL`，参见 [Storage Backends](CONFIGURATION.md#storage-backends)），该版本开始支持 `VECTOR`。在 JSON 后端或旧版 MariaDB 上，MCP 标签页会说明知识库不受支持的原因并拒绝启用；其他任何内容不受影响。无需数据迁移：`kb_*` 数据表会在首次使用时自动创建。

**设置步骤：**

1. 启动一个启用了 **Embedding model** 的实例。任何 embedding GGUF 模型均可（`bge-m3`、`nomic-embed-text` 等）。
2. 前往 **Settings → MCP** → 打开 **Knowledge base**，并在 **Embedding model** 下选择模型。下拉菜单会列出整个集群中所有启用了 embedding 的模型。
3. 使用 MCP `kb_ingest_document` 工具（需开启 **Allow ingest over MCP**）或通过已登录会话调用 `POST /api/kb/documents` 添加文档。该标签页支持列出、筛选和删除主题，并提供测试搜索功能，但暂无添加表单。
4. 打开 **Expose over MCP**。该标签页会显示 endpoint URL（`http://<your-host>:42069/mcp/knowledge`），点击 **Copy config.json** 可复制立即可用的客户端配置：

```json
{
  "mcpServers": {
    "llamaman-kb": {
      "type": "http",
      "url": "http://my-nas:42069/mcp/knowledge",
      "headers": { "Authorization": "Bearer <your-api-key>" }
    }
  }
}
```

仅在需要密钥时才需要 `headers` 代码块（详见下文访问模式）。

**工具。** `kb_list_topics`、`kb_list_documents`、`kb_search`（按匹配度优先返回段落，分数 0–1）和 `kb_get_document` 始终可用。`kb_ingest_document`（若主题不存在则自动创建；每次调用上限 200 KB；重新提交相同内容为无操作）仅在开启 **Allow ingest over MCP** 时出现。`kb_delete_document` / `kb_delete_topic` 仅在开启 **Allow delete over MCP** 时出现。删除主题会连同其所有文档一并删除，且删除操作无法撤销。

**访问模式**（**MCP access**）：

- **Global**（默认）：所有客户端均在**共享池**上操作，可以看到它，并且在允许导入/删除时对其进行修改。新主题会进入共享池。认证遵循 **Require Authentication**（API Keys 标签页）：开启表示必须提供有效的 API key，关闭表示 endpoint 完全公开。
- **Per API key**：即使 Require Authentication 处于关闭状态，每个请求也必须提供有效的 API key。由某个密钥创建的主题对该密钥是**私有**的，两个密钥各自可以拥有同名主题。每个密钥对共享池也拥有**只读**权限：可以搜索和读取共享池，但不能向其中导入或从中删除。其他密钥的主题和文档会返回“未找到”。

切换模式绝不会移动或删除数据，私有主题在两种模式下都保持私有。从 Global 切换到 Per API key 会将已存在的所有内容归入共享池。从 Per API key 切换到 Global 会对所有客户端隐藏私有主题，直到切换回来为止。MCP 标签页始终显示所有内容，并将每个主题标记为 `shared` 或归属密钥的名称。`POST /api/kb/topics` 还可以为指定密钥创建私有主题（`owner_key_id`），这在关闭 MCP 导入时非常实用。

**Embedding 模型。** 按模型名称选择，而非按实例选择。每次 embed 调用会按以下优先级运行：
1. 本节点上健康的实例；
2. 本节点上处于休眠或停止状态的实例（会被唤醒，或者若正在启动中则等待其启动完毕）；
3. 同时开启了 **Knowledge base** 与 **Expose over MCP** 并且拥有该模型的对等节点（那里的休眠实例会被唤醒）；
4. 在上次服务该模型的节点上重新启动实例，每 10 分钟最多尝试 3 次。

首次 embed 会记录向量维度。拥有不同维度的模型会被拒绝，直到你点击 **Re-embed everything** 为止 —— 这是一个在后台重新构建所有文档分块的任务（一次仅运行一个任务，受集群范围的 DB 锁保护）。系统只检查维度，因此在两个维度相同的模型之间切换时，请自行重新生成 embedding。

**集群说明。** 主题和文档是共享数据库中的集群全局数据，而非按节点划分的数据。每个节点看到相同的知识库，且任何节点上的 MCP endpoint 都可以提供服务。参见 [Clustering](CLUSTERING.md)。

**禁用时**，`/mcp/knowledge` 在两种访问模式下均返回 404，且任何请求都不会触碰知识库数据表。

> **安全性：** `/api/kb/*` 仅限管理员使用：接受已登录的 UI 会话和集群对等节点，使用 API key 认证的请求将收到 403。但 API key 并不局限于知识库：提供给 MCP 客户端的密钥同样可以调用推理 endpoint 和大多数其他 `/api/*` 路由。建议为每个客户端使用专用密钥，如有疑虑请及时撤销。
