# Knowledge Base & MCP

Store documents grouped by **topic**, and llamaman chunks them (~1200 chars, paragraph-aligned, 200 overlap), embeds them, and stores the vectors in your MariaDB database (`VECTOR` columns + a cosine ANN index). Any MCP client (Claude Desktop, Claude Code, agents) can then search them by meaning. Everything runs on hardware you already run: embeddings come from a llama.cpp instance you launched with **Embedding model** enabled, vectors live in your database, and llamaman serves the MCP endpoint itself.

**Requirements:** the MariaDB backend (`DATABASE_URL`, see [Storage Backends](CONFIGURATION.md#storage-backends)) on server **11.8+**, which is where `VECTOR` lands. On the JSON backend or an older MariaDB the MCP tab says why the knowledge base is unsupported and enabling is refused; nothing else changes. There is no migration: the `kb_*` tables are created on first use.

**Setup:**

1. Launch an instance with **Embedding model** enabled. Any embedding GGUF works (`bge-m3`, `nomic-embed-text`, …).
2. **Settings → MCP** → turn on **Knowledge base** and pick the model under **Embedding model**. The dropdown lists every embedding-enabled model across the cluster.
3. Add documents with the MCP `kb_ingest_document` tool (turn on **Allow ingest over MCP**) or `POST /api/kb/documents` from a logged-in session. The tab lists, filters and deletes topics and has a test search, but no add form yet.
4. Turn on **Expose over MCP**. The tab shows the endpoint URL (`http://<your-host>:42069/mcp/knowledge`), and **Copy config.json** copies a ready-to-paste client config:

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

The `headers` block is only needed when a key is required (see access modes below).

**Tools.** `kb_list_topics`, `kb_list_documents`, `kb_search` (passages best-first, score 0–1) and `kb_get_document` are always available. `kb_ingest_document` (creates the topic if missing; 200 KB per call; resubmitting identical content is a no-op) appears only with **Allow ingest over MCP** on. `kb_delete_document` / `kb_delete_topic` appear only with **Allow delete over MCP** on. Deleting a topic deletes its documents, and deletes can't be undone.

**Access modes** (**MCP access**):

- **Global** (default): every client works on the **shared pool**, seeing it and, if ingest/delete are allowed, editing it. New topics go into the shared pool. Auth follows **Require Authentication** (API Keys tab): on means a valid API key is required, off means the endpoint is open.
- **Per API key**: every request needs a valid API key, even with Require Authentication off. Topics a key creates are **private** to that key, and two keys can each have a topic with the same name. Every key also sees the shared pool **read-only**: it can search and read it, but not ingest into or delete from it. Other keys' topics and documents come back as "not found".

Switching modes never moves or deletes data, and private topics stay private in both modes. Going Global → Per API key turns everything that exists into the shared pool. Going Per API key → Global hides private topics from every client until you switch back. The MCP tab always shows everything, labeling each topic `shared` or with the owning key's name. `POST /api/kb/topics` can also create a private topic for a given key (`owner_key_id`), which is useful when MCP ingest is off.

**Embedding model.** Selection is by model name, not instance. Each embed call runs on the first of:
1. a healthy instance on this node;
2. a sleeping or stopped one on this node, which gets woken (or one that's already starting, which it waits for);
3. a peer node that has both **Knowledge base** and **Expose over MCP** on and has the model (a sleeping one there gets woken);
4. a relaunch on the node that last served it, at most 3 attempts per 10 minutes.

The first embed records the vector dimension. A model with a different dimension is refused until you press **Re-embed everything**, a background job that rebuilds every document's chunks (one job at a time, guarded by a cluster-wide DB lock). Only the dimension is checked, so re-embed yourself when you switch between two models with the same dimension.

**Cluster notes.** Topics and documents are cluster-wide data in the shared database, not per node. Every node sees the same knowledge base, and the MCP endpoint on any node serves it. See [Clustering](CLUSTERING.md).

**When disabled**, `/mcp/knowledge` returns 404 in both access modes, and no request touches the knowledge base tables.

> **Security:** `/api/kb/*` is admin-only: logged-in UI sessions and cluster peers are accepted, requests authenticated with an API key get 403. API keys are not limited to the knowledge base, though: a key you give an MCP client can also call the inference endpoints and most other `/api/*` routes. Use a dedicated key per client and revoke it if in doubt.
