<a id="knowledge-base--mcp"></a>
# Base de conhecimento e MCP

Armazene documentos agrupados por **topic**, e o llamaman os divide em pedaços (~1200 caracteres, alinhados aos parágrafos, 200 de sobreposição), gera seus embeddings e armazena os vetores no seu banco de dados MariaDB (colunas `VECTOR` + um índice ANN de cosseno). Qualquer cliente MCP (Claude Desktop, Claude Code, agentes) pode então pesquisá-los por significado. Tudo roda no hardware que você já roda: os embeddings vêm de uma instância llama.cpp lançada com **Embedding model** ativado, os vetores vivem no seu banco de dados, e o llamaman mesmo serve o endpoint MCP.

**Requisitos:** o backend MariaDB (`DATABASE_URL`, veja [Backends de armazenamento](CONFIGURATION.md#storage-backends)) no servidor **11.8+**, que é onde o `VECTOR` chega. No backend JSON ou num MariaDB mais antigo, a aba MCP diz por que a base de conhecimento não é suportada e a ativação é recusada; nada mais muda. Não há migração: as tabelas `kb_*` são criadas no primeiro uso.

**Configuração:**

1. Lance uma instância com **Embedding model** ativado. Qualquer GGUF de embedding funciona (`bge-m3`, `nomic-embed-text`, …).
2. **Settings → MCP** → ative **Knowledge base** e escolha o modelo em **Embedding model**. O menu suspenso lista todos os modelos com embedding ativado pelo cluster inteiro.
3. Adicione documentos com a ferramenta MCP `kb_ingest_document` (ative **Allow ingest over MCP**) ou `POST /api/kb/documents` de uma sessão logada. A aba lista, filtra e exclui topics e tem uma busca de teste, mas ainda não tem formulário de adição.
4. Ative **Expose over MCP**. A aba mostra a URL do endpoint (`http://<your-host>:42069/mcp/knowledge`), e **Copy config.json** copia uma configuração de cliente pronta para colar:

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

O bloco `headers` só é necessário quando uma chave é exigida (veja os modos de acesso abaixo).

**Ferramentas.** `kb_list_topics`, `kb_list_documents`, `kb_search` (passagens melhores primeiro, score 0–1) e `kb_get_document` estão sempre disponíveis. `kb_ingest_document` (cria o topic se não existir; 200 KB por chamada; reenviar conteúdo idêntico é um no-op) só aparece com **Allow ingest over MCP** ativado. `kb_delete_document` / `kb_delete_topic` só aparecem com **Allow delete over MCP** ativado. Excluir um topic exclui seus documentos, e exclusões não podem ser desfeitas.

**Modos de acesso** (**MCP access**):

- **Global** (padrão): todo cliente trabalha no **pool compartilhado**, vendo-o e, se ingest/delete são permitidos, editando-o. Novos topics vão para o pool compartilhado. A autenticação segue **Require Authentication** (aba API Keys): ligado significa que uma chave API válida é exigida, desligado significa que o endpoint está aberto.
- **Per API key**: toda requisição precisa de uma chave API válida, mesmo com Require Authentication desligado. Os topics que uma chave cria são **privados** para aquela chave, e duas chaves podem cada ter um topic com o mesmo nome. Toda chave também vê o pool compartilhado **somente leitura**: pode pesquisá-lo e lê-lo, mas não ingerir nele nem excluir dele. Os topics e documentos de outras chaves voltam como "not found".

Trocar de modo nunca move nem exclui dados, e topics privados permanecem privados nos dois modos. Ir de Global → Per API key transforma tudo o que existe no pool compartilhado. Ir de Per API key → Global esconde os topics privados de todo cliente até você voltar. A aba MCP sempre mostra tudo, rotulando cada topic `shared` ou com o nome da chave dona. O `POST /api/kb/topics` também pode criar um topic privado para uma chave dada (`owner_key_id`), o que é útil quando o ingest MCP está desligado.

**Modelo de embedding.** A seleção é por nome de modelo, não por instância. Cada chamada de embed roda na primeira de:
1. uma instância healthy neste node;
2. uma dormindo ou parada neste node, que é acordada (ou uma que já está iniciando, pela qual a chamada espera);
3. um node peer que tem **Knowledge base** e **Expose over MCP** ligados e que tem o modelo (uma dormindo lá é acordada);
4. um relançamento no node que o serviu por último, no máximo 3 tentativas a cada 10 minutos.

O primeiro embed registra a dimensão do vetor. Um modelo com dimensão diferente é recusado até você apertar **Re-embed everything**, um job em segundo plano que reconstrói os pedaços de cada documento (um job por vez, guardado por um lock de DB em todo o cluster). Só a dimensão é verificada, então refaça você mesmo o re-embed quando trocar entre dois modelos com a mesma dimensão.

**Notas de cluster.** Topics e documentos são dados de todo o cluster no banco compartilhado, não por node. Cada node vê a mesma base de conhecimento, e o endpoint MCP de qualquer node a serve. Veja [Clustering](CLUSTERING.md).

**Quando desativado**, `/mcp/knowledge` retorna 404 nos dois modos de acesso, e nenhuma requisição toca as tabelas da base de conhecimento.

> **Segurança:** `/api/kb/*` é só para admin: sessões de UI logadas e peers do cluster são aceitos, requisições autenticadas com uma chave API recebem 403. As chaves API, porém, não são limitadas à base de conhecimento: uma chave que você dá a um cliente MCP também pode chamar os endpoints de inferência e a maioria das outras rotas `/api/*`. Use uma chave dedicada por cliente e revogue-a em caso de dúvida.
