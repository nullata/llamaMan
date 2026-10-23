# API REST

Todos os endpoints retornam / aceitam JSON. Os endpoints de gestão precisam de um cookie de sessão (login no navegador) ou `Authorization: Bearer <key>`. Os endpoints de serviço de modelos também exigem um bearer quando `require_auth` está ativado (padrão). Detalhes de auth: [Authentication](CONFIGURATION.md#authentication-detail).

## Índice

- [Autenticação e chaves](#auth--keys) · [Instâncias](#instances) · [Downloads, modelos, presets](#downloads-models-presets) · [Configurações, sistema, log de requisições](#settings-system-request-log) · [Compatibilidade com Ollama e OpenAI](#ollama--openai-compat)

<a id="auth--keys"></a>
## Autenticação e chaves

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` / `POST` | `/login` | Página de login / autenticar |
| `GET` / `POST` | `/setup` | Configuração do primeiro uso |
| `GET` | `/logout` | Encerrar sessão |
| `GET` / `POST` | `/api/api-keys` | Listar / criar chaves API |
| `DELETE` | `/api/api-keys/<id>` | Revogar |

<a id="instances"></a>
## Instâncias

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Listar / lançar |
| `GET` / `DELETE` | `/api/instances/<id>` | Obter / parar e remover |
| `POST` | `/api/instances/<id>/restart` | Reiniciar uma instância parada/dormindo |
| `DELETE` | `/api/instances/<id>/remove` | Remover o registro de uma instância parada |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Tail / stream SSE dos logs |
| `GET` | `/api/next-port` | Obter a próxima porta disponível do pool |
| `POST` | `/api/preview-command` | Renderizar o comando `llama-server ...` equivalente para um corpo de lançamento (usado pela ferramenta "Copy llama.cpp command" da UI; não lança nada) |

**Corpo de lançamento** (`POST /api/instances`): os campos do formulário de lançamento, serializados como JSON. Veja a [Referência das configurações de lançamento](LAUNCH-SETTINGS.md#launch-settings-reference) para cada campo e seu padrão; campos com valor enum (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) aceitam apenas os valores listados lá e usam por padrão o padrão da referência. Campos numéricos deixados como `null` ou omitidos usam o padrão da referência; campos desconhecidos são ignorados.

Casos-limite que vale a pena registrar:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp estima a VRAM), `-2` = **all** (força cada camada), `0` = só CPU (nenhum dispositivo GPU anexado ao container), N positivo = exatamente N camadas.
- `n_cpu_moe_layers`: `0` (padrão) omite ambas as flags MoE. `-1` emite `--cpu-moe` (todos os experts roteados na CPU). N positivo emite `--n-cpu-moe N` (experts das primeiras N camadas na CPU). Inerte em modelos densos.
- `threads`: quando definido, aplica `--threads N` ao llama-server **e** define a cota de CPU do container para N núcleos.
- `threads_batch`: null / omitido → a flag não é emitida e o llama-server recorre ao `--threads`.
- `flash_attn`: valores legados `true` / `false` de configs pré-tri-state são convertidos na leitura (`true` → `"on"`, `false` → `"off"`), então nenhuma migração de armazenamento é necessária.
- Valores quantizados de `cache_type_v` exigem `flash_attn: "on"` - `"auto"` não é uma garantia; o llama-server se recusa a iniciar caso contrário.
- `spec_draft_model` é obrigatório quando `spec_enabled: true` e `spec_type` é qualquer coisa diferente de `"draft-mtp"` - uma requisição sem ele é rejeitada com 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / omitido omite a flag. `0` é um valor real distinto de null e é repassado.

<a id="downloads-models-presets"></a>
## Downloads, modelos, presets

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Listar / iniciar |
| `GET` / `DELETE` | `/api/downloads/<id>` | Obter / cancelar |
| `DELETE` | `/api/downloads/<id>/remove` | Remover uma entrada concluída/falha |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Tail / stream SSE |
| `GET` | `/api/models` | Modelos descobertos (inclui `repo_id` quando conhecido; modelos arquivados sinalizados `archived` com `restore_path`) |
| `POST` | `/api/models/delete` | Excluir do disco (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Ler a contagem de camadas dos metadados GGUF |
| `GET` | `/api/disk-space` | Espaço livre/usado no volume de modelos |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Obter / salvar / excluir um preset (`GET /api/presets` lista todos) |
| `GET` | `/api/engines` | Motores de inferência deste node: disponibilidade (com motivo), capacidades, campos de lançamento, catálogo de modelos do Strata |
| `GET` | `/api/archive` | Status do archive (`enabled`, `available`, `reason`, `free_bytes`) e jobs de movimentação |
| `POST` | `/api/archive` | Arquivar um modelo: `{"path": "/models/..."}` (move a pasta inteira). 409 quando em uso |
| `POST` | `/api/archive/restore` | Restaurar um modelo arquivado: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Cancelar uma movimentação na fila/em execução (origem mantida), ou remover uma concluída |
| `POST` | `/api/engines/strata/download` | Pré-baixar os shards de um modelo Strata (`{"model": "strata/qwen-IQ2_XS"}`, `hf_token_id` / `speed_limit_mbps` opcionais) |

Corpo de download: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. `filename` em branco baixa o repositório inteiro.

Instâncias e presets aceitam um `engine` opcional (`llamacpp` padrão, ou `strata`; inferido de um caminho de modelo `/strata/...`) mais, para Strata, `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) e `strata_low_ram` (`auto`/`on`/`off`). Os caminhos de modelos Strata são virtuais: `/strata/<family>-<SIZE>`, p. ex. `PUT /api/presets/strata/qwen-IQ2_XS`. Veja [Strata](STRATA.md).

<a id="settings-system-request-log"></a>
## Configurações, sistema, log de requisições

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Obter / salvar configurações globais |
| `GET` | `/api/system-info` | Uso de CPU, contagem de núcleos, uso de RAM |
| `GET` | `/api/gpu-info` | VRAM e utilização por GPU (nativo; recorre a exec no container) |
| `GET` | `/health` | `{"status": "ok"}` - sempre aberto, sem auth necessária |
| `GET` | `/api/request-log/conversations` | Conversas recentes com metadados agregados (`limit` na query, padrão 100, máx. 500) |
| `GET` | `/api/request-log/conversations/<id>` | Todos os turnos registrados de uma conversa, mais antigos primeiro |
| `GET` | `/api/request-log/stats` | Métricas agregadas (tokens, média/pico tokens/s, TTFT, latência, erros, contagens transmitidas). `inst_id` + `window_hours` opcionais |

<a id="ollama--openai-compat"></a>
## Compatibilidade com Ollama e OpenAI

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate auto-iniciam).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat auto-inicia).
Modelos de decisão: `/v1/systemone` (auto-inicia) - veja [Modelos de decisão](MODELS.md#decision-models).
Base de conhecimento: `/api/kb/*` (só admin) e o endpoint MCP `/mcp/knowledge` - veja [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Os endpoints node-a-node de cluster usam `X-Cluster-Secret` - veja [Clustering](CLUSTERING.md).
