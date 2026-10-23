# Modelos, downloads e desalojo

Tudo sobre a biblioteca de modelos: adicionar modelos, mantê-los atuais, como o proxy os escolhe e desaloja, e os ajustes de manutenção.

## Índice

- [Modelos](#models) · [Modelos de decisão](#decision-models) · [Desalojo de modelos](#model-eviction) · [Configurações de download](#download-settings) · [Gerenciamento de imagens Docker](#docker-image-management) · [Arquivamento de modelos](#model-archive) · [Backup e restauração de modelos](#model-backup-and-restore) · [Configurações de limpeza](#cleanup-settings)

<a id="models"></a>
## Modelos

Coloque modelos sob o volume `/models` como arquivos `.gguf`, diretórios de repositório HuggingFace (contendo `config.json`), ou use o botão **Download** na UI.

<a id="display-name"></a>
### Nome de exibição

Os modelos são listados pelo radical do nome de arquivo GGUF por padrão. Cada modelo pode ganhar um **Display Name** amigável (no formulário de lançamento). Quando definido, é o id que os clientes de API veem em `/api/tags` e `/v1/models`, então o OpenWebUI mostra `Qwen 2.5 14B` em vez de `Qwen2.5-14B-Instruct-Q4_K_M`. Precisa ser único e não colidir com o nome de arquivo de outro modelo nem com um nome de grupo de fila do cluster.

<a id="model-updates"></a>
### Atualizações de modelos

O botão **Check for updates** (mostrado para modelos com repositório de origem conhecido) compara o arquivo local contra o hash de conteúdo publicado do repositório:

- **Up to date** - o hash registrado casa
- **Update available** - re-baixa para uma pasta de staging e troca atomicamente só quando o download conclui (um pull falho deixa o modelo atual intacto). Recusa com 409 enquanto o modelo está carregado.
- **Verify hash** - hasheia o arquivo local (job em segundo plano com progresso) quando nenhum hash foi registrado ainda

Uma varredura em segundo plano opt-in (**Settings → Download Settings**, veja [Configurações de download](#download-settings)) mantém essas respostas prontas com antecedência.

<a id="decision-models"></a>
## Modelos de decisão

Modelos de decisão respondem perguntas tipadas sobre uma entrada com probabilidades em vez de gerar texto: a API System One introduzida pelo Jev da TypeSafe, servida pelo llama.cpp (imagens construídas depois de 2026-10-05) em `POST /v1/systemone`. GGUFs abertos: Julia-1, Laya, Kev-4B, lev, Clef-Flash, Clef, OpenJev (OpenJev e Clef também aceitam imagens, com seu `--mmproj`). O llama-server os reconhece por `<arch>.decision.type` no GGUF; nenhuma flag é necessária.

- **Detecção.** Escolher um no formulário de lançamento diz isso sob Model Path. Sem um preset o formulário é preenchido para ele: **Exclude from Max Models** ligado, **Web UI** desligado (a página de chat dele não consegue conduzir o modelo), **Parallel Slots** `8` (cada pergunta de uma requisição pega uma vaga, e elas compartilham a entrada), **Embedding Model** desligado (o llama.cpp liga o modo embedding sozinho onde o modelo precisa; forçá-lo não é necessário). Um modelo codificador (estilo BERT) também tem Context, Batch e Micro-batch Size definidos ao comprimento de contexto dele: a entrada inteira precisa caber num micro-batch. O **Clef** (e o Clef-Flash) faz todas as perguntas num único prompt e as lê de uma única passada, então ele pega **Parallel Slots** `1` e Batch = Micro-batch = Context (16384, seu padrão treinado, quando Context está em branco): a requisição inteira (entrada, cada pergunta e opção) precisa caber num micro-batch, ou o llama-server responde "increase the physical batch size".
- **Roteamento.** `POST :42069/v1/systemone` escolhe o modelo por `"model"` como o `/v1/chat/completions` (auto-início, regras de desalojo, encaminhamento de cluster, fila). A requisição é encaminhada sem mudanças (sem sampling overrides). Na porta própria da instância ela passa pela fila do proxy por instância quando esse proxy roda.
- **Os endpoints de chat os recusam** (HTTP 422) - eles não geram texto.
- **Request log:** as respostas, e `input_tokens` como os tokens do prompt (`output_tokens` é sempre 0).
- **Erros:** uma pergunta cujas opções não cabem num batch é rejeitada pelo llama-server ("increase the physical batch size") - aumente Batch Size / Micro-batch Size.

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
## Desalojo de modelos

`LLAMAMAN_MAX_MODELS` limita os modelos de **chat** concorrentes via o proxy. Quando o teto está cheio e um novo modelo de chat é requisitado, o modelo de chat LRU é desalojado:

| Lançador | Desaloja | Não pode desalojar |
|---|---|---|
| **Admin UI** | Gerenciados por Ollama primeiro (LRU), depois lançados por admin | - |
| **Ollama API** | Gerenciados por Ollama (LRU) | Lançados por admin *(por padrão)* |
| **OpenAI API** | Nada *(por padrão; retorna 503 quando cheio)* | Tudo *(por padrão)* |

Três interruptores sob **Settings → App Settings** relaxam esses padrões:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - desalojo LRU silencioso antes do lançamento (gerenciados por Ollama primeiro). Desligado por padrão → a UI pergunta antes.
- **Allow Ollama API to evict admin-launched models** - fallback quando nenhum modelo gerenciado por Ollama está disponível. Desligado por padrão.
- **Allow OpenAI API to evict admin-launched models** - concede à API OpenAI o mesmo desalojo LRU do Ollama. Desligado por padrão.

Outros detalhes:

- Todas as instâncias de chat rodando contam (tanto admin UI quanto gerenciadas pelo proxy).
- **Instâncias dormindo ainda contam** (a reserva da vaga persiste durante a pausa ociosa) - mas acordar uma instância dormindo para o próprio modelo nunca é bloqueado pelo teto.
- Instâncias com **Exclude from Max Models** ligado (formulário de lançamento) não contam e nunca são desalojadas: modelos de embedding, modelos de decisão, qualquer coisa pequena que deva ficar ativa. Ligar **Embedding Model** liga-o também, e presets salvos antes de o interruptor existir o herdam da sua configuração de Embedding Model.
- `LLAMAMAN_MAX_MODELS=0` desabilita o desalojo inteiramente.

<a id="download-settings"></a>
## Configurações de download

Sob **Settings → Download Settings**:

- **Global speed limit (Mbps)** (0 = ilimitado) - limita todo download neste node e se aplica a downloads rodando dentro de um segundo. Por node: num cluster cada node tem seu próprio limite (um valor salvo antes de ele se tornar por-node se aplica a todo node até aquele node salvar o seu)
- **Auto-retry failed downloads** (desligado por padrão) + **Retry count per failed download** (padrão 3)
- **Check models for updates in the background** (desligado por padrão) - worker opt-in que pergunta a cada repositório de origem se um arquivo foi republicado, mais calcula um checksum para qualquer modelo que ainda não tem um. **Update check interval (hours)** padrão 24. Hasheia no máximo um modelo por passada; nunca roda enquanto um download está em andamento.

<a id="docker-image-management"></a>
## Gerenciamento de imagens Docker

**Settings → Docker Images**: baixe qualquer imagem llama.cpp pelo nome, delete imagens locais antigas (desabilitado para a `LLAMA_IMAGE` ativa, e retorna um erro se o Docker recusar porque um container a está usando), e opcionalmente auto-atualize as imagens llama.cpp do node (`LLAMA_IMAGE` e toda imagem baixada lá) numa agenda. A linha de pull tem uma seleção de motor: com `STRATA_ENABLED`, uma imagem baixada como *Strata* vai para *Manage Strata images* (não há fonte Strata padrão; digite o nome). **Pull repo & build latest** baixa o commit mais recente do repositório Strata (um tarball do GitHub, em `STRATA_SRC_DIR`) e constrói `STRATA_BUILD_IMAGE` com BuildKit através do socket Docker. O auto-update próprio do Strata re-baixa as imagens Strata baixadas e, quando o repositório foi baixado, busca seu commit mais recente e reconstrói quando ele mudou (ou a imagem está ausente); ele nunca baixa o repositório por conta própria. **Delete image & repo** remove a imagem construída (recusado enquanto uma instância rodando a usa) e o repositório baixado, para um node que roda uma imagem baixada. Os lançamentos Strata usam `STRATA_IMAGE` quando ela está no node, senão a imagem construída, senão uma baixada. Veja [Strata](STRATA.md).

<a id="model-archive"></a>
## Arquivamento de modelos

*(opt-in)* Mova modelos de armazenamento rápido para um segundo volume e de volta pela UI, em vez de mexer nos arquivos à mão no host.

**Ative-o** montando o volume de archive no container llamaMan e apontando `ARCHIVE_DIR` para ele:

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

Não definido, o recurso fica escondido. O llamaMan move os arquivos ele mesmo, então nenhum caminho `HOST_` é necessário.

**Use-o.**

- Passe o mouse sobre um modelo na biblioteca e clique no ícone de arquivar.
- Modelos arquivados ficam na lista, esmaecidos, com um badge *archived* e um botão **Restore**. Eles não podem ser selecionados para lançamento e nunca aparecem em `/api/tags` / `/v1/models`.
- As movimentações aparecem no painel **Downloads** com progresso, velocidade e tempo restante, e podem ser canceladas.
- Restore devolve um modelo ao seu caminho original, então presets, favoritos, notas e display names voltam com ele.

**O que se move.** A pasta de nível superior inteira de um modelo sob `/models`: a pasta de um download, com cada shard de um GGUF dividido. Um `.gguf` localizado diretamente em `/models` é movido junto com seus irmãos divididos. A pasta de shards de um modelo Strata (`strata/<tag>`) é movida sozinha.

**Segurança.**

- **Mesmo disco:** a movimentação é um rename instantâneo.
- **Entre discos:** o llamaMan copia para uma pasta oculta `.llamaman-partial-*` no destino, faz fsync, confere o tamanho de cada arquivo, renomeia para o lugar, e só então deleta a origem.
- **Falhas:** um cancelamento, uma checagem falha ou pouco espaço livre deixam a origem intacta. Cópias parciais deixadas por um restart no meio de uma movimentação são removidas na inicialização.
- **Uma movimentação por vez.**
- **Recusado** enquanto uma instância usa o modelo (incluindo uma dormindo, ou como modelo draft / mmproj, ou uma instância Strata usando os shards), enquanto um download escreve nele, ou quando o nome de destino já existe. Lançar ou deletar um modelo que está sendo movido também é recusado.

**Limites.** Os jobs de movimentação vivem em memória: depois de um restart do llamaMan a lista fica vazia, e uma movimentação interrompida precisa ser recomeçada (nada é perdido). No modo de cluster cada node arquiva para o seu próprio volume, e os botões aparecem para o node que você está navegando.

<a id="model-backup-and-restore"></a>
## Backup e restauração de modelos

**Settings → App Settings**:

- **Download Stored Models JSON** - exporta todos os modelos varridos + configurações de preset para um JSON com timestamp
- **Restore from JSON** - por modelo: já presente → preset mesclado (valores existentes preservados); ausente mas com uma fonte HuggingFace → na fila para baixar com o preset pré-preenchido; ausente e sem fonte conhecida → reportado como não restaurável

<a id="cleanup-settings"></a>
## Configurações de limpeza

Sob **Settings → Cleanup Settings** (todas rodam periodicamente em segundo plano; só removem/atualizam registros - nunca deletam arquivos de modelo):

- **Auto-clean completed/failed downloads** (padrão 24h) - downloads ativos nunca tocados
- **Auto-clean stopped instances** (padrão 24h) - instâncias rodando nunca removidas
- **Auto-remove stale instance records** (padrão intervalo de 5 min) - reconfere registros `starting`/`healthy`/`sleeping` contra o Docker; aqueles cujo container sumiu são marcados como parados
