# Clustering

> **Observação:** esta é uma tradução e pode estar desatualizada em relação à [versão em inglês](../en/CLUSTERING.md), que é a referência.

*Opcional, desligado por padrão - instalações de node único não são afetadas de forma alguma.*

Vários deployments do llamaMan agem como **um único cluster lógico**: um dashboard único agregando as GPUs, instâncias e downloads de cada node, com lançamentos/pulls/downloads entre nodes e balanceamento de carga de fila compartilhada multi-node. Os nodes se descobrem através do backend de armazenamento compartilhado - sem troca de chaves aos pares.

**Requisitos:**

- **Backend de armazenamento compartilhado** - cada node aponta para o **mesmo** `DATABASE_URL` (MariaDB/MySQL, veja [Backends de armazenamento](CONFIGURATION.md#storage-backends)). JSON é por host e não pode ser compartilhado.
- **`LLAMAMAN_NODE_NAME` único por node** - a identidade de cluster de cada node (e obrigatório para toda instalação de qualquer forma).
- **O mesmo `CLUSTER_SECRET` em cada node** - o token bearer (enviado como `X-Cluster-Secret`) para todo HTTP node-a-node.
- **`CLUSTER_ADVERTISE_URL` por node** para *ações* entre nodes - um nome de host/IP roteável dos **outros** hosts (não `localhost`), p. ex. `http://srv1:5000`. Um node sem um aparece no dashboard mas é de somente visualização.

Defina em **cada** node (só `LLAMAMAN_NODE_NAME` e `CLUSTER_ADVERTISE_URL` diferem):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Cada node manda heartbeat a cada ~5s; um node silencioso além de `CLUSTER_NODE_ONLINE_WINDOW_S` (padrão 45s) aparece offline. Inspecione e gerencie em **Settings → Cluster**.

Alguns ajustes têm escopo por node porque são específicos do host: **imagens Docker** rastreadas, os interruptores de desalojo do teto de modelos e os [ajustes de acordar sob carga](#waking-under-load). Todo o resto é compartilhado por todo o cluster. Os aliases de grupos de fila compartilhada ativos são anunciados como modelos selecionáveis em `/api/tags` e `/v1/models` (deduplicados no cluster), assim um cliente pode enviar o alias e tê-lo roteado ao node com menos carga servindo-o.

**O comprimento de contexto do grupo é de todo o cluster.** O `/api/show` e o `/v1/models` reportam o contexto de um grupo de fila como o **mínimo entre todos os membros ativos em nodes que podem receber tráfego**, seja qual for o node que o cliente pergunta. Membros em nodes offline ou somente visualização não contam. Um grupo com membros em valores de ctx diferentes anuncia o valor menor para que um prompt dimensionado a ele seja seguro para qualquer alvo de dispatch — a folga extra no membro maior simplesmente fica sem uso. Um node sem membro local do grupo ainda responde (`context_length` preenchido do ctx runtime do peer no estado compartilhado de instâncias, ou da linha de preset no DB compartilhado); um grupo sem membros ativos em lugar nenhum ainda retorna 404 / sem entrada. O card da instância na UI do operador acrescenta `ctx <min>, capped by <node/model>` quando o ctx deste membro excede o mínimo anunciado, assim você vê qual peer está puxando o grupo para baixo.

> **Segurança:** o secret do cluster deixa qualquer peer comandar ações neste node. Rode o tráfego node-a-node numa rede confiável ou atrás de TLS.

Variáveis de ambiente relacionadas a cluster: [Clustering variables](CONFIGURATION.md#clustering).

<a id="queue-groups"></a>
## Grupos de fila

Um grupo de fila reúne instâncias sob um **Queue group name** para que um cliente possa enviar esse nome a qualquer node. A tabela mostra o que acontece com cada membro do ponto de vista de outro node:

| Estado do membro | Listado em `/api/tags` e `/v1/models` | Recebe tráfego roteado |
|---|---|---|
| Healthy | Sim | Primeira escolha |
| Iniciando (carregando) | Sim | Depois dos membros healthy |
| Dormindo | Sim | Por último, e só depois de todo membro acordado, incluindo fallbacks. A requisição o acorda. |
| Parando ou parado | Não | Nunca |
| Em um node sem `CLUSTER_ADVERTISE_URL` (somente visualização) | Não | Nunca |
| Em um node offline (sem heartbeat por 45s) | Não | Nunca |

**`CLUSTER_ADVERTISE_URL` é necessário para receber trabalho roteado.** Sem ele, um node ainda pode enviar requisições aos seus peers, mas nenhum consegue alcançá-lo.

**Dormir e parar significam coisas diferentes aqui.** Um tempo ocioso coloca uma instância para dormir, e um membro dormindo acorda quando o tráfego precisa dele. Um membro parado continua parado: uma parada pelo operador, um desalojo e uma falha terminam todos assim, e nenhuma requisição de outro node o reinicia. Perguntar ao node que o mantém retorna um `503`, a menos que o nome também seja o próprio arquivo desse membro, o que ainda o acorda.

<a id="waking-under-load"></a>
### Acordar sob carga

Membros acordados enfileiram requisições em vez de recusá-las, então só o tráfego alcançaria um membro dormindo depois que uma fila inteira transbordasse. Por isso o llamaMan acorda um antes. Quando todos os membros healthy do grupo estão cheios, há requisições na fila e nada está carregando, por 3 consultas seguidas (cerca de 15 segundos), um membro dormindo acorda. Em todo o cluster, só o node de menor nome que mantém um membro dormindo faz isso, e no máximo um membro por grupo por consulta. Um primário acorda antes de um fallback.

Só conta um membro dormindo que acrescentaria capacidade: ele precisa de **Max Concurrent** acima de `0` e não pode compartilhar a fila com um membro acordado no mesmo node.

O despertar é ignorado, com uma linha de log, se os pesos do modelo não couberem na VRAM livre do node. A verificação soma a memória livre de todas as GPUs e conta apenas os pesos, então um modelo aprovado ainda pode falhar ao carregar. Se a VRAM não puder ser lida, ou o modelo não tiver um arquivo para medir (os modelos virtuais do Strata), o despertar acontece mesmo assim.

Os dois ajustes são por node, via `POST /api/settings`:

| Ajuste | Padrão | Significado |
|---|---|---|
| `eager_wake_enabled` | `true` | Se este node acorda seus membros dormindo sob carga |
| `eager_wake_polls` | `3` | Consultas saturadas consecutivas (5 segundos cada) antes de acordar |

<a id="when-a-group-cant-be-served"></a>
### Quando um grupo não pode ser atendido

Se outros nodes hospedam o grupo, mas nenhum consegue aceitar uma requisição, o cliente recebe um erro nomeando cada node e o motivo, não "model not found":

| Status | Código (rotas OpenAI) | Significado |
|---|---|---|
| `429` | `group_full` | A fila de pelo menos um peer está cheia. Tentar de novo pode funcionar. |
| `503` | `group_unavailable` | Os peers estão offline, somente visualização, inalcançáveis ou não prontos. |

O node dono retorna estes códigos `503` para seus próprios membros, e um node de entrada tenta o próximo membro ao receber um deles:

| Código | Significado |
|---|---|
| `model_stopping` | O membro está sendo encerrado |
| `group_stopped` | Todos os membros neste node estão parados |
| `wake_failed` | O despertar falhou |
| `wake_timeout` | O membro não ficou healthy a tempo |

Um nome de grupo não lança um arquivo local cujo nome apenas o contém. Um arquivo com exatamente esse nome ainda é executado.

<a id="troubleshooting-a-group-that-isnt-visible"></a>
### Solucionar um grupo que não aparece

- **O backend JSON precisa de um único `DATA_DIR` compartilhado.** Os nodes se encontram pelo registro, então diretórios separados significam que cada node não vê nenhum peer. Use MariaDB, a menos que o diretório seja realmente compartilhado.
- **Um banco de dados degradado esconde peers de propósito.** Com o espelho do DB atendendo as leituras, um node vê apenas a si mesmo e atende localmente.
- **Uma sondagem de carga que falha não esconde um grupo.** Um peer que não responde continua roteável pelo seu snapshot (de até 45 segundos atrás), mas fica atrás de todo peer que respondeu. O log registra quando as sondagens a um peer começam a falhar e quando se recuperam.
