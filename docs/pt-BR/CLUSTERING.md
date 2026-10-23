# Clustering

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

Alguns ajustes têm escopo por node porque são específicos do host: **imagens Docker** rastreadas e os interruptores de desalojo do teto de modelos. Todo o resto é compartilhado por todo o cluster. Os aliases de grupos de fila compartilhada ativos são anunciados como modelos selecionáveis em `/api/tags` e `/v1/models` (deduplicados no cluster), assim um cliente pode enviar o alias e tê-lo roteado ao node com menos carga servindo-o.

**O comprimento de contexto do grupo é de todo o cluster.** O `/api/show` e o `/v1/models` reportam o contexto de um grupo de fila como o **mínimo entre todos os membros ativos**, seja qual for o node que o cliente pergunta. Um grupo com membros em valores de ctx diferentes anuncia o valor menor para que um prompt dimensionado a ele seja seguro para qualquer alvo de dispatch — a folga extra no membro maior simplesmente fica sem uso. Um node sem membro local do grupo ainda responde (`context_length` preenchido do ctx runtime do peer no estado compartilhado de instâncias, ou da linha de preset no DB compartilhado); um grupo sem membros ativos em lugar nenhum ainda retorna 404 / sem entrada. O card da instância na UI do operador acrescenta `ctx <min>, capped by <node/model>` quando o ctx deste membro excede o mínimo anunciado, assim você vê qual peer está puxando o grupo para baixo.

> **Segurança:** o secret do cluster deixa qualquer peer comandar ações neste node. Rode o tráfego node-a-node numa rede confiável ou atrás de TLS.

Variáveis de ambiente relacionadas a cluster: [Clustering variables](CONFIGURATION.md#clustering).
