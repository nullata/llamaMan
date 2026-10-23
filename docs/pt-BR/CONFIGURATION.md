# Configuração e armazenamento

Instalações bare-metal, o modelo de auth em detalhe, estatísticas de GPU, registro de requisições, backends de armazenamento e a referência completa de variáveis de ambiente.

## Índice

- [Execução bare-metal](#running-bare-metal) · [Autenticação em detalhe](#authentication-detail) · [Estatísticas de GPU](#gpu-stats) · [Gravação de requisições e estatísticas](#request-recording--stats) · [Backends de armazenamento](#storage-backends) · [Sobrevivendo a uma queda do banco de dados (espelho local)](#surviving-a-database-outage-local-mirror) · [Variáveis de ambiente](#environment-variables)

<a id="running-bare-metal"></a>
## Execução bare-metal

Útil para dev ou hosts (p. ex. WSL) onde rodar o próprio gerenciador em Docker é complicado. O llama-server ainda roda em containers.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

Um único processo sob o gunicorn serve a UI/API em `:5000` e o proxy em `:42069` (`python app.py` sem gunicorn roda o servidor de dev Flask em `:5000` apenas). Container vs bare-metal é autodetectado; se a detecção alguma vez estiver errada para o seu ambiente, defina `LLAMAMAN_IN_DOCKER=true` ou `false` explicitamente.

<a id="authentication-detail"></a>
## Autenticação em detalhe

Dois tipos de credenciais:

- **Cookies de sessão** para a UI (criados via `/setup` no primeiro lançamento)
- **Chaves API** para clientes externos (`Authorization: Bearer llm-xxxxxxxxxx`), gerenciadas em **API Keys** na UI

O interruptor **"Require authentication for all endpoints"** (ligado por padrão) controla se os endpoints de serviço de modelos exigem um token bearer:

| Interruptor | Endpoints de modelos (`/api/chat`, `/v1/chat/completions`, portas por instância) | Endpoints de gestão (`/api/instances`, etc.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **LIGADO** (padrão) | Token bearer exigido | Token bearer ou sessão |
| **DESLIGADO** | Aberto | Token bearer ou sessão |

As três superfícies de porta (5000, 42069, 8000-8020) passam pelo mesmo hook de auth.

<a id="gpu-stats"></a>
## Estatísticas de GPU

VRAM e utilização por GPU, consultadas nativamente - sem llama-server rodando necessário.

| Fabricante | Método | Requisito |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML direto) | Descomente `deploy.resources.reservations` no `docker-compose.yml` para a capacidade `utility` do toolkit NVIDIA |
| AMD | sysfs `/sys/class/drm` | Montagem de volume `/sys/class/drm:ro` (incluída por padrão) |
| Intel Arc | sysfs `/sys/class/drm` | A mesma montagem do AMD |

Sem acesso nativo, o llamaMan recorre a executar `nvidia-smi` / `rocm-smi` dentro de um container rodando. As estatísticas sempre refletem o estado completo da GPU do host.

<a id="request-recording--stats"></a>
## Gravação de requisições e estatísticas

Em **Settings → App Settings → Request recording**: **Off** (padrão) / **Per request** / **Per conversation** (turnos agrupados por um hash de conteúdo do prompt de sistema + primeira mensagem do usuário). Cada registro captura os corpos + campos de envelope (modelo, endpoint, status, duração, contagens de tokens) mais **métricas precisas por turno** (throughput medido sobre a janela de geração; TTFT). Os registros vivem em `RECORDINGS_DIR` (JSON) ou na tabela `request_log` (MariaDB). **Retention (days)** poda a cada hora (`0` = manter para sempre).

Cada card de instância expõe um botão **Stats** (contagem de requisições, throughput médio/pico, TTFT/latência médios, totais de tokens, intervalo de tempo ativo) agregado do request log - assim ele persiste depois que a instância é parada e mostra um estado vazio quando o registro está desligado. O link **Logging** no cabeçalho abre um dashboard de página inteira (tiles de resumo, conversas recentes, drill-down por conversa) com um seletor de janela de 24h / 7d / 30d / Todo o tempo mais filtros por modelo e por dia sobre a lista de conversas.

<a id="storage-backends"></a>
## Backends de armazenamento

### JSON (padrão)

Zero configuração. Arquivos sob `DATA_DIR` (`/data`): `state.json`, `presets.json`, `users.json`, `settings.json`, `api_keys.json`, `request_log/`. Os logs de instância e download vão para `LOGS_DIR` (`/tmp/llama-logs`).

### MariaDB / MySQL

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
```

```yaml
environment:
  - DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

As tabelas se auto-criam na primeira conexão. Os request logs vão para a tabela `request_log` em vez do diretório JSON `request_log/` (`RECORDINGS_DIR` não tem efeito com MariaDB).

Os metadados de modelos por node (repositório de origem + hash de conteúdo) vivem numa tabela `model_files` com escopo de node, indexada por `(node_id, model_path)`. Na atualização, cada node copia suas próprias entradas do blob de configurações compartilhado legado (só os caminhos que existem no seu próprio disco) e mantém o blob como fallback de leitura e caminho de rollback. As migrações são versionadas por node.

> **Nota InnoDB:** `(node_id, model_path)` é `(64 + 700) × 4 = 3056` bytes sob utf8mb4, pouco abaixo do limite de índice de 3072 bytes do InnoDB, então a tabela é criada com `ROW_FORMAT=DYNAMIC`.

O MariaDB também é o pré-requisito do [Clustering](CLUSTERING.md) e da [Knowledge Base](KNOWLEDGE-BASE.md).

<a id="surviving-a-database-outage-local-mirror"></a>
## Sobrevivendo a uma queda do banco de dados (espelho local)

*Opcional, desligado por padrão.*

Com `DATABASE_URL` definido, o banco de dados está no caminho crítico de **toda** requisição (as checagens de auth leem configurações e verificam chaves API a cada chamada). Se ele ficar inalcançável, o node para de servir, e um restart de container no meio da queda não consegue nem iniciar.

Ative **Settings → App settings → "Keep a local mirror of the database"** (ou defina `LLAMAMAN_DB_MIRROR=1`) e o node mantém uma cópia write-through em `DATA_DIR/db_mirror/`.

| Ainda funciona offline | Bloqueado offline |
|---|---|
| Toda a inferência (ambas as portas) | Criar o primeiro usuário (`/setup`) - uma instalação novinha em folha não tem espelho para o qual recorrer de qualquer forma |
| Lançar / parar / dormir / acordar modelos | |
| Downloads, hashing de modelos e checagens de atualização | |
| Criar e editar **presets** (incluindo de modelos que você acabou de baixar) | |
| Criar e revogar **chaves API** | |
| Mudar **configurações** de download e do app | |
| Adicionar e remover **tokens Hugging Face** | |

Mudanças offline vão para um journal somente de acréscimo e são reaplicadas na reconexão (sonda de 10s). O replay registra a *edição*, não a linha resultante, então uma mudança que outro node fez durante a queda sobrevive (um campo de preset diferente, uma chave de configuração diferente, um token HF diferente ficam todos intactos).

**Requisitos e ressalvas:**

- **`DATA_DIR` precisa ser um volume persistente** (já é no arquivo compose de exemplo).
- **Cada node precisa do seu próprio `DATA_DIR`.** O espelho é carimbado com o id do node dono; dois nodes compartilhando um desligam o espelhamento em vez de corromper qualquer visão.
- **Segredos aterrissam no disco local.** As configurações são copiadas literalmente; **tokens Hugging Face são espelhados em texto plano.** Senhas e chaves API já são armazenadas apenas como hashes. Se você escolheu MariaDB em parte para manter segredos fora dos hosts individuais, este compromisso é o motivo de o recurso ser opt-in.
- **O balanceamento cross-node para durante uma queda.** O estado de atividade (liveness) dos peers fica no banco de dados; um node degradado vê só a si mesmo.
- **Mudanças de chave API são locais até a reconexão.** Uma chave criada offline funciona só neste node; uma chave **revogada** offline continua válida nos outros nodes até a reconexão. Planeje as revogações de acordo.
- **Os request logs são descartados, não enfileirados,** enquanto offline (os registros carregam corpos completos; acumulá-los em disco durante uma queda longa poderia encher o volume).
- **Desligar o espelho no meio de uma queda é adiado** até a reconexão.
- Migrações de schema nunca rodam enquanto degradado; elas rodam contra o banco de dados real na reconexão, antes de qualquer replay do journal.

<a id="environment-variables"></a>
## Variáveis de ambiente

### Core

| Variável | Padrão | Descrição |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(obrigatória)* | **Obrigatória.** Identidade única estável - chave de particionamento do estado armazenado e identidade de cluster. Qualquer string; escolha uma vez, conserve (mudar depois deixa o estado armazenado órfão). |
| `MODELS_DIR` | `/models` | Diretório varrido em busca de arquivos de modelo (caminho de container) |
| `DATA_DIR` | `/data` | Diretório para configuração/estado persistentes (arquivos JSON) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Diretório para registros do request log por conversa. Só backend JSON; ignorado quando `DATABASE_URL` está definido |
| `LOGS_DIR` | `/tmp/llama-logs` | Diretório para logs de instância e download (caminho de container) |
| `HOST_MODELS_DIR` | *(igual a `MODELS_DIR`)* | Caminho absoluto **do lado do host** do volume de modelos. Deve casar com o lado esquerdo de `-v /host/path:/models` - passado ao daemon Docker ao spawnar containers irmãos |
| `HOST_LOGS_DIR` | *(igual a `LOGS_DIR`)* | Caminho absoluto do lado do host do volume de logs. O mesmo requisito do `HOST_MODELS_DIR` |
| `ARCHIVE_DIR` | *(não definido)* | Caminho de container onde um volume de archive é montado (p. ex. `/archive`). Definí-lo ativa o [Arquivamento de modelos](MODELS.md#model-archive); não definido o esconde |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Pool de portas públicas de llama-server/proxy |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Pool de portas internas usado quando o modo proxy está habilitado |
| `LLAMAMAN_PROXY_PORT` | `42069` | Porta para o proxy compatível com Ollama |
| `LLAMAMAN_MAX_MODELS` | `0` | Máximo de modelos de chat concorrentes via o proxy (desalojo LRU; 0 = ilimitado). Regras: [Desalojo de modelos](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Minutos de idle-timeout para instâncias gerenciadas pelo proxy (0 = desabilitado) |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Máximo de rasterizações de PDF concorrentes em todo o processo. Cada rasterização spawna o poppler e consome algumas centenas de MB de RAM transiente em DPI alto |
| `SECRET_KEY` | *(auto)* | Segredo de sessão Flask. Auto-derivado do machine-id se não definido. Defina para deployments multi-replica |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Nome do cookie de sessão. Namespaced para que o llamaman coexista com outras apps Flask no mesmo host - cookies são escopados por host+caminho, não por porta |
| `DATABASE_URL` | *(não definido)* | String de conexão MariaDB/MySQL. Não definido = arquivos JSON |
| `LLAMAMAN_DB_MIRROR` | *(não definido)* | Força o espelho de DB local ligado (`1`) / desligado (`0`), sobrescrevendo o ajuste por node. Só tem sentido com `DATABASE_URL` |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout em segundos para checagens de saúde de instância |
| `MODEL_LOAD_TIMEOUT` | `300` | Segundos de espera para um modelo ficar healthy. Aumente para modelos muito grandes |
| `REQUEST_TIMEOUT` | `300` | Timeout de **leitura** em segundos para requisições upstream ao llama-server, encaminhamento cross-node e esperas de aquisição de gate. **Não** governa o tempo de conexão a peers (limite separado de 5s) |

### Docker / GPU

| Variável | Padrão | Descrição |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | Imagem Docker do llama.cpp para todos os containers spawnados. Auto-selecionada do fabricante GPU detectado; defina para fixar uma versão/backend |
| `LLAMA_NETWORK` | `llamaman-net` | Rede Docker para o llamaman + todos os containers llama-server. Criada se não existir |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Prefixo de nome para containers llama-server spawnados |
| `LLAMAMAN_IN_DOCKER` | *(auto)* | Se o llamaman mesmo roda num container. Autodetectado de arquivos marcador + cgroups. Defina `true`/`false` para sobrescrever |
| `LLAMA_HOST_ADDR` | `localhost` | Endereço de host usado para alcançar as portas publicadas dos containers spawnados ao rodar bare-metal |
| `GPU_TYPE` | *(auto)* | Sobrescreve a detecção do fabricante GPU: `cuda`, `rocm`, `intel`, ou `vulkan` (`vulkan` é só opt-in; escolhe `/dev/dri` + os GIDs render/video do host e a imagem `server-vulkan`) |
| `LLAMA_GPU_DEVICES` | *(todas)* | Índices de GPU separados por vírgula visíveis a todos os containers spawnados (p. ex. `0,1,3`). O **GPU Devices** por instância sobrescreve. Não suportado no Intel Arc |

### Strata

Opcional; veja [Strata](STRATA.md).

| Variável | Padrão | Descrição |
|---|---|---|
| `STRATA_ENABLED` | `false` | Ofereça modelos Strata neste node. Também precisa de uma GPU NVIDIA (`GPU_TYPE=cuda` ou detectada) |
| `STRATA_IMAGE` | `strata:latest` | Imagem Strata preferida para lançamentos. Quando ela não está no node, a imagem construída e depois uma baixada são usadas |
| `STRATA_BUILD_IMAGE` | `strata:latest` | Tag que o *Pull repo & build latest* constrói |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | Volume Docker nomeado montado em `/data` em todo container Strata (arquivos de modelo, packs, camada MTP, setup por modelo; 70-120 GB por modelo). Criado pelo Docker no primeiro uso |
| `HOST_STRATA_DATA_DIR` | *(não definido)* | Caminho de host para montar em `/data` em vez do volume nomeado. Vence sobre `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | Segundos que uma requisição (ou relançamento) espera por uma instância Strata ficar pronta. Alto porque uma primeira inicialização baixa e prepara o modelo antes de a porta abrir |
| `STRATA_REPO` | `Niko1221/Strata` | Repositório GitHub (dono/nome) que o *Pull repo & build latest* baixa |
| `STRATA_REPO_REF` | `main` | Branch, tag ou commit de `STRATA_REPO` a construir |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | Onde o repositório baixado é mantido (o contexto de build). Seu commit é registrado ao lado dele em `Strata.source.json` |
| `STRATA_WEB_HOSTS` | *(não definido)* | Nomes de host / IPs extras (separados por vírgula) pelos quais você abre o app web próprio de uma instância Strata (`http://<host>:<instance port>/`). Sem uma chave API o Strata recusa requisições de chat de páginas em outros nomes; o host de `CLUSTER_ADVERTISE_URL` é sempre permitido. Vale para instâncias iniciadas depois |

### Clustering

Opcional. O `LLAMAMAN_NODE_NAME` (sob Core) também serve como identidade de cluster e é obrigatório para todas as instalações.

| Variável | Padrão | Descrição |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Defina truthy para juntar este node a um cluster. Requer `CLUSTER_SECRET`; ignorado com um aviso se o secret estiver vazio |
| `CLUSTER_SECRET` | *(não definido)* | Token bearer compartilhado (`X-Cluster-Secret`) para todo HTTP node-a-node. Idêntico em cada node. Use um valor aleatório longo numa rede confiável ou atrás de TLS |
| `CLUSTER_ADVERTISE_URL` | *(não definido)* | Como os peers alcançam a UI/API **deste** node - nome de host/IP roteável dos outros hosts (não `localhost`). Sem ele o node é de somente visualização |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Segundos desde o último heartbeat antes de um node aparecer offline. Aumente se nodes flutuam sob carga ou defasagem de relógio |
| `LLAMAMAN_PERF_LOG` | *(off)* | Diagnóstico. Defina truthy para logar tempos por fase (`perf <name> <ms>`) de caminhos suspeitos de lentidão: parada remota de instância, forwards de proxy de cluster, sondas de alcançabilidade, snapshot de heartbeat, checagens de auth, `save_state`, stats de container. Desligado por padrão; sobrecarga negligenciável quando off |
