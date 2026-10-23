# Motor Strata

*(opt-in, só NVIDIA)* O [Strata](https://github.com/Niko1221/Strata) roda a família de mixture-of-experts Qwen3.8-Flash-Next (e suas variantes Coder, Swift 1.5 e Unsloth) em GPUs NVIDIA de consumo mais RAM do sistema. O llamaMan pode lançar, monitorar, fazer proxy de, dormir/acordar e parar instâncias Strata ao lado das de llama.cpp. O comportamento do llama.cpp não muda: a camada de motores (`core/engines/`) mantém o lançamento do llama.cpp byte por byte idêntico (testado por snapshot).

## Índice

- [Primeiros passos](#getting-started) · [Limites e comportamento](#limits-and-behaviour) · [Notas de design e verificação](#design-notes-and-verification)

<a id="getting-started"></a>
## Primeiros passos

**1. Consiga uma imagem** (o upstream não publica nenhuma; o host precisa do driver NVIDIA **580+**, i.e. CUDA 13, e do NVIDIA Container Toolkit). Depois de habilitar o Strata (passo 2), ou baixe uma em que você confie pelo nome em **Settings → Docker Images** (motor *Strata* na linha de pull), ou use **Manage Strata images → Pull repo & build latest**, que baixa o repositório para `STRATA_SRC_DIR` e constrói `STRATA_BUILD_IMAGE` através do socket Docker para as gerações de GPU do node (lidas do driver; todas as suportadas se isso falhar), ou construa-a à mão:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. Habilite-o** no node llamaMan: `STRATA_ENABLED=true` (veja [Strata variables](CONFIGURATION.md#strata)). Settings → Docker Images então mostra *Manage Strata images*: as imagens Strata baixadas, a construída (commits baixado e construído, o botão de build) e o auto-update delas. O formulário de lançamento oferece Strata quando uma delas está no node.

**3. Lance.** A biblioteca de modelos ganha uma entrada virtual por família × tamanho: `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(experimental)*. Cada uma é sua própria instância e container, então trocar de modelos funciona como no llama.cpp: lance ou acorde outra instância, com as regras usuais de [desalojo](MODELS.md#model-eviction) (uma instância Strata é um modelo de chat). As entradas também aparecem em `/api/tags` e `/v1/models`, e as requisições podem usar o id (`strata/qwen-IQ2_XS`, qualquer caixa, `:tag` ignorado) ou o nome servido próprio do Strata (`qwen3.8-flash-next-iq2_xs`).

O formulário de lançamento esconde os campos exclusivos de llama.cpp (GPU layers, MoE, threads, parallel, KV types, flash attention, spec decoding, mmproj, DRY, extra args) e mostra **Image Input** (no / GPU / CPU), **KV Cache** (padrão do Strata int8, `int8`, `q4_0`, `k8v4`) e **Low-RAM Mode** (auto / on / off). A API rejeita campos exclusivos de llama.cpp definidos para um modelo Strata.

**Primeira inicialização.** O Strata baixa ~58-111 GB, prepara seu "pack" e busca sua camada MTP antes de o servidor abrir sua porta; inicializações seguintes vão direto ao carregamento (1-3 minutos). O card da instância mostra *setting up → downloading (com %) → preparing pack → loading* parseado do log do container, e o botão Logs transmite o log. Duas formas de baixar:

- **Deixe o Strata baixar** no seu volume `/data` (padrão).
- **Pré-baixe com o llamaMan** (botão na seção Strata, ou `POST /api/engines/strata/download`): uma entrada comum na aba Downloads (progresso, pausa/retomada, retry, tokens HF, limites de velocidade) que busca os shards da revisão do Hugging Face fixada pelo Strata para `MODELS_DIR/strata/<tag>/`. Concluídos, eles são bind-montados no container e o setup do Strata os usa em vez de baixar. Um lançamento é recusado enquanto esse download ainda está rodando. A camada MTP (~5 GB) e o pack ainda são preparados pelo Strata na primeira inicialização.

<a id="limits-and-behaviour"></a>
## Limites e comportamento

- **Uma requisição por vez.** Max Concurrent é forçado a 1; requisições seguintes enfileiram no gate do llamaMan (Max Queue Depth, depois 429).
- **RAM.** O Strata carrega 32-62 GB na RAM (`--ulimit memlock=-1` é sempre definido). Um **System Memory Limit** força o modo Low-RAM para ligado, porque o setup do Strata lê a RAM do host e não consegue ver um limite de container; a UI avisa abaixo de ~64 GB.
- **As configurações ficam fixadas no setup do Strata.** Context / image input / KV / Low-RAM / GPU pinning são registrados por modelo no volume de dados. O llamaMan lembra as configurações com que cada modelo ficou healthy por último e passa `REINSTALL=1` só quando elas mudam (os arquivos são reutilizados).
- **Idle timeout** funciona (o sono para o container; a próxima requisição o acorda), mas um cold start leva minutos e a requisição que desperta espera por ele (`STRATA_LOAD_TIMEOUT`).
- **Uma instância por modelo Strata por node** (instâncias de um modelo compartilham seu setup no volume).
- **Só NVIDIA.** A opção fica indisponível, com o motivo, em nodes ROCm / Intel / Vulkan / CPU e quando `STRATA_ENABLED` está desligado. Num cluster cada node anuncia seus motores, então só nodes capazes oferecem modelos Strata.
- O llamaMan bare-metal alcança o Strata do mesmo jeito que o llama-server (porta publicada em `LLAMA_HOST_ADDR`).

---

<a id="design-notes-and-verification"></a>
# Notas de design e verificação

Esta página cobre como a integração funciona, em que ela se apoia no upstream, seus limites, e como verificá-la num host NVIDIA
real.

As referências de upstream apontam para o commit `99f3dbd` (2026-10-03) do Strata.

<a id="how-it-fits"></a>
## Como se encaixa

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- O motor de uma instância é `config["engine"]`. Um valor ausente significa llama.cpp, então cada linha de instância,
  preset e rótulo de container existente continua significando o que significava. Configurações llama.cpp nunca são
  carimbadas com a chave, porque isso mudaria o rótulo `llamaman.config` do container. `tests/fixtures/llamacpp_container_spec.json` guarda os
  kwargs de `containers.run` pré-refatoração de 9 configs × 5 fabricantes GPU, e `tests/test_engines.py` verifica que eles
  estão inalterados.
- Nenhuma migração de armazenamento é necessária. Presets e instâncias são blobs JSON em ambos os backends (`presets.json` /
  `state.json`, e as colunas TEXT `data` do MariaDB), então `engine` e `strata_*` são chaves aditivas.
- Cada modelo Strata é uma entrada virtual, `/strata/<family>-<SIZE>` (id `strata/<family>-<SIZE>`), e é sua própria
  instância e container. Trocar de modelos, desalojar, dormir e acordar funcionam como funcionam para arquivos GGUF.
- `_run_container` pega o `container_spec()` do motor e acrescenta os dispositivos GPU que o motor pede. Um container
  Strata recebe:

  | | |
  |---|---|
  | image | `STRATA_IMAGE` (senão a imagem Strata construída, depois uma baixada) |
  | command | nenhuma (o entrypoint só lê variáveis de ambiente) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | volume de dados → `/data`; logs; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>` (`<family>-<SIZE>` para não-qwen; a pasta de shards sensível a caixa do setup.py) (rw), quando o llamaMan tem shards completos |
  | mem_limit | do System Memory Limit (força `LOW_RAM=on`) |
  | network / ports / labels | como para o llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

<a id="answers-to-the-open-questions"></a>
## Respostas às questões em aberto

1. **Prontidão.** O `/health` responde `{"status": "ok", ..., "service": "strata"}` antes do gate de chave de API
   (`serve/server.py:2172`). A porta HTTP só abre depois que o modelo carregou: `serve()` roda em
   `server.py:3008`, depois de o motor iniciar em `:2925`. Enquanto o Strata está baixando ou carregando, conexões são
   recusadas, então a checagem de saúde existente do llamaMan (`status == "ok"`) não precisa de caso especial. O `/v1/models` retorna uma lista
   vazia enquanto o modelo está descarregado no modo lazy (`server.py:2190`), então não é um bom sinal de prontidão.
   O `/v1/status` precisa da chave de API quando uma está definida.
2. **Configurações.** Tudo é passado como uma variável de ambiente lida pelo `docker-entrypoint.sh`: `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Argumentos de comando são ignorados,
   porque o entrypoint reconstrói `$@` (`:42`, `:65`). O GPU pinning usa a numeração de container: o runtime NVIDIA
   renumera as GPUs do host que o llamaMan anexa, então host `1,3` vira `GPUS=0,1`. O llamaMan não define `API_KEY`, o que
   casa com como ele roda o llama-server (sem chave); seu próprio proxy e auth ficam na frente.
3. **Reusar GGUFs no disco.** O `setup.py` tem `--gguf-dir` (`setup.py:2954`), mas o entrypoint nunca o encaminha.
   O setup, porém, aceita shards inteiros que já estão em `/data/models/<fam tag><SIZE>/<original name>` (p. ex. `IQ3_S`, `coder-IQ1_M`; sensível a caixa): ele confere cada
   um contra seu próprio diretório de tensores e o marca como feito (`setup.py:3389-3405`, "#173"). O llamaMan portanto bind-monta
   sua cópia lá (leitura-escrita, para as marcas `.done`), e só depois que seu próprio download desses shards concluiu.
   Os shards são buscados das revisões do Hugging Face que o Strata fixa (`setup.py:65-70`, espelhadas em
   `core/engines/strata.py`). A camada MTP (~5 GB) e o pack ainda são preparados dentro do container.
4. **Limite de memória / memlock.** O setup lê a RAM de `/proc/meminfo`, que é o total do host, então um container com teto
   precisa pedir `LOW_RAM=on` (comentário do Dockerfile em `:31`, `docs/INSTALL.md:116-118`). Sem ele, o setup escolhe o
   modo in-RAM e o container provavelmente será morto por OOM. O llamaMan força `LOW_RAM=on` sempre que um limite de memória é
   definido, e a UI avisa abaixo de 64 GB. O `memlock=-1` importa porque o motor trava em memória parte de sua arena para DMA de GPU,
   e experts que ele não consegue fixar passam por cópias de host mais lentas (`src/prefill/prefill.cpp:110`, `:207-214`). Pelo
   código-fonte, um limite de memlock ausente parece uma lentidão em vez de uma falha, mas **não consegui verificar isso sem uma
   GPU**. O llamaMan sempre o define.

Também do código-fonte: o Strata serve uma requisição por vez (`server.py:1323`). O Dockerfile define `STRATA_EXECV=1`,
então o servidor é o PID 1 e o `docker stop` o alcança. Um container parado no meio do download retoma de seus arquivos `.part`
na inicialização seguinte.

<a id="design-decisions"></a>
## Decisões de design

- **A concorrência é forçada a 1** no lançamento, na mesclagem de preset, nas edições ao vivo de preset e nos presets
  salvos. O gate força o proxy sidecar, então o enfileiramento acontece no llamaMan (`RequestGate`), não dentro do Strata.
- **`REINSTALL` por fingerprint.** Context, vision, KV, Low-RAM e GPU pinning são registrados na config por modelo do
  Strata no volume. `DATA_DIR/strata_setups.json` lembra as configurações com que cada modelo ficou healthy por última
  vez, indexado por volume de dados e setup tag, e `REINSTALL=1` é enviado só quando diferem. Um fingerprint desconhecido (primeiro
  lançamento via llamaMan, ou um volume montado à mão) também reinstala. Isso custa uma passada de setup que reutiliza os arquivos.
- **Uma instância ativa por modelo Strata por node.** Duas instâncias de um modelo rodariam setup sobre os mesmos arquivos.
- **Um lançamento é recusado enquanto o llamaMan ainda está baixando aquele modelo**, para que o container não o baixe uma
  segunda vez no seu volume.
- **Shards pré-baixados pulam a procedência de checagem de atualização.** Um arquivo fixado não deve ser "atualizado" ao `main` do repo.
- **Os modelos virtuais são listados só onde podem rodar** (`STRATA_ENABLED` mais NVIDIA): em `/api/models`,
  `/api/tags`, `/v1/models` e na resolução de nomes. As varreduras de atualização e o snapshot de modelos do cluster
  continuam baseados apenas em arquivos. Os peers aprendem a disponibilidade de motores de `snapshot.system.engines` (aninhado sob `system`,
  então peers mais antigos o ignoram).
- **O estágio de carregamento** sai dos últimos 64 KB do log do container: os cabeçalhos `=== Step N ===` do setup, a
  linha de progresso `\r` do download, e o "loading the model" do servidor. É best-effort: se falhar, simplesmente não há linha de estágio.
- **vLLM (#53)** seria outro `Engine`: sua própria imagem, comando, porta (`internal_port`), semântica de saúde e campos
  de lançamento. Os lugares que assumem que ambos os motores usam a porta 8080 (os chamadores de `resolve_llama_endpoint`)
  são os que precisariam ser parametrizados nesse caso.

<a id="not-verified-without-a-gpu"></a>
## Não verificado sem GPU

- Um lançamento Strata real, download da primeira inicialização, construção do pack, carregamento e chat. Tudo acima foi
  conferido contra o código-fonte de upstream e com testes unitários que mockeiam o Docker; nenhum container Strata rodou.
- O formato exato do log numa primeira inicialização real. O parser de estágios segue a saída `step()` / `download()` do
  `setup.py` e os prints de inicialização do servidor.
- O comportamento sem `memlock=-1`, e a RAM realmente usada com `LOW_RAM=on` com um limite de container.
- O bind mount aninhado (`/data/models/<tag>` sobre um volume nomeado em `/data`) nas versões de Docker em uso. O Docker
  suporta montagens aninhadas, mas isto não foi executado aqui.

<a id="known-limitations"></a>
## Limitações conhecidas

- Só NVIDIA. O caminho AMD (HIP) do Strata é um build manual experimental no upstream e não é oferecido.
- `LAYER_SPLIT` (colocação de camadas multi-GPU) não está exposto. Fixar várias GPUs usa o split automático do Strata.
- O codificador de visão (mmproj) é sempre baixado pelo próprio Strata, para o volume.
- A própria UI web do Strata (`/`, `/settings`, `/load`, `/unload`) é alcançável pela porta da instância, mas o llamaMan
  não a usa nem a gerencia. Sua função de idle-unload fica sem uso; o sono do llamaMan para o container.
- Mudar uma configuração que está registrada no setup re-executa a passada de setup do Strata na inicialização seguinte (os arquivos são reutilizados).
- Uma primeira inicialização pode demorar mais que o timeout próprio de um cliente HTTP (Open WebUI e outros). O llamaMan segue
  esperando até `STRATA_LOAD_TIMEOUT`, mas o cliente pode desistir antes. Pré-baixe, ou lance uma vez pela UI antes de colocá-lo em uso.
- O catálogo (famílias, tamanhos, revisões fixadas, nomes de arquivo) é um espelho do `setup.py` do Strata. Uma nova
  família ou tamanho do upstream precisa de uma edição correspondente em `core/engines/strata.py`.

<a id="manual-verification-checklist-nvidia-host"></a>
## Checklist de verificação manual (host NVIDIA)

Pré-requisitos: um driver NVIDIA 580+, um `docker run --gpus all` funcionando, 64 GB de RAM recomendados, e uns 150 GB
de disco livre.

1. **Build:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Opcionalmente
   acrescente `--build-arg CUDA_ARCHITECTURES=<your arch>`. Ou, depois do passo 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** Ele constrói só para as gerações de GPU deste node.
2. **Habilite:** defina `STRATA_ENABLED=true` e reinicie o llamaMan.
   - `GET /api/engines` mostra strata `available: true`, com seus modelos (9 na lista embutida).
   - Settings → Docker Images → *Manage Strata images* mostra `strata:latest` como **local**.
3. **Gating:** num node não-NVIDIA (ou com `GPU_TYPE=rocm`), os modelos Strata não são listados, e
   `POST /api/instances` para `/strata/qwen-IQ2_XS` retorna 400 com o motivo NVIDIA.
4. **Lance pela UI:** selecione `strata/qwen-IQ2_XS`.
   - Os campos exclusivos de llama.cpp ficam escondidos, a seção Strata é mostrada, e Max Concurrent fica travado em 1.
   - Defina um limite de memória de 32g e confira que o aviso abaixo de 64 GB aparece; limpe-o de novo.
   - Lance.
5. **Primeira execução (download pelo container):**
   - O card da instância mostra *setting up → downloading NN% → preparing pack → loading*.
   - O botão Logs transmite a saída do setup do Strata.
   - O card fica **healthy**, e o `docker inspect` mostra `Ulimits: memlock -1`, as variáveis de ambiente e o volume `/data`.
6. **Chat:**
   - Pelo proxy Ollama: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Pelo `/v1/chat/completions` em `:42069`, e diretamente na porta da instância.
   - Teste também `model: "qwen3.8-flash-next-iq2_xs"`.
   - Envie duas requisições de uma vez: a segunda enfileira (o card mostra 1/1 active · 1 queued).
7. **Sono e despertar por idle:**
   - Defina Idle Timeout para 1 min (o aviso de cold-start aparece). Depois de um minuto a instância está **sleeping** e seu
     container sumiu.
   - A próxima requisição o acorda (minutos) e é respondida.
   - Nenhuma passada de setup no log (sem linhas `=== Step`; `REINSTALL` ausente do `docker inspect`).
8. **Mudança de configuração:**
   - Mude o contexto para 65536 e reinicie.
   - O `docker inspect` mostra `REINSTALL=1`, e o log mostra uma passada de setup sem download.
   - No restart seguinte, `REINSTALL` está ausente de novo.
9. **Caminho de pré-download:**
   - Escolha outro modelo (p. ex. `strata/qwen-Q2_0`) e clique *Pre-download with llamaMan*.
   - A aba Downloads mostra o download da revisão fixada.
   - Lançar durante o download é recusado.
   - Depois que conclui, lance: o `docker inspect` mostra `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`, e o
     log não mostra nenhum download de shard (só a camada MTP e o pack).
10. **Parar / remover:** pare a instância. O container é removido, e Remove limpa o card.
11. **Reinicie o llamaMan com uma instância Strata rodando:**
    - Ela é reanexada (state restore), ou adotada se o estado foi perdido: delete-a de `state.json` e reinicie.
    - O card mostra o badge Strata e `strata/qwen-IQ2_XS`.
    - As requisições ainda funcionam.
12. **Modo bare-metal:**
    - Rode o llamaMan com `python app.py` (`LLAMAMAN_IN_DOCKER=false`) e repita 4-6.
    - O llamaMan alcança a porta publicada em `LLAMA_HOST_ADDR`.
13. **Cluster:**
    - Adicione um peer não-NVIDIA. Os modelos Strata aparecem para o node NVIDIA só quando ele é o Target node.
    - Lançar no peer NVIDIA a partir da UI do outro node funciona.
14. **llama.cpp inalterado:** lance um modelo GGUF como antes.
    - A linha de comando (Args do `docker inspect`) casa com uma de antes da atualização.
    - Presets salvos antes da atualização carregam e lançam inalterados.
