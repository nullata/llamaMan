# Configurações de lançamento

Cada campo do formulário de lançamento, no que ele mapeia, e as notas de comportamento. Para o passo a passo de "como lançar" em quatro etapas, veja o [README](../../README.md#your-first-model).

## Índice

- [Ferramentas do formulário de lançamento](#launch-form-tools) · [Referência das configurações de lançamento](#launch-settings-reference) · [Atualizações de preset ao vivo](#live-preset-updates) · [Concorrência e enfileiramento](#concurrency--queueing) · [Entrada de imagem e PDF](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Proxy por instância](#per-instance-proxy) · [Tempo ocioso (Idle Timeout)](#idle-timeout)

As notas de comportamento abaixo são concisas; passe o mouse sobre o info-tip do campo na UI para a explicação completa. Todas as flags mapeiam para o llama-server salvo indicação - veja a [referência CLI do llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) para a semântica exata e os padrões atuais.

<a id="launch-form-tools"></a>
## Ferramentas do formulário de lançamento

A linha de ações do formulário de lançamento traz três botões de ferramenta à direita:

- **Export preset** - baixe os valores atuais do formulário como um arquivo JSON (portável entre nodes; não salva no armazenamento).
- **Import preset** - carregue um preset JSON no formulário. Preenche campos apenas; você ainda aperta Save Preset para persistir.
- **Copy llama.cpp command** - copia a invocação `llama-server ...` equivalente para os valores atuais do formulário no seu clipboard. Renderizado no servidor pelo mesmo código de emissão de flags que o lançamento real usa, então ele se mantém em sincronia conforme as flags evoluem.

Com as configurações recolhidas, um botão **Quick Launch** inicia o modelo selecionado direto do seu preset.

Cada instância expõe uma API compatível com OpenAI na porta atribuída. Quando um GGUF é selecionado, o llamaMan lê seus metadados para detectar a contagem de camadas e a mostra ao lado de **GPU Layers** (p. ex. `/ 32`).

<a id="launch-settings-reference"></a>
## Referência das configurações de lançamento

**Model Settings**

| Configuração | Padrão | Flag | Notas |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Teto de tokens de prompt + resposta por requisição |
| **Parallel Slots** | `1` | `--parallel` | Vagas de decodificação concorrentes dentro do llama-server (vagas de KV) |
| **Batch Size** | `2048` | `--batch-size` | Máximo de tokens de prompt processados num único passo. Em branco omite a flag |
| **Micro-batch Size** | `512` | `--ubatch-size` | Batch físico enviado à GPU; maior = prefill mais rápido, mais VRAM. Modelos de embedding e modelos de decisão codificadores (estilo BERT) precisam da entrada inteira num micro-batch. Em branco omite a flag |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`. Quantizado reduz a memória K; funciona com ou sem Flash Attention |
| **V Cache Type** | `f16` | `--cache-type-v` | Os mesmos valores. Qualquer valor quantizado **exige Flash Attention = On** - a UI acinzenta o dropdown e auto-ajusta o V de volta a `f16` quando o FA sai de On |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` omite a flag (o próprio padrão do llama.cpp); `on` força (falha se o backend não consegue); `off` desabilita |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`. Controla como as tags de thinking são extraídas (roteadas a `reasoning_content` vs deixadas inline). `auto` omite a flag |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. Como os pesos do modelo são carregados na memória (sucessor dos deprecados `--mlock`/`--mmap`/`--direct-io`). `auto` = mmap a menos que não suportado, casa com o padrão do llama.cpp e omite a flag |

**Container & Proxy Settings**

| Configuração | Padrão | Flag | Notas |
|---|---|---|---|
| **Port** | *(auto)* | - | Porta pública; precisa cair dentro de `PORT_RANGE_START/END` |
| **CPU Threads** | *(auto)* | `--threads` | Também define a cota de CPU do container (`--cpus N`). Em branco = deixar container e llama-server usarem todos os núcleos |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | Contagem de threads separada para prefill/batch. Em branco recorre a `--threads` no llama-server |
| **System Memory Limit** | *(nenhum)* | - | Teto de RAM Docker (p. ex. `32g`, `8192m`) |
| **Idle Timeout min** | `0` | - | Auto-dormir após N minutos ocioso. `0` = off. Atualizado ao vivo do preset (sem relançamento). Veja [Tempo ocioso (Idle Timeout)](#idle-timeout) |
| **Max Concurrent** | `0` | - | Máximo de requisições de inferência em voo; o excedente enfileira. `0` = sem gate. Veja [Concorrência e enfileiramento](#concurrency--queueing) |
| **Max Queue Depth** | `200` | - | Teto de fila antes do HTTP 429 |
| **Exclude from Max Models** | off | - | Não conta para `LLAMAMAN_MAX_MODELS`, nunca desalojado. Veja [Desalojo de modelos](MODELS.md#model-eviction). Atualizado ao vivo do preset |

**GPU Settings**

| Configuração | Padrão | Flag | Notas |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (aliases) | `-1` = **auto** (o llama.cpp estima a VRAM e escolhe; pode deixar camadas na CPU). `-2` = **all** (força cada camada; falha se não couber). `0` = só CPU, nenhuma GPU anexada ao container. Qualquer N positivo descarrega exatamente N camadas de baixo |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | Só tem sentido em modelos Mixture-of-Experts. `0` = não emite nada. `-1` = `--cpu-moe` (pesos de experts roteados de todas as camadas fixados na CPU). N positivo = `--n-cpu-moe N` (experts das primeiras N camadas na CPU). Encolhe a VRAM ~4-10x em MoEs grandes porque os experts dominam a contagem de pesos mas só alguns disparam por token. Inerte em modelos densos: o regex do llama.cpp (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`) não casa nada e o override é descartado silenciosamente. **Não cobre experts compartilhados** (`_shexp` em DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE) - adicione um override `-ot` em Extra Args para esses |
| **GPU Devices** | *(todas)* | *(nível de container)* | Índices de GPUs do host separados por vírgula a anexar (p. ex. `0,1`). Dentro do container eles são renumerados a partir de 0. Não suportado no Intel Arc |
| **Split Mode** | `layer` | `--split-mode` | `none` (GPU única, ignora Tensor Split) / `layer` (o padrão do llama.cpp; divide camadas inteiras) / `row` (divide linhas de tensor; tipicamente só vence com interconexão rápida como NVLink) |
| **Tensor Split** | *(auto)* | `--tensor-split` | Pesos relativos separados por vírgula por GPU visível ao container (p. ex. `24,16`). Em branco auto-preenche da VRAM total de cada GPU no lançamento. Ignorado quando Split Mode = `none` ou só uma GPU visível |

**Extra Args** *(campo de texto)* - flags adicionais passadas diretamente ao llama-server (p. ex. `--mlock`, ou tipos de especulação n-gram não expostos no formulário).

**Proxy-Side Sampling Overrides** *(toggle)* - quando ligado, o proxy força parâmetros de sampling em toda requisição encaminhada:

| Configuração | Padrão | Faixa |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0 (`0` desabilita) |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 a 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0 (`0` = desabilitado) |

**Speculative Decoding** *(toggle)*

| Configuração | Padrão | Flag | Notas |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`. Só o `draft-mtp` aceita um drafter em branco (recorre às cabeças MTP embutidas do modelo principal) |
| Draft Model | *(nenhum)* | `-md` | Caminho do GGUF do drafter. Obrigatório para todos os tipos exceto `draft-mtp` |
| Draft N Max | `2` | `--spec-draft-n-max` | Máximo de tokens rascunhados por passo |
| Draft N Min *(Advanced)* | *(auto)* | `--spec-draft-n-min` | Em branco omite a flag |
| Draft P Split *(Advanced)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`. Em branco omite a flag |
| Draft P Min *(Advanced)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`. Em branco omite a flag. `0` é um valor real distinto de em branco |

**Image & PDF Input** *(toggle)* - veja [Entrada de imagem e PDF](#image--pdf-input) abaixo.

**Anti-Loop** — dois sub-toggles independentes (ambos desligados por padrão). Veja [Anti-Loop](#anti-loop) abaixo.

<a id="live-preset-updates"></a>
## Atualizações de preset ao vivo

Salvar um preset atualiza instâncias já rodando daquele modelo no lugar onde possível:

- **Ao vivo (sem relançamento):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, os seis campos de proxy-sampling, os cinco campos de loop-detection (a próxima requisição os pega)
- **Exigem relançamento:** tudo que é fixado no container no lançamento (gpu layers, MoE CPU layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, sampler DRY, campos de spec-decoding, flag de embedding, campos de mmproj/PDF, extra args, memory limit)

**Ressalva do proxy-sampling:** se a instância foi lançada com todos de `idle_timeout=0`, `max_concurrent=0`, e `override_enabled=false`, nenhum proxy sidecar foi spawnado. Alternar `override_enabled=true` ao vivo ainda aplica os overrides em requisições roteadas pelos endpoints de compatibilidade da app, mas acessos diretos à porta pública o contornam. Relance para spawnar o proxy nesse caso.

<a id="concurrency--queueing"></a>
## Concorrência e enfileiramento

Quando **Max Concurrent** > 0, o llamaMan coloca um gate FIFO na frente da instância. As requisições excedentes esperam até **Max Queue Depth** antes de retornar HTTP 429. As estatísticas do gate (ativas + enfileiradas) ficam visíveis via API.

**Parallel vs Max Concurrent:** o `Parallel` controla as vagas KV internas do llama-server; o `Max Concurrent` limita quantas requisições o llamaMan encaminha de uma vez. Use ambos juntos (p. ex. `Parallel=4`, `Max Concurrent=4`) para manter o servidor bem alimentado sem transbordar suas vagas.

<a id="image--pdf-input"></a>
## Entrada de imagem e PDF

Ative o toggle **Image & PDF Input** para carregar o projetor multimodal de um modelo de visão (`--mmproj`) junto ao GGUF principal. O campo **MMPROJ Model** recebe o GGUF do projetor (nome de arquivo típico: `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (ligado por padrão): o llama.cpp mantém o projetor na GPU a menos que receba `--no-mmproj-offload`, então *ligado* omite a flag inteiramente (CLI byte-idêntica ao de antes desta opção existir) e *desligado* passa `--no-mmproj-offload` para rodar o codificador de visão na CPU — útil em configurações de VRAM apertada, onde o codificador fixa alguns GB.

**PDFs** são suportados rasterizando cada página para um PNG **antes** de encaminhar ao llama-server (que não tem suporte nativo a PDF). Ative **Accept PDF uploads** (dentro do grupo Image & PDF Input) para ligar isto para uma instância; o toggle mestre **Image & PDF Input** controla o grupo inteiro — desligá-lo também limpa os sub-toggles de PDF, já que páginas rasterizadas não têm modelo de visão para alcançar sem o projetor. Reconhecido em ambos:

- **OpenAI**: `image_url` com uma URL `data:application/pdf;base64,...`, ou o bloco `file` mais recente
- **Ollama**: bytes de PDF em base64 misturados em `images[]` - o llamaMan detecta `%PDF-` e expande PDFs; imagens comuns mantêm seu caminho existente

**Atalho de camada de texto** (desligado por padrão): com **Try text layer first** ligado, o llamaMan roda `pypdf.extract_text()` antes de rasterizar. PDFs nascidos-digitais (Word/LaTeX/impressão de navegador) com texto substantivo (≥ ~50 chars/página) são inseridos como um bloco de texto e pulam o caminho de visão inteiramente. Texto esparso recorre à rasterização. Desligado por padrão porque a rasterização preserva o layout para o modelo de visão, e comportamento previsível em todo PDF vence um atalho silencioso.

**Tetos por requisição:** `PDF DPI` (72-600, padrão 200) e `Max PDF Pages` (1-200, padrão 20). PDFs acima do teto retornam HTTP 400 em vez de serem truncados.

**Concorrência de rasterização:** a renderização de PDF consome muita CPU e RAM e roda na thread do worker gunicorn antes de qualquer gate por instância. Um semáforo de todo o processo limita as rasterizações concorrentes; ajuste via `LLAMAMAN_PDF_MAX_CONCURRENT` (padrão 4).

## Anti-Loop

Dois controles independentes na seção **Anti-Loop** do formulário de lançamento, ambos desligados por padrão. Eles miram o mesmo problema (modelos presos em loops estáveis de saída) em dois pontos diferentes do pipeline; use-os juntos para a melhor cobertura.

**DRY sampler (soft, no sampling)**. Mapeia para `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n` do llama-server. Todo token que estenderia um n-gram visto recentemente recebe uma penalidade de probabilidade no sampling, então o modelo é empurrado para longe de repetir a si mesmo antes de emitir qualquer trecho duplicado. Esta é a linha de defesa *soft* — geralmente suficiente sozinha para prosa. Fixado no lançamento do container; uma edição de preset exige relançamento. Multiplier = 0 desabilita inteiramente (o próprio padrão do llama.cpp).

*DRY Sampler* (fixado no lançamento — exige relançamento):

| Configuração | Padrão | Flag | Notas |
|---|---|---|---|
| Multiplier | `0.8` (valor da UI; `0` desabilita) | `--dry-multiplier` | Força da penalidade; o llama.cpp trata `0` como desabilitado |
| Base | `1.75` | `--dry-base` | Base exponencial ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | N-grams até este comprimento são isentos |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Janela de retrospecto. Em branco omite a flag |

**Output loop detection (hard, a posteriori)**. Quando o DRY não basta — o modelo ainda fica preso num atrator estável — o proxy do llamaman vigia o texto de saída visível do assistente (content + reasoning) enquanto ele transmite ao cliente. A cada cadência de varredura (padrão: inline a cada 64 chunks, mais um fallback de worker de 10s), os últimos `min_chunk_chars` do buffer são conferidos quanto a repetição. Se eles aparecem ≥ `min_repetitions` vezes, o turno é encerrado:

- **A conexão upstream ao llama-server é fechada** — o stop-on-client-disconnect do llama-server interrompe a geração, liberando o modelo para a próxima requisição.
- **Um terminador sintético é injetado no stream do cliente** — clientes OpenAI veem um último delta com `finish_reason=stop` e conteúdo `[llamaman: output loop detected, ending turn]`; clientes Ollama veem `done_reason: "loop_detected"` numa linha `done: true`.

*Output Loop Detection* (no proxy; atualizado ao vivo do preset — a próxima requisição pega os novos limiares):

| Configuração | Padrão | Notas |
|---|---|---|
| Min chunk chars | `200` | Faixa 60-4096. Menor pega loops mais curtos mas dá falsos positivos em conteúdo repetitivo-mas-legítimo |
| Min repetitions | `3` | Faixa 2-20 |
| Buffer size (chars) | `8192` | Janela rolante por requisição em voo. Faixa 512-65536 |
| Scan every N chunks | `64` | Cadência de varredura inline |
| Fallback scan (seconds) | `10` | Worker em segundo plano para streams que emitem tokens devagar demais para atingir o limiar inline |

**Onde ele vigia:**
- **Rotas de compatibilidade em `:42069`** (Ollama `/api/chat`, `/api/generate`; OpenAI `/v1/chat/completions`, `/v1/completions`) — sempre que o toggle está ligado para o preset da instância alvo.
- **Portas por instância (`8000-8020`)** — só quando um proxy sidecar já está presente para aquela instância (isto é, Idle Timeout, Max Concurrent, ou Proxy Sampling Overrides também ligados). A loop detection não spawna um proxy sidecar por si só — habilite um dos outros três para também vigiar acessos diretos.

**Ajuste fino:**
- `min_chunk_chars` (60-4096, padrão 200) — o menor período repetitivo que dispara. Valores menores pegam loops mais curtos mas arriscam falsos positivos em conteúdo repetitivo-mas-legítimo (listas numeradas, refrões de poesia, tabelas markdown). 60-100 é agressivo; 200 é o padrão balanceado para prosa; 400+ só dispara em loops de escala de parágrafo.
- `min_repetitions` (2-20, padrão 3) — quantas cópias exatas do chunk precisam aparecer antes de encerrar. 2 dispara em qualquer chunk dobrado (agressivo); 3-5 equilibram responsividade contra falsos positivos.
- `max_buffer_chars` (512-65536, padrão 8192) — janela rolante mantida por requisição ativa. Precisa ser pelo menos `min_chunk_chars × min_repetitions`.
- `scan_every_n_tokens` (8-4096, padrão 64) — cadência de varredura inline. Menor = kill mais rápido, mais CPU.
- `scan_interval_s` (1-600, padrão 10) — com que frequência o worker em segundo plano revarre buffers que não atingiram o limiar inline.

Mudanças nos limiares de loop-detection valem na *próxima* requisição (streams em voo mantêm os limiares capturados no momento do attach). Mudanças de DRY exigem relançamento já que as flags são fixadas no container.

**Algoritmo de detecção** (v1): pegue os últimos `min_chunk_chars` do buffer como alvo; conte ocorrências exatas; sinalize quando ≥ `min_repetitions`. Não pega quase-repetições à deriva ("The answer is: The answer is: The answer is:" com leve deriva de pontuação entre repetições) — essas são pegas pelas defesas de sampling-time do DRY. Deliberadamente simples e barato.

**Invariante defensivo:** cada hook no caminho de streaming é envolto em `try/except Exception` que loga e segue adiante — um bug no detector nunca pode quebrar o stream de um usuário real.

<a id="per-instance-proxy"></a>
## Proxy por instância

Quando **Idle Timeout**, **Max Concurrent**, ou **Proxy Sampling Overrides** estão ligados para uma instância, o llamaMan insere um proxy WSGI na frente do container llama-server: a porta pública (p. ex. 8000) é o proxy; o llama-server escuta numa porta interna separada. O proxy faz validação de nome de modelo, wake-on-request para instâncias dormindo, limitação de concorrência `RequestGate`, e injeção de sampling.

**Validação de nome de modelo:** nos endpoints de inferência, se o corpo da requisição tem um campo `"model"` ele é comparado ao radical do nome de arquivo do modelo carregado (minúsculas, sem extensão). **Coincidência de prefixo é aceita** (p. ex. `"qwen2.5-0.5b-instruct-q2"` casa com `"qwen2.5-0.5b-instruct-q2_k"`). Não correspondência retorna HTTP 404; para instâncias dormindo, sem despertar. Requisições sem campo `"model"` são encaminhadas incondicionalmente.

<a id="idle-timeout"></a>
## Tempo ocioso (Idle Timeout)

Defina **Idle Timeout min** no formulário de lançamento (0 = desabilitado). Quando habilitado, após N minutos sem requisições o container llama-server é parado para liberar VRAM; a próxima requisição acorda um container novo com a mesma configuração. Os clientes veem a mesma porta com um atraso de cold-start.

Para instâncias gerenciadas pelo proxy (OpenWebUI), use `LLAMAMAN_IDLE_TIMEOUT` em vez disso.
