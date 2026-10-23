# Ajustes de lanzamiento

Cada campo del formulario de lanzamiento, a qué mapea, y las notas de comportamiento. Para el recorrido de cuatro pasos "cómo lanzar", ver el [README](../../README.md#your-first-model).

## Contenido

- [Herramientas del formulario de lanzamiento](#launch-form-tools) · [Referencia de ajustes de lanzamiento](#launch-settings-reference) · [Actualizaciones de presets en vivo](#live-preset-updates) · [Concurrencia y encolado](#concurrency--queueing) · [Entrada de imagen y PDF](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Proxy por instancia](#per-instance-proxy) · [Tiempo de inactividad (Idle Timeout)](#idle-timeout)

Las notas de comportamiento abajo son concisas; pasa el cursor sobre el info-tip del campo en la UI para la explicación completa. Todas las flags mapean a llama-server salvo que se indique - ver la [referencia CLI de llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) para la semántica exacta y los valores por defecto actuales.

<a id="launch-form-tools"></a>
## Herramientas del formulario de lanzamiento

La fila de acciones del formulario de lanzamiento lleva tres botones de herramienta a la derecha:

- **Export preset** - descarga los valores actuales del formulario como archivo JSON (portable entre nodes; no guarda en el almacenamiento).
- **Import preset** - carga un preset JSON en el formulario. Solo rellena campos; aún pulsas Save Preset para persistir.
- **Copy llama.cpp command** - copia la invocación `llama-server ...` equivalente para los valores actuales del formulario a tu portapapeles. Renderizado en el servidor por el mismo código de emisión de flags que usa el lanzamiento real, así que se mantiene sincronizado a medida que evolucionan las flags.

Con los ajustes colapsados, un botón **Quick Launch** inicia el modelo seleccionado directamente desde su preset.

Cada instancia expone una API compatible con OpenAI en su puerto asignado. Cuando se selecciona un GGUF, llamaMan lee sus metadatos para detectar el recuento de capas y lo muestra junto a **GPU Layers** (p. ej. `/ 32`).

<a id="launch-settings-reference"></a>
## Referencia de ajustes de lanzamiento

**Model Settings**

| Ajuste | Por defecto | Flag | Notas |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Tope de tokens de prompt + respuesta por petición |
| **Parallel Slots** | `1` | `--parallel` | Slots de decodificación concurrentes dentro de llama-server (slots KV) |
| **Batch Size** | `2048` | `--batch-size` | Máximo de tokens de prompt procesados en un paso. En blanco omite la flag |
| **Micro-batch Size** | `512` | `--ubatch-size` | Batch físico enviado a la GPU; más grande = prefill más rápido, más VRAM. Los modelos de embedding y de decisión encoder (estilo BERT) necesitan toda su entrada en un solo micro-batch. En blanco omite la flag |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`. La cuantización reduce la memoria K; funciona con o sin Flash Attention |
| **V Cache Type** | `f16` | `--cache-type-v` | Los mismos valores. Cualquier valor cuantizado **requiere Flash Attention = On** - la UI grisea el desplegable y devuelve V a `f16` automáticamente cuando FA deja de estar On |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` omite la flag (el propio valor por defecto de llama.cpp); `on` la fuerza (falla si el backend no puede); `off` la desactiva |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`. Controla cómo se extraen las etiquetas de thinking (enrutadas a `reasoning_content` vs dejadas en línea). `auto` omite la flag |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. Cómo se cargan los pesos del modelo en memoria (sucesor de los deprecados `--mlock`/`--mmap`/`--direct-io`). `auto` = mmap salvo que no sea compatible, coincide con el valor por defecto de llama.cpp y omite la flag |

**Container & Proxy Settings**

| Ajuste | Por defecto | Flag | Notas |
|---|---|---|---|
| **Port** | *(auto)* | - | Puerto público; debe caer dentro de `PORT_RANGE_START/END` |
| **CPU Threads** | *(auto)* | `--threads` | También fija la cuota CPU del container (`--cpus N`). En blanco = dejar que container y llama-server usen todos los núcleos |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | Recuento de threads separado para prefill/batch. En blanco se repliega a `--threads` en llama-server |
| **System Memory Limit** | *(ninguna)* | - | Tope de RAM Docker (p. ej. `32g`, `8192m`) |
| **Idle Timeout min** | `0` | - | Auto-sueño tras N minutos idle. `0` = off. Actualizado en vivo desde el preset (sin relanzar). Ver [Tiempo de inactividad (Idle Timeout)](#idle-timeout) |
| **Max Concurrent** | `0` | - | Máximo de peticiones de inferencia en vuelo; el exceso se encola. `0` = sin control. Ver [Concurrencia y encolado](#concurrency--queueing) |
| **Max Queue Depth** | `200` | - | Tope de cola antes de HTTP 429 |
| **Exclude from Max Models** | off | - | No cuenta hacia `LLAMAMAN_MAX_MODELS`, nunca se desaloja. Ver [Desalojo de modelos](MODELS.md#model-eviction). Actualizado en vivo desde el preset |

**GPU Settings**

| Ajuste | Por defecto | Flag | Notas |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (alias) | `-1` = **auto** (llama.cpp estima la VRAM y elige; puede dejar capas en CPU). `-2` = **all** (fuerza todas las capas; falla si no cabe). `0` = solo CPU, sin GPU adjunta al container. Cualquier N positivo descarga exactamente N capas desde abajo |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | Solo significativo en modelos Mixture-of-Experts. `0` = no emitir nada. `-1` = `--cpu-moe` (pesos de expertos enrutados de todas las capas fijados en CPU). N positivo = `--n-cpu-moe N` (expertos de las primeras N capas en CPU). Reduce la VRAM ~4-10x en MoEs grandes porque los expertos dominan el recuento de pesos pero solo unos pocos se activan por token. Inerte en modelos densos: la regex de llama.cpp (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`) no coincide con nada y el override se descarta silenciosamente. **No cubre expertos compartidos** (`_shexp` en DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE) - añade un override `-ot` en Extra Args para esos |
| **GPU Devices** | *(todas)* | *(a nivel de container)* | Índices de GPU del host separados por comas a adjuntar (p. ej. `0,1`). Dentro del container se renumeran desde 0. No soportado en Intel Arc |
| **Split Mode** | `layer` | `--split-mode` | `none` (GPU única, ignora Tensor Split) / `layer` (el valor por defecto de llama.cpp; parte capas enteras) / `row` (parte filas de tensor; típicamente solo gana con interconexión rápida como NVLink) |
| **Tensor Split** | *(auto)* | `--tensor-split` | Pesos relativos separados por comas por GPU visible por el container (p. ej. `24,16`). En blanco se autorrellena de la VRAM total de cada GPU en el lanzamiento. Ignorado cuando Split Mode = `none` o solo una GPU visible |

**Extra Args** *(campo de texto)* - flags adicionales pasadas directamente a llama-server (p. ej. `--mlock`, o tipos de especulación n-gram no representados en el formulario).

**Proxy-Side Sampling Overrides** *(interruptor)* - cuando está activado, el proxy fuerza parámetros de muestreo en cada petición reenviada:

| Ajuste | Por defecto | Rango |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0 (`0` desactiva) |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 a 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0 (`0` = desactivado) |

**Speculative Decoding** *(interruptor)*

| Ajuste | Por defecto | Flag | Notas |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`. Solo `draft-mtp` acepta un drafter en blanco (se repliega a las cabezas MTP integradas del modelo principal) |
| Draft Model | *(ninguno)* | `-md` | Ruta al GGUF del drafter. Requerido para todos los tipos salvo `draft-mtp` |
| Draft N Max | `2` | `--spec-draft-n-max` | Máximo de tokens de borrador generados por paso |
| Draft N Min *(Advanced)* | *(auto)* | `--spec-draft-n-min` | En blanco omite la flag |
| Draft P Split *(Advanced)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`. En blanco omite la flag |
| Draft P Min *(Advanced)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`. En blanco omite la flag. `0` es un valor real distinto de en blanco |

**Image & PDF Input** *(interruptor)* - ver [Entrada de imagen y PDF](#image--pdf-input) abajo.

**Anti-Loop** — dos sub-interruptores independientes (ambos off por defecto). Ver [Anti-Loop](#anti-loop) abajo.

<a id="live-preset-updates"></a>
## Actualizaciones de presets en vivo

Guardar un preset actualiza en el sitio las instancias ya en ejecución de ese modelo donde es posible:

- **En vivo (sin relanzar):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, los seis campos de proxy-sampling, los cinco campos de loop-detection (la siguiente petición los toma)
- **Requieren relanzar:** todo lo que queda fijado en el container en el lanzamiento (gpu layers, MoE CPU layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, sampler DRY, campos de spec-decoding, flag embedding, campos mmproj/PDF, extra args, memory limit)

**Advertencia de proxy-sampling:** si la instancia se lanzó con todos `idle_timeout=0`, `max_concurrent=0`, y `override_enabled=false`, no se spawneó ningún proxy sidecar. Activar en vivo `override_enabled=true` entonces sigue aplicando overrides en peticiones enrutadas por los endpoints de compatibilidad de la app, pero las peticiones directas al puerto público lo saltan. Relanza para spawnear el proxy en ese caso.

<a id="concurrency--queueing"></a>
## Concurrencia y encolado

Cuando **Max Concurrent** > 0, llamaMan coloca un gate FIFO delante de la instancia. Las peticiones excedentes esperan hasta **Max Queue Depth** antes de devolver HTTP 429. Las estadísticas del gate (activas + encoladas) son visibles vía la API.

**Parallel vs Max Concurrent:** `Parallel` controla los slots KV internos de llama-server; `Max Concurrent` controla cuántas peticiones reenvía llamaMan a la vez. Úsalos juntos (p. ej. `Parallel=4`, `Max Concurrent=4`) para mantener el servidor bien alimentado sin desbordar sus slots.

<a id="image--pdf-input"></a>
## Entrada de imagen y PDF

Activa el interruptor **Image & PDF Input** para cargar el proyector multimodal de un modelo de visión (`--mmproj`) junto a su GGUF principal. El campo **MMPROJ Model** recibe el GGUF del proyector (nombre de archivo típico: `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (activado por defecto): llama.cpp mantiene el proyector en la GPU salvo que reciba `--no-mmproj-offload`, así que *on* omite la flag por completo (CLI idéntico byte a byte al de antes de que existiera esta opción) y *off* pasa `--no-mmproj-offload` para ejecutar el encoder de visión en CPU — útil en configuraciones de VRAM justa, donde el encoder fija unos pocos GB.

Los **PDF** se soportan rasterizando cada página a PNG **antes** de reenviar a llama-server (que no tiene soporte nativo de PDF). Activa **Accept PDF uploads** (dentro del grupo Image & PDF Input) para encender esto en una instancia; el interruptor maestro **Image & PDF Input** controla todo el grupo — apagarlo también limpia los sub-interruptores de PDF, ya que las páginas rasterizadas no tienen modelo de visión al que llegar sin el proyector. Reconocido en ambos:

- **OpenAI**: `image_url` con una URL `data:application/pdf;base64,...`, o el bloque `file` más reciente
- **Ollama**: bytes PDF en base64 mezclados en `images[]` - llamaMan detecta `%PDF-` y expande los PDF; las imágenes ordinarias mantienen su ruta existente

**Atajo de capa de texto** (desactivado por defecto): con **Try text layer first** activado, llamaMan ejecuta `pypdf.extract_text()` antes de rasterizar. Los PDF nacidos digitales (Word/LaTeX/impresión de navegador) con texto sustancial (≥ ~50 caracteres/página) se insertan en línea como un bloque de texto y se saltan la ruta de visión por completo. El texto escaso se repliega a la rasterización. Desactivado por defecto porque la rasterización preserva el layout para el modelo de visión, y un comportamiento predecible en cada PDF gana a un atajo silencioso.

**Topes por petición:** `PDF DPI` (72-600, por defecto 200) y `Max PDF Pages` (1-200, por defecto 20). Los PDF sobre el tope devuelven HTTP 400 en lugar de ser truncados.

**Concurrencia de rasterización:** el renderizado de PDF consume mucha CPU y RAM y se ejecuta en el thread worker de gunicorn antes de cualquier gate por instancia. Un semáforo a nivel de proceso limita las rasterizaciones concurrentes; ajústalo con `LLAMAMAN_PDF_MAX_CONCURRENT` (por defecto 4).

## Anti-Loop

Dos controles independientes en la sección **Anti-Loop** del formulario de lanzamiento, ambos off por defecto. Apuntan al mismo problema (modelos atascados en bucles de salida estables) en dos puntos distintos del pipeline; úsalos juntos para la mejor cobertura.

**Sampler DRY (blando, en el muestreo).** Mapea a `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n` de llama-server. Cada token que extendería un n-gram visto recientemente recibe una penalización de probabilidad en el muestreo, así que se aparta al modelo de repetirse antes de que emita un tramo duplicado. Esta es la línea de defensa *blanda* — suele bastar por sí sola para prosa. Queda fijado en el lanzamiento del container; una edición del preset requiere relanzar. Multiplier = 0 desactiva por completo (el propio valor por defecto de llama.cpp).

*DRY Sampler* (fijado en el lanzamiento — requiere relanzar):

| Ajuste | Por defecto | Flag | Notas |
|---|---|---|---|
| Multiplier | `0.8` (valor de UI; `0` desactiva) | `--dry-multiplier` | Fuerza de la penalización; llama.cpp trata `0` como desactivado |
| Base | `1.75` | `--dry-base` | Base exponencial ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | Los n-grams de hasta esta longitud están exentos |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Ventana de retroceso. En blanco omite la flag |

**Detección de bucles de salida (dura, a posteriori).** Cuando DRY no basta — el modelo sigue atascado en un atractor estable — el proxy de llamaman vigila el texto de salida visible del asistente (content + reasoning) mientras se transmite al cliente. En cada cadencia de escaneo (por defecto: en línea cada 64 chunks, más un fallback de worker de 10 s), se comprueba la repetición de los últimos `min_chunk_chars` del búfer. Si aparecen ≥ `min_repetitions` veces, el turno se termina:

- **Se cierra la conexión upstream a llama-server** — el stop-on-client-disconnect de llama-server detiene la generación, liberando el modelo para la siguiente petición.
- **Se inyecta un terminador sintético en el stream del cliente** — los clientes OpenAI ven un último delta con `finish_reason=stop` y contenido `[llamaman: output loop detected, ending turn]`; los clientes Ollama ven `done_reason: "loop_detected"` en una línea `done: true`.

*Output Loop Detection* (lado del proxy; actualizado en vivo desde el preset — la siguiente petición toma los nuevos umbrales):

| Ajuste | Por defecto | Notas |
|---|---|---|
| Min chunk chars | `200` | Rango 60-4096. Valores más bajos capturan bucles más cortos pero producen falsos positivos en contenido repetitivo-pero-legítimo |
| Min repetitions | `3` | Rango 2-20 |
| Buffer size (chars) | `8192` | Ventana móvil por petición en vuelo. Rango 512-65536 |
| Scan every N chunks | `64` | Cadencia de escaneo en línea |
| Fallback scan (seconds) | `10` | Worker en segundo plano para streams que emiten tokens demasiado lentos para alcanzar el umbral en línea |

**Dónde vigila:**
- **Rutas de compatibilidad en `:42069`** (Ollama `/api/chat`, `/api/generate`; OpenAI `/v1/chat/completions`, `/v1/completions`) — siempre, cuando el interruptor está activado para el preset de la instancia destino.
- **Puertos por instancia (`8000-8020`)** — solo cuando ya hay un proxy sidecar para esa instancia (es decir, Idle Timeout, Max Concurrent, o Proxy Sampling Overrides también activados). La detección de bucles no spawnea por sí sola un proxy sidecar — activa uno de los otros tres para vigilar también las peticiones directas.

**Ajuste fino:**
- `min_chunk_chars` (60-4096, por defecto 200) — el período repetido más pequeño que disparará. Valores más bajos capturan bucles más cortos pero arriesgan falsos positivos en contenido repetitivo-pero-legítimo (listas numeradas, estribillos de poesía, tablas markdown). 60-100 es agresivo; 200 es el valor por defecto equilibrado para prosa; 400+ solo dispara en bucles de escala de párrafo.
- `min_repetitions` (2-20, por defecto 3) — cuántas copias exactas del chunk deben aparecer antes de terminar. 2 dispara en cualquier chunk duplicado (agresivo); 3-5 equilibra capacidad de respuesta contra falsos positivos.
- `max_buffer_chars` (512-65536, por defecto 8192) — ventana móvil mantenida por petición activa. Debe ser al menos `min_chunk_chars × min_repetitions`.
- `scan_every_n_tokens` (8-4096, por defecto 64) — cadencia de escaneo en línea. Más bajo = kill más rápido, más CPU.
- `scan_interval_s` (1-600, por defecto 10) — cada cuánto el worker en segundo plano reescanea los búferes que no alcanzaron el umbral en línea.

Los cambios a los umbrales de loop-detection surten efecto en la *siguiente* petición (los streams en vuelo mantienen los umbrales capturados en el momento de adjuntarse). Los cambios DRY requieren relanzar ya que las flags quedan fijadas en el container.

**Algoritmo de detección** (v1): toma los últimos `min_chunk_chars` del búfer como objetivo; cuenta ocurrencias exactas; marca cuando ≥ `min_repetitions`. No captura casi-repeticiones a la deriva ("The answer is: The answer is: The answer is:" con ligero desvío de puntuación entre reps) — esas las capturan las defensas de muestreo de DRY. Deliberadamente simple y barato.

**Invariante defensivo:** cada hook en la ruta de streaming está envuelto en un `try/except Exception` que registra y continúa — un bug del detector nunca puede romper el stream de un usuario real.

<a id="per-instance-proxy"></a>
## Proxy por instancia

Cuando **Idle Timeout**, **Max Concurrent**, o **Proxy Sampling Overrides** están activados para una instancia, llamaMan inserta un proxy WSGI delante del container llama-server: el puerto público (p. ej. 8000) es el proxy; llama-server escucha en un puerto interno separado. El proxy hace validación de nombre de modelo, wake-on-request para instancias dormidas, limitación de concurrencia `RequestGate`, e inyección de muestreo.

**Validación de nombre de modelo:** en endpoints de inferencia, si el cuerpo de la petición tiene un campo `"model"` se compara contra el nombre base del archivo del modelo cargado (en minúsculas, sin extensión). **Se acepta coincidencia de prefijo** (p. ej. `"qwen2.5-0.5b-instruct-q2"` coincide con `"qwen2.5-0.5b-instruct-q2_k"`). El desajuste devuelve HTTP 404; para instancias dormidas, sin despertar. Las peticiones sin campo `"model"` se reenvían incondicionalmente.

<a id="idle-timeout"></a>
## Tiempo de inactividad (Idle Timeout)

Define **Idle Timeout min** en el formulario de lanzamiento (0 = desactivado). Cuando está habilitado, tras N minutos sin peticiones el container llama-server se detiene para liberar VRAM; la siguiente petición despierta un container nuevo con la misma config. Los clientes ven el mismo puerto con un retardo de cold-start.

Para instancias gestionadas por proxy (OpenWebUI), usa `LLAMAMAN_IDLE_TIMEOUT` en su lugar.
