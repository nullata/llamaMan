# Motor Strata

*(opt-in, solo NVIDIA)* [Strata](https://github.com/Niko1221/Strata) ejecuta la familia mixture-of-experts Qwen3.8-Flash-Next (y sus variantes Coder, Swift 1.5 y Unsloth) en GPUs NVIDIA de consumo más RAM del sistema. llamaMan puede lanzar, monitorizar, hacer de proxy, dormir/despertar y detener instancias Strata junto a las de llama.cpp. El comportamiento de llama.cpp no cambia: la capa de motor (`core/engines/`) mantiene el lanzamiento de llama.cpp idéntico byte a byte (probado con snapshots).

## Contenido

- [Primeros pasos](#getting-started) · [Límites y comportamiento](#limits-and-behaviour) · [Notas de diseño y verificación](#design-notes-and-verification)

<a id="getting-started"></a>
## Primeros pasos

**1. Consigue una imagen** (upstream no publica ninguna; el host necesita driver NVIDIA **580+**, es decir CUDA 13, y el NVIDIA Container Toolkit). Tras habilitar Strata (paso 2), o bien haz pull de una de confianza por nombre en **Settings → Docker Images** (motor *Strata* en la fila de pull), o usa **Manage Strata images → Pull repo & build latest**, que descarga el repositorio a `STRATA_SRC_DIR` y construye `STRATA_BUILD_IMAGE` a través del socket de Docker para las generaciones GPU del node (leídas del driver; todas las soportadas si eso falla), o constrúyela a mano:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. Habilítalo** en el node llamaMan: `STRATA_ENABLED=true` (ver [Strata variables](CONFIGURATION.md#strata)). Settings → Docker Images muestra entonces *Manage Strata images*: las imágenes Strata descargadas, la construida (commits descargado y construido, el botón de build) y su auto-update. El formulario de lanzamiento ofrece Strata cuando una de ellas está en el node.

**3. Lanza.** La biblioteca de modelos gana una entrada virtual por familia × tamaño: `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(experimental)*. Cada una es su propia instancia y container, así que cambiar de modelos funciona como en llama.cpp: lanza o despierta otra instancia, con las reglas habituales de [desalojo](MODELS.md#model-eviction) (una instancia Strata es un modelo de chat). Las entradas también aparecen en `/api/tags` y `/v1/models`, y las peticiones pueden usar el id (`strata/qwen-IQ2_XS`, sin distinguir mayúsculas/minúsculas, `:tag` ignorado) o el nombre servido por Strata (`qwen3.8-flash-next-iq2_xs`).

El formulario de lanzamiento oculta los campos solo-llama.cpp (GPU layers, MoE, threads, parallel, KV types, flash attention, spec decoding, mmproj, DRY, extra args) y muestra **Image Input** (no / GPU / CPU), **KV Cache** (int8 por defecto de Strata, `int8`, `q4_0`, `k8v4`) y **Low-RAM Mode** (auto / on / off). La API rechaza los campos solo-llama.cpp definidos para un modelo Strata.

**Primer arranque.** Strata descarga ~58-111 GB, prepara su "pack" y obtiene su capa MTP antes de que el servidor abra su puerto; los arranques posteriores van directo a la carga (1-3 minutos). La tarjeta de instancia muestra *setting up → downloading (con %) → preparing pack → loading* parseado del log del container, y el botón Logs transmite el log. Dos formas de descargar:

- **Deja que Strata descargue** en su volumen `/data` (por defecto).
- **Pre-descarga con llamaMan** (botón en la sección Strata, o `POST /api/engines/strata/download`): una entrada ordinaria en la pestaña Downloads (progreso, pausa/reanudar, reintento, tokens HF, límites de velocidad) que obtiene los shards de la revisión de Hugging Face fijada por Strata en `MODELS_DIR/strata/<tag>/`. Una vez completos se montan en el container mediante bind mount y el setup de Strata los usa en lugar de descargar. Un lanzamiento se rechaza mientras esa descarga sigue en curso. La capa MTP (~5 GB) y el pack los sigue preparando Strata en el primer arranque.

<a id="limits-and-behaviour"></a>
## Límites y comportamiento

- **Una petición a la vez.** Max Concurrent se fuerza a 1; las peticiones siguientes se encolan en el gate de llamaMan (Max Queue Depth, y luego 429).
- **RAM.** Strata carga 32-62 GB en RAM (`--ulimit memlock=-1` se define siempre). Un **System Memory Limit** fuerza Low-RAM mode activado, porque el setup de Strata lee la RAM del host y no puede ver un límite de container; la UI avisa por debajo de ~64 GB.
- **Los ajustes quedan fijados en el setup de Strata.** Context / image input / KV / Low-RAM / GPU pinning se registran por modelo en el volumen de datos. llamaMan recuerda los ajustes con los que cada modelo quedó healthy por última vez y pasa `REINSTALL=1` solo cuando cambian (los archivos se reutilizan).
- **Idle timeout** funciona (sleep detiene el container; la siguiente petición lo despierta), pero un cold start tarda minutos y la petición que despierta espera por él (`STRATA_LOAD_TIMEOUT`).
- **Una instancia por modelo Strata por node** (las instancias de un modelo comparten su setup en el volumen).
- **Solo NVIDIA.** La opción no está disponible, con la razón, en nodes ROCm / Intel / Vulkan / CPU y cuando `STRATA_ENABLED` está off. En un cluster cada node anuncia sus motores, así que solo los nodes capaces ofrecen modelos Strata.
- llamaMan en bare-metal alcanza Strata igual que llama-server (puerto publicado en `LLAMA_HOST_ADDR`).

---

<a id="design-notes-and-verification"></a>
# Notas de diseño y verificación

Esta página cubre cómo funciona la integración, en qué depende de upstream, sus límites, y cómo verificarla en un host NVIDIA
real.

Las referencias a upstream apuntan al commit `99f3dbd` de Strata (2026-10-03).

<a id="how-it-fits"></a>
## Cómo encaja

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- El motor de una instancia es `config["engine"]`. Un valor ausente significa llama.cpp, así que cada fila de instancia
  existente, preset y etiqueta de container sigue significando lo mismo. Las configs de llama.cpp nunca se estampan con
  la clave, porque cambiaría la etiqueta `llamaman.config` del container. `tests/fixtures/llamacpp_container_spec.json`
  guarda los kwargs de `containers.run` pre-refactor para 9 configs × 5 fabricantes GPU, y `tests/test_engines.py`
  verifica que no han cambiado.
- No hace falta migración de almacenamiento. Presets e instancias son blobs JSON en ambos backends (`presets.json` /
  `state.json`, y las columnas TEXT `data` de MariaDB), así que `engine` y `strata_*` son claves aditivas.
- Cada modelo Strata es una entrada virtual, `/strata/<family>-<SIZE>` (id `strata/<family>-<SIZE>`), y es su propia
  instancia y container. Cambiar de modelos, desalojo, sueño y despertar funcionan como con los archivos GGUF.
- `_run_container` toma el `container_spec()` del motor y añade los dispositivos GPU que el motor pide. Un container
  Strata obtiene:

  | | |
  |---|---|
  | image | `STRATA_IMAGE` (si no la construida, y si no una imagen Strata descargada) |
  | command | ninguna (el entrypoint solo lee variables de entorno) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | volumen de datos → `/data`; logs; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>` (`<family>-<SIZE>` para los no-qwen; la carpeta de shards sensible a mayúsculas de setup.py) (rw), cuando llamaMan tiene shards completos |
  | mem_limit | de System Memory Limit (fuerza `LOW_RAM=on`) |
  | network / ports / labels | como para llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

<a id="answers-to-the-open-questions"></a>
## Respuestas a las preguntas abiertas

1. **Readiness.** `/health` responde `{"status": "ok", ..., "service": "strata"}` antes de la gate de clave API
   (`serve/server.py:2172`). El puerto HTTP solo se abre después de que el modelo se haya cargado: `serve()` corre en
   `server.py:3008`, tras arrancar el motor en `:2925`. Mientras Strata descarga o carga, las conexiones se rechazan,
   así que el health check existente de llamaMan (`status == "ok"`) no necesita caso especial. `/v1/models` devuelve una
   lista vacía mientras el modelo está descargado en modo perezoso (`server.py:2190`), así que no es buena señal de
   readiness. `/v1/status` necesita la clave API cuando hay una definida.
2. **Ajustes.** Todo se pasa como variable de entorno leída por `docker-entrypoint.sh`: `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Los argumentos de comando se ignoran,
   porque el entrypoint reconstruye `$@` (`:42`, `:65`). El GPU pinning usa numeración de container: el runtime NVIDIA
   renumera las GPUs del host que llamaMan adjunta, así que host `1,3` se convierte en `GPUS=0,1`. llamaMan no define
   `API_KEY`, lo cual coincide con cómo ejecuta llama-server (sin clave); su propio proxy y auth están delante.
3. **Reutilizar GGUFs en disco.** `setup.py` tiene `--gguf-dir` (`setup.py:2954`), pero el entrypoint nunca lo reenvía.
   Setup, eso sí, acepta shards completos que ya están en `/data/models/<fam tag><SIZE>/<original name>` (p. ej. `IQ3_S`, `coder-IQ1_M`; sensible a mayúsculas): los comprueba
   contra su propio directorio de tensores y los marca hechos (`setup.py:3389-3405`, "#173"). llamaMan por tanto monta
   bind su copia ahí (lectura-escritura, para las marcas `.done`), y solo cuando su propia descarga de esos shards ha
   completado. Los shards se obtienen de las revisiones de Hugging Face que Strata fija (`setup.py:65-70`, replicadas en
   `core/engines/strata.py`). La capa MTP (~5 GB) y el pack se siguen preparando dentro del container.
4. **Límite de memoria / memlock.** Setup lee RAM de `/proc/meminfo`, que es el total del host, así que un container
   limitado debe pedir `LOW_RAM=on` (comentario del Dockerfile en `:31`, `docs/INSTALL.md:116-118`). Sin él, setup elige
   el modo en-RAM y es probable que el container sea OOM-killed. llamaMan fuerza `LOW_RAM=on` siempre que hay un límite
   de memoria definido, y la UI avisa por debajo de 64 GB. `memlock=-1` importa porque el motor bloquea en memoria (page-lock) parte de su
   arena para DMA de GPU, y los expertos que no puede fijar pasan por copias de host más lentas (`src/prefill/prefill.cpp:110`, `:207-214`). Según el
   código fuente, un límite de memlock ausente parece una ralentización más que un fallo, pero **no pude verificarlo sin
   una GPU**. llamaMan siempre lo define.

También del código fuente: Strata sirve una petición a la vez (`server.py:1323`). El Dockerfile define `STRATA_EXECV=1`,
así que el servidor es PID 1 y `docker stop` lo alcanza. Un container detenido a mitad de descarga se reanuda desde sus
archivos `.part` en el siguiente arranque.

<a id="design-decisions"></a>
## Decisiones de diseño

- **La concurrencia se fuerza a 1** en el lanzamiento, en la fusión de preset, en ediciones en vivo y en los presets
  guardados. El gate fuerza el proxy sidecar, así que el encolado ocurre en llamaMan (`RequestGate`), no dentro de Strata.
- **`REINSTALL` por huella digital.** Context, vision, KV, Low-RAM y GPU pinning se registran en la config por modelo de
  Strata en el volumen. `DATA_DIR/strata_setups.json` recuerda los ajustes con los que cada modelo se puso healthy por
  última vez, indexado por volumen de datos y setup tag, y `REINSTALL=1` se envía solo cuando difieren. Una huella
  desconocida (primer lanzamiento vía llamaMan, o un volumen montado a mano) también reinstala. Cuesta una pasada de setup que reutiliza los archivos.
- **Una instancia viva por modelo Strata por node.** Dos instancias de un modelo ejecutarían setup sobre los mismos archivos.
- **Un lanzamiento se rechaza mientras llamaMan sigue descargando ese modelo**, para que el container no lo descargue una
  segunda vez en su volumen.
- **Los shards pre-descargados se saltan la procedencia de comprobación de actualización.** Un archivo fijo no debe "actualizarse" al `main` del repo.
- **Los modelos virtuales se listan solo donde pueden correr** (`STRATA_ENABLED` más NVIDIA): en `/api/models`,
  `/api/tags`, `/v1/models` y la resolución de nombres. Los escaneos de actualización y el snapshot de modelos del
  cluster siguen basándose solo en archivos. Los peers aprenden la disponibilidad de motores de `snapshot.system.engines` (anidado bajo `system`, así que peers más antiguos lo ignoran).
- **La etapa de carga** sale de los últimos 64 KB del log del container: las cabeceras `=== Step N ===` de setup, la
  línea de progreso `\r` de la descarga, y el "loading the model" del servidor. Es best-effort: si falla, simplemente no se muestra línea de etapa.
- **vLLM (#53)** sería otro `Engine`: su propia imagen, comando, puerto (`internal_port`), semántica de salud y campos
  de lanzamiento. Los sitios que asumen que ambos motores usan el puerto 8080 (los llamantes de `resolve_llama_endpoint`)
  son los que parametrizar entonces.

<a id="not-verified-without-a-gpu"></a>
## No verificado sin GPU

- Un lanzamiento Strata real, descarga del primer arranque, construcción del pack, carga y chat. Todo lo anterior se
  comprobó contra el código fuente de upstream y con tests unitarios que mockean Docker; ningún container Strata ha corrido.
- El formato exacto del log en un primer arranque real. El parser de etapas sigue la salida `step()` / `download()` de
  `setup.py` y los prints de arranque del servidor.
- El comportamiento sin `memlock=-1`, y la RAM realmente usada con `LOW_RAM=on` con un límite de container.
- El bind mount anidado (`/data/models/<tag>` sobre un volumen nombrado en `/data`) en las versiones de Docker en uso.
  Docker soporta montajes anidados, pero esto no se ha ejecutado aquí.

<a id="known-limitations"></a>
## Limitaciones conocidas

- Solo NVIDIA. La ruta AMD (HIP) de Strata es un build manual experimental en upstream y no se ofrece.
- `LAYER_SPLIT` (colocación de capas multi-GPU) no está expuesto. Fijar varias GPUs usa el split automático de Strata.
- El encoder de visión (mmproj) siempre lo descarga Strata mismo, al volumen.
- La propia UI web de Strata (`/`, `/settings`, `/load`, `/unload`) es alcanzable por el puerto de la instancia, pero
  llamaMan no la usa ni la gestiona. Su función de idle-unload está sin usar; el sleep de llamaMan detiene el container.
- Cambiar un ajuste registrado en el setup vuelve a ejecutar la pasada de setup de Strata en el siguiente arranque (los archivos se reutilizan).
- Un primer arranque puede tardar más que el timeout propio de un cliente HTTP (Open WebUI y otros). llamaMan sigue
  esperando hasta `STRATA_LOAD_TIMEOUT`, pero el cliente puede rendirse antes. Pre-descarga, o lanza una vez desde la UI antes de ponerlo en servicio.
- El catálogo (familias, tamaños, revisiones fijas, nombres de archivo) es un espejo del `setup.py` de Strata. Una
  familia o tamaño nuevo de upstream necesita una edición coincidente en `core/engines/strata.py`.

<a id="manual-verification-checklist-nvidia-host"></a>
## Lista de verificación manual (host NVIDIA)

Requisitos previos: un driver NVIDIA 580+, un `docker run --gpus all` funcional, 64 GB de RAM recomendados, y unos 150 GB
de disco libre.

1. **Build:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Opcionalmente
   añade `--build-arg CUDA_ARCHITECTURES=<your arch>`. O, tras el paso 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** Construye solo para las generaciones GPU de este node.
2. **Habilitar:** define `STRATA_ENABLED=true` y reinicia llamaMan.
   - `GET /api/engines` muestra strata `available: true`, con sus modelos (9 en la lista integrada).
   - Settings → Docker Images → *Manage Strata images* muestra `strata:latest` como **local**.
3. **Gating:** en un node no-NVIDIA (o con `GPU_TYPE=rocm`), los modelos Strata no se listan, y
   `POST /api/instances` para `/strata/qwen-IQ2_XS` devuelve 400 con la razón de NVIDIA.
4. **Lanzar desde la UI:** selecciona `strata/qwen-IQ2_XS`.
   - Los campos solo-llama.cpp están ocultos, la sección Strata se muestra, y Max Concurrent queda fijado en 1.
   - Define un límite de memoria de 32g y comprueba que aparece el aviso de por debajo de 64 GB; límpialo otra vez.
   - Lanza.
5. **Primera ejecución (descarga del container):**
   - La tarjeta de instancia muestra *setting up → downloading NN% → preparing pack → loading*.
   - El botón Logs transmite la salida del setup de Strata.
   - La tarjeta pasa a **healthy**, y `docker inspect` muestra `Ulimits: memlock -1`, las variables de entorno y el
     volumen `/data`.
6. **Chat:**
   - Por el proxy Ollama: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Por `/v1/chat/completions` en `:42069`, y directamente en el puerto de la instancia.
   - Prueba también `model: "qwen3.8-flash-next-iq2_xs"`.
   - Envía dos peticiones a la vez: la segunda se encola (la tarjeta muestra 1/1 active · 1 queued).
7. **Sleep y despertar por idle:**
   - Define Idle Timeout a 1 min (aparece el aviso de cold-start). Tras un minuto la instancia está **sleeping** y su
     container ha desaparecido.
   - La siguiente petición lo despierta (minutos) y es respondida.
   - Sin pasada de setup en el log (sin líneas `=== Step`; `REINSTALL` ausente de `docker inspect`).
8. **Cambio de ajustes:**
   - Cambia el contexto a 65536 y reinicia.
   - `docker inspect` muestra `REINSTALL=1`, y el log muestra una pasada de setup sin descarga.
   - En el siguiente reinicio, `REINSTALL` vuelve a estar ausente.
9. **Ruta de pre-descarga:**
   - Elige otro modelo (p. ej. `strata/qwen-Q2_0`) y haz clic en *Pre-download with llamaMan*.
   - La pestaña Downloads muestra la descarga de la revisión fija.
   - Lanzar durante la descarga se rechaza.
   - Tras completarse, lanza: `docker inspect` muestra `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`, y el
     log no muestra descarga de shards (solo la capa MTP y el pack).
10. **Stop / remove:** detén la instancia. El container se elimina, y Remove limpia la tarjeta.
11. **Reinicia llamaMan con una instancia Strata corriendo:**
    - Se readjunta (restore de estado), o se adopta si el estado se perdió: bórrala de `state.json` y reinicia.
    - La tarjeta muestra la insignia Strata y `strata/qwen-IQ2_XS`.
    - Las peticiones siguen funcionando.
12. **Modo bare-metal:**
    - Ejecuta llamaMan con `python app.py` (`LLAMAMAN_IN_DOCKER=false`) y repite 4-6.
    - llamaMan alcanza el puerto publicado en `LLAMA_HOST_ADDR`.
13. **Cluster:**
    - Añade un peer no-NVIDIA. Los modelos Strata aparecen para el node NVIDIA solo cuando es el Target node.
    - Lanzar en el peer NVIDIA desde la UI del otro node funciona.
14. **llama.cpp sin afectar:** lanza un modelo GGUF como siempre.
    - La línea de comando (Args de `docker inspect`) coincide con una de antes de la actualización.
    - Los presets guardados antes de la actualización se cargan y lanzan sin cambios.
