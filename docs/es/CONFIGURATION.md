# Configuración y almacenamiento

Instalaciones bare-metal, el modelo de autenticación en detalle, estadísticas de GPU, registro de peticiones, backends de almacenamiento, y la referencia completa de variables de entorno.

## Contenido

- [Ejecución en bare-metal](#running-bare-metal) · [Autenticación en detalle](#authentication-detail) · [Estadísticas de GPU](#gpu-stats) · [Registro de peticiones y estadísticas](#request-recording--stats) · [Backends de almacenamiento](#storage-backends) · [Sobrevivir a una caída de la base de datos (espejo local)](#surviving-a-database-outage-local-mirror) · [Variables de entorno](#environment-variables)

<a id="running-bare-metal"></a>
## Ejecución en bare-metal

Útil para desarrollo o hosts (p. ej. WSL) donde ejecutar el propio gestor en Docker es engorroso. llama-server sigue corriendo en containers.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

Un único proceso bajo gunicorn sirve la UI/API en `:5000` y el proxy en `:42069` (`python app.py` sin gunicorn ejecuta el servidor de desarrollo Flask solo en `:5000`). Container-vs-bare-metal se autodetecta; si la detección alguna vez es errónea para tu entorno, define `LLAMAMAN_IN_DOCKER=true` o `false` explícitamente.

<a id="authentication-detail"></a>
## Autenticación en detalle

Dos tipos de credenciales:

- **Cookies de sesión** para la UI (creadas vía `/setup` en el primer lanzamiento)
- **Claves API** para clientes externos (`Authorization: Bearer llm-xxxxxxxxxx`), gestionadas en **API Keys** de la UI

El interruptor **"Require authentication for all endpoints"** (activado por defecto) controla si los endpoints de servicio de modelos exigen un token bearer:

| Interruptor | Endpoints de modelos (`/api/chat`, `/v1/chat/completions`, puertos por instancia) | Endpoints de gestión (`/api/instances`, etc.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (por defecto) | Token bearer requerido | Token bearer o sesión |
| **OFF** | Abierto | Token bearer o sesión |

Las tres superficies de puerto (5000, 42069, 8000-8020) pasan por el mismo hook de autenticación.

<a id="gpu-stats"></a>
## Estadísticas de GPU

VRAM y utilización por GPU, consultadas nativamente - no se requiere ningún llama-server en ejecución.

| Fabricante | Método | Requisito |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML directo) | Descomenta `deploy.resources.reservations` en `docker-compose.yml` para la capability `utility` del toolkit NVIDIA |
| AMD | sysfs `/sys/class/drm` | Montaje de volumen `/sys/class/drm:ro` (incluido por defecto) |
| Intel Arc | sysfs `/sys/class/drm` | El mismo montaje que AMD |

Sin acceso nativo, llamaMan se repliega a ejecutar `nvidia-smi` / `rocm-smi` dentro de un container en marcha. Las estadísticas siempre reflejan el estado GPU completo del host.

<a id="request-recording--stats"></a>
## Registro de peticiones y estadísticas

En **Settings → App Settings → Request recording**: **Off** (por defecto) / **Per request** / **Per conversation** (turnos agrupados por un hash de contenido del system prompt + primer mensaje de usuario). Cada registro captura los cuerpos + campos de sobre (modelo, endpoint, estado, duración, conteos de tokens) más **métricas precisas por turno** (throughput medido sobre la ventana de generación; TTFT). Los registros viven en `RECORDINGS_DIR` (JSON) o la tabla `request_log` (MariaDB). **Retention (days)** poda cada hora (`0` = conservar para siempre).

Cada tarjeta de instancia expone un botón **Stats** (conteo de peticiones, throughput medio/pico, TTFT/latencia medios, totales de tokens, franja de tiempo activo) agregado desde el request log - por lo que persiste después de detener la instancia y muestra un estado vacío cuando el registro está desactivado. El enlace **Logging** de la cabecera abre un dashboard a página completa (mosaicos de resumen, conversaciones recientes, desglose por conversación) con un selector de ventana temporal 24 h / 7 d / 30 d / Todo más filtros por modelo y por día sobre la lista de conversaciones.

<a id="storage-backends"></a>
## Backends de almacenamiento

### JSON (por defecto)

Cero configuración. Archivos bajo `DATA_DIR` (`/data`): `state.json`, `presets.json`, `users.json`, `settings.json`, `api_keys.json`, `request_log/`. Los registros de instancia y descarga van a `LOGS_DIR` (`/tmp/llama-logs`).

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

Las tablas se crean automáticamente en la primera conexión. Los registros de peticiones van a la tabla `request_log` en lugar del directorio JSON `request_log/` (`RECORDINGS_DIR` no tiene efecto con MariaDB).

Los metadatos de modelos por node (repo de origen + hash de contenido) viven en una tabla `model_files` de ámbito node indexada por `(node_id, model_path)`. Al actualizar, cada node copia sus propias entradas fuera del blob heredado de ajustes compartidos (solo rutas que existen en su propio disco) y conserva el blob como repliegue de lectura y ruta de rollback. Las migraciones están versionadas por node.

> **Nota InnoDB:** `(node_id, model_path)` son `(64 + 700) × 4 = 3056` bytes en utf8mb4, justo por debajo del límite de índice de 3072 bytes de InnoDB, así que la tabla se crea con `ROW_FORMAT=DYNAMIC`.

MariaDB es también el prerrequisito del [Clustering](CLUSTERING.md) y la [Knowledge Base](KNOWLEDGE-BASE.md).

<a id="surviving-a-database-outage-local-mirror"></a>
## Sobrevivir a una caída de la base de datos (espejo local)

*Opcional, desactivado por defecto.*

Con `DATABASE_URL` definido, la base de datos está en la ruta crítica de **cada** petición (las comprobaciones de autenticación leen ajustes y verifican claves API en cada llamada). Si se vuelve inalcanzable, el node deja de servir, y un reinicio de container en plena caída no puede arrancar en absoluto.

Activa **Settings → App settings → "Keep a local mirror of the database"** (o define `LLAMAMAN_DB_MIRROR=1`) y el node mantiene una copia write-through en `DATA_DIR/db_mirror/`.

| Sigue funcionando sin conexión | Bloqueado sin conexión |
|---|---|
| Toda la inferencia (ambos puertos) | Crear el primer usuario (`/setup`) - una instalación nueva no tiene espejo al que replegarse de todos modos |
| Lanzar / detener / dormir / despertar modelos | |
| Descargas, hashing de modelos y comprobaciones de actualización | |
| Crear y editar **presets** (incluidos los de modelos recién descargados) | |
| Crear y revocar **claves API** | |
| Cambiar **ajustes** de descarga y de la aplicación | |
| Añadir y quitar **tokens de Hugging Face** | |

Los cambios sin conexión van a un journal de solo adición y se reproducen al reconectar (sonda de 10 s). La reproducción registra la *edición*, no la fila resultante, así que un cambio hecho por otro node durante la caída sobrevive (un campo de preset distinto, una clave de ajustes distinta, un token HF distinto quedan todos intactos).

**Requisitos y advertencias:**

- **`DATA_DIR` debe ser un volumen persistente** (ya lo es en el archivo compose de ejemplo).
- **Cada node necesita su propio `DATA_DIR`.** El espejo está estampado con el id del node propietario; dos nodes compartiendo uno desactivan el espejo en lugar de corromper ninguna de las vistas.
- **Los secretos aterrizan en el disco local.** Los ajustes se copian literalmente; **los tokens de Hugging Face se espejan en texto plano.** Los hashes de contraseña + clave API ya están hasheados. Si elegiste MariaDB en parte para mantener los secretos fuera de hosts individuales, este compromiso es la razón de que la función sea opt-in.
- **El balanceo entre nodes se detiene durante una caída.** El estado de actividad (liveness) de los peers se guarda en la base de datos; un node degradado solo se ve a sí mismo.
- **Los cambios de claves API son locales hasta reconectar.** Una clave creada sin conexión funciona solo en este node; una clave **revocada** sin conexión sigue siendo válida en los demás nodes hasta reconectar. Planifica las revocaciones en consecuencia.
- **Los registros de peticiones se descartan, no se almacenan en búfer,** sin conexión (los registros llevan cuerpos completos; acumularlos en disco durante una caída larga podría llenar el volumen).
- **Desactivar el espejo en plena caída se difiere** hasta reconectar.
- Las migraciones de esquema nunca se ejecutan degradadas; se ejecutan contra la base de datos real al reconectar, antes de cualquier reproducción del journal.

<a id="environment-variables"></a>
## Variables de entorno

### Core

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(requerida)* | **Requerida.** Identidad única estable - clave de partición del estado almacenado e identidad de cluster. Cualquier cadena; elige una vez, consérvala (cambiarla después deja huérfano el estado almacenado). |
| `MODELS_DIR` | `/models` | Directorio escaneado en busca de archivos de modelos (ruta de container) |
| `DATA_DIR` | `/data` | Directorio para config/estado persistente (archivos JSON) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Directorio para registros del request log por conversación. Solo backend JSON; ignorado cuando `DATABASE_URL` está definido |
| `LOGS_DIR` | `/tmp/llama-logs` | Directorio para registros de instancia y descarga (ruta de container) |
| `HOST_MODELS_DIR` | *(igual que `MODELS_DIR`)* | Ruta absoluta **del lado del host** del volumen de modelos. Debe coincidir con el lado izquierdo de `-v /host/path:/models` - se pasa al daemon de Docker al spawnear containers hermanos |
| `HOST_LOGS_DIR` | *(igual que `LOGS_DIR`)* | Ruta absoluta del lado del host del volumen de registros. El mismo requisito que `HOST_MODELS_DIR` |
| `ARCHIVE_DIR` | *(sin definir)* | Ruta de container donde se monta un volumen de archivo (p. ej. `/archive`). Definirlo habilita [Archivo de modelos](MODELS.md#model-archive); sin definir lo oculta |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Pool de puertos públicos de llama-server/proxy |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Pool de puertos internos usado cuando el modo proxy está habilitado |
| `LLAMAMAN_PROXY_PORT` | `42069` | Puerto para el proxy compatible con Ollama |
| `LLAMAMAN_MAX_MODELS` | `0` | Máximo de modelos de chat concurrentes vía el proxy (desalojo LRU; 0 = ilimitado). Reglas: [Desalojo de modelos](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Minutos de idle-timeout para instancias gestionadas por el proxy (0 = desactivado) |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Máximo de rasterizaciones de PDF concurrentes en todo el proceso. Cada rasterización spawnea poppler y consume unos cientos de MB de RAM transitoria a DPI altos |
| `SECRET_KEY` | *(auto)* | Secreto de sesión Flask. Auto-derivado del machine-id si no se define. Defínelo para despliegues multi-réplica |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Nombre de la cookie de sesión. Con espacio de nombres para que llamaman coexista con otras apps Flask en el mismo host - las cookies se delimitan por host+ruta, no por puerto |
| `DATABASE_URL` | *(sin definir)* | Cadena de conexión MariaDB/MySQL. Sin definir = archivos JSON |
| `LLAMAMAN_DB_MIRROR` | *(sin definir)* | Fuerza el espejo de DB local activado (`1`) / desactivado (`0`), anulando el ajuste por node. Solo tiene sentido con `DATABASE_URL` |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout en segundos para las comprobaciones de salud de instancia |
| `MODEL_LOAD_TIMEOUT` | `300` | Segundos de espera a que un modelo quede healthy. Súbelo para modelos muy grandes |
| `REQUEST_TIMEOUT` | `300` | Timeout de **lectura** en segundos para peticiones upstream a llama-server, reenvío entre nodes y esperas de adquisición de gate. **No** rige el tiempo de conexión a peers (límite separado de 5 s) |

### Docker / GPU

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | Imagen Docker de llama.cpp para todos los containers spawneados. Auto-seleccionada del fabricante GPU detectado; defínela para fijar una versión/backend |
| `LLAMA_NETWORK` | `llamaman-net` | Red Docker para llamaman + todos los containers llama-server. Se crea si falta |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Prefijo de nombre para los containers llama-server spawneados |
| `LLAMAMAN_IN_DOCKER` | *(auto)* | Si llamaman mismo corre en un container. Auto-detectado desde archivos marcador + cgroups. Define `true`/`false` para anular |
| `LLAMA_HOST_ADDR` | `localhost` | Dirección de host usada para alcanzar los puertos publicados de los containers spawneados al correr en bare-metal |
| `GPU_TYPE` | *(auto)* | Anula la detección del fabricante GPU: `cuda`, `rocm`, `intel`, o `vulkan` (`vulkan` es solo opt-in; elige `/dev/dri` + los GIDs render/video del host y la imagen `server-vulkan`) |
| `LLAMA_GPU_DEVICES` | *(todas)* | Índices de GPU separados por comas visibles para todos los containers spawneados (p. ej. `0,1,3`). **GPU Devices** por instancia anula. No soportado en Intel Arc |

### Strata

Opcional; ver [Strata](STRATA.md).

| Variable | Por defecto | Descripción |
|---|---|---|
| `STRATA_ENABLED` | `false` | Ofrecer modelos Strata en este node. Necesita también una GPU NVIDIA (`GPU_TYPE=cuda` o detectada) |
| `STRATA_IMAGE` | `strata:latest` | Imagen Strata preferida para los lanzamientos. Cuando no está en el node, se usa la imagen construida, y si no, una descargada |
| `STRATA_BUILD_IMAGE` | `strata:latest` | Tag que *Pull repo & build latest* construye |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | Volumen Docker nombrado montado en `/data` en cada container Strata (archivos de modelos, packs, capa MTP, setup por modelo; 70-120 GB por modelo). Lo crea Docker en el primer uso |
| `HOST_STRATA_DATA_DIR` | *(sin definir)* | Ruta de host para montar en `/data` en lugar del volumen nombrado. Gana sobre `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | Segundos que una petición (o relanzamiento) espera a que una instancia Strata esté lista. Alto porque un primer arranque descarga y prepara el modelo antes de que el puerto se abra |
| `STRATA_REPO` | `Niko1221/Strata` | Repositorio GitHub (owner/name) que *Pull repo & build latest* descarga |
| `STRATA_REPO_REF` | `main` | Rama, tag o commit de `STRATA_REPO` a construir |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | Dónde se guarda el repositorio descargado (el contexto de build). Su commit se registra a su lado en `Strata.source.json` |
| `STRATA_WEB_HOSTS` | *(sin definir)* | Nombres de host / IPs adicionales (separados por comas) por los que abres la propia app web de una instancia Strata (`http://<host>:<instance port>/`). Sin una clave API, Strata rechaza peticiones de chat de páginas en otros nombres; el host de `CLUSTER_ADVERTISE_URL` siempre está permitido. Se aplica a las instancias iniciadas después |

### Clustering

Opcional. `LLAMAMAN_NODE_NAME` (bajo Core) hace también de identidad de cluster y es requerida para todas las instalaciones.

| Variable | Por defecto | Descripción |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Define truthy para unir este node a un cluster. Requiere `CLUSTER_SECRET`; se ignora con un aviso si el secreto está vacío |
| `CLUSTER_SECRET` | *(sin definir)* | Token bearer compartido (`X-Cluster-Secret`) para todo el HTTP node-a-node. Idéntico en cada node. Usa un valor aleatorio largo sobre una red de confianza o detrás de TLS |
| `CLUSTER_ADVERTISE_URL` | *(sin definir)* | Cómo los peers alcanzan la UI/API de **este** node - nombre de host/IP enrutable desde los otros hosts (no `localhost`). Sin él, el node es de solo visualización |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Segundos desde el último heartbeat antes de que un node se muestre sin conexión. Súbelo si los nodes fluctúan bajo carga o por desvío de reloj |
| `LLAMAMAN_PERF_LOG` | *(off)* | Diagnóstico. Define truthy para registrar tiempos por fase (`perf <name> <ms>`) de rutas sospechosas de lentas: stop de instancia remota, reenvíos del proxy de cluster, sondas de alcanzabilidad, snapshot de heartbeat, comprobaciones de auth, `save_state`, estadísticas de container. Desactivado por defecto; sobrecarga insignificante cuando está apagado |
