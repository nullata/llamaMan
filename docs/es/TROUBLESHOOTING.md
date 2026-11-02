# Solución de problemas

> **Nota:** esta es una traducción y puede ir por detrás de la [versión en inglés](../en/TROUBLESHOOTING.md), que es la de referencia.

La tabla completa síntoma → corrección. El [README](../../README.md#troubleshooting) conserva las nueve filas más comunes del primer arranque; todo lo demás está aquí.

| Síntoma | Corrección |
|---|---|
| Instancia atascada en **starting** | Revisa los registros con el botón Logs. Comunes: OOM, errata en la ruta del modelo, GGUF corrupto, imagen no descargada |
| _"Docker image not found"_ | Descarga la imagen correspondiente: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (o `server-rocm` / `server-sycl` / `server`) |
| _"Docker API error"_ al lanzar | Asegúrate de que `/var/run/docker.sock` está montado (por defecto en `docker-compose.yml`) |
| Sin GPU / error CUDA | Asegúrate de que NVIDIA Container Toolkit está instalado y que `docker run --gpus all` funciona en el host |
| Sin GPU / error ROCm | Asegúrate de que `/dev/kfd` y `/dev/dri` existen y tu usuario está en `video`/`render` |
| Sin GPU / error Intel Arc | Asegúrate de que `/dev/dri` es accesible y tu usuario está en `video`/`render` |
| La imagen Vulkan falla con _"Unable to find group render"_ | Define `GPU_TYPE=vulkan` para que llamaMan adjunte los GIDs render/video del host numéricamente (el group_add por nombre falla en imágenes cuyo /etc/group no incluye una entrada `render`) |
| Estadísticas de GPU no disponibles | NVIDIA: descomenta el bloque `deploy.resources.reservations`. AMD/Intel: asegúrate de que `/sys/class/drm:ro` está montado |
| Fabricante de GPU detectado incorrectamente | Define `GPU_TYPE=cuda`/`rocm`/`intel` para anular |
| Instancia atascada en **starting** ejecutándose en bare-metal | El container está healthy pero llamaman no puede alcanzarlo. Define `LLAMAMAN_IN_DOCKER=false`/`true` explícitamente si la autodetección es errónea para tu entorno |
| Strata: _"Docker image 'strata:latest' not found"_ | Descarga o construye una imagen Strata en Settings → Docker Images (o `docker build -t strata .` en un clon del repositorio Strata). Ver [Strata](STRATA.md) |
| Modelos Strata no listados | Define `STRATA_ENABLED=true`; el node debe ser NVIDIA (`GPU_TYPE=cuda` si falla la detección). `GET /api/engines` muestra la razón |
| Instancia Strata atascada en **starting** | Normal durante minutos en un primer arranque (descarga + pack). La tarjeta muestra la etapa; el registro muestra el setup de Strata. "setup failed" significa que su setup se detuvo - lee la línea `[X]` del registro |
| La tarjeta Strata indica _"this Strata image can't run ... yet: update it"_ | El motor de la imagen no cuenta con esa familia de modelos. Haz pull o construye una imagen Strata más reciente en Settings → Docker Images |
| El modal de Stats está vacío | Activa **Settings → App Settings → Request recording** (ver [Registro de peticiones y estadísticas](CONFIGURATION.md#request-recording--stats)) |
| El lanzamiento falla con error GPU/CDI en un host sin passthrough de GPU | Define **GPU Layers** a `0` para solo CPU sin dispositivo GPU adjunto, o arregla el runtime de GPU |
| Conflicto de puerto | El formulario sugiere automáticamente un puerto libre; ajusta si es necesario |
| El modelo no aparece en OpenWebUI | Asegúrate de `OLLAMA_BASE_URL=http://llamaman:42069`. Verifica que `/api/tags` devuelve modelos |
| OpenWebUI recibe 401 | `require_auth` está activado (por defecto). Crea una clave API y define `OPENAI_API_KEYS` en OpenWebUI |
| Un cliente OpenAI recibe 503 cuando los modelos están llenos | Alcanzado el tope de `LLAMAMAN_MAX_MODELS` y la API OpenAI no puede desalojar por defecto. Sube el tope o activa los interruptores de desalojo: [Desalojo de modelos](MODELS.md#model-eviction) |
| Containers sin limpiar tras el stop | llamaMan elimina containers al detener. Huérfanos tras un fallo: `docker ps --filter name=llamaman-`, o reinicia llamaMan (la adopción de huérfanos se ejecuta al arrancar) |
| Un cliente (Hermes / OpenWebUI / etc.) reporta la ventana de contexto entrenada en lugar del tope del preset | Actualiza a 1.1.2+. `/api/ps` ahora incluye `context_length` con el ctx runtime con el que se lanzó la instancia, y `model_info["<arch>.context_length"]` de `/api/show` se anula con el tope efectivo |
| Un cliente que pregunta por un nombre de queue-group recibe su ctx predeterminado (p. ej. 128k) en lugar del ctx real del despliegue | `/api/show` y `/v1/models` ahora responden por un queue group en cualquier node clusterizado — incluso uno sin miembro local — usando el ctx runtime del peer del estado compartido. Los grupos multi-miembro reportan el mínimo entre los miembros activos para que el valor anunciado sea seguro para cualquier destino de despacho. Ver [Clustering](CLUSTERING.md) |
| La petición al grupo de cola devuelve `group_full` o `group_unavailable` | Ningún node pudo aceptarla, y el mensaje nombra a cada node y el motivo. `group_full` puede funcionar con un reintento. Para el resto, revisa los nodes nombrados: [Cuando un grupo no se puede servir](CLUSTERING.md#when-a-group-cant-be-served) |
| Falta el nombre del grupo de cola en `/api/tags` y `/v1/models` | Los miembros detenidos y los miembros en nodes desconectados o de solo visualización no se listan. Comprueba que el miembro esté en ejecución y verifica `CLUSTER_ADVERTISE_URL` en su node: [Grupos de cola](CLUSTERING.md#queue-groups) |
| Un miembro dormido del grupo de cola arranca sin que le llegue ninguna petición | Comportamiento esperado cuando su grupo permanece saturado: [Despertar bajo carga](CLUSTERING.md#waking-under-load). Establece `eager_wake_enabled` a `false` en ese node para desactivarlo |
