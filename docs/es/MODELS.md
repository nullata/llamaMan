# Modelos, descargas y desalojo

Todo sobre la biblioteca de modelos: añadir modelos, mantenerlos al día, cómo el proxy los elige y desaloja, y los ajustes de mantenimiento.

## Contenido

- [Modelos](#models) · [Modelos de decisión](#decision-models) · [Desalojo de modelos](#model-eviction) · [Ajustes de descarga](#download-settings) · [Gestión de imágenes Docker](#docker-image-management) · [Archivo de modelos](#model-archive) · [Copia de seguridad y restauración de modelos](#model-backup-and-restore) · [Ajustes de limpieza](#cleanup-settings)

<a id="models"></a>
## Modelos

Coloca modelos bajo el volumen `/models` como archivos `.gguf`, directorios de repositorio de HuggingFace (conteniendo `config.json`), o usa el botón **Download** de la UI.

<a id="display-name"></a>
### Nombre visible

Los modelos se listan por el nombre base del archivo GGUF por defecto. Cada modelo puede recibir un **Display Name** amigable (en el formulario Launch). Cuando está definido, es el id que los clientes de API ven en `/api/tags` y `/v1/models`, así que OpenWebUI muestra `Qwen 2.5 14B` en lugar de `Qwen2.5-14B-Instruct-Q4_K_M`. Debe ser único y no chocar con el nombre de archivo de otro modelo ni con un nombre de queue-group del cluster.

<a id="model-updates"></a>
### Actualizaciones de modelos

El botón **Check for updates** (mostrado para modelos con un repo de origen conocido) compara el archivo local contra el hash de contenido publicado del repo:

- **Up to date** - el hash registrado coincide
- **Update available** - vuelve a descargar a una carpeta de staging e intercambia atómicamente solo cuando la descarga se completa (un pull fallado deja el modelo actual intacto). Rechaza con 409 mientras el modelo está cargado.
- **Verify hash** - hashea el archivo local (trabajo en segundo plano con progreso) cuando aún no se ha registrado ningún hash

Un escaneo en segundo plano opt-in (**Settings → Download Settings**, ver [Ajustes de descarga](#download-settings)) mantiene estas respuestas listas de antemano.

<a id="decision-models"></a>
## Modelos de decisión

Los modelos de decisión responden preguntas tipadas sobre una entrada con probabilidades en lugar de generar texto: la API System One introducida por Jev de TypeSafe, servida por llama.cpp (imágenes construidas después de 2026-10-05) en `POST /v1/systemone`. GGUF abiertos: Julia-1, Laya, Kev-4B, lev, Clef-Flash, Clef, OpenJev (OpenJev y Clef también aceptan imágenes, con su `--mmproj`). llama-server los reconoce desde `<arch>.decision.type` en el GGUF; no se necesita ninguna flag.

- **Detección.** Elegir uno en el formulario de lanzamiento lo indica bajo Model Path. Sin un preset, el formulario se rellena para él: **Exclude from Max Models** activado, **Web UI** desactivado (su página de chat no puede manejar el modelo), **Parallel Slots** `8` (cada pregunta de una petición toma un slot, y comparten la entrada), **Embedding Model** desactivado (llama.cpp activa el modo embedding él mismo donde el modelo lo necesita; forzarlo no hace falta). Un modelo encoder (estilo BERT) también tiene Context, Batch y Micro-batch Size fijados a su longitud de contexto: toda su entrada debe caber en un micro-batch. **Clef** (y Clef-Flash) hace todas las preguntas en un prompt y las lee de una pasada, así que obtiene **Parallel Slots** `1` y Batch = Micro-batch = Context (16384, su valor entrenado por defecto, cuando Context está en blanco): toda la petición (entrada, cada pregunta y opción) debe caber en un micro-batch, o llama-server responde "increase the physical batch size".
- **Enrutamiento.** `POST :42069/v1/systemone` elige el modelo por `"model"` como `/v1/chat/completions` (auto-arranque, reglas de desalojo, reenvío de cluster, cola). La petición se reenvía sin cambios (sin overrides de muestreo). En el puerto propio de la instancia pasa por la cola del proxy por instancia cuando ese proxy corre.
- **Los endpoints de chat los rechazan** (HTTP 422) - no generan texto.
- **Request log:** las respuestas, y `input_tokens` como los tokens del prompt (`output_tokens` es siempre 0).
- **Errores:** una pregunta cuyas opciones no caben en un batch es rechazada por llama-server ("increase the physical batch size") - sube Batch Size / Micro-batch Size.

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

`LLAMAMAN_MAX_MODELS` limita los modelos de **chat** concurrentes vía el proxy. Cuando el tope está lleno y se pide un modelo de chat nuevo, se desaloja el modelo de chat LRU:

| Lanzador | Desaloja | No puede desalojar |
|---|---|---|
| **Admin UI** | Gestionados por Ollama primero (LRU), luego lanzados por admin | - |
| **Ollama API** | Gestionados por Ollama (LRU) | Lanzados por admin *(por defecto)* |
| **OpenAI API** | Nada *(por defecto; devuelve 503 cuando está lleno)* | Todo *(por defecto)* |

Tres interruptores bajo **Settings → App Settings** relajan estos valores por defecto:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - desalojo LRU silencioso antes del lanzamiento (gestionados por Ollama primero). Desactivado por defecto → la UI pregunta primero.
- **Allow Ollama API to evict admin-launched models** - repliegue cuando no hay modelos gestionados por Ollama disponibles. Desactivado por defecto.
- **Allow OpenAI API to evict admin-launched models** - concede a la API OpenAI el mismo desalojo LRU que Ollama. Desactivado por defecto.

Otros detalles:

- Todas las instancias de chat en ejecución cuentan (tanto admin UI como gestionadas por proxy).
- **Las instancias dormidas siguen contando** (la reserva del slot persiste durante la pausa idle) - pero despertar una instancia dormida para su propio modelo nunca lo bloquea el tope.
- Las instancias con **Exclude from Max Models** activado (formulario de lanzamiento) no cuentan y nunca se desalojan: modelos de embedding, modelos de decisión, cualquier cosa pequeña que deba seguir activa. Activar **Embedding Model** también lo activa, y los presets guardados antes de que existiera el interruptor lo obtienen de su ajuste Embedding Model.
- `LLAMAMAN_MAX_MODELS=0` desactiva el desalojo por completo.

<a id="download-settings"></a>
## Ajustes de descarga

Bajo **Settings → Download Settings**:

- **Global speed limit (Mbps)** (0 = ilimitado) - limita todas las descargas en este node y se aplica a las descargas en curso en el plazo de un segundo. Por node: en un cluster cada node tiene su propio límite (un valor guardado antes de que se volviera por-node se aplica a todos los nodes hasta que ese node guarde el suyo propio)
- **Auto-retry failed downloads** (desactivado por defecto) + **Retry count per failed download** (por defecto 3)
- **Check models for updates in the background** (desactivado por defecto) - worker opt-in que pregunta a cada repo de origen si un archivo ha sido republicado, además de calcular un checksum para cualquier modelo que aún no tenga uno. **Update check interval (hours)** por defecto 24. Hashea como máximo un modelo por pasada; nunca se ejecuta mientras hay una descarga en curso.

<a id="docker-image-management"></a>
## Gestión de imágenes Docker

**Settings → Docker Images**: pull de cualquier imagen llama.cpp por nombre, elimina imágenes locales antiguas (desactivado para la `LLAMA_IMAGE` activa, y devuelve un error si Docker se niega porque un container la está usando), y opcionalmente auto-actualiza las imágenes llama.cpp del node (`LLAMA_IMAGE` y cada imagen descargada allí) según un calendario. La fila de pull tiene un selector de motor: con `STRATA_ENABLED`, una imagen descargada como *Strata* va a *Manage Strata images* (no hay fuente Strata por defecto; escribe el nombre). **Pull repo & build latest** descarga el commit más reciente del repositorio Strata (un tarball de GitHub, en `STRATA_SRC_DIR`) y construye `STRATA_BUILD_IMAGE` con BuildKit a través del socket de Docker. El auto-update de Strata vuelve a hacer pull de las imágenes Strata descargadas y, cuando el repositorio fue descargado, obtiene su commit más reciente y reconstruye cuando cambió (o la imagen falta); nunca descarga el repositorio por su cuenta. **Delete image & repo** elimina la imagen construida (rechazado mientras una instancia en ejecución la usa) y el repositorio descargado, para un node que corre una imagen descargada. Los lanzamientos Strata usan `STRATA_IMAGE` cuando está en el node, si no la imagen construida, si no una descargada. Ver [Strata](STRATA.md).

<a id="model-archive"></a>
## Archivo de modelos

*(opt-in)* Mueve modelos fuera del almacenamiento rápido a un segundo volumen y de vuelta desde la UI, en lugar de trasegar archivos a mano en el host.

**Habilitarlo** montando el volumen de archivo en el container de llamaMan y apuntando `ARCHIVE_DIR` a él:

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

Sin definir, la función está oculta. llamaMan mueve los archivos él mismo, así que no se necesita ninguna ruta `HOST_`.

**Usarlo.**

- Pasa el cursor sobre un modelo en la biblioteca y haz clic en el icono de archivar.
- Los modelos archivados siguen en la lista, atenuados, con una insignia *archived* y un botón **Restore**. No pueden seleccionarse para lanzar y nunca aparecen en `/api/tags` / `/v1/models`.
- Los movimientos se muestran en el panel **Downloads** con progreso, velocidad y tiempo restante, y pueden cancelarse.
- Restore devuelve un modelo a su ruta original, así que presets, favoritos, notas y display names vuelven con él.

**Qué se mueve.** La carpeta de primer nivel entera de un modelo bajo `/models`: la carpeta de una descarga, con cada shard de un GGUF partido. Un `.gguf` ubicado directamente en `/models` se mueve junto con sus hermanos partidos. La carpeta de shards de un modelo Strata (`strata/<tag>`) se mueve sola.

**Seguridad.**

- **Mismo disco:** el movimiento es un rename instantáneo.
- **Entre discos:** llamaMan copia a una carpeta oculta `.llamaman-partial-*` en el destino, fsync, comprueba el tamaño de cada archivo, lo renombra a su sitio, y solo entonces borra la fuente.
- **Fallos:** una cancelación, una comprobación fallida o demasiado poco espacio libre dejan la fuente intacta. Las copias parciales dejadas por un reinicio a mitad de movimiento se eliminan al arrancar.
- **Un movimiento a la vez.**
- **Rechazado** mientras una instancia usa el modelo (incluida una dormida, o como modelo draft / mmproj, o una instancia Strata usando los shards), mientras una descarga escribe en él, o cuando el nombre de destino ya existe. Lanzar o eliminar un modelo que se está moviendo también se rechaza.

**Límites.** Los trabajos de movimiento viven en memoria: tras un reinicio de llamaMan la lista está vacía, y un movimiento interrumpido debe volver a iniciarse (no se pierde nada). En modo cluster cada node archiva a su propio volumen, y los botones se muestran para el node que estás navegando.

<a id="model-backup-and-restore"></a>
## Copia de seguridad y restauración de modelos

**Settings → App Settings**:

- **Download Stored Models JSON** - exporta todos los modelos escaneados + configs de preset a un JSON con timestamp
- **Restore from JSON** - por modelo: ya presente → preset fusionado (se conservan los valores existentes); ausente pero con fuente HuggingFace → en cola para descargar con el preset pre-rellenado; ausente y sin fuente conocida → reportado como no restaurable

<a id="cleanup-settings"></a>
## Ajustes de limpieza

Bajo **Settings → Cleanup Settings** (todo se ejecuta periódicamente en segundo plano; solo elimina/actualiza registros - nunca borra archivos de modelos):

- **Auto-clean completed/failed downloads** (por defecto 24 h) - las descargas activas nunca se tocan
- **Auto-clean stopped instances** (por defecto 24 h) - las instancias en ejecución nunca se eliminan
- **Auto-remove stale instance records** (por defecto intervalo de 5 min) - vuelve a comprobar los registros `starting`/`healthy`/`sleeping` contra Docker; aquellos cuyo container desapareció se marcan como detenidos
