# API REST

Todos los endpoints devuelven / aceptan JSON. Los endpoints de gestión necesitan una cookie de sesión (login en navegador) o `Authorization: Bearer <key>`. Los endpoints de servicio de modelos también exigen un bearer cuando `require_auth` está activado (por defecto). Detalles de autenticación: [Authentication](CONFIGURATION.md#authentication-detail).

## Contenido

- [Autenticación y claves](#auth--keys) · [Instancias](#instances) · [Descargas, modelos, presets](#downloads-models-presets) · [Ajustes, sistema, registro de peticiones](#settings-system-request-log) · [Compatibilidad con Ollama y OpenAI](#ollama--openai-compat)

<a id="auth--keys"></a>
## Autenticación y claves

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` / `POST` | `/login` | Página de login / autenticarse |
| `GET` / `POST` | `/setup` | Configuración del primer arranque |
| `GET` | `/logout` | Finalizar sesión |
| `GET` / `POST` | `/api/api-keys` | Listar / crear claves API |
| `DELETE` | `/api/api-keys/<id>` | Revocar |

<a id="instances"></a>
## Instancias

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Listar / lanzar |
| `GET` / `DELETE` | `/api/instances/<id>` | Obtener / detener y eliminar |
| `POST` | `/api/instances/<id>/restart` | Reiniciar una instancia detenida/dormida |
| `DELETE` | `/api/instances/<id>/remove` | Eliminar un registro de instancia detenida |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Cola / stream SSE de registros |
| `GET` | `/api/next-port` | Obtener el siguiente puerto disponible del pool |
| `POST` | `/api/preview-command` | Renderiza el comando `llama-server ...` equivalente para un cuerpo de lanzamiento (usado por la herramienta "Copy llama.cpp command" de la UI; no lanza nada) |

**Cuerpo de lanzamiento** (`POST /api/instances`): los campos del formulario de lanzamiento, serializados como JSON. Ver la [referencia de ajustes de lanzamiento](LAUNCH-SETTINGS.md#launch-settings-reference) para cada campo y su valor por defecto; los campos de valor enum (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) solo aceptan los valores listados allí y por defecto usan el valor por defecto de la referencia. Los campos numéricos dejados como `null` u omitidos usan el valor por defecto de la referencia; los campos desconocidos se ignoran.

Rarezas de frontera que conviene fijar:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp estima la VRAM), `-2` = **all** (fuerza todas las capas), `0` = solo CPU (sin GPU adjunta al container), N positivo = exactamente N capas.
- `n_cpu_moe_layers`: `0` (por defecto) omite ambas flags MoE. `-1` emite `--cpu-moe` (todos los expertos enrutados en CPU). N positivo emite `--n-cpu-moe N` (expertos de las primeras N capas en CPU). Inerte en modelos densos.
- `threads`: cuando se define, aplica `--threads N` a llama-server **y** fija la cuota CPU del container a N núcleos.
- `threads_batch`: null / omitido → la flag no se emite y llama-server se repliega a `--threads`.
- `flash_attn`: los valores heredados `true` / `false` de configs pre-tri-estado se pliegan al leer (`true` → `"on"`, `false` → `"off"`), así que no hace falta migración de almacenamiento.
- Los valores cuantizados de `cache_type_v` requieren `flash_attn: "on"` - `"auto"` no es una garantía; de lo contrario llama-server se niega a arrancar.
- `spec_draft_model` es obligatorio cuando `spec_enabled: true` y `spec_type` es cualquier cosa distinta de `"draft-mtp"` - una petición sin él se rechaza con 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / omitido omite la flag. `0` es un valor real distinto de null y se transmite.

<a id="downloads-models-presets"></a>
## Descargas, modelos, presets

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Listar / iniciar |
| `GET` / `DELETE` | `/api/downloads/<id>` | Obtener / cancelar |
| `DELETE` | `/api/downloads/<id>/remove` | Eliminar una entrada completada/fallida |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Cola / stream SSE |
| `GET` | `/api/models` | Modelos descubiertos (incluye `repo_id` cuando se conoce; los modelos archivados se marcan `archived` con `restore_path`) |
| `POST` | `/api/models/delete` | Eliminar del disco (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Leer el recuento de capas de los metadatos GGUF |
| `GET` | `/api/disk-space` | Espacio libre/usado en el volumen de modelos |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Obtener / guardar / eliminar un preset (`GET /api/presets` lista todos) |
| `GET` | `/api/engines` | Motores de inferencia de este node: disponibilidad (con razón), capacidades, campos de lanzamiento, catálogo de modelos de Strata |
| `GET` | `/api/archive` | Estado del archivo (`enabled`, `available`, `reason`, `free_bytes`) y trabajos de movimiento |
| `POST` | `/api/archive` | Archivar un modelo: `{"path": "/models/..."}` (mueve su carpeta entera). 409 cuando está en uso |
| `POST` | `/api/archive/restore` | Restaurar un modelo archivado: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Cancelar un movimiento en cola/en ejecución (se conserva la fuente), o eliminar uno terminado |
| `POST` | `/api/engines/strata/download` | Pre-descargar los shards de un modelo Strata (`{"model": "strata/qwen-IQ2_XS"}`, opcional `hf_token_id` / `speed_limit_mbps`) |

Cuerpo de descarga: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. `filename` vacío descarga el repositorio completo.

Las instancias y presets aceptan un `engine` opcional (`llamacpp` por defecto, o `strata`; inferido de una ruta de modelo `/strata/...`) más, para Strata, `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) y `strata_low_ram` (`auto`/`on`/`off`). Las rutas de modelos Strata son virtuales: `/strata/<family>-<SIZE>`, p. ej. `PUT /api/presets/strata/qwen-IQ2_XS`. Ver [Strata](STRATA.md).

<a id="settings-system-request-log"></a>
## Ajustes, sistema, registro de peticiones

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Obtener / guardar ajustes globales |
| `GET` | `/api/system-info` | Uso de CPU, número de núcleos, uso de RAM |
| `GET` | `/api/gpu-info` | VRAM y utilización por GPU (nativo; se repliega a exec en container) |
| `GET` | `/health` | `{"status": "ok"}` - siempre abierto, sin autenticación |
| `GET` | `/api/request-log/conversations` | Conversaciones recientes con metadatos agregados (`limit` en query, por defecto 100, máximo 500) |
| `GET` | `/api/request-log/conversations/<id>` | Todos los turnos registrados de una conversación, del más antiguo primero |
| `GET` | `/api/request-log/stats` | Métricas agregadas (tokens, tokens/s medio/pico, TTFT, latencia, errores, conteos en streaming). Opcional `inst_id` + `window_hours` |

<a id="ollama--openai-compat"></a>
## Compatibilidad con Ollama y OpenAI

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate auto-arrancan).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat auto-arranca).
Modelos de decisión: `/v1/systemone` (auto-arranca) - ver [Modelos de decisión](MODELS.md#decision-models).
Base de conocimiento: `/api/kb/*` (solo admin) y el endpoint MCP `/mcp/knowledge` - ver [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Los endpoints node-a-node del cluster usan `X-Cluster-Secret` - ver [Clustering](CLUSTERING.md).
