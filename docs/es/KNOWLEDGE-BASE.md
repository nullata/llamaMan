<a id="knowledge-base--mcp"></a>
# Base de conocimiento y MCP

Almacena documentos agrupados por **topic**, y llamaman los trocea (~1200 caracteres, alineados al párrafo, 200 de solape), genera sus embeddings y guarda los vectores en tu base de datos MariaDB (columnas `VECTOR` + un índice ANN coseno). Cualquier cliente MCP (Claude Desktop, Claude Code, agentes) puede entonces buscarlos por significado. Todo se ejecuta en el hardware que ya tienes: los embeddings vienen de una instancia llama.cpp que lanzaste con **Embedding model** activado, los vectores viven en tu base de datos, y llamaman sirve el endpoint MCP él mismo.

**Requisitos:** el backend MariaDB (`DATABASE_URL`, ver [Backends de almacenamiento](CONFIGURATION.md#storage-backends)) en servidor **11.8+**, que es donde aterriza `VECTOR`. En el backend JSON o un MariaDB más antiguo, la pestaña MCP dice por qué la base de conocimiento no es compatible y se rechaza habilitarla; nada más cambia. No hay migración: las tablas `kb_*` se crean en el primer uso.

**Configuración:**

1. Lanza una instancia con **Embedding model** activado. Cualquier GGUF de embedding funciona (`bge-m3`, `nomic-embed-text`, …).
2. **Settings → MCP** → activa **Knowledge base** y elige el modelo en **Embedding model**. El desplegable lista todos los modelos con embedding activado en todo el cluster.
3. Añade documentos con la herramienta MCP `kb_ingest_document` (activa **Allow ingest over MCP**) o `POST /api/kb/documents` desde una sesión autenticada. La pestaña lista, filtra y elimina topics y tiene una búsqueda de prueba, pero aún no tiene formulario de alta.
4. Activa **Expose over MCP**. La pestaña muestra la URL del endpoint (`http://<your-host>:42069/mcp/knowledge`), y **Copy config.json** copia una configuración de cliente lista para pegar:

```json
{
  "mcpServers": {
    "llamaman-kb": {
      "type": "http",
      "url": "http://my-nas:42069/mcp/knowledge",
      "headers": { "Authorization": "Bearer <your-api-key>" }
    }
  }
}
```

El bloque `headers` solo se necesita cuando se exige una clave (ver modos de acceso abajo).

**Herramientas.** `kb_list_topics`, `kb_list_documents`, `kb_search` (pasajes del mejor primero, puntuación 0–1) y `kb_get_document` están siempre disponibles. `kb_ingest_document` (crea el topic si falta; 200 KB por llamada; reenviar contenido idéntico es un no-op) aparece solo con **Allow ingest over MCP** activado. `kb_delete_document` / `kb_delete_topic` aparecen solo con **Allow delete over MCP** activado. Eliminar un topic elimina sus documentos, y los borrados no se pueden deshacer.

**Modos de acceso** (**MCP access**):

- **Global** (por defecto): cada cliente trabaja sobre el **pool compartido**, viéndolo y, si ingest/delete están permitidos, editándolo. Los topics nuevos van al pool compartido. La autenticación sigue a **Require Authentication** (pestaña API Keys): activado significa que se exige una clave API válida, desactivado significa que el endpoint está abierto.
- **Per API key**: cada petición necesita una clave API válida, incluso con Require Authentication desactivado. Los topics que una clave crea son **privados** para esa clave, y dos claves pueden cada una tener un topic con el mismo nombre. Cada clave también ve el pool compartido **solo lectura**: puede buscarlo y leerlo, pero no ingerir en él ni borrar de él. Los topics y documentos de otras claves devuelven "not found".

Cambiar de modo nunca mueve ni borra datos, y los topics privados siguen privados en ambos modos. Ir de Global → Per API key convierte todo lo existente en el pool compartido. Ir de Per API key → Global oculta los topics privados a todos los clientes hasta que vuelvas. La pestaña MCP siempre muestra todo, etiquetando cada topic como `shared` o con el nombre de la clave propietaria. `POST /api/kb/topics` también puede crear un topic privado para una clave dada (`owner_key_id`), lo cual es útil cuando la ingesta por MCP está desactivada.

**Modelo de embedding.** La selección es por nombre de modelo, no por instancia. Cada llamada de embed se ejecuta en la primera de:
1. una instancia healthy en este node;
2. una dormida o detenida en este node, que se despierta (o una que ya está arrancando, a la que espera);
3. un node peer que tenga tanto **Knowledge base** como **Expose over MCP** activados y tenga el modelo (una dormida allí se despierta);
4. un relanzamiento en el node que la sirvió por última vez, como máximo 3 intentos cada 10 minutos.

El primer embed registra la dimensión del vector. Un modelo con dimensión distinta se rechaza hasta que pulses **Re-embed everything**, un trabajo en segundo plano que reconstruye los trozos de cada documento (un trabajo a la vez, protegido por un lock de base de datos a nivel de cluster). Solo se comprueba la dimensión, así que lanza tú mismo el re-embedding cuando cambies entre dos modelos con la misma dimensión.

**Notas de cluster.** Los topics y documentos son datos a nivel de cluster en la base de datos compartida, no por node. Cada node ve la misma base de conocimiento, y el endpoint MCP de cualquier node la sirve. Ver [Clustering](CLUSTERING.md).

**Cuando está desactivado**, `/mcp/knowledge` devuelve 404 en ambos modos de acceso, y ninguna petición toca las tablas de la base de conocimiento.

> **Seguridad:** `/api/kb/*` es solo para admin: se aceptan sesiones de UI autenticadas y peers del cluster, las peticiones autenticadas con una clave API reciben 403. Las claves API no están limitadas a la base de conocimiento, eso sí: una clave que des a un cliente MCP también puede llamar a los endpoints de inferencia y a la mayoría de las demás rutas `/api/*`. Usa una clave dedicada por cliente y revócala en caso de duda.
