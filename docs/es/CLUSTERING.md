# Clustering

> **Nota:** esta es una traducción y puede ir por detrás de la [versión en inglés](../en/CLUSTERING.md), que es la de referencia.

*Opcional, desactivado por defecto - las instalaciones de un solo node no se ven afectadas en absoluto.*

Varios despliegues de llamaMan actúan como **un único cluster lógico**: un solo dashboard que agrega las GPUs, instancias y descargas de cada node, con lanzamientos/pulls/descargas entre nodes y balanceo de carga de cola compartida multi-node. Los nodes se descubren entre sí a través del backend de almacenamiento compartido - sin intercambio de claves por pares.

**Requisitos:**

- **Backend de almacenamiento compartido** - cada node apunta a la **misma** `DATABASE_URL` (MariaDB/MySQL, ver [Backends de almacenamiento](CONFIGURATION.md#storage-backends)). JSON es por host y no se puede compartir.
- **`LLAMAMAN_NODE_NAME` único por node** - la identidad de cluster de cada node (y requerido para toda instalación de todos modos).
- **El mismo `CLUSTER_SECRET` en cada node** - el token bearer (enviado como `X-Cluster-Secret`) para todo el HTTP node-a-node.
- **`CLUSTER_ADVERTISE_URL` por node** para *acciones* entre nodes - un nombre de host/IP enrutable desde los **otros** hosts (no `localhost`), p. ej. `http://srv1:5000`. Un node sin uno aparece en el dashboard pero es de solo visualización.

Define en **cada** node (solo `LLAMAMAN_NODE_NAME` y `CLUSTER_ADVERTISE_URL` difieren):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Cada node emite un heartbeat cada ~5 s; un node en silencio más allá de `CLUSTER_NODE_ONLINE_WINDOW_S` (por defecto 45 s) se muestra sin conexión. Inspecciona y gestiona en **Settings → Cluster**.

Algunos ajustes están delimitados por node porque son específicos del host: las **imágenes Docker** seguidas, los interruptores de desalojo del tope de modelos y los [ajustes de despertar bajo carga](#waking-under-load). Todo lo demás se comparte a nivel de cluster. Los alias de grupos de cola compartida activos se anuncian como modelos seleccionables en `/api/tags` y `/v1/models` (deduplicados a nivel de cluster), así que un cliente puede enviar el alias y que lo enruten al node con menos carga que lo sirva.

**La longitud de contexto del grupo es a nivel de cluster.** `/api/show` y `/v1/models` reportan el contexto de un grupo de cola como el **mínimo entre todos los miembros activos en nodes que pueden recibir tráfico**, sea cual sea el node al que pregunte el cliente. Los miembros en nodes desconectados o de solo visualización (view-only) no cuentan. Un grupo con miembros en valores de ctx distintos anuncia el valor menor para que un prompt dimensionado a él sea seguro para cualquier destino de despacho — el margen extra del miembro más grande simplemente queda sin usar. Un node sin miembro local del grupo sigue respondiendo (`context_length` rellenado desde el ctx runtime del peer en el estado de instancia compartido, o desde la fila del preset en la base de datos compartida); un grupo sin miembros activos en ninguna parte sigue devolviendo 404 / sin entrada. La tarjeta de instancia en la UI de operador añade `ctx <min>, capped by <node/model>` cuando el ctx de este miembro excede el mínimo anunciado, para que puedas ver qué peer está arrastrando al grupo hacia abajo.

> **Seguridad:** el secreto de cluster permite que cualquier peer ejecute acciones en este node. Ejecuta el tráfico node-a-node sobre una red de confianza o detrás de TLS.

Variables de entorno relacionadas con el cluster: [Clustering variables](CONFIGURATION.md#clustering).

<a id="queue-groups"></a>
## Grupos de cola

Un grupo de cola agrupa instancias bajo un **Nombre de grupo de cola (Queue group name)** para que un cliente pueda enviar ese nombre a cualquier node. La tabla muestra lo que le ocurre a cada miembro desde el punto de vista de otro node:

| Estado del miembro | Listado en `/api/tags` y `/v1/models` | Recibe tráfico enrutado |
|---|---|---|
| Healthy | Sí | Primera opción |
| Iniciando (cargando) | Sí | Después de los miembros healthy |
| Dormido (Sleeping) | Sí | Al final, y solo después de cada miembro activo, incluidos los fallbacks. La petición lo despierta. |
| Deteniéndose o detenido | No | Nunca |
| En un node sin `CLUSTER_ADVERTISE_URL` (solo visualización) | No | Nunca |
| En un node desconectado (sin heartbeat por 45s) | No | Nunca |

**`CLUSTER_ADVERTISE_URL` es obligatorio para recibir trabajo enrutado.** Sin él, un node puede seguir enviando peticiones a sus peers, pero ninguna puede llegar a él.

**Dormido y detenido significan cosas distintas aquí.** Un idle timeout pone una instancia a dormir, y un miembro dormido se despierta cuando el tráfico lo necesita. Un miembro detenido permanece detenido: una parada por operador, un desalojo y una caída terminan ahí, y ninguna petición de otro node lo reinicia. Preguntar al node que lo contiene devuelve un `503`, a menos que el nombre coincida exactamente con el archivo propio de ese miembro, lo que aún lo despierta.

<a id="waking-under-load"></a>
### Despertar bajo carga

Los miembros activos encolan peticiones en lugar de rechazarlas, por lo que el tráfico por sí solo solo llegaría a un dormido después de que se desbordara una cola entera. Por tanto, llamaMan despierta a uno anticipadamente. Cuando todos los miembros saludables del grupo están llenos, hay peticiones en cola y no hay nada cargándose actualmente, durante 3 sondeos consecutivos (unos 15 segundos), un miembro dormido se despierta. En todo el cluster, solo el node con el nombre más bajo que contenga un miembro dormido hace esto, y como máximo un miembro por grupo por sondeo. Un miembro principal (primary) se despierta antes que un fallback.

Solo cuenta un miembro dormido que aportaría capacidad: necesita **Max Concurrent** por encima de `0`, y no puede compartir cola con un miembro activo en el mismo node.

El despertar se omite, con una línea de log, si los pesos del modelo no caben en la VRAM libre del node. La comprobación suma la memoria libre de todas las GPUs y cuenta únicamente los pesos, por lo que un modelo que pasa la comprobación aún puede fallar al cargarse. Si no se puede leer la VRAM, o el modelo no tiene archivo que medir (los modelos virtuales de Strata), el despertar sigue adelante.

Ambos ajustes son por node, a través de `POST /api/settings`:

| Ajuste | Por defecto | Significado |
|---|---|---|
| `eager_wake_enabled` | `true` | Si este node despierta a sus miembros dormidos bajo carga |
| `eager_wake_polls` | `3` | Sondeos saturados consecutivos (5 segundos cada uno) antes de despertar |

<a id="when-a-group-cant-be-served"></a>
### Cuando un grupo no se puede servir

Si otros nodes alojan el grupo pero ninguno puede aceptar una petición, el cliente recibe un error nombrando a cada node y el motivo, en lugar de "model not found":

| Estado | Código (rutas de OpenAI) | Significado |
|---|---|---|
| `429` | `group_full` | La cola de al menos un peer está llena. Reintentar puede funcionar. |
| `503` | `group_unavailable` | Los peers están desconectados, en modo solo lectura, inalcanzables o no están listos. |

El node propietario devuelve estos códigos `503` para sus propios miembros, y un node de entrada prueba con su siguiente miembro cuando recibe uno:

| Código | Significado |
|---|---|
| `model_stopping` | El miembro se está apagando |
| `group_stopped` | Todos los miembros en este node están detenidos |
| `wake_failed` | El despertar falló |
| `wake_timeout` | El miembro no llegó al estado healthy a tiempo |

Un nombre de grupo no lanza un archivo local cuyo nombre simplemente lo contenga. Un archivo con exactamente ese nombre todavía se ejecuta.

<a id="troubleshooting-a-group-that-isnt-visible"></a>
### Solución de problemas con un grupo que no es visible

- **El backend JSON necesita un único `DATA_DIR` compartido.** Los nodes se descubren entre sí a través del registro, por lo que directorios separados hacen que cada node no vea peers. Usa MariaDB a menos que el directorio esté realmente compartido.
- **Una base de datos degradada oculta los peers a propósito.** Con el espejo de DB sirviendo lecturas, un node solo se ve a sí mismo y sirve localmente.
- **Un sondeo de carga fallido no oculta un grupo.** Un peer que no responde sigue siendo enrutable a partir de su snapshot (de hasta 45 segundos de antigüedad), pero se clasifica detrás de cada peer que sí respondió. El log registra cuándo empiezan a fallar los sondeos a un peer y cuándo se recuperan.
