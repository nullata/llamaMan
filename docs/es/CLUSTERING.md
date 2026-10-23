# Clustering

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

Algunos ajustes están delimitados por node porque son específicos del host: las **imágenes Docker** seguidas y los interruptores de desalojo del tope de modelos. Todo lo demás se comparte a nivel de cluster. Los alias de grupos de cola compartida activos se anuncian como modelos seleccionables en `/api/tags` y `/v1/models` (deduplicados a nivel de cluster), así que un cliente puede enviar el alias y que lo enruten al node con menos carga que lo sirva.

**La longitud de contexto del grupo es a nivel de cluster.** `/api/show` y `/v1/models` reportan el contexto de un grupo de cola como el **mínimo entre todos los miembros activos**, sea cual sea el node al que pregunte el cliente. Un grupo con miembros en valores de ctx distintos anuncia el valor menor para que un prompt dimensionado a él sea seguro para cualquier destino de despacho — el margen extra del miembro más grande simplemente queda sin usar. Un node sin miembro local del grupo sigue respondiendo (`context_length` rellenado desde el ctx runtime del peer en el estado de instancia compartido, o desde la fila del preset en la base de datos compartida); un grupo sin miembros activos en ninguna parte sigue devolviendo 404 / sin entrada. La tarjeta de instancia en la UI de operador añade `ctx <min>, capped by <node/model>` cuando el ctx de este miembro excede el mínimo anunciado, para que puedas ver qué peer está arrastrando al grupo hacia abajo.

> **Seguridad:** el secreto de cluster permite que cualquier peer ejecute acciones en este node. Ejecuta el tráfico node-a-node sobre una red de confianza o detrás de TLS.

Variables de entorno relacionadas con el cluster: [Clustering variables](CONFIGURATION.md#clustering).
