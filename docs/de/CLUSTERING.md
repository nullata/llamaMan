# Clustering

*Optional, standardmäßig deaktiviert – Single-Node-Installationen sind davon völlig unberührt.*

Mehrere llamaMan-Deployments agieren als **ein logischer Cluster**: ein zentrales Dashboard, das GPUs, Instanzen und Downloads aller Nodes aggregiert, mit knotenübergreifenden Starts/Pulls/Downloads und Multi-Node-Lastverteilung über gemeinsame Warteschlangen (Shared Queues). Die Nodes erkennen einander über das gemeinsame Speicher-Backend – kein paarweiser Schlüsselaustausch erforderlich.

**Anforderungen:**

- **Gemeinsames Speicher-Backend** – jeder Node verweist auf **dieselbe** `DATABASE_URL` (MariaDB/MySQL, siehe [Speicher-Backends](CONFIGURATION.md#storage-backends)). JSON ist hostspezifisch und kann nicht geteilt werden.
- **Eindeutiger `LLAMAMAN_NODE_NAME` pro Node** – die Cluster-Identität des jeweiligen Nodes (und ohnehin für jede Installation erforderlich).
- **Derselbe `CLUSTER_SECRET` auf jedem Node** – das Bearer-Token (gesendet als `X-Cluster-Secret`) für die gesamte Node-zu-Node-HTTP-Kommunikation.
- **`CLUSTER_ADVERTISE_URL` pro Node** für knotenübergreifende *Aktionen* – ein Hostname/eine IP, der/die von den **anderen** Hosts aus erreichbar ist (nicht `localhost`), z. B. `http://srv1:5000`. Ein Node ohne diese Angabe erscheint im Dashboard, ist jedoch schreibgeschützt.

Auf **jedem** Node festlegen (nur `LLAMAMAN_NODE_NAME` und `CLUSTER_ADVERTISE_URL` unterscheiden sich):

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Jeder Node sendet alle ~5 Sekunden einen Heartbeat; ein Node, der länger als `CLUSTER_NODE_ONLINE_WINDOW_S` (Standard: 45s) stumm bleibt, wird als offline angezeigt. Einsehen und verwalten unter **Settings → Cluster**.

Einige Einstellungen sind pro Node definiert, da sie hostspezifisch sind: getrackte **Docker-Images** und die Umschalter für die Model-Cap-Eviction. Alles andere wird clusterweit geteilt. Aktive Shared-Queue-Gruppenaliase werden in `/api/tags` und `/v1/models` als auswählbare Modelle bereitgestellt (clusterweit dedupliziert), sodass ein Client den Alias anfragen kann und an den am wenigsten ausgelasteten Node weitergeleitet wird, der ihn bedient.

**Die Gruppen-Kontextlänge gilt clusterweit.** `/api/show` und `/v1/models` melden den Kontext einer Queue-Gruppe als das **Minimum aller aktiven Member**, unabhängig davon, welchen Node der Client anfragt. Eine Gruppe mit Membern unterschiedlicher ctx-Werte kündigt den kleineren Wert an, sodass ein darauf abgestimmter Prompt für jedes Dispatch-Ziel sicher ist – der zusätzliche Puffer auf dem größeren Member bleibt einfach ungenutzt. Ein Node ohne lokalen Member der Gruppe antwortet dennoch (`context_length` wird aus dem Runtime-ctx des Peers im geteilten Instanzstatus oder aus der Preset-Zeile in der geteilten DB befüllt); eine Gruppe ohne aktive Member an irgendeinem Ort liefert weiterhin 404 / keinen Eintrag zurück. Die Instanzkarte in der Betreiber-UI ergänzt `ctx <min>, capped by <node/model>`, wenn der ctx dieses Members das angekündigte Minimum überschreitet, sodass Sie sehen können, welcher Peer die Gruppe limitiert.

> **Sicherheit:** Das Cluster-Secret erlaubt es jedem Peer, Aktionen auf diesem Node auszulösen. Führen Sie den Node-zu-Node-Traffic über ein vertrauenswürdiges Netzwerk oder hinter TLS aus.

Clusterbezogene Umgebungsvariablen: [Cluster-Variablen](CONFIGURATION.md#clustering).
