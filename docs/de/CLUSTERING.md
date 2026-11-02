# Clustering

> **Hinweis:** Dies ist eine Übersetzung und kann hinter der [englischen Fassung](../en/CLUSTERING.md) zurückliegen; maßgeblich ist die englische Fassung.

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

Einige Einstellungen sind pro Node definiert, da sie hostspezifisch sind: getrackte **Docker-Images**, die Umschalter für die Model-Cap-Eviction und die [Einstellungen zum Aufwecken unter Last](#waking-under-load). Alles andere wird clusterweit geteilt. Aktive Shared-Queue-Gruppenaliase werden in `/api/tags` und `/v1/models` als auswählbare Modelle bereitgestellt (clusterweit dedupliziert), sodass ein Client den Alias anfragen kann und an den am wenigsten ausgelasteten Node weitergeleitet wird, der ihn bedient.

**Die Gruppen-Kontextlänge gilt clusterweit.** `/api/show` und `/v1/models` melden den Kontext einer Queue-Gruppe als das **Minimum aller aktiven Member auf Nodes, die Traffic empfangen können**, unabhängig davon, welchen Node der Client anfragt. Member auf Offline- oder View-Only-Nodes zählen nicht. Eine Gruppe mit Membern unterschiedlicher ctx-Werte kündigt den kleineren Wert an, sodass ein darauf abgestimmter Prompt für jedes Dispatch-Ziel sicher ist – der zusätzliche Puffer auf dem größeren Member bleibt einfach ungenutzt. Ein Node ohne lokalen Member der Gruppe antwortet dennoch (`context_length` wird aus dem Runtime-ctx des Peers im geteilten Instanzstatus oder aus der Preset-Zeile in der geteilten DB befüllt); eine Gruppe ohne aktive Member an irgendeinem Ort liefert weiterhin 404 / keinen Eintrag zurück. Die Instanzkarte in der Betreiber-UI ergänzt `ctx <min>, capped by <node/model>`, wenn der ctx dieses Members das angekündigte Minimum überschreitet, sodass Sie sehen können, welcher Peer die Gruppe limitiert.

> **Sicherheit:** Das Cluster-Secret erlaubt es jedem Peer, Aktionen auf diesem Node auszulösen. Führen Sie den Node-zu-Node-Traffic über ein vertrauenswürdiges Netzwerk oder hinter TLS aus.

Clusterbezogene Umgebungsvariablen: [Cluster-Variablen](CONFIGURATION.md#clustering).

<a id="queue-groups"></a>
## Queue-Gruppen

Eine Queue-Gruppe fasst Instanzen unter einem **Warteschlangengruppennamen (Queue group name)** zusammen, sodass ein Client diesen Namen an jeden beliebigen Node senden kann. Die Tabelle zeigt, was mit jedem Mitglied aus der Sicht eines anderen Nodes geschieht:

| Mitgliedsstatus | Gelistet in `/api/tags` und `/v1/models` | Erhält gerouteten Traffic |
|---|---|---|
| Healthy | Ja | Erste Wahl |
| Starting (lädt) | Ja | Nach healthy Mitgliedern |
| Sleeping | Ja | Zuletzt, und erst nach jedem wachen Mitglied, einschließlich Fallbacks. Der Request weckt es auf. |
| Stopping oder gestoppt | Nein | Niemals |
| Auf einem Node ohne `CLUSTER_ADVERTISE_URL` (nur Anzeige) | Nein | Niemals |
| Auf einem Offline-Node (kein Heartbeat für 45s) | Nein | Niemals |

**`CLUSTER_ADVERTISE_URL` ist erforderlich, um geroutete Arbeit zu empfangen.** Ohne diese kann ein Node zwar weiterhin Requests an seine Peers senden, aber keine können ihn erreichen.

**Schlafen und Stoppen bedeuten hier unterschiedliche Dinge.** Ein Idle-Timeout versetzt eine Instanz in den Schlafzustand, und ein schlafendes Mitglied wacht auf, wenn Traffic es erfordert. Ein gestopptes Mitglied bleibt gestoppt: Ein Stoppen durch den Operator, eine Verdrängung (Eviction) und ein Absturz enden alle dort, und kein Request von einem anderen Node startet es neu. Fragt man den Node an, der es hält, erhält man ein `503`, es sei denn, der Name entspricht auch der eigenen Datei dieses Mitglieds, was es weiterhin aufweckt.

<a id="waking-under-load"></a>
### Aufwecken unter Last

Wache Mitglieder stellen Requests eher in die Warteschlange, als sie abzulehnen, sodass der Traffic allein ein schlafendes Mitglied erst erreichen würde, nachdem eine gesamte Queue überläuft. llamaMan weckt daher eines vorzeitig auf. Wenn jedes gesunde Mitglied in der Gruppe voll ist, Requests in der Warteschlange stehen und gerade nichts lädt, wird nach 3 aufeinanderfolgenden Polls (ca. 15 Sekunden) ein schlafendes Mitglied aufgeweckt. Clusterweit führt dies nur der Node mit dem alphabetisch niedrigsten Namen durch, der ein schlafendes Mitglied hält, und maximal ein Mitglied pro Gruppe pro Poll. Ein Primary wacht vor einem Fallback auf.

Nur ein schlafendes Mitglied, das zusätzliche Kapazität bringen würde, zählt: Es benötigt **Max Concurrent** über `0` und darf die Warteschlange nicht mit einem wachen Mitglied auf demselben Node teilen.

Das Aufwecken wird mit einer Logzeile übersprungen, wenn die Gewichte des Modells nicht in den freien VRAM des Nodes passen. Die Prüfung summiert den freien Speicher über alle GPUs und zählt nur die Gewichte, sodass ein Modell, das die Prüfung besteht, beim Laden dennoch scheitern kann. Kann der VRAM nicht gelesen werden oder hat das Modell keine Datei zur Größenmessung (die virtuellen Modelle von Strata), wird das Aufwecken dennoch durchgeführt.

Beide Einstellungen sind pro Node über `POST /api/settings` konfigurierbar:

| Einstellung | Standard | Bedeutung |
|---|---|---|
| `eager_wake_enabled` | `true` | Ob dieser Node seine schlafenden Mitglieder unter Last aufweckt |
| `eager_wake_polls` | `3` | Aufeinanderfolgende gesättigte Polls (je 5 Sekunden) vor dem Aufwecken |

<a id="when-a-group-cant-be-served"></a>
### Wenn eine Gruppe nicht bedient werden kann

Wenn andere Nodes die Gruppe hosten, aber keiner einen Request annehmen kann, erhält der Client einen Fehler, der jeden Node und den Grund benennt, anstelle von "model not found":

| Status | Code (OpenAI-Routen) | Bedeutung |
|---|---|---|
| `429` | `group_full` | Die Queue mindestens eines Peers ist voll. Ein erneuter Versuch kann funktionieren. |
| `503` | `group_unavailable` | Die Peers sind offline, im View-Only-Modus, nicht erreichbar oder nicht bereit. |

Der besitzende Node gibt diese `503`-Codes für seine eigenen Mitglieder zurück, und ein Eingangs-Node versucht sein nächstes Mitglied, sobald er einen solchen erhält:

| Code | Bedeutung |
|---|---|
| `model_stopping` | Das Mitglied wird heruntergefahren |
| `group_stopped` | Jedes Mitglied auf diesem Node ist gestoppt |
| `wake_failed` | Das Aufwecken ist fehlgeschlagen |
| `wake_timeout` | Das Mitglied wurde nicht rechtzeitig healthy |

Ein Gruppenname startet keine lokale Datei, deren Name ihn lediglich enthält. Eine Datei mit exakt diesem Namen wird weiterhin gestartet.

<a id="troubleshooting-a-group-that-isnt-visible"></a>
### Fehlerbehebung bei einer nicht sichtbaren Gruppe

- **Das JSON-Backend benötigt ein gemeinsames `DATA_DIR`.** Nodes finden einander über die Registry; separate Verzeichnisse bedeuten daher, dass jeder Node keine Peers sieht. Nutzen Sie MariaDB, es sei denn, das Verzeichnis wird tatsächlich geteilt.
- **Eine beeinträchtigte Datenbank verbirgt Peers absichtlich.** Wenn das DB-Mirror Lesezugriffe bedient, sieht ein Node nur sich selbst und verarbeitet Requests lokal.
- **Ein fehlgeschlagener Load-Probe blendet eine Gruppe nicht aus.** Ein Peer, der nicht antwortet, bleibt über seinen Snapshot (bis zu 45 Sekunden alt) routingfähig, wird jedoch hinter jeden Peer eingestuft, der geantwortet hat. Das Log zeichnet auf, wann Probes zu einem Peer fehlschlagen und wann sie sich wieder erholen.
