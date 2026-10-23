# Konfiguration & Speicherung

Bare-Metal-Installationen, das Auth-Modell im Detail, GPU-Stats, Request-Aufzeichnung, Speicher-Backends und die vollständige Umgebungsvariablen-Referenz.

## Inhalt

- [Bare-Metal-Betrieb](#running-bare-metal) · [Authentifizierung im Detail](#authentication-detail) · [GPU-Statistiken](#gpu-stats) · [Request-Aufzeichnung & Statistiken](#request-recording--stats) · [Speicher-Backends](#storage-backends) · [Einen Datenbankausfall überstehen (lokaler Spiegel)](#surviving-a-database-outage-local-mirror) · [Umgebungsvariablen](#environment-variables)

<a id="running-bare-metal"></a>
## Bare-Metal-Betrieb

Nützlich für die Entwicklung oder auf Hosts (z. B. WSL), auf denen das Betreiben des Managers selbst in Docker unbequem ist. llama-server läuft weiterhin in Containern.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

Ein einzelner Prozess unter gunicorn bedient UI/API auf `:5000` und den Proxy auf `:42069` (`python app.py` ohne gunicorn startet nur den Flask-Dev-Server auf `:5000`). Container- vs. Bare-Metal-Betrieb wird automatisch erkannt; falls die Erkennung für Ihre Laufzeitumgebung jemals falsch liegt, setzen Sie `LLAMAMAN_IN_DOCKER=true` oder `false` explizit.

<a id="authentication-detail"></a>
## Authentifizierung im Detail

Zwei Anmeldeinformations-Typen:

- **Session-Cookies** für das UI (über `/setup` beim ersten Start angelegt)
- **API-Keys** für externe Clients (`Authorization: Bearer llm-xxxxxxxxxx`), verwaltet unter **API Keys** im UI

Der Umschalter **„Require authentication for all endpoints“** (standardmäßig aktiv) steuert, ob modellbereitstellende Endpoints ein Bearer-Token verlangen:

| Umschalter | Modell-Endpoints (`/api/chat`, `/v1/chat/completions`, Instanz-Ports) | Verwaltungs-Endpoints (`/api/instances` usw.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (Standard) | Bearer-Token erforderlich | Bearer-Token oder Session |
| **OFF** | Offen | Bearer-Token oder Session |

Alle drei Port-Ebenen (5000, 42069, 8000-8020) durchlaufen denselben Auth-Hook.

<a id="gpu-stats"></a>
## GPU-Statistiken

VRAM und Auslastung pro GPU, nativ abgefragt - kein laufender llama-server nötig.

| Hersteller | Methode | Voraussetzung |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML direkt) | `deploy.resources.reservations` in `docker-compose.yml` einkommentieren für die `utility`-Capability des NVIDIA Toolkits |
| AMD | `/sys/class/drm` sysfs | `/sys/class/drm:ro` Volume-Mount (standardmäßig enthalten) |
| Intel Arc | `/sys/class/drm` sysfs | Gleicher Mount wie bei AMD |

Ohne nativen Zugriff fällt llamaMan darauf zurück, `nvidia-smi` / `rocm-smi` in einem laufenden Container auszuführen. Die Stats bilden stets den vollständigen GPU-Zustand des Hosts ab.

<a id="request-recording--stats"></a>
## Request-Aufzeichnung & Statistiken

Unter **Settings → App Settings → Request recording**: **Off** (Standard) / **Per request** / **Per conversation** (Turns gruppiert über einen Inhalts-Hash aus System-Prompt + erster User-Nachricht). Jeder Datensatz erfasst die Bodies + Envelope-Felder (Modell, Endpoint, Status, Dauer, Token-Anzahlen) plus **präzise Metriken pro Turn** (Durchsatz über das Generierungsfenster gemessen; TTFT). Die Datensätze liegen unter `RECORDINGS_DIR` (JSON) oder in der Tabelle `request_log` (MariaDB). **Retention (days)** bereinigt stündlich (`0` = unbegrenzt behalten).

Jede Instanzkarte stellt einen **Stats**-Button bereit (Request-Anzahl, durchschn./Peak-Durchsatz, durchschn. TTFT/Latenz, Token-Summen, aktiver Zeitraum), aufbereitet aus dem Request-Log - sie bleibt daher erhalten, nachdem die Instanz gestoppt wurde, und zeigt einen Leerzustand, wenn die Aufzeichnung aus ist. Der Link **Logging** in der Kopfzeile öffnet ein seitenfüllendes Dashboard (Übersichtskacheln, kürzliche Konversationen, Drill-down pro Konversation) mit einem 24h- / 7d- / 30d- / All time-Zeitfenster-Wähler sowie Filtern pro Modell und pro Tag über der Konversationsliste.

<a id="storage-backends"></a>
## Speicher-Backends

### JSON (Standard)

Kein Konfigurationsaufwand. Dateien unter `DATA_DIR` (`/data`): `state.json`, `presets.json`, `users.json`, `settings.json`, `api_keys.json`, `request_log/`. Instanz- und Download-Logs gehen nach `LOGS_DIR` (`/tmp/llama-logs`).

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

Tabellen werden bei der ersten Verbindung automatisch angelegt. Request-Logs gehen in die Tabelle `request_log` statt in das JSON-Verzeichnis `request_log/` (`RECORDINGS_DIR` hat bei MariaDB keine Auswirkung).

Knotenbezogene Modell-Metadaten (Quell-Repository + Inhalts-Hash) liegen in einer node-scoped-Tabelle `model_files` mit dem Schlüssel `(node_id, model_path)`. Beim Upgrade kopiert jeder Node seine eigenen Einträge aus dem alten gemeinsamen Settings-Blob (nur Pfade, die auf seiner eigenen Festplatte existieren) und behält den Blob als Lese-Fallback und Rollback-Pfad. Migrationen werden pro Node versioniert.

> **InnoDB-Hinweis:** `(node_id, model_path)` sind unter utf8mb4 `(64 + 700) × 4 = 3056` Bytes, knapp unter der InnoDB-Indexgrenze von 3072 Bytes, daher wird die Tabelle mit `ROW_FORMAT=DYNAMIC` angelegt.

MariaDB ist außerdem die Voraussetzung für [Clustering](CLUSTERING.md) und die [Knowledge Base](KNOWLEDGE-BASE.md).

<a id="surviving-a-database-outage-local-mirror"></a>
## Einen Datenbankausfall überstehen (lokaler Spiegel)

*Optional, standardmäßig deaktiviert.*

Mit gesetztem `DATABASE_URL` liegt die Datenbank auf dem kritischen Pfad **jedes** Requests (Auth-Prüfungen lesen Einstellungen und verifizieren API-Keys bei jedem Aufruf). Wird sie unerreichbar, stellt der Node die Dienste ein, und ein Container-Neustart mitten im Ausfall kann gar nicht mehr hochfahren.

Schalten Sie **Settings → App settings → „Keep a local mirror of the database“** ein (oder setzen Sie `LLAMAMAN_DB_MIRROR=1`), und der Node führt eine Write-Through-Kopie in `DATA_DIR/db_mirror/`.

| Funktioniert offline weiter | Ist offline blockiert |
|---|---|
| Gesamte Inferenz (beide Ports) | Anlegen des ersten Benutzers (`/setup`) - eine brandneue Installation hat ohnehin keinen Spiegel als Fallback |
| Modelle starten / stoppen / schlafen legen / wecken | |
| Downloads, Modell-Hashing und Update-Prüfungen | |
| **Presets** anlegen und bearbeiten (auch für gerade heruntergeladene Modelle) | |
| **API-Keys** anlegen und widerrufen | |
| Download- und App-**Einstellungen** ändern | |
| **Hugging-Face-Tokens** hinzufügen und entfernen | |

Offline-Änderungen landen in einem Append-only-Journal und werden bei Wiederverbindung nachgespielt (10s-Probe). Das Journal protokolliert die *Änderung*, nicht die resultierende Zeile, daher bleibt eine Änderung, die ein anderer Node während des Ausfalls vorgenommen hat, erhalten (ein anderes Preset-Feld, ein anderer Settings-Key, ein anderes HF-Token bleiben jeweils intakt).

**Voraussetzungen und Einschränkungen:**

- **`DATA_DIR` muss ein persistentes Volume sein** (in der Beispiel-Compose-Datei bereits so).
- **Jeder Node braucht sein eigenes `DATA_DIR`.** Der Spiegel ist mit der Node-ID des Besitzers gestempelt; zwei Nodes, die sich einen teilen, schalten die Spiegelung ab, statt eine der beiden Ansichten zu beschädigen.
- **Secrets landen auf der lokalen Festplatte.** Einstellungen werden wortgetreu kopiert; **Hugging-Face-Tokens werden im Klartext gespiegelt.** Passwort- + API-Key-Hashes sind bereits gehasht. Wer MariaDB auch deshalb gewählt hat, um Secrets von einzelnen Hosts fernzuhalten, für den ist dieser Trade-off der Grund, warum das Feature opt-in ist.
- **Knotenübergreifendes Balancing stoppt während eines Ausfalls.** Die Peer-Liveness lebt in der Datenbank; ein degradierter Node sieht nur sich selbst.
- **API-Key-Änderungen sind lokal bis zur Wiederverbindung.** Ein offline erstellter Key funktioniert nur auf diesem Node; ein offline **widerrufener** Key bleibt auf anderen Nodes gültig, bis die Wiederverbindung erfolgt. Planen Sie Widerrufe entsprechend.
- **Request-Logs werden verworfen, nicht gepuffert,** solange offline (die Datensätze tragen vollständige Bodies; sie durch einen langen Ausfall zu spoolen könnte das Volume füllen).
- **Das Abschalten des Spiegels mitten im Ausfall wird aufgeschoben** bis zur Wiederverbindung.
- Schema-Migrationen laufen niemals im degradierten Zustand; sie laufen bei Wiederverbindung gegen die echte Datenbank, vor jedem Journal-Replay.

<a id="environment-variables"></a>
## Umgebungsvariablen

### Core

| Variable | Standard | Beschreibung |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(erforderlich)* | **Erforderlich.** Eindeutige stabile Identität - Partitionsschlüssel für den gespeicherten Zustand und die Cluster-Identität. Beliebiger String; einmal wählen, beibehalten (eine spätere Änderung verwaist den gespeicherten Zustand). |
| `MODELS_DIR` | `/models` | Verzeichnis, das nach Modelldateien durchsucht wird (Container-Pfad) |
| `DATA_DIR` | `/data` | Verzeichnis für persistente Konfiguration/den persistenten Zustand (JSON-Dateien) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Verzeichnis für Request-Log-Datensätze pro Konversation. Nur JSON-Backend; ignoriert, wenn `DATABASE_URL` gesetzt ist |
| `LOGS_DIR` | `/tmp/llama-logs` | Verzeichnis für Instanz- und Download-Logs (Container-Pfad) |
| `HOST_MODELS_DIR` | *(wie `MODELS_DIR`)* | **Host-seitiger** absoluter Pfad des Modell-Volumes. Muss zur linken Seite von `-v /host/path:/models` passen - wird beim Spawnen von Geschwister-Containern an den Docker-Daemon übergeben |
| `HOST_LOGS_DIR` | *(wie `LOGS_DIR`)* | Host-seitiger absoluter Pfad des Log-Volumes. Gleiche Anforderung wie `HOST_MODELS_DIR` |
| `ARCHIVE_DIR` | *(nicht gesetzt)* | Container-Pfad, an dem ein Archiv-Volume gemountet ist (z. B. `/archive`). Setzen aktiviert [Modell-Archiv](MODELS.md#model-archive); nicht gesetzt verbirgt es |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Öffentlicher Port-Pool für llama-server/Proxy |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Interner Port-Pool, der bei aktiviertem Proxy-Modus verwendet wird |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port für den Ollama-kompatiblen Proxy |
| `LLAMAMAN_MAX_MODELS` | `0` | Maximale gleichzeitige Chat-Modelle über den Proxy (LRU-Eviction; 0 = unbegrenzt). Regeln: [Modell-Eviction](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle-Timeout-Minuten für proxy-verwaltete Instanzen (0 = deaktiviert) |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Maximale gleichzeitige PDF-Rasterisierungen prozessweit. Jede Rasterisierung startet poppler und belegt bei hoher DPI einige hundert MB flüchtigen RAM |
| `SECRET_KEY` | *(auto)* | Flask-Session-Secret. Wird, falls nicht gesetzt, automatisch aus der machine-id abgeleitet. Für Multi-Replica-Deployments setzen |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Name des Session-Cookies. Namensraum-isoliert, damit llamaman auf demselben Host mit anderen Flask-Apps koexistiert - Cookies sind über Host+Pfad scoped, nicht über den Port |
| `DATABASE_URL` | *(nicht gesetzt)* | MariaDB/MySQL-Verbindungsstring. Nicht gesetzt = JSON-Dateien |
| `LLAMAMAN_DB_MIRROR` | *(nicht gesetzt)* | Lokalen DB-Spiegel erzwingen: an (`1`) / aus (`0`), überstimmt die node-bezogene Einstellung. Nur sinnvoll mit `DATABASE_URL` |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in Sekunden für Instanz-Health-Checks |
| `MODEL_LOAD_TIMEOUT` | `300` | Sekunden, die auf ein healthy werdendes Modell gewartet wird. Für sehr große Modelle erhöhen |
| `REQUEST_TIMEOUT` | `300` | **Read**-Timeout in Sekunden für Upstream-Requests an llama-server, knotenübergreifendes Forwarding und Gate-Acquire-Waits. Steuert **nicht** die Peer-Connect-Zeit (separate 5s-Grenze) |

### Docker / GPU

| Variable | Standard | Beschreibung |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | llama.cpp-Docker-Image für alle gespawnten Container. Automatisch nach erkanntem GPU-Hersteller gewählt; zum Festlegen einer Version/Backend setzen |
| `LLAMA_NETWORK` | `llamaman-net` | Docker-Netzwerk für llamaman + alle llama-server-Container. Wird angelegt, falls es fehlt |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Namenspräfix für gespawnte llama-server-Container |
| `LLAMAMAN_IN_DOCKER` | *(auto)* | Ob llamaman selbst in einem Container läuft. Automatisch aus Marker-Dateien + cgroups erkannt. Mit `true`/`false` übersteuerbar |
| `LLAMA_HOST_ADDR` | `localhost` | Host-Adresse, über die bei Bare-Metal-Betrieb die publizierten Ports der gespawnten Container erreicht werden |
| `GPU_TYPE` | *(auto)* | GPU-Hersteller-Erkennung übersteuern: `cuda`, `rocm`, `intel` oder `vulkan` (`vulkan` ist nur opt-in; wählt `/dev/dri` + die render/video-GIDs des Hosts und das `server-vulkan`-Image) |
| `LLAMA_GPU_DEVICES` | *(alle)* | Kommagetrennte GPU-Indizes, die für alle gespawnten Container sichtbar sind (z. B. `0,1,3`). Override über **GPU Devices** pro Instanz. Nicht unterstützt auf Intel Arc |

### Strata

Optional; siehe [Strata](STRATA.md).

| Variable | Standard | Beschreibung |
|---|---|---|
| `STRATA_ENABLED` | `false` | Strata-Modelle auf diesem Node anbieten. Braucht zusätzlich eine NVIDIA-GPU (`GPU_TYPE=cuda` oder erkannt) |
| `STRATA_IMAGE` | `strata:latest` | Bevorzugtes Strata-Image für Launches. Wenn es auf dem Node fehlt, wird das gebaute, danach ein gezogenes Image verwendet |
| `STRATA_BUILD_IMAGE` | `strata:latest` | Tag, den *Pull repo & build latest* baut |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | Benanntes Docker-Volume, das in jedem Strata-Container unter `/data` gemountet wird (Modelldateien, Packs, MTP-Layer, Setup pro Modell; 70-120 GB pro Modell). Wird von Docker bei der ersten Nutzung angelegt |
| `HOST_STRATA_DATA_DIR` | *(nicht gesetzt)* | Host-Pfad, der anstelle des benannten Volumes unter `/data` gemountet wird. Hat Vorrang vor `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | Sekunden, die ein Request (oder Relaunch) wartet, bis eine Strata-Instanz bereit ist. Hoch, weil ein Erststart das Modell herunterlädt und vorbereitet, bevor der Port öffnet |
| `STRATA_REPO` | `Niko1221/Strata` | GitHub-Repository (owner/name), das *Pull repo & build latest* herunterlädt |
| `STRATA_REPO_REF` | `main` | Branch, Tag oder Commit von `STRATA_REPO`, der gebaut wird |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | Wo das heruntergeladene Repository gehalten wird (der Build-Kontext). Sein Commit wird daneben in `Strata.source.json` vermerkt |
| `STRATA_WEB_HOSTS` | *(nicht gesetzt)* | Zusätzliche Hostnamen / IPs (kommagetrennt), über die Sie die eigene Web-App einer Strata-Instanz öffnen (`http://<host>:<instance port>/`). Ohne API-Key lehnt Strata Chat-Requests von Seiten unter anderen Namen ab; der Host von `CLUSTER_ADVERTISE_URL` ist immer erlaubt. Gilt für danach gestartete Instanzen |

### Clustering

Optional. `LLAMAMAN_NODE_NAME` (unter Core) dient zugleich als Cluster-Identität und ist für alle Installationen erforderlich.

| Variable | Standard | Beschreibung |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Auf truthy setzen, um diesen Node in einen Cluster einzugliedern. Erfordert `CLUSTER_SECRET`; bei leerem Secret mit Warnung ignoriert |
| `CLUSTER_SECRET` | *(nicht gesetzt)* | Gemeinsames Bearer-Token (`X-Cluster-Secret`) für die gesamte Node-zu-Node-HTTP-Kommunikation. Auf jedem Node identisch. Über ein vertrauenswürdiges Netzwerk oder hinter TLS einen langen Zufallswert verwenden |
| `CLUSTER_ADVERTISE_URL` | *(nicht gesetzt)* | Wie Peers die UI/API **dieses** Nodes erreichen - ein von den anderen Hosts aus routbarer Hostname/IP (nicht `localhost`). Ohne ihn ist der Node schreibgeschützt |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Sekunden seit dem letzten Heartbeat, bevor ein Node als offline angezeigt wird. Erhöhen, wenn Nodes unter Last oder bei Uhrdrift flattern |
| `LLAMAMAN_PERF_LOG` | *(aus)* | Diagnose. Auf truthy setzen, um Phasen-Timings (`perf <name> <ms>`) für vermutete Slow Paths zu protokollieren: Stoppen entfernter Instanzen, Cluster-Proxy-Forwards, Erreichbarkeits-Probes, Heartbeat-Snapshot, Auth-Checks, `save_state`, Container-Stats. Standardmäßig aus; im aus-Zustand vernachlässigbarer Overhead |
