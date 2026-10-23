# Knowledge Base & MCP

Speichern Sie Dokumente gruppiert nach **Topics** (Themen). llamaman segmentiert (chunkt) diese (~1200 Zeichen, absatzorientiert, 200 Zeichen Überlappung), generiert Embeddings und speichert die Vektoren in Ihrer MariaDB-Datenbank (`VECTOR`-Spalten + ein Cosine-ANN-Index). Jeder MCP-Client (Claude Desktop, Claude Code, Agenten) kann diese anschließend semantisch durchsuchen. Alles läuft auf Hardware, die Sie bereits betreiben: Embeddings stammen von einer llama.cpp-Instanz, die Sie mit aktiviertem **Embedding model** gestartet haben, Vektoren liegen in Ihrer Datenbank und llamaman stellt den MCP-Endpoint selbst bereit.

**Anforderungen:** Das MariaDB-Backend (`DATABASE_URL`, siehe [Speicher-Backends](CONFIGURATION.md#storage-backends)) auf Server **11.8+**, ab dem `VECTOR` unterstützt wird. Beim JSON-Backend oder einer älteren MariaDB-Version erläutert der MCP-Tab, warum die Knowledge Base nicht unterstützt wird, und die Aktivierung wird verweigert; alles andere bleibt unverändert. Es ist keine Migration erforderlich: Die `kb_*`-Tabellen werden bei der ersten Nutzung automatisch erstellt.

**Einrichtung:**

1. Starten Sie eine Instanz mit aktiviertem **Embedding model**. Jedes Embedding-GGUF funktioniert (`bge-m3`, `nomic-embed-text`, …).
2. **Settings → MCP** → aktivieren Sie **Knowledge base** und wählen Sie das Modell unter **Embedding model** aus. Das Dropdown-Menü listet jedes embedding-fähige Modell im gesamten Cluster auf.
3. Fügen Sie Dokumente mit dem MCP-Tool `kb_ingest_document` hinzu (aktivieren Sie **Allow ingest over MCP**) oder über `POST /api/kb/documents` aus einer angemeldeten Sitzung. Der Tab listet Topics auf, filtert und löscht sie und bietet eine Testsuche, besitzt jedoch noch kein eigenes Formular zum Hinzufügen.
4. Aktivieren Sie **Expose over MCP**. Der Tab zeigt die Endpoint-URL (`http://<your-host>:42069/mcp/knowledge`) an, und **Copy config.json** kopiert eine direkt einsetzbare Client-Konfiguration:

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

Der `headers`-Block ist nur erforderlich, wenn ein Key benötigt wird (siehe Zugriffsmodi unten).

**Tools.** `kb_list_topics`, `kb_list_documents`, `kb_search` (Passagen nach Relevanz sortiert, Score 0–1) und `kb_get_document` sind immer verfügbar. `kb_ingest_document` (erstellt das Topic, falls nicht vorhanden; 200 KB pro Aufruf; das erneute Senden identischer Inhalte ist eine No-Op) erscheint nur bei aktiviertem **Allow ingest over MCP**. `kb_delete_document` / `kb_delete_topic` erscheinen nur bei aktiviertem **Allow delete over MCP**. Das Löschen eines Topics löscht alle zugehörigen Dokumente, und Löschvorgänge können nicht rückgängig gemacht werden.

**Zugriffsmodi** (**MCP access**):

- **Global** (Standard): Jeder Client arbeitet auf dem **gemeinsamen Pool**, kann ihn einsehen und – sofern Ingest/Löschen erlaubt sind – bearbeiten. Neue Topics landen im gemeinsamen Pool. Die Authentifizierung folgt **Require Authentication** (Tab „API Keys“): Aktiv bedeutet, dass ein gültiger API-Key erforderlich ist, inaktiv bedeutet, dass der Endpoint öffentlich zugänglich ist.
- **Per API key**: Jeder Request benötigt einen gültigen API-Key, selbst wenn Require Authentication deaktiviert ist. Topics, die von einem Key erstellt werden, sind für diesen Key **privat**, und zwei Keys können jeweils ein Topic mit demselben Namen besitzen. Jeder Key sieht den gemeinsamen Pool zudem **schreibgeschützt**: Er kann ihn durchsuchen und lesen, jedoch nichts hinzufügen (Ingest) oder daraus löschen. Topics und Dokumente anderer Keys werden als „nicht gefunden“ gemeldet.

Ein Moduswechsel verschiebt oder löscht niemals Daten, und private Topics bleiben in beiden Modi privat. Der Wechsel von Global → Per API key überführt alle vorhandenen Daten in den gemeinsamen Pool. Der Wechsel von Per API key → Global verbirgt private Topics vor allen Clients, bis Sie zurückwechseln. Der MCP-Tab zeigt stets alles an und kennzeichnet jedes Topic als `shared` oder mit dem Namen des besitzenden Keys. `POST /api/kb/topics` kann zudem ein privates Topic für einen bestimmten Key (`owner_key_id`) anlegen, was nützlich ist, wenn MCP-Ingest deaktiviert ist.

**Embedding-Modell.** Die Auswahl erfolgt nach Modellname, nicht nach Instanz. Jeder Embed-Aufruf läuft auf dem ersten zutreffenden Ziel:
1. einer funktionierenden (healthy) Instanz auf diesem Node;
2. einer schlafenden oder gestoppten Instanz auf diesem Node, die aufgeweckt wird (oder einer Instanz, die gerade startet und auf die gewartet wird);
3. einem Peer-Node, auf dem sowohl **Knowledge base** als auch **Expose over MCP** aktiv sind und der über das Modell verfügt (eine dort schlafende Instanz wird aufgeweckt);
4. einem Neustart auf dem Node, der das Modell zuletzt bereitgestellt hat, maximal 3 Versuche pro 10 Minuten.

Der erste Embedding-Vorgang erfasst die Vektordimension. Ein Modell mit abweichender Dimension wird verweigert, bis Sie **Re-embed everything** ausführen – einen Hintergrundjob, der die Chunks aller Dokumente neu aufbaut (jeweils ein Job zur Zeit, geschützt durch ein clusterweites DB-Lock). Da nur die Dimension geprüft wird, sollten Sie den Re-Embed-Vorgang manuell anstoßen, wenn Sie zwischen zwei Modellen mit identischer Dimension wechseln.

**Hinweise zum Cluster.** Topics und Dokumente sind clusterweite Daten in der gemeinsamen Datenbank, nicht knotenspezifisch. Jeder Node sieht dieselbe Knowledge Base, und der MCP-Endpoint auf jedem Node stellt sie bereit. Siehe [Clustering](CLUSTERING.md).

**Bei Deaktivierung** gibt `/mcp/knowledge` in beiden Zugriffsmodi den Status 404 zurück, und kein Request greift auf die Tabellen der Knowledge Base zu.

> **Sicherheit:** `/api/kb/*` ist Administratoren vorbehalten: Angemeldete UI-Sitzungen und Cluster-Peers werden akzeptiert, mit einem API-Key authentifizierte Requests erhalten 403. API-Keys sind jedoch nicht auf die Knowledge Base beschränkt: Ein Key, den Sie einem MCP-Client übergeben, kann auch die Inferenz-Endpoints und die meisten anderen `/api/*`-Routen aufrufen. Verwenden Sie für jeden Client einen dedizierten Key und widerrufen Sie ihn im Zweifel.
