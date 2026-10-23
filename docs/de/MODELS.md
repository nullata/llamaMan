# Modelle, Downloads & Eviction

Alles rund um die Modellbibliothek: Modelle hinzufügen, aktuell halten, wie der Proxy sie auswählt und verdrängt, und die Aufräum-Einstellungen.

## Inhalt

- [Modelle](#models) · [Entscheidungsmodelle](#decision-models) · [Modell-Eviction](#model-eviction) · [Download-Einstellungen](#download-settings) · [Docker-Image-Verwaltung](#docker-image-management) · [Modell-Archiv](#model-archive) · [Modell-Backup und -Wiederherstellung](#model-backup-and-restore) · [Bereinigungs-Einstellungen](#cleanup-settings)

<a id="models"></a>
## Modelle

Modelle liegen als `.gguf`-Dateien oder HuggingFace-Repository-Verzeichnisse (mit `config.json`) unter dem `/models`-Volume, oder Sie nutzen den **Download**-Button im UI.

<a id="display-name"></a>
### Anzeigename

Modelle werden standardmäßig nach GGUF-Dateinamen-Stem gelistet. Jedes Modell kann einen freundlichen **Display Name** erhalten (im Launch-Formular). Wenn gesetzt, ist er die ID, die API-Clients auf `/api/tags` und `/v1/models` sehen, sodass OpenWebUI `Qwen 2.5 14B` statt `Qwen2.5-14B-Instruct-Q4_K_M` anzeigt. Muss eindeutig sein und darf nicht mit dem Dateinamen eines anderen Modells oder dem Namen einer Cluster-Queue-Gruppe kollidieren.

<a id="model-updates"></a>
### Modell-Updates

Der Button **Check for updates** (angezeigt für Modelle mit bekanntem Quell-Repository) vergleicht die lokale Datei mit dem veröffentlichten Inhalts-Hash des Repositories:

- **Up to date** - aufgezeichneter Hash stimmt überein
- **Update available** - lädt in einen Staging-Ordner und tauscht atomar ein, erst wenn der Download vollständig ist (ein fehlgeschlagener Pull lässt das aktuelle Modell intakt). Lehnt mit 409 ab, solange das Modell geladen ist.
- **Verify hash** - hasht die lokale Datei (Hintergrund-Job mit Fortschritt), wenn noch kein Hash aufgezeichnet wurde

Ein opt-in Hintergrund-Scan (**Settings → Download Settings**, siehe [Download-Einstellungen](#download-settings)) hält diese Antworten vorsorglich bereit.

<a id="decision-models"></a>
## Entscheidungsmodelle

Decision Models beantworten typisierte Fragen zu einer Eingabe mit Wahrscheinlichkeiten, statt Text zu generieren: die von TypeSafe's Jev eingeführte System-One-API, bedient von llama.cpp (Images nach dem 2026-10-05) auf `POST /v1/systemone`. Offene GGUFs: Julia-1, Laya, Kev-4B, lev, Clef-Flash, Clef, OpenJev (OpenJev und Clef akzeptieren auch Bilder, mit ihrem `--mmproj`). llama-server erkennt sie an `<arch>.decision.type` im GGUF; kein Flag nötig.

- **Erkennung.** Wer eines im Launch-Formular wählt, bekommt das unter Model Path angezeigt. Ohne Preset wird das Formular für es vorbelegt: **Exclude from Max Models** an, **Web UI** aus (dessen Chat-Seite kann das Modell nicht ansteuern), **Parallel Slots** `8` (jede Frage eines Requests belegt einen Slot, und sie teilen sich die Eingabe), **Embedding Model** aus (llama.cpp schaltet den Embedding-Modus selbst ein, wo das Modell ihn braucht; Erzwingen ist nicht nötig). Ein Encoder-Modell (BERT-Stil) bekommt außerdem Context, Batch und Micro-batch Size auf seine Kontextlänge gesetzt: die gesamte Eingabe muss in einen Micro-batch passen. **Clef** (und Clef-Flash) stellt alle Fragen in einem Prompt und liest sie aus einem Durchgang, daher bekommt es **Parallel Slots** `1` und Batch = Micro-batch = Context (16384, sein trainierter Default, wenn Context leer ist): der gesamte Request (Eingabe, jede Frage und Option) muss in einen Micro-batch passen, sonst antwortet llama-server mit „increase the physical batch size“.
- **Routing.** `POST :42069/v1/systemone` wählt das Modell über `"model"` wie `/v1/chat/completions` (Auto-Start, Eviction-Regeln, Cluster-Forwarding, Queue). Der Request wird unverändert weitergeleitet (keine Sampling-Overrides). Auf dem eigenen Port der Instanz läuft er über die Queue des Instanz-Proxys, wenn dieser läuft.
- **Chat-Endpoints lehnen sie ab** (HTTP 422) - sie generieren keinen Text.
- **Request-Log:** die Antworten, und `input_tokens` als Prompt-Tokens (`output_tokens` ist immer 0).
- **Fehler:** Eine Frage, deren Optionen nicht in einen Batch passen, wird von llama-server abgewiesen („increase the physical batch size“) - Batch Size / Micro-batch Size erhöhen.

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
## Modell-Eviction

`LLAMAMAN_MAX_MODELS` begrenzt gleichzeitige **Chat**-Modelle über den Proxy. Ist das Cap voll und ein neues Chat-Modell angefragt, wird das LRU-Chat-Modell verdrängt:

| Launcher | Verdrängt | Kann nicht verdrängen |
|---|---|---|
| **Admin UI** | Ollama-verwaltete zuerst (LRU), dann admin-gestartete | - |
| **Ollama API** | Ollama-verwaltete (LRU) | Admin-gestartete *(standardmäßig)* |
| **OpenAI API** | Nichts *(standardmäßig; liefert 503 wenn voll)* | Alles *(standardmäßig)* |

Drei Umschalter unter **Settings → App Settings** lockern diese Defaults:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - stille LRU-Eviction vor dem Launch (Ollama-verwaltete zuerst). Standardmäßig aus → das UI fragt erst nach.
- **Allow Ollama API to evict admin-launched models** - Fallback, wenn keine Ollama-verwalteten Modelle verfügbar sind. Standardmäßig aus.
- **Allow OpenAI API to evict admin-launched models** - gewährt der OpenAI-API dieselbe LRU-Eviction wie Ollama. Standardmäßig aus.

Weitere Details:

- Alle laufenden Chat-Instanzen zählen (sowohl Admin-UI- als auch proxy-verwaltete).
- **Schlafende Instanzen zählen weiterhin** (die Slot-Belegung bleibt über die Idle-Pause bestehen) - aber das Wecken eines Schläfers für sein eigenes Modell wird niemals durch das Cap blockiert.
- Instanzen mit **Exclude from Max Models** an (Launch-Formular) zählen nicht und werden nie verdrängt: Embedding-Modelle, Decision Models, alles Kleine, das dauerhaft laufen soll. **Embedding Model** einzuschalten schaltet es ebenfalls ein, und Presets, die vor Einführung des Umschalters gespeichert wurden, erhalten ihn aus ihrer Embedding-Model-Einstellung.
- `LLAMAMAN_MAX_MODELS=0` deaktiviert die Eviction vollständig.

<a id="download-settings"></a>
## Download-Einstellungen

Unter **Settings → Download Settings**:

- **Global speed limit (Mbps)** (0 = unbegrenzt) - drosselt jeden Download auf diesem Node und wirkt auf laufende Downloads innerhalb einer Sekunde. Pro Node: In einem Cluster hat jeder Node sein eigenes Limit (ein Wert, der gespeichert wurde, bevor er node-bezogen wurde, gilt für alle Nodes, bis dieser Node sein eigenes speichert)
- **Auto-retry failed downloads** (standardmäßig aus) + **Retry count per failed download** (Standard 3)
- **Check models for updates in the background** (standardmäßig aus) - opt-in-Worker, der jedes Quell-Repository fragt, ob eine Datei neu veröffentlicht wurde, plus einen Prüfsummen-Hash für jedes Modell berechnet, das noch keinen hat. **Update check interval (hours)** Standard 24. Hasht höchstens ein Modell pro Durchgang; läuft niemals, während ein Download läuft.

<a id="docker-image-management"></a>
## Docker-Image-Verwaltung

**Settings → Docker Images**: beliebige llama.cpp-Images per Namen ziehen, alte lokale Images löschen (deaktiviert für das aktive `LLAMA_IMAGE`, und liefert einen Fehler, wenn Docker ablehnt, weil ein Container es nutzt), und optional die llama.cpp-Images des Nodes (`LLAMA_IMAGE` und jedes dort gezogene Image) zeitgesteuert automatisch aktualisieren. Die Pull-Zeile hat einen Engine-Wähler: mit `STRATA_ENABLED` geht ein als *Strata* gezogenes Image zu *Manage Strata images* (es gibt keine Standard-Strata-Quelle; den Namen eintippen). **Pull repo & build latest** lädt den neuesten Commit des Strata-Repositorys herunter (ein GitHub-Tarball, nach `STRATA_SRC_DIR`) und baut `STRATA_BUILD_IMAGE` mit BuildKit über den Docker-Socket. Stratas eigene Auto-Aktualisierung zieht die gezogenen Strata-Images erneut und holt, wenn das Repository heruntergeladen wurde, dessen neuesten Commit und baut neu, wenn er sich geändert hat (oder das Image fehlt); sie lädt das Repository niemals von selbst herunter. **Delete image & repo** entfernt das gebaute Image (abgelehnt, solange eine laufende Instanz es nutzt) und das heruntergeladene Repository, für einen Node, der ein gezogenes Image betreibt. Strata-Launches nutzen `STRATA_IMAGE`, wenn es auf dem Node ist, sonst das gebaute, sonst ein gezogenes. Siehe [Strata](STRATA.md).

<a id="model-archive"></a>
## Modell-Archiv

*(opt-in)* Modelle per UI von schnellem Speicher auf ein zweites Volume und zurück verschieben, statt Dateien auf dem Host von Hand zu schieben.

**Aktivieren**, indem Sie das Archiv-Volume in den llamaMan-Container mounten und `ARCHIVE_DIR` darauf zeigen lassen:

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

Nicht gesetzt ist das Feature verborgen. llamaMan verschiebt die Dateien selbst, daher wird kein `HOST_`-Pfad benötigt.

**Nutzung.**

- Mit der Maus über ein Modell in der Bibliothek fahren und auf das Archiv-Symbol klicken.
- Archivierte Modelle bleiben abgeblendet in der Liste, mit einem *archived*-Badge und einem **Restore**-Button. Sie können zum Launch nicht ausgewählt werden und erscheinen niemals in `/api/tags` / `/v1/models`.
- Verschiebungen erscheinen im **Downloads**-Panel mit Fortschritt, Geschwindigkeit und Restzeit, und können abgebrochen werden.
- Restore legt ein Modell an seinem ursprünglichen Pfad ab, daher kommen Presets, Favoriten, Notizen und Display Names mit zurück.

**Was verschoben wird.** Der gesamte Top-Level-Ordner eines Modells unter `/models`: der Ordner eines Downloads, mit jedem Shard eines gesplitteten GGUF. Eine `.gguf` direkt in `/models` wandert zusammen mit ihren Split-Geschwistern. Der Shard-Ordner eines Strata-Modells (`strata/<tag>`) wandert allein.

**Sicherheit.**

- **Gleiche Festplatte:** die Verschiebung ist ein sofortiges Rename.
- **Festplattenübergreifend:** llamaMan kopiert in einen versteckten `.llamaman-partial-*`-Ordner am Ziel, fsynced, prüft die Größe jeder Datei, benennt sie an ihren Platz um und löscht erst dann die Quelle.
- **Fehler:** Ein Abbruch, eine fehlgeschlagene Prüfung oder zu wenig freier Speicher lassen die Quelle unberührt. Teil-Kopien, die ein Neustart mitten in der Verschiebung hinterlassen hat, werden beim Start entfernt.
- **Nur eine Verschiebung gleichzeitig.**
- **Abgelehnt**, während eine Instanz das Modell nutzt (einschließlich einer schlafenden, oder als Draft-Modell / mmproj, oder einer Strata-Instanz, die die Shards nutzt), während ein Download hineinschreibt, oder wenn der Zielname bereits existiert. Auch das Starten oder Löschen eines Modells, das gerade verschoben wird, wird abgelehnt.

**Grenzen.** Move-Jobs leben im Speicher: Nach einem llamaMan-Neustart ist die Liste leer, und eine unterbrochene Verschiebung muss neu gestartet werden (nichts geht verloren). Im Cluster-Modus archiviert jeder Node auf sein eigenes Volume, und die Buttons erscheinen für den Node, den Sie gerade durchsuchen.

<a id="model-backup-and-restore"></a>
## Modell-Backup und -Wiederherstellung

**Settings → App Settings**:

- **Download Stored Models JSON** - exportiert alle gescannten Modelle + Preset-Konfigurationen in ein zeitgestempeltes JSON
- **Restore from JSON** - pro Modell: bereits vorhanden → Preset wird eingepflegt (bestehende Werte bleiben erhalten); fehlt, aber mit HuggingFace-Quelle → wird zum Download eingereiht mit vorbelegtem Preset; fehlt und ohne bekannte Quelle → als nicht wiederherstellbar gemeldet

<a id="cleanup-settings"></a>
## Bereinigungs-Einstellungen

Unter **Settings → Cleanup Settings** (alle laufen regelmäßig im Hintergrund; entfernen/aktualisieren nur Datensätze - löschen niemals Modelldateien):

- **Auto-clean completed/failed downloads** (Standard 24h) - aktive Downloads werden nie angefasst
- **Auto-clean stopped instances** (Standard 24h) - laufende Instanzen werden nie entfernt
- **Auto-remove stale instance records** (Standard 5-Min-Intervall) - prüft `starting`/`healthy`/`sleeping`-Datensätze erneut gegen Docker; Datensätze, deren Container verschwunden ist, werden als gestoppt markiert
