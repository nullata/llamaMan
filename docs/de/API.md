# REST API

> **Hinweis:** Dies ist eine Übersetzung und kann hinter der [englischen Fassung](../en/API.md) zurückliegen; maßgeblich ist die englische Fassung.

Alle Endpoints akzeptieren / liefern JSON. Verwaltungs-Endpoints erfordern ein Session-Cookie (Browser-Login) oder `Authorization: Bearer <key>`. Modellbereitstellende Endpoints erfordern ebenfalls ein Bearer-Token, wenn `require_auth` aktiv ist (Standard). Authentifizierungsdetails: [Authentication](CONFIGURATION.md#authentication-detail).

## Inhalt

- [Authentifizierung und Keys](#authentifizierung-und-keys) · [Instanzen](#instanzen) · [Downloads, Modelle und Presets](#downloads-modelle-und-presets) · [Modellfehler auf Inferenz-Routen](#model-errors-on-inference-routes) · [Einstellungen, System und Request-Log](#einstellungen-system-und-request-log) · [Ollama und OpenAI-Kompatibilität](#ollama-und-openai-kompatibilität)

## Authentifizierung und Keys

| Methode | Endpoint | Beschreibung |
|---|---|---|
| `GET` / `POST` | `/login` | Login-Seite / Authentifizierung |
| `GET` / `POST` | `/setup` | Ersteinrichtung |
| `GET` | `/logout` | Sitzung beenden |
| `GET` / `POST` | `/api/api-keys` | API-Keys auflisten / erstellen |
| `DELETE` | `/api/api-keys/<id>` | Widerrufen |

## Instanzen

| Methode | Endpoint | Beschreibung |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Auflisten / starten |
| `GET` / `DELETE` | `/api/instances/<id>` | Abrufen / stoppen und entfernen |
| `POST` | `/api/instances/<id>/restart` | Gestoppte/schlafende Instanz neustarten |
| `DELETE` | `/api/instances/<id>/remove` | Datensatz einer gestoppten Instanz entfernen |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Tail / SSE-Stream der Logs |
| `GET` | `/api/next-port` | Nächsten verfügbaren Port aus dem Pool abrufen |
| `POST` | `/api/preview-command` | Den äquivalenten `llama-server ...`-Befehl für einen Launch-Body rendern (genutzt vom UI-Tool „Copy llama.cpp command“; startet keine Instanz) |

**Launch-Body** (`POST /api/instances`): Die Felder des Startformulars, serialisiert als JSON. Siehe das [Launch-Settings-Referenzdokument](LAUNCH-SETTINGS.md#launch-settings-reference) für jedes Feld und seinen Standardwert; Enum-Felder (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) akzeptieren nur die dort aufgeführten Werte und fallen auf den Referenzstandard zurück. Numerische Felder, die als `null` übergeben oder weggelassen werden, nutzen den Standardwert der Referenz; unbekannte Felder werden ignoriert.

Wichtige Randfall-Besonderheiten:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp schätzt VRAM), `-2` = **all** (alle Layer erzwingen), `0` = nur CPU (kein GPU-Gerät am Container angebunden), positives N = exakt N Layer.
- `n_cpu_moe_layers`: `0` (Standard) lässt beide MoE-Flags weg. `-1` übergibt `--cpu-moe` (alle gerouteten Experten auf CPU). Positives N übergibt `--n-cpu-moe N` (Experten der ersten N Layer auf CPU). Wirkungslos bei dichten (Dense-) Modellen.
- `threads`: Wenn gesetzt, übergibt dies `--threads N` an llama-server **und** setzt das CPU-Kontingent des Containers auf N Kerne.
- `threads_batch`: null / weggelassen → das Flag wird nicht übergeben und llama-server fällt auf `--threads` zurück.
- `flash_attn`: Veraltete `true` / `false`-Werte aus Konfigurationen vor der Tri-State-Einführung werden beim Lesen normalisiert (`true` → `"on"`, `false` → `"off"`), sodass keine Speichermigration erforderlich ist.
- `cache_type_v` quantisierte Werte erfordern `flash_attn: "on"` – `"auto"` ist keine Garantie; llama-server verweigert andernfalls den Start.
- `spec_draft_model` ist erforderlich, wenn `spec_enabled: true` und `spec_type` nicht `"draft-mtp"` ist – ein Request ohne dieses Feld wird mit 400 abgewiesen.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / weggelassen lässt das Flag weg. `0` ist ein gültiger Wert ungleich null und wird durchgereicht.

## Downloads, Modelle und Presets

| Methode | Endpoint | Beschreibung |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Auflisten / starten |
| `GET` / `DELETE` | `/api/downloads/<id>` | Abrufen / abbrechen |
| `DELETE` | `/api/downloads/<id>/remove` | Abgeschlossenen/fehlgeschlagenen Eintrag entfernen |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Tail / SSE-Stream |
| `GET` | `/api/models` | Gefundene Modelle (enthält `repo_id`, falls bekannt; archivierte Modelle sind als `archived` mit `restore_path` markiert) |
| `POST` | `/api/models/delete` | Von Festplatte löschen (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Layer-Anzahl aus GGUF-Metadaten lesen |
| `GET` | `/api/disk-space` | Freier/belegter Speicherplatz auf dem Modell-Volume |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Preset abrufen / speichern / löschen (`GET /api/presets` listet alle auf) |
| `GET` | `/api/engines` | Inferenz-Engines dieses Nodes: Verfügbarkeit (mit Begründung), Fähigkeiten, Startfelder, Strata-Modellkatalog |
| `GET` | `/api/archive` | Archivstatus (`enabled`, `available`, `reason`, `free_bytes`) und Verschiebe-Jobs |
| `POST` | `/api/archive` | Modell archivieren: `{"path": "/models/..."}` (verschiebt den gesamten Ordner). 409 bei Nutzung |
| `POST` | `/api/archive/restore` | Archiviertes Modell wiederherstellen: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Eingereihten/laufenden Move-Job abbrechen (Quelle bleibt erhalten) oder abgeschlossenen entfernen |
| `POST` | `/api/engines/strata/download` | Shards eines Strata-Modells vorab herunterladen (`{"model": "strata/qwen-IQ2_XS"}`, optional `hf_token_id` / `speed_limit_mbps`) |

Download-Body: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Ein leerer `filename` zieht das gesamte Repository.

Instanzen und Presets akzeptieren ein optionales `engine` (`llamacpp` als Standard oder `strata`; abgeleitet aus einem `/strata/...`-Modellpfad) sowie für Strata zusätzlich `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) und `strata_low_ram` (`auto`/`on`/`off`). Qwen3.6-35B-A3B akzeptiert außerdem `strata_parallel` (`1` bis `8`, Standard `1`), was jede andere Strata-Familie über `1` ablehnt. Strata-Modellpfade sind virtuell: `/strata/<family>-<SIZE>`, z. B. `PUT /api/presets/strata/qwen-IQ2_XS`. Siehe [Strata](STRATA.md).

<a id="model-errors-on-inference-routes"></a>
## Modellfehler auf Inferenz-Routen

Wenn ein Modell nicht bedient werden kann, geben die Ollama-Routen `{"error": "<message>"}` zurück und die OpenAI-Routen `{"error": {"message": "<message>", "code": "<code>"}}`.

| Status | Code | Bedeutung |
|---|---|---|
| `429` | `group_full` | Andere Nodes hosten die Gruppe und die Warteschlange mindestens eines Peers ist voll. Ein erneuter Versuch kann funktionieren. |
| `503` | `group_unavailable` | Andere Nodes hosten die Gruppe, aber keiner kann die Anfrage annehmen. Die Meldung benennt jeden Node und den Grund. |
| `503` | `group_stopped` | Jedes Mitglied der Gruppe auf diesem Node ist gestoppt |
| `503` | `model_stopping` | Das Mitglied wird heruntergefahren |
| `503` | `wake_failed` | Das Aufwecken des Modells ist fehlgeschlagen |
| `503` | `wake_timeout` | Das Modell wurde nicht rechtzeitig healthy |
| `503` | `model_limit` | `LLAMAMAN_MAX_MODELS` ist erreicht und nichts kann verdrängt werden |

Ein Modell, das schlicht unbekannt ist, gibt `model '<name>' not found` zurück (als `500` auf Ollama-Routen und `503` auf OpenAI-Routen, ohne Code). Eine weitergeleitete Anfrage, die auf einem Peer in ein Timeout läuft, gibt `504` zurück. Ein Modell, das gestartet wurde oder bereits geladen wurde, aber nie bereit wird, gibt `500` ohne Code zurück, und die Anfrage versucht kein anderes Mitglied.

## Einstellungen, System und Request-Log

| Methode | Endpoint | Beschreibung |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Globale Einstellungen abrufen / speichern. Node-spezifische Schlüssel umfassen `eager_wake_enabled` und `eager_wake_polls`; siehe [Aufwecken unter Last](CLUSTERING.md#waking-under-load) |
| `GET` | `/api/system-info` | CPU-Auslastung, Kernanzahl, RAM-Auslastung |
| `GET` | `/api/gpu-info` | VRAM und Auslastung pro GPU (nativ; Fallback auf Container-Exec) |
| `GET` | `/health` | `{"status": "ok"}` – stets erreichbar, keine Authentifizierung erforderlich |
| `GET` | `/api/request-log/conversations` | Kürzliche Konversationen mit aggregierten Metadaten (`limit`-Query, Standard: 100, Max: 500) |
| `GET` | `/api/request-log/conversations/<id>` | Alle aufgezeichneten Turns einer Konversation, älteste zuerst |
| `GET` | `/api/request-log/stats` | Aggregierte Metriken (Token, durchschn./Peak-Token/s, TTFT, Latenz, Fehler, Stream-Anzahl). Optional `inst_id` + `window_hours` |

## Ollama und OpenAI-Kompatibilität

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate starten automatisch).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat startet automatisch). In einem Cluster wird `/v1/embeddings` wie die anderen Inferenz-Routen über die Nodes geroutet.
Entscheidungsmodelle: `/v1/systemone` (startet automatisch) – siehe [Entscheidungsmodelle](MODELS.md#decision-models).
Knowledge Base: `/api/kb/*` (nur Administratoren) und der MCP-Endpoint `/mcp/knowledge` – siehe [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Node-zu-Node-Endpoints im Cluster nutzen `X-Cluster-Secret` – siehe [Clustering](CLUSTERING.md).
