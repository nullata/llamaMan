# Fehlerbehebung

Die vollständige Tabelle „Symptom → Behebung“. Die [README](../../README.md#troubleshooting) enthält die neun häufigsten Zeilen für den Erstbetrieb; alles Weitere finden Sie hier.

| Symptom | Behebung |
|---|---|
| Instanz verbleibt auf **starting** | Logs über die Schaltfläche „Logs“ prüfen. Häufig: OOM, Tippfehler im Modellpfad, beschädigtes GGUF, Image nicht gepullt |
| _"Docker image not found"_ | Passendes Image pullen: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (oder `server-rocm` / `server-sycl` / `server`) |
| _"Docker API error"_ beim Start | Sicherstellen, dass `/var/run/docker.sock` gemountet ist (Standard in `docker-compose.yml`) |
| Keine GPU / CUDA-Fehler | Sicherstellen, dass das NVIDIA Container Toolkit installiert ist und `docker run --gpus all` auf dem Host funktioniert |
| Keine GPU / ROCm-Fehler | Sicherstellen, dass `/dev/kfd` und `/dev/dri` existieren und Ihr Benutzer in der Gruppe `video`/`render` ist |
| Keine GPU / Intel-Arc-Fehler | Sicherstellen, dass `/dev/dri` zugänglich ist und Ihr Benutzer in der Gruppe `video`/`render` ist |
| Vulkan-Image schlägt mit _"Unable to find group render"_ fehl | `GPU_TYPE=vulkan` setzen, damit llamaMan die Host-GIDs für render/video numerisch anhängt (namensbasiertes group_add schlägt bei Images fehl, deren /etc/group keinen `render`-Eintrag enthält) |
| GPU-Statistiken nicht verfügbar | NVIDIA: Block `deploy.resources.reservations` einkommentieren. AMD/Intel: sicherstellen, dass `/sys/class/drm:ro` gemountet ist |
| Falscher GPU-Hersteller erkannt | `GPU_TYPE=cuda`/`rocm`/`intel` setzen, um die Erkennung zu überschreiben |
| Instanz verbleibt im Bare-Metal-Betrieb auf **starting** | Der Container ist healthy, aber llamaman kann ihn nicht erreichen. `LLAMAMAN_IN_DOCKER=false`/`true` explizit setzen, wenn die automatische Erkennung für Ihre Laufzeitumgebung fehlschlägt |
| Strata: _"Docker image 'strata:latest' not found"_ | Strata-Image unter Settings → Docker Images pullen oder bauen (oder `docker build -t strata .` in einem Klon des Strata-Repos ausführen). Siehe [Strata](STRATA.md) |
| Strata-Modelle werden nicht aufgeführt | `STRATA_ENABLED=true` setzen; der Node muss NVIDIA sein (`GPU_TYPE=cuda`, falls die Erkennung fehlschlägt). `GET /api/engines` zeigt die Ursache an |
| Strata-Instanz verbleibt auf **starting** | Beim ersten Start für einige Minuten normal (Download + Pack). Die Karte zeigt die Phase; das Log zeigt das Setup von Strata. „setup failed“ bedeutet, dass das Setup abbrach – Zeile `[X]` im Log prüfen |
| Stats-Modal ist leer | **Settings → App Settings → Request recording** aktivieren (siehe [Request-Aufzeichnung & Statistiken](CONFIGURATION.md#request-recording--stats)) |
| Start schlägt mit GPU/CDI-Fehler auf einem Host ohne GPU-Passthrough fehl | Für reine CPU-Nutzung ohne GPU-Gerät **GPU Layers** auf `0` setzen oder die GPU-Laufzeitumgebung korrigieren |
| Port-Konflikt | Das Formular schlägt automatisch einen freien Port vor; bei Bedarf anpassen |
| Modell wird in OpenWebUI nicht angezeigt | Sicherstellen, dass `OLLAMA_BASE_URL=http://llamaman:42069` gesetzt ist. Prüfen, ob `/api/tags` Modelle zurückgibt |
| OpenWebUI erhält 401 | `require_auth` ist aktiv (Standard). API-Key erstellen und `OPENAI_API_KEYS` in OpenWebUI setzen |
| OpenAI-Client erhält 503 bei voller Modellauslastung | `LLAMAMAN_MAX_MODELS`-Limit erreicht und die OpenAI-API kann standardmäßig nicht verdrängen. Limit erhöhen oder die Eviction-Umschalter aktivieren: [Modell-Eviction](MODELS.md#model-eviction) |
| Container werden nach dem Stoppen nicht aufgeräumt | llamaMan entfernt Container beim Stoppen. Bei verwaisten Containern nach einem Absturz: `docker ps --filter name=llamaman-` oder llamaMan neustarten (Orphan-Adoption läuft beim Start) |
| Client (Hermes / OpenWebUI / etc.) meldet das trainierte Kontextfenster statt des Preset-Limits | Auf 1.1.2+ aktualisieren. `/api/ps` enthält nun `context_length` mit dem Runtime-ctx, mit dem die Instanz gestartet wurde, und `model_info["<arch>.context_length"]` von `/api/show` wird mit dem effektiven Limit überschrieben |
| Client, der nach einem Queue-Gruppennamen fragt, erhält seinen eigenen Standard-ctx (z. B. 128k) statt des realen Deployment-ctx | `/api/show` und `/v1/models` antworten nun für eine Queue-Gruppe auf jedem geclusterten Node – selbst ohne lokalen Member – unter Nutzung des Runtime-ctx des Peers aus dem geteilten Status. Multi-Member-Gruppen melden das Minimum aller aktiven Member, sodass der angekündigte Wert für jedes Dispatch-Ziel sicher ist. Siehe [Clustering](CLUSTERING.md) |
