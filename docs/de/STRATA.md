# Strata-Engine

*(opt-in, nur NVIDIA)* [Strata](https://github.com/Niko1221/Strata) betreibt die Mixture-of-Experts-Familie Qwen3.8-Flash-Next (und ihre Coder-, Swift-1.5- und Unsloth-Varianten) auf Consumer-NVIDIA-GPUs plus System-RAM. llamaMan kann Strata-Instanzen neben llama.cpp-Instanzen starten, überwachen, proxyen, schlafen/wecken und stoppen. Das Verhalten von llama.cpp ändert sich nicht: die Engine-Schicht (`core/engines/`) hält den llama.cpp-Launch byte-für-byte identisch (snapshot-getestet).

## Inhalt

- [Erste Schritte](#getting-started) · [Grenzen und Verhalten](#limits-and-behaviour) · [Design-Notizen und Verifikation](#design-notes-and-verification)

<a id="getting-started"></a>
## Erste Schritte

**1. Image besorgen** (upstream veröffentlicht keines; der Host braucht den NVIDIA-Treiber **580+**, also CUDA 13, und das NVIDIA Container Toolkit). Nach dem Aktivieren von Strata (Schritt 2) entweder ein vertrauenswürdiges Image per Namen unter **Settings → Docker Images** ziehen (Engine *Strata* in der Pull-Zeile), oder **Manage Strata images → Pull repo & build latest** nutzen, was das Repository nach `STRATA_SRC_DIR` herunterlädt und `STRATA_BUILD_IMAGE` über den Docker-Socket für die GPU-Generationen des Nodes baut (aus dem Treiber gelesen; alle unterstützten, falls das fehlschlägt), oder es von Hand bauen:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. Aktivieren** auf dem llamaMan-Node: `STRATA_ENABLED=true` (siehe [Strata-Variablen](CONFIGURATION.md#strata)). Settings → Docker Images zeigt dann *Manage Strata images*: die gezogenen Strata-Images, das gebaute (heruntergeladene und gebaute Commits, der Build-Button) und deren Auto-Update. Das Launch-Formular bietet Strata an, sobald eines davon auf dem Node ist.

**3. Launch.** Die Modellbibliothek erhält einen virtuellen Eintrag pro Familie × Größe: `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(experimentell)*. Jeder ist seine eigene Instanz und sein eigener Container, daher funktioniert das Modellwechseln wie bei llama.cpp: eine andere Instanz starten oder wecken, mit den üblichen [Eviction](MODELS.md#model-eviction)-Regeln (eine Strata-Instanz ist ein Chat-Modell). Die Einträge erscheinen auch in `/api/tags` und `/v1/models`, und Requests können die ID nutzen (`strata/qwen-IQ2_XS`, beliebige Groß-/Kleinschreibung, `:tag` ignoriert) oder Stratas eigenen Served Name (`qwen3.8-flash-next-iq2_xs`).

Das Launch-Formular verbirgt die nur für llama.cpp geltenden Felder (GPU-Layer, MoE, Threads, parallel, KV-Typen, Flash Attention, Spec-Decoding, mmproj, DRY, Extra Args) und zeigt **Image Input** (no / GPU / CPU), **KV Cache** (Strata-Default int8, `int8`, `q4_0`, `k8v4`) und **Low-RAM Mode** (auto / on / off). Die API lehnt nur für llama.cpp geltende Felder ab, die für ein Strata-Modell gesetzt sind.

**Erststart.** Strata lädt ~58-111 GB herunter, bereitet seinen „Pack“ vor und holt seinen MTP-Layer, bevor der Server seinen Port öffnet; spätere Starts gehen direkt zum Laden über (1-3 Minuten). Die Instanzkarte zeigt *setting up → downloading (mit %) → preparing pack → loading*, geparst aus dem Container-Log, und der Logs-Button streamt das Log. Zwei Wege zum Herunterladen:

- **Strata herunterladen lassen** in sein `/data`-Volume (Standard).
- **Mit llamaMan vorab herunterladen** (Button im Strata-Abschnitt, oder `POST /api/engines/strata/download`): ein gewöhnlicher Eintrag im Downloads-Tab (Fortschritt, Pause/Fortsetzen, Retry, HF-Tokens, Speed-Limits), der die Shards aus der von Strata gepinnten Hugging-Face-Revision nach `MODELS_DIR/strata/<tag>/` holt. Nach Abschluss werden sie per Bind-Mount in den Container eingebunden, und Stratas Setup nutzt sie, statt herunterzuladen. Ein Launch wird abgelehnt, während dieser Download noch läuft. Der MTP-Layer (~5 GB) und der Pack werden beim Erststart weiterhin von Strata vorbereitet.

<a id="limits-and-behaviour"></a>
## Grenzen und Verhalten

- **Nur ein Request gleichzeitig.** Max Concurrent ist auf 1 erzwungen; weitere Requests warten in llamaMans Gate (Max Queue Depth, dann 429).
- **RAM.** Strata lädt 32-62 GB in den RAM (`--ulimit memlock=-1` wird immer gesetzt). Ein **System Memory Limit** erzwingt Low-RAM-Modus, denn Stratas Setup liest den RAM des Hosts und kann ein Container-Limit nicht sehen; das UI warnt unterhalb von ~64 GB.
- **Einstellungen sind in Stratas Setup eingebrannt.** Context / Image Input / KV / Low-RAM / GPU-Pinning werden pro Modell auf dem Data-Volume aufgezeichnet. llamaMan merkt sich die Einstellungen, mit denen jedes Modell zuletzt healthy wurde, und übergibt `REINSTALL=1` nur, wenn sie sich ändern (die Dateien werden wiederverwendet).
- **Idle-Timeout** funktioniert (Sleep stoppt den Container; der nächste Request weckt ihn), aber ein Cold Start dauert Minuten, und der weckende Request wartet darauf (`STRATA_LOAD_TIMEOUT`).
- **Eine Instanz pro Strata-Modell pro Node** (Instanzen eines Modells teilen sich dessen Setup auf dem Volume).
- **Nur NVIDIA.** Die Option ist auf ROCm- / Intel- / Vulkan- / CPU-Nodes und wenn `STRATA_ENABLED` aus ist nicht verfügbar, mit Begründung. In einem Cluster meldet jeder Node seine Engines, daher bieten nur fähige Nodes Strata-Modelle an.
- Bare-Metal-llamaMan erreicht Strata auf demselben Weg wie llama-server (publizierter Port auf `LLAMA_HOST_ADDR`).

---

<a id="design-notes-and-verification"></a>
# Design-Notizen und Verifikation

Diese Seite beschreibt, wie die Integration funktioniert, wovon sie upstream abhängt, ihre Grenzen, und wie man sie auf einem echten NVIDIA-Host
verifiziert.

Upstream-Referenzen zeigen auf Strata-Commit `99f3dbd` (2026-10-03).

## Wie es sich einfügt

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- Die Engine einer Instanz ist `config["engine"]`. Ein fehlender Wert bedeutet llama.cpp, daher bleibt jede bestehende
  Instanz-Zeile, jedes Preset und jedes Container-Label bedeutungsgleich. llama.cpp-Konfigurationen werden niemals mit dem
  Schlüssel gestempelt, weil das das `llamaman.config`-Label des Containers ändern würde.
  `tests/fixtures/llamacpp_container_spec.json` enthält die `containers.run`-Kwargs vor dem Refactoring für 9 Konfigurationen × 5 GPU-Hersteller, und `tests/test_engines.py`
  stellt sicher, dass sie unverändert sind.
- Keine Speicher-Migration nötig. Presets und Instanzen sind in beiden Backends JSON-Blobs (`presets.json` /
  `state.json`, und die MariaDB-`data`-TEXT-Spalten), daher sind `engine` und `strata_*` additive Schlüssel.
- Jedes Strata-Modell ist ein virtueller Eintrag, `/strata/<family>-<SIZE>` (ID `strata/<family>-<SIZE>`), und ist seine
  eigene Instanz und sein eigener Container. Modellwechsel, Eviction, Sleep und Wake funktionieren wie bei GGUF-Dateien.
- `_run_container` nimmt den `container_spec()` der Engine und ergänzt die GPU-Geräte, die die Engine anfordert. Ein
  Strata-Container bekommt:

  | | |
  |---|---|
  | image | `STRATA_IMAGE` (sonst das gebaute, dann ein gezogenes Strata-Image) |
  | command | keine (der Entrypoint liest nur Umgebungsvariablen) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | Data-Volume → `/data`; Logs; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>` (`<family>-<SIZE>` bei nicht-qwen; setup.py's groß-/kleinschreibungs-sensitiver Shard-Ordner) (rw), wenn llamaMan vollständige Shards besitzt |
  | mem_limit | aus System Memory Limit (erzwingt `LOW_RAM=on`) |
  | network / ports / labels | wie bei llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

## Antworten auf die offenen Fragen

1. **Bereitschaft.** `/health` antwortet `{"status": "ok", ..., "service": "strata"}` vor dem API-Key-Gate
   (`serve/server.py:2172`). Der HTTP-Port öffnet erst, nachdem das Modell geladen ist: `serve()` läuft bei
   `server.py:3008`, nachdem die Engine bei `:2925` startet. Während Strata herunterlädt oder lädt, werden Verbindungen
   abgelehnt, daher braucht llamaMans bestehender Health-Check (`status == "ok"`) keinen Sonderfall. `/v1/models` liefert
   im Lazy-Modus eine leere Liste, solange das Modell nicht geladen ist (`server.py:2190`), taugt also nicht als
   Bereitschaftssignal. `/v1/status` braucht den API-Key, wenn einer gesetzt ist.
2. **Einstellungen.** Alles wird als Umgebungsvariable übergeben, die `docker-entrypoint.sh` liest: `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Kommandoargumente werden ignoriert,
   weil der Entrypoint `$@` neu aufbaut (`:42`, `:65`). GPU-Pinning nutzt Container-Nummerierung: Das NVIDIA-Runtime
   nummeriert die von llamaMan angebundenen Host-GPUs um, daher wird Host `1,3` zu `GPUS=0,1`. llamaMan setzt `API_KEY`
   nicht, was zu seiner Arbeitsweise bei llama-server passt (kein Key); sein eigener Proxy und seine Auth sitzen davor.
3. **GGUFs auf der Festplatte wiederverwenden.** `setup.py` hat `--gguf-dir` (`setup.py:2954`), aber der Entrypoint
   reicht es nie weiter. Setup akzeptiert jedoch vollständige Shards, die bereits unter `/data/models/<fam tag><SIZE>/<original name>` liegen (z. B. `IQ3_S`, `coder-IQ1_M`; groß-/kleinschreibungs-sensitiv): es prüft jedes
   gegen sein eigenes Tensor-Verzeichnis und markiert es als fertig (`setup.py:3389-3405`, „#173“). llamaMan bind-mountet
   daher seine Kopie dorthin (read-write, für die `.done`-Markierungen), und erst nachdem sein eigener Download dieser
   Shards abgeschlossen ist. Die Shards werden von den Hugging-Face-Revisionen geholt, die Strata pinnt (`setup.py:65-70`,
   gespiegelt in `core/engines/strata.py`). Der MTP-Layer (~5 GB) und der Pack werden weiterhin im Container vorbereitet.
4. **Memory-Limit / memlock.** Setup liest RAM aus `/proc/meminfo`, also den Host-Gesamt-RAM, daher muss ein gedeckelter
   Container um `LOW_RAM=on` bitten (Dockerfile-Kommentar bei `:31`, `docs/INSTALL.md:116-118`). Ohne es wählt Setup den
   In-RAM-Modus, und der Container wird wahrscheinlich OOM-gekillt. llamaMan erzwingt `LOW_RAM=on`, wann immer ein
   Memory-Limit gesetzt ist, und das UI warnt unter 64 GB. `memlock=-1` ist wichtig, weil die Engine einen Teil ihrer Arena
   für GPU-DMA seitenfixiert, und Experten, die sie nicht pinnen kann, über langsamere Host-Kopien laufen
   (`src/prefill/prefill.cpp:110`, `:207-214`). Aus dem Quellcode sieht ein fehlendes memlock-Limit wie eine Verlangsamung
   aus, nicht wie ein Fehler, aber **das konnte ich ohne GPU nicht verifizieren**. llamaMan setzt es immer.

Ebenfalls aus dem Quellcode: Strata bedient einen Request gleichzeitig (`server.py:1323`). Das Dockerfile setzt `STRATA_EXECV=1`,
daher ist der Server PID 1, und `docker stop` erreicht ihn. Ein mitten im Download gestoppter Container setzt beim nächsten
Start aus seinen `.part`-Dateien fort.

## Design-Entscheidungen

- **Nebenläufigkeit ist auf 1 erzwungen** beim Launch, beim Preset-Merge, bei Live-Preset-Änderungen und in gespeicherten
  Presets. Das Gate erzwingt den Sidecar-Proxy, daher passiert Queueing in llamaMan (`RequestGate`), nicht in Strata.
- **`REINSTALL` per Fingerprint.** Context, Vision, KV, Low-RAM und GPU-Pinning werden in Stratas Konfiguration pro Modell
  auf dem Volume aufgezeichnet. `DATA_DIR/strata_setups.json` merkt sich die Einstellungen, mit denen jedes Modell zuletzt healthy wurde,
  mit Data-Volume und Setup-Tag als Schlüssel, und `REINSTALL=1` wird nur gesendet, wenn sie abweichen. Ein unbekannter Fingerprint
  (Erststart über llamaMan, oder ein von Hand eingerichtetes Volume) installiert ebenfalls neu. Das kostet einen Setup-Durchgang, der die Dateien wiederverwendet.
- **Eine laufende Instanz pro Strata-Modell pro Node.** Zwei Instanzen eines Modells würden Setup über denselben Dateien ausführen.
- **Ein Launch wird abgelehnt, während llamaMan das Modell noch herunterlädt**, damit der Container es nicht ein zweites
  Mal in sein Volume herunterlädt.
- **Vorab heruntergeladene Shards überspringen die Update-Check-Herkunft.** Eine gepinnte Datei darf nicht auf das `main` des Repositories „aktualisiert“ werden.
- **Virtuelle Modelle werden nur aufgelistet, wo sie laufen können** (`STRATA_ENABLED` plus NVIDIA): in `/api/models`,
  `/api/tags`, `/v1/models` und der Namensauflösung. Update-Scans und der Cluster-Modell-Snapshot bleiben dateienbasiert. Peers
  erfahren Engine-Verfügbarkeit aus `snapshot.system.engines` (verschachtelt unter `system`, damit ältere Peers es ignorieren).
- **Ladestufe** kommt aus den letzten 64 KB des Container-Logs: Setups `=== Step N ===`-Header, die `\r`-Fortschrittszeile
  des Downloads und das „loading the model“ des Servers. Sie ist best-effort; schlägt sie fehl, wird einfach keine Stufenzeile angezeigt.
- **vLLM (#53)** wäre eine weitere `Engine`: eigenes Image, Kommando, Port (`internal_port`), Health-Semantik und
  Launch-Felder. Die Stellen, die davon ausgehen, dass beide Engines Port 8080 nutzen (die `resolve_llama_endpoint`-Aufrufer),
  sind die, die man dann parametrisieren müsste.

## Ohne GPU nicht verifiziert

- Ein echter Strata-Launch, Erststart-Download, Pack-Build, Laden und Chat. Alles oben wurde gegen den Upstream-Quellcode
  und mit Unit-Tests geprüft, die Docker mocken; kein Strata-Container ist gelaufen.
- Das exakte Log-Format bei einem echten Erststart. Der Stufen-Parser folgt `setup.py`'s `step()`- / `download()`-Ausgabe
  und den Startup-Ausgaben des Servers.
- Verhalten ohne `memlock=-1`, und der tatsächlich genutzte RAM unter `LOW_RAM=on` mit Container-Limit.
- Verschachtelter Bind-Mount (`/data/models/<tag>` über einem benannten Volume unter `/data`) auf den verwendeten
  Docker-Versionen. Docker unterstützt verschachtelte Mounts, aber das wurde hier nicht ausgeführt.

## Bekannte Einschränkungen

- Nur NVIDIA. Stratas AMD-(HIP-)Pfad ist upstream ein experimenteller manueller Build und wird nicht angeboten.
- `LAYER_SPLIT` (Layer-Platzierung über mehrere GPUs) ist nicht exponiert. Mehrere GPUs zu pinnen nutzt Stratas automatischen Split.
- Der Vision-Encoder (mmproj) wird immer von Strata selbst heruntergeladen, in das Volume.
- Stratas eigenes Web-UI (`/`, `/settings`, `/load`, `/unload`) ist über den Instanz-Port erreichbar, aber llamaMan
  nutzt und verwaltet es nicht. Dessen Idle-Unload-Funktion ist ungenutzt; llamaMans Sleep stoppt stattdessen den Container.
- Eine Änderung an einer im Setup aufgezeichneten Einstellung führt Stratas Setup-Durchgang beim nächsten Start erneut aus (die Dateien werden wiederverwendet).
- Ein Erststart kann länger dauern als das eigene Timeout eines HTTP-Clients (Open WebUI und andere). llamaMan wartet weiter
  bis zu `STRATA_LOAD_TIMEOUT`, aber der Client gibt vielleicht zuerst auf. Vorab herunterladen, oder einmal aus dem UI starten, bevor Sie es Clients bereitstellen.
- Der Katalog (Familien, Größen, gepinnte Revisionen, Dateinamen) ist ein Spiegel von Stratas `setup.py`. Eine neue
  Upstream-Familie oder -Größe braucht einen passenden Edit in `core/engines/strata.py`.

## Checkliste zur manuellen Verifikation (NVIDIA-Host)

Voraussetzungen: ein NVIDIA-Treiber 580+, ein funktionierendes `docker run --gpus all`, empfohlene 64 GB RAM und etwa 150 GB
freier Festplattenspeicher.

1. **Build:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Optional
   `--build-arg CUDA_ARCHITECTURES=<your arch>` ergänzen. Oder, nach Schritt 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** Es baut nur für die GPU-Generationen dieses Nodes.
2. **Aktivieren:** `STRATA_ENABLED=true` setzen und llamaMan neu starten.
   - `GET /api/engines` zeigt strata `available: true`, mit seinen Modellen (9 in der eingebauten Liste).
   - Settings → Docker Images → *Manage Strata images* zeigt `strata:latest` als **local**.
3. **Gating:** Auf einem Nicht-NVIDIA-Node (oder mit `GPU_TYPE=rocm`) werden Strata-Modelle nicht gelistet, und
   `POST /api/instances` für `/strata/qwen-IQ2_XS` liefert 400 mit der NVIDIA-Begründung.
4. **Aus dem UI starten:** `strata/qwen-IQ2_XS` auswählen.
   - Die nur für llama.cpp geltenden Felder sind verborgen, der Strata-Abschnitt wird angezeigt, und Max Concurrent ist auf 1 fixiert.
   - Ein 32g-Memory-Limit setzen und prüfen, dass die Warnung unter 64 GB erscheint; wieder löschen.
   - Launch.
5. **Erstlauf (Container-Download):**
   - Die Instanzkarte zeigt *setting up → downloading NN% → preparing pack → loading*.
   - Der Logs-Button streamt Stratas Setup-Ausgabe.
   - Die Karte wird **healthy**, und `docker inspect` zeigt `Ulimits: memlock -1`, die Umgebungsvariablen und das `/data`-
     Volume.
6. **Chat:**
   - Über den Ollama-Proxy: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Über `/v1/chat/completions` auf `:42069`, und direkt auf dem Instanz-Port.
   - Auch `model: "qwen3.8-flash-next-iq2_xs"` probieren.
   - Zwei Requests gleichzeitig senden: der zweite steht in der Warteschlange (die Karte zeigt 1/1 aktiv · 1 queued).
7. **Idle-Sleep und Wake:**
   - Idle Timeout auf 1 Minute setzen (die Cold-Start-Warnung erscheint). Nach einer Minute ist die Instanz **sleeping**, und ihr
     Container ist weg.
   - Der nächste Request weckt sie (Minuten) und wird beantwortet.
   - Kein Setup-Durchgang im Log (keine `=== Step`-Zeilen; `REINSTALL` fehlt in `docker inspect`).
8. **Einstellungsänderung:**
   - Den Context auf 65536 ändern und neu starten.
   - `docker inspect` zeigt `REINSTALL=1`, und das Log zeigt einen Setup-Durchgang ohne Download.
   - Beim nächsten Neustart fehlt `REINSTALL` wieder.
9. **Vorab-Download-Pfad:**
   - Ein anderes Modell wählen (z. B. `strata/qwen-Q2_0`) und *Pre-download with llamaMan* klicken.
   - Der Downloads-Tab zeigt den Download der gepinnten Revision.
   - Das Starten während des Downloads wird abgelehnt.
   - Nach Abschluss launchen: `docker inspect` zeigt `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`, und das
     Log zeigt keinen Shard-Download (nur der MTP-Layer und der Pack).
10. **Stop / Remove:** Instanz stoppen. Der Container wird entfernt, und Remove räumt die Karte.
11. **llamaMan mit laufender Strata-Instanz neu starten:**
    - Sie wird wieder angebunden (State-Restore), oder adoptiert, falls der State verloren war: sie aus `state.json` löschen und neu starten.
    - Die Karte zeigt das Strata-Badge und `strata/qwen-IQ2_XS`.
    - Requests funktionieren weiterhin.
12. **Bare-Metal-Modus:**
    - llamaMan mit `python app.py` (`LLAMAMAN_IN_DOCKER=false`) laufen lassen und 4-6 wiederholen.
    - llamaMan erreicht den publizierten Port auf `LLAMA_HOST_ADDR`.
13. **Cluster:**
    - Einen Nicht-NVIDIA-Peer hinzufügen. Strata-Modelle erscheinen für den NVIDIA-Node nur, wenn er der Target-Node ist.
    - Das Starten auf dem NVIDIA-Peer aus dem UI des anderen Nodes funktioniert.
14. **llama.cpp unberührt:** Ein GGUF-Modell wie bisher starten.
    - Die Kommandozeile (`docker inspect` Args) stimmt mit einer vor dem Upgrade überein.
    - Vor dem Upgrade gespeicherte Presets laden und starten unverändert.
