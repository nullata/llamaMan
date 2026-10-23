# Launch-Einstellungen

Jedes Feld im Launch-Formular, worauf es sich abbildet, und die Verhaltensnotizen. Für die Vier-Schritte-Anleitung „Wie starte ich“ siehe die [README](../../README.md#your-first-model).

## Inhalt

- [Launch-Formular-Werkzeuge](#launch-form-tools) · [Referenz der Launch-Einstellungen](#launch-settings-reference) · [Live-Preset-Updates](#live-preset-updates) · [Nebenläufigkeit & Queueing](#concurrency--queueing) · [Bild- & PDF-Input](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Proxy pro Instanz](#per-instance-proxy) · [Idle-Timeout](#idle-timeout)

Die Verhaltensnotizen unten sind knapp gehalten; fahren Sie mit dem Mauszeiger über den Info-Tipp des Feldes im UI für die vollständige Erklärung. Alle Flags werden auf llama-server gemappt, sofern nicht anders angegeben - siehe die [llama-server CLI reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) für exakte Semantik und aktuelle Defaults.

<a id="launch-form-tools"></a>
## Launch-Formular-Werkzeuge

Die Aktionszeile des Launch-Formulars trägt rechts drei Werkzeug-Buttons:

- **Export preset** - die aktuellen Formularwerte als JSON-Datei herunterladen (portabel über Nodes hinweg; speichert nicht im Storage).
- **Import preset** - ein Preset-JSON in das Formular laden. Befüllt nur die Felder; Speichern müssen Sie weiterhin mit Save Preset.
- **Copy llama.cpp command** - kopiert den äquivalenten `llama-server ...`-Aufruf für die aktuellen Formularwerte in die Zwischenablage. Serverseitig durch denselben Flag-Emissions-Code gerendert, den auch der echte Launch nutzt, bleibt es bei Flag-Änderungen synchron.

Mit eingeklappten Einstellungen startet ein **Quick Launch**-Button das ausgewählte Modell direkt aus seinem Preset.

Jede Instanz stellt eine OpenAI-kompatible API auf ihrem zugewiesenen Port bereit. Wenn ein GGUF ausgewählt ist, liest llamaMan seine Metadaten, um die Layer-Anzahl zu erkennen, und zeigt sie neben **GPU Layers** an (z. B. `/ 32`).

<a id="launch-settings-reference"></a>
## Referenz der Launch-Einstellungen

**Model Settings**

| Einstellung | Standard | Flag | Notizen |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Prompt + Response-Token-Obergrenze pro Request |
| **Parallel Slots** | `1` | `--parallel` | Gleichzeitige Decoding-Slots innerhalb von llama-server (KV-Slots) |
| **Batch Size** | `2048` | `--batch-size` | Maximale Anzahl Prompt-Tokens, die in einem Schritt verarbeitet werden. Leer lässt das Flag weg |
| **Micro-batch Size** | `512` | `--ubatch-size` | Physischer Batch an die GPU; größer = schnelleres Prefill, mehr VRAM. Embedding- und Encoder-Modelle (BERT-Stil) brauchen ihre gesamte Eingabe in einem Micro-batch. Leer lässt das Flag weg |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`. Quantisiert reduziert K-Speicher; funktioniert mit oder ohne Flash Attention |
| **V Cache Type** | `f16` | `--cache-type-v` | Dieselben Werte. Jeder quantisierte Wert **erfordert Flash Attention = On** - das UI graut das Dropdown aus und setzt V automatisch auf `f16` zurück, wenn FA nicht mehr On ist |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` lässt das Flag weg (llama.cpps eigener Default); `on` erzwingt es (schlägt fehl, wenn das Backend es nicht kann); `off` deaktiviert |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`. Steuert, wie Thinking-Tags herausgeparst werden (nach `reasoning_content` geroutet vs. inline belassen). `auto` lässt das Flag weg |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. Wie Modellgewichte in den Speicher geladen werden (Nachfolger der veralteten `--mlock`/`--mmap`/`--direct-io`). `auto` = mmap, außer nicht unterstützt, entspricht llama.cpps Default und lässt das Flag weg |

**Container & Proxy Settings**

| Einstellung | Standard | Flag | Notizen |
|---|---|---|---|
| **Port** | *(auto)* | - | Öffentlicher Port; muss in `PORT_RANGE_START/END` fallen |
| **CPU Threads** | *(auto)* | `--threads` | Setzt außerdem das CPU-Kontingent des Containers (`--cpus N`). Leer = Container und llama-server alle Kerne nutzen lassen |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | Separate Thread-Anzahl für Prefill/Batch. Leer fällt in llama-server auf `--threads` zurück |
| **System Memory Limit** | *(keins)* | - | Docker-RAM-Obergrenze (z. B. `32g`, `8192m`) |
| **Idle Timeout min** | `0` | - | Auto-Sleep nach N Minuten Inaktivität. `0` = aus. Live aus dem Preset aktualisiert (kein Relaunch). Siehe [Idle-Timeout](#idle-timeout) |
| **Max Concurrent** | `0` | - | Maximale In-Flight-Inferenz-Requests; der Überschuss steht in der Warteschlange. `0` = kein Gating. Siehe [Nebenläufigkeit & Queueing](#concurrency--queueing) |
| **Max Queue Depth** | `200` | - | Queue-Obergrenze vor HTTP 429 |
| **Exclude from Max Models** | aus | - | Zählt nicht zu `LLAMAMAN_MAX_MODELS`, wird nie verdrängt. Siehe [Modell-Eviction](MODELS.md#model-eviction). Live aus dem Preset aktualisiert |

**GPU Settings**

| Einstellung | Standard | Flag | Notizen |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (Aliase) | `-1` = **auto** (llama.cpp schätzt VRAM und wählt; kann Layer auf CPU belassen). `-2` = **all** (jeden Layer erzwingen; schlägt fehl, wenn es nicht passt). `0` = nur CPU, keine GPU am Container. Jedes positive N lagert exakt N Layer von unten aus |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | Nur bei Mixture-of-Experts-Modellen sinnvoll. `0` = gar nichts emittieren. `-1` = `--cpu-moe` (geroutete Experten-Gewichte aller Layer auf CPU fixiert). Positives N = `--n-cpu-moe N` (Experten der ersten N Layer auf CPU). Schrumpft VRAM bei großen MoEs um ~4-10x, weil Experten die Gewichtsanzahl dominieren, aber pro Token nur wenige feuern. Wirkungslos bei dichten Modellen: llama.cpps Regex (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`) matcht nichts, und der Override wird stillschweigend verworfen. **Deckt keine Shared Experts ab** (`_shexp` bei DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE) - dafür in Extra Args einen `-ot`-Override ergänzen |
| **GPU Devices** | *(alle)* | *(Container-Ebene)* | Kommagetrennte Host-GPU-Indizes zum Anbinden (z. B. `0,1`). Innerhalb des Containers werden sie ab 0 neu nummeriert. Nicht unterstützt auf Intel Arc |
| **Split Mode** | `layer` | `--split-mode` | `none` (eine GPU, ignoriert Tensor Split) / `layer` (llama.cpps Default; splittet ganze Layer) / `row` (splittet Tensor-Zeilen; typischerweise nur mit schnellem Interconnect wie NVLink von Vorteil) |
| **Tensor Split** | *(auto)* | `--tensor-split` | Kommagetrennte relative Gewichte pro container-sichtbarer GPU (z. B. `24,16`). Leer wird beim Start automatisch aus dem Gesamt-VRAM jeder GPU befüllt. Ignoriert, wenn Split Mode = `none` oder nur eine GPU sichtbar ist |

**Extra Args** *(Textfeld)* - zusätzliche Flags, die direkt an llama-server übergeben werden (z. B. `--mlock`, oder n-Gramm-Speculative-Typen, die im Formular nicht auftauchen).

**Proxy-Side Sampling Overrides** *(Umschalter)* - wenn an, erzwingt der Proxy Sampling-Parameter auf jedem weitergeleiteten Request:

| Einstellung | Standard | Bereich |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0 (`0` deaktiviert) |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 bis 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0 (`0` = deaktiviert) |

**Speculative Decoding** *(Umschalter)*

| Einstellung | Standard | Flag | Notizen |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`. Nur `draft-mtp` akzeptiert einen leeren Drafter (fällt auf die eingebauten MTP-Heads des Hauptmodells zurück) |
| Draft Model | *(keins)* | `-md` | Pfad zum Drafter-GGUF. Für alle Typen außer `draft-mtp` erforderlich |
| Draft N Max | `2` | `--spec-draft-n-max` | Maximale Tokens, die pro Schritt drafted werden |
| Draft N Min *(Advanced)* | *(auto)* | `--spec-draft-n-min` | Leer lässt das Flag weg |
| Draft P Split *(Advanced)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`. Leer lässt das Flag weg |
| Draft P Min *(Advanced)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`. Leer lässt das Flag weg. `0` ist ein echter Wert, verschieden von leer |

**Image & PDF Input** *(Umschalter)* - siehe [Bild- & PDF-Input](#image--pdf-input) unten.

**Anti-Loop** — zwei unabhängige Unter-Umschalter (beide standardmäßig aus). Siehe [Anti-Loop](#anti-loop) unten.

<a id="live-preset-updates"></a>
## Live-Preset-Updates

Ein Preset zu speichern aktualisiert bereits laufende Instanzen dieses Modells nach Möglichkeit an Ort und Stelle:

- **Live (kein Relaunch):** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, alle sechs Proxy-Sampling-Felder, alle fünf Loop-Detection-Felder (der nächste Request übernimmt sie)
- **Erfordern Relaunch:** alles, was beim Start in den Container eingebrannt wird (gpu layers, MoE CPU layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, DRY-Sampler, Spec-Decoding-Felder, Embedding-Flag, mmproj/PDF-Felder, extra args, memory limit)

**Proxy-Sampling-Einschränkung:** Wenn die Instanz mit `idle_timeout=0`, `max_concurrent=0` und `override_enabled=false` gestartet wurde, wurde kein Sidecar-Proxy gespawnt. Live auf `override_enabled=true` zu schalten wendet die Overrides dann weiterhin auf Requests an, die über die Compat-Endpoints der App laufen, aber direkte Zugriffe auf den öffentlichen Port umgehen ihn. In diesem Fall relaunchen, um den Proxy zu spawnen.

<a id="concurrency--queueing"></a>
## Nebenläufigkeit & Queueing

Wenn **Max Concurrent** > 0, setzt llamaMan ein FIFO-Gate vor die Instanz. Überschüssige Requests warten bis zu **Max Queue Depth**, bevor HTTP 429 zurückgegeben wird. Gate-Stats (aktiv + wartend) sind über die API sichtbar.

**Parallel vs Max Concurrent:** `Parallel` steuert llama-server-interne KV-Slots; `Max Concurrent` begrenzt, wie viele Requests llamaMan gleichzeitig weiterleitet. Beides zusammen verwenden (z. B. `Parallel=4`, `Max Concurrent=4`), um den Server gut zu versorgen, ohne seine Slots zu überfüllen.

<a id="image--pdf-input"></a>
## Bild- & PDF-Input

Aktivieren Sie den **Image & PDF Input**-Umschalter, um den multimodalen Projektor eines Vision-Modells (`--mmproj`) neben seinem Haupt-GGUF zu laden. Das Feld **MMPROJ Model** nimmt das Projektor-GGUF (typischer Dateiname: `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (standardmäßig an): llama.cpp hält den Projektor auf der GPU, außer er bekommt `--no-mmproj-offload`, daher lässt *an* das Flag komplett weg (byte-identische CLI wie vor Einführung dieser Option) und *aus* übergibt `--no-mmproj-offload`, um den Vision-Encoder auf CPU laufen zu lassen — nützlich bei knappen VRAM-Setups, wo der Encoder einige GB festhält.

**PDFs** werden unterstützt, indem jede Seite zu einem PNG gerastert wird, **bevor** sie an llama-server weitergeleitet wird (das keine native PDF-Unterstützung hat). Aktivieren Sie **Accept PDF uploads** (innerhalb der Image-&-PDF-Input-Gruppe), um das pro Instanz einzuschalten; der Master-Umschalter **Image & PDF Input** steuert die gesamte Gruppe — ihn abzuschalten löscht auch die PDF-Unter-Umschalter, da gerasterte Seiten ohne Projektor kein Vision-Modell erreichen. In beiden erkannt:

- **OpenAI**: `image_url` mit einer `data:application/pdf;base64,...`-URL, oder der neuere `file`-Block
- **Ollama**: Base64-PDF-Bytes gemischt in `images[]` - llamaMan erkennt `%PDF-` und expandiert PDFs; gewöhnliche Bilder behalten ihren bestehenden Pfad

**Text-Layer-Abkürzung** (standardmäßig aus): Mit **Try text layer first** an führt llamaMan `pypdf.extract_text()` vor dem Rastern aus. Digital geborene PDFs (Word/LaTeX/Browser-Druck) mit wesentlichem Text (≥ ~50 Zeichen/Seite) werden als ein Textblock inline eingefügt und umgehen den Vision-Pfad komplett. PDFs mit wenig Text fallen aufs Rastern zurück. Standardmäßig aus, weil das Rastern das Layout für das Vision-Modell erhält, und vorhersagbares Verhalten bei jedem PDF besser ist als eine stille Abkürzung.

**Obergrenzen pro Request:** `PDF DPI` (72-600, Standard 200) und `Max PDF Pages` (1-200, Standard 20). PDFs über der Grenze liefern HTTP 400, statt gekürzt zu werden.

**Rasterisierungs-Nebenläufigkeit:** PDF-Rendering ist CPU- und RAM-hungrig und läuft im gunicorn-Worker-Thread vor jedem Instanz-Gate. Ein prozessweites Semaphore begrenzt gleichzeitige Rasterungen; abstimmbar über `LLAMAMAN_PDF_MAX_CONCURRENT` (Standard 4).

## Anti-Loop

Zwei unabhängige Regler im Abschnitt **Anti-Loop** des Launch-Formulars, beide standardmäßig aus. Sie zielen auf dasselbe Problem (Modelle, die in stabilen Output-Schleifen hängen bleiben) an zwei verschiedenen Stellen der Pipeline; zusammen verwenden für die beste Abdeckung.

**DRY-Sampler (weich, zur Sampling-Zeit).** Mappt auf llama-servers `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n`. Jedes Token, das ein kürzlich gesehenes n-Gramm verlängern würde, bekommt zur Sampling-Zeit eine Wahrscheinlichkeits-Strafe, sodass das Modell weg von Wiederholungen geschubst wird, bevor es überhaupt eine doppelte Spanne emittiert. Dies ist die *weiche* Verteidigungslinie — für Prosa üblicherweise allein ausreichend. Beim Container-Start eingebrannt; eine Preset-Änderung erfordert einen Relaunch. Multiplier = 0 deaktiviert vollständig (llama.cpps eigener Default).

*DRY Sampler* (beim Start eingebrannt — erfordert Relaunch):

| Einstellung | Standard | Flag | Notizen |
|---|---|---|---|
| Multiplier | `0.8` (UI-Wert; `0` deaktiviert) | `--dry-multiplier` | Strafstärke; llama.cpp behandelt `0` als deaktiviert |
| Base | `1.75` | `--dry-base` | Exponentialbasis ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | n-Gramme bis zu dieser Länge sind ausgenommen |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Rückblickfenster. Leer lässt das Flag weg |

**Output-Loop-Erkennung (hart, nachträglich).** Wenn DRY nicht reicht — das Modell hängt weiterhin in einem stabilen Attraktor — beobachtet llamaMans Proxy den für den Client sichtbaren Assistant-Output-Text (content + reasoning), während er zum Client streamt. Jeden Scan-Takt (Standard: inline alle 64 Chunks, plus ein 10s-Worker-Fallback) werden die letzten `min_chunk_chars` des Puffers auf Wiederholung geprüft. Erscheinen sie ≥ `min_repetitions`-mal, wird der Turn beendet:

- **Die Upstream-Verbindung zu llama-server wird geschlossen** — llama-servers Stop-on-Client-Disconnect stoppt die Generierung und gibt das Modell für den nächsten Request frei.
- **Ein synthetischer Terminator wird in den Client-Stream injiziert** — OpenAI-Clients sehen einen letzten Delta mit `finish_reason=stop` und content `[llamaman: output loop detected, ending turn]`; Ollama-Clients sehen `done_reason: "loop_detected"` auf einer `done: true`-Zeile.

*Output Loop Detection* (Proxy-Seite; live aus dem Preset aktualisiert — der nächste Request übernimmt neue Schwellen):

| Einstellung | Standard | Notizen |
|---|---|---|
| Min chunk chars | `200` | Bereich 60-4096. Niedriger fängt kürzere Schleifen, produziert aber False Positives bei repetitiv-legitimen Inhalten |
| Min repetitions | `3` | Bereich 2-20 |
| Buffer size (chars) | `8192` | Rollierendes Fenster pro In-Flight-Request. Bereich 512-65536 |
| Scan every N chunks | `64` | Inline-Scan-Takt |
| Fallback scan (seconds) | `10` | Hintergrund-Worker für Streams, die Tokens zu langsam emittieren, um die Inline-Schwelle zu erreichen |

**Wo beobachtet wird:**
- **Compat-Routen auf `:42069`** (Ollama `/api/chat`, `/api/generate`; OpenAI `/v1/chat/completions`, `/v1/completions`) — immer, wenn der Umschalter für das Preset der Zielinstanz an ist.
- **Instanz-Ports (`8000-8020`)** — nur wenn für diese Instanz bereits ein Sidecar-Proxy existiert (d. h. Idle Timeout, Max Concurrent oder Proxy Sampling Overrides ebenfalls an). Loop-Erkennung spawnt für sich allein keinen Sidecar-Proxy — einen der anderen drei aktivieren, um auch direkte Zugriffe zu beobachten.

**Abstimmung:**
- `min_chunk_chars` (60-4096, Standard 200) — die kleinste wiederholende Periode, die auslöst. Niedrigere Werte fangen kürzere Schleifen, riskieren aber False Positives bei repetitiv-legitimen Inhalten (nummerierte Listen, Gedicht-Refrains, Markdown-Tabellen). 60-100 ist aggressiv; 200 ist der ausgewogene Default für Prosa; 400+ löst nur bei Schleifen auf Absatz-Niveau aus.
- `min_repetitions` (2-20, Standard 3) — wie viele exakte Kopien des Chunks erscheinen müssen, bevor beendet wird. 2 feuert auf jeden verdoppelten Chunk (aggressiv); 3-5 balancieren Reaktionsstärke gegen False Positives.
- `max_buffer_chars` (512-65536, Standard 8192) — rollierendes Fenster pro aktiver Anfrage. Muss mindestens `min_chunk_chars × min_repetitions` sein.
- `scan_every_n_tokens` (8-4096, Standard 64) — Inline-Scan-Takt. Niedriger = schnellerer Kill, mehr CPU.
- `scan_interval_s` (1-600, Standard 10) — wie oft der Hintergrund-Worker Puffer nachscannt, die die Inline-Schwelle nicht erreicht haben.

Änderungen an den Loop-Detection-Schwellen wirken beim *nächsten* Request (In-Flight-Streams behalten die beim Andocken erfassten Schwellen). DRY-Änderungen erfordern einen Relaunch, da die Flags in den Container eingebrannt sind.

**Erkennungsalgorithmus** (v1): die letzten `min_chunk_chars` des Puffers als Ziel nehmen; exakte Vorkommen zählen; markieren, wenn ≥ `min_repetitions`. Fängt driftende Near-Repeats nicht („The answer is: The answer is: The answer is:“ mit leichter Zeichensetzungsdrift zwischen Wiederholungen) — die werden von DRYs Sampling-Zeit-Verteidigung gefangen. Bewusst einfach und billig.

**Defensive Invariante:** Jeder Hook im Streaming-Pfad ist in ein `try/except Exception` gewickelt, das loggt und durchfällt — ein Detector-Bug kann den Stream eines echten Nutzers niemals brechen.

<a id="per-instance-proxy"></a>
## Proxy pro Instanz

Wenn **Idle Timeout**, **Max Concurrent** oder **Proxy Sampling Overrides** für eine Instanz an sind, schiebt llamaMan einen WSGI-Proxy vor den llama-server-Container: Der öffentliche Port (z. B. 8000) ist der Proxy; llama-server lauscht auf einem separaten internen Port. Der Proxy macht Modellnamen-Validierung, Wake-on-Request für schlafende Instanzen, `RequestGate`-Nebenläufigkeitsbegrenzung und Sampling-Injektion.

**Modellnamen-Validierung:** Auf Inferenz-Endpoints wird ein `"model"`-Feld im Request-Body, falls vorhanden, gegen den Dateinamen-Stem des geladenen Modells verglichen (lowercased, ohne Endung). **Präfix-Match wird akzeptiert** (z. B. matcht `"qwen2.5-0.5b-instruct-q2"` auf `"qwen2.5-0.5b-instruct-q2_k"`). Mismatch liefert HTTP 404; für schlafende Instanzen kein Wake. Requests ohne `"model"`-Feld werden bedingungslos weitergeleitet.

<a id="idle-timeout"></a>
## Idle-Timeout

Setzen Sie **Idle Timeout min** im Launch-Formular (0 = deaktiviert). Wenn aktiviert, wird der llama-server-Container nach N Minuten ohne Requests gestoppt, um VRAM freizugeben; der nächste Request weckt einen neuen Container mit derselben Konfiguration. Clients sehen denselben Port mit einer Cold-Start-Verzögerung.

Für proxy-verwaltete (OpenWebUI-)Instanzen stattdessen `LLAMAMAN_IDLE_TIMEOUT` verwenden.
