# Paramètres de lancement

Chaque champ du formulaire de lancement, ce à quoi il correspond et les notes de comportement. Pour le parcours « comment lancer » en quatre étapes, voir le [README](../../README.md#your-first-model).

## Sommaire

- [Outils du formulaire de lancement](#launch-form-tools) · [Référence des paramètres de lancement](#launch-settings-reference) · [Mises à jour en direct des préréglages](#live-preset-updates) · [Concurrence et mise en file](#concurrency--queueing) · [Entrée image et PDF](#image--pdf-input) · [Anti-Loop](#anti-loop) · [Proxy par instance](#per-instance-proxy) · [Délai d'inactivité (Idle Timeout)](#idle-timeout)

Les notes de comportement ci-dessous sont brèves ; survolez l'info-bulle du champ dans l'UI pour l'explication complète. Tous les flags correspondent à llama-server sauf indication contraire - voir la [référence CLI de llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) pour la sémantique exacte et les valeurs par défaut actuelles.

<a id="launch-form-tools"></a>
## Outils du formulaire de lancement

La ligne d'actions du formulaire de lancement porte trois boutons d'outil à droite :

- **Export preset** - télécharge les valeurs actuelles du formulaire en fichier JSON (portable entre nodes ; ne sauvegarde pas dans le stockage).
- **Import preset** - charge un préréglage JSON dans le formulaire. Ne remplit que les champs ; vous devez encore appuyer sur **Save Preset** pour persister.
- **Copy llama.cpp command** - copie l'invocation `llama-server ...` équivalente pour les valeurs actuelles du formulaire dans votre presse-papiers. Rendue côté serveur par le même code d'émission de flags que le vrai lancement utilise, donc elle reste synchronisée à mesure que les flags évoluent.

Avec les paramètres repliés, un bouton **Quick Launch** démarre le modèle sélectionné directement depuis son préréglage.

Chaque instance expose une API compatible OpenAI sur son port attribué. Quand un GGUF est sélectionné, llamaMan lit ses métadonnées pour détecter le nombre de couches et l'affiche à côté de **GPU Layers** (p. ex. `/ 32`).

<a id="launch-settings-reference"></a>
## Référence des paramètres de lancement

**Model Settings**

| Réglage | Défaut | Flag | Notes |
|---|---|---|---|
| **Context Size** | `4096` | `--ctx-size` | Plafond de tokens prompt + réponse par requête |
| **Parallel Slots** | `1` | `--parallel` | Slots de décodage concurrents dans llama-server (slots KV) |
| **Batch Size** | `2048` | `--batch-size` | Nombre maximal de tokens de prompt traités en une étape. Vide omet le flag |
| **Micro-batch Size** | `512` | `--ubatch-size` | Batch physique envoyé au GPU ; plus grand = prefill plus rapide, plus de VRAM. Les modèles d'embedding et les modèles de décision encodeurs (type BERT) ont besoin de toute leur entrée dans un seul micro-batch. Vide omet le flag |
| **K Cache Type** | `f16` | `--cache-type-k` | `f16`/`f32`/`bf16`/`q8_0`/`q5_1`/`q5_0`/`iq4_nl`/`q4_1`/`q4_0`. La quantification réduit la mémoire K ; fonctionne avec ou sans Flash Attention |
| **V Cache Type** | `f16` | `--cache-type-v` | Mêmes valeurs. Toute valeur quantifiée **exige Flash Attention = On** - l'UI grise le menu déroulant et remet V à `f16` automatiquement quand FA quitte On |
| **Flash Attention** | `auto` | `--flash-attn on\|off\|auto` | `auto` omet le flag (le propre défaut de llama.cpp) ; `on` le force (échoue si le backend ne peut pas) ; `off` le désactive |
| **Reasoning Format** | `auto` | `--reasoning-format` | `none` / `auto` / `deepseek` / `deepseek-legacy`. Contrôle comment les balises de réflexion sont extraites (routées vers `reasoning_content` vs laissées en ligne). `auto` omet le flag |
| **Load Mode** | `auto` | `--load-mode` | `auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. Comment les poids du modèle sont chargés en mémoire (successeur des `--mlock`/`--mmap`/`--direct-io` dépréciés). `auto` = mmap sauf si non pris en charge, correspond au défaut de llama.cpp et omet le flag |

**Container & Proxy Settings**

| Réglage | Défaut | Flag | Notes |
|---|---|---|---|
| **Port** | *(auto)* | - | Port public ; doit tomber dans `PORT_RANGE_START/END` |
| **CPU Threads** | *(auto)* | `--threads` | Définit aussi le quota CPU du conteneur (`--cpus N`). Vide = laisser le conteneur et llama-server utiliser tous les cœurs |
| **CPU Threads Batch** | *(= CPU Threads)* | `--threads-batch` | Nombre de threads séparé pour le prefill/batch. Vide se rabat sur `--threads` dans llama-server |
| **System Memory Limit** | *(aucune)* | - | Plafond RAM Docker (p. ex. `32g`, `8192m`) |
| **Idle Timeout min** | `0` | - | Auto-sommeil après N minutes d'inactivité. `0` = désactivé. Mis à jour en direct depuis le préréglage (pas de relancement). Voir [Délai d'inactivité (Idle Timeout)](#idle-timeout) |
| **Max Concurrent** | `0` | - | Max de requêtes d'inférence en vol ; l'excédent est mis en file. `0` = pas de gating. Voir [Concurrence et mise en file](#concurrency--queueing) |
| **Max Queue Depth** | `200` | - | Plafond de la file avant un HTTP 429 |
| **Exclude from Max Models** | off | - | Ne compte pas dans `LLAMAMAN_MAX_MODELS`, jamais évincé. Voir [Éviction de modèles](MODELS.md#model-eviction). Mis à jour en direct depuis le préréglage |

**GPU Settings**

| Réglage | Défaut | Flag | Notes |
|---|---|---|---|
| **GPU Layers** | `-1` | `--n-gpu-layers` / `--gpu-layers` / `-ngl` (alias) | `-1` = **auto** (llama.cpp estime la VRAM et choisit ; peut laisser des couches sur CPU). `-2` = **all** (forcer chaque couche ; échoue si ça ne tient pas). `0` = CPU seulement, aucun GPU attaché au conteneur. Tout N positif décharge exactement N couches depuis le bas |
| **MoE CPU Layers** | `0` | `--cpu-moe` / `--n-cpu-moe` | N'a de sens que sur les modèles Mixture-of-Experts. `0` = ne rien émettre. `-1` = `--cpu-moe` (les poids experts routés de toutes les couches épinglés sur CPU). N positif = `--n-cpu-moe N` (les experts des N premières couches sur CPU). Réduit la VRAM d'environ 4 à 10 fois sur les gros MoEs car les experts dominent le compte de poids mais seuls quelques-uns s'activent par token. Inerte sur les modèles denses : la regex de llama.cpp (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`) ne correspond à rien et l'override est silencieusement abandonné. **Ne couvre pas les experts partagés** (`_shexp` dans DeepSeek-MoE / Qwen2-MoE / Hunyuan-MoE) - ajoutez un override `-ot` dans Extra Args pour ceux-là |
| **GPU Devices** | *(tous)* | *(niveau conteneur)* | Indices de GPU hôtes séparés par virgules à attacher (p. ex. `0,1`). Dans le conteneur ils sont renumérotés depuis 0. Non pris en charge sur Intel Arc |
| **Split Mode** | `layer` | `--split-mode` | `none` (GPU unique, ignore Tensor Split) / `layer` (le défaut de llama.cpp ; découpe des couches entières) / `row` (découpe des lignes de tenseur ; ne gagne typiquement qu'avec une interconnexion rapide comme NVLink) |
| **Tensor Split** | *(auto)* | `--tensor-split` | Poids relatifs séparés par virgules par GPU visible dans le conteneur (p. ex. `24,16`). Vide se remplit automatiquement depuis la VRAM totale de chaque GPU au lancement. Ignoré quand Split Mode = `none` ou un seul GPU visible |

**Extra Args** *(champ texte)* - flags supplémentaires passés directement à llama-server (p. ex. `--mlock`, ou des types de spéculation n-gram non exposés dans le formulaire).

**Proxy-Side Sampling Overrides** *(interrupteur)* - quand activé, le proxy force les paramètres d'échantillonnage sur chaque requête transférée :

| Réglage | Défaut | Plage |
|---|---|---|
| Temperature | `0.8` | 0.0-2.0 |
| Top K | `40` | ≥ 0 (`0` désactive) |
| Top P | `0.95` | 0.01-1.0 |
| Presence Penalty | `0.0` | -2.0 à 2.0 |
| Repeat Penalty | `0.0` | 0.0-2.0 (`0` = désactivé) |

**Speculative Decoding** *(interrupteur)*

| Réglage | Défaut | Flag | Notes |
|---|---|---|---|
| Draft Type | `draft-mtp` | `--spec-type` | `draft-simple` / `draft-mtp` / `draft-dflash` / `draft-dspark` / `draft-eagle3`. Seul `draft-mtp` accepte un drafter vide (se rabat sur les têtes MTP intégrées du modèle principal) |
| Draft Model | *(aucun)* | `-md` | Chemin du GGUF drafter. Requis pour tous les types sauf `draft-mtp` |
| Draft N Max | `2` | `--spec-draft-n-max` | Max de tokens draftés par étape |
| Draft N Min *(Advanced)* | *(auto)* | `--spec-draft-n-min` | Vide omet le flag |
| Draft P Split *(Advanced)* | *(auto)* | `--spec-draft-p-split` | `0.0`-`1.0`. Vide omet le flag |
| Draft P Min *(Advanced)* | *(auto)* | `--spec-draft-p-min` | `0.0`-`1.0`. Vide omet le flag. `0` est une valeur réelle distincte de vide |

**Image & PDF Input** *(interrupteur)* - voir [Entrée image et PDF](#image--pdf-input) ci-dessous.

**Anti-Loop** — deux sous-interrupteurs indépendants (les deux désactivés par défaut). Voir [Anti-Loop](#anti-loop) ci-dessous.

<a id="live-preset-updates"></a>
## Mises à jour en direct des préréglages

Sauvegarder un préréglage met à jour sur place les instances déjà en cours de ce modèle quand c'est possible :

- **En direct (pas de relancement) :** `idle_timeout_min`, `max_concurrent`, `max_queue_depth`, `share_queue`, les six champs de proxy-sampling, les cinq champs de loop-detection (la requête suivante les prend en compte)
- **Nécessitent un relancement :** tout ce qui est intégré au conteneur au lancement (gpu layers, MoE CPU layers, ctx size, threads, threads-batch, cache types, flash-attn, reasoning-format, load-mode, DRY sampler, champs spec-decoding, flag embedding, champs mmproj/PDF, extra args, memory limit)

**Avertissement proxy-sampling :** si l'instance a été lancée avec `idle_timeout=0`, `max_concurrent=0` et `override_enabled=false`, aucun proxy sidecar n'a été spawné. Basculer `override_enabled=true` en direct applique quand même les overrides sur les requêtes routées par les endpoints de compatibilité de l'appli, mais les accès directs au port public les contournent. Relancez pour lancer le proxy dans ce cas.

<a id="concurrency--queueing"></a>
## Concurrence et mise en file

Quand **Max Concurrent** > 0, llamaMan place une gate FIFO devant l'instance. Les requêtes en excès attendent jusqu'à **Max Queue Depth** avant de renvoyer un HTTP 429. Les stats de la gate (actives + en file) sont visibles via l'API.

**Parallel vs Max Concurrent :** `Parallel` contrôle les slots KV internes de llama-server ; `Max Concurrent` limite combien de requêtes llamaMan transfère à la fois. Utilisez les deux ensemble (p. ex. `Parallel=4`, `Max Concurrent=4`) pour bien nourrir le serveur sans saturer ses slots.

<a id="image--pdf-input"></a>
## Entrée image et PDF

Activez l'interrupteur **Image & PDF Input** pour charger le projecteur multimodal d'un modèle de vision (`--mmproj`) à côté de son GGUF principal. Le champ **MMPROJ Model** prend le GGUF du projecteur (nom de fichier typique : `mmproj-<model>-<precision>.gguf`).

**Offload mmproj to GPU** (activé par défaut) : llama.cpp garde le projecteur sur le GPU sauf s'il reçoit `--no-mmproj-offload`, donc *on* omet entièrement le flag (CLI identique octet pour octet à avant l'existence de cette option) et *off* passe `--no-mmproj-offload` pour faire tourner l'encodeur de vision sur CPU — utile sur les configurations à VRAM serrée, où l'encodeur épingle quelques Go.

Les **PDF** sont pris en charge en rastérisant chaque page en PNG **avant** le transfert à llama-server (qui n'a pas de support PDF natif). Activez **Accept PDF uploads** (dans le groupe Image & PDF Input) pour l'activer sur une instance ; l'interrupteur maître **Image & PDF Input** contrôle tout le groupe — le désactiver efface aussi les sous-interrupteurs PDF, puisque les pages rastérisées n'ont aucun modèle de vision à atteindre sans le projecteur. Reconnu dans les deux :

- **OpenAI** : `image_url` avec une URL `data:application/pdf;base64,...`, ou le bloc `file` plus récent
- **Ollama** : des octets PDF base64 mélangés dans `images[]` - llamaMan détecte `%PDF-` et développe les PDF ; les images ordinaires gardent leur chemin existant

**Raccourci couche texte** (désactivé par défaut) : avec **Try text layer first** activé, llamaMan exécute `pypdf.extract_text()` avant la rastérisation. Les PDF nés numériques (Word/LaTeX/impression navigateur) avec un texte substantiel (≥ ~50 caractères/page) sont insérés en ligne comme un seul bloc de texte et évitent entièrement le chemin vision. Un texte clairsemé se rabat sur la rastérisation. Désactivé par défaut parce que la rastérisation préserve la mise en page pour le modèle de vision, et un comportement prévisible sur chaque PDF vaut mieux qu'un raccourci silencieux.

**Plafonds par requête :** `PDF DPI` (72-600, défaut 200) et `Max PDF Pages` (1-200, défaut 20). Les PDF au-delà du plafond renvoient un HTTP 400 au lieu d'être tronqués.

**Concurrence de rastérisation :** le rendu PDF est gourmand en CPU et en RAM et s'exécute dans le thread worker gunicorn avant toute gate par instance. Un sémaphore à l'échelle du processus plafonne les rastérisations concurrentes ; réglez via `LLAMAMAN_PDF_MAX_CONCURRENT` (4 par défaut).

## Anti-Loop

Deux contrôles indépendants dans la section **Anti-Loop** du formulaire de lancement, tous les deux désactivés par défaut. Ils visent le même problème (des modèles bloqués dans des boucles de sortie stables) à deux endroits différents du pipeline ; utilisez-les ensemble pour la meilleure couverture.

**DRY sampler (doux, au moment de l'échantillonnage)**. Correspond à `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n` de llama-server. Chaque token qui prolongerait un n-gramme vu récemment reçoit une pénalité de probabilité au moment de l'échantillonnage, donc le modèle est éloigné de la répétition avant même d'émettre un doublon. C'est la ligne de défense *douce* — souvent suffisante à elle seule pour de la prose. Intégré au lancement du conteneur ; une modification de préréglage exige un relancement. Multiplier = 0 désactive entièrement (le propre défaut de llama.cpp).

*DRY Sampler* (intégré au lancement — nécessite un relancement) :

| Réglage | Défaut | Flag | Notes |
|---|---|---|---|
| Multiplier | `0.8` (valeur UI ; `0` désactive) | `--dry-multiplier` | Force de la pénalité ; llama.cpp traite `0` comme désactivé |
| Base | `1.75` | `--dry-base` | Base exponentielle ≥ 1.0 |
| Allowed Length | `2` | `--dry-allowed-length` | Les n-grammes jusqu'à cette longueur sont exemptés |
| Penalty Last N | *(auto)* | `--dry-penalty-last-n` | Fenêtre d'historique prise en compte. Vide omet le flag |

**Output loop detection (dur, a posteriori)**. Quand DRY ne suffit pas — le modèle reste bloqué dans un attracteur stable — le proxy de llamaman surveille le texte de sortie visible de l'assistant (content + reasoning) pendant son streaming vers le client. À chaque cadence de scan (défaut : en ligne toutes les 64 portions, plus un repli worker de 10 s), les derniers `min_chunk_chars` du tampon sont vérifiés pour répétition. S'ils apparaissent ≥ `min_repetitions` fois, le tour est terminé :

- **La connexion upstream vers llama-server est fermée** — le stop-on-client-disconnect de llama-server arrête la génération, libérant le modèle pour la requête suivante.
- **Un terminateur synthétique est injecté dans le stream du client** — les clients OpenAI voient un dernier delta avec `finish_reason=stop` et le contenu `[llamaman: output loop detected, ending turn]` ; les clients Ollama voient `done_reason: "loop_detected"` sur une ligne `done: true`.

*Output Loop Detection* (côté proxy ; mis à jour en direct depuis le préréglage — la requête suivante prend les nouveaux seuils) :

| Réglage | Défaut | Notes |
|---|---|---|
| Min chunk chars | `200` | Plage 60-4096. Plus bas attrape des boucles plus courtes mais fait des faux positifs sur du contenu répétitif mais légitime |
| Min repetitions | `3` | Plage 2-20 |
| Buffer size (chars) | `8192` | Fenêtre glissante par requête en vol. Plage 512-65536 |
| Scan every N chunks | `64` | Cadence de scan en ligne |
| Fallback scan (seconds) | `10` | Worker en arrière-plan pour les streams qui émettent des tokens trop lentement pour atteindre le seuil en ligne |

**Où il surveille :**
- **Routes de compatibilité sur `:42069`** (Ollama `/api/chat`, `/api/generate` ; OpenAI `/v1/chat/completions`, `/v1/completions`) — toujours, quand l'interrupteur est activé sur le préréglage de l'instance cible.
- **Ports par instance (`8000-8020`)** — seulement quand un proxy sidecar est déjà présent pour cette instance (c'est-à-dire Idle Timeout, Max Concurrent, ou Proxy Sampling Overrides aussi activés). La loop detection ne lance pas de proxy sidecar à elle seule — activez un des trois autres pour surveiller aussi les accès directs.

**Réglage :**
- `min_chunk_chars` (60-4096, défaut 200) — la plus courte période répétitive qui déclenchera. Plus bas attrape des boucles plus courtes mais risque des faux positifs sur du contenu répétitif mais légitime (listes numérotées, refrains de poésie, tableaux markdown). 60-100 est agressif ; 200 est le défaut équilibré pour la prose ; 400+ ne déclenche que sur des boucles à l'échelle du paragraphe.
- `min_repetitions` (2-20, défaut 3) — combien de copies exactes de la portion doivent apparaître avant de terminer. 2 se déclenche sur n'importe quelle portion doublée (agressif) ; 3-5 équilibrent réactivité et faux positifs.
- `max_buffer_chars` (512-65536, défaut 8192) — fenêtre glissante conservée par requête active. Doit valoir au moins `min_chunk_chars × min_repetitions`.
- `scan_every_n_tokens` (8-4096, défaut 64) — cadence de scan en ligne. Plus bas = kill plus rapide, plus de CPU.
- `scan_interval_s` (1-600, défaut 10) — à quelle fréquence le worker en arrière-plan rescane les tampons qui n'ont pas atteint le seuil en ligne.

Les modifications des seuils de loop-detection prennent effet à la *requête suivante* (les streams en vol gardent les seuils capturés au moment de l'attachement). Les changements DRY exigent un relancement puisque les flags sont intégrés au conteneur.

**Algorithme de détection** (v1) : prendre les derniers `min_chunk_chars` du tampon comme cible ; compter les occurrences exactes ; signaler quand ≥ `min_repetitions`. N'attrape pas les quasi-répétitions dérivantes (« The answer is: The answer is: The answer is: » avec une légère dérive de ponctuation entre répétitions) — celles-là sont attrapées par les défenses DRY au moment de l'échantillonnage. Volontairement simple et bon marché.

**Invariant défensif :** chaque hook dans le chemin de streaming est enveloppé dans un `try/except Exception` qui journalise et passe — un bug du détecteur ne peut jamais casser le stream d'un vrai utilisateur.

<a id="per-instance-proxy"></a>
## Proxy par instance

Quand **Idle Timeout**, **Max Concurrent**, ou **Proxy Sampling Overrides** sont activés pour une instance, llamaMan insère un proxy WSGI devant le conteneur llama-server : le port public (p. ex. 8000) est le proxy ; llama-server écoute sur un port interne séparé. Le proxy fait la validation de nom de modèle, le wake-on-request pour les instances endormies, la limitation de concurrence `RequestGate`, et l'injection d'échantillonnage.

**Validation de nom de modèle :** sur les endpoints d'inférence, si le corps de la requête a un champ `"model"` il est comparé à la racine du nom de fichier du modèle chargé (minuscules, sans extension). **La correspondance par préfixe est acceptée** (p. ex. `"qwen2.5-0.5b-instruct-q2"` correspond à `"qwen2.5-0.5b-instruct-q2_k"`). Une non-correspondance renvoie un HTTP 404 ; pour les instances endormies, pas de réveil. Les requêtes sans champ `"model"` sont transférées inconditionnellement.

<a id="idle-timeout"></a>
## Délai d'inactivité (Idle Timeout)

Définissez **Idle Timeout min** dans le formulaire de lancement (0 = désactivé). Quand activé, après N minutes sans requêtes le conteneur llama-server est stoppé pour libérer la VRAM ; la requête suivante réveille un nouveau conteneur avec la même config. Les clients voient le même port avec un délai de cold-start.

Pour les instances gérées par proxy (OpenWebUI), utilisez `LLAMAMAN_IDLE_TIMEOUT` à la place.
