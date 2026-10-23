# Moteur Strata

*(opt-in, NVIDIA uniquement)* [Strata](https://github.com/Niko1221/Strata) fait tourner la famille mixture-of-experts Qwen3.8-Flash-Next (et ses variantes Coder, Swift 1.5 et Unsloth) sur des GPUs NVIDIA grand public plus la RAM système. llamaMan peut lancer, surveiller, proxyser, endormir/réveiller et stopper des instances Strata à côté de celles de llama.cpp. Le comportement de llama.cpp ne change pas : la couche moteur (`core/engines/`) garde le lancement llama.cpp identique octet pour octet (testé par snapshots).

## Sommaire

- [Premiers pas](#getting-started) · [Limites et comportement](#limits-and-behaviour) · [Notes de conception et vérification](#design-notes-and-verification)

<a id="getting-started"></a>
## Premiers pas

**1. Obtenir une image** (upstream n'en publie aucune ; l'hôte a besoin du pilote NVIDIA **580+**, c'est-à-dire CUDA 13, et du NVIDIA Container Toolkit). Après avoir activé Strata (étape 2), soit tirer par nom une image de confiance dans **Settings → Docker Images** (moteur *Strata* dans la ligne de pull), soit utiliser **Manage Strata images → Pull repo & build latest**, qui télécharge le dépôt dans `STRATA_SRC_DIR` et construit `STRATA_BUILD_IMAGE` via le socket Docker pour les générations GPU du node (lues depuis le pilote ; toutes celles prises en charge si cela échoue), ou la construire à la main :

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. L'activer** sur le node llamaMan : `STRATA_ENABLED=true` (voir [variables Strata](CONFIGURATION.md#strata)). Settings → Docker Images affiche alors *Manage Strata images* : les images Strata tirées, celle construite (commits téléchargé et construit, le bouton de build) et leur auto-mise à jour. Le formulaire de lancement propose Strata dès qu'une d'elles est sur le node.

**3. Lancement.** La bibliothèque de modèles gagne une entrée virtuelle par famille × taille : `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(expérimental)*. Chacune est sa propre instance et son propre conteneur, donc changer de modèle fonctionne comme avec llama.cpp : lancer ou réveiller une autre instance, avec les règles habituelles d'[éviction](MODELS.md#model-eviction) (une instance Strata est un modèle de chat). Les entrées apparaissent aussi dans `/api/tags` et `/v1/models`, et les requêtes peuvent utiliser l'id (`strata/qwen-IQ2_XS`, n'importe quelle casse, `:tag` ignoré) ou le propre nom servi de Strata (`qwen3.8-flash-next-iq2_xs`).

Le formulaire de lancement masque les champs propres à llama.cpp (GPU layers, MoE, threads, parallel, types KV, flash attention, spec decoding, mmproj, DRY, extra args) et montre **Image Input** (no / GPU / CPU), **KV Cache** (défaut Strata int8, `int8`, `q4_0`, `k8v4`) et **Low-RAM Mode** (auto / on / off). L'API rejette les champs propres à llama.cpp définis pour un modèle Strata.

**Premier démarrage.** Strata télécharge ~58-111 GB, prépare son « pack » et récupère sa couche MTP avant que le serveur n'ouvre son port ; les démarrages suivants vont directement au chargement (1-3 minutes). La carte d'instance affiche *setting up → downloading (avec %) → preparing pack → loading* parsé depuis le log du conteneur, et le bouton Logs streame le log. Deux façons de télécharger :

- **Laisser Strata télécharger** dans son volume `/data` (défaut).
- **Pré-télécharger avec llamaMan** (bouton dans la section Strata, ou `POST /api/engines/strata/download`) : une entrée ordinaire dans l'onglet Downloads (progression, pause/reprise, retry, jetons HF, plafonds de vitesse) qui récupère les shards depuis la révision Hugging Face épinglée par Strata vers `MODELS_DIR/strata/<tag>/`. Une fois terminés, ils sont bind-mountés dans le conteneur et le setup de Strata les utilise au lieu de télécharger. Un lancement est refusé pendant que ce téléchargement est en cours. La couche MTP (~5 GB) et le pack restent préparés par Strata au premier démarrage.

<a id="limits-and-behaviour"></a>
## Limites et comportement

- **Une requête à la fois.** Max Concurrent est forcé à 1 ; les requêtes suivantes patientent dans la gate de llamaMan (Max Queue Depth, puis 429).
- **RAM.** Strata charge 32-62 GB en RAM (`--ulimit memlock=-1` est toujours défini). Une **System Memory Limit** force le mode Low-RAM, car le setup de Strata lit la RAM de l'hôte et ne voit pas une limite de conteneur ; l'UI avertit sous ~64 GB.
- **Les réglages sont intégrés au setup de Strata.** Context / image input / KV / Low-RAM / GPU pinning sont enregistrés par modèle sur le volume de données. llamaMan se souvient des réglages avec lesquels chaque modèle est devenu healthy pour la dernière fois et ne passe `REINSTALL=1` que quand ils changent (les fichiers sont réutilisés).
- **L'idle timeout** fonctionne (le sommeil stoppe le conteneur ; la requête suivante le réveille), mais un cold start prend des minutes et la requête qui réveille attend (`STRATA_LOAD_TIMEOUT`).
- **Une instance par modèle Strata par node** (les instances d'un même modèle partagent son setup sur le volume).
- **NVIDIA uniquement.** L'option est indisponible, avec la raison, sur les nodes ROCm / Intel / Vulkan / CPU et quand `STRATA_ENABLED` est désactivé. Dans un cluster chaque node annonce ses moteurs, donc seuls les nodes capables proposent des modèles Strata.
- llamaMan en bare-metal atteint Strata de la même façon que llama-server (port publié sur `LLAMA_HOST_ADDR`).

---

<a id="design-notes-and-verification"></a>
# Notes de conception et vérification

Cette page couvre le fonctionnement de l'intégration, ce dont elle dépend côté upstream, ses limites, et comment la vérifier sur un vrai hôte
NVIDIA.

Les références upstream pointent vers le commit Strata `99f3dbd` (2026-10-03).

## Comment ça s'intègre

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- Le moteur d'une instance est `config["engine"]`. Une valeur manquante signifie llama.cpp, donc chaque ligne
  d'instance, préréglage et étiquette de conteneur existants garde sa signification. Les configs llama.cpp ne sont jamais estampillées de la clé, car cela
  changerait l'étiquette `llamaman.config` du conteneur. `tests/fixtures/llamacpp_container_spec.json` contient les
  kwargs `containers.run` d'avant le refactor pour 9 configs × 5 fabricants GPU, et `tests/test_engines.py` vérifie qu'ils
  sont inchangés.
- Aucune migration de stockage n'est nécessaire. Préréglages et instances sont des blobs JSON dans les deux backends (`presets.json` /
  `state.json`, et les colonnes TEXT `data` de MariaDB), donc `engine` et `strata_*` sont des clés additives.
- Chaque modèle Strata est une entrée virtuelle, `/strata/<family>-<SIZE>` (id `strata/<family>-<SIZE>`), et est sa propre
  instance et son propre conteneur. Changer de modèle, éviction, sommeil et réveil fonctionnent comme pour les fichiers GGUF.
- `_run_container` prend le `container_spec()` du moteur et ajoute les périphériques GPU que le moteur demande. Un conteneur
  Strata reçoit :

  | | |
  |---|---|
  | image | `STRATA_IMAGE` (sinon l'image construite, puis une image Strata tirée) |
  | command | aucune (l'entrypoint ne lit que des variables d'environnement) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | `device_requests` NVIDIA (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | volume de données → `/data` ; logs ; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>` (`<family>-<SIZE>` pour non-qwen ; le dossier de shards sensible à la casse de setup.py) (rw), quand llamaMan détient des shards complets |
  | mem_limit | depuis System Memory Limit (force `LOW_RAM=on`) |
  | network / ports / labels | comme pour llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

## Réponses aux questions ouvertes

1. **Préparation.** `/health` répond `{"status": "ok", ..., "service": "strata"}` avant la gate de clé API
   (`serve/server.py:2172`). Le port HTTP ne s'ouvre qu'après le chargement du modèle : `serve()` tourne à
   `server.py:3008`, après le démarrage du moteur à `:2925`. Pendant que Strata télécharge ou charge, les connexions sont
   refusées, donc le health check existant de llamaMan (`status == "ok"`) n'a besoin d'aucun cas particulier. `/v1/models` renvoie une liste vide
   tant que le modèle est déchargé en mode lazy (`server.py:2190`), ce n'est donc pas un bon signal de préparation.
   `/v1/status` exige la clé API quand une est définie.
2. **Réglages.** Tout est passé en variable d'environnement lue par `docker-entrypoint.sh` : `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Les arguments de commande sont ignorés,
   car l'entrypoint reconstruit `$@` (`:42`, `:65`). Le GPU pinning utilise la numérotation conteneur : le runtime NVIDIA
   renumérote les GPUs hôtes que llamaMan attache, donc l'hôte `1,3` devient `GPUS=0,1`. llamaMan ne définit pas `API_KEY`, ce qui
   correspond à sa façon de faire tourner llama-server (pas de clé) ; son propre proxy et son auth sont devant.
3. **Réutiliser des GGUF sur disque.** `setup.py` a `--gguf-dir` (`setup.py:2954`), mais l'entrypoint ne le transmet jamais.
   Setup accepte cependant des shards complets déjà placés sous `/data/models/<fam tag><SIZE>/<original name>` (p. ex. `IQ3_S`, `coder-IQ1_M`; sensible à la casse) : il vérifie chacun
   contre son propre répertoire de tenseurs et le marque terminé (`setup.py:3389-3405`, « #173 »). llamaMan bind-monte donc
   sa copie là (lecture-écriture, pour les marques `.done`), et seulement une fois son propre téléchargement de ces shards
   terminé. Les shards sont récupérés depuis les révisions Hugging Face que Strata épingle (`setup.py:65-70`,
   reproduites dans `core/engines/strata.py`). La couche MTP (~5 GB) et le pack restent préparés dans le conteneur.
4. **Limite mémoire / memlock.** Setup lit la RAM depuis `/proc/meminfo`, qui est le total de l'hôte, donc un conteneur
   plafonné doit demander `LOW_RAM=on` (commentaire du Dockerfile à `:31`, `docs/INSTALL.md:116-118`). Sans cela, setup choisit
   le mode in-RAM et le conteneur sera probablement tué par OOM. llamaMan force `LOW_RAM=on` dès qu'une limite mémoire est
   définie, et l'UI avertit sous 64 GB. `memlock=-1` compte car le moteur verrouille en mémoire (page-lock) une partie de son arena pour le DMA GPU,
   et les experts qu'il ne peut pas épingler passent par des copies hôte plus lentes
   (`src/prefill/prefill.cpp:110`, `:207-214`). D'après le source, une limite memlock manquante ressemble à un ralentissement
   plutôt qu'à une panne, mais **je n'ai pas pu vérifier sans GPU**. llamaMan la définit toujours.

Également d'après le source : Strata sert une requête à la fois (`server.py:1323`). Le Dockerfile définit `STRATA_EXECV=1`,
donc le serveur est PID 1 et `docker stop` l'atteint. Un conteneur stoppé en plein téléchargement reprend depuis ses fichiers
`.part` au démarrage suivant.

## Décisions de conception

- **La concurrence est forcée à 1** au lancement, à la fusion de préréglage, aux modifications live de préréglage et dans les préréglages
  sauvegardés. La gate force le proxy sidecar, donc la mise en file se passe dans llamaMan (`RequestGate`), pas dans Strata.
- **`REINSTALL` par fingerprint.** Context, vision, KV, Low-RAM et GPU pinning sont enregistrés dans la config par modèle de
  Strata sur le volume. `DATA_DIR/strata_setups.json` se souvient des réglages avec lesquels chaque modèle est devenu healthy pour la dernière fois,
  indexés par volume de données et tag de setup, et `REINSTALL=1` n'est envoyé que quand ils diffèrent. Un fingerprint inconnu
  (premier lancement via llamaMan, ou un volume configuré à la main) réinstalle aussi. Cela coûte une passe de setup qui réutilise les fichiers.
- **Une instance active par modèle Strata par node.** Deux instances d'un même modèle exécuteraient le setup sur les mêmes fichiers.
- **Un lancement est refusé pendant que llamaMan télécharge encore ce modèle**, pour que le conteneur ne le télécharge pas une
  deuxième fois dans son volume.
- **Les shards pré-téléchargés court-circuitent la provenance du check de mise à jour.** Un fichier épinglé ne doit pas être « mis à jour » vers le `main` du dépôt.
- **Les modèles virtuels ne sont listés que là où ils peuvent tourner** (`STRATA_ENABLED` plus NVIDIA) : dans `/api/models`,
  `/api/tags`, `/v1/models` et la résolution de noms. Les scans de mise à jour et le snapshot de modèles du cluster restent uniquement basés sur les fichiers. Les peers
  apprennent la disponibilité des moteurs depuis `snapshot.system.engines` (imbriqué sous `system`, donc les peers plus anciens l'ignorent).
- **L'étape de chargement** vient des 64 derniers Ko du log du conteneur : les en-têtes `=== Step N ===` du setup, la ligne de
  progression `\r` du téléchargement, et le « loading the model » du serveur. C'est best-effort : en cas d'échec, aucune ligne d'étape n'est affichée.
- **vLLM (#53)** serait un autre `Engine` : sa propre image, commande, port (`internal_port`), sémantique de health et
  champs de lancement. Les endroits qui supposent que les deux moteurs utilisent le port 8080 (les appelants de `resolve_llama_endpoint`)
  sont ceux à paramétrer alors.

## Non vérifié sans GPU

- Un vrai lancement Strata, le téléchargement de premier démarrage, la construction du pack, le chargement et le chat. Tout ce qui précède a été vérifié contre le
  source upstream et avec des tests unitaires qui mockent Docker ; aucun conteneur Strata n'a tourné.
- Le format de log exact d'un vrai premier démarrage. Le parseur d'étapes suit la sortie `step()` / `download()` de `setup.py`
  et les affichages de démarrage du serveur.
- Le comportement sans `memlock=-1`, et la RAM réellement utilisée sous `LOW_RAM=on` avec une limite de conteneur.
- Le bind mount imbriqué (`/data/models/<tag>` par-dessus un volume nommé dans `/data`) sur les versions de Docker utilisées. Docker
  supporte les mounts imbriqués, mais cela n'a pas été exécuté ici.

## Limitations connues

- NVIDIA uniquement. Le chemin AMD (HIP) de Strata est un build manuel expérimental upstream et n'est pas proposé.
- `LAYER_SPLIT` (placement de couches multi-GPU) n'est pas exposé. Épingler plusieurs GPUs utilise le split automatique de Strata.
- L'encodeur de vision (mmproj) est toujours téléchargé par Strata elle-même, dans le volume.
- La propre UI web de Strata (`/`, `/settings`, `/load`, `/unload`) est atteignable par le port de l'instance, mais llamaMan
  ne l'utilise ni ne la gère. Sa fonction d'idle-unload est inutilisée ; le sommeil de llamaMan stoppe le conteneur à la place.
- Changer un réglage enregistré dans le setup relance la passe de setup de Strata au démarrage suivant (les fichiers sont réutilisés).
- Un premier démarrage peut durer plus longtemps que le propre timeout d'un client HTTP (Open WebUI et autres). llamaMan continue d'attendre
  jusqu'à `STRATA_LOAD_TIMEOUT`, mais le client peut abandonner avant. Pré-téléchargez, ou lancez une fois depuis l'UI avant de l'exposer aux clients.
- Le catalogue (familles, tailles, révisions épinglées, noms de fichiers) est un miroir du `setup.py` de Strata. Une nouvelle
  famille ou taille upstream demande une modification correspondante dans `core/engines/strata.py`.

## Checklist de vérification manuelle (hôte NVIDIA)

Prérequis : un pilote NVIDIA 580+, un `docker run --gpus all` fonctionnel, 64 GB de RAM recommandés, et environ 150 GB de
disque libre.

1. **Build :** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Optionnellement
   ajoutez `--build-arg CUDA_ARCHITECTURES=<your arch>`. Ou, après l'étape 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** Il ne construit que pour les générations GPU de ce node.
2. **Activer :** définissez `STRATA_ENABLED=true` et redémarrez llamaMan.
   - `GET /api/engines` montre strata `available: true`, avec ses modèles (9 dans la liste intégrée).
   - Settings → Docker Images → *Manage Strata images* montre `strata:latest` comme **local**.
3. **Gating :** sur un node non-NVIDIA (ou avec `GPU_TYPE=rocm`), les modèles Strata ne sont pas listés, et
   `POST /api/instances` pour `/strata/qwen-IQ2_XS` renvoie 400 avec la raison NVIDIA.
4. **Lancer depuis l'UI :** sélectionnez `strata/qwen-IQ2_XS`.
   - Les champs propres à llama.cpp sont masqués, la section Strata est affichée, et Max Concurrent est verrouillé à 1.
   - Définissez une limite mémoire de 32g et vérifiez que l'avertissement sous 64 GB apparaît ; retirez-la.
   - Lancez.
5. **Premier run (téléchargement conteneur) :**
   - La carte d'instance affiche *setting up → downloading NN% → preparing pack → loading*.
   - Le bouton Logs streame la sortie de setup de Strata.
   - La carte passe **healthy**, et `docker inspect` montre `Ulimits: memlock -1`, les variables d'environnement et le
     volume `/data`.
6. **Chat :**
   - Via le proxy Ollama : `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Via `/v1/chat/completions` sur `:42069`, et directement sur le port de l'instance.
   - Essayez aussi `model: "qwen3.8-flash-next-iq2_xs"`.
   - Envoyez deux requêtes à la fois : la seconde est mise en file (la carte montre 1/1 active · 1 queued).
7. **Sommeil idle et réveil :**
   - Réglez Idle Timeout à 1 min (l'avertissement de cold-start s'affiche). Après une minute l'instance est **sleeping** et son
     conteneur a disparu.
   - La requête suivante la réveille (des minutes) et reçoit une réponse.
   - Aucune passe de setup dans le log (pas de lignes `=== Step` ; `REINSTALL` absent de `docker inspect`).
8. **Changement de réglage :**
   - Changez le context à 65536 et redémarrez.
   - `docker inspect` montre `REINSTALL=1`, et le log montre une passe de setup sans téléchargement.
   - Au redémarrage suivant, `REINSTALL` est de nouveau absent.
9. **Chemin de pré-téléchargement :**
   - Choisissez un autre modèle (p. ex. `strata/qwen-Q2_0`) et cliquez *Pre-download with llamaMan*.
   - L'onglet Downloads montre le téléchargement de la révision épinglée.
   - Lancer pendant le téléchargement est refusé.
   - Une fois terminé, lancez : `docker inspect` montre `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`, et le
     log ne montre aucun téléchargement de shard (seulement la couche MTP et le pack).
10. **Stop / Remove :** stoppez l'instance. Le conteneur est supprimé, et Remove vide la carte.
11. **Redémarrer llamaMan avec une instance Strata en cours :**
    - Elle est rattachée à nouveau (restauration d'état), ou adoptée si l'état était perdu : supprimez-la de `state.json` et redémarrez.
    - La carte montre le badge Strata et `strata/qwen-IQ2_XS`.
    - Les requêtes fonctionnent toujours.
12. **Mode bare-metal :**
    - Faites tourner llamaMan avec `python app.py` (`LLAMAMAN_IN_DOCKER=false`) et répétez 4-6.
    - llamaMan atteint le port publié sur `LLAMA_HOST_ADDR`.
13. **Cluster :**
    - Ajoutez un peer non-NVIDIA. Les modèles Strata n'apparaissent pour le node NVIDIA que quand c'est le Target node.
    - Lancer sur le peer NVIDIA depuis l'UI de l'autre node fonctionne.
14. **llama.cpp inchangé :** lancez un modèle GGUF comme avant.
    - La ligne de commande (Args de `docker inspect`) correspond à celle d'avant la mise à niveau.
    - Les préréglages sauvegardés avant la mise à niveau se chargent et se lancent inchangés.
