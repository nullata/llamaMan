# Configuration et stockage

Installations bare-metal, le modèle d'authentification en détail, les statistiques GPU, l'enregistrement des requêtes, les backends de stockage, et la référence complète des variables d'environnement.

## Sommaire

- [Exécution en bare-metal](#running-bare-metal) · [Authentification en détail](#authentication-detail) · [Statistiques GPU](#gpu-stats) · [Enregistrement des requêtes et statistiques](#request-recording--stats) · [Backends de stockage](#storage-backends) · [Survivre à une panne de la base de données (miroir local)](#surviving-a-database-outage-local-mirror) · [Variables d'environnement](#environment-variables)

<a id="running-bare-metal"></a>
## Exécution en bare-metal

Utile pour le dev ou des hôtes (p. ex. WSL) où faire tourner le gestionnaire lui-même dans Docker est pénible. llama-server tourne toujours dans des conteneurs.

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

Un processus unique sous gunicorn sert l'UI/API sur `:5000` et le proxy sur `:42069` (`python app.py` sans gunicorn n'exécute le serveur de dev Flask que sur `:5000`). Conteneur-vs-bare-metal est auto-détecté ; si la détection se trompe pour votre environnement, définissez `LLAMAMAN_IN_DOCKER=true` ou `false` explicitement.

<a id="authentication-detail"></a>
## Authentification en détail

Deux types d'identifiants :

- **Cookies de session** pour l'UI (créés via `/setup` au premier lancement)
- **Clés API** pour les clients externes (`Authorization: Bearer llm-xxxxxxxxxx`), gérées sous **API Keys** dans l'UI

L'interrupteur **"Require authentication for all endpoints"** (activé par défaut) contrôle si les endpoints de service de modèles exigent un jeton bearer :

| Interrupteur | Endpoints de modèles (`/api/chat`, `/v1/chat/completions`, ports par instance) | Endpoints d'administration (`/api/instances`, etc.) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (par défaut) | Jeton bearer requis | Jeton bearer ou session |
| **OFF** | Ouvert | Jeton bearer ou session |

Les trois surfaces de port (5000, 42069, 8000-8020) passent par le même hook d'authentification.

<a id="gpu-stats"></a>
## Statistiques GPU

VRAM et utilisation par GPU, interrogées nativement - aucun llama-server en cours d'exécution requis.

| Fabricant | Méthode | Exigence |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML direct) | Décommentez `deploy.resources.reservations` dans `docker-compose.yml` pour la capability `utility` du toolkit NVIDIA |
| AMD | sysfs `/sys/class/drm` | Montage du volume `/sys/class/drm:ro` (inclus par défaut) |
| Intel Arc | sysfs `/sys/class/drm` | Même montage que AMD |

Sans accès natif, llamaMan se rabat sur l'exec de `nvidia-smi` / `rocm-smi` dans un conteneur en cours d'exécution. Les statistiques reflètent toujours l'état complet du GPU de l'hôte.

<a id="request-recording--stats"></a>
## Enregistrement des requêtes et statistiques

Sous **Settings → App Settings → Request recording** : **Off** (par défaut) / **Per request** / **Per conversation** (tours groupés par un hash de contenu du prompt système + premier message utilisateur). Chaque enregistrement capture les corps + champs d'enveloppe (modèle, endpoint, statut, durée, comptes de tokens) plus des **métriques précises par tour** (débit mesuré sur la fenêtre de génération ; TTFT). Les enregistrements vivent sous `RECORDINGS_DIR` (JSON) ou dans la table `request_log` (MariaDB). **Retention (days)** purge toutes les heures (`0` = conserver indéfiniment).

Chaque carte d'instance expose un bouton **Stats** (nombre de requêtes, débit moyen/crête, TTFT/latence moyens, totaux de tokens, plage de temps actif) agrégé depuis le request-log - il persiste donc après l'arrêt de l'instance et affiche un état vide quand l'enregistrement est désactivé. Le lien **Logging** dans l'en-tête ouvre un tableau de bord pleine page (tuiles de résumé, conversations récentes, vue détaillée par conversation) avec un sélecteur de fenêtre 24 h / 7 j / 30 j / Toutes les périodes plus des filtres par modèle et par jour sur la liste des conversations.

<a id="storage-backends"></a>
## Backends de stockage

### JSON (par défaut)

Zéro configuration. Fichiers sous `DATA_DIR` (`/data`) : `state.json`, `presets.json`, `users.json`, `settings.json`, `api_keys.json`, `request_log/`. Les logs d'instance et de téléchargement vont dans `LOGS_DIR` (`/tmp/llama-logs`).

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

Les tables se créent automatiquement à la première connexion. Les logs de requêtes vont dans la table `request_log` au lieu du répertoire JSON `request_log/` (`RECORDINGS_DIR` n'a aucun effet avec MariaDB).

Les métadonnées de modèles par node (dépôt source + hash de contenu) vivent dans une table `model_files` de portée node, indexée par `(node_id, model_path)`. À la mise à niveau, chaque node copie ses propres entrées hors du blob de réglages partagés hérité (seulement les chemins qui existent sur son propre disque) et garde le blob comme repli en lecture et chemin de retour arrière. Les migrations sont versionnées par node.

> **Note InnoDB :** `(node_id, model_path)` fait `(64 + 700) × 4 = 3056` octets en utf8mb4, juste sous la limite d'index de 3072 octets d'InnoDB, donc la table est créée avec `ROW_FORMAT=DYNAMIC`.

MariaDB est aussi le prérequis du [Clustering](CLUSTERING.md) et de la [Knowledge Base](KNOWLEDGE-BASE.md).

<a id="surviving-a-database-outage-local-mirror"></a>
## Survivre à une panne de la base de données (miroir local)

*Optionnel, désactivé par défaut.*

Avec `DATABASE_URL` défini, la base de données est sur le chemin critique de **chaque** requête (les vérifications d'auth lisent les réglages et vérifient les clés API à chaque appel). Si elle devient injoignable, le node cesse de servir, et un redémarrage de conteneur en pleine panne ne peut même pas amorcer.

Activez **Settings → App settings → "Keep a local mirror of the database"** (ou définissez `LLAMAMAN_DB_MIRROR=1`) et le node garde une copie write-through dans `DATA_DIR/db_mirror/`.

| Fonctionne encore hors ligne | Bloqué hors ligne |
|---|---|
| Toute l'inférence (les deux ports) | Créer le premier utilisateur (`/setup`) - une installation toute neuve n'a de toute façon pas de miroir sur lequel se rabattre |
| Lancer / stopper / endormir / réveiller des modèles | |
| Téléchargements, hachage de modèles et vérifications de mise à jour | |
| Créer et modifier des **préréglages** (y compris pour des modèles que vous venez de télécharger) | |
| Créer et révoquer des **clés API** | |
| Modifier les **réglages** de téléchargement et d'application | |
| Ajouter et retirer des **jetons Hugging Face** | |

Les changements hors ligne vont dans un journal append-only et sont rejoués à la reconnexion (sonde de 10 s). Le rejeu enregistre la *modification*, pas la ligne résultante, donc un changement fait par un autre node pendant la panne survit (un champ de préréglage différent, une clé de réglage différente, un jeton HF différent sont tous laissés intacts).

**Exigences et avertissements :**

- **`DATA_DIR` doit être un volume persistant** (il l'est déjà dans le fichier compose d'exemple).
- **Chaque node a besoin de son propre `DATA_DIR`.** Le miroir est estampillé de l'id du node propriétaire ; deux nodes qui en partagent un désactivent la mise en miroir plutôt que de corrompre l'une des vues.
- **Les secrets atterrissent sur le disque local.** Les réglages sont copiés tels quels ; **les jetons Hugging Face sont copiés en clair dans le miroir.** Les hashes de mot de passe + clé API sont déjà hachés. Si vous avez choisi MariaDB en partie pour garder les secrets hors des hôtes individuels, ce compromis est la raison pour laquelle la fonctionnalité est opt-in.
- **L'équilibrage inter-nodes s'arrête pendant une panne.** La vivacité des peers vit dans la base de données ; un node dégradé ne voit que lui-même.
- **Les changements de clé API sont locaux jusqu'à la reconnexion.** Une clé créée hors ligne ne fonctionne que sur ce node ; une clé **révoquée** hors ligne reste valide sur les autres nodes jusqu'à la reconnexion. Planifiez les révocations en conséquence.
- **Les logs de requêtes sont jetés, pas mis en tampon,** tant que le node est hors ligne (les enregistrements portent les corps complets ; les spooler à travers une longue panne pourrait remplir le volume).
- **Désactiver le miroir en pleine panne est différé** jusqu'à la reconnexion.
- Les migrations de schéma ne s'exécutent jamais en mode dégradé ; elles s'exécutent contre la vraie base de données à la reconnexion, avant tout rejeu de journal.

<a id="environment-variables"></a>
## Variables d'environnement

### Core

| Variable | Défaut | Description |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(requis)* | **Requis.** Identité unique stable - clé de partition pour l'état stocké et identité cluster. N'importe quelle chaîne ; choisissez-en une une fois, gardez-la (la changer plus tard rend orphelin l'état stocké). |
| `MODELS_DIR` | `/models` | Répertoire scanné pour les fichiers de modèles (chemin conteneur) |
| `DATA_DIR` | `/data` | Répertoire pour la config/état persistant (fichiers JSON) |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | Répertoire pour les enregistrements du request-log par conversation. Backend JSON uniquement ; ignoré quand `DATABASE_URL` est défini |
| `LOGS_DIR` | `/tmp/llama-logs` | Répertoire pour les logs d'instance et de téléchargement (chemin conteneur) |
| `HOST_MODELS_DIR` | *(comme `MODELS_DIR`)* | Chemin absolu **côté hôte** du volume de modèles. Doit correspondre au côté gauche de `-v /host/path:/models` - passé au démon Docker en spawnant les conteneurs frères |
| `HOST_LOGS_DIR` | *(comme `LOGS_DIR`)* | Chemin absolu côté hôte du volume de logs. Même exigence que `HOST_MODELS_DIR` |
| `ARCHIVE_DIR` | *(non défini)* | Chemin conteneur où un volume d'archive est monté (p. ex. `/archive`). Le définir active [Archive de modèles](MODELS.md#model-archive) ; non défini le masque |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | Pool de ports publics llama-server/proxy |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | Pool de ports internes utilisé quand le mode proxy est activé |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port pour le proxy compatible Ollama |
| `LLAMAMAN_MAX_MODELS` | `0` | Max de modèles de chat concurrents via le proxy (éviction LRU ; 0 = illimité). Règles : [Éviction de modèles](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Minutes d'idle-timeout pour les instances gérées par le proxy (0 = désactivé) |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | Max de rastérisations PDF concurrentes dans tout le processus. Chaque rastérisation lance poppler et consomme quelques centaines de Mo de RAM transitoire en haute DPI |
| `SECRET_KEY` | *(auto)* | Secret de session Flask. Auto-dérivé de machine-id si non défini. Définissez-le pour les déploiements multi-répliques |
| `SESSION_COOKIE_NAME` | `llamaman_session` | Nom du cookie de session. Nom de cookie mis en espace de noms pour que llamaman coexiste avec d'autres applis Flask sur le même hôte - les cookies sont délimités par hôte+chemin, pas par port |
| `DATABASE_URL` | *(non défini)* | Chaîne de connexion MariaDB/MySQL. Non défini = fichiers JSON |
| `LLAMAMAN_DB_MIRROR` | *(non défini)* | Force le miroir DB local activé (`1`) / désactivé (`0`), en surchargeant le réglage par node. N'a de sens qu'avec `DATABASE_URL` |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout en secondes pour les health checks d'instance |
| `MODEL_LOAD_TIMEOUT` | `300` | Secondes à attendre qu'un modèle devienne healthy. Augmentez pour les très grands modèles |
| `REQUEST_TIMEOUT` | `300` | Timeout en secondes en **lecture** pour les requêtes upstream vers llama-server, le forwarding cross-node et les attentes d'acquisition de gate. Ne régit **pas** le temps de connexion peer (borne séparée de 5 s) |

### Docker / GPU

| Variable | Défaut | Description |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | Image Docker llama.cpp pour tous les conteneurs spawnés. Auto-sélectionnée depuis le fabricant GPU détecté ; définissez-la pour épingler une version/backend |
| `LLAMA_NETWORK` | `llamaman-net` | Réseau Docker pour llamaman + tous les conteneurs llama-server. Créé s'il manque |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | Préfixe de nom pour les conteneurs llama-server spawnés |
| `LLAMAMAN_IN_DOCKER` | *(auto)* | Si llamaman lui-même tourne dans un conteneur. Auto-détecté depuis les fichiers marqueurs + cgroups. Définissez `true`/`false` pour forcer |
| `LLAMA_HOST_ADDR` | `localhost` | Adresse d'hôte utilisée pour atteindre les ports publiés des conteneurs spawnés en bare-metal |
| `GPU_TYPE` | *(auto)* | Force la détection du fabricant GPU : `cuda`, `rocm`, `intel`, ou `vulkan` (`vulkan` est uniquement opt-in ; choisit `/dev/dri` + les GIDs render/video de l'hôte et l'image `server-vulkan`) |
| `LLAMA_GPU_DEVICES` | *(tous)* | Indices GPU séparés par virgules visibles par tous les conteneurs spawnés (p. ex. `0,1,3`). **GPU Devices** par instance a priorité. Non pris en charge sur Intel Arc |

### Strata

Optionnel ; voir [Strata](STRATA.md).

| Variable | Défaut | Description |
|---|---|---|
| `STRATA_ENABLED` | `false` | Proposer des modèles Strata sur ce node. A aussi besoin d'un GPU NVIDIA (`GPU_TYPE=cuda` ou détecté) |
| `STRATA_IMAGE` | `strata:latest` | Image Strata préférée pour les lancements. Quand elle n'est pas sur le node, l'image construite puis une tirée est utilisée |
| `STRATA_BUILD_IMAGE` | `strata:latest` | Tag que *Pull repo & build latest* construit |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | Volume Docker nommé monté dans `/data` dans chaque conteneur Strata (fichiers de modèles, packs, couche MTP, setup par modèle ; 70-120 Go par modèle). Créé par Docker à la première utilisation |
| `HOST_STRATA_DATA_DIR` | *(non défini)* | Chemin hôte à monter dans `/data` à la place du volume nommé. L'emporte sur `STRATA_DATA_VOLUME` |
| `STRATA_LOAD_TIMEOUT` | `3600` | Secondes qu'une requête (ou un relancement) attend qu'une instance Strata soit prête. Élevé car un premier lancement télécharge et prépare le modèle avant que le port s'ouvre |
| `STRATA_REPO` | `Niko1221/Strata` | Dépôt GitHub (owner/name) que *Pull repo & build latest* télécharge |
| `STRATA_REPO_REF` | `main` | Branche, tag ou commit de `STRATA_REPO` à construire |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | Où le dépôt téléchargé est conservé (le contexte de build). Son commit est enregistré à côté dans `Strata.source.json` |
| `STRATA_WEB_HOSTS` | *(non défini)* | Noms d'hôtes / IP supplémentaires (séparés par virgules) par lesquels vous ouvrez la propre appli web d'une instance Strata (`http://<host>:<instance port>/`). Sans clé API, Strata refuse les requêtes de chat venant de pages sur d'autres noms ; l'hôte de `CLUSTER_ADVERTISE_URL` est toujours autorisé. S'applique aux instances démarrées ensuite |

### Clustering

Optionnel. `LLAMAMAN_NODE_NAME` (sous Core) sert aussi d'identité cluster et est requis pour toutes les installations.

| Variable | Défaut | Description |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | Définir truthy pour joindre ce node à un cluster. Requiert `CLUSTER_SECRET` ; ignoré avec un avertissement si le secret est vide |
| `CLUSTER_SECRET` | *(non défini)* | Jeton bearer partagé (`X-Cluster-Secret`) pour tout le HTTP node-à-node. Identique sur chaque node. Utilisez une longue valeur aléatoire sur un réseau de confiance ou derrière TLS |
| `CLUSTER_ADVERTISE_URL` | *(non défini)* | Comment les peers atteignent l'UI/API de **ce** node - nom d'hôte/IP routable depuis les autres hôtes (pas `localhost`). Sans lui le node est en consultation seule |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | Secondes depuis le dernier heartbeat avant qu'un node s'affiche hors ligne. Augmentez si les nodes oscillent en ligne/hors ligne sous charge ou en cas de dérive d'horloge |
| `LLAMAMAN_PERF_LOG` | *(off)* | Diagnostics. Définir truthy pour journaliser les temps par phase (`perf <name> <ms>`) des chemins soupçonnés lents : stop d'instance distante, forwards du proxy cluster, sondes d'accessibilité, snapshot de heartbeat, vérifications d'auth, `save_state`, stats de conteneur. Désactivé par défaut ; overhead négligeable quand désactivé |
