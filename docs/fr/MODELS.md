# Modèles, téléchargements et éviction

Tout sur la bibliothèque de modèles : ajouter des modèles, les garder à jour, comment le proxy les choisit et les évince, et les réglages d'entretien.

## Sommaire

- [Modèles](#models) · [Modèles de décision](#decision-models) · [Éviction de modèles](#model-eviction) · [Réglages de téléchargement](#download-settings) · [Gestion des images Docker](#docker-image-management) · [Archive de modèles](#model-archive) · [Sauvegarde et restauration des modèles](#model-backup-and-restore) · [Réglages de nettoyage](#cleanup-settings)

<a id="models"></a>
## Modèles

Placez les modèles sous le volume `/models` en fichiers `.gguf`, en répertoires de dépôt HuggingFace (contenant `config.json`), ou utilisez le bouton **Download** dans l'UI.

<a id="display-name"></a>
### Nom d'affichage

Les modèles sont listés par racine de nom de fichier GGUF par défaut. Chaque modèle peut recevoir un **Display Name** convivial (dans le formulaire de lancement). Quand il est défini, c'est l'id que les clients API voient sur `/api/tags` et `/v1/models`, donc OpenWebUI affiche `Qwen 2.5 14B` au lieu de `Qwen2.5-14B-Instruct-Q4_K_M`. Doit être unique et ne pas entrer en conflit avec le nom de fichier d'un autre modèle ou un nom de queue-group du cluster.

<a id="model-updates"></a>
### Mises à jour des modèles

Le bouton **Check for updates** (affiché pour les modèles avec un dépôt source connu) compare le fichier local au hash de contenu publié du dépôt :

- **Up to date** - le hash enregistré correspond
- **Update available** - re-télécharge dans un dossier de staging et n'effectue la bascule atomique qu'une fois le téléchargement terminé (un pull échoué laisse le modèle actuel intact). Refuse avec 409 tant que le modèle est chargé.
- **Verify hash** - hache le fichier local (travail en arrière-plan avec progression) quand aucun hash n'a encore été enregistré

Un scan en arrière-plan opt-in (**Settings → Download Settings**, voir [Réglages de téléchargement](#download-settings)) garde ces réponses prêtes à l'avance.

<a id="decision-models"></a>
## Modèles de décision

Les decision models répondent à des questions typées sur une entrée avec des probabilités au lieu de générer du texte : l'API System One introduite par le Jev de TypeSafe, servie par llama.cpp (images construites après le 2026-10-05) sur `POST /v1/systemone`. GGUF ouverts : Julia-1, Laya, Kev-4B, lev, Clef-Flash, Clef, OpenJev (OpenJev et Clef acceptent aussi des images, avec leur `--mmproj`). llama-server les reconnaît depuis `<arch>.decision.type` dans le GGUF ; aucun flag n'est nécessaire.

- **Détection.** En choisir un dans le formulaire de lancement le signale sous Model Path. Sans préréglage, le formulaire est prérempli pour lui : **Exclude from Max Models** activé, **Web UI** désactivé (sa page de chat ne peut pas piloter le modèle), **Parallel Slots** `8` (chaque question d'une requête prend un slot, et elles partagent l'entrée), **Embedding Model** désactivé (llama.cpp active le mode embedding lui-même là où le modèle en a besoin ; le forcer n'est pas nécessaire). Un modèle encodeur (type BERT) reçoit aussi Context, Batch et Micro-batch Size réglés à sa longueur de contexte : toute son entrée doit tenir dans un seul micro-batch. **Clef** (et Clef-Flash) pose toutes les questions dans un seul prompt et les lit en une seule passe, donc il reçoit **Parallel Slots** `1` et Batch = Micro-batch = Context (16384, son défaut entraîné, quand Context est vide) : toute la requête (entrée, chaque question et option) doit tenir dans un seul micro-batch, sinon llama-server répond « increase the physical batch size ».
- **Routage.** `POST :42069/v1/systemone` choisit le modèle par `"model"` comme `/v1/chat/completions` (auto-démarrage, règles d'éviction, forwarding cluster, file). La requête est transférée inchangée (pas d'overrides d'échantillonnage). Sur le port propre de l'instance, elle passe par la file du proxy par instance quand ce proxy tourne.
- **Les endpoints de chat les refusent** (HTTP 422) - ils ne génèrent pas de texte.
- **Request log :** les réponses, et `input_tokens` comme tokens du prompt (`output_tokens` est toujours 0).
- **Erreurs :** une question dont les options ne tiennent pas dans un batch est rejetée par llama-server (« increase the physical batch size ») - augmentez Batch Size / Micro-batch Size.

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
## Éviction de modèles

`LLAMAMAN_MAX_MODELS` plafonne les modèles de **chat** concurrents via le proxy. Quand le plafond est plein et qu'un nouveau modèle de chat est demandé, le modèle de chat LRU est évincé :

| Lanceur | Évince | Ne peut pas évincer |
|---|---|---|
| **Admin UI** | Gérés par Ollama d'abord (LRU), puis lancés par admin | - |
| **Ollama API** | Gérés par Ollama (LRU) | Lancés par admin *(par défaut)* |
| **OpenAI API** | Rien *(par défaut ; renvoie 503 quand plein)* | Tout *(par défaut)* |

Trois interrupteurs sous **Settings → App Settings** assouplissent ces défauts :

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - éviction LRU silencieuse avant lancement (gérés par Ollama d'abord). Désactivé par défaut → l'UI demande d'abord.
- **Allow Ollama API to evict admin-launched models** - repli quand aucun modèle géré par Ollama n'est disponible. Désactivé par défaut.
- **Allow OpenAI API to evict admin-launched models** - accorde à l'API OpenAI la même éviction LRU qu'Ollama. Désactivé par défaut.

Autres détails :

- Toutes les instances de chat en cours comptent (admin UI et gérées par proxy).
- **Les instances endormies comptent quand même** (la réservation du slot persiste pendant la pause idle) - mais réveiller une instance endormie pour son propre modèle n'est jamais bloqué par le plafond.
- Les instances avec **Exclude from Max Models** activé (formulaire de lancement) ne comptent pas et ne sont jamais évincées : modèles d'embedding, decision models, tout ce qui est petit et devrait rester actif. Activer **Embedding Model** l'active aussi, et les préréglages sauvegardés avant l'existence de l'interrupteur le déduisent de leur réglage Embedding Model.
- `LLAMAMAN_MAX_MODELS=0` désactive entièrement l'éviction.

<a id="download-settings"></a>
## Réglages de téléchargement

Sous **Settings → Download Settings** :

- **Global speed limit (Mbps)** (0 = illimité) - plafonne chaque téléchargement sur ce node et s'applique aux téléchargements en cours en moins d'une seconde. Par node : dans un cluster chaque node a son propre plafond (une valeur sauvegardée avant qu'il devienne par node s'applique à tous les nodes jusqu'à ce que ce node sauvegarde le sien)
- **Auto-retry failed downloads** (désactivé par défaut) + **Retry count per failed download** (3 par défaut)
- **Check models for updates in the background** (désactivé par défaut) - worker opt-in qui demande à chaque dépôt source si un fichier a été republié, et calcule une checksum pour tout modèle qui n'en a pas encore. **Update check interval (hours)** 24 par défaut. Hache au plus un modèle par passe ; ne s'exécute jamais pendant qu'un téléchargement est en cours.

<a id="docker-image-management"></a>
## Gestion des images Docker

**Settings → Docker Images** : tirez n'importe quelle image llama.cpp par nom, supprimez les vieilles images locales (désactivé pour la `LLAMA_IMAGE` active, et renvoie une erreur si Docker refuse parce qu'un conteneur l'utilise), et éventuellement auto-mettez à jour les images llama.cpp du node (`LLAMA_IMAGE` et chaque image tirée là) selon un calendrier. La ligne de pull a un sélecteur de moteur : avec `STRATA_ENABLED`, une image tirée en tant que *Strata* va dans *Manage Strata images* (il n'y a pas de source Strata par défaut ; tapez le nom). **Pull repo & build latest** télécharge le commit le plus récent du dépôt Strata (un tarball GitHub, dans `STRATA_SRC_DIR`) et construit `STRATA_BUILD_IMAGE` avec BuildKit via le socket Docker. L'auto-mise à jour propre à Strata re-tire les images Strata tirées et, quand le dépôt a été téléchargé, récupère son commit le plus récent et reconstruit s'il a changé (ou si l'image manque) ; elle ne télécharge jamais le dépôt d'elle-même. **Delete image & repo** supprime l'image construite (refusé tant qu'une instance en cours l'utilise) et le dépôt téléchargé, pour un node qui fait tourner une image tirée. Les lancements Strata utilisent `STRATA_IMAGE` quand elle est sur le node, sinon l'image construite, sinon une tirée. Voir [Strata](STRATA.md).

<a id="model-archive"></a>
## Archive de modèles

*(opt-in)* Déplacez des modèles du stockage rapide vers un second volume et revenez-en depuis l'UI, au lieu de déplacer des fichiers à la main sur l'hôte.

**Activez-le** en montant le volume d'archive dans le conteneur llamaMan et en pointant `ARCHIVE_DIR` dessus :

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

Non défini, la fonctionnalité est masquée. llamaMan déplace les fichiers lui-même, donc aucun chemin `HOST_` n'est nécessaire.

**Utilisez-le.**

- Survolez un modèle dans la bibliothèque et cliquez sur l'icône d'archivage.
- Les modèles archivés restent dans la liste, estompés, avec un badge *archived* et un bouton **Restore**. Ils ne peuvent pas être sélectionnés pour le lancement et n'apparaissent jamais dans `/api/tags` / `/v1/models`.
- Les déplacements s'affichent dans le panneau **Downloads** avec progression, vitesse et temps restant, et peuvent être annulés.
- Restore remet un modèle à son chemin d'origine, donc préréglages, favoris, notes et display names reviennent avec lui.

**Ce qui bouge.** Tout le dossier de premier niveau d'un modèle sous `/models` : le dossier d'un téléchargement, avec chaque shard d'un GGUF scindé. Un `.gguf` posé directement dans `/models` bouge avec ses frères scindés. Le dossier de shards d'un modèle Strata (`strata/<tag>`) bouge seul.

**Sécurité.**

- **Même disque :** le déplacement est un renommage instantané.
- **À travers des disques :** llamaMan copie dans un dossier caché `.llamaman-partial-*` sur la destination, fsync, vérifie la taille de chaque fichier, le renomme en place, et seulement alors supprime la source.
- **Échecs :** une annulation, une vérification échouée ou trop peu d'espace libre laissent la source intacte. Les copies partielles laissées par un redémarrage en plein déplacement sont supprimées au démarrage.
- **Un déplacement à la fois.**
- **Refusé** pendant qu'une instance utilise le modèle (y compris une endormie, ou comme modèle draft / mmproj, ou une instance Strata utilisant les shards), pendant qu'un téléchargement y écrit, ou quand le nom de destination existe déjà. Lancer ou supprimer un modèle en cours de déplacement est refusé aussi.

**Limites.** Les travaux de déplacement vivent en mémoire : après un redémarrage de llamaMan la liste est vide, et un déplacement interrompu doit être relancé (rien n'est perdu). En mode cluster chaque node archive vers son propre volume, et les boutons s'affichent pour le node que vous parcourez.

<a id="model-backup-and-restore"></a>
## Sauvegarde et restauration des modèles

**Settings → App Settings** :

- **Download Stored Models JSON** - exporte tous les modèles scannés + configs de préréglage dans un JSON horodaté
- **Restore from JSON** - par modèle : déjà présent → préréglage fusionné (valeurs existantes préservées) ; manquant mais avec une source HuggingFace → mis en file pour téléchargement avec préréglage pré-rempli ; manquant et sans source connue → signalé comme non restaurable

<a id="cleanup-settings"></a>
## Réglages de nettoyage

Sous **Settings → Cleanup Settings** (tout s'exécute périodiquement en arrière-plan ; ne supprime/met à jour que des enregistrements - ne supprime jamais de fichiers de modèles) :

- **Auto-clean completed/failed downloads** (24 h par défaut) - les téléchargements actifs ne sont jamais touchés
- **Auto-clean stopped instances** (24 h par défaut) - les instances en cours ne sont jamais supprimées
- **Auto-remove stale instance records** (intervalle de 5 min par défaut) - revérifie les enregistrements `starting`/`healthy`/`sleeping` contre Docker ; ceux dont le conteneur a disparu sont marqués stopped
