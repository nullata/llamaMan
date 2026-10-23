# API REST

Tous les endpoints renvoient / acceptent du JSON. Les endpoints d'administration exigent un cookie de session (connexion navigateur) ou `Authorization: Bearer <key>`. Les endpoints de service de modèles exigent aussi un bearer quand `require_auth` est activé (par défaut). Détails d'authentification : [Authentication](CONFIGURATION.md#authentication-detail).

## Sommaire

- [Authentification et clés](#auth--keys) · [Instances](#instances) · [Téléchargements, modèles, préréglages](#downloads-models-presets) · [Réglages, système, journal des requêtes](#settings-system-request-log) · [Compatibilité Ollama et OpenAI](#ollama--openai-compat)

<a id="auth--keys"></a>
## Authentification et clés

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/login` | Page de connexion / authentifier |
| `GET` / `POST` | `/setup` | Configuration au premier lancement |
| `GET` | `/logout` | Terminer la session |
| `GET` / `POST` | `/api/api-keys` | Lister / créer des clés API |
| `DELETE` | `/api/api-keys/<id>` | Révoquer |

## Instances

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Lister / lancer |
| `GET` / `DELETE` | `/api/instances/<id>` | Récupérer / stopper et supprimer |
| `POST` | `/api/instances/<id>/restart` | Redémarrer une instance arrêtée/endormie |
| `DELETE` | `/api/instances/<id>/remove` | Supprimer l'enregistrement d'une instance arrêtée |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Suivre / stream SSE des logs |
| `GET` | `/api/next-port` | Obtenir le prochain port disponible du pool |
| `POST` | `/api/preview-command` | Rendre la commande `llama-server ...` équivalente pour un corps de lancement (utilisée par l'outil « Copy llama.cpp command » de l'UI ; ne lance rien) |

**Corps de lancement** (`POST /api/instances`) : les champs du formulaire de lancement, sérialisés en JSON. Voir la [Référence des paramètres de lancement](LAUNCH-SETTINGS.md#launch-settings-reference) pour chaque champ et sa valeur par défaut ; les champs à valeur énumérée (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) n'acceptent que les valeurs listées là et prennent par défaut la valeur par défaut de la référence. Les champs numériques laissés à `null` ou omis utilisent la valeur par défaut de la référence ; les champs inconnus sont ignorés.

Cas limites à retenir :

- `n_gpu_layers` : `-1` = **auto** (llama.cpp estime la VRAM), `-2` = **all** (forcer chaque couche), `0` = CPU uniquement (aucun GPU attaché au conteneur), N positif = exactement N couches.
- `n_cpu_moe_layers` : `0` (par défaut) omet les deux flags MoE. `-1` émet `--cpu-moe` (tous les experts routés sur CPU). N positif émet `--n-cpu-moe N` (les experts des N premières couches sur CPU). Inerte sur les modèles denses.
- `threads` : quand défini, applique `--threads N` à llama-server **et** définit le quota CPU du conteneur à N cœurs.
- `threads_batch` : null / omis → le flag n'est pas émis et llama-server se rabat sur `--threads`.
- `flash_attn` : les anciennes valeurs `true` / `false` des configs pré-tri-state sont repliées à la lecture (`true` → `"on"`, `false` → `"off"`), donc aucune migration de stockage n'est nécessaire.
- Les valeurs quantifiées de `cache_type_v` exigent `flash_attn: "on"` - `"auto"` n'est pas une garantie ; sinon llama-server refuse de démarrer.
- `spec_draft_model` est requis quand `spec_enabled: true` et que `spec_type` est autre chose que `"draft-mtp"` - une requête sans ce champ est rejetée avec 400.
- `spec_draft_n_min` / `_p_split` / `_p_min` : null / omis omet le flag. `0` est une valeur réelle distincte de null et est transmise telle quelle.

<a id="downloads-models-presets"></a>
## Téléchargements, modèles, préréglages

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Lister / démarrer |
| `GET` / `DELETE` | `/api/downloads/<id>` | Récupérer / annuler |
| `DELETE` | `/api/downloads/<id>/remove` | Supprimer une entrée terminée/échouée |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Suivre / stream SSE |
| `GET` | `/api/models` | Modèles découverts (inclut `repo_id` quand connu ; les modèles archivés sont marqués `archived` avec `restore_path`) |
| `POST` | `/api/models/delete` | Supprimer du disque (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Lire le nombre de couches depuis les métadonnées GGUF |
| `GET` | `/api/disk-space` | Espace libre/utilisé du volume de modèles |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Récupérer / sauvegarder / supprimer un préréglage (`GET /api/presets` liste tout) |
| `GET` | `/api/engines` | Les moteurs d'inférence de ce node : disponibilité (avec raison), capacités, champs de lancement, catalogue de modèles de Strata |
| `GET` | `/api/archive` | État de l'archive (`enabled`, `available`, `reason`, `free_bytes`) et travaux de déplacement |
| `POST` | `/api/archive` | Archiver un modèle : `{"path": "/models/..."}` (déplace tout son dossier). 409 quand en cours d'utilisation |
| `POST` | `/api/archive/restore` | Restaurer un modèle archivé : `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Annuler un déplacement en file/en cours (source conservée), ou supprimer un terminé |
| `POST` | `/api/engines/strata/download` | Pré-télécharger les shards d'un modèle Strata (`{"model": "strata/qwen-IQ2_XS"}`, `hf_token_id` / `speed_limit_mbps` optionnels) |

Corps de téléchargement : `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Un `filename` vide télécharge tout le dépôt.

Les instances et les préréglages acceptent un `engine` optionnel (`llamacpp` par défaut, ou `strata` ; déduit d'un chemin de modèle `/strata/...`) plus, pour Strata, `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) et `strata_low_ram` (`auto`/`on`/`off`). Les chemins de modèles Strata sont virtuels : `/strata/<family>-<SIZE>`, p. ex. `PUT /api/presets/strata/qwen-IQ2_XS`. Voir [Strata](STRATA.md).

<a id="settings-system-request-log"></a>
## Réglages, système, journal des requêtes

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Récupérer / sauvegarder les réglages globaux |
| `GET` | `/api/system-info` | Utilisation CPU, nombre de cœurs, utilisation RAM |
| `GET` | `/api/gpu-info` | VRAM et utilisation par GPU (natif ; se rabat sur l'exec conteneur) |
| `GET` | `/health` | `{"status": "ok"}` - toujours ouvert, aucune authentification requise |
| `GET` | `/api/request-log/conversations` | Conversations récentes avec métadonnées agrégées (`limit` en query, défaut 100, max 500) |
| `GET` | `/api/request-log/conversations/<id>` | Tous les tours enregistrés d'une conversation, du plus ancien au plus récent |
| `GET` | `/api/request-log/stats` | Métriques agrégées (tokens, tokens/s moyen/crête, TTFT, latence, erreurs, comptes streamés). `inst_id` + `window_hours` optionnels |

<a id="ollama--openai-compat"></a>
## Compatibilité Ollama et OpenAI

Ollama : `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate auto-démarrent).
OpenAI : `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat auto-démarre).
Decision models : `/v1/systemone` (auto-démarre) - voir [Modèles de décision](MODELS.md#decision-models).
Base de connaissances : `/api/kb/*` (réservé aux admins) et le point de terminaison MCP `/mcp/knowledge` - voir [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Les endpoints node-à-node du cluster utilisent `X-Cluster-Secret` - voir [Clustering](CLUSTERING.md).
