# Dépannage

Le tableau complet symptôme → correctif. Le [README](../../README.md#troubleshooting) garde les neuf lignes les plus courantes au premier lancement ; tout le reste est ici.

| Symptôme | Correctif |
|---|---|
| Instance bloquée en **starting** | Consultez les logs via le bouton Logs. Causes fréquentes : OOM, faute de frappe dans le chemin du modèle, GGUF corrompu, image non téléchargée |
| _« Docker image not found »_ | Téléchargez l'image correspondante : `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (ou `server-rocm` / `server-sycl` / `server`) |
| _« Docker API error »_ au lancement | Vérifiez que `/var/run/docker.sock` est monté (par défaut dans `docker-compose.yml`) |
| Pas de GPU / erreur CUDA | Vérifiez que le NVIDIA Container Toolkit est installé et que `docker run --gpus all` fonctionne sur l'hôte |
| Pas de GPU / erreur ROCm | Vérifiez que `/dev/kfd` et `/dev/dri` existent et que votre utilisateur est dans `video`/`render` |
| Pas de GPU / erreur Intel Arc | Vérifiez que `/dev/dri` est accessible et que votre utilisateur est dans `video`/`render` |
| L'image Vulkan échoue avec _« Unable to find group render »_ | Définissez `GPU_TYPE=vulkan` pour que llamaMan attache les GIDs render/video de l'hôte numériquement (le group_add par nom échoue sur les images dont /etc/group ne contient pas d'entrée `render`) |
| Statistiques GPU indisponibles | NVIDIA : décommentez le bloc `deploy.resources.reservations`. AMD/Intel : vérifiez que `/sys/class/drm:ro` est monté |
| Mauvais fabricant de GPU détecté | Définissez `GPU_TYPE=cuda`/`rocm`/`intel` pour forcer |
| Instance bloquée en **starting** en mode bare-metal | Le conteneur est healthy mais llamaman ne l'atteint pas. Définissez `LLAMAMAN_IN_DOCKER=false`/`true` explicitement si l'auto-détection est erronée pour votre environnement |
| Strata : _« Docker image 'strata:latest' not found »_ | Téléchargez ou construisez une image Strata dans Settings → Docker Images (ou `docker build -t strata .` dans un clone du dépôt Strata). Voir [Strata](STRATA.md) |
| Modèles Strata non listés | Définissez `STRATA_ENABLED=true` ; le node doit être NVIDIA (`GPU_TYPE=cuda` si la détection échoue). `GET /api/engines` affiche la raison |
| Instance Strata bloquée en **starting** | Normal pendant plusieurs minutes au premier lancement (téléchargement + pack). La carte affiche l'étape ; le log montre le setup de Strata. « setup failed » signifie que son setup s'est arrêté - lisez la ligne `[X]` du log |
| Le modal de statistiques est vide | Activez **Settings → App Settings → Request recording** (voir [Enregistrement des requêtes et statistiques](CONFIGURATION.md#request-recording--stats)) |
| Le lancement échoue avec une erreur GPU/CDI sur un hôte sans passthrough GPU | Mettez **GPU Layers** à `0` pour du CPU-only sans périphérique GPU attaché, ou corrigez le runtime GPU |
| Conflit de port | Le formulaire suggère automatiquement un port libre ; ajustez si besoin |
| Modèle absent d'OpenWebUI | Vérifiez `OLLAMA_BASE_URL=http://llamaman:42069`. Vérifiez que `/api/tags` renvoie des modèles |
| OpenWebUI reçoit 401 | `require_auth` est activé (par défaut). Créez une clé API et définissez `OPENAI_API_KEYS` dans OpenWebUI |
| Le client OpenAI reçoit 503 quand les modèles sont pleins | Le plafond `LLAMAMAN_MAX_MODELS` est atteint et l'API OpenAI ne peut pas évincer par défaut. Relevez le plafond ou activez les interrupteurs d'éviction : [Éviction de modèles](MODELS.md#model-eviction) |
| Conteneurs non nettoyés après un stop | llamaMan supprime les conteneurs au stop. Orphelins après un crash : `docker ps --filter name=llamaman-`, ou redémarrez llamaMan (l'adoption des orphelins s'exécute au démarrage) |
| Le client (Hermes / OpenWebUI / etc.) rapporte la fenêtre de contexte entraînée au lieu du plafond du préréglage | Passez à la 1.1.2+. `/api/ps` inclut désormais `context_length` défini au ctx runtime avec lequel l'instance a été lancée, et `model_info["<arch>.context_length"]` de `/api/show` est surchargé avec le plafond effectif |
| Un client qui interroge un nom de queue-group obtient son propre ctx par défaut (p. ex. 128k) au lieu du ctx réel du déploiement | `/api/show` et `/v1/models` répondent désormais pour un queue-group sur n'importe quel node en cluster — même sans membre local — en utilisant le ctx runtime du peer depuis l'état partagé. Les groupes multi-membres rapportent le minimum sur les membres actifs afin que la valeur annoncée soit sûre pour toute cible de dispatch. Voir [Clustering](CLUSTERING.md) |
