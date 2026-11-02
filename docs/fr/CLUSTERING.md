# Clustering

> **Remarque :** ceci est une traduction et peut être en retard sur la [version anglaise](../en/CLUSTERING.md), qui fait foi.

*Optionnel, désactivé par défaut - les installations mono-node ne sont absolument pas affectées.*

Plusieurs déploiements llamaMan agissent comme **un seul cluster logique** : un unique tableau de bord agrégeant les GPU, instances et téléchargements de chaque node, avec des lancements/pulls/téléchargements inter-nodes et un équilibrage de charge multi-node en file partagée. Les nodes se découvrent mutuellement via le backend de stockage partagé - aucun échange de clés par paires.

**Prérequis :**

- **Backend de stockage partagé** - chaque node pointe vers le **même** `DATABASE_URL` (MariaDB/MySQL, voir [Backends de stockage](CONFIGURATION.md#storage-backends)). JSON est par hôte et ne peut pas être partagé.
- **Un `LLAMAMAN_NODE_NAME` unique par node** - l'identité cluster de chaque node (et de toute façon requis pour chaque installation).
- **Le même `CLUSTER_SECRET` sur chaque node** - le jeton bearer (envoyé en `X-Cluster-Secret`) pour tout le HTTP node-à-node.
- **`CLUSTER_ADVERTISE_URL` par node** pour les *actions* inter-nodes - un nom d'hôte/IP routable depuis les **autres** hôtes (pas `localhost`), p. ex. `http://srv1:5000`. Un node sans celui-ci apparaît dans le tableau de bord mais est en consultation seule.

Définis sur **chaque** node (seuls `LLAMAMAN_NODE_NAME` et `CLUSTER_ADVERTISE_URL` diffèrent) :

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

Chaque node envoie un heartbeat toutes les ~5 s ; un node silencieux au-delà de `CLUSTER_NODE_ONLINE_WINDOW_S` (45 s par défaut) s'affiche hors ligne. Consultez et gérez sous **Settings → Cluster**.

Quelques réglages sont par node car ils sont spécifiques à l'hôte : les **Docker images** suivies, les interrupteurs d'éviction du plafond de modèles et les [réglages de réveil sous charge](#waking-under-load). Tout le reste est partagé à l'échelle du cluster. Les alias de groupes en file partagée actifs sont annoncés comme des modèles sélectionnables dans `/api/tags` et `/v1/models` (dédupliqués à l'échelle du cluster), donc un client peut envoyer l'alias et le voir routé vers le node le moins chargé qui le sert.

**La longueur de contexte des groupes est à l'échelle du cluster.** `/api/show` et `/v1/models` rapportent le contexte d'un groupe de file comme le **minimum sur tous les membres actifs sur des nodes pouvant recevoir du trafic**, quel que soit le node interrogé. Les membres sur des nodes hors ligne ou en affichage seul (view-only) ne comptent pas. Un groupe dont les membres ont des valeurs de ctx différentes annonce la plus petite afin qu'un prompt dimensionné pour elle soit sûr pour toute cible de dispatch — la marge supplémentaire du membre le plus grand reste simplement inutilisée. Un node sans membre local du groupe répond quand même (`context_length` rempli depuis le ctx runtime du peer dans l'état d'instance partagé, ou depuis la ligne de préréglage dans la DB partagée) ; un groupe sans membre actif nulle part renvoie quand même 404 / aucune entrée. La carte d'instance de l'UI opérateur ajoute `ctx <min>, capped by <node/model>` quand le ctx de ce membre dépasse le minimum annoncé, afin de voir quel peer tire le groupe vers le bas.

> **Sécurité :** le secret cluster permet à n'importe quel peer de piloter des actions sur ce node. Faites circuler le trafic node-à-node sur un réseau de confiance ou derrière TLS.

Variables d'environnement liées au cluster : [Clustering variables](CONFIGURATION.md#clustering).

<a id="queue-groups"></a>
## Groupes de files d'attente

Un groupe de files rassemble des instances sous un **Nom de groupe de files (Queue group name)** afin qu'un client puisse envoyer ce nom à n'importe quel node. Le tableau montre ce qui arrive à chaque membre du point de vue d'un autre node :

| État du membre | Listé dans `/api/tags` et `/v1/models` | Reçoit le trafic routé |
|---|---|---|
| Healthy | Oui | Premier choix |
| En démarrage (chargement) | Oui | Après les membres healthy |
| Endormi (Sleeping) | Oui | En dernier, et seulement après chaque membre éveillé, fallbacks compris. La requête le réveille. |
| En arrêt ou arrêté | Non | Jamais |
| Sur un node sans `CLUSTER_ADVERTISE_URL` (affichage seul) | Non | Jamais |
| Sur un node hors ligne (aucun heartbeat depuis 45s) | Non | Jamais |

**`CLUSTER_ADVERTISE_URL` est requis pour recevoir du travail routé.** Sans cela, un node peut toujours envoyer des requêtes à ses pairs, mais aucune ne peut l'atteindre.

**Dormir et arrêter ont des significations différentes ici.** Un idle timeout met une instance en sommeil, et un membre endormi se réveille lorsque le trafic le nécessite. Un membre arrêté reste arrêté : un arrêt par l'opérateur, une éviction ou un crash se terminent là, et aucune requête d'un autre node ne le redémarre. Interroger le node qui le détient renvoie un `503`, sauf si le nom est également le propre fichier de ce membre, ce qui le réveille quand même.

<a id="waking-under-load"></a>
### Réveil sous charge

Les membres éveillés mettent les requêtes en file d'attente plutôt que de les refuser, de sorte que le trafic seul n'atteindrait un membre endormi qu'après le débordement complet d'une file. llamaMan en réveille donc un par anticipation. Lorsque chaque membre sain du groupe est saturé, que des requêtes sont en attente et que rien n'est déjà en cours de chargement, pendant 3 sondages consécutifs (environ 15 secondes), un membre endormi se réveille. À l'échelle du cluster, seul le node portant le nom le plus bas abritant un membre endormi effectue cette action, et au plus un membre par groupe par sondage. Un membre principal (primary) se réveille avant un fallback.

Seul un membre endormi apportant de la capacité compte : il doit avoir un **Max Concurrent** supérieur à `0`, et il ne peut pas partager une file avec un membre éveillé sur le même node.

Le réveil est ignoré, avec une ligne de log, si les poids du modèle ne rentrent pas dans la VRAM libre du node. La vérification additionne la mémoire libre sur tous les GPU et ne compte que les poids, de sorte qu'un modèle qui passe le contrôle peut quand même échouer au chargement. Si la VRAM ne peut pas être lue, ou si le modèle n'a pas de fichier à mesurer (modèles virtuels de Strata), le réveil se poursuit.

Les deux réglages sont configurables par node via `POST /api/settings` :

| Réglage | Par défaut | Signification |
|---|---|---|
| `eager_wake_enabled` | `true` | Si ce node réveille ses membres endormis sous charge |
| `eager_wake_polls` | `3` | Nombre de sondages saturés consécutifs (5 secondes chacun) avant réveil |

<a id="when-a-group-cant-be-served"></a>
### Quand un groupe ne peut pas être servi

Si d'autres nodes hébergent le groupe mais qu'aucun ne peut accepter une requête, le client reçoit une erreur nommant chaque node et la raison, plutôt que "model not found" :

| Statut | Code (routes OpenAI) | Signification |
|---|---|---|
| `429` | `group_full` | La file d'au moins un peer est pleine. Réessayer peut fonctionner. |
| `503` | `group_unavailable` | Les pairs sont hors ligne, en lecture seule, inatteignables ou pas prêts. |

Le node propriétaire renvoie ces codes `503` pour ses propres membres, et un node d'entrée essaie son membre suivant lorsqu'il en reçoit un :

| Code | Signification |
|---|---|
| `model_stopping` | Le membre est en cours d'arrêt |
| `group_stopped` | Chaque membre sur ce node est arrêté |
| `wake_failed` | Le réveil a échoué |
| `wake_timeout` | Le membre n'est pas devenu healthy à temps |

Un nom de groupe ne lance pas un fichier local dont le nom le contient simplement. Un fichier portant exactement ce nom s'exécute toujours.

<a id="troubleshooting-a-group-that-isnt-visible"></a>
### Dépannage d'un groupe non visible

- **Le backend JSON nécessite un unique `DATA_DIR` partagé.** Les nodes se découvrent via le registre, donc des répertoires distincts signifient qu'aucun node ne voit de pairs. Utilisez MariaDB à moins que le répertoire ne soit réellement partagé.
- **Une base de données dégradée masque volontairement les pairs.** Lorsque le miroir DB sert les lectures, un node ne voit que lui-même et dessert localement.
- **Un échec de sonde de charge ne masque pas un groupe.** Un peer qui ne répond pas reste routable d'après son snapshot (jusqu'à 45 secondes d'ancienneté), mais se classe derrière chaque peer ayant répondu. Le journal enregistre le moment où les sondes vers un peer commencent à échouer et celui où elles récupèrent.
