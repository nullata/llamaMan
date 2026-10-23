# Clustering

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

Quelques réglages sont par node car ils sont spécifiques à l'hôte : les **Docker images** suivies et les interrupteurs d'éviction du plafond de modèles. Tout le reste est partagé à l'échelle du cluster. Les alias de groupes en file partagée actifs sont annoncés comme des modèles sélectionnables dans `/api/tags` et `/v1/models` (dédupliqués à l'échelle du cluster), donc un client peut envoyer l'alias et le voir routé vers le node le moins chargé qui le sert.

**La longueur de contexte des groupes est à l'échelle du cluster.** `/api/show` et `/v1/models` rapportent le contexte d'un groupe de file comme le **minimum sur tous les membres actifs**, quel que soit le node interrogé. Un groupe dont les membres ont des valeurs de ctx différentes annonce la plus petite afin qu'un prompt dimensionné pour elle soit sûr pour toute cible de dispatch — la marge supplémentaire du membre le plus grand reste simplement inutilisée. Un node sans membre local du groupe répond quand même (`context_length` rempli depuis le ctx runtime du peer dans l'état d'instance partagé, ou depuis la ligne de préréglage dans la DB partagée) ; un groupe sans membre actif nulle part renvoie quand même 404 / aucune entrée. La carte d'instance de l'UI opérateur ajoute `ctx <min>, capped by <node/model>` quand le ctx de ce membre dépasse le minimum annoncé, afin de voir quel peer tire le groupe vers le bas.

> **Sécurité :** le secret cluster permet à n'importe quel peer de piloter des actions sur ce node. Faites circuler le trafic node-à-node sur un réseau de confiance ou derrière TLS.

Variables d'environnement liées au cluster : [Clustering variables](CONFIGURATION.md#clustering).
