<a id="knowledge-base--mcp"></a>
# Base de connaissances et MCP

Stockez des documents groupés par **topic**, et llamaman les découpe en morceaux (~1200 caractères, alignés sur les paragraphes, 200 de recouvrement), calcule leurs embeddings et stocke les vecteurs dans votre base MariaDB (colonnes `VECTOR` + un index ANN en cosinus). N'importe quel client MCP (Claude Desktop, Claude Code, agents) peut alors les rechercher par sens. Tout tourne sur le matériel que vous faites déjà tourner : les embeddings viennent d'une instance llama.cpp lancée avec **Embedding model** activé, les vecteurs vivent dans votre base de données, et llamaman sert lui-même le point de terminaison MCP.

**Prérequis :** le backend MariaDB (`DATABASE_URL`, voir [Backends de stockage](CONFIGURATION.md#storage-backends)) sur serveur **11.8+**, qui est là où arrive `VECTOR`. Sur le backend JSON ou un MariaDB plus ancien, l'onglet MCP explique pourquoi la base de connaissances n'est pas prise en charge et l'activation est refusée ; rien d'autre ne change. Il n'y a pas de migration : les tables `kb_*` sont créées à la première utilisation.

**Configuration :**

1. Lancez une instance avec **Embedding model** activé. N'importe quel GGUF d'embedding fonctionne (`bge-m3`, `nomic-embed-text`, …).
2. **Settings → MCP** → activez **Knowledge base** et choisissez le modèle sous **Embedding model**. Le menu déroulant liste tous les modèles avec embedding activé à travers le cluster.
3. Ajoutez des documents avec l'outil MCP `kb_ingest_document` (activez **Allow ingest over MCP**) ou `POST /api/kb/documents` depuis une session connectée. L'onglet liste, filtre et supprime les topics et propose une recherche de test, mais pas encore de formulaire d'ajout.
4. Activez **Expose over MCP**. L'onglet affiche l'URL du point de terminaison (`http://<your-host>:42069/mcp/knowledge`), et **Copy config.json** copie une configuration client prête à coller :

```json
{
  "mcpServers": {
    "llamaman-kb": {
      "type": "http",
      "url": "http://my-nas:42069/mcp/knowledge",
      "headers": { "Authorization": "Bearer <your-api-key>" }
    }
  }
}
```

Le bloc `headers` n'est nécessaire que lorsqu'une clé est requise (voir les modes d'accès ci-dessous).

**Outils.** `kb_list_topics`, `kb_list_documents`, `kb_search` (passages les meilleurs d'abord, score 0–1) et `kb_get_document` sont toujours disponibles. `kb_ingest_document` (crée le topic s'il manque ; 200 Ko par appel ; renvoyer un contenu identique est sans effet) n'apparaît qu'avec **Allow ingest over MCP** activé. `kb_delete_document` / `kb_delete_topic` n'apparaissent qu'avec **Allow delete over MCP** activé. Supprimer un topic supprime ses documents, et les suppressions sont irréversibles.

**Modes d'accès** (**MCP access**) :

- **Global** (par défaut) : chaque client travaille sur le **pool partagé**, le voit et, si ingest/delete sont autorisés, le modifie. Les nouveaux topics vont dans le pool partagé. L'authentification suit **Require Authentication** (onglet API Keys) : activé signifie qu'une clé API valide est requise, désactivé signifie que le point de terminaison est ouvert.
- **Per API key** : chaque requête a besoin d'une clé API valide, même avec Require Authentication désactivé. Les topics créés par une clé sont **privés** pour cette clé, et deux clés peuvent chacune avoir un topic du même nom. Chaque clé voit aussi le pool partagé en **lecture seule** : elle peut le rechercher et le lire, mais ni y ajouter (ingest) ni y supprimer. Les topics et documents des autres clés reviennent comme « not found ».

Changer de mode ne déplace ni ne supprime jamais de données, et les topics privés restent privés dans les deux modes. Passer de Global → Per API key transforme tout ce qui existe en pool partagé. Passer de Per API key → Global masque les topics privés pour tous les clients jusqu'à ce que vous reveniez en arrière. L'onglet MCP affiche toujours tout, en étiquetant chaque topic `shared` ou du nom de la clé propriétaire. `POST /api/kb/topics` peut aussi créer un topic privé pour une clé donnée (`owner_key_id`), utile quand l'ingest MCP est désactivé.

**Modèle d'embedding.** La sélection se fait par nom de modèle, pas par instance. Chaque appel d'embedding s'exécute sur la première option disponible :
1. une instance healthy sur ce node ;
2. une endormie ou arrêtée sur ce node, qui est réveillée (ou une déjà en démarrage, que l'appel attend) ;
3. un node peer qui a **Knowledge base** et **Expose over MCP** activés et qui possède le modèle (une instance endormie là-bas est réveillée) ;
4. un relancement sur le node qui l'a servi en dernier, au plus 3 tentatives par 10 minutes.

Le premier embedding enregistre la dimension du vecteur. Un modèle avec une dimension différente est refusé tant que vous n'appuyez pas sur **Re-embed everything**, un travail en arrière-plan qui reconstruit les morceaux de chaque document (un travail à la fois, gardé par un verrou DB à l'échelle du cluster). Seule la dimension est vérifiée, donc relancez vous-même le ré-embedding quand vous changez entre deux modèles de même dimension.

**Notes sur le cluster.** Les topics et documents sont des données à l'échelle du cluster dans la base partagée, pas par node. Chaque node voit la même base de connaissances, et le point de terminaison MCP de n'importe quel node la sert. Voir [Clustering](CLUSTERING.md).

**Quand désactivé**, `/mcp/knowledge` renvoie 404 dans les deux modes d'accès, et aucune requête ne touche les tables de la base de connaissances.

> **Sécurité :** `/api/kb/*` est réservé aux admins : les sessions UI connectées et les peers du cluster sont acceptés, les requêtes authentifiées par une clé API reçoivent 403. Les clés API ne sont toutefois pas limitées à la base de connaissances : une clé que vous donnez à un client MCP peut aussi appeler les endpoints d'inférence et la plupart des autres routes `/api/*`. Utilisez une clé dédiée par client et révoquez-la en cas de doute.
