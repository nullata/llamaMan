# クラスタリング

*オプション機能、デフォルトは無効 — シングルノードのインストールには一切影響しません。*

複数の llamaMan デプロイメントが**1つの論理クラスタ**として機能します。単一のダッシュボードですべてのノードの GPU、インスタンス、ダウンロードを集約し、ノード間の起動/プル/ダウンロードや、複数ノードの共有キューによる負荷分散を実現します。ノード同士は共有ストレージバックエンドを介して互いを検出するため、ペアごとのキー交換は不要です。

**要件:**

- **共有ストレージバックエンド** - すべてのノードが**同一の** `DATABASE_URL`（MariaDB/MySQL、[Storage Backends](CONFIGURATION.md#storage-backends) を参照）を指定していること。JSON バックエンドはホスト単位であり共有できません。
- **ノードごとに一意の `LLAMAMAN_NODE_NAME`** - 各ノードのクラスタ識別子（すべてのインストールで設定必須）。
- **すべてのノードで同一の `CLUSTER_SECRET`** - ノード間のすべての HTTP 通信で使用される Bearer トークン（`X-Cluster-Secret` として送信）。
- **クロスノード*アクション*用の各ノードの `CLUSTER_ADVERTISE_URL`** - **他の**ホストからルーティング可能なホスト名/IP（`localhost` は不可、例: `http://srv1:5000`）。これが設定されていないノードはダッシュボードに表示されますが、表示専用（view-only）となります。

**各**ノードで設定（`LLAMAMAN_NODE_NAME` と `CLUSTER_ADVERTISE_URL` のみが異なります）:

```yaml
environment:
  - LLAMAMAN_NODE_NAME=srv1                 # unique per node
  - DATABASE_URL=mysql+pymysql://llamaman:pass@db-host:3306/llamaman   # identical
  - CLUSTER_ENABLED=true
  - CLUSTER_SECRET=a-long-shared-random-secret   # identical
  - CLUSTER_ADVERTISE_URL=http://srv1:5000  # this node's address, routable from peers
```

各ノードは約 5 秒ごとにハートビートを送信します。`CLUSTER_NODE_ONLINE_WINDOW_S`（デフォルト 45 秒）を超えて応答がないノードはオフラインとして表示されます。**Settings → Cluster** で確認および管理できます。

ホスト固有であるため、一部の設定はノード単位でスコープされます。追跡対象の **Docker images** とモデル上限エビクションのトグルです。それ以外のすべてはクラスタ全体で共有されます。アクティブな共有キューグループのエイリアスは、選択可能なモデルとして `/api/tags` および `/v1/models` で通知（クラスタ全体で重複排除）されるため、クライアントがエイリアスを送信すると、それを提供している最も負荷の低いノードへルーティングされます。

**グループのコンテキスト長はクラスタ全体で統一されます。** クライアントがどのノードに問い合わせても、`/api/show` および `/v1/models` はキューグループのコンテキストを**すべてのアクティブメンバーの最小値（min）**としてレポートします。異なる ctx 値のメンバーを持つグループは小さい方の値を通知するため、そのサイズに合わせたプロンプトはどのディスパッチ先でも安全です — 大きいメンバーの追加の余白は単に使用されません。グループのローカルメンバーを持たないノードでも応答します（`context_length` は共有インスタンス状態のピアのランタイム ctx、または共有 DB のプリセット行から設定されます）。どこにもアクティブなメンバーが存在しないグループは 404 / エントリなしを返します。このメンバーの ctx が通知された最小値を超えている場合、オペレーター UI のインスタンスカードには `ctx <min>, capped by <node/model>` と表示され、どのピアがグループのコンテキストを引き下げているかを確認できます。

> **セキュリティ:** クラスタシークレットにより、任意のピアがこのノード上でアクションを実行できるようになります。ノード間トラフィックは、信頼できるネットワーク上または TLS の背後で実行してください。

クラスタ関連の環境変数: [Clustering variables](CONFIGURATION.md#clustering)。
