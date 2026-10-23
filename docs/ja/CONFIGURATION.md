# 設定とストレージ

ベアメタルインストール、認証モデルの詳細、GPU 統計情報、リクエストの記録、ストレージバックエンド、および完全な環境変数リファレンス。

## 目次

- [ベアメタル実行](#running-bare-metal) · [認証の詳細](#authentication-detail) · [GPU 統計情報](#gpu-統計情報) · [リクエストの記録と統計](#request-recording--stats) · [ストレージバックエンド](#storage-backends) · [データベース障害の克服 (ローカルミラー)](#surviving-a-database-outage-local-mirror) · [環境変数](#environment-variables)

<a id="running-bare-metal"></a>
## ベアメタル実行

マネージャー自体を Docker 内で実行するのが不便な開発環境やホスト（例: WSL）で便利です。llama-server は引き続きコンテナ内で実行されます。

```bash
pip install -r requirements.txt
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev python app.py
# or under gunicorn:
MODELS_DIR=./models DATA_DIR=./data LOGS_DIR=./logs \
  LLAMAMAN_NODE_NAME=dev gunicorn -c gunicorn.conf.py app:app
```

gunicorn 下の単一プロセスで `:5000` の UI/API と `:42069` のプロキシを提供します（gunicorn を使わない `python app.py` は `:5000` のみで Flask 開発サーバーを実行します）。コンテナ実行かベアメタル実行かは自動検出されます。実行環境に対して検出が誤っている場合は、明示的に `LLAMAMAN_IN_DOCKER=true` または `false` を設定してください。

<a id="authentication-detail"></a>
## 認証の詳細

2 種類の認証情報:

- UI 用の **セッション cookie**（初回起動時に `/setup` 経由で作成）
- 外部クライアント用の **API キー**（`Authorization: Bearer llm-xxxxxxxxxx`）、UI の **API Keys** で管理

**"Require authentication for all endpoints"** トグル（デフォルトで有効）は、モデル配信エンドポイントで Bearer トークンを要求するかどうかを制御します:

| トグル | モデルエンドポイント (`/api/chat`、`/v1/chat/completions`、インスタンス別ポート) | 管理エンドポイント (`/api/instances` など) |
|--------|--------------------------------------------------------------------------|-----------------------------------------------|
| **ON** (デフォルト) | Bearer トークン必須 | Bearer トークンまたはセッション |
| **OFF** | オープン | Bearer トークンまたはセッション |

3 つのポート公開面（5000、42069、8000-8020）はすべて同じ認証フックを通過します。

## GPU 統計情報

GPU ごとの VRAM と使用率をネイティブに照会します - llama-server が実行中である必要はありません。

| ベンダー | メソッド | 要件 |
|---|---|---|
| NVIDIA | `nvidia-ml-py` (NVML 直接呼び出し) | NVIDIA toolkit の `utility` ケーパビリティを利用するため、`docker-compose.yml` 内の `deploy.resources.reservations` のコメントアウトを解除 |
| AMD | `/sys/class/drm` sysfs | `/sys/class/drm:ro` ボリュームマウント（デフォルトで含まれています） |
| Intel Arc | `/sys/class/drm` sysfs | AMD と同一のマウント |

ネイティブアクセスがない場合、llamaMan は実行中のコンテナ内で `nvidia-smi` / `rocm-smi` を実行することにフォールバックします。統計情報は常にホスト GPU 全体の状態を反映します。

<a id="request-recording--stats"></a>
## リクエストの記録と統計

**Settings → App Settings → Request recording** で設定: **Off**（デフォルト）/ **Per request** / **Per conversation**（システムプロンプト + 最初のユーザーメッセージのコンテンツハッシュによってターンをグループ化）。各レコードはリクエスト本文 + エンベロープフィールド（モデル、エンドポイント、ステータス、所要時間、トークン数）に加え、**正確なターン別メトリクス**（生成ウィンドウ全体で測定されたスループット、TTFT）を記録します。レコードは `RECORDINGS_DIR` (JSON) または `request_log` テーブル (MariaDB) に保存されます。**Retention (days)** により毎時プルーニング（古いレコードの削除）が行われます（`0` = 永久に保持）。

各インスタンスカードには、リクエストログから集計された **Stats** ボタン（リクエスト数、平均/ピークスループット、平均 TTFT/レイテンシ、合計トークン数、アクティブ期間）が表示されます。そのため、インスタンス停止後も保持され、記録が無効な場合は空の状態で表示されます。ヘッダーの **Logging** リンクをクリックすると全画面ダッシュボード（サマリータイル、最近の会話、会話別の詳細ドリルダウン）が開き、24h / 7d / 30d / すべて の時間枠セレクターと、会話リストに対するモデル別および日付別のフィルターが利用できます。

<a id="storage-backends"></a>
## ストレージバックエンド

### JSON (デフォルト)

設定不要（ゼロコンフィグ）。`DATA_DIR` (`/data`) 配下のファイル: `state.json`、`presets.json`、`users.json`、`settings.json`、`api_keys.json`、`request_log/`。インスタンスログとダウンロードログは `LOGS_DIR` (`/tmp/llama-logs`) に出力されます。

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

初回接続時にテーブルが自動作成されます。リクエストログは JSON の `request_log/` ディレクトリではなく `request_log` テーブルに保存されます（MariaDB 使用時は `RECORDINGS_DIR` は効果がありません）。

ノードごとのモデルメタデータ（ソースリポジトリ + コンテンツハッシュ）は、`(node_id, model_path)` をキーとするノードスコープの `model_files` テーブルに保持されます。アップグレード時、各ノードは従来の共有設定 blob から自身のエントリ（自身のディスクに存在するパスのみ）をコピーし、読み取りフォールバックおよびロールバックパスとして blob を保持します。マイグレーションはノードごとにバージョン管理されます。

> **InnoDB に関する注意:** `(node_id, model_path)` は utf8mb4 環境下で `(64 + 700) × 4 = 3056` バイトとなり、InnoDB の 3072 バイトインデックス制限の直下となるため、テーブルは `ROW_FORMAT=DYNAMIC` で作成されます。

MariaDB は、[クラスタリング](CLUSTERING.md) および [ナレッジベース](KNOWLEDGE-BASE.md) を利用するための前提条件でもあります。

<a id="surviving-a-database-outage-local-mirror"></a>
## データベース障害の克服 (ローカルミラー)

*オプション、デフォルトでは無効。*

`DATABASE_URL` が設定されている場合、データベースは**すべて**のリクエストのクリティカルパス上に位置します（認証チェックは呼び出しごとに設定を読み取り、API キーを検証します）。データベースに接続できなくなるとノードは配信を停止し、障害の最中にコンテナが再起動されるとまったく起動できなくなります。

**Settings → App settings → "Keep a local mirror of the database"** を有効にするか（または `LLAMAMAN_DB_MIRROR=1` を設定すると）、ノードは `DATA_DIR/db_mirror/` にライトスルーコピーを保持します。

| オフライン中も動作するもの | オフライン中にブロックされるもの |
|---|---|
| すべての推論（両ポート） | 最初のユーザー作成 (`/setup`) - 新規インストールではそもそもフォールバック先のミラーが存在しないため |
| モデルの起動 / 停止 / スリープ / 復帰 | |
| ダウンロード、モデルのハッシュ化、更新チェック | |
| **プリセット** の作成と編集（ダウンロードしたばかりのモデル向けを含む） | |
| **API キー** の作成と失効 | |
| ダウンロードおよびアプリ **設定** の変更 | |
| **Hugging Face トークン** の追加と削除 | |

オフライン中の変更は追記専用ジャーナルに記録され、再接続時に再生（リプレイ）されます（10 秒ごとのプローブ）。リプレイは結果の行ではなく*編集内容*を記録するため、障害中に別のノードが行った変更も残ります（異なるプリセットフィールド、異なる設定キー、異なる HF トークンはすべてそのまま維持されます）。

**要件と注意点:**

- **`DATA_DIR` は永続ボリュームである必要があります**（サンプルの compose ファイルでは設定済み）。
- **各ノードに専用の `DATA_DIR` が必要です。** ミラーには所有するノード ID が刻印されます。2 つのノードが 1 つを共有すると、どちらかのビューが破損するのを防ぐためにミラーリングが無効化されます。
- **シークレットがローカルディスクに保存されます。** 設定はそのままコピーされます。**Hugging Face トークンはプレーンテキストでミラーリングされます。** パスワードと API キーは元々ハッシュとしてのみ保存されています。シークレットを個々のホストに置かないために MariaDB を選択した場合は、このトレードオフがあるため本機能はオプトインになっています。
- **障害中はノード間の負荷分散が停止します。** ピアの生存情報はデータベースに保存されているため、縮退状態のノードは自分自身しか認識できません。
- **API キーの変更は再接続までローカルに留まります。** オフラインで作成されたキーはこのノードでのみ機能します。オフラインで**失効**されたキーは、再接続まで他のノードで有効なままになります。失効はその点を考慮して計画してください。
- オフライン中は**リクエストログはバッファリングされず破棄されます**（レコードには完全なリクエスト/レスポンス本文が含まれており、長時間の障害中にそれらをスプールするとボリュームがいっぱいになる可能性があるため）。
- **障害の最中にミラーをオフにする操作は、再接続まで保留されます**。
- 縮退状態の最中にスキーママイグレーションが実行されることはありません。再接続後、ジャーナルのリプレイ前に本物のデータベースに対して実行されます。

<a id="environment-variables"></a>
## 環境変数

### コア

| 変数名 | デフォルト値 | 説明 |
|---|---|---|
| `LLAMAMAN_NODE_NAME` | *(必須)* | **必須。** 一意で安定した識別子 - 保存された状態のパーティションキーおよびクラスタ識別子。任意の文字列。一度決めたら変更しないでください（後で変更すると保存された状態が孤立します）。 |
| `MODELS_DIR` | `/models` | モデルファイルをスキャンするディレクトリ（コンテナ内パス） |
| `DATA_DIR` | `/data` | 永続的な設定/状態（JSON ファイル）用のディレクトリ |
| `RECORDINGS_DIR` | `{DATA_DIR}/request_log` | 会話ごとのリクエストログレコード用のディレクトリ。JSON バックエンドのみ。`DATABASE_URL` が設定されている場合は無視されます |
| `LOGS_DIR` | `/tmp/llama-logs` | インスタンスおよびダウンロードログ用のディレクトリ（コンテナ内パス） |
| `HOST_MODELS_DIR` | *(`MODELS_DIR` と同じ)* | モデルボリュームの**ホスト側**絶対パス。`-v /host/path:/models` の左側と一致している必要があります - 兄弟コンテナを生成する際に Docker デーモンに渡されます |
| `HOST_LOGS_DIR` | *(`LOGS_DIR` と同じ)* | ログボリュームのホスト側絶対パス。`HOST_MODELS_DIR` と同じ要件 |
| `ARCHIVE_DIR` | *(未設定)* | アーカイブボリュームがマウントされるコンテナ内パス（例: `/archive`）。設定すると [モデルアーカイブ](MODELS.md#model-archive) が有効になり、未設定の場合は非表示になります |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `8000` / `8020` | パブリックな llama-server/プロキシ ポートプール |
| `INTERNAL_PORT_RANGE_START` / `INTERNAL_PORT_RANGE_END` | `9000` / `9020` | プロキシモードが有効な場合に使用される内部ポートプール |
| `LLAMAMAN_PROXY_PORT` | `42069` | Ollama 互換プロキシのポート |
| `LLAMAMAN_MAX_MODELS` | `0` | プロキシ経由で実行可能な同時チャットモデルの最大数（LRU エビクション、0 = 無制限）。ルール: [モデルのエビクション](MODELS.md#model-eviction) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | プロキシ管理インスタンスのアイドルタイムアウト（分単位、0 = 無効） |
| `LLAMAMAN_PDF_MAX_CONCURRENT` | `4` | プロセス全体での最大同時 PDF ラスタライズ数。各ラスタライズは poppler を起動し、高 DPI では数百 MB の一時 RAM を消費します |
| `SECRET_KEY` | *(自動)* | Flask セッションシークレット。未設定の場合は machine-id から自動生成。マルチレプリカデプロイの場合は明示的に設定してください |
| `SESSION_COOKIE_NAME` | `llamaman_session` | セッション cookie 名。同一ホスト上の他の Flask アプリと共存できるよう名前空間化されています - cookie はポートではなく host+path でスコープされます |
| `DATABASE_URL` | *(未設定)* | MariaDB/MySQL 接続文字列。未設定 = JSON ファイル |
| `LLAMAMAN_DB_MIRROR` | *(未設定)* | ローカル DB ミラーの有効化 (`1`) / 無効化 (`0`) を強制し、ノードごとの設定を上書きします。`DATABASE_URL` が設定されている場合のみ意味を持ちます |
| `HEALTH_CHECK_TIMEOUT` | `3` | インスタンスのヘルスチェックタイムアウト（秒） |
| `MODEL_LOAD_TIMEOUT` | `300` | モデルが正常（healthy）になるのを待機する秒数。超巨大モデルの場合は増やしてください |
| `REQUEST_TIMEOUT` | `300` | llama-server へのアップストリームリクエスト、ノード間転送、ゲート取得待機時の**読み取り**タイムアウト（秒）。ピア接続時間（別の 5 秒制限）は制御**しません** |

### Docker / GPU

| 変数名 | デフォルト値 | 説明 |
|---|---|---|
| `LLAMA_IMAGE` | *(自動)* | 起動されるすべてのコンテナ用の llama.cpp Docker イメージ。検出された GPU ベンダーから自動選択されます。バージョン/バックエンドを固定する場合に設定します |
| `LLAMA_NETWORK` | `llamaman-net` | llamaman + すべての llama-server コンテナ用の Docker ネットワーク。存在しない場合は作成されます |
| `LLAMA_CONTAINER_PREFIX` | `llamaman-` | 起動される llama-server コンテナの名前プレフィックス |
| `LLAMAMAN_IN_DOCKER` | *(自動)* | llamaman 自体がコンテナ内で実行されているかどうか。マーカーファイル + cgroups から自動検出されます。上書きする場合は `true`/`false` を設定します |
| `LLAMA_HOST_ADDR` | `localhost` | ベアメタル実行時に起動されたコンテナの公開ポートに接続するために使用されるホストアドレス |
| `GPU_TYPE` | *(自動)* | GPU ベンダー検出を上書き: `cuda`、`rocm`、`intel`、または `vulkan`（`vulkan` はオプトインのみ。`/dev/dri` + ホストの render/video GID および `server-vulkan` イメージを選択します） |
| `LLAMA_GPU_DEVICES` | *(すべて)* | 起動されるすべてのコンテナから見えるカンマ区切りの GPU インデックス（例: `0,1,3`）。インスタンスごとの **GPU Devices** が優先されます。Intel Arc ではサポートされていません |

### Strata

オプション。[Strata](STRATA.md) を参照してください。

| 変数名 | デフォルト値 | 説明 |
|---|---|---|
| `STRATA_ENABLED` | `false` | このノードで Strata モデルを提供します。NVIDIA GPU も必要です（`GPU_TYPE=cuda` または検出済み） |
| `STRATA_IMAGE` | `strata:latest` | 起動に使用する優先 Strata イメージ。ノード上に存在しない場合は、ビルド済みイメージ、次にプルされたイメージが使用されます |
| `STRATA_BUILD_IMAGE` | `strata:latest` | *Pull repo & build latest* でビルドされるタグ |
| `STRATA_DATA_VOLUME` | `llamaman-strata-data` | すべての Strata コンテナ内の `/data` にマウントされる名前付き Docker ボリューム（モデルファイル、パック、MTP レイヤー、モデルごとのセットアップ。モデルあたり 70〜120 GB）。初回使用時に Docker によって作成されます |
| `HOST_STRATA_DATA_DIR` | *(未設定)* | 名前付きボリュームの代わりに `/data` にマウントするホストパス。`STRATA_DATA_VOLUME` より優先されます |
| `STRATA_LOAD_TIMEOUT` | `3600` | リクエスト（または再起動）が Strata インスタンスの準備完了を待機する秒数。初回起動時はポートが開く前にモデルのダウンロードと準備が行われるため、大きな値になっています |
| `STRATA_REPO` | `Niko1221/Strata` | *Pull repo & build latest* がダウンロードする GitHub リポジトリ（owner/name） |
| `STRATA_REPO_REF` | `main` | ビルドする `STRATA_REPO` のブランチ、タグ、またはコミット |
| `STRATA_SRC_DIR` | `$DATA_DIR/engines/Strata` | ダウンロードしたリポジトリが保持される場所（ビルドコンテキスト）。コミット情報は隣の `Strata.source.json` に記録されます |
| `STRATA_WEB_HOSTS` | *(未設定)* | Strata インスタンス自体の Web アプリ（`http://<host>:<instance port>/`）を開くための追加のホスト名 / IP（カンマ区切り）。API キーがない場合、Strata は他の名前のページからのチャットリクエストを拒否します。`CLUSTER_ADVERTISE_URL` のホストは常に許可されます。以降に起動されたインスタンスに適用されます |

<a id="clustering"></a>
### クラスタリング

オプション。Core の `LLAMAMAN_NODE_NAME` はクラスタ識別子も兼ねており、すべてのインストールで必須です。

| 変数名 | デフォルト値 | 説明 |
|---|---|---|
| `CLUSTER_ENABLED` | `false` | このノードをクラスタに参加させる場合は真値を設定します。`CLUSTER_SECRET` が必要です。シークレットが空の場合は警告付きで無視されます |
| `CLUSTER_SECRET` | *(未設定)* | すべてのノード間 HTTP 通信で使用される共有 Bearer トークン（`X-Cluster-Secret`）。すべてのノードで同一である必要があります。信頼できるネットワーク上または TLS の背後で、十分な長さのランダムな値を使用してください |
| `CLUSTER_ADVERTISE_URL` | *(未設定)* | ピアが**この**ノードの UI/API にアクセスする方法 - 他のホストからルーティング可能なホスト名/IP（`localhost` 以外）。これがない場合、ノードは読み取り専用になります |
| `CLUSTER_NODE_ONLINE_WINDOW_S` | `45` | ノードがオフラインと表示されるまでの最終ハートビートからの秒数。負荷やクロックスキューによってノードが頻繁に切り替わる（フラッピングする）場合は増やしてください |
| `LLAMAMAN_PERF_LOG` | *(無効)* | 診断用。遅延が疑われるパスについてフェーズごとの所要時間（`perf <name> <ms>`）をログ出力する場合は真値を設定します: リモートインスタンス停止、クラスタプロキシ転送、到達性プローブ、ハートビートスナップショット、認証チェック、`save_state`、コンテナ統計。デフォルトは無効。無効時のオーバーヘッドはごくわずかです |
