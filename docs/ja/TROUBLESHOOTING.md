# トラブルシューティング

完全な「症状 → 解決策」の一覧表。[README](../../README.md#troubleshooting) には初回実行時によくある9つの代表的な問題が記載されています。その他の問題はすべてここに記載されています。

| 症状 | 解決策 |
|---|---|
| インスタンスが **starting** のまま停止する | Logs ボタンからログを確認してください。よくある原因: OOM（メモリ不足）、モデルパスの誤字、破損した GGUF、イメージがプルされていない |
| _"Docker image not found"_ | 対応するイメージをプルしてください: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda`（または `server-rocm` / `server-sycl` / `server`） |
| 起動時に _"Docker API error"_ が発生する | `/var/run/docker.sock` がマウントされていることを確認してください（`docker-compose.yml` ではデフォルト） |
| GPU が認識されない / CUDA エラー | NVIDIA Container Toolkit がインストールされており、ホスト上で `docker run --gpus all` が機能することを確認してください |
| GPU が認識されない / ROCm エラー | `/dev/kfd` と `/dev/dri` が存在し、ユーザーが `video`/`render` グループに所属していることを確認してください |
| GPU が認識されない / Intel Arc エラー | `/dev/dri` にアクセス可能であり、ユーザーが `video`/`render` グループに所属していることを確認してください |
| Vulkan イメージが _"Unable to find group render"_ で失敗する | `GPU_TYPE=vulkan` を設定し、llamaMan がホストの render/video GID を数値でアタッチするようにしてください（/etc/group に `render` エントリが含まれていないイメージでは名前ベースの group_add が失敗するため） |
| GPU 統計情報が利用できない | NVIDIA: `deploy.resources.reservations` ブロックのコメントアウトを解除してください。AMD/Intel: `/sys/class/drm:ro` がマウントされていることを確認してください |
| 誤った GPU ベンダーが検出される | `GPU_TYPE=cuda`/`rocm`/`intel` を設定して上書きしてください |
| ベアメタル実行時にインスタンスが **starting** のまま停止する | コンテナ自体は正常ですが、llamaman が接続できません。ランタイム環境に対する自動検出が誤っている場合は、明示的に `LLAMAMAN_IN_DOCKER=false`/`true` を設定してください |
| Strata: _"Docker image 'strata:latest' not found"_ | Settings → Docker Images で Strata イメージをプルまたはビルドしてください（または Strata リポジトリのクローン内で `docker build -t strata .` を実行）。[Strata](STRATA.md) を参照 |
| Strata モデルが表示されない | `STRATA_ENABLED=true` を設定してください。ノードは NVIDIA である必要があります（検出に失敗する場合は `GPU_TYPE=cuda`）。`GET /api/engines` で理由を確認できます |
| Strata インスタンスが **starting** のまま停止する | 初回起動時は数分かかるのが正常です（ダウンロード + パック準備）。カードに進捗ステージが表示され、ログに Strata のセットアップが出力されます。「setup failed」はセットアップが停止したことを意味します — ログの `[X]` 行を確認してください |
| 統計モーダルが空である | **Settings → App Settings → Request recording** を有効にしてください（[Request Recording & Stats](CONFIGURATION.md#request-recording--stats) を参照） |
| GPU パススルーのないホストで起動が GPU/CDI エラーで失敗する | GPU デバイスをアタッチせず CPU のみを使用する場合は **GPU Layers** を `0` に設定するか、GPU ランタイムを修復してください |
| ポートの競合 | フォームは未使用のポートを自動提案します。必要に応じて調整してください |
| モデルが OpenWebUI に表示されない | `OLLAMA_BASE_URL=http://llamaman:42069` が設定されていることを確認してください。`/api/tags` がモデルを返すか確認してください |
| OpenWebUI で 401 エラーが発生する | `require_auth` がオンになっています（デフォルト）。API キーを作成し、OpenWebUI で `OPENAI_API_KEYS` を設定してください |
| モデルが満杯のときに OpenAI クライアントが 503 を受け取る | `LLAMAMAN_MAX_MODELS` の上限に達しており、OpenAI API はデフォルトでエビクションを行いません。上限を引き上げるか、エビクショントグルを有効にしてください: [Model Eviction](MODELS.md#model-eviction) |
| 停止後にコンテナがクリーンアップされない | llamaMan は停止時にコンテナを削除します。クラッシュ後に孤立したコンテナがある場合: `docker ps --filter name=llamaman-` を実行するか、llamaMan を再起動してください（起動時に孤立コンテナの引き取り処理が実行されます） |
| クライアント（Hermes / OpenWebUI など）がプリセットの上限ではなく学習時のコンテキストウィンドウを報告する | 1.1.2+ にアップグレードしてください。`/api/ps` にインスタンス起動時のランタイム ctx を反映した `context_length` が含まれるようになり、`/api/show` の `model_info["<arch>.context_length"]` も有効な上限値で上書きされます |
| キューグループ名を問い合わせたクライアントが実際のデプロイの ctx ではなく独自のデフォルト ctx（例: 128k）を取得する | `/api/show` および `/v1/models` は、共有状態から得られるピアのランタイム ctx を使用して、クラスタ化された任意のノード（ローカルメンバーを持たないノードを含む）上でキューグループに応答するようになりました。複数メンバーのグループは、通知値がどのディスパッチ先でも安全になるよう、アクティブメンバー全体の最小値を報告します。[Clustering](CLUSTERING.md) を参照 |
