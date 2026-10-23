# Strata エンジン

*(オプトイン、NVIDIA 専用)* [Strata](https://github.com/Niko1221/Strata) は、Qwen3.8-Flash-Next の Mixture-of-Experts ファミリー（およびその Coder、Swift 1.5、Unsloth バリアント）をコンシューマー向け NVIDIA GPU とシステム RAM 上で実行します。llamaMan は、llama.cpp インスタンスの隣で Strata インスタンスの起動、監視、プロキシ、スリープ/復帰、停止を行うことができます。llama.cpp の動作は変わりません: エンジン層（`core/engines/`）により、llama.cpp の起動はバイト単位で同一に保たれます（スナップショットテスト済み）。

## 目次

- [クイックスタート](#クイックスタート) · [制限と動作](#制限と動作) · [設計上の留意点と検証](#設計上の留意点と検証)

## クイックスタート

**1. イメージの取得**（上流はイメージを公開していません。ホストには NVIDIA ドライバ **580+**（つまり CUDA 13）および NVIDIA Container Toolkit が必要です）。Strata を有効にした後（ステップ2）、**Settings → Docker Images** で名前を指定して信頼できるイメージをプルするか（プル行でエンジン *Strata* を選択）、**Manage Strata images → Pull repo & build latest** を使用してリポジトリを `STRATA_SRC_DIR` にダウンロードし、Docker socket 経由でノードの GPU 世代向けに `STRATA_BUILD_IMAGE` をビルドします（ドライバから読み取り、失敗した場合はサポートされているすべての世代を対象とします）。または手動でビルドします:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. 有効化** llamaMan ノードで設定: `STRATA_ENABLED=true`（[Strata variables](CONFIGURATION.md#strata) を参照）。設定後、Settings → Docker Images に *Manage Strata images* が表示されます: プルされた Strata イメージ、ビルドされたイメージ（ダウンロード済みおよびビルド済みのコミット、ビルドボタン）、およびそれらの自動更新が表示されます。いずれかがノード上に存在すると、起動フォームで Strata が選択可能になります。

**3. 起動。** モデルライブラリには、ファミリー × サイズごとに1つの仮想エントリが追加されます: `strata/qwen-Q2_0`、`strata/qwen-IQ2_XS`、`strata/qwen-IQ3_XXS`、`strata/qwen-IQ3_S`、`strata/swift-IQ2_XS`、`strata/swift-IQ3_XXS`、`strata/coder-IQ1_M`、`strata/unsloth-UD-Q4_K_XL` *(実験的)*。それぞれが独自のインスタンスおよびコンテナであるため、モデルの切り替えは llama.cpp と同様に機能します: 通常の[エビクション](MODELS.md#model-eviction)ルール（Strata インスタンスは1つのチャットモデル）に従って、別のインスタンスを起動または復帰させます。エントリは `/api/tags` および `/v1/models` にも表示され、リクエストには ID（`strata/qwen-IQ2_XS`、大文字小文字不問、`:tag` は無視）または Strata 自身の提供名（`qwen3.8-flash-next-iq2_xs`）を使用できます。

起動フォームでは、llama.cpp 専用フィールド（GPU レイヤー、MoE、スレッド、パラレル、KV タイプ、Flash Attention、投機的デコーディング、mmproj、DRY、追加引数）が非表示になり、**Image Input**（no / GPU / CPU）、**KV Cache**（Strata デフォルト int8、`int8`、`q4_0`、`k8v4`）、**Low-RAM Mode**（auto / on / off）が表示されます。API は Strata モデルに対して設定された llama.cpp 専用フィールドを拒否します。

**初回起動。** Strata は約 58〜111 GB をダウンロードし、「パック」を準備して、サーバーがポートを開く前に MTP レイヤーを取得します。以降の起動は直接ロードに進みます（1〜3分）。インスタンスカードには、コンテナログから解析された *setting up → downloading (with %) → preparing pack → loading* が表示され、Logs ボタンでログをストリーミングできます。ダウンロードには2つの方法があります:

- **Strata にダウンロードさせる** `/data` ボリュームに直接ダウンロードします（デフォルト）。
- **llamaMan で事前ダウンロード**（Strata セクションのボタン、または `POST /api/engines/strata/download`）: Downloads タブの通常のエントリ（進捗、一時停止/再開、再試行、HF トークン、速度制限）として、Strata が固定している Hugging Face リビジョンからシャードを `MODELS_DIR/strata/<tag>/` に取得します。完了するとコンテナにバインドマウントされ、Strata のセットアップはダウンロードの代わりにこれらを使用します。そのダウンロードの実行中は起動が拒否されます。MTP レイヤー（約 5 GB）とパックは、初回起動時に引き続き Strata によって準備されます。

## 制限と動作

- **一度に1リクエストのみ。** Max Concurrent は強制的に 1 になります。以降のリクエストは llamaMan のゲートでキューイングされます（Max Queue Depth に達すると 429）。
- **RAM。** Strata は 32〜62 GB を RAM にロードします（`--ulimit memlock=-1` は常に設定されます）。**System Memory Limit** を設定すると Low-RAM モードが強制的にオンになります。これは、Strata のセットアップがホストの RAM を読み取り、コンテナの制限を認識できないためです。UI は約 64 GB 未満で警告を表示します。
- **設定は Strata のセットアップに焼き込まれます。** Context / 画像入力 / KV / Low-RAM / GPU ピニングは、データボリューム上のモデルごとに記録されます。llamaMan は各モデルが最後に正常起動したときの設定を記憶し、それらが変更された場合にのみ `REINSTALL=1` を渡します（ファイルは再利用されます）。
- **アイドルタイムアウト** は機能します（スリープによりコンテナが停止し、次のリクエストで復帰します）が、コールドスタートには数分かかり、復帰リクエストはその完了を待ちます（`STRATA_LOAD_TIMEOUT`）。
- **ノードあたり Strata モデルごとに1インスタンス**（同一モデルのインスタンスはボリューム上のセットアップを共有します）。
- **NVIDIA 専用。** ROCm / Intel / Vulkan / CPU ノード、および `STRATA_ENABLED` がオフの場合は理由付きで利用不可となります。クラスタでは各ノードが自身のエンジンを通知するため、対応ノードのみが Strata モデルを提供します。
- ベアメタル実行の llamaMan は、llama-server と同様の方法で Strata にアクセスします（`LLAMA_HOST_ADDR` 上の公開ポート）。

---

# 設計上の留意点と検証

このページでは、統合の仕組み、上流への依存関係、制限事項、および実際の NVIDIA ホスト上での検証方法について説明します。

上流のリファレンスは Strata コミット `99f3dbd`（2026-10-03）を指しています。

## 構成の適合方法

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- インスタンスのエンジンは `config["engine"]` です。値がない場合は llama.cpp を意味するため、既存のすべてのインスタンス行、プリセット、コンテナラベルは以前と同じ意味を持ち続けます。llama.cpp の設定にこのキーが付与されることはありません。付与するとコンテナの `llamaman.config` ラベルが変更されてしまうためです。`tests/fixtures/llamacpp_container_spec.json` は、9つの設定 × 5つの GPU ベンダーについてリファクタリング前の `containers.run` kwargs を保持しており、`tests/test_engines.py` がそれらが変更されていないことをアサートします。
- ストレージのマイグレーションは不要です。プリセットとインスタンスは両方のバックエンド（`presets.json` / `state.json`、および MariaDB の `data` TEXT 列）で JSON blob であるため、`engine` と `strata_*` は追加のキーとなります。
- 各 Strata モデルは仮想エントリ `/strata/<family>-<SIZE>`（ID `strata/<family>-<SIZE>`）であり、それぞれ独自のインスタンスおよびコンテナとなります。モデルの切り替え、エビクション、スリープ、復帰は GGUF ファイルと同様に機能します。
- `_run_container` はエンジンの `container_spec()` を受け取り、エンジンが要求する GPU デバイスを追加します。Strata コンテナは以下を取得します:

  | | |
  |---|---|
  | image | `STRATA_IMAGE`（それ以外の場合はビルド済み、次いでプルされた Strata イメージ） |
  | command | なし（entrypoint は環境変数のみを読み取ります） |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`?（ハングウォッチドッグ / 低速ストレージ許容時間）, `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests`（`gpu_devices` / `LLAMA_GPU_DEVICES`） |
  | mounts | データボリューム → `/data`; ログ; llamaMan が完全なシャードを保持している場合、`MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>`（非 qwen の場合は `<family>-<SIZE>`; setup.py の大文字小文字を区別するシャードフォルダ）(rw) |
  | mem_limit | System Memory Limit から（`LOW_RAM=on` を強制） |
  | network / ports / labels | llama-server と同様（`llamaman-net`、`8080 → host port`、`llamaman.*`） |

## 未解決の疑問への回答

1. **準備完了判定。** API キーゲートの前に、`/health` は `{"status": "ok", ..., "service": "strata"}` を返します（`serve/server.py:2172`）。HTTP ポートはモデルがロードされた後にのみ開きます: `server.py:3008` で `serve()` が実行され、その前に `:2925` でエンジンが起動します。Strata がダウンロード中またはロード中の間は接続が拒否されるため、llamaMan の既存のヘルスチェック（`status == "ok"`）に特別なケースは不要です。lazy モードでモデルがアンロードされている間、`/v1/models` は空のリストを返すため（`server.py:2190`）、準備完了のシグナルとしては適していません。API キーが設定されている場合、`/v1/status` にはそのキーが必要です。
2. **設定。** すべて `docker-entrypoint.sh` によって読み取られる環境変数として渡されます: `HOST`, `PORT`, `API_KEY`, `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION`（`:14-24`）。コマンド引数は無視されます。entrypoint が `$@` を再構築するためです（`:42`, `:65`）。GPU ピニングはコンテナの番号付けを使用します: NVIDIA ランタイムは llamaMan がアタッチするホスト GPU に番号を振り直すため、ホストの `1,3` は `GPUS=0,1` になります。llamaMan は `API_KEY` を設定しません。これは llama-server の実行方法（キーなし）と一致しており、前面に自身のプロキシと認証が配置されます。
3. **ディスク上の GGUF の再利用。** `setup.py` には `--gguf-dir` がありますが（`setup.py:2954`）、entrypoint はこれを転送しません。ただし、セットアップは `/data/models/<fam tag><SIZE>/<original name>`（例: `IQ3_S`, `coder-IQ1_M`; 大文字小文字を区別）にすでに存在する完全なシャードを受け入れます: 自身のテンソルディレクトリと照合して完了マークを付けます（`setup.py:3389-3405`, "#173"）。したがって、llamaMan は自身のシャードのダウンロードが完了した後にのみ、ローカルコピーをそこにバインドマウントします（`.done` マークのために読み書き可能）。シャードは Strata が固定している Hugging Face リビジョンから取得されます（`setup.py:65-70`、`core/engines/strata.py` にミラーリング）。MTP レイヤー（約 5 GB）とパックは引き続きコンテナ内で準備されます。
4. **メモリ制限 / memlock。** セットアップはホストの合計である `/proc/meminfo` から RAM を読み取るため、制限されたコンテナは `LOW_RAM=on` を要求する必要があります（Dockerfile のコメント `:31`、`docs/INSTALL.md:116-118`）。これがないとセットアップは RAM 内モードを選択し、コンテナが OOM キルされる可能性が高くなります。llamaMan はメモリ制限が設定されている場合は常に `LOW_RAM=on` を強制し、UI は 64 GB 未満で警告を表示します。`memlock=-1` は重要です。エンジンが GPU DMA 用にアリーナの一部をページロックし、固定できないエキスパートはより低速なホストコピーを経由するためです（`src/prefill/prefill.cpp:110`, `:207-214`）。ソースコード上、memlock 制限がないと失敗というよりは速度低下のように見えますが、**GPU なしでは検証できませんでした**。llamaMan は常にこれを設定します。

また、ソースコードより: Strata は一度に1つのリクエストを処理します（`server.py:1323`）。Dockerfile は `STRATA_EXECV=1` を設定しているため、サーバーは PID 1 となり `docker stop` が届きます。ダウンロードの途中で停止したコンテナは、次回の起動時に `.part` ファイルから再開します。

## 設計上の決定事項

- **並行性は起動時、プリセットマージ時、ライブプリセット編集時、保存されたプリセット内で強制的に 1 になります。** ゲートによりサイドカープロキシが強制されるため、キューイングは Strata の内部ではなく llamaMan（`RequestGate`）内で行われます。
- **フィンガープリントによる `REINSTALL`。** Context、ビジョン、KV、Low-RAM、GPU ピニングは、ボリューム上の Strata のモデルごとの設定に記録されます。`DATA_DIR/strata_setups.json` は各モデルが最後に正常に稼働した設定を記憶し（データボリュームとセットアップタグをキーとする）、異なる場合にのみ `REINSTALL=1` が送信されます。不明なフィンガープリント（llamaMan 経由の初回起動、または手動でセットアップされたボリューム）も再インストールされます。これにはファイルを再利用する1回のセットアップパスのコストがかかります。
- **ノードあたり Strata モデルごとに1つのアクティブインスタンス。** 1つのモデルの2つのインスタンスが同じファイルに対してセットアップを実行することを防ぎます。
- **llamaMan がそのモデルをダウンロードしている間は起動が拒否されます。** コンテナがボリュームに2回目のダウンロードを行うのを防ぐためです。
- **事前ダウンロードされたシャードは更新チェックの出所確認をスキップします。** 固定されたファイルがリポジトリの `main` に「更新」されてはならないためです。
- **仮想モデルは実行可能な場合にのみ一覧表示されます**（`STRATA_ENABLED` かつ NVIDIA）: `/api/models`、`/api/tags`、`/v1/models`、および名前解決。更新スキャンとクラスタモデルスナップショットはファイルのみのままです。ピアは `snapshot.system.engines`（`system` の下にネストされているため、古いピアは無視します）からエンジンの利用可能性を学習します。
- **ロード段階** はコンテナログの最後の 64 KB から取得されます: セットアップの `=== Step N ===` ヘッダー、ダウンロードの `\r` 進捗行、サーバーの "loading the model"。これはベストエフォートであり、取得できない場合は段階行の非表示にグレースフルにフォールバックします。
- **vLLM (#53)** は別の `Engine` となります: 独自のイメージ、コマンド、ポート（`internal_port`）、ヘルスセマンティクス、起動フィールド。両方のエンジンがポート 8080 を使用すると想定している箇所（`resolve_llama_endpoint` の呼び出し元）は、その際にパラメータ化する必要があります。

## GPU なしでは未検証の項目

- 実際の Strata の起動、初回実行のダウンロード、パックビルド、ロード、およびチャット。上記はすべて上流のソースコードと Docker をモックした単体テストに対して確認されており、Strata コンテナは実機で実行されていません。
- 実際の初回起動時の正確なログ形式。段階パーサーは `setup.py` の `step()` / `download()` の出力とサーバーの起動出力を追跡します。
- `memlock=-1` がない場合の挙動、およびコンテナ制限下で `LOW_RAM=on` の場合に実際に使用される RAM。
- 使用中の Docker バージョンでのネストされたバインドマウント（`/data` の名前付きボリューム上の `/data/models/<tag>`）。Docker はネストされたマウントをサポートしていますが、ここでは実機テストされていません。

## 既知の制限事項

- NVIDIA 専用。Strata の AMD (HIP) パスは上流で実験的な手動ビルドであり、提供されていません。
- `LAYER_SPLIT`（マルチ GPU レイヤー配置）は公開されていません。複数の GPU をピニングすると Strata の自動分割が使用されます。
- ビジョンエンコーダ（mmproj）は常に Strata 自身によってボリュームにダウンロードされます。
- Strata 独自の Web UI（`/`, `/settings`, `/load`, `/unload`）はインスタンスポート経由でアクセス可能ですが、llamaMan はこれを使用または管理しません。アイドル時のアンロード機能は使用されず、代わりに llamaMan のスリープがコンテナを停止します。
- セットアップに記録されている設定を変更すると、次回の起動時に Strata のセットアップパスが再実行されます（ファイルは再利用されます）。
- 初回起動は HTTP クライアント自身のタイムアウト（Open WebUI など）よりも長くかかる場合があります。llamaMan は `STRATA_LOAD_TIMEOUT` まで待機し続けますが、クライアントが先に諦める可能性があります。事前にダウンロードするか、サービスを提供する前に UI から一度起動してください。
- カタログ（ファミリー、サイズ、固定リビジョン、ファイル名）は Strata の `setup.py` のミラーです。上流で新しいファミリーやサイズが追加された場合は、`core/engines/strata.py` で対応する編集が必要です。

## 手動検証チェックリスト (NVIDIA ホスト)

前提条件: NVIDIA ドライバ 580+、動作する `docker run --gpus all`、推奨 64 GB RAM、約 150 GB の空きディスク容量。

1. **ビルド:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`。オプションで `--build-arg CUDA_ARCHITECTURES=<your arch>` を追加。またはステップ2の後、Settings → Docker Images → *Manage Strata images* → **Pull repo & build latest**。このノードの GPU 世代向けにのみビルドされます。
2. **有効化:** `STRATA_ENABLED=true` を設定し、llamaMan を再起動します。
   - `GET /api/engines` で strata の `available: true` とそのモデル（組み込みリストに9個）が表示されること。
   - Settings → Docker Images → *Manage Strata images* で `strata:latest` が **local** として表示されること。
3. **ゲーティング:** 非 NVIDIA ノード（または `GPU_TYPE=rocm`）では Strata モデルは一覧表示されず、`/strata/qwen-IQ2_XS` に対する `POST /api/instances` は NVIDIA の理由とともに 400 を返すこと。
4. **UI からの起動:** `strata/qwen-IQ2_XS` を選択。
   - llama.cpp 専用フィールドが非表示になり、Strata セクションが表示され、Max Concurrent が 1 に固定されること。
   - 32g のメモリ制限を設定し、64 GB 未満の警告が表示されることを確認。再度クリアする。
   - 起動。
5. **初回実行（コンテナ内ダウンロード）:**
   - インスタンスカードに *setting up → downloading NN% → preparing pack → loading* と表示されること。
   - Logs ボタンで Strata のセットアップ出力がストリーミングされること。
   - カードが **healthy** になり、`docker inspect` で `Ulimits: memlock -1`、環境変数、`/data` ボリュームが表示されること。
6. **チャット:**
   - Ollama プロキシ経由: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - `:42069` 上の `/v1/chat/completions` 経由、およびインスタンスポートへ直接。
   - `model: "qwen3.8-flash-next-iq2_xs"` も試行。
   - 2つのリクエストを同時に送信: 2つ目はキューイングされること（カードに 1/1 active · 1 queued と表示）。
7. **アイドルスリープと復帰:**
   - Idle Timeout を 1分に設定（コールドスタート警告が表示されます）。1分後、インスタンスは **sleeping** になり、コンテナは削除されること。
   - 次のリクエストで復帰し（数分）、応答が返ること。
   - ログにセットアップパスがないこと（`=== Step` 行なし。`docker inspect` に `REINSTALL` なし）。
8. **設定変更:**
   - コンテキストを 65536 に変更して再起動。
   - `docker inspect` に `REINSTALL=1` が表示され、ログにダウンロードなしのセットアップパスが表示されること。
   - 次回の再起動時、`REINSTALL` が再びなくなること。
9. **事前ダウンロードパス:**
   - 別のモデル（例: `strata/qwen-Q2_0`）を選択し、*Pre-download with llamaMan* をクリック。
   - Downloads タブに固定リビジョンのダウンロードが表示されること。
   - ダウンロード中の起動は拒否されること。
   - 完了後に起動: `docker inspect` で `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0` が表示され、ログにシャードのダウンロードがないこと（MTP レイヤーとパックのみ）。
10. **停止 / 削除:** インスタンスを停止。コンテナが削除され、Remove でカードが消去されること。
11. **Strata インスタンスが実行中の状態で llamaMan を再起動:**
    - 再アタッチされること（状態の復元）、または状態が失われた場合は引き取られること: `state.json` から削除して再起動。
    - カードに Strata バッジと `strata/qwen-IQ2_XS` が表示されること。
    - リクエストが引き続き機能すること。
12. **ベアメタルモード:**
    - `python app.py`（`LLAMAMAN_IN_DOCKER=false`）で llamaMan を実行し、ステップ 4〜6 を繰り返す。
    - llamaMan は `LLAMA_HOST_ADDR` 上の公開ポートに接続すること。
13. **クラスタ:**
    - 非 NVIDIA ピアを追加。Strata モデルは、NVIDIA ノードがターゲットノードの場合にのみ表示されること。
    - 他のノードの UI から NVIDIA ピア上で起動できること。
14. **llama.cpp への影響なし:** 以前と同様に GGUF モデルを起動。
    - コマンドライン（`docker inspect` Args）がアップグレード前と一致すること。
    - アップグレード前に保存されたプリセットが変更なしでロードおよび起動できること。
