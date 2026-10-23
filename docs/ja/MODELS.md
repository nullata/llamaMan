# モデル、ダウンロード、エビクション

モデルライブラリに関するすべて: モデルの追加、最新状態の維持、プロキシによるモデルの選択とエビクション、ハウスキーピング設定。

## 目次

- [モデル](#モデル) · [意思決定モデル](#decision-models) · [モデルのエビクション](#model-eviction) · [ダウンロード設定](#ダウンロード設定) · [Docker イメージ管理](#docker-image-management) · [モデルアーカイブ](#model-archive) · [モデルのバックアップと復元](#モデルのバックアップと復元) · [クリーンアップ設定](#クリーンアップ設定)

## モデル

モデルは `.gguf` ファイル、HuggingFace リポジトリディレクトリ（`config.json` を含む）として `/models` ボリューム下に配置するか、UI の **Download** ボタンを使用します。

### 表示名

デフォルトでは、モデルは GGUF ファイル名の幹（stem）で一覧表示されます。各モデルにはわかりやすい **Display Name**（起動フォーム内）を設定できます。設定すると、`/api/tags` および `/v1/models` で API クライアントが参照する ID になるため、OpenWebUI では `Qwen2.5-14B-Instruct-Q4_K_M` の代わりに `Qwen 2.5 14B` が表示されます。一意である必要があり、他のモデルのファイル名やクラスタキューグループ名と重複してはなりません。

### モデルの更新

**Check for updates** ボタン（元のリポジトリが判明しているモデルに表示）は、ローカルファイルとリポジトリの公開されたコンテンツハッシュを比較します:

- **Up to date** - 記録されたハッシュと一致
- **Update available** - ステージングフォルダに再プルし、ダウンロードが完了したときにのみアトミックに入れ替えます（プルの失敗時は現在のモデルがそのまま維持されます）。モデルのロード中は 409 で拒否されます。
- **Verify hash** - ハッシュがまだ記録されていない場合にローカルファイルのハッシュを計算します（進捗表示付きのバックグラウンドジョブ）

オプトインのバックグラウンドスキャン（**Settings → Download Settings**、[ダウンロード設定](#ダウンロード設定) を参照）により、これらの結果を事前に準備しておくことができます。

<a id="decision-models"></a>
## 意思決定モデル

意思決定モデル（Decision models）は、テキストを生成する代わりに入力に対する型付きの質問に確率で回答します。TypeSafe の Jev によって導入された System One API であり、llama.cpp（2026-10-05 以降にビルドされたイメージ）によって `POST /v1/systemone` で提供されます。オープンな GGUF: Julia-1、Laya、Kev-4B、lev、Clef-Flash、Clef、OpenJev（OpenJev と Clef は `--mmproj` を使用して画像も受け入れます）。llama-server は GGUF 内の `<arch>.decision.type` からこれらを認識するため、追加のフラグは不要です。

- **検出。** 起動フォームで選択すると、Model Path の下にその旨が表示されます。プリセットがない場合、フォームは専用に設定されます: **Exclude from Max Models** がオン、**Web UI** がオフ（そのチャットページではモデルを操作できないため）、**Parallel Slots** が `8`（リクエストの各質問がスロットを消費し、入力を共有するため）、**Embedding Model** がオフ（モデルが必要とする場合、llama.cpp が自身で埋め込みモードをオンにするため強制は不要）。エンコーダ（BERT スタイル）モデルの場合、Context、Batch、Micro-batch Size もそのコンテキスト長に設定されます。入力全体が1つのマイクロバッチに収まる必要があります。**Clef**（および Clef-Flash）は1つのプロンプトですべての質問を行い、1回のパスで読み取るため、**Parallel Slots** が `1`、Batch = Micro-batch = Context（Context が空の場合は学習時のデフォルトである 16384）に設定されます。リクエスト全体（入力、すべての質問、選択肢）が1つのマイクロバッチに収まる必要があり、そうでない場合 llama-server は "increase the physical batch size" と返答します。
- **ルーティング。** `POST :42069/v1/systemone` は、`/v1/chat/completions` と同様に `"model"` によってモデルを選択します（自動起動、エビクションルール、クラスタ転送、キュー）。リクエストは変更されずに転送されます（サンプリングオーバーライドなし）。インスタンス自身のポート上では、インスタンスごとのプロキシが実行されている場合、そのプロキシのキューを通過します。
- **チャットエンドポイントはこれらを拒否します**（HTTP 422） - テキストを生成しないためです。
- **リクエストログ:** 回答内容、およびプロンプトトークンとしての `input_tokens`（`output_tokens` は常に 0）。
- **エラー:** 選択肢が1つのバッチに収まらない質問は、llama-server によって拒否されます（"increase the physical batch size"） - Batch Size / Micro-batch Size を引き上げてください。

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
## モデルのエビクション

`LLAMAMAN_MAX_MODELS` は、プロキシ経由で同時に稼働する**チャット**モデルの数を制限します。上限に達した状態で新しいチャットモデルがリクエストされると、LRU チャットモデルがエビクションされます:

| 起動元 | エビクション対象 | エビクション不可 |
|---|---|---|
| **Admin UI** | Ollama 管理を優先（LRU）、次いで管理者が起動したもの | - |
| **Ollama API** | Ollama 管理（LRU） | 管理者が起動したもの *(デフォルト)* |
| **OpenAI API** | なし *(デフォルト。満杯時は 503 を返却)* | すべて *(デフォルト)* |

**Settings → App Settings** にある3つのトグルで、これらのデフォルトを緩和できます:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - 起動前にサイレントで LRU エビクションを実行（Ollama 管理を優先）。デフォルトはオフ → UI が先に確認を促します。
- **Allow Ollama API to evict admin-launched models** - Ollama 管理のモデルが利用できない場合のフォールバック。デフォルトはオフ。
- **Allow OpenAI API to evict admin-launched models** - OpenAI API に Ollama と同様の LRU エビクションを許可。デフォルトはオフ。

その他の詳細:

- 実行中のすべてのチャットインスタンスがカウントされます（管理 UI とプロキシ管理の両方）。
- **スリープ中のインスタンスもカウントされます**（アイドル一時停止中もスロットの確保が継続します） - ただし、自身のモデルのためにスリープ中のインスタンスを起動することは、上限によってブロックされません。
- **Exclude from Max Models** がオン（起動フォーム）のインスタンスはカウントされず、エビクションされることもありません: 埋め込みモデル、意思決定モデル、常駐させるべき小型のモデルなど。**Embedding Model** をオンにするとこれも自動的にオンになり、トグルが存在する前に保存されたプリセットは Embedding Model 設定からこれを継承します。
- `LLAMAMAN_MAX_MODELS=0` はエビクションを完全に無効化します。

## ダウンロード設定

**Settings → Download Settings** 内:

- **Global speed limit (Mbps)**（0 = 無制限）- このノード上のすべてのダウンロードを制限し、1秒以内に実行中のダウンロードに適用されます。ノード単位: クラスタでは各ノードが独自の制限を持ちます（ノード単位になる前に保存された値は、各ノードが独自に保存するまで全ノードに適用されます）
- **Auto-retry failed downloads**（デフォルトはオフ）+ **Retry count per failed download**（デフォルト 3）
- **Check models for updates in the background**（デフォルトはオフ）- 各ソースリポジトリにファイルが再公開されたかを問い合わせ、まだチェックサムがないモデルのチェックサムを計算するオプトインワーカー。**Update check interval (hours)** のデフォルトは 24。1回のパスでハッシュを計算するのは最大1モデル。ダウンロードの進行中は実行されません。

<a id="docker-image-management"></a>
## Docker イメージ管理

**Settings → Docker Images**: 名前を指定して任意の llama.cpp イメージをプルし、古いローカルイメージを削除（アクティブな `LLAMA_IMAGE` では無効化され、コンテナが使用中のため Docker が拒否した場合はエラーを返却）、およびスケジュールに従ってノードの llama.cpp イメージ（`LLAMA_IMAGE` およびそこでプルされたすべてのイメージ）を自動更新（オプション）。プル行にはエンジンの選択があります: `STRATA_ENABLED` が有効な場合、*Strata* としてプルされたイメージは *Manage Strata images* に入ります（デフォルトの Strata ソースはありません。名前を入力してください）。**Pull repo & build latest** は、Strata リポジトリの最新コミット（GitHub の tarball、`STRATA_SRC_DIR` 内）をダウンロードし、Docker socket 経由で BuildKit を使用して `STRATA_BUILD_IMAGE` をビルドします。Strata 自身の自動更新は、プルされた Strata イメージを再プルし、リポジトリがダウンロードされていた場合は最新コミットを取得して変更があった場合（またはイメージがない場合）に再ビルドします。独自にリポジトリをダウンロードすることはありません。**Delete image & repo** は、プルされたイメージを実行するノードのために、ビルドされたイメージ（実行中のインスタンスが使用している間は拒否）とダウンロードされたリポジトリを削除します。Strata の起動は、ノード上に存在する場合は `STRATA_IMAGE` を使用し、存在しない場合はビルドされたイメージ、それもない場合はプルされたイメージを使用します。[Strata](STRATA.md) を参照してください。

<a id="model-archive"></a>
## モデルアーカイブ

*(オプトイン)* ホスト上で手動でファイルを移動する代わりに、UI からモデルを高速ストレージからセカンダリボリュームへ、またはその逆へと移動できます。

アーカイブボリュームを llamaMan コンテナにマウントし、`ARCHIVE_DIR` に指定して**有効化**します:

```yaml
    volumes:
      - /opt/backup/llms:/archive
    environment:
      - ARCHIVE_DIR=/archive
```

未設定の場合、この機能は非表示になります。llamaMan 自身がファイルを移動するため、`HOST_` パスは不要です。

**使い方。**

- ライブラリ内のモデルにホバーし、アーカイブアイコンをクリックします。
- アーカイブされたモデルはリストに薄く残り、*archived* バッジと **Restore** ボタンが表示されます。起動用に選択することはできず、`/api/tags` / `/v1/models` にも表示されません。
- 移動の進捗、速度、残り時間は **Downloads** パネルに表示され、キャンセルも可能です。
- 復元するとモデルが元のパスに戻るため、プリセット、お気に入り、メモ、表示名も一緒に復元されます。

**移動対象。** `/models` 下のモデルの最上位フォルダ全体: ダウンロードのフォルダと、分割された GGUF のすべてのシャード。`/models` に直接配置された `.gguf` は、その分割された兄弟ファイルと一緒に移動します。Strata モデルのシャードフォルダ（`strata/<tag>`）は単独で移動します。

**安全性。**

- **同一ディスク:** 移動は即座のリネームで行われます。
- **ディスク間:** llamaMan は移動先の隠し `.llamaman-partial-*` フォルダにコピーし、fsync を行い、各ファイルのサイズをチェックし、適切な位置にリネームした後にのみソースを削除します。
- **障害:** キャンセル、チェックの失敗、または空き容量不足の場合、ソースはそのまま残ります。移動途中の再起動によって残された不完全なコピーは起動時に削除されます。
- **一度に1つの移動のみ。**
- **拒否条件:** インスタンスがモデルを使用している間（スリープ中のもの、ドラフトモデル / mmproj として使用されているもの、またはシャードを使用している Strata インスタンスを含む）、ダウンロードがそのモデルに書き込んでいる間、または移動先の名前がすでに存在する場合は拒否されます。移動中のモデルの起動や削除も拒否されます。

**制限事項。** 移動ジョブはメモリ上に保持されます: llamaMan の再起動後はリストが空になり、中断された移動は再度開始する必要があります（データは失われません）。クラスタモードでは各ノードが独自のボリュームにアーカイブし、閲覧中のノードに応じたボタンが表示されます。

## モデルのバックアップと復元

**Settings → App Settings**:

- **Download Stored Models JSON** - スキャンされたすべてのモデル + プリセット設定をタイムスタンプ付きの JSON にエクスポート
- **Restore from JSON** - モデルごと: 既存モデル → プリセットをマージ（既存の値を保持）。未存在だが HuggingFace ソースがある場合 → プリセットが事前入力された状態でダウンロードキューに追加。未存在かつ既知のソースがない場合 → 復元不能として報告

## クリーンアップ設定

**Settings → Cleanup Settings**（すべてバックグラウンドで定期的に実行。レコードの削除/更新のみを行い、モデルファイルを削除することは決してありません）:

- **Auto-clean completed/failed downloads**（デフォルト 24h）- アクティブなダウンロードには一切触れません
- **Auto-clean stopped instances**（デフォルト 24h）- 実行中のインスタンスは決して削除されません
- **Auto-remove stale instance records**（デフォルト 5分間隔）- Docker に対して `starting`/`healthy`/`sleeping` レコードを再チェックし、コンテナが存在しないものは停止済みとしてマークされます
