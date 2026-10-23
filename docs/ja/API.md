# REST API

すべてのエンドポイントは JSON を返却 / 受信します。管理エンドポイントにはセッション Cookie（ブラウザログイン）または `Authorization: Bearer <key>` が必要です。`require_auth` がオン（デフォルト）の場合、モデル配信エンドポイントにも Bearer トークンが必要です。認証の詳細: [Authentication](CONFIGURATION.md#authentication-detail)。

## 目次

- [認証とキー](#認証とキー) · [インスタンス](#インスタンス) · [ダウンロード、モデル、プリセット](#ダウンロードモデルプリセット) · [設定、システム、リクエストログ](#設定システムリクエストログ) · [Ollama と OpenAI の互換性](#ollama-と-openai-の互換性)

## 認証とキー

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` / `POST` | `/login` | ログインページ / 認証 |
| `GET` / `POST` | `/setup` | 初回セットアップ |
| `GET` | `/logout` | セッション終了 |
| `GET` / `POST` | `/api/api-keys` | API キーの一覧表示 / 作成 |
| `DELETE` | `/api/api-keys/<id>` | 失効 |

## インスタンス

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` / `POST` | `/api/instances` | 一覧表示 / 起動 |
| `GET` / `DELETE` | `/api/instances/<id>` | 取得 / 停止および削除 |
| `POST` | `/api/instances/<id>/restart` | 停止中/スリープ中のインスタンスを再起動 |
| `DELETE` | `/api/instances/<id>/remove` | 停止済みインスタンスのレコードを削除 |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | ログの末尾表示 / SSE ストリーム |
| `GET` | `/api/next-port` | プールから次の利用可能なポートを取得 |
| `POST` | `/api/preview-command` | 起動ボディと同等の `llama-server ...` コマンドをレンダリング（UI の「Copy llama.cpp command」ツールで使用。起動は行わない） |

**起動ボディ** (`POST /api/instances`): JSON としてシリアライズされた起動フォームのフィールド。[Launch settings reference](LAUNCH-SETTINGS.md#launch-settings-reference) で各フィールドとそのデフォルト値を確認してください。列挙値フィールド（`flash_attn`、`reasoning_format`、`split_mode`、`cache_type_k/v`、`spec_type`）はそこにリストされている値のみを受け入れ、リファレンスのデフォルト値が初期値となります。`null` のままか省略された数値フィールドはリファレンスのデフォルト値が使用され、不明なフィールドは無視されます。

注意すべき境界の挙動:

- `n_gpu_layers`: `-1` = **auto**（llama.cpp が VRAM を推定）、`-2` = **all**（全レイヤーを強制）、`0` = CPU のみ（コンテナに GPU がアタッチされない）、正の整数 N = 正確に N レイヤー。
- `n_cpu_moe_layers`: `0`（デフォルト）は両方の MoE フラグを省略。`-1` は `--cpu-moe` を出力（すべてのルーティングされたエキスパートを CPU に配置）。正の整数 N は `--n-cpu-moe N` を出力（最初の N レイヤーのエキスパートを CPU に配置）。dense モデルでは無効。
- `threads`: 設定時、llama-server に `--threads N` を適用**し**、コンテナの CPU クォータを N コアに設定。
- `threads_batch`: null / 省略 → フラグは出力されず、llama-server は `--threads` にフォールバック。
- `flash_attn`: 3値（tri-state）化前の設定からのレガシーな `true` / `false` 値は読み取り時に畳み込まれるため（`true` → `"on"`、`false` → `"off"`）、ストレージのマイグレーションは不要。
- `cache_type_v` の量子化値には `flash_attn: "on"` が必要です — `"auto"` では保証されず、それ以外の場合 llama-server は起動を拒否します。
- `spec_draft_model` は、`spec_enabled: true` かつ `spec_type` が `"draft-mtp"` 以外の場合に必須です — これがないリクエストは 400 で拒否されます。
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / 省略時はフラグを省略します。`0` は null とは異なる有効な値であり、そのまま渡されます。

## ダウンロード、モデル、プリセット

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | 一覧表示 / 開始 |
| `GET` / `DELETE` | `/api/downloads/<id>` | 取得 / キャンセル |
| `DELETE` | `/api/downloads/<id>/remove` | 完了/失敗したエントリを削除 |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | ログの末尾表示 / SSE ストリーム |
| `GET` | `/api/models` | 検出されたモデル（判明している場合は `repo_id` を含む。アーカイブされたモデルは `archived` フラグと `restore_path` を保持） |
| `POST` | `/api/models/delete` | ディスクから削除（`{"path": "/models/..."}`） |
| `GET` | `/api/model-layers?path=<path>` | GGUF メタデータからレイヤー数を読み取り |
| `GET` | `/api/disk-space` | モデルボリュームの空き/使用済み容量 |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | プリセットの取得 / 保存 / 削除（`GET /api/presets` で一覧表示） |
| `GET` | `/api/engines` | このノードの推論エンジン: 利用可能性（理由付き）、機能、起動フィールド、Strata のモデルカタログ |
| `GET` | `/api/archive` | アーカイブ状態（`enabled`、`available`、`reason`、`free_bytes`）と移動ジョブ |
| `POST` | `/api/archive` | モデルのアーカイブ: `{"path": "/models/..."}`（フォルダ全体を移動）。使用中は 409 |
| `POST` | `/api/archive/restore` | アーカイブされたモデルの復元: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | キュー中/実行中の移動をキャンセル（ソースは保持）、または完了したジョブを削除 |
| `POST` | `/api/engines/strata/download` | Strata モデルのシャードを事前ダウンロード（`{"model": "strata/qwen-IQ2_XS"}`、オプションで `hf_token_id` / `speed_limit_mbps`） |

ダウンロードボディ: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`。`filename` を空にするとリポジトリ全体をプルします。

インスタンスとプリセットは、オプションの `engine`（デフォルトは `llamacpp`、または `strata`。`/strata/...` モデルパスから推測）、および Strata の場合は `strata_vision`（`no`/`yes`/`cpu`）、`strata_kv`（`""`/`int8`/`q4_0`/`k8v4`）、`strata_low_ram`（`auto`/`on`/`off`）を受け入れます。Strata のモデルパスは仮想パスです: `/strata/<family>-<SIZE>`（例: `PUT /api/presets/strata/qwen-IQ2_XS`）。[Strata](STRATA.md) を参照してください。

## 設定、システム、リクエストログ

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` / `POST` | `/api/settings` | グローバル設定の取得 / 保存 |
| `GET` | `/api/system-info` | CPU 使用率、コア数、RAM 使用量 |
| `GET` | `/api/gpu-info` | GPU ごとの VRAM と使用率（ネイティブ取得。コンテナ内 exec にフォールバック） |
| `GET` | `/health` | `{"status": "ok"}` - 常に公開、認証不要 |
| `GET` | `/api/request-log/conversations` | 集約されたメタデータを含む最近の会話（`limit` クエリ、デフォルト 100、最大 500） |
| `GET` | `/api/request-log/conversations/<id>` | 1つの会話のすべての記録されたターン（古い順） |
| `GET` | `/api/request-log/stats` | 集約メトリクス（トークン数、平均/ピークのトークン/秒、TTFT、レイテンシ、エラー、ストリーム回数）。オプションで `inst_id` + `window_hours` |

## Ollama と OpenAI の互換性

Ollama: `/api/tags`、`/api/version`、`/api/show`、`/api/ps`、`/api/chat`、`/api/generate`（chat/generate は自動起動）。
OpenAI: `/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`（chat は自動起動）。
意思決定モデル: `/v1/systemone`（自動起動）- [Decision Models](MODELS.md#decision-models) を参照。
ナレッジベース: `/api/kb/*`（管理者専用）および MCP エンドポイント `/mcp/knowledge` - [Knowledge Base & MCP](KNOWLEDGE-BASE.md) を参照。クラスタのノード間エンドポイントは `X-Cluster-Secret` を使用 - [Clustering](CLUSTERING.md) を参照。
