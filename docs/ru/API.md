# REST API

> **Примечание:** это перевод, он может отставать от [английской версии](../en/API.md), которая является основной.

Все эндпоинты возвращают / принимают JSON. Управленческим эндпоинтам нужен cookie-сессии (браузерный вход) или `Authorization: Bearer <key>`. Эндпоинты обслуживания моделей также требуют bearer, когда `require_auth` включён (по умолчанию). Подробности аутентификации: [Authentication](CONFIGURATION.md#authentication-detail).

## Содержание

- [Аутентификация и ключи](#auth--keys) · [Инстансы](#instances) · [Загрузки, модели, пресеты](#downloads-models-presets) · [Ошибки моделей на маршрутах инференса](#model-errors-on-inference-routes) · [Настройки, система, журнал запросов](#settings-system-request-log) · [Совместимость с Ollama и OpenAI](#ollama--openai-compat)

<a id="auth--keys"></a>
## Аутентификация и ключи

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` / `POST` | `/login` | Страница входа / аутентификация |
| `GET` / `POST` | `/setup` | Настройка первого запуска |
| `GET` | `/logout` | Завершить сессию |
| `GET` / `POST` | `/api/api-keys` | Список / создание API-ключей |
| `DELETE` | `/api/api-keys/<id>` | Отозвать |

<a id="instances"></a>
## Инстансы

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Список / запуск |
| `GET` / `DELETE` | `/api/instances/<id>` | Получить / остановить и удалить |
| `POST` | `/api/instances/<id>/restart` | Перезапустить остановленный/спящий инстанс |
| `DELETE` | `/api/instances/<id>/remove` | Удалить запись остановленного инстанса |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Хвост / SSE-поток логов |
| `GET` | `/api/next-port` | Получить следующий доступный порт из пула |
| `POST` | `/api/preview-command` | Отрендерить эквивалентную команду `llama-server ...` для тела запуска (используется инструментом UI "Copy llama.cpp command"; ничего не запускает) |

**Тело запуска** (`POST /api/instances`): поля формы запуска, сериализованные в JSON. См. [Справочник настроек запуска](LAUNCH-SETTINGS.md#launch-settings-reference) по каждому полю и его значению по умолчанию; поля с enum (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) принимают только перечисленные там значения и по умолчанию берут значение из справочника. Числовые поля, оставленные `null` или пропущенные, используют значение из справочника; неизвестные поля игнорируются.

Пограничные особенности, которые стоит запомнить:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp оценивает VRAM), `-2` = **all** (принудительно все слои), `0` = только CPU (GPU к контейнеру не подключается), положительное N = ровно N слоёв.
- `n_cpu_moe_layers`: `0` (по умолчанию) не выпускает оба флага MoE. `-1` выпускает `--cpu-moe` (все маршрутизированные эксперты на CPU). Положительное N выпускает `--n-cpu-moe N` (эксперты первых N слоёв на CPU). На плотных моделях не действует.
- `threads`: когда задано, применяет `--threads N` к llama-server **и** ставит контейнеру CPU-квоту в N ядер.
- `threads_batch`: null / пропущено → флаг не выпускается, и llama-server откатывается на `--threads`.
- `flash_attn`: легаси-значения `true` / `false` из конфигов до трёхпозиционной схемы сворачиваются при чтении (`true` → `"on"`, `false` → `"off"`), так что миграция хранилища не нужна.
- Квантованные значения `cache_type_v` требуют `flash_attn: "on"` — `"auto"` не является гарантией; иначе llama-server откажется запускаться.
- `spec_draft_model` обязателен, когда `spec_enabled: true` и `spec_type` — что угодно, кроме `"draft-mtp"`: запрос без него отклоняется с 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / пропущено — флаг не выпускается. `0` — реальное значение, отличное от null, и оно передаётся.

<a id="downloads-models-presets"></a>
## Загрузки, модели, пресеты

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Список / начать |
| `GET` / `DELETE` | `/api/downloads/<id>` | Получить / отменить |
| `DELETE` | `/api/downloads/<id>/remove` | Удалить завершённую/упавшую запись |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Хвост / SSE-поток |
| `GET` | `/api/models` | Обнаруженные модели (включает `repo_id`, когда известен; архивированные модели помечены `archived` с `restore_path`) |
| `POST` | `/api/models/delete` | Удалить с диска (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Прочитать число слоёв из метаданных GGUF |
| `GET` | `/api/disk-space` | Свободное/занятое место на томе моделей |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Получить / сохранить / удалить пресет (`GET /api/presets` перечисляет все) |
| `GET` | `/api/engines` | Движки инференса этого узла: доступность (с причиной), возможности, поля запуска, каталог моделей Strata |
| `GET` | `/api/archive` | Статус архива (`enabled`, `available`, `reason`, `free_bytes`) и задачи перемещения |
| `POST` | `/api/archive` | Архивировать модель: `{"path": "/models/..."}` (перемещает её папку целиком). 409, когда используется |
| `POST` | `/api/archive/restore` | Восстановить архивированную модель: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Отменить поставленное в очередь/идущее перемещение (источник сохраняется) или убрать завершённую |
| `POST` | `/api/engines/strata/download` | Предварительно скачать шарды модели Strata (`{"model": "strata/qwen-IQ2_XS"}`, опционально `hf_token_id` / `speed_limit_mbps`) |

Тело загрузки: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Пустой `filename` тянет весь репозиторий.

Инстансы и пресеты принимают опциональный `engine` (`llamacpp` по умолчанию, или `strata`; выводится из пути модели `/strata/...`) плюс, для Strata, `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) и `strata_low_ram` (`auto`/`on`/`off`). Qwen3.6-35B-A3B также принимает `strata_parallel` (от `1` до `8`, по умолчанию `1`); любое другое семейство Strata отклоняет значения выше `1`. Пути моделей Strata виртуальные: `/strata/<family>-<SIZE>`, например `PUT /api/presets/strata/qwen-IQ2_XS`. См. [Strata](STRATA.md).

<a id="model-errors-on-inference-routes"></a>
## Ошибки моделей на маршрутах инференса

Когда модель не может быть обслужена, маршруты Ollama возвращают `{"error": "<message>"}`, а маршруты OpenAI — `{"error": {"message": "<message>", "code": "<code>"}}`.

| Статус | Код | Значение |
|---|---|---|
| `429` | `group_full` | Группу размещают другие узлы, и очередь хотя бы одного пира заполнена. Повторная попытка может сработать. |
| `503` | `group_unavailable` | Группу размещают другие узлы, но ни один не может принять запрос. В сообщении указаны все узлы и причины. |
| `503` | `group_stopped` | Все члены группы на этом узле остановлены |
| `503` | `model_stopping` | Член группы завершает работу |
| `503` | `wake_failed` | Не удалось разбудить модель |
| `503` | `wake_timeout` | Модель не стала healthy вовремя |
| `503` | `model_limit` | Достигнут `LLAMAMAN_MAX_MODELS`, и вытеснить нечего |

Просто неизвестная модель возвращает `model '<name>' not found`: `500` на маршрутах Ollama и `503` на маршрутах OpenAI, без кода. Пересланный запрос, превысивший тайм-аут на пире, возвращает `504`. Модель, которая была запущена или уже загружалась, но так и не стала готовой, возвращает `500` без кода, и запрос не пробует другого члена группы.

<a id="settings-system-request-log"></a>
## Настройки, система, журнал запросов

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Получить / сохранить глобальные настройки. Ключи на уровне узла включают `eager_wake_enabled` и `eager_wake_polls`; см. [Пробуждение под нагрузкой](CLUSTERING.md#waking-under-load) |
| `GET` | `/api/system-info` | Загрузка CPU, число ядер, использование RAM |
| `GET` | `/api/gpu-info` | VRAM и загрузка по каждому GPU (нативно; откатывается к exec в контейнере) |
| `GET` | `/health` | `{"status": "ok"}` — всегда открыт, аутентификация не требуется |
| `GET` | `/api/request-log/conversations` | Недавние беседы со свёрнутыми метаданными (`limit` в query, по умолчанию 100, максимум 500) |
| `GET` | `/api/request-log/conversations/<id>` | Все записанные ходы одной беседы, от старых к новым |
| `GET` | `/api/request-log/stats` | Сводные метрики (токены, средние/пиковые токены/с, TTFT, задержка, ошибки, счётчики стриминга). Опционально `inst_id` + `window_hours` |

<a id="ollama--openai-compat"></a>
## Совместимость с Ollama и OpenAI

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate автозапускаются).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat автозапускается). В кластере `/v1/embeddings` маршрутизируется между узлами, как и другие маршруты инференса.
Модели решений: `/v1/systemone` (автозапускается) — см. [Decision-модели](MODELS.md#decision-models).
База знаний: `/api/kb/*` (только для админов) и MCP-эндпоинт `/mcp/knowledge` — см. [Knowledge Base & MCP](KNOWLEDGE-BASE.md). Межузловые эндпоинты кластера используют `X-Cluster-Secret` — см. [Clustering](CLUSTERING.md).
