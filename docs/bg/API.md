# REST API

Всички ендпойнти връщат / приемат JSON. Ендпойнтите за управление изискват сесийна бисквитка (вход през браузъра) или `Authorization: Bearer <key>`. Ендпойнтите, обслужващи модели, също изискват bearer токен, когато `require_auth` е включено (по подразбиране). Подробности за удостоверяването: [Удостоверяване](CONFIGURATION.md#authentication-detail).

## Съдържание

- [Удостоверяване и ключове](#auth--keys) · [Инстанции](#instances) · [Изтегляния, модели, пресети](#downloads-models-presets) · [Настройки, система, журнал на заявките](#settings-system-request-log) · [Съвместимост с Ollama и OpenAI](#ollama--openai-compat)

<a id="auth--keys"></a>
## Удостоверяване и ключове

| Метод | Ендпойнт | Описание |
|---|---|---|
| `GET` / `POST` | `/login` | Страница за вход / удостоверяване |
| `GET` / `POST` | `/setup` | Първоначална настройка |
| `GET` | `/logout` | Край на сесията |
| `GET` / `POST` | `/api/api-keys` | Списък / създаване на API ключове |
| `DELETE` | `/api/api-keys/<id>` | Отнемане |

<a id="instances"></a>
## Инстанции

| Метод | Ендпойнт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/instances` | Списък / стартиране |
| `GET` / `DELETE` | `/api/instances/<id>` | Получаване / спиране и премахване |
| `POST` | `/api/instances/<id>/restart` | Рестартиране на спряна/заспала инстанция |
| `DELETE` | `/api/instances/<id>/remove` | Премахване на записа на спряна инстанция |
| `GET` | `/api/instances/<id>/logs`, `.../logs/stream` | Опашка / SSE поток на логовете |
| `GET` | `/api/next-port` | Следващ свободен порт от пула |
| `POST` | `/api/preview-command` | Генерира еквивалентната команда `llama-server ...` за тяло на стартиране (използва се от инструмента "Copy llama.cpp command" в UI; нищо не стартира) |

**Тяло на стартирането** (`POST /api/instances`): полетата от формата за стартиране, сериализирани като JSON. Вижте [справочника за настройки при стартиране](LAUNCH-SETTINGS.md#launch-settings-reference) за всяко поле и стойността му по подразбиране; полетата с изброени стойности (`flash_attn`, `reasoning_format`, `split_mode`, `cache_type_k/v`, `spec_type`) приемат само изброените там стойности и по подразбиране ползват стойността от справочника. Числови полета, оставени `null` или пропуснати, ползват стойността по подразбиране от справочника; непознати полета се игнорират.

Гранични особености, които си струва да се запомнят:

- `n_gpu_layers`: `-1` = **auto** (llama.cpp оценява VRAM), `-2` = **all** (принудително всички слоеве), `0` = само CPU (към контейнера не се закача GPU), положително N = точно N слоя.
- `n_cpu_moe_layers`: `0` (по подразбиране) пропуска и двата MoE флага. `-1` подава `--cpu-moe` (всички маршрутизирани експерти на CPU). Положително N подава `--n-cpu-moe N` (експертите от първите N слоя на CPU). Няма ефект при плътни модели.
- `threads`: когато е зададено, подава `--threads N` на llama-server **и** задава CPU квота на контейнера от N ядра.
- `threads_batch`: null / пропуснато → флагът не се подава и llama-server използва `--threads`.
- `flash_attn`: старите стойности `true` / `false` от конфигурации отпреди тристепенния режим се преобразуват при четене (`true` → `"on"`, `false` → `"off"`), така че не е нужна миграция на хранилището.
- Квантуваните стойности на `cache_type_v` изискват `flash_attn: "on"` - `"auto"` не е гаранция; иначе llama-server отказва да стартира.
- `spec_draft_model` е задължително, когато `spec_enabled: true` и `spec_type` е различно от `"draft-mtp"` - заявка без него се отхвърля с 400.
- `spec_draft_n_min` / `_p_split` / `_p_min`: null / пропуснато пропуска флага. `0` е реална стойност, различна от null, и се подава.

<a id="downloads-models-presets"></a>
## Изтегляния, модели, пресети

| Метод | Ендпойнт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/downloads` | Списък / стартиране |
| `GET` / `DELETE` | `/api/downloads/<id>` | Получаване / отказ |
| `DELETE` | `/api/downloads/<id>/remove` | Премахване на завършен/неуспешен запис |
| `GET` | `/api/downloads/<id>/logs`, `.../logs/stream` | Опашка / SSE поток |
| `GET` | `/api/models` | Открити модели (включва `repo_id`, когато е известен; архивираните модели са маркирани с `archived` и `restore_path`) |
| `POST` | `/api/models/delete` | Изтриване от диска (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Брой слоеве от метаданните на GGUF |
| `GET` | `/api/disk-space` | Свободно/заето място на тома с модели |
| `GET` / `PUT` / `DELETE` | `/api/presets/<model_path>` | Получаване / запис / изтриване на пресет (`GET /api/presets` връща всички) |
| `GET` | `/api/engines` | Inference двигателите на този възел: наличност (с причина), възможности, полета за стартиране, каталогът с модели на Strata |
| `GET` | `/api/archive` | Състояние на архива (`enabled`, `available`, `reason`, `free_bytes`) и задачи за преместване |
| `POST` | `/api/archive` | Архивиране на модел: `{"path": "/models/..."}` (премества цялата му папка). 409, когато е в употреба |
| `POST` | `/api/archive/restore` | Възстановяване на архивиран модел: `{"path": "<ARCHIVE_DIR>/..."}` |
| `DELETE` | `/api/archive/jobs/<id>` | Отказ на чакащо/текущо преместване (източникът се запазва) или премахване на завършено |
| `POST` | `/api/engines/strata/download` | Предварително изтегляне на шардовете на Strata модел (`{"model": "strata/qwen-IQ2_XS"}`, по избор `hf_token_id` / `speed_limit_mbps`) |

Тяло на изтеглянето: `{"repo_id": "...", "filename": "...", "hf_token": "...", "speed_limit_mbps": 0}`. Празно `filename` изтегля цялото хранилище.

Инстанциите и пресетите приемат незадължително `engine` (`llamacpp` по подразбиране или `strata`; извежда се от път на модел `/strata/...`), а за Strata и `strata_vision` (`no`/`yes`/`cpu`), `strata_kv` (`""`/`int8`/`q4_0`/`k8v4`) и `strata_low_ram` (`auto`/`on`/`off`). Пътищата на Strata моделите са виртуални: `/strata/<family>-<SIZE>`, напр. `PUT /api/presets/strata/qwen-IQ2_XS`. Вижте [Strata](STRATA.md).

<a id="settings-system-request-log"></a>
## Настройки, система, журнал на заявките

| Метод | Ендпойнт | Описание |
|---|---|---|
| `GET` / `POST` | `/api/settings` | Получаване / запис на глобалните настройки |
| `GET` | `/api/system-info` | Натоварване на CPU, брой ядра, използвана RAM |
| `GET` | `/api/gpu-info` | VRAM и натоварване по GPU (нативно; при нужда през exec в контейнер) |
| `GET` | `/health` | `{"status": "ok"}` - винаги отворен, без удостоверяване |
| `GET` | `/api/request-log/conversations` | Скорошни разговори с обобщени метаданни (параметър `limit`, по подразбиране 100, максимум 500) |
| `GET` | `/api/request-log/conversations/<id>` | Всички записани реплики за един разговор, от най-старата |
| `GET` | `/api/request-log/stats` | Обобщени метрики (токени, средни/пикови токени/s, TTFT, латентност, грешки, брой стриймнати). По избор `inst_id` + `window_hours` |

<a id="ollama--openai-compat"></a>
## Съвместимост с Ollama и OpenAI

Ollama: `/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` (chat/generate стартират автоматично).
OpenAI: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (chat стартира автоматично).
Decision модели: `/v1/systemone` (стартира автоматично) - вижте [Decision модели](MODELS.md#decision-models).
База знания: `/api/kb/*` (само за администратори) и MCP ендпойнтът `/mcp/knowledge` - вижте [База знания и MCP](KNOWLEDGE-BASE.md). Ендпойнтите между възлите в клъстера използват `X-Cluster-Secret` - вижте [Клъстериране](CLUSTERING.md).
