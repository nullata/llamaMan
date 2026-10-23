# Движок Strata

*(opt-in, только NVIDIA)* [Strata](https://github.com/Niko1221/Strata) гоняет семейство mixture-of-experts Qwen3.8-Flash-Next (и его варианты Coder, Swift 1.5 и Unsloth) на потребительских NVIDIA GPU плюс системной RAM. llamaMan умеет запускать, мониторить, проксировать, укладывать в сон/будить и останавливать Strata-инстансы рядом с llama.cpp. Поведение llama.cpp не меняется: слой движков (`core/engines/`) сохраняет запуск llama.cpp побайтово идентичным (snapshot-тесты).

## Содержание

- [Начало работы](#getting-started) · [Ограничения и поведение](#limits-and-behaviour) · [Заметки по дизайну и проверка](#design-notes-and-verification)

<a id="getting-started"></a>
## Начало работы

**1. Достаньте образ** (upstream никаких не публикует; хосту нужен NVIDIA драйвер **580+**, т.е. CUDA 13, и NVIDIA Container Toolkit). После включения Strata (шаг 2) либо подтяните доверенный по имени в **Settings → Docker Images** (движок *Strata* в строке pull), либо воспользуйтесь **Manage Strata images → Pull repo & build latest**, что скачивает репозиторий в `STRATA_SRC_DIR` и собирает `STRATA_BUILD_IMAGE` через Docker-сокет для GPU-поколений узла (считываются из драйвера; всех поддерживаемых, если это не удаётся), либо соберите вручную:

```bash
git clone https://github.com/Niko1221/Strata && cd Strata
docker build -t strata .                                        # all supported GPU generations
docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .      # RTX 40 only (faster build)
```

**2. Включите** на узле llamaMan: `STRATA_ENABLED=true` (см. [Strata variables](CONFIGURATION.md#strata)). Settings → Docker Images тогда покажет *Manage Strata images*: подтянутые Strata-образы, собранный (скачанные и собранные коммиты, кнопка сборки) и их автообновление. Форма запуска предложит Strata, как только один из них окажется на узле.

**3. Запуск.** Библиотека моделей получает по одной виртуальной записи на семейство × размер: `strata/qwen-Q2_0`, `strata/qwen-IQ2_XS`, `strata/qwen-IQ3_XXS`, `strata/qwen-IQ3_S`, `strata/swift-IQ2_XS`, `strata/swift-IQ3_XXS`, `strata/coder-IQ1_M`, `strata/unsloth-UD-Q4_K_XL` *(experimental)*. Каждая — отдельный инстанс и контейнер, поэтому переключение моделей работает как у llama.cpp: запустите или разбудите другой инстанс, с обычными правилами [вытеснения](MODELS.md#model-eviction) (Strata-инстанс — одна chat-модель). Записи также появляются в `/api/tags` и `/v1/models`, и запросы могут использовать id (`strata/qwen-IQ2_XS`, любой регистр, `:tag` игнорируется) или собственное обслуживаемое имя Strata (`qwen3.8-flash-next-iq2_xs`).

Форма запуска скрывает поля только для llama.cpp (GPU layers, MoE, threads, parallel, KV types, flash attention, spec decoding, mmproj, DRY, extra args) и показывает **Image Input** (no / GPU / CPU), **KV Cache** (дефолт Strata int8, `int8`, `q4_0`, `k8v4`) и **Low-RAM Mode** (auto / on / off). API отвергает поля только для llama.cpp, заданные для Strata-модели.

**Первый старт.** Strata скачивает ~58-111 ГБ, готовит свой "pack" и тянет свой MTP-слой до того, как сервер откроет порт; последующие старты сразу идут в загрузку (1-3 минуты). Карточка инстанса показывает *setting up → downloading (с %) → preparing pack → loading*, разобранные из лога контейнера, а кнопка Logs стримит лог. Два способа скачать:

- **Пусть скачает Strata** в свой том `/data` (по умолчанию).
- **Предварительно скачать через llamaMan** (кнопка в секции Strata, или `POST /api/engines/strata/download`): обычная запись во вкладке Downloads (прогресс, пауза/продолжение, ретраи, HF-токены, лимиты скорости), которая тянет шарды из ревизии Hugging Face, закрепленной Strata, в `MODELS_DIR/strata/<tag>/`. По завершении они bind-монтируются в контейнер, и setup Strata использует их вместо скачивания. Запуск отвергается, пока та загрузка ещё идёт. MTP-слой (~5 ГБ) и pack по-прежнему готовит Strata при первом старте.

<a id="limits-and-behaviour"></a>
## Ограничения и поведение

- **Один запрос за раз.** Max Concurrent принудительно 1; дальнейшие запросы встают в гейт llamaMan (Max Queue Depth, затем 429).
- **RAM.** Strata загружает 32-62 ГБ в RAM (`--ulimit memlock=-1` ставится всегда). **System Memory Limit** принудительно включает Low-RAM режим, потому что setup Strata читает RAM хоста и не видит контейнерный лимит; UI предупреждает ниже ~64 ГБ.
- **Настройки запекаются в setup Strata.** Context / image input / KV / Low-RAM / закрепление GPU записываются по моделям на томе данных. llamaMan помнит настройки, с которыми каждая модель последний раз стала healthy, и передаёт `REINSTALL=1` только при их изменении (файлы переиспользуются).
- **Idle timeout** работает (сон останавливает контейнер; следующий запрос будит), но холодный старт занимает минуты, и будящий запрос ждёт (`STRATA_LOAD_TIMEOUT`).
- **Один инстанс на Strata-модель на узел** (инстансы одной модели делят её setup на томе).
- **Только NVIDIA.** Опция недоступна, с указанием причины, на ROCm / Intel / Vulkan / CPU узлах и когда `STRATA_ENABLED` выкл. В кластере каждый узел рекламирует свои движки, поэтому Strata-модели предлагают только способные узлы.
- llamaMan в bare-metal режиме обращается к Strata так же, как к llama-server (опубликованный порт на `LLAMA_HOST_ADDR`).

---

<a id="design-notes-and-verification"></a>
# Заметки по дизайну и проверка

Эта страница описывает, как устроена интеграция, на что она опирается в upstream, её ограничения и как проверить её на настоящем NVIDIA
хосте.

Ссылки на upstream указывают на коммит Strata `99f3dbd` (2026-10-03).

<a id="how-it-fits"></a>
## Как это устроено

```
core/engines/
  base.py       Engine interface: image, command, env, volumes, ulimits, GPU attachment,
                readiness timeout, capabilities, launch fields, served names, availability
  llamacpp.py   LlamaCppEngine - the pre-existing launch path, unchanged (snapshot-tested)
  strata.py     StrataEngine, the family x size catalogue, load-stage parsing
  __init__.py   registry, engine resolution, launch validation
```

- Движок инстанса — это `config["engine"]`. Отсутствие значения означает llama.cpp, поэтому каждая существующая строка инстанса,
  пресет и метка (label) контейнера значат по-прежнему то же, что значили. Конфигурации llama.cpp никогда не помечаются этим ключом, потому что это
  изменило бы метку `llamaman.config` контейнера. `tests/fixtures/llamacpp_container_spec.json` хранит
  kwargs `containers.run` до рефакторинга для 9 конфигураций × 5 вендоров GPU, и `tests/test_engines.py` проверяет, что они
  не изменились.
- Миграция хранилища не нужна. Пресеты и инстансы — JSON-блобы в обоих бэкендах (`presets.json` /
  `state.json`, и TEXT-колонки `data` MariaDB), поэтому `engine` и `strata_*` — аддитивные ключи.
- Каждая Strata-модель — виртуальная запись, `/strata/<family>-<SIZE>` (id `strata/<family>-<SIZE>`), и это отдельный
  инстанс и контейнер. Переключение моделей, вытеснение, сон и пробуждение работают так же, как для GGUF-файлов.
- `_run_container` берёт `container_spec()` движка и добавляет GPU-устройства, которые движок запрашивает. Strata-
  контейнер получает:

  | | |
  |---|---|
  | image | `STRATA_IMAGE` (иначе собранный, затем подтянутый Strata-образ) |
  | command | нет (entrypoint читает только переменные окружения) |
  | env | `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `HOST=0.0.0.0`, `PORT=8080`, `LOW_RAM`, `KV`?, `GPU`/`GPUS`?, `LAYER_SPLIT`?, `STRATA_WATCHDOG_S`? / `STRATA_WATCHDOG_IO_S`? (Hang Watchdog / Slow-Storage Allowance), `STRATA_ALLOWED_HOSTS`, `REINSTALL=1`? |
  | ulimits | `memlock=-1` |
  | GPU | NVIDIA `device_requests` (`gpu_devices` / `LLAMA_GPU_DEVICES`) |
  | mounts | том данных → `/data`; логи; `MODELS_DIR/strata/<tag>[/<SIZE>]` → `/data/models/<SIZE>` (`<family>-<SIZE>` для не-qwen; чувствительная к регистру папка шардов setup.py) (rw), когда llamaMan держит полные шарды |
  | mem_limit | из System Memory Limit (принудительно `LOW_RAM=on`) |
  | network / ports / labels | как для llama-server (`llamaman-net`, `8080 → host port`, `llamaman.*`) |

<a id="answers-to-the-open-questions"></a>
## Ответы на открытые вопросы

1. **Готовность.** `/health` отвечает `{"status": "ok", ..., "service": "strata"}` до гейта API-ключа
   (`serve/server.py:2172`). HTTP-порт открывается только после загрузки модели: `serve()` выполняется на
   `server.py:3008`, после старта движка на `:2925`. Пока Strata скачивает или загружает, соединения отклоняются, поэтому
   существующая health-проверка llamaMan (`status == "ok"`) не требует специального случая. `/v1/models` возвращает пустой
   список, пока модель не загружена в lazy-режиме (`server.py:2190`), поэтому это не хороший сигнал готовности.
   `/v1/status` требует API-ключ, когда тот задан.
2. **Настройки.** Всё передаётся переменными окружения, которые читает `docker-entrypoint.sh`: `HOST`, `PORT`, `API_KEY`,
   `GPU` / `GPUS` / `LAYER_SPLIT`, `LOW_RAM`, `KV`, `CONTEXT`, `VISION` (`:14-24`). Аргументы команды игнорируются,
   потому что entrypoint пересобирает `$@` (`:42`, `:65`). Закрепление GPU использует контейнерную нумерацию: NVIDIA runtime
   перенумеровывает хост-GPU, которые подключает llamaMan, поэтому хостовые `1,3` становятся `GPUS=0,1`. llamaMan не задаёт `API_KEY`, что
   совпадает с тем, как он гоняет llama-server (без ключа); его собственный прокси и auth стоят спереди.
3. **Переиспользование GGUF на диске.** У `setup.py` есть `--gguf-dir` (`setup.py:2954`), но entrypoint его никогда не передаёт.
   Setup, впрочем, принимает целые шарды, уже лежащие в `/data/models/<fam tag><SIZE>/<original name>` (например `IQ3_S`, `coder-IQ1_M`; чувствительно к регистру): он проверяет каждый
   против своего каталога тензоров и помечает готовым (`setup.py:3389-3405`, "#173"). Поэтому llamaMan bind-монтирует
   свою копию туда (чтение-запись, ради меток `.done`), и только когда собственное скачивание тех шардов завершено.
   Шарды тянутся из ревизий Hugging Face, закрепленных Strata (`setup.py:65-70`, продублировано в
   `core/engines/strata.py`). MTP-слой (~5 ГБ) и pack по-прежнему готовятся внутри контейнера.
4. **Лимит памяти / memlock.** Setup читает RAM из `/proc/meminfo`, а это полный объём хоста, поэтому контейнеру с лимитом
   нужно просить `LOW_RAM=on` (комментарий в Dockerfile на `:31`, `docs/INSTALL.md:116-118`). Без этого setup выбирает
   in-RAM режим, и контейнер, скорее всего, убьёт OOM. llamaMan принудительно ставит `LOW_RAM=on`, когда задан лимит памяти,
   и UI предупреждает ниже 64 ГБ. `memlock=-1` важен, потому что движок page-lock'ит часть своей арены для GPU DMA,
   а эксперты, которые не удаётся закрепить, идут через более медленные копии по хосту (`src/prefill/prefill.cpp:110`, `:207-214`). По
   исходнику, отсутствующий лимит memlock выглядит как замедление, а не отказ, но **без GPU
   я это проверить не смог**. llamaMan ставит его всегда.

Также из исходника: Strata обслуживает один запрос за раз (`server.py:1323`). Dockerfile задаёт `STRATA_EXECV=1`,
так что сервер — PID 1, и `docker stop` доходит до него. Контейнер, остановленный посреди скачивания, продолжает со своих `.part`
файлов при следующем старте.

<a id="design-decisions"></a>
## Проектные решения

- **Параллелизм принудительно 1** при запуске, при слиянии пресета, при живых правках пресета и в сохранённых пресетах. Гейт
  принуждает sidecar-прокси, поэтому очередь происходит в llamaMan (`RequestGate`), а не внутри Strata.
- **`REINSTALL` по отпечатку.** Context, vision, KV, Low-RAM и закрепление GPU записываются в по-модельный конфиг Strata
  на томе. `DATA_DIR/strata_setups.json` помнит настройки, с которыми каждая модель последний раз стала healthy,
  с ключом по тому данных и тегу setup, и `REINSTALL=1` отправляется только при расхождении. Неизвестный отпечаток (первый
  запуск через llamaMan, или том, настроенный руками) тоже переустанавливает. Это стоит один проход setup, переиспользующий файлы.
- **Один живой инстанс на Strata-модель на узел.** Два инстанса одной модели гоняли бы setup по одним и тем же файлам.
- **Запуск отвергается, пока llamaMan ещё скачивает эту модель**, чтобы контейнер не скачал её второй раз в свой
  том.
- **Предварительно скачанные шарды пропускают provenance проверки обновлений.** Закреплённый файл нельзя "обновить" до `main` репозитория.
- **Виртуальные модели перечисляются только там, где могут работать** (`STRATA_ENABLED` плюс NVIDIA): в `/api/models`,
  `/api/tags`, `/v1/models` и разрешении имён. Сканы обновлений и кластерный снимок моделей остаются только по файлам. Пиры
  узнают доступность движков из `snapshot.system.engines` (вложены в `system`, поэтому старые пиры их игнорируют).
- **Стадия загрузки** берётся из последних 64 КБ лога контейнера: заголовки setup `=== Step N ===`, строка прогресса
  скачивания с `\r` и "loading the model" сервера. Это best-effort: при неудаче строка стадии просто не показывается.
- **vLLM (#53)** был бы ещё одним `Engine`: свой образ, команда, порт (`internal_port`), семантика здоровья и
  поля запуска. Места, предполагающие, что оба движка используют порт 8080 (вызывающие `resolve_llama_endpoint`), — именно их
  тогда нужно будет параметризовать.

<a id="not-verified-without-a-gpu"></a>
## Не проверено без GPU

- Настоящий запуск Strata, скачивание при первом запуске, сборка pack, загрузка и chat. Всё выше сверялось с исходником
  upstream и юнит-тестами, мокающими Docker; ни один Strata-контейнер не запускался.
- Точный формат лога при настоящем первом старте. Парсер стадий следует выводу `step()` / `download()` из `setup.py`
  и startup-печатям сервера.
- Поведение без `memlock=-1` и RAM, реально используемая при `LOW_RAM=on` с контейнерным лимитом.
- Вложенное bind-монтирование (`/data/models/<tag>` поверх именованного тома в `/data`) на используемых версиях Docker. Docker
  вложенные монтирования поддерживает, но здесь это не прогонялось.

<a id="known-limitations"></a>
## Известные ограничения

- Только NVIDIA. Путь Strata для AMD (HIP) — экспериментальная ручная сборка в upstream, и не предлагается.
- `LAYER_SPLIT` (размещение слоёв на нескольких GPU) не представлен. Закрепление нескольких GPU использует автоматический сплит Strata.
- Vision-энкодер (mmproj) всегда скачивает сам Strata, в том.
- Собственный веб-UI Strata (`/`, `/settings`, `/load`, `/unload`) достижим через порт инстанса, но llamaMan его
  не использует и не управляет им. Его idle-unload функция не используется; сон llamaMan вместо этого останавливает контейнер.
- Смена настройки, записываемой в setup, перезапускает проход setup Strata при следующем старте (файлы переиспользуются).
- Первый старт может длиться дольше таймаута самого HTTP-клиента (Open WebUI и другие). llamaMan продолжает ждать до
  `STRATA_LOAD_TIMEOUT`, но клиент может сдаться первым. Предварительно скачайте, или запустите один раз из UI, прежде чем открывать его клиентам.
- Каталог (семейства, размеры, закрепленные ревизии, имена файлов) — зеркало `setup.py` Strata. Новое upstream-семейство
  или размер требует парной правки в `core/engines/strata.py`.

<a id="manual-verification-checklist-nvidia-host"></a>
## Чек-лист ручной проверки (хост NVIDIA)

Предпосылки: NVIDIA драйвер 580+, работающий `docker run --gpus all`, рекомендовано 64 ГБ RAM и около 150 ГБ
свободного диска.

1. **Сборка:** `git clone https://github.com/Niko1221/Strata && cd Strata && docker build -t strata .`. Опционально
   добавьте `--build-arg CUDA_ARCHITECTURES=<your arch>`. Или, после шага 2, Settings → Docker Images → *Manage Strata
   images* → **Pull repo & build latest** Собирает только для GPU-поколений этого узла.
2. **Включение:** задайте `STRATA_ENABLED=true` и перезапустите llamaMan.
   - `GET /api/engines` показывает strata `available: true`, с его моделями (9 во встроенном списке).
   - Settings → Docker Images → *Manage Strata images* показывает `strata:latest` как **local**.
3. **Гейтинг:** на не-NVIDIA узле (или с `GPU_TYPE=rocm`) Strata-модели не перечисляются, а
   `POST /api/instances` для `/strata/qwen-IQ2_XS` возвращает 400 с причиной про NVIDIA.
4. **Запуск из UI:** выберите `strata/qwen-IQ2_XS`.
   - Поля только для llama.cpp скрыты, секция Strata показана, и Max Concurrent зафиксирован на 1.
   - Задайте лимит памяти 32g и проверьте, что предупреждение ниже 64 ГБ появилось; снова снимите.
   - Запустите.
5. **Первый прогон (скачивание контейнером):**
   - Карточка инстанса показывает *setting up → downloading NN% → preparing pack → loading*.
   - Кнопка Logs стримит вывод setup Strata.
   - Карточка становится **healthy**, и `docker inspect` показывает `Ulimits: memlock -1`, переменные окружения и
     том `/data`.
6. **Chat:**
   - Через Ollama-прокси: `curl :42069/api/chat -d '{"model":"strata/qwen-IQ2_XS","messages":[...]}'`.
   - Через `/v1/chat/completions` на `:42069` и напрямую на порту инстанса.
   - Попробуйте также `model: "qwen3.8-flash-next-iq2_xs"`.
   - Отправьте два запроса разом: второй встанет в очередь (карточка покажет 1/1 active · 1 queued).
7. **Idle сон и пробуждение:**
   - Поставьте Idle Timeout 1 мин (предупреждение о холодном старте покажется). Через минуту инстанс **sleeping**, и его
     контейнер исчез.
   - Следующий запрос будит его (минуты) и получает ответ.
   - В логе нет прохода setup (нет строк `=== Step`; `REINSTALL` отсутствует в `docker inspect`).
8. **Смена настроек:**
   - Поменяйте context на 65536 и перезапустите.
   - `docker inspect` показывает `REINSTALL=1`, и лог показывает проход setup без скачивания.
   - При следующем перезапуске `REINSTALL` снова отсутствует.
9. **Путь предварительного скачивания:**
   - Выберите другую модель (например `strata/qwen-Q2_0`) и нажмите *Pre-download with llamaMan*.
   - Вкладка Downloads показывает скачивание закреплённой ревизии.
   - Запуск во время скачивания отвергается.
   - После завершения запустите: `docker inspect` показывает `MODELS_DIR/strata/q2_0/Q2_0 → /data/models/Q2_0`, и
     лог не показывает скачивания шардов (только MTP-слой и pack).
10. **Остановка / удаление:** остановите инстанс. Контейнер удалён, и Remove убирает карточку.
11. **Перезапустите llamaMan с работающим Strata-инстансом:**
    - Он переприсоединяется (восстановление состояния), или усыновляется, если состояние потеряно: удалите его из `state.json` и перезапустите.
    - Карточка показывает бейдж Strata и `strata/qwen-IQ2_XS`.
    - Запросы по-прежнему работают.
12. **Режим bare-metal:**
    - Запустите llamaMan через `python app.py` (`LLAMAMAN_IN_DOCKER=false`) и повторите 4-6.
    - llamaMan обращается к опубликованному порту на `LLAMA_HOST_ADDR`.
13. **Кластер:**
    - Добавьте не-NVIDIA пир. Strata-модели появляются для NVIDIA-узла только когда он — Target node.
    - Запуск на NVIDIA-пире из UI другого узла работает.
14. **llama.cpp не затронут:** запустите GGUF-модель как раньше.
    - Командная строка (Args в `docker inspect`) совпадает с прежней, до апгрейда.
    - Пресеты, сохранённые до апгрейда, загружаются и запускаются без изменений.
