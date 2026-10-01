# YandexSmartAI — детекция графических объектов (иконок) с YOLOv8

Конвейер из четырёх частей: генерация синтетического датасета → обучение YOLOv8 →
оценка → GUI для инференса и анализа результатов. 139 классов иконок из `objects/`,
изображения 320×180.

## Структура

| Путь | Назначение |
|---|---|
| `generate_dataset.py`, `config.py`, `utils/` | генератор синтетического датасета (JSON + YOLO разметка) |
| `train_yolo.py` | обучение / дообучение / возобновление (обёртка над Ultralytics) |
| `evaluate_model.py` | оценка модели на любом размеченном наборе (в т.ч. реальных изображениях) |
| `app_flet.py`, `gui/` | GUI на Flet: загрузка изображений, инференс, графики, экспорт JSON/CSV |
| `prepare_yolo_model.py` | скачивание предобученных весов в `weights/` |
| `download_backgrounds_uuid.py` | скачивание фонов |
| `make_archive.py` | сборка архива проекта для сдачи/переноса |
| `tests/` | автотесты (`unittest`) |

## Установка

```bat
setup_train.bat      :: окружение для обучения (torch 2.5.1 + CUDA 12.1)
run_gui.bat          :: запуск GUI (при первом запуске ставит зависимости)
```

Версии зависимостей закреплены в `requirements.txt` и `requirements_gui.txt`
(Python 3.10, ultralytics 8.4.23, flet 0.82.2).

## Генерация датасета

```bat
python generate_dataset.py --output output_50k --count 50000 --num-workers 8 --no-preview
```

- Результат полностью определяется `--seed` (по умолчанию 42), профилем `--bg-profile`,
  `--num-workers` и `--worker-chunk-size`. **Тот же seed с теми же параметрами даёт тот же
  датасет** — для нового набора данных меняйте `--seed`.
- Выходная папка пересоздаётся, только если в ней лежит датасет этого генератора
  (`train/`, `val/`, `data.yaml`, `classes.json`). Корень проекта, `objects/`, `backgrounds/`
  и их родительские папки удалить нельзя.
- Рамки берутся по видимому объекту (порог альфа-канала `bbox_alpha_threshold = 128`),
  без полупрозрачной каймы, которую оставляет бикубическая интерполяция (раньше рамки
  были в среднем на 2–3 px шире).
- Классы-«буквы» (`no_flip_classes`: `a_word`, `m_word`, `t_word`, `f_x`, `circle_s`)
  не отражаются — зеркальных букв в реальных данных нет.
- `data.yaml` не содержит ключа `path`: пути считаются относительно самого yaml, поэтому
  датасет работает из любой текущей папки и после переноса.

## Обучение

```bat
python prepare_yolo_model.py
python train_yolo.py --data output_50k/data.yaml --weights weights/yolov8n.pt ^
    --epochs 40 --imgsz 416 --batch 32 --workers 10 --name yolov8n_50k
```

`--fliplr` по умолчанию 0: отражения делает генератор (с учётом `no_flip_classes`),
а отражение всего кадра в Ultralytics зеркалило бы и буквы.

### Дообучение (fine-tuning)

Дообучать имеет смысл **на новых данных** (другой `--seed` или профиль) и **с малым LR без
прогрева**. При `--optimizer auto` Ultralytics игнорирует `--lr0` и снова стартует
с LR ≈ 0.01, поэтому оптимизатор нужно указать явно:

```bat
python train_yolo.py --data output_50k_seed7/data.yaml ^
    --weights runs/train/yolov8n_50k_maxspeed2/weights/best.pt ^
    --optimizer SGD --lr0 0.001 --warmup-epochs 0 --epochs 15 --imgsz 416 --batch 32 --name finetune_seed7
```

### Возобновление

`--resume-from runs/train/<run>/weights/last.pt`. При возобновлении Ultralytics берёт
гиперпараметры из чекпойнта; менять можно только `--batch`, `--device`, `--workers`
(`--amp` игнорируется — скрипт об этом предупреждает).

## Оценка модели

Валидационная выборка синтетическая и порождена тем же генератором, поэтому её метрики
(mAP50 ≈ 0.995) почти ничего не говорят о качестве на реальных изображениях. Для честной
оценки нужен отдельный размеченный набор реальных изображений:

```
real_test/
  images/      *.png / *.jpg
  labels/      *.txt  (YOLO: class x_center y_center width height, нормированные)
  data.yaml
```

```yaml
# real_test/data.yaml
train: images
val: images
nc: 139
names: { ... скопировать из output_50k/data.yaml ... }
```

```bat
python evaluate_model.py --weights runs/train/<run>/weights/best.pt --data real_test/data.yaml
```

Метрики и 10 худших классов сохраняются в `runs/val/<run>_<dataset>_val/metrics.json`.

> Модели, обученные до исправления рамок, предсказывают «раздутые» рамки. На наборе
> с новой разметкой у них mAP50-95 занижен (0.93 → 0.75 на одних и тех же изображениях),
> поэтому сравнивать модели нужно на одной и той же разметке, а для выигрыша от новой
> разметки — перегенерировать датасет и переобучить модель.

## GUI

```bat
run_gui.bat
```

- Модель по умолчанию — последняя выбранная вручную, иначе `best.pt` из `runs/train`
  с наибольшим mAP50-95 (затем обученные веса из `weights/`; предобученные COCO-веса
  не предлагаются).
- Инференс выполняется с порогом 0.05, ползунок фильтрует результаты от 0.05 до 0.95.
  Аннотированное изображение, таблица, графики и экспортируемые JSON/CSV соответствуют
  текущему положению ползунка.
- Аннотированные изображения сохраняются в `gui_outputs/annotated/`, тема и выбранная
  модель — в `gui_outputs/gui_state.json`.

## Тесты

```bat
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

## Архив проекта

```bat
python make_archive.py --extra "presentation.pptx"
```

Включает код, `objects/`, `backgrounds/` (отключается `--no-backgrounds`),
`weights/yolov8n.pt`, артефакты запусков `runs/train` с `best.pt`. Имена файлов
сохраняются в UTF-8.

## История экспериментов

| Запуск | Данные | Эпох | Время | mAP50 | mAP50-95 (val) |
|---|---|---|---|---|---|
| `yolov8n_gpu` (18.03) | `output`, ~7.4 тыс. изображений, imgsz 320 (папка удалена) | 100 | — | 0.994 | 0.953 |
| `yolov8n_50k_maxspeed2` | `output_50k`, imgsz 416 | 40 | 4.6 ч | 0.995 | 0.939 |
| `yolov8n_50k_finetune_20260324` | `output_50k_finetune_20260324` | 20 | 2.2 ч | 0.995 | 0.938 |

- Датасет `output_50k_finetune_20260324` побайтно совпадает с `output_50k` (тот же seed),
  а дообучение шло с `optimizer=auto` (LR снова 0.01) — поэтому прироста нет.
- Стартовые веса дообучения (`backups/models/yolov8n_50k_maxspeed2_20260324_031220/best.pt`)
  не сохранились; по времени создания это копия
  `runs/train/yolov8n_50k_maxspeed2/weights/best.pt`.
- Первая строка не сравнима с остальными: другая валидационная выборка и параметры генерации.
- Все запуски обучены на разметке с «раздутыми» рамками (до исправления генератора).
