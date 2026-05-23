# APS Failure at Scania Trucks

End-to-end мини-проект по предиктивному обслуживанию грузовиков Scania. Сервис оценивает вероятность того, что отказ грузовика связан с Air Pressure System (APS), по 170 анонимизированным сенсорным признакам.

## Паспорт

- **Название:** APS Failure at Scania Trucks
- **Автор:** Балан Фёдор Михайлович
- **Группа:** ИКБО-71-24
- **Контакт:** @latee_fog
- **Задача:** бинарная классификация редкого отказа APS (`pos`) против отказов не-APS (`neg`)
- **Главный акцент:** дисбаланс классов и cost-sensitive оценка, где пропуск реального APS-отказа сильно дороже ложной тревоги

## Структура

- `data/raw/` — исходные UCI-файлы Scania APS, перенесённые в проект.
- `data/demo_payload.json` — пример запроса для API.
- `src/aps_failure/` — загрузка данных, обучение, метрики, графики и FastAPI-сервис.
- `configs/default.yaml` — пути, random seed, цены ошибок и параметры обучения.
- `configs/.env.example` — шаблон переменных окружения без секретов.
- `artifacts/` — обученная модель, метрики, schema и графики.
- `notebooks/` — EDA, data quality deep dive, feature signal analysis и диагностика ошибок модели.
- `tests/` — smoke/sanity тесты загрузки данных и API.
- `Dockerfile` — контейнер для API.

## Установка

```bash
cd project
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Обучение модели

```bash
cd project
.venv/bin/python -m src.train
```

Скрипт обучает `LogisticRegression`, `RandomForest`, `HistGradientBoosting` и `MLP`, подбирает decision threshold по стоимости ошибок APS и сохраняет:

- `artifacts/best_model.joblib`
- `artifacts/metrics.json`
- `artifacts/runs.csv`
- `artifacts/figures/*.png`
- `data/demo_payload.json`

## Запуск API

```bash
cd project
.venv/bin/uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Основные endpoints:

- `GET /health` — состояние сервиса и факт загрузки модели.
- `GET /metrics` — простые Prometheus-style счётчики.
- `POST /predict-failure` — предсказание APS-риска.
- `POST /predict` — короткий alias для того же сценария.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict-failure \
  -H "Content-Type: application/json" \
  --data @data/demo_payload.json
```

Ответ содержит вероятность APS-отказа, класс `pos/neg`, уровень риска, используемый порог, cost policy и топ-признаки для объяснения.

## Docker

```bash
cd project
docker build -t aps-failure-scania .
docker run -p 8000:8000 aps-failure-scania
```

Перед сборкой контейнера желательно один раз выполнить обучение, чтобы `artifacts/best_model.joblib` уже был внутри проекта.

## Результаты

Финально выбран `HistGradientBoosting` с порогом `0.121241`.

| Модель | PR-AUC test | Recall pos | F1 | Cost test |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.7980 | 0.9440 | 0.5549 | 15970 |
| RandomForest | 0.7908 | 0.9707 | 0.5661 | 10970 |
| HistGradientBoosting | 0.9033 | 0.9653 | 0.6290 | 10640 |
| MLP | 0.8481 | 0.9147 | 0.5241 | 21910 |

Для выбранной модели на официальном test set: `TP=362`, `FN=13`, `FP=414`, `TN=15211`, `ROC-AUC=0.9934`.

## Тесты

```bash
cd project
.venv/bin/python -m pytest
```

В `pytest.ini` отключены сторонние ROS-плагины pytest, чтобы тесты проекта не зависели от системного окружения.

## Демонстрация

1. Показать `artifacts/runs.csv` и графики из `artifacts/figures/`.
2. Запустить API через `uvicorn`.
3. Проверить `/health`.
4. Отправить `data/demo_payload.json` в `/predict-failure`.
5. Пояснить, почему выбран cost-sensitive threshold: пропуск APS-отказа стоит `500`, ложная тревога стоит `10`.
