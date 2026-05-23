# Отчёт по проекту: APS Failure at Scania Trucks

## 1. Паспорт проекта

- **Название проекта:** APS Failure at Scania Trucks
- **Автор:** Балан Фёдор Михайлович
- **Группа:** ИКБО-71-24
- **Контакт:** @latee_fog
- **Домен:** predictive maintenance, табличные сенсорные данные

Проект реализует сервис, который по анонимизированным сенсорным признакам грузовика оценивает риск отказа компонента Air Pressure System. APS используется в важных функциях грузовика, включая торможение и переключение передач, поэтому пропуск реального отказа должен считаться намного более дорогой ошибкой, чем лишняя проверка.

## 2. Постановка задачи

Задача формулируется как бинарная классификация:

- `pos` — отказ связан с APS;
- `neg` — отказ связан с другим компонентом.

Потенциальный пользователь сервиса — инженер технического обслуживания или backend-система мониторинга парка грузовиков. Сценарий: сервис получает текущие агрегированные сенсорные признаки, возвращает вероятность APS-отказа, риск-класс и признаки, которые сильнее всего повлияли на объяснение.

Особенность задачи — сильный дисбаланс классов и разная цена ошибок. В описании UCI задана стоимость:

- ложная тревога `FP`: `10`;
- пропуск APS-отказа `FN`: `500`.

Поэтому модель сравнивается не только по PR-AUC/F1/recall, но и по итоговой стоимости ошибок.

## 3. Данные

Источник — открытый датасет UCI **APS Failure at Scania Trucks**. Исходные файлы перенесены в проект:

- `data/raw/aps_failure_training_set.csv`
- `data/raw/aps_failure_test_set.csv`
- `data/raw/aps_failure_description.txt`

В датасете 170 анонимизированных числовых признаков. Пропуски обозначены строкой `na`; загрузчик `src/aps_failure/data.py` читает первые 20 строк метаданных корректно, преобразует `na` в `NaN`, а таргет `pos/neg` — в `1/0`.

Ключевые факты EDA:

- train: `60000` строк, `1000` positive и `59000` negative;
- test: `16000` строк, `375` positive и `15625` negative;
- доля positive в train: `1.67%`;
- средняя доля пропусков по train: `8.33%`;
- наиболее пропущенные признаки: `br_000`, `bq_000`, `bp_000`, `bo_000`, `ab_000`, `cr_000`.

EDA и воспроизводимый сценарий лежат в `notebooks/01_eda_and_experiments.ipynb`, автоматические summary — в `artifacts/data_quality.json`.

## 4. Модели

Были обучены четыре подхода:

- `LogisticRegression` — линейный baseline с median imputation, missing indicators, scaling и `class_weight=balanced`;
- `RandomForest` — ансамбль деревьев с median imputation, missing indicators и `balanced_subsample`;
- `HistGradientBoosting` — градиентный бустинг sklearn с поддержкой пропусков и `class_weight=balanced`;
- `MLP` — простая нейросеть для табличного baseline.

Для каждой модели decision threshold подбирался на validation split по минимальной стоимости `10 * FP + 500 * FN`. Затем выбранный порог применялся к официальному UCI test set.

## 5. Результаты

| Модель | Threshold | PR-AUC test | Recall pos | F1 | Cost test |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.365731 | 0.7980 | 0.9440 | 0.5549 | 15970 |
| RandomForest | 0.105209 | 0.7908 | 0.9707 | 0.5661 | 10970 |
| HistGradientBoosting | 0.121241 | 0.9033 | 0.9653 | 0.6290 | 10640 |
| MLP | 0.002270 | 0.8481 | 0.9147 | 0.5241 | 21910 |

Финальная модель — `HistGradientBoosting`, потому что она дала лучший validation cost и одновременно лучший test PR-AUC. На test set:

- `TP=362`;
- `FN=13`;
- `FP=414`;
- `TN=15211`;
- `Recall=0.9653`;
- `PR-AUC=0.9033`;
- `ROC-AUC=0.9934`;
- `Total cost=10640`.

Важные признаки по permutation importance: `ck_000`, `aa_000`, `ay_006`, `bj_000`, `ai_000`, `az_002`. Графики сохранены в `artifacts/figures/`.

## 6. Архитектура и сервис

Пайплайн:

1. `src/aps_failure/data.py` загружает UCI CSV и готовит признаки/таргет.
2. `src/aps_failure/modeling.py` создаёт набор моделей.
3. `src/aps_failure/train.py` обучает модели, выбирает порог, сохраняет артефакт.
4. `src/aps_failure/service.py` загружает `artifacts/best_model.joblib` и обслуживает REST API.

API:

- `GET /health` — health-check и информация о модели;
- `GET /metrics` — простые счётчики запросов/ошибок;
- `POST /predict-failure` — основной endpoint;
- `POST /predict` — alias.

Контракт `/predict-failure`:

```json
{
  "request_id": "demo-positive-scania-aps",
  "features": {
    "aa_000": 153204.0,
    "ab_000": null
  }
}
```

Сервис допускает частично заполненный JSON: отсутствующие признаки превращаются в `NaN`, а неизвестные признаки возвращаются в поле `unknown_features`.

## 7. Наблюдаемость, конфигурация, безопасность

Наблюдаемость:

- endpoints логируют health-check, metrics-запросы и prediction latency;
- `/health` показывает состояние модели;
- `/metrics` отдаёт counters в Prometheus-style формате.

Конфигурация:

- `configs/default.yaml` хранит пути, random seed, цены ошибок, параметры обучения;
- `configs/.env.example` показывает переменные окружения;
- реальные `.env` игнорируются `.gitignore`.

Безопасность:

- датасет открытый и обезличенный;
- секреты не используются и не добавлены в репозиторий;
- домашние задания не изменялись.

## 8. Ограничения и дальнейшая работа

Ограничения текущей версии:

- признаки анонимизированы, поэтому предметная интерпретация ограничена;
- API использует один локальный joblib-артефакт без model registry;
- explanation основан на permutation importance и отклонении от training median, без SHAP;
- нет онлайн-мониторинга data drift.

Развитие:

- добавить SHAP для локальных объяснений;
- добавить MLflow/DVC для версионирования экспериментов и данных;
- настроить мониторинг дрейфа входных признаков;
- сравнить с LightGBM/XGBoost при доступности зависимостей.

## 9. Сценарий защиты

1. Показать структуру `project/`, данные в `data/raw/`, артефакты в `artifacts/`.
2. Открыть `artifacts/runs.csv` и сравнить модели по PR-AUC, recall и cost.
3. Запустить API:

```bash
.venv/bin/uvicorn src.service:app --host 0.0.0.0 --port 8000
```

4. Проверить:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict-failure \
  -H "Content-Type: application/json" \
  --data @data/demo_payload.json
```

5. Объяснить, почему выбран не порог `0.5`, а cost-sensitive threshold `0.121241`.
