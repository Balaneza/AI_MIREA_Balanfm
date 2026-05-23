# Исходный код проекта

Команды запуска из папки `project/`:

```bash
.venv/bin/python -m src.train
.venv/bin/uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Основные модули:

- `src/train.py` — точка входа для обучения.
- `src/service.py` — точка входа FastAPI-приложения.
- `src/aps_failure/data.py` — корректная загрузка UCI CSV, обработка `na`, сбор входного DataFrame для API.
- `src/aps_failure/modeling.py` — фабрика моделей: Logistic Regression, Random Forest, HistGradientBoosting, MLP.
- `src/aps_failure/metrics.py` — PR-AUC, recall/F1, confusion matrix и cost-sensitive threshold.
- `src/aps_failure/train.py` — полный training pipeline и сохранение артефактов.
- `src/aps_failure/service.py` — `/health`, `/metrics`, `/predict-failure`, `/predict`.
