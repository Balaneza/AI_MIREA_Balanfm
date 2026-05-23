# Артефакты

Содержимое после запуска:

- `best_model.joblib` — финальный sklearn pipeline и метаданные сервиса.
- `metrics.json` — все результаты обучения, выбранная модель, threshold, data quality и top features.
- `runs.csv` — компактная таблица сравнения моделей.
- `data_quality.json` — EDA summary по train/test.
- `feature_schema.json` — список 170 признаков и target schema.
- `figures/` — class balance, missingness, PR-curve, confusion matrix, permutation importance.

Финальная модель выбрана по validation total cost с политикой `FP=10`, `FN=500`.
