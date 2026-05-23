# Ноутбуки проекта

- `01_eda_and_experiments.ipynb` — стартовый EDA для Scania APS, проверка дисбаланса, пропусков и связь с training pipeline.
- `02_data_quality_deep_dive.ipynb` — детальный анализ качества данных: пропуски по признакам и строкам, train/test drift, PSI, кандидаты на ручную проверку.
- `03_feature_signal_analysis.ipynb` — анализ сигнальности признаков: univariate Average Precision, missingness-as-signal, распределения топ-сенсоров и корреляции.
- `04_model_diagnostics_and_error_analysis.ipynb` — диагностика обученной модели: сравнение моделей, cost-sensitive threshold, FP/FN, калибровочные бины и примеры ошибок.

Основная логика проекта вынесена в `src/`, поэтому ноутбуки играют роль воспроизводимых аналитических сценариев, а не единственного места с кодом.
