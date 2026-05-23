# Самопроверка проекта

| # | Критерий | Да/Нет (студент) | Где смотреть / комментарий |
|---|---|---|---|
| 1 | Сервис запускается по инструкциям из `project/README.md` и работает | ✅ | `README.md`, `src/aps_failure/service.py`, `/health` |
| 2 | Endpoint `/predict` использует реальную модель, а не заглушку | ✅ | `artifacts/best_model.joblib`, `src/aps_failure/service.py` |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/01_eda_and_experiments.ipynb`, `artifacts/metrics.json`, `report.md` |
| 4 | Есть baseline и улучшенная модель, есть сравнение по метрикам | ✅ | `artifacts/runs.csv`, `report.md` |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/aps_failure/data.py`, `modeling.py`, `train.py`, `service.py` |
| 6 | Есть Dockerfile или понятный сценарий развёртывания без Docker | ✅ | `Dockerfile`, `README.md` |
| 7 | Есть `.env.example` и нет реальных секретов/паролей | ✅ | `configs/.env.example`, `.gitignore` |
| 8 | Реализованы логи/наблюдаемость | ✅ | endpoint logging, `/health`, `/metrics` |
| 9 | В `report.md` обоснован выбор финальной модели | ✅ | `report.md`, разделы 4-5 |
| 10 | `project/README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md`, `report.md`, раздел «Сценарий защиты» |

Итого: **10/10**.
