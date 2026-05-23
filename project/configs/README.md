# Конфигурация

- `default.yaml` — основной конфиг проекта: пути к данным, random seed, validation split, цены ошибок, параметры permutation importance и настройки сервиса.
- `.env.example` — пример переменных окружения без секретов.

Поддерживаемые переменные:

- `APS_MODEL_PATH` — путь к `best_model.joblib`.
- `APS_CONFIG_PATH` — путь к YAML-конфигу.
- `APS_LOG_LEVEL` — уровень логирования.
- `APS_SERVICE_HOST` — host для сервиса.
- `APS_SERVICE_PORT` — port для сервиса.

Реальные `.env` файлы игнорируются `.gitignore`.
