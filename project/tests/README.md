# Тесты

Запуск:

```bash
cd project
.venv/bin/python -m pytest
```

Что проверяется:

- `test_data.py` — UCI CSV читается корректно: 60 000 строк, 170 признаков, 1 000 positive.
- `test_service.py` — FastAPI-приложение загружает реальный model artifact, а модель скорит `data/demo_payload.json`.

В `pytest.ini` отключены ROS-плагины pytest из системного окружения, чтобы тесты не падали из-за внешних зависимостей, не относящихся к проекту.
