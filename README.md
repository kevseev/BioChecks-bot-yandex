# BioChecks Bot (Yandex Messenger + LUNA)

Бот для Яндекс Мессенджера, который через LUNA Platform выполняет биометрические проверки и анализ изображений.

## Возможности

- 🧑‍🤝‍🧑 **1 к 1**: сравнение двух лиц (SDK descriptor + matcher/raw)
- 👤 **Атрибуты лица**: все лица на фото + атрибуты
- 🧍 **Атрибуты тела**: все тела на фото + body-атрибуты
- 🫀 **Liveness**
- 🎭 **Deepfake**
- 🖼️ **Качество изображения** (`estimate_quality`)
- 🧑‍🤝‍🧑 **Детекция толпы** (`estimate_people_count`)
- 👥 **Детекция лиц**
- 🧍 **Детекция тел**
- 🛠️ **Модификация изображения** (`estimate_image_modification`)

Поддерживаются:
- отправка файлов в чат
- ссылки на изображения (включая популярные public-share ссылки)

## Архитектура

- `app/runner.py` — polling `getUpdates` для Яндекс Мессенджера
- `app/handlers.py` — сценарии кнопок и маршрутизация событий
- `app/luna.py` — вызовы LUNA SDK/matcher и форматирование ответов
- `app/yandex_api.py` — клиент Bot API Яндекс Мессенджера
- `app/url_images.py` — загрузка изображений по URL
- `Dockerfile`, `docker-compose.yml` — контейнерный запуск

## Настройка

1. Скопируйте `.env.example` в `.env`
2. Заполните:
   - `YANDEX_BOT_TOKEN`
   - `LUNA_BASE_URL` (например `http://host.docker.internal:5000/6`)
   - `LUNA_HTTP_USER` / `LUNA_HTTP_PASSWORD` (или bearer)

## Запуск (Docker)

```bash
docker compose up -d --build
docker compose logs -f bot
```

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.runner
```

## Release/CI

В репозитории настроен GitHub Actions workflow:
- сборка и публикация Docker-образа в GHCR при тегах `v*`
- автоматическое создание GitHub Release

