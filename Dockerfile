FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# offset и сессии на томе
RUN mkdir -p /data
ENV YANDEX_OFFSET_FILE=/data/yandex_updates_offset.txt

CMD ["python", "-m", "app.runner"]
