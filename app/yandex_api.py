from __future__ import annotations

from typing import Any

import httpx

from app.config import settings

BOT_API = "https://botapi.messenger.yandex.net/bot/v1"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"OAuth {settings.yandex_bot_token}"}


async def get_updates(*, offset: int, limit: int) -> dict[str, Any]:
    """Короткий polling: период задаётся YANDEX_POLL_INTERVAL_SEC."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            f"{BOT_API}/messages/getUpdates/",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"limit": limit, "offset": offset},
        )
    r.raise_for_status()
    return r.json()


async def send_text(
    *,
    login: str | None,
    chat_id: str | None,
    text: str,
    suggest_buttons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"text": text}
    if login:
        body["login"] = login
    if chat_id:
        body["chat_id"] = chat_id
    if suggest_buttons:
        body["suggest_buttons"] = suggest_buttons
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"{BOT_API}/messages/sendText/",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=body,
        )
    r.raise_for_status()
    return r.json()


async def send_image_bytes(
    *,
    login: str | None,
    chat_id: str | None,
    image_bytes: bytes,
    caption: str | None = None,
    filename: str = "result.jpg",
) -> dict[str, Any]:
    """Отправка jpeg; caption через отдельное сообщение — sendImage не всегда поддерживает подпись."""
    data: dict[str, Any] = {}
    if login:
        data["login"] = login
    if chat_id:
        data["chat_id"] = chat_id
    files = {"image": (filename, image_bytes, "image/jpeg")}
    if caption:
        await send_text(login=login, chat_id=chat_id, text=caption[:5900])
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(
            f"{BOT_API}/messages/sendImage/",
            headers=_auth_headers(),
            data=data,
            files=files,
        )
    r.raise_for_status()
    return r.json()


async def download_file(file_id: str) -> tuple[bytes, str]:
    clean_id = file_id.split("?")[0]
    # Рабочий вариант (как в yandex-bot-py): GET + file_id в query, без тела — иначе часто 415.
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.get(
            f"{BOT_API}/messages/getFile/",
            headers=_auth_headers(),
            params={"file_id": clean_id},
        )
    r.raise_for_status()
    ctype = r.headers.get("content-type", "application/octet-stream")
    return r.content, ctype


def main_menu_buttons() -> dict[str, Any]:
    return {
        "layout": "true",
        "persist": True,
        "buttons": [
            [
                {
                    "id": "btn_compare",
                    "title": "🧑‍🤝‍🧑 1 к 1",
                    "directives": [{"type": "server_action", "name": "flow_compare", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_attrs",
                    "title": "👤 Атрибуты",
                    "directives": [{"type": "server_action", "name": "flow_attributes", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_body_attrs",
                    "title": "🧍 Атрибуты тела",
                    "directives": [{"type": "server_action", "name": "flow_body_attributes", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_liveness",
                    "title": "🫀 Лайфнесс",
                    "directives": [{"type": "server_action", "name": "flow_liveness", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_deepfake",
                    "title": "🎭 Дипфейк",
                    "directives": [{"type": "server_action", "name": "flow_deepfake", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_quality",
                    "title": "🖼️ Качество изображения",
                    "directives": [{"type": "server_action", "name": "flow_quality", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_faces_detect",
                    "title": "👥 Детекция лиц",
                    "directives": [{"type": "server_action", "name": "flow_face_detect", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_bodies_detect",
                    "title": "🧍 Тела",
                    "directives": [{"type": "server_action", "name": "flow_body_detect", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_crowd_detect",
                    "title": "🧑‍🤝‍🧑 Детекция толпы",
                    "directives": [{"type": "server_action", "name": "flow_crowd_detect", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_image_modification",
                    "title": "🛠️ Модификация изображения",
                    "directives": [{"type": "server_action", "name": "flow_image_modification", "payload": {}}],
                }
            ],
            [
                {
                    "id": "btn_cancel",
                    "title": "♻️ Сброс",
                    "directives": [{"type": "server_action", "name": "flow_reset", "payload": {}}],
                }
            ],
        ],
    }
