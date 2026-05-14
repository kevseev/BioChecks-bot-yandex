from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from app import access_control
from app.config import settings
from app import luna, draw
from app.state import Flow, get_session, session_key
from app.url_images import extract_urls, fetch_image_from_url
from app.yandex_api import (
    download_file,
    main_menu_buttons,
    send_image_bytes,
    send_text,
)

log = logging.getLogger(__name__)

SEEN: deque[int] = deque(maxlen=20000)
SEEN_SET: set[int] = set()

COMPARE_PARAMS: dict[str, int | float] = {
    "multiface_policy": 2,
    "detect_face": 1,
    "detect_body": 0,
    "estimate_face_descriptor": 1,
    "score_threshold": 0.5,
}

ATTR_PARAMS: dict[str, int | float] = {
    "multiface_policy": 1,
    "detect_face": 1,
    "detect_body": 0,
    "estimate_head_pose": 1,
    "estimate_emotions": 1,
    "estimate_mask": 1,
    "estimate_glasses": 1,
    "estimate_eyes_attributes": 1,
    "estimate_mouth_attributes": 1,
    "estimate_face_occlusion": 1,
    "estimate_deepfake": 1,
    "score_threshold": 0.5,
}

BODY_ATTR_PARAMS: dict[str, int | float] = {
    "multiface_policy": 1,
    "detect_face": 0,
    "detect_body": 1,
    "estimate_body_descriptor": 1,
    "estimate_body_basic_attributes": 1,
    "estimate_upper_body": 1,
    "estimate_lower_body": 1,
    "estimate_accessories": 1,
    "score_threshold": 0.5,
}

LIVENESS_PARAMS: dict[str, int | float] = {
    "multiface_policy": 2,
    "detect_face": 1,
    "detect_body": 0,
    "estimate_liveness": 1,
    "score_threshold": 0.5,
}

DEEPFAKE_PARAMS: dict[str, int | float] = {
    "multiface_policy": 2,
    "detect_face": 1,
    "detect_body": 0,
    "estimate_deepfake": 1,
    "score_threshold": 0.5,
}

QUALITY_PARAMS: dict[str, int | float] = {
    "multiface_policy": 2,
    "detect_face": 1,
    "detect_body": 0,
    "estimate_quality": 1,
    "score_threshold": 0.5,
}

FACE_DETECT_PARAMS: dict[str, int | float] = {
    "multiface_policy": 1,
    "detect_face": 1,
    "detect_body": 0,
    "score_threshold": 0.5,
}

BODY_DETECT_PARAMS: dict[str, int | float] = {
    "multiface_policy": 1,
    "detect_face": 0,
    "detect_body": 1,
    "score_threshold": 0.5,
}

IMAGE_MODIFICATION_PARAMS: dict[str, int | float] = {
    "multiface_policy": 0,
    "detect_face": 0,
    "detect_body": 0,
    "estimate_image_modification": 1,
    "score_threshold": 0.5,
}

CROWD_DETECT_PARAMS: dict[str, int | float] = {
    "multiface_policy": 1,
    "detect_face": 0,
    "detect_body": 0,
    "estimate_people_count": 1,
    "people_count_coordinates": 1,
}


def _principal(update: dict[str, Any]) -> str | None:
    """Идентификатор пользователя для пароля: login, иначе id из from."""
    from_ = update.get("from") or {}
    login = from_.get("login")
    if isinstance(login, str) and login.strip():
        return login.strip().lower()
    uid = from_.get("id")
    if uid is not None and str(uid).strip():
        return f"id:{uid}"
    return None


async def _admin_text_commands(
    target: dict[str, str | None],
    raw_text: str,
    principal: str,
) -> bool:
    if principal not in access_control.admin_logins():
        return False
    t = raw_text.strip()
    low = t.lower()
    if low in ("!users", "/users"):
        users = access_control.list_user_logins()
        body = "Пользователи с доступом:\n" + (
            "\n".join(f"• {u}" for u in users) if users else "(пока никого)"
        )
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=body[:5900],
        )
        return True
    revoked = access_control.parse_admin_revoke(t)
    if revoked:
        if access_control.delete_user(revoked):
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Доступ для `{revoked}` отозван.",
            )
        else:
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Логин `{revoked}` не найден в списке.",
            )
        return True
    issue_login = access_control.parse_admin_issue(t)
    if issue_login:
        pw = access_control.generate_password()
        access_control.set_user_password(issue_login, pw)
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=(
                f"Пароль для пользователя `{issue_login}`:\n{pw}\n\n"
                "Передайте его по защищённому каналу; повторно тот же текст бот не пришлёт."
            )[:5900],
        )
        return True
    return False


def _target(update: dict[str, Any]) -> dict[str, str | None]:
    chat = update.get("chat") or {}
    from_ = update.get("from") or {}
    if chat.get("type") == "private":
        return {"login": from_.get("login"), "chat_id": None}
    return {"login": None, "chat_id": chat.get("id")}


def _norm_ct(ctype: str) -> str:
    c = (ctype or "").split(";")[0].strip().lower()
    if c in ("application/octet-stream", "", "binary/octet-stream"):
        return "image/jpeg"
    return c


async def _reply_menu(msg: str, target: dict[str, str | None]) -> None:
    await send_text(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        text=msg,
        suggest_buttons=main_menu_buttons(),
    )


def _extract_server_action(update: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    br = update.get("bot_request") or {}
    sa = br.get("server_action") or update.get("server_action") or {}
    name = sa.get("name")
    payload = sa.get("payload") or {}
    return name, payload if isinstance(payload, dict) else {}


def _gallery_file_ids(update: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for group in update.get("images") or []:
        if not group:
            continue
        fid = luna.pick_original_file_id(group)
        if fid:
            out.append(fid)
    return out


def _seen(u: int) -> bool:
    if u in SEEN_SET:
        return True
    SEEN.append(u)
    SEEN_SET.add(u)
    if len(SEEN) > 15000:
        old = SEEN.popleft()
        SEEN_SET.discard(old)
    return False


def _similarity_percent(match_json: dict[str, Any]) -> float | None:
    try:
        m = match_json["matches"][0]["matches"][0]["similarity"]
        return float(m) * 100.0
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _guess_image_content_type(data: bytes, reported: str) -> str:
    r = (reported or "").split(";")[0].strip().lower()
    if r.startswith("image/") and r != "image/octet-stream":
        return r
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


async def _run_compare(
    target: dict[str, str | None],
    images: list[tuple[bytes, str]],
) -> None:
    if len(images) < 2:
        await _reply_menu("Нужно два снимка.", target)
        return
    (a, cta), (b, ctb) = images[0], images[1]
    cta = _guess_image_content_type(a, cta)
    ctb = _guess_image_content_type(b, ctb)
    try:
        ja = await luna.sdk_analyze(a, cta, COMPARE_PARAMS)
        jb = await luna.sdk_analyze(b, ctb, COMPARE_PARAMS)
    except Exception as e:
        log.exception("sdk compare")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    fa = luna.iter_faces_from_sdk(ja)
    fb = luna.iter_faces_from_sdk(jb)
    if not fa or not fb:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="На одном из снимков лицо не найдено. Пришлите другое фото.",
        )
        return

    d0 = luna.get_descriptor_b64(fa[0])
    d1 = luna.get_descriptor_b64(fb[0])
    if not d0 or not d1:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Не удалось извлечь дескриптор (face descriptor).",
        )
        return

    try:
        mraw = await luna.matcher_raw_sdk(d0, d1)
    except Exception as e:
        log.exception("matcher")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка matcher/raw: {e}"[:3000],
        )
        return

    pct = _similarity_percent(mraw)
    out_a = draw.draw_boxes_on_image(a, [fa[0]], highlight_index=0, colors=[(255, 0, 0)])
    out_b = draw.draw_boxes_on_image(b, [fb[0]], highlight_index=0, colors=[(0, 180, 0)])
    sim_text = f"🧑‍🤝‍🧑 Схожесть: {pct:.2f}%" if pct is not None else "🧑‍🤝‍🧑 Схожесть: (нет в ответе matcher)"
    await send_text(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        text=sim_text,
    )
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=out_a,
        filename="face_a.jpg",
    )
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=out_b,
        filename="face_b.jpg",
    )
    await _reply_menu("✅ Готово. Можно выбрать следующее действие.", target)


async def _run_attributes(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j, iso_exc = await asyncio.gather(
            luna.sdk_analyze(image, ct, ATTR_PARAMS),
            luna.check_iso(image, ct, multiface_policy=2),
            return_exceptions=True,
        )
    except Exception as e:
        log.exception("sdk attr")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    if isinstance(j, Exception):
        log.exception("sdk attr", exc_info=j)
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {j}"[:3000],
        )
        return

    iso_payload: dict[str, Any] | None = None
    iso_text: str | None = None
    if isinstance(iso_exc, Exception):
        log.warning("check_iso: %s", iso_exc)
    else:
        iso_payload = iso_exc
        iso_text = luna.format_iso_check_ru(iso_exc)

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="👤 Лица не обнаружены.",
        )
        return

    if iso_text:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=iso_text[:5900],
        )

    for i, face in enumerate(faces):
        pict = draw.draw_boxes_on_image(
            image, [face], highlight_index=0, colors=[(64, 64, 255)]
        )
        cap = f"👤 Лицо {i + 1} / {len(faces)}\n\n" + luna.format_face_attributes(
            face, iso_json=iso_payload, face_index=i
        )
        await send_image_bytes(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            image_bytes=pict,
            caption=cap,
            filename=f"face_{i+1}.jpg",
        )
    await _reply_menu("✅ Атрибуты отправлены.", target)


async def _run_body_attributes(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, BODY_ATTR_PARAMS)
    except Exception as e:
        log.exception("sdk body attrs")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    bodies = luna.iter_bodies_from_sdk(j)
    if not bodies:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="🧍 Тела не обнаружены.",
        )
        return

    for i, body in enumerate(bodies):
        pict = draw.draw_boxes_on_image(
            image, [body], highlight_index=0, colors=[(0, 200, 140)]
        )
        cap = f"🧍 Тело {i + 1} / {len(bodies)}\n\n" + luna.format_body_attributes(body)
        await send_image_bytes(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            image_bytes=pict,
            caption=cap,
            filename=f"body_{i+1}.jpg",
        )
    await _reply_menu("✅ Атрибуты тела отправлены.", target)


async def _run_liveness(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, LIVENESS_PARAMS)
    except Exception as e:
        log.exception("sdk liveness")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="👤 Лицо не обнаружено — нужен снимок с лицом.",
        )
        return

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(200, 100, 0)])
    cap = "🫀 Liveness (лучшее лицо на кадре)\n\n" + luna.format_liveness(face)
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=pict,
        caption=cap,
        filename="liveness.jpg",
    )
    await _reply_menu("✅ Готово.", target)


async def _run_deepfake(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, DEEPFAKE_PARAMS)
    except Exception as e:
        log.exception("sdk deepfake")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="👤 Лицо не обнаружено — нужен снимок с лицом.",
        )
        return

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(160, 60, 200)])
    cap = "🎭 Дипфейк (лучшее лицо на кадре)\n\n" + luna.format_deepfake(face)
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=pict,
        caption=cap,
        filename="deepfake.jpg",
    )
    await _reply_menu("✅ Готово.", target)


async def _run_quality(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, QUALITY_PARAMS)
    except Exception as e:
        log.exception("sdk quality")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="👤 Лицо не обнаружено — нужен снимок с лицом.",
        )
        return

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(80, 160, 255)])
    cap = luna.format_quality_ru(j)
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=pict,
        caption=cap,
        filename="quality.jpg",
    )
    await _reply_menu("✅ Готово.", target)


async def _run_face_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, FACE_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk face_detect")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="👥 Лица на кадре не найдены.",
        )
        await _reply_menu("✅ Готово.", target)
        return

    for i, face in enumerate(faces):
        pict = draw.draw_boxes_on_image(
            image,
            [face],
            highlight_index=0,
            colors=[(255, 60, 60)],
            line_width=3,
        )
        cap = f"👥 Детекция лиц — лицо {i + 1} из {len(faces)}."
        await send_image_bytes(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            image_bytes=pict,
            caption=cap,
            filename=f"face_detect_{i+1}.jpg",
        )
    await _reply_menu("✅ Готово.", target)


async def _run_body_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, BODY_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk body_detect")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    bodies = luna.iter_bodies_from_sdk(j)
    if not bodies:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="🧍 Тела на кадре не найдены.",
        )
        await _reply_menu("✅ Готово.", target)
        return

    for i, body in enumerate(bodies):
        pict = draw.draw_boxes_on_image(
            image,
            [body],
            highlight_index=0,
            colors=[(0, 200, 140)],
            line_width=3,
        )
        cap = f"🧍 Детекция тел — тело {i + 1} из {len(bodies)}."
        await send_image_bytes(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            image_bytes=pict,
            caption=cap,
            filename=f"body_detect_{i+1}.jpg",
        )
    await _reply_menu("✅ Готово.", target)


async def _run_image_modification(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, IMAGE_MODIFICATION_PARAMS)
    except Exception as e:
        log.exception("sdk image_modification")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    text = luna.format_image_modification_ru(j)
    await send_text(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        text=text[:5900],
    )
    await _reply_menu("✅ Готово.", target)


async def _run_crowd_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    ct = _guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, CROWD_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk crowd_detect")
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=f"Ошибка SDK: {e}"[:3000],
        )
        return

    count, points = luna.get_people_estimation(j)
    if count is None and not points:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="🧑‍🤝‍🧑 Детекция толпы: блок people не вернулся.",
        )
        await _reply_menu("✅ Готово.", target)
        return

    out = draw.draw_people_points_on_image(image, points) if points else image
    c = count if count is not None else len(points)
    cap = f"🧑‍🤝‍🧑 Детекция толпы: людей на кадре — {c}."
    if points and (count is None or count != len(points)):
        cap += f" Точек: {len(points)}."
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=out,
        caption=cap,
        filename="crowd_detect.jpg",
    )
    await _reply_menu("✅ Готово.", target)


async def handle_update(update: dict[str, Any]) -> None:
    uid = int(update.get("update_id", 0) or 0)
    if uid and _seen(uid):
        return

    target = _target(update)
    if not target.get("login") and not target.get("chat_id"):
        return

    from_ = update.get("from") or {}
    from_login = from_.get("login")
    if not from_login and from_.get("id") is not None:
        from_login = f"id:{from_.get('id')}"
    sk = session_key(
        (update.get("chat") or {}).get("type"),
        (update.get("chat") or {}).get("id"),
        from_login,
    )
    sess = get_session(sk)

    raw_text = (update.get("text") or "").strip()

    if settings.bot_auth_enabled:
        pr = _principal(update)
        if not pr:
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text="Не удалось сопоставить профиль (нет логина или id в событии). Включите корпоративный аккаунт или обратитесь к администратору.",
            )
            return
        if not access_control.user_has_record(pr):
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=(
                    f"Доступ не настроен для учётной записи `{pr}`.\n\n"
                    "Админ может выдать пароль командой "
                    f"`!issue {pr}` (логин должен совпадать с вашим в мессенджере).\n\n"
                    "Если вы админ: в `BOT_ADMIN_LOGINS` и в JSON должен быть **тот же** логин, "
                    "что выше (как в Яндекс Мессенджере). Проверьте `.env` без пробелов в начале строк, "
                    "`BOT_BOOTSTRAP_ADMIN_PASSWORD`, перезапустите бота — при старте создаются записи "
                    "для всех логинов из `BOT_ADMIN_LOGINS`."
                )[:5900],
            )
            return
        if not sess.authorized:
            act_early, _ = _extract_server_action(update)
            if act_early:
                await send_text(
                    login=target.get("login"),
                    chat_id=target.get("chat_id"),
                    text="Сначала введите пароль доступа обычным текстом (без кнопок меню и без вложений).",
                )
                return
            if _gallery_file_ids(update) or extract_urls(raw_text):
                await send_text(
                    login=target.get("login"),
                    chat_id=target.get("chat_id"),
                    text="Сначала введите пароль одним текстовым сообщением, без фото и без ссылок.",
                )
                return
            if not raw_text:
                return
            if access_control.verify_password(pr, raw_text):
                sess.authorized = True
                await _reply_menu("✅ Вход выполнён.", target)
                return
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text="Неверный пароль.",
            )
            return

    if settings.bot_auth_enabled and sess.authorized:
        pr_a = _principal(update)
        if pr_a:
            low = raw_text.lower()
            if low in ("!logout", "/logout"):
                sess.authorized = False
                sess.flow = Flow.IDLE
                sess.compare_buffers = []
                await send_text(
                    login=target.get("login"),
                    chat_id=target.get("chat_id"),
                    text="Вы вышли. Введите пароль снова для входа.",
                )
                return
            if await _admin_text_commands(target, raw_text, pr_a):
                return

    act, _payload = _extract_server_action(update)
    if act == "flow_compare":
        sess.flow = Flow.WAIT_COMPARE
        sess.compare_buffers = []
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Сравнение 1:1. Отправьте два фото или две ссылки на фото (вместе или по одному).",
        )
        return
    if act == "flow_attributes":
        sess.flow = Flow.WAIT_ATTRIBUTES
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Атрибуты. Отправьте одно фото или ссылку — по каждому лицу отдельная сводка и контур только этого лица.",
        )
        return
    if act == "flow_body_attributes":
        sess.flow = Flow.WAIT_BODY_ATTRIBUTES
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Атрибуты тела. Отправьте одно фото или ссылку — по каждому телу отдельная сводка и контур только этого тела.",
        )
        return
    if act == "flow_liveness":
        sess.flow = Flow.WAIT_LIVENESS
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Liveness. Пришлите одно фото или ссылку (основное лицо на кадре).",
        )
        return
    if act == "flow_deepfake":
        sess.flow = Flow.WAIT_DEEPFAKE
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Дипфейк. Пришлите одно фото или ссылку (основное лицо на кадре).",
        )
        return
    if act == "flow_quality":
        sess.flow = Flow.WAIT_QUALITY
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Качество изображения. Пришлите фото или ссылку (основное лицо на кадре).",
        )
        return
    if act == "flow_face_detect":
        sess.flow = Flow.WAIT_FACE_DETECT
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Детекция лиц. Пришлите фото или ссылку — по каждому лицу отдельное фото с контуром только этого лица.",
        )
        return
    if act == "flow_body_detect":
        sess.flow = Flow.WAIT_BODY_DETECT
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Детекция тел. Пришлите фото или ссылку — по каждому телу отдельное фото с контуром только этого тела.",
        )
        return
    if act == "flow_image_modification":
        sess.flow = Flow.WAIT_IMAGE_MODIFICATION
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Модификация изображения. Пришлите фото или ссылку — оценю признаки обработки/модификации кадра.",
        )
        return
    if act == "flow_crowd_detect":
        sess.flow = Flow.WAIT_CROWD_DETECT
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Детекция толпы. Пришлите фото или ссылку — посчитаю людей и отмечу координаты.",
        )
        return
    if act == "flow_reset":
        sess.flow = Flow.IDLE
        sess.compare_buffers = []
        await _reply_menu("♻️ Состояние сброшено.", target)
        return

    text = raw_text.lower()
    if text in ("/start", "/menu", "меню", "start"):
        sess.flow = Flow.IDLE
        sess.compare_buffers = []
        await _reply_menu(
            "Luna: 1 к 1, атрибуты, лайфнесс, дипфейк. Остальное — отдельной кнопкой.",
            target,
        )
        return
    if text in ("отмена", "/cancel", "сброс"):
        sess.flow = Flow.IDLE
        sess.compare_buffers = []
        await _reply_menu("♻️ Ок, сброс.", target)
        return

    fids = _gallery_file_ids(update)
    parts: list[tuple[bytes, str]] = []
    failed_urls: list[tuple[str, str]] = []

    for fid in fids:
        try:
            data, ct = await download_file(fid)
            parts.append((data, _norm_ct(ct)))
        except Exception as e:
            log.warning("getFile %s: %s", fid, e)

    for url in extract_urls(raw_text):
        try:
            data, ct, _final = await fetch_image_from_url(url)
            parts.append((data, _norm_ct(ct)))
        except Exception as e:
            log.warning("url image %s: %s", url, e)
            failed_urls.append((url, str(e)))

    if not parts:
        if failed_urls:
            details = "\n".join(f"• {u} — {err[:160]}" for u, err in failed_urls[:5])
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=(
                    "🔗 Не удалось скачать фото по ссылкам.\n"
                    "Проверьте, что ссылка публичная и ведет к файлу изображения.\n\n"
                    f"{details}"
                )[:5900],
            )
        return

    if failed_urls:
        details = "\n".join(f"• {u} — {err[:160]}" for u, err in failed_urls[:5])
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text=(
                "⚠️ Некоторые ссылки не удалось скачать, обрабатываю только успешные.\n\n"
                f"{details}"
            )[:5900],
        )

    if sess.flow == Flow.WAIT_COMPARE:
        for data, ct in parts:
            if len(sess.compare_buffers) >= 2:
                break
            sess.compare_buffers.append((data, ct))
        if len(sess.compare_buffers) >= 2:
            await _run_compare(target, sess.compare_buffers[:2])
            sess.flow = Flow.IDLE
            sess.compare_buffers = []
        else:
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Принято фото {len(sess.compare_buffers)}/2. Пришлите ещё {2 - len(sess.compare_buffers)}.",
            )
        return

    if sess.flow == Flow.WAIT_ATTRIBUTES:
        data, ct = parts[0]
        try:
            await _run_attributes(target, data, ct)
        except Exception as e:
            log.exception("attributes")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_BODY_ATTRIBUTES:
        data, ct = parts[0]
        try:
            await _run_body_attributes(target, data, ct)
        except Exception as e:
            log.exception("body_attributes")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_LIVENESS:
        data, ct = parts[0]
        try:
            await _run_liveness(target, data, ct)
        except Exception as e:
            log.exception("liveness")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_DEEPFAKE:
        data, ct = parts[0]
        try:
            await _run_deepfake(target, data, ct)
        except Exception as e:
            log.exception("deepfake")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_QUALITY:
        data, ct = parts[0]
        try:
            await _run_quality(target, data, ct)
        except Exception as e:
            log.exception("quality")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_FACE_DETECT:
        data, ct = parts[0]
        try:
            await _run_face_detect(target, data, ct)
        except Exception as e:
            log.exception("face_detect")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_BODY_DETECT:
        data, ct = parts[0]
        try:
            await _run_body_detect(target, data, ct)
        except Exception as e:
            log.exception("body_detect")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_IMAGE_MODIFICATION:
        data, ct = parts[0]
        try:
            await _run_image_modification(target, data, ct)
        except Exception as e:
            log.exception("image_modification")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if sess.flow == Flow.WAIT_CROWD_DETECT:
        data, ct = parts[0]
        try:
            await _run_crowd_detect(target, data, ct)
        except Exception as e:
            log.exception("crowd_detect")
            await send_text(
                login=target.get("login"),
                chat_id=target.get("chat_id"),
                text=f"Ошибка: {e}"[:3000],
            )
        sess.flow = Flow.IDLE
        return

    if len(parts) == 2 and sess.flow == Flow.IDLE:
        await _run_compare(target, parts[:2])
        return

    if len(parts) == 1 and sess.flow == Flow.IDLE:
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Выберите режим кнопкой под сообщением.",
            suggest_buttons=main_menu_buttons(),
        )
