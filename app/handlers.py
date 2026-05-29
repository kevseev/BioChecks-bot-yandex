from __future__ import annotations

import logging
from collections import deque
from typing import Any

from app import access_control
from app import pipeline
from app.config import settings
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
    from app import luna as luna_mod

    out: list[str] = []
    for group in update.get("images") or []:
        if not group:
            continue
        fid = luna_mod.pick_original_file_id(group)
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


async def _emit_bundle(target: dict[str, str | None], bundle: pipeline.BotBundle) -> None:
    login, chat_id = target.get("login"), target.get("chat_id")
    for p in bundle.parts:
        if isinstance(p, pipeline.PipeText):
            await send_text(login=login, chat_id=chat_id, text=p.text)
        else:
            await send_image_bytes(
                login=login,
                chat_id=chat_id,
                image_bytes=p.data,
                caption=p.caption,
                filename=p.filename,
            )
    if bundle.menu_message:
        await _reply_menu(bundle.menu_message, target)


async def _run_compare(
    target: dict[str, str | None],
    images: list[tuple[bytes, str]],
) -> None:
    bundle = await pipeline.run_compare(images)
    await _emit_bundle(target, bundle)


async def _run_attributes(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_attributes(image, content_type))


async def _run_body_attributes(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_body_attributes(image, content_type))


async def _run_liveness(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_liveness(image, content_type))


async def _run_deepfake(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_deepfake(image, content_type))


async def _run_quality(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_quality(image, content_type))


async def _run_face_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_face_detect(image, content_type))


async def _run_body_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_body_detect(image, content_type))


async def _run_image_modification(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_image_modification(image, content_type))


async def _run_crowd_detect(
    target: dict[str, str | None], image: bytes, content_type: str = "image/jpeg"
) -> None:
    await _emit_bundle(target, await pipeline.run_crowd_detect(image, content_type))


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
