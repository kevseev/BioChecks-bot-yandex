from __future__ import annotations

import logging
from collections import deque
from typing import Any

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
        j = await luna.sdk_analyze(image, ct, ATTR_PARAMS)
    except Exception as e:
        log.exception("sdk attr")
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
            text="👤 Лица не обнаружены.",
        )
        return

    for i, face in enumerate(faces):
        pict = draw.draw_boxes_on_image(
            image, faces, highlight_index=i, colors=[(64, 64, 255)] * len(faces)
        )
        cap = f"👤 Лицо {i + 1} / {len(faces)}\n\n" + luna.format_face_attributes(face)
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
            image, bodies, highlight_index=i, colors=[(0, 200, 140)] * len(bodies)
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

    pict = draw.draw_boxes_on_image(
        image,
        faces,
        highlight_index=None,
        colors=[(255, 60, 60), (60, 180, 255), (80, 220, 120), (255, 180, 40), (200, 80, 255)],
        line_width=3,
    )
    cap = f"👥 Детекция лиц: найдено {len(faces)}."
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=pict,
        caption=cap,
        filename="faces_detect.jpg",
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

    pict = draw.draw_boxes_on_image(
        image,
        bodies,
        highlight_index=None,
        colors=[(0, 200, 140), (40, 140, 255), (200, 200, 60), (180, 100, 255), (255, 120, 80)],
        line_width=3,
    )
    cap = f"🧍 Детекция тел: найдено {len(bodies)}."
    await send_image_bytes(
        login=target.get("login"),
        chat_id=target.get("chat_id"),
        image_bytes=pict,
        caption=cap,
        filename="bodies_detect.jpg",
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
    sk = session_key(
        (update.get("chat") or {}).get("type"),
        (update.get("chat") or {}).get("id"),
        from_.get("login"),
    )
    sess = get_session(sk)

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
            text="Атрибуты. Отправьте одно фото или ссылку — обработаю все найденные лица.",
        )
        return
    if act == "flow_body_attributes":
        sess.flow = Flow.WAIT_BODY_ATTRIBUTES
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Атрибуты тела. Отправьте одно фото или ссылку — обработаю все найденные тела.",
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
            text="Детекция лиц. Пришлите фото или ссылку — отмечу все найденные лица.",
        )
        return
    if act == "flow_body_detect":
        sess.flow = Flow.WAIT_BODY_DETECT
        await send_text(
            login=target.get("login"),
            chat_id=target.get("chat_id"),
            text="Детекция тел. Пришлите фото или ссылку — отмечу все найденные тела.",
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

    raw_text = (update.get("text") or "").strip()
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
