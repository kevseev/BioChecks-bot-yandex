"""Общая логика вызовов LUNA и подготовки ответов для бота и веб-интерфейса."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app import draw, luna

log = logging.getLogger(__name__)

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
    "estimate_basic_attributes": 1,
    "estimate_face_occlusion": 1,
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


@dataclass
class PipeText:
    text: str


@dataclass
class PipeImage:
    data: bytes
    filename: str
    caption: str | None = None


PipePart = PipeText | PipeImage


@dataclass
class BotBundle:
    """Результат для бота: фрагменты ответа и опционально финальное меню (_reply_menu)."""

    parts: list[PipePart]
    menu_message: str | None = None


def guess_image_content_type(data: bytes, reported: str) -> str:
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


def similarity_percent(match_json: dict[str, Any]) -> float | None:
    try:
        m = match_json["matches"][0]["matches"][0]["similarity"]
        return float(m) * 100.0
    except (KeyError, IndexError, TypeError, ValueError):
        return None


async def run_compare(images: list[tuple[bytes, str]]) -> BotBundle:
    if len(images) < 2:
        return BotBundle([], "Нужно два снимка.")
    out: list[PipePart] = []
    (a, cta), (b, ctb) = images[0], images[1]
    cta = guess_image_content_type(a, cta)
    ctb = guess_image_content_type(b, ctb)
    try:
        ja = await luna.sdk_analyze(a, cta, COMPARE_PARAMS)
        jb = await luna.sdk_analyze(b, ctb, COMPARE_PARAMS)
    except Exception as e:
        log.exception("sdk compare")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    fa = luna.iter_faces_from_sdk(ja)
    fb = luna.iter_faces_from_sdk(jb)
    if not fa or not fb:
        out.append(PipeText("На одном из снимков лицо не найдено. Пришлите другое фото."))
        return BotBundle(out)

    d0 = luna.get_descriptor_b64(fa[0])
    d1 = luna.get_descriptor_b64(fb[0])
    if not d0 or not d1:
        out.append(PipeText("Не удалось извлечь дескриптор (face descriptor)."))
        return BotBundle(out)

    try:
        mraw = await luna.matcher_raw_sdk(d0, d1)
    except Exception as e:
        log.exception("matcher")
        out.append(PipeText(f"Ошибка matcher/raw: {e}"[:3000]))
        return BotBundle(out)

    pct = similarity_percent(mraw)
    out_a = draw.draw_boxes_on_image(a, [fa[0]], highlight_index=0, colors=[(255, 0, 0)])
    out_b = draw.draw_boxes_on_image(b, [fb[0]], highlight_index=0, colors=[(0, 180, 0)])
    sim_text = (
        f"🧑‍🤝‍🧑 Схожесть: {pct:.2f}%"
        if pct is not None
        else "🧑‍🤝‍🧑 Схожесть: (нет в ответе matcher)"
    )
    out.append(PipeText(sim_text))
    out.append(PipeImage(out_a, "face_a.jpg"))
    out.append(PipeImage(out_b, "face_b.jpg"))
    return BotBundle(out, "✅ Готово. Можно выбрать следующее действие.")


async def run_attributes(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j, iso_exc = await asyncio.gather(
            luna.sdk_analyze(image, ct, ATTR_PARAMS),
            luna.check_iso(image, ct, multiface_policy=2),
            return_exceptions=True,
        )
    except Exception as e:
        log.exception("sdk attr")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    if isinstance(j, Exception):
        log.exception("sdk attr", exc_info=j)
        out.append(PipeText(f"Ошибка SDK: {j}"[:3000]))
        return BotBundle(out)

    iso_payload: dict[str, Any] | None = None
    if isinstance(iso_exc, Exception):
        log.warning("check_iso: %s", iso_exc)
    else:
        iso_payload = iso_exc

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        out.append(PipeText("👤 Лица не обнаружены."))
        return BotBundle(out)

    for i, face in enumerate(faces):
        pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(64, 64, 255)])
        cap = f"👤 Лицо {i + 1} / {len(faces)}\n\n" + luna.format_face_attributes(
            face, iso_json=iso_payload, face_index=i
        )
        out.append(PipeImage(pict, f"face_{i+1}.jpg", caption=cap))
    return BotBundle(out, "✅ Атрибуты отправлены.")


async def run_body_attributes(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, BODY_ATTR_PARAMS)
    except Exception as e:
        log.exception("sdk body attrs")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    bodies = luna.iter_bodies_from_sdk(j)
    if not bodies:
        out.append(PipeText("🧍 Тела не обнаружены."))
        return BotBundle(out)

    for i, body in enumerate(bodies):
        pict = draw.draw_boxes_on_image(image, [body], highlight_index=0, colors=[(0, 200, 140)])
        cap = f"🧍 Тело {i + 1} / {len(bodies)}\n\n" + luna.format_body_attributes(body)
        out.append(PipeImage(pict, f"body_{i+1}.jpg", caption=cap))
    return BotBundle(out, "✅ Атрибуты тела отправлены.")


async def run_liveness(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, LIVENESS_PARAMS)
    except Exception as e:
        log.exception("sdk liveness")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        out.append(PipeText("👤 Лицо не обнаружено — нужен снимок с лицом."))
        return BotBundle(out)

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(200, 100, 0)])
    cap = "🫀 Liveness (лучшее лицо на кадре)\n\n" + luna.format_liveness(face)
    out.append(PipeImage(pict, "liveness.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")


async def run_deepfake(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, DEEPFAKE_PARAMS)
    except Exception as e:
        log.exception("sdk deepfake")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        out.append(PipeText("👤 Лицо не обнаружено — нужен снимок с лицом."))
        return BotBundle(out)

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(160, 60, 200)])
    cap = "🎭 Дипфейк (лучшее лицо на кадре)\n\n" + luna.format_deepfake(face)
    out.append(PipeImage(pict, "deepfake.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")


async def run_quality(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, QUALITY_PARAMS)
    except Exception as e:
        log.exception("sdk quality")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        out.append(PipeText("👤 Лицо не обнаружено — нужен снимок с лицом."))
        return BotBundle(out)

    face = faces[0]
    pict = draw.draw_boxes_on_image(image, [face], highlight_index=0, colors=[(80, 160, 255)])
    cap = luna.format_quality_ru(j)
    out.append(PipeImage(pict, "quality.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")


async def run_face_detect(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, FACE_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk face_detect")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    faces = luna.iter_faces_from_sdk(j)
    if not faces:
        out.append(PipeText("👥 Лица на кадре не найдены."))
        return BotBundle(out, "✅ Готово.")

    for i, face in enumerate(faces):
        pict = draw.draw_boxes_on_image(
            image, [face], highlight_index=0, colors=[(255, 60, 60)], line_width=3
        )
        cap = f"👥 Детекция лиц — лицо {i + 1} из {len(faces)}."
        out.append(PipeImage(pict, f"face_detect_{i+1}.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")


async def run_body_detect(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, BODY_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk body_detect")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    bodies = luna.iter_bodies_from_sdk(j)
    if not bodies:
        out.append(PipeText("🧍 Тела на кадре не найдены."))
        return BotBundle(out, "✅ Готово.")

    for i, body in enumerate(bodies):
        pict = draw.draw_boxes_on_image(
            image, [body], highlight_index=0, colors=[(0, 200, 140)], line_width=3
        )
        cap = f"🧍 Детекция тел — тело {i + 1} из {len(bodies)}."
        out.append(PipeImage(pict, f"body_detect_{i+1}.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")


async def run_image_modification(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, IMAGE_MODIFICATION_PARAMS)
    except Exception as e:
        log.exception("sdk image_modification")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    text = luna.format_image_modification_ru(j)
    out.append(PipeText(text[:5900]))
    return BotBundle(out, "✅ Готово.")


async def run_crowd_detect(image: bytes, content_type: str = "image/jpeg") -> BotBundle:
    out: list[PipePart] = []
    ct = guess_image_content_type(image, content_type)
    try:
        j = await luna.sdk_analyze(image, ct, CROWD_DETECT_PARAMS)
    except Exception as e:
        log.exception("sdk crowd_detect")
        out.append(PipeText(f"Ошибка SDK: {e}"[:3000]))
        return BotBundle(out)

    count, points = luna.get_people_estimation(j)
    if count is None and not points:
        out.append(PipeText("🧑‍🤝‍🧑 Детекция толпы: блок people не вернулся."))
        return BotBundle(out, "✅ Готово.")

    out_img = draw.draw_people_points_on_image(image, points) if points else image
    c = count if count is not None else len(points)
    cap = f"🧑‍🤝‍🧑 Детекция толпы: людей на кадре — {c}."
    if points and (count is None or count != len(points)):
        cap += f" Точек: {len(points)}."
    out.append(PipeImage(out_img, "crowd_detect.jpg", caption=cap))
    return BotBundle(out, "✅ Готово.")
