from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from app.config import settings


def _luna_headers() -> dict[str, str]:
    h: dict[str, str] = {
        "Luna-Request-Id": f"{int(time.time())},{uuid.uuid4()}",
        "Accept": "application/json",
    }
    if settings.luna_bearer_token:
        h["Authorization"] = f"Bearer {settings.luna_bearer_token}"
    return h


async def sdk_analyze(
    image_bytes: bytes,
    content_type: str,
    extra_params: dict[str, int | float | str],
) -> dict[str, Any]:
    base = settings.luna_base_url.rstrip("/")
    url = f"{base}/sdk"
    params: dict[str, Any] = dict(extra_params)
    headers = {**_luna_headers(), "Content-Type": content_type}
    auth = (
        (settings.luna_http_user, settings.luna_http_password)
        if settings.luna_http_user and not settings.luna_bearer_token
        else None
    )
    async with httpx.AsyncClient(auth=auth, timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(
            url,
            content=image_bytes,
            headers=headers,
            params=params,
        )
    r.raise_for_status()
    return r.json()


async def matcher_raw_sdk(
    ref_sdk_b64: str,
    cand_sdk_b64: str,
    ref_id: str = "ref-0",
    cand_id: str = "cand-0",
) -> dict[str, Any]:
    base = settings.luna_base_url.rstrip("/")
    url = f"{base}/matcher/raw"
    body = {
        "references": [
            {"id": ref_id, "type": "sdk_descriptor", "data": ref_sdk_b64},
        ],
        "candidates": [
            {"id": cand_id, "type": "sdk_descriptor", "data": cand_sdk_b64},
        ],
    }
    auth = (
        (settings.luna_http_user, settings.luna_http_password)
        if settings.luna_http_user and not settings.luna_bearer_token
        else None
    )
    async with httpx.AsyncClient(auth=auth, timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            url,
            json=body,
            headers={**_luna_headers(), "Content-Type": "application/json"},
        )
    r.raise_for_status()
    return r.json()


def pick_original_file_id(image_variants: list[dict[str, Any]]) -> str | None:
    """Берём оригинал (обычно последний вариант с name/size)."""
    if not image_variants:
        return None
    best = image_variants[-1]
    fid = best.get("file_id") or ""
    return fid.split("?")[0] if fid else None


def iter_faces_from_sdk(sdk_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Список блоков face из ответа sdk (плоский список по всем estimations)."""
    out: list[dict[str, Any]] = []
    for img in sdk_json.get("images_estimations") or []:
        for est in img.get("estimations") or []:
            face = est.get("face")
            if face:
                out.append(face)
    return out


def get_image_modification_block(sdk_json: dict[str, Any]) -> dict[str, Any] | None:
    """Блок image_modification из images_estimations[].image_estimations."""
    for img in sdk_json.get("images_estimations") or []:
        ie = img.get("image_estimations")
        if not isinstance(ie, dict):
            continue
        im = ie.get("image_modification")
        if isinstance(im, dict):
            return im
    return None


def format_image_modification_ru(sdk_json: dict[str, Any]) -> str:
    im = get_image_modification_block(sdk_json)
    if not im:
        return (
            "Модификация изображения: в ответе SDK нет блока image_modification "
            "(проверьте лицензию/параметр estimate_image_modification)."
        )
    score = im.get("score")
    status = im.get("status")
    lines: list[str] = ["Модификация изображения (Luna SDK)"]
    if score is not None:
        lines.append(f"• score: {score}")
    if status is not None:
        status_hint = {
            0: "не изменено",
            1: "изменено",
        }.get(status, "неизвестное значение")
        lines.append(f"• status: {status} — {status_hint} (enum: 0=не изменено, 1=изменено)")
    lines.append("")
    lines.append(json.dumps(im, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def get_people_estimation(sdk_json: dict[str, Any]) -> tuple[int | None, list[tuple[int, int]]]:
    """Возвращает count и координаты людей из images_estimations[].image_estimations.people."""
    for img in sdk_json.get("images_estimations") or []:
        ie = img.get("image_estimations")
        if not isinstance(ie, dict):
            continue
        people = ie.get("people")
        if not isinstance(people, dict):
            continue
        count_raw = people.get("count")
        count = int(count_raw) if isinstance(count_raw, (int, float)) else None
        coords: list[tuple[int, int]] = []
        for p in people.get("coordinates") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    coords.append((int(p[0]), int(p[1])))
                except (TypeError, ValueError):
                    continue
        return count, coords
    return None, []


def get_quality_block(sdk_json: dict[str, Any]) -> dict[str, Any] | None:
    """Ищет quality в face-блоке (включая face.detection.quality), затем fallback."""
    for img in sdk_json.get("images_estimations") or []:
        for est in img.get("estimations") or []:
            face = est.get("face") or {}
            # Чаще всего quality внутри face.attributes / face.detection.attributes.
            for cand in (
                (face.get("detection") or {}).get("quality"),
                (face.get("attributes") or {}).get("quality"),
                ((face.get("detection") or {}).get("attributes") or {}).get("quality"),
            ):
                if isinstance(cand, dict) and cand:
                    return cand
            q = est.get("quality")
            if isinstance(q, dict) and q:
                return q
    return None


def format_quality_ru(sdk_json: dict[str, Any]) -> str:
    q = get_quality_block(sdk_json)
    if not q:
        return "🖼️ Качество: в ответе SDK нет блока quality."
    lines: list[str] = ["🖼️ Качество изображения (0..1, где 1 — лучше)"]
    for k in ("blurriness", "dark", "illumination", "specularity", "light"):
        if k in q:
            lines.append(f"• {k}: {q[k]}")
    return "\n".join(lines)


def iter_bodies_from_sdk(sdk_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Список блоков body из ответа sdk (все детекции тел при multiface_policy=1)."""
    out: list[dict[str, Any]] = []
    for img in sdk_json.get("images_estimations") or []:
        for est in img.get("estimations") or []:
            body = est.get("body")
            if body:
                out.append(body)
    return out


def _face_attrs(face: dict[str, Any]) -> dict[str, Any]:
    """В OpenAPI-ответе атрибуты под face.attributes; на части инсталляций — под face.detection.attributes."""
    top = face.get("attributes")
    if isinstance(top, dict) and top:
        return top
    nested = (face.get("detection") or {}).get("attributes")
    if isinstance(nested, dict) and nested:
        return nested
    return {}


def get_descriptor_b64(face: dict[str, Any]) -> str | None:
    desc = _face_attrs(face).get("descriptor") or {}
    raw = desc.get("sdk_descriptor")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _without_iris_landmarks(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _without_iris_landmarks(v)
            for k, v in obj.items()
            if k != "iris_landmarks"
        }
    if isinstance(obj, list):
        return [_without_iris_landmarks(x) for x in obj]
    return obj


_VALUE_HINTS: dict[str, dict[str, str]] = {
    "glasses": {
        "glasses": "в очках",
        "no_glasses": "без очков",
        "sunglasses": "солнцезащитные очки",
    },
    "state": {
        "open": "открыто",
        "closed": "закрыто",
    },
    "predominant_mask": {
        "medical_mask": "медицинская маска",
        "missing": "маска отсутствует",
        "occluded": "окклюзия/перекрытие",
    },
    "predominant_occlusion": {
        "full": "полное перекрытие",
        "clear": "без перекрытия",
        "correct": "корректное ношение",
        "partially": "частичное перекрытие",
        "mouth": "перекрыт рот",
        "chin": "перекрыт подбородок",
    },
    "prediction": {
        "real": "реальный",
        "spoof": "спуф/подмена",
        "fake": "дипфейк/подделка",
    },
}


def _annotate_estimations(obj: Any) -> Any:
    """Добавляет к известным enum-полям текстовые подсказки."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            vv = _annotate_estimations(v)
            if isinstance(v, str):
                hint = _VALUE_HINTS.get(k, {}).get(v.lower())
                if hint:
                    out[k] = f"{v} ({hint})"
                    continue
            out[k] = vv
        return out
    if isinstance(obj, list):
        return [_annotate_estimations(x) for x in obj]
    return obj


def format_liveness(face: dict[str, Any]) -> str:
    liv = _face_attrs(face).get("liveness")
    if not liv:
        return "🫀 Liveness: в ответе SDK нет блока liveness."
    pred = liv.get("prediction")
    est = liv.get("estimations")
    parts = []
    if pred is not None:
        pred_hint = {
            "real": "реальный человек",
            "spoof": "подозрение на спуф/подмену",
        }.get(str(pred).lower(), "см. документацию модели")
        parts.append(f"🫀 prediction: {pred} — {pred_hint}")
    if est is not None:
        parts.append(f"📊 estimations:\n{json.dumps(est, ensure_ascii=False, indent=2)}")
    return "\n".join(parts) if parts else json.dumps(liv, ensure_ascii=False, indent=2)


def format_deepfake(face: dict[str, Any]) -> str:
    df = _face_attrs(face).get("deepfake")
    if not df:
        return "🎭 Deepfake: в ответе SDK нет блока deepfake."
    pred = df.get("prediction")
    score = df.get("score")
    lines = []
    if pred is not None:
        pred_hint = {
            "real": "признаков дипфейка не обнаружено",
            "fake": "обнаружены признаки дипфейка",
        }.get(str(pred).lower(), "см. документацию модели")
        lines.append(f"🎭 prediction: {pred} — {pred_hint}")
    if score is not None:
        lines.append(f"📊 score: {score}")
    return "\n".join(lines) if lines else json.dumps(df, ensure_ascii=False, indent=2)


def _face_bbox_size(face: dict[str, Any]) -> dict[str, int] | None:
    """Геометрия лица из detection.rect (размер области детекции в пикселях относительно кадра SDK)."""
    det = face.get("detection") or {}
    r = det.get("rect")
    if not isinstance(r, dict):
        return None
    try:
        w = int(r["width"])
        h = int(r["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"width_px": w, "height_px": h, "area_px2": w * h}


def format_face_attributes(face: dict[str, Any]) -> str:
    attrs = _annotate_estimations(_without_iris_landmarks(_face_attrs(face)))
    parts: dict[str, Any] = {}
    size = _face_bbox_size(face)
    if size is not None:
        parts["📐 размер лица (bbox)"] = size
    parts.update(
        {
        "👄 mouth_attributes": attrs.get("mouth_attributes"),
        "👀 eyes_attributes": attrs.get("eyes_attributes"),
        "🙂 emotions": attrs.get("emotions"),
        "😷 mask": attrs.get("mask"),
        "🧭 head_pose": attrs.get("head_pose"),
        "👓 glasses": attrs.get("glasses"),
        "🪪 basic_attributes": attrs.get("basic_attributes"),
        "🎭 deepfake": attrs.get("deepfake"),
        })
    occ = attrs.get("face_occlusion")
    if occ is not None:
        parts["🚫 face_occlusion"] = occ
    else:
        m = attrs.get("mask")
        if isinstance(m, dict) and "face_occlusion" in m:
            parts["🚫 face_occlusion"] = m.get("face_occlusion")
    lines = []
    for k, v in parts.items():
        if v is not None:
            lines.append(f"{k}:\n{json.dumps(v, ensure_ascii=False, indent=2)}")
    return "\n\n".join(lines) if lines else json.dumps(attrs, ensure_ascii=False, indent=2)


def format_body_attributes(body: dict[str, Any]) -> str:
    det = body.get("detection") or {}
    attrs = _annotate_estimations(_without_iris_landmarks((det.get("attributes") or {})))
    parts = {
        "🧬 descriptor": (attrs.get("descriptor") or {}).get("score"),
        "🪪 basic_attributes": attrs.get("basic_attributes"),
        "👕 upper_body": attrs.get("upper_body"),
        "👖 lower_body": attrs.get("lower_body"),
        "🎒 accessories": attrs.get("accessories"),
    }
    lines = []
    for k, v in parts.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            lines.append(f"{k}:\n{json.dumps(v, ensure_ascii=False, indent=2)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n\n".join(lines) if lines else "🧍 Атрибуты тела не вернулись в ответе SDK."
