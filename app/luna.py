from __future__ import annotations

import json
import math
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


async def check_iso(
    image_bytes: bytes,
    content_type: str,
    *,
    multiface_policy: int = 2,
) -> dict[str, Any]:
    """POST /iso — операция checkISO (ISO/IEC 19794-5).

    Документация: https://docs.visionlabs.ru/luna/v.5.152.0/ReferenceManuals/APIReferenceManual.html#tag/iso/operation/checkISO
    Параметр multiface_policy=2 — несколько лиц на кадре; расстояние между глазами — элемент проверки
    ``eye_distance`` (поле ``object_value``, пиксели) в ``face.detection.iso.checks``.
    """
    base = settings.luna_base_url.rstrip("/")
    url = f"{base}/iso"
    params: dict[str, Any] = {"multiface_policy": multiface_policy}
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


def _eyes_attributes_compact_for_caption(raw: dict[str, Any]) -> dict[str, Any]:
    """Только состояние глаз со скобкой-переводом, порядок: правый, левый (как в выдаче операторов)."""
    hints = _VALUE_HINTS.get("state", {})
    out: dict[str, Any] = {}
    for side in ("right_eye", "left_eye"):
        block = raw.get(side)
        if not isinstance(block, dict):
            continue
        st = block.get("state")
        if isinstance(st, str):
            hint = hints.get(st.lower())
            label = f"{st} ({hint})" if hint else st
            out[side] = {"state": label}
        elif st is not None:
            out[side] = {"state": st}
    return out


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


def _parse_xy(p: Any) -> tuple[float, float] | None:
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return float(p[0]), float(p[1])
        except (TypeError, ValueError):
            return None
    if isinstance(p, dict):
        for a, b in (("x", "y"), ("X", "Y")):
            if a in p and b in p:
                try:
                    return float(p[a]), float(p[b])
                except (TypeError, ValueError):
                    continue
    return None


def _euclid_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _iso_images_list(iso_json: dict[str, Any]) -> list[dict[str, Any]]:
    imgs = iso_json.get("images")
    if isinstance(imgs, list) and imgs:
        return imgs
    alt = iso_json.get("images_estimations")
    return alt if isinstance(alt, list) else []


def eye_distance_px_from_check_iso(
    iso_json: dict[str, Any],
    *,
    image_index: int = 0,
    face_index: int = 0,
) -> float | None:
    """Расстояние между глазами из ответа checkISO: ``checks[]`` с ``name`` = ``eye_distance`` → ``object_value`` (px).

    Ожидаемая ветка (v.5.152.0): ``images[i].estimations[j].face.detection.iso.checks``.
    """
    imgs = _iso_images_list(iso_json)
    if image_index >= len(imgs):
        return None
    ests = imgs[image_index].get("estimations") if isinstance(imgs[image_index], dict) else None
    if not isinstance(ests, list) or face_index >= len(ests):
        return None
    est = ests[face_index]
    if not isinstance(est, dict):
        return None

    face = est.get("face")
    if isinstance(face, dict):
        det = face.get("detection")
        if isinstance(det, dict):
            iso = det.get("iso")
            if isinstance(iso, dict):
                checks = iso.get("checks")
                if isinstance(checks, list):
                    for ch in checks:
                        if not isinstance(ch, dict):
                            continue
                        if ch.get("name") != "eye_distance":
                            continue
                        ov = ch.get("object_value")
                        if isinstance(ov, (int, float)) and not isinstance(ov, bool) and float(ov) > 0:
                            return float(ov)

    def _walk_find_eye_distance(obj: Any, depth: int = 0) -> float | None:
        if depth > 14:
            return None
        if isinstance(obj, dict):
            if obj.get("name") == "eye_distance":
                ov = obj.get("object_value")
                if isinstance(ov, (int, float)) and not isinstance(ov, bool) and float(ov) > 0:
                    return float(ov)
            for v in obj.values():
                d = _walk_find_eye_distance(v, depth + 1)
                if d is not None:
                    return d
        elif isinstance(obj, list):
            for x in obj:
                d = _walk_find_eye_distance(x, depth + 1)
                if d is not None:
                    return d
        return None

    return _walk_find_eye_distance(est)


def _inter_eye_from_iris_landmarks(raw_attrs: dict[str, Any]) -> float | None:
    eyes = raw_attrs.get("eyes_attributes")
    if not isinstance(eyes, dict):
        eyes = {}
    ilm = eyes.get("iris_landmarks")
    if ilm is None:
        ilm = raw_attrs.get("iris_landmarks")
    if isinstance(ilm, dict):
        for lk, rk in (
            ("left", "right"),
            ("left_iris", "right_iris"),
            ("left_eye", "right_eye"),
        ):
            l, r = _parse_xy(ilm.get(lk)), _parse_xy(ilm.get(rk))
            if l and r:
                return _euclid_px(l, r)
    if isinstance(ilm, list) and len(ilm) >= 2:
        l, r = _parse_xy(ilm[0]), _parse_xy(ilm[1])
        if l and r:
            return _euclid_px(l, r)
    return None


def _centroid_points(lm: list[Any], indices: range) -> tuple[float, float] | None:
    pts: list[tuple[float, float]] = []
    for i in indices:
        if i >= len(lm):
            continue
        p = _parse_xy(lm[i])
        if p:
            pts.append(p)
    if not pts:
        return None
    sx = sum(p[0] for p in pts) / len(pts)
    sy = sum(p[1] for p in pts) / len(pts)
    return sx, sy


def _inter_eye_from_face_landmarks(face: dict[str, Any]) -> float | None:
    det = face.get("detection") or {}
    lm = det.get("landmarks") or det.get("points")
    if not isinstance(lm, list) or len(lm) < 4:
        return None
    # Раскладка как у 68-point: глаза — точки 36–41 и 42–47.
    if len(lm) >= 48:
        left = _centroid_points(lm, range(36, 42))
        right = _centroid_points(lm, range(42, 48))
        if left and right:
            return _euclid_px(left, right)
    if len(lm) == 2:
        a, b = _parse_xy(lm[0]), _parse_xy(lm[1])
        if a and b:
            return _euclid_px(a, b)
    return None


def _inter_eye_distance_info(
    face: dict[str, Any],
    *,
    iso_json: dict[str, Any] | None = None,
    face_index: int = 0,
) -> tuple[float, str] | None:
    """Расстояние между центрами глаз в пикселях кадра и источник значения."""
    if iso_json is not None:
        d_iso = eye_distance_px_from_check_iso(iso_json, image_index=0, face_index=face_index)
        if d_iso is not None:
            return d_iso, "iso"

    raw = _face_attrs(face)
    eyes = raw.get("eyes_attributes") if isinstance(raw.get("eyes_attributes"), dict) else {}

    def try_numeric(container: dict[str, Any]) -> float | None:
        for key in (
            "interocular_distance",
            "inter_eye_distance",
            "eye_distance",
            "iod",
            "interocular",
            "pupillary_distance",
        ):
            v = container.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        return None

    if isinstance(eyes, dict):
        d = try_numeric(eyes)
        if d is not None:
            return d, "sdk"
        for nested in (eyes.get("geometry"), eyes.get("measurements"), eyes.get("sizes")):
            if isinstance(nested, dict):
                d = try_numeric(nested)
                if d is not None:
                    return d, "sdk"

    if isinstance(raw, dict):
        d = try_numeric(raw)
        if d is not None:
            return d, "sdk"

    d = _inter_eye_from_iris_landmarks(raw)
    if d is not None:
        return d, "iris"

    d = _inter_eye_from_face_landmarks(face)
    if d is not None:
        return d, "landmarks"

    return None


def format_face_attributes(
    face: dict[str, Any],
    *,
    iso_json: dict[str, Any] | None = None,
    face_index: int = 0,
) -> str:
    attrs = _annotate_estimations(_without_iris_landmarks(_face_attrs(face)))
    parts: dict[str, Any] = {}
    size = _face_bbox_size(face)
    if size is not None:
        parts["📐 размер лица (bbox)"] = size
    iei = _inter_eye_distance_info(face, iso_json=iso_json, face_index=face_index)
    if iei is not None:
        dist, src = iei
        hint = {
            "iso": "checkISO, проверка eye_distance (object_value), docs v.5.152.0",
            "sdk": "оценка SDK (/sdk)",
            "iris": "по iris_landmarks",
            "landmarks": "по меткам лица (68-point)",
        }[src]
        parts["📏 расстояние между глазами"] = {"distance_px": float(round(dist, 2)), "источник": hint}
    eyes_raw = attrs.get("eyes_attributes")
    eyes_out: Any
    if isinstance(eyes_raw, dict):
        compact = _eyes_attributes_compact_for_caption(eyes_raw)
        eyes_out = compact if compact else _annotate_estimations(eyes_raw)
    else:
        eyes_out = eyes_raw
    parts.update(
        {
        "👄 mouth_attributes": attrs.get("mouth_attributes"),
        "👀 eyes_attributes": eyes_out,
        "🙂 emotions": attrs.get("emotions"),
        "😷 mask": attrs.get("mask"),
        "🧭 head_pose": attrs.get("head_pose"),
        "👓 glasses": attrs.get("glasses"),
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
