from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw


def _rect_from_detection(face: dict[str, Any]) -> tuple[int, int, int, int] | None:
    det = face.get("detection") or {}
    r = det.get("rect")
    if not r:
        return None
    try:
        x, y = int(r["x"]), int(r["y"])
        w, h = int(r["width"]), int(r["height"])
        return x, y, x + w, y + h
    except (KeyError, TypeError, ValueError):
        return None


def draw_boxes_on_image(
    image_bytes: bytes,
    faces: list[dict[str, Any]],
    highlight_index: int | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    line_width: int = 3,
    highlight_width: int = 5,
) -> bytes:
    """Рисует прямоугольники по detection.rect (лица или тела); highlight_index — более толстая рамка."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    default_colors = colors or [(255, 0, 0), (0, 180, 0), (0, 128, 255), (255, 128, 0)]
    for i, item in enumerate(faces):
        box = _rect_from_detection(item)
        if not box:
            continue
        color = default_colors[i % len(default_colors)]
        w = highlight_width if highlight_index is not None and i == highlight_index else line_width
        for offset in range(w):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=color,
            )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def draw_people_points_on_image(
    image_bytes: bytes,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 80, 0),
    radius: int = 8,
) -> bytes:
    """Рисует точки (центры людей) для crowd/people_count."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
