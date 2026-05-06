from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Flow(Enum):
    IDLE = "idle"
    WAIT_COMPARE = "wait_compare"
    WAIT_ATTRIBUTES = "wait_attributes"
    WAIT_LIVENESS = "wait_liveness"
    WAIT_DEEPFAKE = "wait_deepfake"
    WAIT_FACE_DETECT = "wait_face_detect"
    WAIT_BODY_DETECT = "wait_body_detect"
    WAIT_IMAGE_MODIFICATION = "wait_image_modification"
    WAIT_CROWD_DETECT = "wait_crowd_detect"
    WAIT_QUALITY = "wait_quality"
    WAIT_BODY_ATTRIBUTES = "wait_body_attributes"


@dataclass
class Session:
    flow: Flow = Flow.IDLE
    compare_buffers: list[tuple[bytes, str]] = field(default_factory=list)


_sessions: dict[str, Session] = {}


def session_key(chat_type: str | None, chat_id: str | None, user_login: str | None) -> str:
    uid = user_login or "unknown"
    if chat_type == "private" or not chat_id:
        return f"p:{uid}"
    return f"g:{chat_id}:{uid}"


def get_session(sk: str) -> Session:
    if sk not in _sessions:
        _sessions[sk] = Session()
    return _sessions[sk]
