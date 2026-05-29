from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

_ISSUE_RE = re.compile(r"^[!/]issue\s+(\S+)", re.IGNORECASE)
_REVOKE_RE = re.compile(r"^[!/]revoke\s+(\S+)", re.IGNORECASE)


def normalize_login(s: str) -> str:
    return _norm_login(s)


def _norm_login(s: str) -> str:
    return (s or "").strip().lower()


def store_path() -> Path:
    return Path(settings.bot_access_store_path)


def _atomic_write_json(path: Path, data: Any) -> None:
    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"{path} — это каталог (Docker так создал, если не было файла на хосте). "
            "Удалите эту папку на хосте, создайте JSON-файл: "
            "mkdir -p access_auth && "
            "cp access_auth/access_users.example.json access_auth/access_users.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_store() -> dict[str, Any]:
    path = store_path()
    if not path.is_file():
        return {"users": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("access store read failed %s: %s", path, e)
        return {"users": {}}
    users = raw.get("users")
    if not isinstance(users, dict):
        users = {}
    return {"users": users}


def _save_users(users: dict[str, Any]) -> None:
    _atomic_write_json(store_path(), {"users": users})


def _hash_pw(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        settings.bot_password_pbkdf2_iterations,
    )
    return dk.hex()


def set_user_password(login: str, plain_password: str) -> None:
    login_n = _norm_login(login)
    if not login_n:
        raise ValueError("empty login")
    salt = secrets.token_bytes(16)
    salt_hex = salt.hex()
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt,
        settings.bot_password_pbkdf2_iterations,
    )
    users = _load_store()["users"]
    users[login_n] = {"salt": salt_hex, "hash": dk.hex()}
    _save_users(users)


def verify_password(login: str, plain_password: str) -> bool:
    login_n = _norm_login(login)
    rec = _load_store()["users"].get(login_n)
    if not isinstance(rec, dict):
        return False
    try:
        expected = rec.get("hash")
        salt = rec.get("salt")
        if not isinstance(expected, str) or not isinstance(salt, str):
            return False
        got = _hash_pw(plain_password, salt)
    except (ValueError, KeyError):
        return False
    return secrets.compare_digest(got, expected)


def delete_user(login: str) -> bool:
    login_n = _norm_login(login)
    users = _load_store()["users"]
    if login_n not in users:
        return False
    del users[login_n]
    _save_users(users)
    return True


def list_user_logins() -> list[str]:
    return sorted(_load_store()["users"].keys())


def user_has_record(login: str) -> bool:
    return _norm_login(login) in _load_store()["users"]


def admin_logins() -> set[str]:
    return {
        _norm_login(x)
        for x in (settings.bot_admin_logins or "").replace(";", ",").split(",")
        if _norm_login(x)
    }


def ensure_bootstrap() -> None:
    """Для каждого логина из BOT_ADMIN_LOGINS (+ BOT_BOOTSTRAP_ADMIN_LOGIN), которого нет в JSON,
    создаётся запись с паролем BOT_BOOTSTRAP_ADMIN_PASSWORD.
    """
    if not settings.bot_auth_enabled:
        return
    pw = (settings.bot_bootstrap_admin_password or "").strip()
    if not pw:
        log.warning(
            "auth bootstrap skipped: задайте BOT_BOOTSTRAP_ADMIN_PASSWORD и непустой BOT_ADMIN_LOGINS "
            "(или BOT_BOOTSTRAP_ADMIN_LOGIN)"
        )
        return

    candidates: set[str] = set(admin_logins())
    extra = _norm_login(settings.bot_bootstrap_admin_login)
    if extra:
        candidates.add(extra)
    if not candidates:
        log.warning(
            "auth bootstrap skipped: BOT_ADMIN_LOGINS и BOT_BOOTSTRAP_ADMIN_LOGIN пусты — "
            "укажите тот же логин, что в Яндекс Мессенджере (например k.evseev@visionlabs.ru)"
        )
        return

    users = _load_store()["users"]
    added_any = False
    for login in sorted(candidates):
        if login in users:
            log.info("auth bootstrap: %s уже есть в %s", login, store_path().resolve())
            continue
        try:
            set_user_password(login, pw)
            added_any = True
            log.info("access bootstrap: добавлен пользователь %s → %s", login, store_path().resolve())
        except OSError:
            log.exception("auth bootstrap: не удалось записать %s", store_path().resolve())
            raise
    if not added_any and candidates:
        log.info("auth bootstrap: все перечисленные логины уже есть в хранилище")


def generate_password() -> str:
    return secrets.token_urlsafe(12)


def parse_admin_issue(text: str) -> str | None:
    m = _ISSUE_RE.match((text or "").strip())
    if not m:
        return None
    return _norm_login(m.group(1))


def parse_admin_revoke(text: str) -> str | None:
    m = _REVOKE_RE.match((text or "").strip())
    if not m:
        return None
    return _norm_login(m.group(1))
