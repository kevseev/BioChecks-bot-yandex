from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yandex_bot_token: str = ""
    yandex_poll_interval_sec: float = 2.0
    yandex_updates_limit: int = 100
    yandex_offset_file: str = str(
        Path(__file__).resolve().parent.parent / "data" / "yandex_updates_offset.txt"
    )

    # Доступ к боту: пароли в JSON; в .env используйте BOT_ACCESS_STORE (или BOT_ACCESS_STORE_PATH)
    bot_auth_enabled: bool = False
    bot_admin_logins: str = ""
    bot_access_store_path: str = Field(
        default=str(Path(__file__).resolve().parent.parent / "data" / "access_users.json"),
        validation_alias=AliasChoices("BOT_ACCESS_STORE", "BOT_ACCESS_STORE_PATH"),
    )
    bot_bootstrap_admin_login: str = ""
    bot_bootstrap_admin_password: str = ""
    bot_password_pbkdf2_iterations: int = 200_000

    luna_base_url: str = "http://127.0.0.1:5000/6"
    luna_http_user: str = ""
    luna_http_password: str = ""
    luna_bearer_token: str = ""

    web_host: str = "0.0.0.0"
    web_port: int = 8080


settings = Settings()
