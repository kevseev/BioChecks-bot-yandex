from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yandex_bot_token: str = ""
    yandex_poll_interval_sec: float = 2.0
    yandex_updates_limit: int = 100
    yandex_offset_file: str = str(
        Path(__file__).resolve().parent.parent / "data" / "yandex_updates_offset.txt"
    )

    luna_base_url: str = "http://127.0.0.1:5000/6"
    luna_http_user: str = ""
    luna_http_password: str = ""
    luna_bearer_token: str = ""


settings = Settings()
