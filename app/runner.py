from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

from app.access_control import ensure_bootstrap
from app.config import settings
from app.handlers import handle_update
from app.yandex_api import get_updates

log = logging.getLogger(__name__)


def _read_offset(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def _write_offset(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value), encoding="utf-8")


async def poll_loop(stop: asyncio.Event) -> None:
    if not settings.yandex_bot_token:
        raise SystemExit("Задайте YANDEX_BOT_TOKEN в окружении или .env")

    path = Path(settings.yandex_offset_file)
    log.info(
        "старт: BOT_AUTH_ENABLED=%s BOT_ACCESS_STORE=%s",
        settings.bot_auth_enabled,
        Path(settings.bot_access_store_path).resolve(),
    )
    try:
        ensure_bootstrap()
    except Exception:
        log.exception("ensure_bootstrap")
        raise
    offset = _read_offset(path)
    log.info("Yandex getUpdates: старт, offset=%s", offset)

    while not stop.is_set():
        try:
            body = await get_updates(offset=offset, limit=settings.yandex_updates_limit)
        except Exception:
            log.exception("getUpdates")
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.yandex_poll_interval_sec)
            except asyncio.TimeoutError:
                pass
            continue

        if not body.get("ok"):
            log.warning("getUpdates ok=false: %s", body)
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.yandex_poll_interval_sec)
            except asyncio.TimeoutError:
                pass
            continue

        updates = body.get("updates") or []
        for u in updates:
            try:
                await handle_update(u)
            except Exception:
                log.exception("handle_update")

        if updates:
            offset = max(int(x.get("update_id", 0) or 0) for x in updates) + 1
            _write_offset(path, offset)
            log.debug("offset -> %s", offset)

        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.yandex_poll_interval_sec)
        except asyncio.TimeoutError:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = asyncio.Event()

    def _shutdown() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(poll_loop(stop))
    except SystemExit:
        raise
    finally:
        loop.close()


if __name__ == "__main__":
    main()
