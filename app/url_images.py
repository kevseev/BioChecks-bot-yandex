from __future__ import annotations

import re
from urllib.parse import quote_plus, urlparse

import httpx


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0).strip("()[]<>{},.;\"'") for m in _URL_RE.finditer(text)]


def _is_image_signature(data: bytes) -> bool:
    if len(data) < 12:
        return False
    return (
        data.startswith(b"\xff\xd8")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
        or (data.startswith(b"RIFF") and data[8:12] == b"WEBP")
        or data.startswith(b"BM")
    )


def _google_drive_candidates(url: str) -> list[str]:
    out: list[str] = []
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        fid = m.group(1)
        out.append(f"https://drive.google.com/uc?export=download&id={fid}")
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        fid = m.group(1)
        out.append(f"https://drive.google.com/uc?export=download&id={fid}")
    return out


async def _yandex_disk_download_href(url: str) -> str | None:
    api = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        f"?public_key={quote_plus(url)}"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True) as c:
        r = await c.get(api)
    if r.status_code != 200:
        return None
    try:
        href = r.json().get("href")
    except ValueError:
        return None
    return href if isinstance(href, str) and href else None


async def fetch_image_from_url(url: str) -> tuple[bytes, str, str]:
    """
    Возвращает (bytes, content_type, final_url).
    Поддерживает обычные прямые URL и шары популярных дисков.
    """
    candidates: list[str] = [url]
    host = (urlparse(url).hostname or "").lower()

    if "drive.google.com" in host or "docs.google.com" in host:
        candidates.extend(_google_drive_candidates(url))

    if "disk.yandex.ru" in host or "yadi.sk" in host:
        href = await _yandex_disk_download_href(url)
        if href:
            candidates.append(href)

    # Dropbox shared link -> direct
    if "dropbox.com" in host and "dl=0" in url:
        candidates.append(url.replace("dl=0", "dl=1"))

    seen: set[str] = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                follow_redirects=True,
                headers=headers,
            ) as c:
                r = await c.get(cand)
            if r.status_code != 200:
                continue
            ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
            data = r.content
            if ctype.startswith("image/") or _is_image_signature(data):
                return data, (ctype or "image/jpeg"), str(r.url)
        except Exception:
            continue

    raise ValueError(f"Не удалось скачать изображение по ссылке: {url}")
