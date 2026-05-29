from __future__ import annotations

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.web_app:app",
        host=settings.web_host,
        port=settings.web_port,
        factory=False,
    )
