from __future__ import annotations

import os

import uvicorn

from app.config import load_config


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "app.server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        workers=1,
        log_level=os.getenv("DABAI_UVICORN_LOG_LEVEL", "warning").strip().lower(),
        access_log=_env_bool("DABAI_UVICORN_ACCESS_LOG", False),
    )
