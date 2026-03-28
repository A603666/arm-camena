from __future__ import annotations

import uvicorn

from app.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run("app.server:app", host=cfg.host, port=cfg.port, reload=False, workers=1)
