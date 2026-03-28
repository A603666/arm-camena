from __future__ import annotations

import sys
from pathlib import Path


CAMERA_RUNTIME_DIR = Path(__file__).resolve().parents[2]
if str(CAMERA_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_RUNTIME_DIR))
