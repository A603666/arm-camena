from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FramePacket:
    meta: dict[str, Any]
    rgb: np.ndarray
    depth: np.ndarray
