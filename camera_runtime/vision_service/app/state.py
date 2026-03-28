from __future__ import annotations

import threading
import time
from typing import Any


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_result: dict[str, Any] = {
            "status": "booting",
            "target": None,
            "depth": {"center_depth_mm": None},
            "size": None,
            "grasp": None,
            "timing": {"frame_id": None, "ts_us": None, "fps": 0.0},
        }
        self._latest_annotated_jpeg: bytes | None = None
        self._last_frame_wall_time: float | None = None
        self._last_update_wall_time: float = time.time()

    def update(self, result: dict[str, Any], annotated_jpeg: bytes | None, frame_wall_time: float) -> None:
        with self._lock:
            self._latest_result = result
            if annotated_jpeg:
                self._latest_annotated_jpeg = annotated_jpeg
            self._last_frame_wall_time = frame_wall_time
            self._last_update_wall_time = time.time()

    def get_latest_result(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_result)

    def get_latest_annotated_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_annotated_jpeg

    def get_health(self, stale_frame_threshold_sec: float) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            age = None if self._last_frame_wall_time is None else now - self._last_frame_wall_time
            update_age = now - self._last_update_wall_time
            return {
                "service": "ok",
                "stream_connected": age is not None and age < stale_frame_threshold_sec,
                "seconds_since_last_frame": age,
                "seconds_since_last_update": update_age,
                "latest_status": self._latest_result.get("status", "unknown"),
            }
