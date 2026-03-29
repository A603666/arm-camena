from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from vision_service.app.processor import VisionProcessor
from vision_service.app.types import FramePacket


class StubCalibrationManager:
    def __init__(self, active: bool) -> None:
        self._active = active
        self.ingest_calls = 0

    def ingest_frame(self, rgb, meta) -> None:
        _ = rgb, meta
        self.ingest_calls += 1

    def is_mode_active(self) -> bool:
        return self._active


def test_process_packet_short_circuits_when_calibration_mode_active() -> None:
    processor = object.__new__(VisionProcessor)
    processor._logger = logging.getLogger("test")
    processor._cfg = SimpleNamespace(
        center_depth_window=3,
        min_depth_mm=10.0,
        max_depth_mm=5000.0,
    )
    processor._calibration_manager = StubCalibrationManager(active=True)

    processor._cached_detections = [{"dummy": True}]
    processor._cached_geometry_payload = {"status": "ok"}
    processor._cached_geometry_bbox = [1, 1, 2, 2]
    processor._cached_geometry_class_id = 0
    processor._cached_geometry_backend = "cpu"
    processor._axis_tracker = {"x": 1}
    processor._grasp_tracker = {"x": 1}
    processor._target_tracker = {"x": 1}
    processor._support_source_tracker = {"x": 1}
    processor._geometry_fallback_count = 0
    processor._geometry_parity_mismatch_count = 0

    processor._encode_annotated = lambda _img: b"jpeg"

    packet = FramePacket(
        meta={"frame_id": 1, "ts_us": 2, "depth_scale": 1.0},
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        depth=np.full((16, 16), 1000, dtype=np.uint16),
    )

    result, jpeg = processor._process_packet(packet, fps=10.0)

    assert jpeg == b"jpeg"
    assert result["status"] == "calibration_mode"
    assert result["timing"]["infer_ran"] is False
    assert result["timing"]["geometry_ran"] is False
    assert "process_rss_mb" in result["timing"]
    assert "geometry_ground_fit_mode" in result["timing"]
    assert processor._calibration_manager.ingest_calls == 1
    assert processor._cached_detections == []
    assert processor._cached_geometry_payload is None
    assert processor._axis_tracker is None
    assert processor._grasp_tracker is None
    assert processor._target_tracker is None
    assert processor._support_source_tracker is None


def test_process_packet_skips_calibration_ingest_when_mode_inactive() -> None:
    processor = object.__new__(VisionProcessor)
    processor._logger = logging.getLogger("test")
    processor._cfg = SimpleNamespace(
        center_depth_window=3,
        min_depth_mm=10.0,
        max_depth_mm=5000.0,
        infer_every_n=1,
    )
    processor._calibration_manager = StubCalibrationManager(active=False)

    processor._frame_index = 0
    processor._cached_detections = []
    processor._cached_geometry_payload = None
    processor._cached_geometry_bbox = None
    processor._cached_geometry_class_id = None
    processor._cached_geometry_backend = None
    processor._axis_tracker = None
    processor._grasp_tracker = None
    processor._target_tracker = None
    processor._support_source_tracker = None
    processor._geometry_fallback_count = 0
    processor._geometry_parity_mismatch_count = 0
    processor._infer = lambda _img: []
    processor._encode_annotated = lambda _img: b"jpeg"

    packet = FramePacket(
        meta={"frame_id": 9, "ts_us": 10, "depth_scale": 1.0},
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        depth=np.full((16, 16), 1200, dtype=np.uint16),
    )

    result, jpeg = processor._process_packet(packet, fps=12.5)

    assert jpeg == b"jpeg"
    assert result["status"] == "no_target"
    assert result["timing"]["infer_ran"] is True
    assert result["timing"]["geometry_ran"] is False
    assert result["timing"]["process_threads"] is not None
    assert processor._calibration_manager.ingest_calls == 0
