from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from vision_service.app.processor import VisionProcessor


def test_bbox_iou_for_identical_boxes() -> None:
    iou = VisionProcessor._bbox_iou([10, 20, 110, 220], [10, 20, 110, 220])
    assert abs(iou - 1.0) < 1e-6


def test_bbox_iou_for_disjoint_boxes() -> None:
    iou = VisionProcessor._bbox_iou([0, 0, 10, 10], [20, 20, 30, 30])
    assert iou == 0.0


def test_should_run_geometry_respects_every_n_and_iou() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(geometry_every_n=2, geometry_force_recalc_iou=0.75)
    processor._cached_geometry_payload = {"status": "ok", "size": None, "grasp": None}
    processor._cached_geometry_bbox = [100, 100, 200, 200]
    processor._cached_geometry_class_id = 0

    processor._frame_index = 3
    assert processor._should_run_geometry(0, [102, 102, 202, 202]) is False

    processor._frame_index = 4
    assert processor._should_run_geometry(0, [102, 102, 202, 202]) is True

    processor._frame_index = 5
    assert processor._should_run_geometry(0, [260, 260, 320, 320]) is True


def test_stabilize_axis_flips_sign_and_smooths_consistently() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        geometry_force_recalc_iou=0.75,
        axis_hold_frames=5,
        axis_smooth_alpha=0.25,
        min_object_points=120,
        support_close_px=5,
        axis_eig_ratio_min=1.35,
    )
    processor._axis_tracker = None

    axis1, state1, quality1 = processor._stabilize_axis(
        axis_dir_cam=[1.0, 0.0, 0.0],
        reliable=True,
        axis_quality=0.9,
        class_id=1,
        bbox=[100, 100, 200, 200],
    )
    assert state1 == "live"
    assert quality1 == 0.9
    assert axis1 is not None
    assert axis1[0] > 0.99

    axis2, state2, _ = processor._stabilize_axis(
        axis_dir_cam=[-1.0, 0.0, 0.0],
        reliable=True,
        axis_quality=0.85,
        class_id=1,
        bbox=[102, 102, 202, 202],
    )
    assert state2 == "live"
    assert axis2 is not None
    assert axis2[0] > 0.99


def test_stabilize_axis_holds_then_marks_unreliable() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        geometry_force_recalc_iou=0.75,
        axis_hold_frames=1,
        axis_smooth_alpha=0.25,
        min_object_points=120,
        support_close_px=5,
        axis_eig_ratio_min=1.35,
    )
    processor._axis_tracker = None

    axis_live, state_live, _ = processor._stabilize_axis(
        axis_dir_cam=[0.0, 1.0, 0.0],
        reliable=True,
        axis_quality=0.8,
        class_id=5,
        bbox=[50, 50, 140, 180],
    )
    assert state_live == "live"
    assert axis_live is not None

    axis_held, state_held, held_quality = processor._stabilize_axis(
        axis_dir_cam=[0.1, 0.9, 0.0],
        reliable=False,
        axis_quality=0.2,
        class_id=5,
        bbox=[52, 50, 142, 180],
    )
    assert state_held == "held"
    assert axis_held is not None
    assert axis_held[1] > 0.95
    assert held_quality > 0.2

    axis_unreliable, state_unreliable, unreliable_quality = processor._stabilize_axis(
        axis_dir_cam=[-0.8, 0.2, 0.0],
        reliable=False,
        axis_quality=0.1,
        class_id=5,
        bbox=[54, 50, 144, 180],
    )
    assert state_unreliable == "unreliable"
    assert axis_unreliable is not None
    assert unreliable_quality == 0.1


def test_geometry_payload_within_threshold_accepts_small_delta() -> None:
    lhs = {
        "status": "ok",
        "size": {"length_mm": 100.0, "width_mm": 30.0, "height_mm": 12.0},
        "grasp": {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 30.0, "yaw_deg": 15.0},
    }
    rhs = {
        "status": "ok",
        "size": {"length_mm": 102.8, "width_mm": 27.5, "height_mm": 14.5},
        "grasp": {"x_mm": 11.6, "y_mm": 19.3, "z_mm": 31.5, "yaw_deg": 16.4},
    }
    ok, _ = VisionProcessor._geometry_payload_within_threshold(lhs, rhs)
    assert ok is True


def test_geometry_payload_within_threshold_rejects_large_delta() -> None:
    lhs = {
        "status": "ok",
        "size": {"length_mm": 100.0, "width_mm": 30.0, "height_mm": 12.0},
        "grasp": {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 30.0, "yaw_deg": 15.0},
    }
    rhs = {
        "status": "ok",
        "size": {"length_mm": 104.0, "width_mm": 30.0, "height_mm": 12.0},
        "grasp": {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 30.0, "yaw_deg": 15.0},
    }
    ok, reason = VisionProcessor._geometry_payload_within_threshold(lhs, rhs)
    assert ok is False
    assert "size mismatch" in reason


def test_run_geometry_with_fallback_uses_cpu_on_backend_error() -> None:
    class DummyBackend:
        def __init__(self, name: str) -> None:
            self.name = name

    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(geom_parity_check=False, geom_parity_every_n=30)
    processor._geometry_backend = DummyBackend("torch")
    processor._cpu_geometry_backend = DummyBackend("cpu")
    processor._geometry_force_cpu = False
    processor._geometry_fallback_count = 0
    processor._geometry_parity_mismatch_count = 0
    processor._frame_index = 1
    processor._axis_tracker = None
    processor._logger = logging.getLogger("test")

    def fake_compute(*, backend, **kwargs):
        if backend.name == "torch":
            raise RuntimeError("gpu failed")
        return {"status": "ok", "size": None, "grasp": None}, {"total_ms": 1.0}

    processor._compute_geometry_payload = fake_compute  # type: ignore[method-assign]

    payload, timing, backend = processor._run_geometry_with_fallback(
        depth=np.zeros((4, 4), dtype=np.uint16),
        bbox=[0, 0, 4, 4],
        class_id=0,
        depth_scale=1.0,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        ref_uv=(0, 0),
    )
    assert payload["status"] == "ok"
    assert timing["total_ms"] == 1.0
    assert backend == "cpu_fallback"
    assert processor._geometry_fallback_count == 1


def test_run_geometry_with_fallback_forces_cpu_after_parity_mismatch() -> None:
    class DummyBackend:
        def __init__(self, name: str) -> None:
            self.name = name

    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(geom_parity_check=True, geom_parity_every_n=5)
    processor._geometry_backend = DummyBackend("torch")
    processor._cpu_geometry_backend = DummyBackend("cpu")
    processor._geometry_force_cpu = False
    processor._geometry_fallback_count = 0
    processor._geometry_parity_mismatch_count = 0
    processor._frame_index = 10
    processor._axis_tracker = None
    processor._logger = logging.getLogger("test")

    def fake_compute(*, backend, **kwargs):
        if backend.name == "torch":
            return (
                {
                    "status": "ok",
                    "size": {"length_mm": 120.0, "width_mm": 20.0, "height_mm": 10.0},
                    "grasp": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "yaw_deg": 0.0},
                },
                {"total_ms": 2.0},
            )
        return (
            {
                "status": "ok",
                "size": {"length_mm": 130.5, "width_mm": 20.0, "height_mm": 10.0},
                "grasp": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "yaw_deg": 0.0},
            },
            {"total_ms": 3.0},
        )

    processor._compute_geometry_payload = fake_compute  # type: ignore[method-assign]

    payload, timing, backend = processor._run_geometry_with_fallback(
        depth=np.zeros((4, 4), dtype=np.uint16),
        bbox=[0, 0, 4, 4],
        class_id=0,
        depth_scale=1.0,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        ref_uv=(0, 0),
    )
    assert backend == "cpu_forced_parity"
    assert timing["total_ms"] == 3.0
    assert abs(payload["size"]["length_mm"] - 130.5) < 1e-6
    assert processor._geometry_force_cpu is True
    assert processor._geometry_parity_mismatch_count == 1
    assert processor._geometry_fallback_count == 1
