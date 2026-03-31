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


def test_map_uv_between_frames_respects_center_and_bounds() -> None:
    mapped_center = VisionProcessor._map_uv_between_frames(
        u=320,
        v=240,
        src_width=640,
        src_height=480,
        dst_width=640,
        dst_height=400,
    )
    assert mapped_center == (320, 200)

    mapped_corner = VisionProcessor._map_uv_between_frames(
        u=639,
        v=479,
        src_width=640,
        src_height=480,
        dst_width=640,
        dst_height=400,
    )
    assert mapped_corner == (639, 399)


def test_map_bbox_between_frames_scales_to_depth_space() -> None:
    mapped = VisionProcessor._map_bbox_between_frames(
        bbox=[267, 206, 357, 468],
        src_width=640,
        src_height=480,
        dst_width=640,
        dst_height=400,
    )
    assert mapped == (267, 172, 357, 390)


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


def test_stabilize_grasp_smooths_stable_measurements() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        geometry_force_recalc_iou=0.75,
        grasp_hold_frames=7,
        grasp_point_smooth_alpha=0.2,
        grasp_yaw_smooth_alpha=0.2,
        grasp_jump_xy_mm=22.0,
        grasp_jump_z_mm=22.0,
        grasp_jump_yaw_deg=20.0,
    )
    processor._grasp_tracker = None

    xyz1, yaw1, state1, score1 = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([10.0, 5.0, 1000.0], dtype=np.float32),
        raw_yaw_deg=10.0,
        class_id=1,
        bbox=[100, 100, 220, 220],
        axis_quality=0.9,
    )
    assert state1 == "live"
    assert xyz1 is not None
    assert yaw1 is not None
    assert score1 > 0.8

    xyz2, yaw2, state2, score2 = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([12.0, 6.0, 1002.0], dtype=np.float32),
        raw_yaw_deg=14.0,
        class_id=1,
        bbox=[102, 102, 222, 222],
        axis_quality=0.9,
    )
    assert state2 == "live"
    assert xyz2 is not None
    assert yaw2 is not None
    assert score2 > 0.7
    assert float(xyz2[0]) < 12.0
    assert 10.0 < float(yaw2) < 14.0


def test_stabilize_grasp_holds_on_large_jump() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        geometry_force_recalc_iou=0.75,
        grasp_hold_frames=2,
        grasp_point_smooth_alpha=0.2,
        grasp_yaw_smooth_alpha=0.2,
        grasp_jump_xy_mm=20.0,
        grasp_jump_z_mm=20.0,
        grasp_jump_yaw_deg=20.0,
    )
    processor._grasp_tracker = None

    xyz_live, yaw_live, _, _ = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([0.0, 0.0, 900.0], dtype=np.float32),
        raw_yaw_deg=5.0,
        class_id=2,
        bbox=[80, 80, 180, 180],
        axis_quality=0.8,
    )
    assert xyz_live is not None
    assert yaw_live is not None

    xyz_held, yaw_held, state_held, _ = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([80.0, 60.0, 950.0], dtype=np.float32),
        raw_yaw_deg=60.0,
        class_id=2,
        bbox=[82, 82, 182, 182],
        axis_quality=0.2,
    )
    assert state_held == "held"
    assert xyz_held is not None
    assert yaw_held is not None
    assert np.allclose(xyz_held, xyz_live, atol=1e-6)
    assert abs(float(yaw_held) - float(yaw_live)) < 1e-6


def test_stabilize_grasp_wraps_yaw_near_180_deg() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        geometry_force_recalc_iou=0.75,
        grasp_hold_frames=4,
        grasp_point_smooth_alpha=0.2,
        grasp_yaw_smooth_alpha=0.2,
        grasp_jump_xy_mm=22.0,
        grasp_jump_z_mm=22.0,
        grasp_jump_yaw_deg=25.0,
    )
    processor._grasp_tracker = None

    _, yaw1, _, _ = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([1.0, 1.0, 900.0], dtype=np.float32),
        raw_yaw_deg=179.0,
        class_id=3,
        bbox=[60, 60, 150, 150],
        axis_quality=0.8,
    )
    _, yaw2, state2, _ = processor._stabilize_grasp(
        raw_grasp_xyz_mm=np.array([1.5, 1.2, 900.5], dtype=np.float32),
        raw_yaw_deg=-179.0,
        class_id=3,
        bbox=[61, 61, 151, 151],
        axis_quality=0.8,
    )
    assert state2 == "live"
    assert yaw1 is not None
    assert yaw2 is not None
    assert VisionProcessor._yaw_delta_deg(yaw1, yaw2) < 5.0


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
    processor._grasp_tracker = None
    processor._support_source_tracker = None
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
    processor._grasp_tracker = None
    processor._support_source_tracker = None
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


def test_select_target_uses_tracker_and_rejects_large_switch() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        center_depth_window=5,
        min_depth_mm=80.0,
        max_depth_mm=5000.0,
        target_lock_iou_min=0.45,
        target_lock_hits=2,
        target_lost_hold_frames=4,
        target_max_center_jump_px=80.0,
        target_max_depth_jump_mm=80.0,
    )
    processor._target_tracker = None

    depth = np.full((480, 640), 300, dtype=np.uint16)
    depth[:220, :] = 200
    top = {"conf": 0.42, "class_id": 31, "class_name": "snowboard", "bbox_xyxy": [267, 1, 415, 200]}
    bottom = {"conf": 0.48, "class_id": 4, "class_name": "airplane", "bbox_xyxy": [139, 313, 538, 478]}

    selected1, state1, _, reason1 = processor._select_target([top], 640, 480, depth, 1.0)
    assert selected1 is None
    assert state1 == "acquire"
    assert reason1 == "acquiring_lock"

    selected2, state2, score2, reason2 = processor._select_target([top], 640, 480, depth, 1.0)
    assert selected2 is not None
    assert state2 == "locked"
    assert score2 is not None and score2 > 0.0
    assert reason2 is None

    selected3, state3, _, reason3 = processor._select_target([bottom], 640, 480, depth, 1.0)
    assert selected3 is None
    assert state3 == "lost"
    assert reason3 in {"iou_break", "center_jump", "depth_jump"}


def test_stabilize_support_source_has_hysteresis() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(support_switch_hold_frames=3)
    processor._support_source_tracker = None

    options = {"support_region": {"points": np.zeros((1, 3))}, "legacy_cluster": {"points": np.zeros((1, 3))}}
    source1, switches1 = processor._stabilize_support_source("support_region", options)
    assert source1 == "support_region"
    assert switches1 == 0

    source2, _ = processor._stabilize_support_source("legacy_cluster", options)
    source3, _ = processor._stabilize_support_source("legacy_cluster", options)
    source4, switches4 = processor._stabilize_support_source("legacy_cluster", options)
    assert source2 == "support_region"
    assert source3 == "support_region"
    assert source4 == "legacy_cluster"
    assert switches4 == 1


def test_evaluate_geometry_quality_rejects_low_quality_frames() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        depth_valid_ratio_min=0.03,
        support_points_min=180,
        support_fill_ratio_min=0.02,
        ground_ratio_max=0.96,
        axis_eig_ratio_min=1.35,
        quality_score_min=0.5,
    )
    low_seg = {
        "depth_valid_ratio": 0.005,
        "support_points": 120,
        "support_fill_ratio": 0.01,
        "ground_ratio": 0.99,
        "axis_eig_ratio": 1.1,
    }
    ok, score, flags = processor._evaluate_geometry_quality(low_seg)
    assert ok is False
    assert score < 0.5
    assert "quality_score" in flags
    assert "depth_valid_ratio" in flags
    assert "support_points" in flags

    good_seg = {
        "depth_valid_ratio": 0.06,
        "support_points": 450,
        "support_fill_ratio": 0.14,
        "ground_ratio": 0.7,
        "axis_eig_ratio": 2.2,
    }
    ok_good, score_good, flags_good = processor._evaluate_geometry_quality(good_seg)
    assert ok_good is True
    assert score_good >= 0.5
    assert flags_good == []


def test_finalize_packet_output_v1_with_shadow_compare_adds_summary() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        vision_pipeline="v1",
        shadow_compare=True,
        metrics_window_size=10,
        metrics_slow_frame_ms=500.0,
    )
    processor._encode_annotated = lambda _img: b"jpeg"

    result = {
        "status": "no_target",
        "target": None,
        "depth": {"center_depth_mm": 1000.0},
        "size": None,
        "grasp": None,
        "segmentation": None,
        "timing": {"infer_ms": 3.0, "geometry_total_ms": 1.0},
    }

    output, encoded = processor._finalize_packet_output(result, np.zeros((8, 8, 3), dtype=np.uint8))
    assert encoded == b"jpeg"
    assert output["status"] == "no_target"
    shadow = output.get("shadow_compare")
    assert isinstance(shadow, dict)
    assert shadow.get("enabled") is True
    assert shadow.get("primary_pipeline") == "v1"
    assert shadow.get("v1", {}).get("status") == "no_target"
    assert shadow.get("v2", {}).get("status") == "lost"


def test_finalize_packet_output_v2_with_shadow_compare_adds_summary() -> None:
    processor = object.__new__(VisionProcessor)
    processor._cfg = SimpleNamespace(
        vision_pipeline="v2",
        shadow_compare=True,
        metrics_window_size=10,
        metrics_slow_frame_ms=500.0,
    )
    processor._encode_annotated = lambda _img: b"jpeg"

    result = {
        "status": "ok",
        "target": {
            "class_id": 0,
            "class_name": "box",
            "conf": 0.9,
            "bbox_xyxy": [10, 20, 30, 40],
            "tracker_state": "locked",
            "tracker_score": 0.88,
            "rejected_reason": None,
        },
        "depth": {"target_center_depth_mm": 900.0},
        "size": {"width_mm": 40.0},
        "grasp": {"x_mm": 1.0, "y_mm": 2.0, "z_mm": 3.0, "axis_dir_cam": [1.0, 0.0, 0.0]},
        "segmentation": {"quality_score": 0.8, "quality_flags": []},
        "timing": {"infer_ms": 4.0, "geometry_total_ms": 2.0},
    }

    output, encoded = processor._finalize_packet_output(result, np.zeros((8, 8, 3), dtype=np.uint8))
    assert encoded == b"jpeg"
    assert output["schema_version"] == 2
    assert output["status"] == "ok"
    shadow = output.get("shadow_compare")
    assert isinstance(shadow, dict)
    assert shadow.get("primary_pipeline") == "v2"
    assert shadow.get("status_match") is True
    assert shadow.get("v2", {}).get("tracking", {}).get("state") == "locked"
