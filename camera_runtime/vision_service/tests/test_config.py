from __future__ import annotations

from vision_service.app.config import load_config


def test_load_config_uses_yolo26n_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DABAI_YOLO_MODEL", raising=False)
    cfg = load_config()
    assert cfg.model_path.name == "yolo26n.pt"
    assert cfg.robot_arm_config_path.name == "pipeline_config.yaml"
    assert cfg.robot_control_enabled is True
    assert cfg.robot_loopback_only is True
    assert cfg.yolo_precision == "fp32"
    assert cfg.yolo_warmup is True
    assert cfg.geom_backend == "auto"
    assert cfg.geom_parity_check is False
    assert cfg.geom_parity_every_n == 30
    assert cfg.ground_fit_fast_enabled is True
    assert cfg.ground_fit_fast_sample_cap == 2500
    assert cfg.ground_fit_fast_max_trials == 40
    assert abs(cfg.ground_fit_fast_min_inlier_ratio - 0.55) < 1e-6
    assert cfg.ground_fit_fast_min_inliers == 80
    assert abs(cfg.target_lock_iou_min - 0.45) < 1e-6
    assert cfg.target_lock_hits == 2
    assert cfg.target_lost_hold_frames == 6
    assert abs(cfg.target_max_center_jump_px - 80.0) < 1e-6
    assert abs(cfg.target_max_depth_jump_mm - 80.0) < 1e-6
    assert cfg.support_switch_hold_frames == 3
    assert abs(cfg.depth_valid_ratio_min - 0.03) < 1e-6
    assert cfg.support_points_min == 180
    assert abs(cfg.support_fill_ratio_min - 0.02) < 1e-6
    assert abs(cfg.ground_ratio_max - 0.96) < 1e-6
    assert abs(cfg.quality_score_min - 0.50) < 1e-6
    assert abs(cfg.grasp_point_smooth_alpha - 0.20) < 1e-6
    assert abs(cfg.grasp_yaw_smooth_alpha - 0.20) < 1e-6
    assert cfg.grasp_hold_frames == 7
    assert abs(cfg.grasp_jump_xy_mm - 22.0) < 1e-6
    assert abs(cfg.grasp_jump_z_mm - 22.0) < 1e-6
    assert abs(cfg.grasp_jump_yaw_deg - 20.0) < 1e-6


def test_load_config_reads_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOM_BACKEND", "torch")
    monkeypatch.setenv("DABAI_GEOM_PARITY_CHECK", "1")
    monkeypatch.setenv("DABAI_GEOM_PARITY_EVERY_N", "45")
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "2")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "30")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_ENABLED", "0")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_SAMPLE_CAP", "1800")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MAX_TRIALS", "22")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO", "0.72")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MIN_INLIERS", "140")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "0.8")
    monkeypatch.setenv("DABAI_OBJECT_HEIGHT_MIN_MM", "6.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "7")
    monkeypatch.setenv("DABAI_AXIS_EIG_RATIO_MIN", "1.6")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "9")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "0.4")
    monkeypatch.setenv("DABAI_TARGET_LOCK_IOU_MIN", "0.62")
    monkeypatch.setenv("DABAI_TARGET_LOCK_HITS", "3")
    monkeypatch.setenv("DABAI_TARGET_LOST_HOLD_FRAMES", "10")
    monkeypatch.setenv("DABAI_TARGET_MAX_CENTER_JUMP_PX", "65")
    monkeypatch.setenv("DABAI_TARGET_MAX_DEPTH_JUMP_MM", "55")
    monkeypatch.setenv("DABAI_SUPPORT_SWITCH_HOLD_FRAMES", "4")
    monkeypatch.setenv("DABAI_DEPTH_VALID_RATIO_MIN", "0.08")
    monkeypatch.setenv("DABAI_SUPPORT_POINTS_MIN", "260")
    monkeypatch.setenv("DABAI_SUPPORT_FILL_RATIO_MIN", "0.04")
    monkeypatch.setenv("DABAI_GROUND_RATIO_MAX", "0.9")
    monkeypatch.setenv("DABAI_QUALITY_SCORE_MIN", "0.66")
    monkeypatch.setenv("DABAI_GRASP_POINT_SMOOTH_ALPHA", "0.18")
    monkeypatch.setenv("DABAI_GRASP_YAW_SMOOTH_ALPHA", "0.19")
    monkeypatch.setenv("DABAI_GRASP_HOLD_FRAMES", "8")
    monkeypatch.setenv("DABAI_GRASP_JUMP_XY_MM", "24")
    monkeypatch.setenv("DABAI_GRASP_JUMP_Z_MM", "23")
    monkeypatch.setenv("DABAI_GRASP_JUMP_YAW_DEG", "18")

    cfg = load_config()
    assert cfg.geom_backend == "torch"
    assert cfg.geom_parity_check is True
    assert cfg.geom_parity_every_n == 45
    assert cfg.geometry_every_n == 2
    assert cfg.ransac_max_trials == 30
    assert cfg.ground_fit_fast_enabled is False
    assert cfg.ground_fit_fast_sample_cap == 1800
    assert cfg.ground_fit_fast_max_trials == 22
    assert abs(cfg.ground_fit_fast_min_inlier_ratio - 0.72) < 1e-6
    assert cfg.ground_fit_fast_min_inliers == 140
    assert abs(cfg.geometry_force_recalc_iou - 0.8) < 1e-6
    assert abs(cfg.object_height_min_mm - 6.5) < 1e-6
    assert cfg.support_close_px == 7
    assert abs(cfg.axis_eig_ratio_min - 1.6) < 1e-6
    assert cfg.axis_hold_frames == 9
    assert abs(cfg.axis_smooth_alpha - 0.4) < 1e-6
    assert abs(cfg.target_lock_iou_min - 0.62) < 1e-6
    assert cfg.target_lock_hits == 3
    assert cfg.target_lost_hold_frames == 10
    assert abs(cfg.target_max_center_jump_px - 65.0) < 1e-6
    assert abs(cfg.target_max_depth_jump_mm - 55.0) < 1e-6
    assert cfg.support_switch_hold_frames == 4
    assert abs(cfg.depth_valid_ratio_min - 0.08) < 1e-6
    assert cfg.support_points_min == 260
    assert abs(cfg.support_fill_ratio_min - 0.04) < 1e-6
    assert abs(cfg.ground_ratio_max - 0.9) < 1e-6
    assert abs(cfg.quality_score_min - 0.66) < 1e-6
    assert abs(cfg.grasp_point_smooth_alpha - 0.18) < 1e-6
    assert abs(cfg.grasp_yaw_smooth_alpha - 0.19) < 1e-6
    assert cfg.grasp_hold_frames == 8
    assert abs(cfg.grasp_jump_xy_mm - 24.0) < 1e-6
    assert abs(cfg.grasp_jump_z_mm - 23.0) < 1e-6
    assert abs(cfg.grasp_jump_yaw_deg - 18.0) < 1e-6


def test_load_config_clamps_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOM_BACKEND", "invalid")
    monkeypatch.setenv("DABAI_GEOM_PARITY_EVERY_N", "0")
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "99")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "-1")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_SAMPLE_CAP", "-3")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MAX_TRIALS", "999")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO", "9")
    monkeypatch.setenv("DABAI_GROUND_FIT_FAST_MIN_INLIERS", "0")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "2.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "0")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "99")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "5")
    monkeypatch.setenv("DABAI_TARGET_LOCK_IOU_MIN", "99")
    monkeypatch.setenv("DABAI_TARGET_LOCK_HITS", "0")
    monkeypatch.setenv("DABAI_TARGET_LOST_HOLD_FRAMES", "999")
    monkeypatch.setenv("DABAI_TARGET_MAX_CENTER_JUMP_PX", "0")
    monkeypatch.setenv("DABAI_TARGET_MAX_DEPTH_JUMP_MM", "-9")
    monkeypatch.setenv("DABAI_SUPPORT_SWITCH_HOLD_FRAMES", "-3")
    monkeypatch.setenv("DABAI_DEPTH_VALID_RATIO_MIN", "-0.1")
    monkeypatch.setenv("DABAI_SUPPORT_POINTS_MIN", "0")
    monkeypatch.setenv("DABAI_SUPPORT_FILL_RATIO_MIN", "9")
    monkeypatch.setenv("DABAI_GROUND_RATIO_MAX", "9")
    monkeypatch.setenv("DABAI_QUALITY_SCORE_MIN", "2.0")
    monkeypatch.setenv("DABAI_GRASP_POINT_SMOOTH_ALPHA", "-1")
    monkeypatch.setenv("DABAI_GRASP_YAW_SMOOTH_ALPHA", "9")
    monkeypatch.setenv("DABAI_GRASP_HOLD_FRAMES", "100")
    monkeypatch.setenv("DABAI_GRASP_JUMP_XY_MM", "0.1")
    monkeypatch.setenv("DABAI_GRASP_JUMP_Z_MM", "500")
    monkeypatch.setenv("DABAI_GRASP_JUMP_YAW_DEG", "999")

    cfg = load_config()
    assert cfg.geom_backend == "auto"
    assert cfg.geom_parity_every_n == 1
    assert cfg.geometry_every_n == 6
    assert cfg.ransac_max_trials == 8
    assert cfg.ground_fit_fast_sample_cap == 500
    assert cfg.ground_fit_fast_max_trials == 120
    assert abs(cfg.ground_fit_fast_min_inlier_ratio - 1.0) < 1e-6
    assert cfg.ground_fit_fast_min_inliers == 60
    assert abs(cfg.geometry_force_recalc_iou - 0.95) < 1e-6
    assert cfg.support_close_px == 1
    assert cfg.axis_hold_frames == 60
    assert abs(cfg.axis_smooth_alpha - 1.0) < 1e-6
    assert abs(cfg.target_lock_iou_min - 0.95) < 1e-6
    assert cfg.target_lock_hits == 1
    assert cfg.target_lost_hold_frames == 60
    assert abs(cfg.target_max_center_jump_px - 1.0) < 1e-6
    assert abs(cfg.target_max_depth_jump_mm - 1.0) < 1e-6
    assert cfg.support_switch_hold_frames == 1
    assert abs(cfg.depth_valid_ratio_min - 0.001) < 1e-6
    assert cfg.support_points_min == 30
    assert abs(cfg.support_fill_ratio_min - 1.0) < 1e-6
    assert abs(cfg.ground_ratio_max - 1.0) < 1e-6
    assert abs(cfg.quality_score_min - 1.0) < 1e-6
    assert abs(cfg.grasp_point_smooth_alpha - 0.0) < 1e-6
    assert abs(cfg.grasp_yaw_smooth_alpha - 1.0) < 1e-6
    assert cfg.grasp_hold_frames == 60
    assert abs(cfg.grasp_jump_xy_mm - 1.0) < 1e-6
    assert abs(cfg.grasp_jump_z_mm - 300.0) < 1e-6
    assert abs(cfg.grasp_jump_yaw_deg - 180.0) < 1e-6


def test_load_config_reads_robot_control_env(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "robot_web.yaml"
    cfg_path.write_text("backend: real\n", encoding="utf-8")

    monkeypatch.setenv("DABAI_ROBOT_CONTROL_ENABLED", "0")
    monkeypatch.setenv("DABAI_ROBOT_LOOPBACK_ONLY", "false")
    monkeypatch.setenv("DABAI_ROBOT_BACKEND_OVERRIDE", "real")
    monkeypatch.setenv("DABAI_ROBOT_CONFIG", str(cfg_path))

    cfg = load_config()
    assert cfg.robot_control_enabled is False
    assert cfg.robot_loopback_only is False
    assert cfg.robot_backend_override == "real"
    assert cfg.robot_arm_config_path == cfg_path.resolve()


def test_load_config_reads_yolo_precision_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_YOLO_PRECISION", "fp16")
    monkeypatch.setenv("DABAI_YOLO_WARMUP", "0")
    cfg = load_config()
    assert cfg.yolo_precision == "fp16"
    assert cfg.yolo_warmup is False
