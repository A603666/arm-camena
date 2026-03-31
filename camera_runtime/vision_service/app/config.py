from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    vision_pipeline: str
    shadow_compare: bool
    zmq_endpoint: str
    zmq_topic: str
    zmq_timeout_ms: int
    host: str
    port: int
    model_path: Path
    yolo_conf: float
    yolo_imgsz: int
    yolo_device: str
    yolo_precision: str
    yolo_warmup: bool
    geom_backend: str
    geom_parity_check: bool
    geom_parity_every_n: int
    infer_every_n: int
    geometry_every_n: int
    geometry_force_recalc_iou: float
    center_depth_window: int
    min_depth_mm: float
    max_depth_mm: float
    ransac_residual_mm: float
    ransac_max_trials: int
    ground_fit_fast_enabled: bool
    ground_fit_fast_sample_cap: int
    ground_fit_fast_max_trials: int
    ground_fit_fast_min_inlier_ratio: float
    ground_fit_fast_min_inliers: int
    dbscan_eps_mm: float
    dbscan_min_samples: int
    min_object_points: int
    max_points_for_geometry: int
    gripper_width_limit_mm: float
    grasp_window_length_mm: float
    grasp_window_step_mm: float
    grasp_window_min_points: int
    object_height_min_mm: float
    support_close_px: int
    segmentation_debug_dbscan: bool
    seed_region_min_radius_px: int
    seed_region_max_radius_px: int
    seed_region_radius_step_px: int
    axis_eig_ratio_min: float
    axis_hold_frames: int
    axis_smooth_alpha: float
    target_lock_iou_min: float
    target_lock_hits: int
    target_lost_hold_frames: int
    target_max_center_jump_px: float
    target_max_depth_jump_mm: float
    support_switch_hold_frames: int
    depth_valid_ratio_min: float
    support_points_min: int
    support_fill_ratio_min: float
    ground_ratio_max: float
    quality_score_min: float
    grasp_point_smooth_alpha: float
    grasp_yaw_smooth_alpha: float
    grasp_hold_frames: int
    grasp_jump_xy_mm: float
    grasp_jump_z_mm: float
    grasp_jump_yaw_deg: float
    annotated_jpeg_quality: int
    stale_frame_threshold_sec: float
    metrics_window_size: int
    metrics_slow_frame_ms: float
    robot_control_enabled: bool
    robot_arm_config_path: Path
    robot_backend_override: str | None
    robot_loopback_only: bool


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _get_env_choice(name: str, default: str, valid: set[str]) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in valid:
        return normalized
    return default


def load_config() -> AppConfig:
    service_root = Path(__file__).resolve().parents[1]
    workspace_root = service_root.parent
    repo_root = workspace_root.parent
    unified_cfg = repo_root / "pipeline_config.yaml"
    default_model = workspace_root / "yolo26n.pt"
    default_arm_config = (
        unified_cfg
        if unified_cfg.is_file()
        else repo_root / "robot_runtime" / "config" / "default.yaml"
    )
    configured_model = Path(os.getenv("DABAI_YOLO_MODEL", str(default_model))).expanduser().resolve()
    configured_robot_cfg = Path(os.getenv("DABAI_ROBOT_CONFIG", str(default_arm_config))).expanduser().resolve()
    backend_override_raw = os.getenv("DABAI_ROBOT_BACKEND_OVERRIDE", "").strip().lower()
    backend_override = backend_override_raw if backend_override_raw else None

    return AppConfig(
        vision_pipeline=_get_env_choice("DABAI_VISION_PIPELINE", "v1", {"v1", "v2"}),
        shadow_compare=_get_env_bool("DABAI_SHADOW_COMPARE", False),
        zmq_endpoint=os.getenv("DABAI_SUB_ENDPOINT", "tcp://127.0.0.1:5557"),
        zmq_topic=os.getenv("DABAI_SUB_TOPIC", "frames.rgbd.v1"),
        zmq_timeout_ms=max(50, min(500, _get_env_int("DABAI_SUB_TIMEOUT_MS", 120))),
        host=os.getenv("DABAI_WEB_HOST", "0.0.0.0"),
        port=max(1, min(65535, _get_env_int("DABAI_WEB_PORT", 8000))),
        model_path=configured_model,
        yolo_conf=max(0.01, min(0.99, _get_env_float("DABAI_YOLO_CONF", 0.35))),
        yolo_imgsz=max(320, min(1280, _get_env_int("DABAI_YOLO_IMGSZ", 512))),
        yolo_device=os.getenv("DABAI_YOLO_DEVICE", "cpu"),
        yolo_precision=_get_env_choice("DABAI_YOLO_PRECISION", "fp32", {"fp32", "fp16"}),
        yolo_warmup=_get_env_bool("DABAI_YOLO_WARMUP", True),
        geom_backend=_get_env_choice("DABAI_GEOM_BACKEND", "auto", {"auto", "cpu", "torch", "cuml"}),
        geom_parity_check=_get_env_bool("DABAI_GEOM_PARITY_CHECK", False),
        geom_parity_every_n=max(1, min(300, _get_env_int("DABAI_GEOM_PARITY_EVERY_N", 30))),
        infer_every_n=max(1, min(6, _get_env_int("DABAI_INFER_EVERY_N", 2))),
        geometry_every_n=max(1, min(6, _get_env_int("DABAI_GEOMETRY_EVERY_N", 1))),
        geometry_force_recalc_iou=max(0.30, min(0.95, _get_env_float("DABAI_GEOMETRY_FORCE_RECALC_IOU", 0.75))),
        center_depth_window=max(1, min(25, _get_env_int("DABAI_CENTER_DEPTH_WINDOW", 5))),
        min_depth_mm=max(10.0, _get_env_float("DABAI_MIN_DEPTH_MM", 80.0)),
        max_depth_mm=max(500.0, _get_env_float("DABAI_MAX_DEPTH_MM", 5000.0)),
        ransac_residual_mm=max(1.0, _get_env_float("DABAI_RANSAC_RESIDUAL_MM", 12.0)),
        ransac_max_trials=max(8, min(200, _get_env_int("DABAI_RANSAC_MAX_TRIALS", 120))),
        ground_fit_fast_enabled=_get_env_bool("DABAI_GROUND_FIT_FAST_ENABLED", True),
        ground_fit_fast_sample_cap=max(500, min(20000, _get_env_int("DABAI_GROUND_FIT_FAST_SAMPLE_CAP", 2500))),
        ground_fit_fast_max_trials=max(8, min(120, _get_env_int("DABAI_GROUND_FIT_FAST_MAX_TRIALS", 40))),
        ground_fit_fast_min_inlier_ratio=max(0.05, min(1.0, _get_env_float("DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO", 0.55))),
        ground_fit_fast_min_inliers=max(60, min(20000, _get_env_int("DABAI_GROUND_FIT_FAST_MIN_INLIERS", 80))),
        dbscan_eps_mm=max(1.0, _get_env_float("DABAI_DBSCAN_EPS_MM", 20.0)),
        dbscan_min_samples=max(3, _get_env_int("DABAI_DBSCAN_MIN_SAMPLES", 30)),
        min_object_points=max(30, _get_env_int("DABAI_MIN_OBJECT_POINTS", 120)),
        max_points_for_geometry=max(500, _get_env_int("DABAI_MAX_POINTS_FOR_GEOMETRY", 12000)),
        gripper_width_limit_mm=max(10.0, _get_env_float("DABAI_GRIPPER_WIDTH_LIMIT_MM", 100.0)),
        grasp_window_length_mm=max(20.0, _get_env_float("DABAI_GRASP_WINDOW_LEN_MM", 80.0)),
        grasp_window_step_mm=max(1.0, _get_env_float("DABAI_GRASP_WINDOW_STEP_MM", 5.0)),
        grasp_window_min_points=max(20, _get_env_int("DABAI_GRASP_WINDOW_MIN_POINTS", 80)),
        object_height_min_mm=max(1.0, _get_env_float("DABAI_OBJECT_HEIGHT_MIN_MM", 4.0)),
        support_close_px=max(1, min(31, _get_env_int("DABAI_SUPPORT_CLOSE_PX", 5))),
        segmentation_debug_dbscan=_get_env_bool("DABAI_SEGMENTATION_DEBUG_DBSCAN", False),
        seed_region_min_radius_px=max(2, min(120, _get_env_int("DABAI_SEED_REGION_MIN_RADIUS_PX", 8))),
        seed_region_max_radius_px=max(8, min(400, _get_env_int("DABAI_SEED_REGION_MAX_RADIUS_PX", 140))),
        seed_region_radius_step_px=max(1, min(80, _get_env_int("DABAI_SEED_REGION_RADIUS_STEP_PX", 8))),
        axis_eig_ratio_min=max(1.01, _get_env_float("DABAI_AXIS_EIG_RATIO_MIN", 1.35)),
        axis_hold_frames=max(0, min(60, _get_env_int("DABAI_AXIS_HOLD_FRAMES", 5))),
        axis_smooth_alpha=max(0.0, min(1.0, _get_env_float("DABAI_AXIS_SMOOTH_ALPHA", 0.25))),
        target_lock_iou_min=max(0.10, min(0.95, _get_env_float("DABAI_TARGET_LOCK_IOU_MIN", 0.45))),
        target_lock_hits=max(1, min(10, _get_env_int("DABAI_TARGET_LOCK_HITS", 2))),
        target_lost_hold_frames=max(0, min(60, _get_env_int("DABAI_TARGET_LOST_HOLD_FRAMES", 6))),
        target_max_center_jump_px=max(1.0, min(800.0, _get_env_float("DABAI_TARGET_MAX_CENTER_JUMP_PX", 80.0))),
        target_max_depth_jump_mm=max(1.0, min(2000.0, _get_env_float("DABAI_TARGET_MAX_DEPTH_JUMP_MM", 80.0))),
        support_switch_hold_frames=max(1, min(20, _get_env_int("DABAI_SUPPORT_SWITCH_HOLD_FRAMES", 3))),
        depth_valid_ratio_min=max(0.001, min(1.0, _get_env_float("DABAI_DEPTH_VALID_RATIO_MIN", 0.03))),
        support_points_min=max(30, min(50000, _get_env_int("DABAI_SUPPORT_POINTS_MIN", 180))),
        support_fill_ratio_min=max(0.001, min(1.0, _get_env_float("DABAI_SUPPORT_FILL_RATIO_MIN", 0.02))),
        ground_ratio_max=max(0.10, min(1.0, _get_env_float("DABAI_GROUND_RATIO_MAX", 0.96))),
        quality_score_min=max(0.0, min(1.0, _get_env_float("DABAI_QUALITY_SCORE_MIN", 0.50))),
        grasp_point_smooth_alpha=max(0.0, min(1.0, _get_env_float("DABAI_GRASP_POINT_SMOOTH_ALPHA", 0.20))),
        grasp_yaw_smooth_alpha=max(0.0, min(1.0, _get_env_float("DABAI_GRASP_YAW_SMOOTH_ALPHA", 0.20))),
        grasp_hold_frames=max(0, min(60, _get_env_int("DABAI_GRASP_HOLD_FRAMES", 7))),
        grasp_jump_xy_mm=max(1.0, min(300.0, _get_env_float("DABAI_GRASP_JUMP_XY_MM", 22.0))),
        grasp_jump_z_mm=max(1.0, min(300.0, _get_env_float("DABAI_GRASP_JUMP_Z_MM", 22.0))),
        grasp_jump_yaw_deg=max(1.0, min(180.0, _get_env_float("DABAI_GRASP_JUMP_YAW_DEG", 20.0))),
        annotated_jpeg_quality=max(50, min(95, _get_env_int("DABAI_ANNOTATED_JPEG_QUALITY", 85))),
        stale_frame_threshold_sec=max(0.2, _get_env_float("DABAI_STALE_FRAME_SEC", 2.0)),
        metrics_window_size=max(10, min(600, _get_env_int("DABAI_METRICS_WINDOW_SIZE", 120))),
        metrics_slow_frame_ms=max(10.0, min(5000.0, _get_env_float("DABAI_METRICS_SLOW_FRAME_MS", 500.0))),
        robot_control_enabled=_get_env_bool("DABAI_ROBOT_CONTROL_ENABLED", True),
        robot_arm_config_path=configured_robot_cfg,
        robot_backend_override=backend_override,
        robot_loopback_only=_get_env_bool("DABAI_ROBOT_LOOPBACK_ONLY", True),
    )
