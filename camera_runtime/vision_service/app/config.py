from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
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
    axis_eig_ratio_min: float
    axis_hold_frames: int
    axis_smooth_alpha: float
    annotated_jpeg_quality: int
    stale_frame_threshold_sec: float
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
    default_model = workspace_root / "yolo26n.pt"
    default_arm_config = repo_root / "robot_runtime" / "config" / "default.yaml"
    configured_model = Path(os.getenv("DABAI_YOLO_MODEL", str(default_model))).expanduser().resolve()
    configured_robot_cfg = Path(os.getenv("DABAI_ROBOT_CONFIG", str(default_arm_config))).expanduser().resolve()
    backend_override_raw = os.getenv("DABAI_ROBOT_BACKEND_OVERRIDE", "").strip().lower()
    backend_override = backend_override_raw if backend_override_raw else None

    return AppConfig(
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
        axis_eig_ratio_min=max(1.01, _get_env_float("DABAI_AXIS_EIG_RATIO_MIN", 1.35)),
        axis_hold_frames=max(0, min(60, _get_env_int("DABAI_AXIS_HOLD_FRAMES", 5))),
        axis_smooth_alpha=max(0.0, min(1.0, _get_env_float("DABAI_AXIS_SMOOTH_ALPHA", 0.25))),
        annotated_jpeg_quality=max(50, min(95, _get_env_int("DABAI_ANNOTATED_JPEG_QUALITY", 85))),
        stale_frame_threshold_sec=max(0.2, _get_env_float("DABAI_STALE_FRAME_SEC", 2.0)),
        robot_control_enabled=_get_env_bool("DABAI_ROBOT_CONTROL_ENABLED", True),
        robot_arm_config_path=configured_robot_cfg,
        robot_backend_override=backend_override,
        robot_loopback_only=_get_env_bool("DABAI_ROBOT_LOOPBACK_ONLY", True),
    )
