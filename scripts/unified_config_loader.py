#!/usr/bin/env python3
"""Unified pipeline config loader.

Supports:
- Unified config shape (pipeline_config.yaml)
- Legacy config shapes (dynamic_grasp / robot_runtime / auto_enable)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

UNIFIED_KEYS = {
    "dynamic_grasp",
    "vision_runtime",
    "robot_runtime",
    "auto_enable",
}

AUTO_ENABLE_REQUIRED_KEYS = {
    "can_channel",
    "can_bitrate",
    "can_interface",
    "usb_bus_info",
    "can_scripts_dir",
    "retry_interval_sec",
    "enable_timeout_sec",
    "health_check_sec",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def is_unified_config(raw: dict[str, Any]) -> bool:
    return any(isinstance(raw.get(key), dict) for key in UNIFIED_KEYS)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resolve_path(config_path: Path, raw_value: Any) -> str:
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)


def load_dynamic_grasp_section(config_path: str | Path) -> tuple[dict[str, Any], bool]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    unified = is_unified_config(raw)

    if unified:
        section = _require_mapping(raw.get("dynamic_grasp"), "dynamic_grasp")
        result = dict(section)
        runtime = dict(_require_mapping(result.get("runtime"), "dynamic_grasp.runtime"))
        # Unified default: point robot arm config to the unified file itself.
        if not runtime.get("arm_config_path"):
            runtime["arm_config_path"] = str(path)
        result["runtime"] = runtime
        return result, True

    legacy_keys = {"vision", "handeye", "scan", "grasp", "route", "runtime"}
    if legacy_keys.issubset(raw.keys()):
        return raw, False

    raise ValueError("config is neither unified dynamic_grasp config nor legacy dynamic_grasp config")


def load_robot_runtime_section(config_path: str | Path) -> tuple[dict[str, Any], bool]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    unified = is_unified_config(raw)

    if unified:
        section = _require_mapping(raw.get("robot_runtime"), "robot_runtime")
        return dict(section), True

    if "backend" in raw or "threepoint" in raw:
        return raw, False

    raise ValueError("config is neither unified robot_runtime config nor legacy robot config")


def load_auto_enable_section(config_path: str | Path) -> tuple[dict[str, Any], bool]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    unified = is_unified_config(raw)

    if unified:
        auto_enable = dict(_require_mapping(raw.get("auto_enable"), "auto_enable"))
        robot_runtime = raw.get("robot_runtime")
        if isinstance(robot_runtime, dict):
            for key in ("can_channel", "can_bitrate", "can_interface"):
                auto_enable.setdefault(key, robot_runtime.get(key))
            can_tools = robot_runtime.get("can_tools")
            if isinstance(can_tools, dict) and can_tools.get("scripts_dir") and not auto_enable.get("can_scripts_dir"):
                auto_enable["can_scripts_dir"] = can_tools.get("scripts_dir")

        missing = sorted(key for key in AUTO_ENABLE_REQUIRED_KEYS if auto_enable.get(key) in (None, ""))
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"auto_enable missing required keys in unified config: {joined}")
        return auto_enable, True

    if AUTO_ENABLE_REQUIRED_KEYS.issubset(raw.keys()):
        return raw, False

    raise ValueError("config is neither unified auto_enable config nor legacy auto_enable config")


def load_vision_env_map(config_path: str | Path) -> tuple[dict[str, str], bool]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    if not is_unified_config(raw):
        return {}, False

    section = _require_mapping(raw.get("vision_runtime"), "vision_runtime")

    def _read_path_value(key_path: str) -> Any:
        # v2 grouped layout only: detector/tracking/segmentation/geometry/stability/rollout/stream/service/...
        cur: Any = section
        for part in key_path.split("."):
            if not isinstance(cur, dict):
                return None
            if part not in cur:
                return None
            cur = cur.get(part)
        return cur

    env_key_map = {
        "stream.sub_endpoint": "DABAI_SUB_ENDPOINT",
        "stream.sub_topic": "DABAI_SUB_TOPIC",
        "stream.sub_timeout_ms": "DABAI_SUB_TIMEOUT_MS",
        "stream.pub_endpoint": "DABAI_PUB_ENDPOINT",
        "service.web_host": "DABAI_WEB_HOST",
        "service.web_port": "DABAI_WEB_PORT",
        "detector.yolo_conf": "DABAI_YOLO_CONF",
        "detector.yolo_imgsz": "DABAI_YOLO_IMGSZ",
        "detector.yolo_device": "DABAI_YOLO_DEVICE",
        "detector.yolo_precision": "DABAI_YOLO_PRECISION",
        "detector.yolo_warmup": "DABAI_YOLO_WARMUP",
        "geometry.geom_backend": "DABAI_GEOM_BACKEND",
        "geometry.geom_parity_check": "DABAI_GEOM_PARITY_CHECK",
        "geometry.geom_parity_every_n": "DABAI_GEOM_PARITY_EVERY_N",
        "detector.infer_every_n": "DABAI_INFER_EVERY_N",
        "geometry.geometry_every_n": "DABAI_GEOMETRY_EVERY_N",
        "geometry.geometry_force_recalc_iou": "DABAI_GEOMETRY_FORCE_RECALC_IOU",
        "geometry.center_depth_window": "DABAI_CENTER_DEPTH_WINDOW",
        "geometry.min_depth_mm": "DABAI_MIN_DEPTH_MM",
        "geometry.max_depth_mm": "DABAI_MAX_DEPTH_MM",
        "geometry.ransac_residual_mm": "DABAI_RANSAC_RESIDUAL_MM",
        "geometry.ransac_max_trials": "DABAI_RANSAC_MAX_TRIALS",
        "geometry.ground_fit_fast_enabled": "DABAI_GROUND_FIT_FAST_ENABLED",
        "geometry.ground_fit_fast_sample_cap": "DABAI_GROUND_FIT_FAST_SAMPLE_CAP",
        "geometry.ground_fit_fast_max_trials": "DABAI_GROUND_FIT_FAST_MAX_TRIALS",
        "geometry.ground_fit_fast_min_inlier_ratio": "DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO",
        "geometry.ground_fit_fast_min_inliers": "DABAI_GROUND_FIT_FAST_MIN_INLIERS",
        "geometry.dbscan_eps_mm": "DABAI_DBSCAN_EPS_MM",
        "geometry.dbscan_min_samples": "DABAI_DBSCAN_MIN_SAMPLES",
        "geometry.min_object_points": "DABAI_MIN_OBJECT_POINTS",
        "geometry.max_points_for_geometry": "DABAI_MAX_POINTS_FOR_GEOMETRY",
        "geometry.gripper_width_limit_mm": "DABAI_GRIPPER_WIDTH_LIMIT_MM",
        "geometry.grasp_window_len_mm": "DABAI_GRASP_WINDOW_LEN_MM",
        "geometry.grasp_window_step_mm": "DABAI_GRASP_WINDOW_STEP_MM",
        "geometry.grasp_window_min_points": "DABAI_GRASP_WINDOW_MIN_POINTS",
        "segmentation.object_height_min_mm": "DABAI_OBJECT_HEIGHT_MIN_MM",
        "segmentation.support_close_px": "DABAI_SUPPORT_CLOSE_PX",
        "segmentation.support_switch_hold_frames": "DABAI_SUPPORT_SWITCH_HOLD_FRAMES",
        "segmentation.depth_valid_ratio_min": "DABAI_DEPTH_VALID_RATIO_MIN",
        "segmentation.support_points_min": "DABAI_SUPPORT_POINTS_MIN",
        "segmentation.support_fill_ratio_min": "DABAI_SUPPORT_FILL_RATIO_MIN",
        "segmentation.ground_ratio_max": "DABAI_GROUND_RATIO_MAX",
        "segmentation.segmentation_debug_dbscan": "DABAI_SEGMENTATION_DEBUG_DBSCAN",
        "segmentation.seed_region_min_radius_px": "DABAI_SEED_REGION_MIN_RADIUS_PX",
        "segmentation.seed_region_max_radius_px": "DABAI_SEED_REGION_MAX_RADIUS_PX",
        "segmentation.seed_region_radius_step_px": "DABAI_SEED_REGION_RADIUS_STEP_PX",
        "tracking.target_lock_iou_min": "DABAI_TARGET_LOCK_IOU_MIN",
        "tracking.target_lock_hits": "DABAI_TARGET_LOCK_HITS",
        "tracking.target_lost_hold_frames": "DABAI_TARGET_LOST_HOLD_FRAMES",
        "tracking.target_max_center_jump_px": "DABAI_TARGET_MAX_CENTER_JUMP_PX",
        "tracking.target_max_depth_jump_mm": "DABAI_TARGET_MAX_DEPTH_JUMP_MM",
        "stability.axis_eig_ratio_min": "DABAI_AXIS_EIG_RATIO_MIN",
        "stability.axis_hold_frames": "DABAI_AXIS_HOLD_FRAMES",
        "stability.axis_smooth_alpha": "DABAI_AXIS_SMOOTH_ALPHA",
        "stability.quality_score_min": "DABAI_QUALITY_SCORE_MIN",
        "stability.grasp_point_smooth_alpha": "DABAI_GRASP_POINT_SMOOTH_ALPHA",
        "stability.grasp_yaw_smooth_alpha": "DABAI_GRASP_YAW_SMOOTH_ALPHA",
        "stability.grasp_hold_frames": "DABAI_GRASP_HOLD_FRAMES",
        "stability.grasp_jump_xy_mm": "DABAI_GRASP_JUMP_XY_MM",
        "stability.grasp_jump_z_mm": "DABAI_GRASP_JUMP_Z_MM",
        "stability.grasp_jump_yaw_deg": "DABAI_GRASP_JUMP_YAW_DEG",
        "service.annotated_jpeg_quality": "DABAI_ANNOTATED_JPEG_QUALITY",
        "service.stale_frame_sec": "DABAI_STALE_FRAME_SEC",
        "publisher.jpeg_quality": "DABAI_JPEG_QUALITY",
        "publisher.wait_ms": "DABAI_WAIT_MS",
        "publisher.color_fps": "DABAI_COLOR_FPS",
        "publisher.depth_fps": "DABAI_DEPTH_FPS",
        "publisher.align_mode": "DABAI_ALIGN_MODE",
        "publisher.frame_sync": "DABAI_FRAME_SYNC",
        "publisher.ob_log_level": "DABAI_OB_LOG_LEVEL",
        "service.uvicorn_log_level": "DABAI_UVICORN_LOG_LEVEL",
        "service.uvicorn_access_log": "DABAI_UVICORN_ACCESS_LOG",
        "build.skip_build": "DABAI_SKIP_BUILD",
        "build.force_rebuild": "DABAI_FORCE_REBUILD",
        "control.robot_control_enabled": "DABAI_ROBOT_CONTROL_ENABLED",
        "control.robot_loopback_only": "DABAI_ROBOT_LOOPBACK_ONLY",
        "control.robot_backend_override": "DABAI_ROBOT_BACKEND_OVERRIDE",
        "rollout.vision_pipeline": "DABAI_VISION_PIPELINE",
        "rollout.dynamic_grasp_api": "DABAI_DYNAMIC_GRASP_API",
        "rollout.shadow_compare": "DABAI_SHADOW_COMPARE",
        "rollout.metrics_window_size": "DABAI_METRICS_WINDOW_SIZE",
        "rollout.metrics_slow_frame_ms": "DABAI_METRICS_SLOW_FRAME_MS",
    }
    grouped_vision_sections = {
        "stream",
        "service",
        "publisher",
        "build",
        "detector",
        "geometry",
        "tracking",
        "segmentation",
        "stability",
        "control",
        "rollout",
        "extra_env",
    }
    deprecated_flat_keys = sorted(
        key
        for key, value in section.items()
        if key not in grouped_vision_sections
        and key in {source_key.split(".")[-1] for source_key in env_key_map.keys()} | {"yolo_model"}
        and not isinstance(value, dict)
    )
    if deprecated_flat_keys:
        joined = ", ".join(deprecated_flat_keys)
        raise ValueError(
            "vision_runtime contains deprecated flat keys: "
            f"{joined}. Run scripts/migrate_pipeline_vision_runtime_v2.py --config <path> --in-place."
        )

    env: dict[str, str] = {}
    for source_key, env_key in env_key_map.items():
        value = _read_path_value(source_key)
        if value is None:
            continue
        if env_key == "DABAI_ROBOT_BACKEND_OVERRIDE" and str(value).strip() == "":
            continue
        if isinstance(value, bool):
            env[env_key] = "1" if value else "0"
        else:
            env[env_key] = str(value)

    yolo_model = str(_read_path_value("detector.yolo_model") or "").strip()
    if yolo_model:
        env["DABAI_YOLO_MODEL"] = _resolve_path(path, yolo_model)

    # Unified robot config path points to the unified file itself.
    env["DABAI_ROBOT_CONFIG"] = str(path)

    extra_env = section.get("extra_env")
    if isinstance(extra_env, dict):
        for key, value in extra_env.items():
            name = str(key).strip()
            if not name:
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                env[name] = "1" if value else "0"
            else:
                env[name] = str(value)

    return env, True


def _print_section_yaml(section: dict[str, Any]) -> None:
    print(yaml.safe_dump(section, sort_keys=False, allow_unicode=True).strip())


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Unified pipeline config loader")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["detect", "component", "vision-env"],
        help="Output mode",
    )
    parser.add_argument(
        "--component",
        choices=["dynamic_grasp", "robot_runtime", "auto_enable"],
        help="Component name for --mode component",
    )
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()

    if args.mode == "detect":
        raw = _load_yaml(path)
        print("unified" if is_unified_config(raw) else "legacy")
        return 0

    if args.mode == "vision-env":
        env, _ = load_vision_env_map(path)
        for key in sorted(env.keys()):
            print(f"{key}={env[key]}")
        return 0

    if args.component is None:
        raise SystemExit("--component is required when --mode component")

    if args.component == "dynamic_grasp":
        section, _ = load_dynamic_grasp_section(path)
        _print_section_yaml(section)
        return 0
    if args.component == "robot_runtime":
        section, _ = load_robot_runtime_section(path)
        _print_section_yaml(section)
        return 0

    section, _ = load_auto_enable_section(path)
    _print_section_yaml(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
