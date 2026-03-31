#!/usr/bin/env python3
"""One-shot migrator for vision_runtime flat keys -> grouped v2 keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# Legacy flat key -> grouped v2 key path under vision_runtime.
FLAT_TO_GROUPED: dict[str, str] = {
    "sub_endpoint": "stream.sub_endpoint",
    "sub_topic": "stream.sub_topic",
    "sub_timeout_ms": "stream.sub_timeout_ms",
    "pub_endpoint": "stream.pub_endpoint",
    "web_host": "service.web_host",
    "web_port": "service.web_port",
    "annotated_jpeg_quality": "service.annotated_jpeg_quality",
    "stale_frame_sec": "service.stale_frame_sec",
    "uvicorn_log_level": "service.uvicorn_log_level",
    "uvicorn_access_log": "service.uvicorn_access_log",
    "jpeg_quality": "publisher.jpeg_quality",
    "wait_ms": "publisher.wait_ms",
    "color_fps": "publisher.color_fps",
    "depth_fps": "publisher.depth_fps",
    "align_mode": "publisher.align_mode",
    "frame_sync": "publisher.frame_sync",
    "ob_log_level": "publisher.ob_log_level",
    "skip_build": "build.skip_build",
    "force_rebuild": "build.force_rebuild",
    "yolo_model": "detector.yolo_model",
    "yolo_conf": "detector.yolo_conf",
    "yolo_imgsz": "detector.yolo_imgsz",
    "yolo_device": "detector.yolo_device",
    "yolo_precision": "detector.yolo_precision",
    "yolo_warmup": "detector.yolo_warmup",
    "infer_every_n": "detector.infer_every_n",
    "geom_backend": "geometry.geom_backend",
    "geom_parity_check": "geometry.geom_parity_check",
    "geom_parity_every_n": "geometry.geom_parity_every_n",
    "geometry_every_n": "geometry.geometry_every_n",
    "geometry_force_recalc_iou": "geometry.geometry_force_recalc_iou",
    "center_depth_window": "geometry.center_depth_window",
    "min_depth_mm": "geometry.min_depth_mm",
    "max_depth_mm": "geometry.max_depth_mm",
    "ransac_residual_mm": "geometry.ransac_residual_mm",
    "ransac_max_trials": "geometry.ransac_max_trials",
    "ground_fit_fast_enabled": "geometry.ground_fit_fast_enabled",
    "ground_fit_fast_sample_cap": "geometry.ground_fit_fast_sample_cap",
    "ground_fit_fast_max_trials": "geometry.ground_fit_fast_max_trials",
    "ground_fit_fast_min_inlier_ratio": "geometry.ground_fit_fast_min_inlier_ratio",
    "ground_fit_fast_min_inliers": "geometry.ground_fit_fast_min_inliers",
    "dbscan_eps_mm": "geometry.dbscan_eps_mm",
    "dbscan_min_samples": "geometry.dbscan_min_samples",
    "min_object_points": "geometry.min_object_points",
    "max_points_for_geometry": "geometry.max_points_for_geometry",
    "gripper_width_limit_mm": "geometry.gripper_width_limit_mm",
    "grasp_window_len_mm": "geometry.grasp_window_len_mm",
    "grasp_window_step_mm": "geometry.grasp_window_step_mm",
    "grasp_window_min_points": "geometry.grasp_window_min_points",
    "target_lock_iou_min": "tracking.target_lock_iou_min",
    "target_lock_hits": "tracking.target_lock_hits",
    "target_lost_hold_frames": "tracking.target_lost_hold_frames",
    "target_max_center_jump_px": "tracking.target_max_center_jump_px",
    "target_max_depth_jump_mm": "tracking.target_max_depth_jump_mm",
    "object_height_min_mm": "segmentation.object_height_min_mm",
    "support_close_px": "segmentation.support_close_px",
    "support_switch_hold_frames": "segmentation.support_switch_hold_frames",
    "depth_valid_ratio_min": "segmentation.depth_valid_ratio_min",
    "support_points_min": "segmentation.support_points_min",
    "support_fill_ratio_min": "segmentation.support_fill_ratio_min",
    "ground_ratio_max": "segmentation.ground_ratio_max",
    "segmentation_debug_dbscan": "segmentation.segmentation_debug_dbscan",
    "seed_region_min_radius_px": "segmentation.seed_region_min_radius_px",
    "seed_region_max_radius_px": "segmentation.seed_region_max_radius_px",
    "seed_region_radius_step_px": "segmentation.seed_region_radius_step_px",
    "axis_eig_ratio_min": "stability.axis_eig_ratio_min",
    "axis_hold_frames": "stability.axis_hold_frames",
    "axis_smooth_alpha": "stability.axis_smooth_alpha",
    "quality_score_min": "stability.quality_score_min",
    "grasp_point_smooth_alpha": "stability.grasp_point_smooth_alpha",
    "grasp_yaw_smooth_alpha": "stability.grasp_yaw_smooth_alpha",
    "grasp_hold_frames": "stability.grasp_hold_frames",
    "grasp_jump_xy_mm": "stability.grasp_jump_xy_mm",
    "grasp_jump_z_mm": "stability.grasp_jump_z_mm",
    "grasp_jump_yaw_deg": "stability.grasp_jump_yaw_deg",
    "robot_control_enabled": "control.robot_control_enabled",
    "robot_loopback_only": "control.robot_loopback_only",
    "robot_backend_override": "control.robot_backend_override",
    "vision_pipeline": "rollout.vision_pipeline",
    "dynamic_grasp_api": "rollout.dynamic_grasp_api",
    "shadow_compare": "rollout.shadow_compare",
    "metrics_window_size": "rollout.metrics_window_size",
    "metrics_slow_frame_ms": "rollout.metrics_slow_frame_ms",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def _has_dotted_key(root: dict[str, Any], dotted: str) -> bool:
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return False
        if part not in cur:
            return False
        cur = cur.get(part)
    return True


def _set_dotted_key(root: dict[str, Any], dotted: str, value: Any) -> None:
    cur = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def migrate_vision_runtime_section(section: dict[str, Any]) -> dict[str, list[str]]:
    moved: list[str] = []
    skipped_existing: list[str] = []
    removed_only: list[str] = []

    for flat_key, grouped_key in FLAT_TO_GROUPED.items():
        if flat_key not in section:
            continue
        value = section.pop(flat_key)
        if _has_dotted_key(section, grouped_key):
            skipped_existing.append(flat_key)
            removed_only.append(flat_key)
            continue
        _set_dotted_key(section, grouped_key, value)
        moved.append(flat_key)

    return {
        "moved": sorted(moved),
        "skipped_existing": sorted(skipped_existing),
        "removed_only": sorted(removed_only),
    }


def migrate_pipeline_config(path: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    raw = _load_yaml(path)
    vision_runtime = raw.get("vision_runtime")
    if not isinstance(vision_runtime, dict):
        raise ValueError("missing required section: vision_runtime")
    summary = migrate_vision_runtime_section(vision_runtime)
    return raw, summary


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(_dump_yaml(data), encoding="utf-8")


def _run() -> int:
    parser = argparse.ArgumentParser(description="Migrate vision_runtime flat keys to grouped v2 layout")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML")
    parser.add_argument("--in-place", action="store_true", help="Write changes back to --config")
    parser.add_argument("--output", help="Write migrated YAML to output path")
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak backup when using --in-place")
    args = parser.parse_args()

    if args.in_place and args.output:
        raise SystemExit("--in-place and --output cannot be used together")
    if args.no_backup and not args.in_place:
        raise SystemExit("--no-backup requires --in-place")

    config_path = Path(args.config).expanduser().resolve()
    data, summary = migrate_pipeline_config(config_path)
    touched = len(summary["moved"]) + len(summary["removed_only"])

    print(f"[migrate] config: {config_path}")
    print(f"[migrate] moved_keys={len(summary['moved'])} removed_legacy_keys={len(summary['removed_only'])}")
    if summary["moved"]:
        print("[migrate] moved:", ", ".join(summary["moved"]))
    if summary["skipped_existing"]:
        print("[migrate] skipped_existing:", ", ".join(summary["skipped_existing"]))

    if touched == 0:
        print("[migrate] no legacy flat keys found")
        return 0

    if args.in_place:
        if not args.no_backup:
            backup_path = config_path.with_suffix(config_path.suffix + ".bak")
            backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[migrate] backup written: {backup_path}")
        _write_yaml(config_path, data)
        print("[migrate] migration written in place")
        return 0

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        _write_yaml(output_path, data)
        print(f"[migrate] migration written: {output_path}")
        return 0

    print("[migrate] dry-run complete (no file written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
