from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VisionConfig:
    base_url: str
    api_version: str
    poll_hz: float
    health_timeout_sec: float
    target_stable_frames: int
    stable_pos_tol_mm: float
    stable_z_tol_mm: float
    stable_yaw_tol_deg: float
    plan_buffer_frames: int
    max_replan_jump_mm: float
    min_quality_score: float
    reject_on_quality_drop: bool


@dataclass(frozen=True)
class HandEyeConfig:
    mode: str
    extrinsics_path: Path


@dataclass(frozen=True)
class ScanConfig:
    max_offset_xy_m: float
    max_step_xy_m: float
    converge_tol_mm: float
    lost_target_timeout_sec: float


@dataclass(frozen=True)
class GraspConfig:
    open_width_m: float
    close_width_m: float
    force_n: float
    verify_enabled: bool
    prepick_offset_m: float
    final_z_offset_m: float
    max_descent_m: float
    min_safe_z_m: float
    yaw_alignment_offset_deg: float
    verify_width_range_m: tuple[float, float]
    verify_timeout_sec: float


@dataclass(frozen=True)
class RouteConfig:
    ready_from: str
    transport_from: str
    dump_from: str


@dataclass(frozen=True)
class RuntimeConfig:
    arm_config_path: Path
    speed_percent: int
    move_timeout_sec: float
    gripper_dwell_sec: float
    log_dir: Path


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    vision: VisionConfig
    handeye: HandEyeConfig
    scan: ScanConfig
    grasp: GraspConfig
    route: RouteConfig
    runtime: RuntimeConfig


def _require_dict(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    return raw


def _require_section(root: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in root:
        raise ValueError(f"missing required config section: {name}")
    return _require_dict(root[name], name)


def _as_float(section: dict[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {key}: {raw!r}") from exc


def _as_int(section: dict[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid int for {key}: {raw!r}") from exc


def _as_path(config_path: Path, raw: Any, default: str | Path) -> Path:
    value = default if raw is None else raw
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _as_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    raw = section.get(key, default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid bool for {key}: {raw!r}")


def _normalize_width_range(raw: Any) -> tuple[float, float]:
    if isinstance(raw, (int, float)):
        high = float(raw)
        return (0.0, high)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("verify_width_range_m must be a number or a two-item list")
    low = float(raw[0])
    high = float(raw[1])
    if low > high:
        raise ValueError("verify_width_range_m lower bound must be <= upper bound")
    return (low, high)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root_raw = _require_dict(raw, "root")
    using_unified_root = False
    if "dynamic_grasp" in root_raw:
        root = _require_dict(root_raw.get("dynamic_grasp"), "dynamic_grasp")
        using_unified_root = True
    else:
        root = root_raw

    vision = _require_section(root, "vision")
    handeye = _require_section(root, "handeye")
    scan = _require_section(root, "scan")
    grasp = _require_section(root, "grasp")
    route = _require_section(root, "route")
    runtime = _require_section(root, "runtime")
    api_version = str(vision.get("api_version", "v2")).strip().lower() or "v2"
    if api_version != "v2":
        raise ValueError(f"vision.api_version must be v2, got {api_version!r}")
    if "prepick_offset_m" in grasp:
        prepick_offset_m = max(0.0, _as_float(grasp, "prepick_offset_m", 0.05))
    elif "hover_clearance_m" in grasp:
        prepick_offset_m = max(0.0, _as_float(grasp, "hover_clearance_m", 0.05))
    else:
        prepick_offset_m = 0.05

    return AppConfig(
        config_path=config_path,
        vision=VisionConfig(
            base_url=str(vision.get("base_url", "http://127.0.0.1:18000")).rstrip("/"),
            api_version=api_version,
            poll_hz=max(0.5, _as_float(vision, "poll_hz", 12.0)),
            health_timeout_sec=max(0.2, _as_float(vision, "health_timeout_sec", 2.0)),
            target_stable_frames=max(1, _as_int(vision, "target_stable_frames", 3)),
            stable_pos_tol_mm=max(0.1, _as_float(vision, "stable_pos_tol_mm", 5.0)),
            stable_z_tol_mm=max(0.1, _as_float(vision, "stable_z_tol_mm", 5.0)),
            stable_yaw_tol_deg=max(0.1, _as_float(vision, "stable_yaw_tol_deg", 5.0)),
            plan_buffer_frames=max(1, min(10, _as_int(vision, "plan_buffer_frames", 3))),
            max_replan_jump_mm=max(1.0, _as_float(vision, "max_replan_jump_mm", 25.0)),
            min_quality_score=max(0.0, min(1.0, _as_float(vision, "min_quality_score", 0.50))),
            reject_on_quality_drop=_as_bool(vision, "reject_on_quality_drop", True),
        ),
        handeye=HandEyeConfig(
            mode=str(handeye.get("mode", "nominal")).strip().lower() or "nominal",
            extrinsics_path=_as_path(
                config_path,
                handeye.get("extrinsics_path"),
                ROOT_DIR / "模型文件" / "nero_description" / "config" / "handeye_extrinsics.yaml",
            ),
        ),
        scan=ScanConfig(
            max_offset_xy_m=max(0.01, _as_float(scan, "max_offset_xy_m", 0.18)),
            max_step_xy_m=max(0.001, _as_float(scan, "max_step_xy_m", 0.03)),
            converge_tol_mm=max(0.1, _as_float(scan, "converge_tol_mm", 5.0)),
            lost_target_timeout_sec=max(0.1, _as_float(scan, "lost_target_timeout_sec", 1.0)),
        ),
        grasp=GraspConfig(
            open_width_m=max(0.0, _as_float(grasp, "open_width_m", 0.05)),
            close_width_m=max(0.0, _as_float(grasp, "close_width_m", 0.0)),
            force_n=max(0.0, _as_float(grasp, "force_n", 1.0)),
            verify_enabled=_as_bool(grasp, "verify_enabled", False),
            prepick_offset_m=prepick_offset_m,
            final_z_offset_m=_as_float(grasp, "final_z_offset_m", 0.0),
            max_descent_m=max(0.01, _as_float(grasp, "max_descent_m", 0.35)),
            min_safe_z_m=max(0.0, _as_float(grasp, "min_safe_z_m", 0.10)),
            yaw_alignment_offset_deg=_as_float(grasp, "yaw_alignment_offset_deg", 0.0),
            verify_width_range_m=_normalize_width_range(grasp.get("verify_width_range_m", [0.003, 0.08])),
            verify_timeout_sec=max(0.1, _as_float(grasp, "verify_timeout_sec", 1.5)),
        ),
        route=RouteConfig(
            ready_from=str(route.get("ready_from", "threepoint.ready")).strip() or "threepoint.ready",
            transport_from=(
                str(route.get("transport_from", route.get("dump_pre_from", "threepoint.transport"))).strip()
                or "threepoint.transport"
            ),
            dump_from=str(route.get("dump_from", "threepoint.dump")).strip() or "threepoint.dump",
        ),
        runtime=RuntimeConfig(
            arm_config_path=_as_path(
                config_path,
                runtime.get("arm_config_path"),
                (
                    config_path
                    if using_unified_root
                    else ROOT_DIR / "robot_runtime" / "config" / "default.yaml"
                ),
            ),
            speed_percent=max(1, min(100, _as_int(runtime, "speed_percent", 35))),
            move_timeout_sec=max(1.0, _as_float(runtime, "move_timeout_sec", 30.0)),
            gripper_dwell_sec=max(0.0, _as_float(runtime, "gripper_dwell_sec", 0.8)),
            log_dir=_as_path(
                config_path,
                runtime.get("log_dir"),
                ROOT_DIR / "logs",
            ),
        ),
    )
