from __future__ import annotations

import importlib.util
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class StaticRoute:
    ready_flange_pose: tuple[float, float, float, float, float, float]
    transport_flange_pose: tuple[float, float, float, float, float, float]
    dump_flange_pose: tuple[float, float, float, float, float, float]
    baseline_rpy: tuple[float, float, float]


def _load_python_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load python module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    return module


def _load_arm_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"arm config must be a mapping: {path}")
    if isinstance(raw.get("robot_runtime"), dict):
        raw = dict(raw.get("robot_runtime"))
    for key in ("pyagxarm_repo", "waypoint_file"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                raw[key] = str((path.parent / candidate).resolve())
    can_tools = raw.get("can_tools")
    if isinstance(can_tools, dict):
        scripts_dir = can_tools.get("scripts_dir")
        if isinstance(scripts_dir, str) and scripts_dir.strip():
            candidate = Path(scripts_dir).expanduser()
            if not candidate.is_absolute():
                can_tools["scripts_dir"] = str((path.parent / candidate).resolve())
    return raw


def _resolve_nero_cli_module_path(arm_config_path: Path) -> Path:
    arm_config_path = Path(arm_config_path).expanduser().resolve()
    candidates = [
        arm_config_path.parent.parent / "nero_test_cli.py",
        arm_config_path.parent / "robot_runtime" / "nero_test_cli.py",
        arm_config_path.parent / "nero_test_cli.py",
        Path(__file__).resolve().parents[1] / "robot_runtime" / "nero_test_cli.py",
    ]
    seen: set[str] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(resolved)
        if resolved.is_file():
            return resolved

    rendered = "\n".join(f"- {path}" for path in ordered_candidates)
    raise FileNotFoundError(
        "nero_test_cli.py not found for arm config "
        f"{arm_config_path}\nsearched candidates:\n{rendered}"
    )


def _pose_map_to_list(raw: Any) -> list[float]:
    if not isinstance(raw, dict):
        raise ValueError("pose definition must be a mapping")
    keys = ["x", "y", "z", "rx", "ry", "rz"]
    if not all(key in raw for key in keys):
        raise ValueError(f"pose definition must contain keys: {keys}")
    return [float(raw[key]) for key in keys]


def _pose_mm_deg_to_m_rad(pose_mm_deg: list[float]) -> list[float]:
    if len(pose_mm_deg) != 6:
        raise ValueError("pose_mm_deg must contain 6 values")
    return [
        pose_mm_deg[0] / 1000.0,
        pose_mm_deg[1] / 1000.0,
        pose_mm_deg[2] / 1000.0,
        math.radians(pose_mm_deg[3]),
        math.radians(pose_mm_deg[4]),
        math.radians(pose_mm_deg[5]),
    ]


def _get_pose_from_threepoint(threepoint: dict[str, Any], name: str) -> list[float] | None:
    section = threepoint.get(name, {})
    if not isinstance(section, dict):
        return None
    pose_raw = section.get("pose_mm_deg")
    if pose_raw is None:
        return None
    return _pose_mm_deg_to_m_rad(_pose_map_to_list(pose_raw))


def _resolve_threepoint_pose(threepoint: dict[str, Any], keys: list[str], label: str) -> list[float]:
    for key in keys:
        pose = _get_pose_from_threepoint(threepoint, key)
        if pose is not None:
            return pose
    raise ValueError(f"arm config is missing {label}.pose_mm_deg")


def build_static_route(arm_cfg: dict[str, Any]) -> StaticRoute:
    threepoint = arm_cfg.get("threepoint")
    if not isinstance(threepoint, dict):
        raise ValueError("arm config is missing threepoint section")

    ready_pose = _resolve_threepoint_pose(threepoint, ["ready", "scan"], "threepoint.ready (or scan)")
    dump_pose = _resolve_threepoint_pose(threepoint, ["dump"], "threepoint.dump")
    transport_pose = _get_pose_from_threepoint(threepoint, "transport")
    if transport_pose is None:
        # Backward compatibility: reuse explicit dump_pre pose if present, otherwise
        # fall back to interpolating between ready and dump.
        transport_pose = _get_pose_from_threepoint(threepoint, "dump_pre")
    if transport_pose is None:
        transport_ratio = clamp(
            float(threepoint.get("transport_ratio", threepoint.get("dump_pre_ratio", 0.50))),
            0.05,
            0.95,
        )
        transport_pose = [
            ready_pose[idx] + (dump_pose[idx] - ready_pose[idx]) * transport_ratio
            for idx in range(3)
        ] + [float(ready_pose[3]), float(ready_pose[4]), float(ready_pose[5])]

    ready_flange_pose = (
        float(ready_pose[0]),
        float(ready_pose[1]),
        float(ready_pose[2]),
        float(ready_pose[3]),
        float(ready_pose[4]),
        float(ready_pose[5]),
    )
    transport_flange_pose = (
        float(transport_pose[0]),
        float(transport_pose[1]),
        float(transport_pose[2]),
        float(transport_pose[3]),
        float(transport_pose[4]),
        float(transport_pose[5]),
    )
    dump_flange_pose = (
        float(dump_pose[0]),
        float(dump_pose[1]),
        float(dump_pose[2]),
        float(dump_pose[3]),
        float(dump_pose[4]),
        float(dump_pose[5]),
    )
    return StaticRoute(
        ready_flange_pose=ready_flange_pose,
        transport_flange_pose=transport_flange_pose,
        dump_flange_pose=dump_flange_pose,
        baseline_rpy=(float(ready_pose[3]), float(ready_pose[4]), float(ready_pose[5])),
    )


class RobotBridge:
    def __init__(self, arm_config_path: Path, move_timeout_sec: float, logger: logging.Logger) -> None:
        self.arm_config_path = Path(arm_config_path).expanduser().resolve()
        self.arm_cfg = _load_arm_config(self.arm_config_path)
        self.logger = logger
        self._move_timeout_sec = float(move_timeout_sec)
        self._tcp_offset_pose: list[float] | None = None
        module_path = _resolve_nero_cli_module_path(self.arm_config_path)
        self._module = _load_python_module("nero_test_cli_dynamic_bridge", module_path)
        self._backend = self._module.RealBackend(self.arm_cfg, logger)
        self._route = build_static_route(self.arm_cfg)

    @property
    def route(self) -> StaticRoute:
        return self._route

    def resolve_route_pose(self, reference: str) -> list[float]:
        mapping = {
            "ready": self._route.ready_flange_pose,
            "scan": self._route.ready_flange_pose,
            "threepoint.ready": self._route.ready_flange_pose,
            "threepoint.scan": self._route.ready_flange_pose,
            "transport": self._route.transport_flange_pose,
            "dump_pre": self._route.transport_flange_pose,
            "threepoint.transport": self._route.transport_flange_pose,
            "threepoint.dump_pre": self._route.transport_flange_pose,
            "dump": self._route.dump_flange_pose,
            "threepoint.dump": self._route.dump_flange_pose,
        }
        key = str(reference).strip().lower()
        if key not in mapping:
            raise ValueError(f"unsupported route reference: {reference}")
        return [float(v) for v in mapping[key]]

    def connect(self) -> bool:
        return bool(self._backend.connect())

    def shutdown(self) -> None:
        try:
            self._backend.disconnect()
        except Exception:
            self.logger.exception("robot disconnect failed")

    def set_speed_percent(self, percent: int) -> None:
        self._backend.set_speed_percent(int(percent))

    def _require_robot(self) -> Any:
        robot = self._backend.robot
        if robot is None:
            raise RuntimeError("robot driver is not connected")
        return robot

    def set_tcp_offset(self, pose_m_rad: list[float] | tuple[float, ...]) -> None:
        pose = [float(v) for v in pose_m_rad]
        self._require_robot().set_tcp_offset(pose)
        self._tcp_offset_pose = list(pose)

    def get_flange_pose(self) -> list[float] | None:
        return self._backend.get_flange_pose()

    def get_joint_positions(self) -> list[float] | None:
        getter = getattr(self._backend, "get_joint_positions", None)
        if not callable(getter):
            return None
        joints = getter()
        if not isinstance(joints, list) or len(joints) != 7:
            return None
        return [float(v) for v in joints]

    def get_tcp_pose(self) -> list[float] | None:
        robot = self._require_robot()
        if not hasattr(robot, "get_tcp_pose"):
            return None
        data = robot.get_tcp_pose()
        if data is None or not hasattr(data, "msg"):
            return None
        pose = list(data.msg)
        return [float(v) for v in pose] if len(pose) == 6 else None

    def tcp_to_flange_pose(self, tcp_pose_m_rad: list[float] | tuple[float, ...]) -> list[float]:
        robot = self._require_robot()
        if not hasattr(robot, "get_tcp2flange_pose"):
            raise RuntimeError("robot driver does not expose get_tcp2flange_pose")
        pose = robot.get_tcp2flange_pose(list(tcp_pose_m_rad))
        return [float(v) for v in pose]

    def joint_enable_flags(self) -> list[bool] | None:
        getter = getattr(self._backend, "_get_joint_enable_flags", None)
        if callable(getter):
            try:
                flags = getter()
            except Exception:
                flags = None
            if isinstance(flags, list) and len(flags) >= 7:
                return [bool(v) for v in flags[:7]]

        raw = self.diagnostics().get("joint_enable_flags", "")
        if isinstance(raw, str) and len(raw) >= 7 and set(raw[:7]).issubset({"0", "1"}):
            return [char == "1" for char in raw[:7]]
        return None

    def motion_session_health(self) -> tuple[bool, str]:
        diag = self.diagnostics()
        if diag.get("connected") == "no":
            return False, "robot backend disconnected"

        flags = self.joint_enable_flags()
        if flags is None:
            return False, "joint enable flags unavailable"
        if not all(flags):
            rendered = "".join("1" if flag else "0" for flag in flags)
            return False, f"joint_enable_flags={rendered}"

        feedback_state = str(diag.get("joint_feedback_alive", "unknown"))
        if feedback_state != "yes":
            return False, f"joint_feedback_alive={feedback_state}"

        return True, ""

    def recover_enable_session(self) -> bool:
        try:
            robot = self._require_robot()
        except Exception:
            self.logger.exception("motion session recover failed: robot handle unavailable")
            return False

        ok = True
        try:
            if hasattr(robot, "set_normal_mode"):
                robot.set_normal_mode()
        except Exception:
            self.logger.exception("motion session recover failed during set_normal_mode")
            ok = False

        try:
            if not bool(self._backend.enable()):
                self.logger.warning("motion session recover failed: enable returned false")
                ok = False
        except Exception:
            self.logger.exception("motion session recover failed during enable")
            ok = False

        try:
            self._backend.set_speed_percent(self._backend.speed_percent)
        except Exception:
            self.logger.exception("motion session recover failed while restoring speed")
            ok = False

        if self._tcp_offset_pose is not None:
            try:
                robot.set_tcp_offset(list(self._tcp_offset_pose))
            except Exception:
                self.logger.exception("motion session recover failed while restoring tcp offset")
                ok = False

        healthy, reason = self.motion_session_health()
        if not healthy:
            self.logger.warning("motion session still unhealthy after recover: %s", reason)
            return False
        return ok

    def _ensure_motion_session_ready(self, action: str) -> bool:
        healthy, reason = self.motion_session_health()
        if healthy:
            return True

        self.logger.warning("%s blocked by unhealthy motion session: %s", action, reason)
        if not self.recover_enable_session():
            self.logger.error("%s rejected: motion session recovery failed", action)
            return False
        return True

    def move_flange_p(self, flange_pose_m_rad: list[float] | tuple[float, ...]) -> bool:
        if not self._ensure_motion_session_ready("move_flange_p"):
            return False
        return bool(self._backend.move_pose_p(list(flange_pose_m_rad), self._move_timeout_sec))

    def move_flange_l(self, flange_pose_m_rad: list[float] | tuple[float, ...]) -> bool:
        if not self._ensure_motion_session_ready("move_flange_l"):
            return False
        return bool(self._backend.move_pose_l(list(flange_pose_m_rad), self._move_timeout_sec))

    def move_tcp_p(self, tcp_pose_m_rad: list[float] | tuple[float, ...]) -> bool:
        return self.move_flange_p(self.tcp_to_flange_pose(tcp_pose_m_rad))

    def move_tcp_l(self, tcp_pose_m_rad: list[float] | tuple[float, ...]) -> bool:
        return self.move_flange_l(self.tcp_to_flange_pose(tcp_pose_m_rad))

    def open_gripper(self, width_m: float, force_n: float) -> bool:
        return bool(self._backend.open_gripper(width_m, force_n))

    def close_gripper(self, width_m: float, force_n: float) -> bool:
        return bool(self._backend.close_gripper(width_m, force_n))

    def get_gripper_status(self) -> Any | None:
        effector = self._backend.effector
        return None if effector is None else effector.get_gripper_status()

    def get_gripper_ctrl_states(self) -> Any | None:
        effector = self._backend.effector
        return None if effector is None else effector.get_gripper_ctrl_states()

    def gripper_is_ok(self) -> bool:
        effector = self._backend.effector
        if effector is None:
            return False
        try:
            return bool(effector.is_ok())
        except Exception:
            return False

    def gripper_fps(self) -> float:
        effector = self._backend.effector
        if effector is None:
            return 0.0
        try:
            return float(effector.get_fps())
        except Exception:
            return 0.0

    def diagnostics(self) -> dict[str, str]:
        return self._backend.diagnostics()

    def estop(self) -> bool:
        return bool(self._backend.estop())

    def recover_after_estop(self, settle_sec: float = 0.3) -> bool:
        try:
            robot = self._require_robot()
        except Exception:
            self.logger.exception("robot recover failed: robot handle unavailable")
            return False

        ok = True
        if settle_sec > 0.0:
            time.sleep(float(settle_sec))

        try:
            if hasattr(robot, "reset"):
                robot.reset()
        except Exception:
            self.logger.exception("robot recover failed during reset")
            ok = False

        time.sleep(0.1)

        try:
            if hasattr(robot, "set_normal_mode"):
                robot.set_normal_mode()
        except Exception:
            self.logger.exception("robot recover failed during set_normal_mode")
            ok = False

        try:
            if not bool(self._backend.enable()):
                self.logger.warning("robot recover failed: enable returned false")
                ok = False
        except Exception:
            self.logger.exception("robot recover failed during enable")
            ok = False

        try:
            self._backend.set_speed_percent(self._backend.speed_percent)
        except Exception:
            self.logger.exception("robot recover failed while restoring speed")
            ok = False

        if self._tcp_offset_pose is not None:
            try:
                robot.set_tcp_offset(list(self._tcp_offset_pose))
            except Exception:
                self.logger.exception("robot recover failed while restoring tcp offset")
                ok = False

        return ok
