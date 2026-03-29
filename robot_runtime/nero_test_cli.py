#!/usr/bin/env python3
"""Terminal tester for NERO real robot using official pyAgxArm SDK."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
THREEPOINT_CONFIG_ROUTE = ["ready", "pick", "transport", "dump"]
THREEPOINT_EXEC_ROUTE = ["ready", "prepick", "pick", "transport", "dump"]
PREPICK_OFFSET_M = 0.05


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


DEFAULT_LOGGING_CFG = {
    "enabled": True,
    "level": "INFO",
    "dir": "./logs",
    "per_run_file": True,
    "filename_prefix": "nero_test",
    "retain_runs": 30,
    "capture_stdout": True,
    "capture_stderr": True,
    "log_commands": True,
}


class LogMirrorStream:
    def __init__(self, stream, logger: logging.Logger, level: int) -> None:
        self.stream = stream
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, data: str) -> int:
        written = self.stream.write(data)
        if data:
            self.buffer += data
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                if line.strip():
                    self.logger.log(self.level, line)
        return written

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


@dataclass
class Waypoint:
    joints: List[float]
    pose: Optional[List[float]] = None
    updated_at: float = 0.0


class WaypointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Waypoint] = {}

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        waypoints = raw.get("waypoints", {})
        loaded: Dict[str, Waypoint] = {}
        for name, item in waypoints.items():
            joints = item.get("joints", [])
            pose = item.get("pose")
            updated_at = float(item.get("updated_at", 0.0))
            if isinstance(joints, list) and len(joints) == 7:
                loaded[name] = Waypoint(
                    joints=[float(v) for v in joints],
                    pose=[float(v) for v in pose] if isinstance(pose, list) and len(pose) == 6 else None,
                    updated_at=updated_at,
                )
        self.data = loaded

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"waypoints": {}}
        for name, wp in self.data.items():
            raw["waypoints"][name] = {
                "joints": wp.joints,
                "pose": wp.pose,
                "updated_at": wp.updated_at,
            }
        self.path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    def set(self, name: str, joints: List[float], pose: Optional[List[float]]) -> None:
        self.data[name] = Waypoint(joints=list(joints), pose=list(pose) if pose else None, updated_at=time.time())

    def get(self, name: str) -> Optional[Waypoint]:
        return self.data.get(name)

    def delete(self, name: str) -> bool:
        if name in self.data:
            del self.data[name]
            return True
        return False

    def names(self) -> List[str]:
        return sorted(self.data.keys())


class RealBackend:
    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.logger = logger if logger is not None else logging.getLogger("nero_test")
        self.robot = None
        self.effector = None
        self.connected = False
        self.speed_percent = int(cfg.get("speed_percent", cfg.get("safety", {}).get("speed_percent_default", 10)))
        self.last_enable_ok = False
        self.last_connect_error = ""
        self.effector_init_error = "not_initialized"
        self.last_joint_feedback_at = 0.0
        self.last_move_error = ""

    def _resolve_effector_symbol(self):
        candidates = [
            ("OPTIONS", "EFFECTOR", "AGX_GRIPPER"),
            ("EFFECTOR", "AGX_GRIPPER"),
        ]
        for path in candidates:
            obj = self.robot
            ok = True
            for seg in path:
                if not hasattr(obj, seg):
                    ok = False
                    break
                obj = getattr(obj, seg)
            if ok:
                return obj
        return None

    def _get_joint_enable_flags(self) -> Optional[List[bool]]:
        if not self.robot or not hasattr(self.robot, "get_joints_enable_status_list"):
            return None
        try:
            raw = self.robot.get_joints_enable_status_list()
            if isinstance(raw, list) and len(raw) == 7:
                return [bool(v) for v in raw]
        except Exception:
            return None
        return None

    def connect(self) -> bool:
        if self.connected:
            return True

        pyagxarm_repo = self.cfg.get("pyagxarm_repo", "")
        if pyagxarm_repo:
            repo = Path(pyagxarm_repo).expanduser().resolve()
            for cand in (repo, repo / "pyAgxArm"):
                c = str(cand)
                if cand.is_dir() and c not in sys.path:
                    sys.path.insert(0, c)

        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except Exception as exc:
            self.last_connect_error = f"pyAgxArm import failed: {exc}"
            self.logger.error(self.last_connect_error)
            return False

        try:
            config = create_agx_arm_config(
                robot="nero",
                comm="can",
                channel=self.cfg.get("can_channel", "can0"),
                interface=self.cfg.get("can_interface", "socketcan"),
                bitrate=int(self.cfg.get("can_bitrate", 1000000)),
            )
            self.robot = AgxArmFactory.create_arm(config)
            self.robot.connect()
            if hasattr(self.robot, "set_normal_mode"):
                self.robot.set_normal_mode()
            self.last_enable_ok = self.enable()
            self.set_speed_percent(self.speed_percent)

            effector_symbol = self._resolve_effector_symbol()
            if effector_symbol is not None and hasattr(self.robot, "init_effector"):
                try:
                    self.effector = self.robot.init_effector(effector_symbol)
                    self.effector_init_error = ""
                except Exception as exc:
                    self.effector_init_error = str(exc)
                    self.logger.warning("effector init failed: %s", self.effector_init_error)

            _ = self.get_joint_positions()
            self.connected = True
            self.last_connect_error = ""
            return True
        except Exception as exc:
            self.last_connect_error = f"Real backend connect failed: {exc}"
            self.logger.exception(self.last_connect_error)
            return False

    def disconnect(self) -> None:
        self.connected = False

    def enable(self) -> bool:
        if not self.robot:
            self.last_enable_ok = False
            return False
        for _ in range(120):
            try:
                if self.robot.enable():
                    self.last_enable_ok = True
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        self.last_enable_ok = False
        return False

    def set_speed_percent(self, percent: int) -> None:
        percent = int(clamp(percent, 1, 100))
        self.speed_percent = percent
        if self.robot and hasattr(self.robot, "set_speed_percent"):
            self.robot.set_speed_percent(percent)

    @staticmethod
    def _max_joint_error(cur: List[float], target: List[float]) -> float:
        return max(abs(cur[i] - target[i]) for i in range(7))

    @staticmethod
    def _angle_abs_diff_rad(a: float, b: float) -> float:
        return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))

    @classmethod
    def _pose_error(cls, cur: List[float], target: List[float]) -> Tuple[float, float]:
        pos_err = math.sqrt(
            (float(cur[0]) - float(target[0])) ** 2
            + (float(cur[1]) - float(target[1])) ** 2
            + (float(cur[2]) - float(target[2])) ** 2
        )
        rot_err = max(
            cls._angle_abs_diff_rad(cur[3], target[3]),
            cls._angle_abs_diff_rad(cur[4], target[4]),
            cls._angle_abs_diff_rad(cur[5], target[5]),
        )
        return pos_err, rot_err

    def _wait_motion_done(
        self,
        timeout: float,
        target: Optional[List[float]] = None,
        start: Optional[List[float]] = None,
    ) -> bool:
        if not self.robot:
            return False

        tol_deg = float(self.cfg.get("move_reach_tolerance_deg", 2.5))
        done_tol_deg = float(self.cfg.get("move_done_tolerance_deg", max(4.0, tol_deg * 1.6)))
        min_progress_deg = float(self.cfg.get("move_min_progress_deg", 0.8))

        tol_rad = math.radians(max(0.1, tol_deg))
        done_tol_rad = math.radians(max(tol_deg, done_tol_deg))
        min_progress_rad = math.radians(max(0.0, min_progress_deg))
        stable_cycles = int(max(1, self.cfg.get("move_reach_stable_cycles", 1)))

        deadline = time.time() + timeout
        stable_hits = 0
        last_err_rad: Optional[float] = None
        last_progress_rad: Optional[float] = None

        while time.time() < deadline:
            status_done = False
            try:
                status = self.robot.get_arm_status()
                if status is not None and hasattr(status, "msg"):
                    motion_status = getattr(status.msg, "motion_status", None)
                    status_done = motion_status == 0
            except Exception:
                pass

            if target is None:
                if status_done:
                    return True
                time.sleep(0.05)
                continue

            cur = self.get_joint_positions()
            if cur is not None:
                last_err_rad = self._max_joint_error(cur, target)
                if start is not None:
                    last_progress_rad = self._max_joint_error(cur, start)

                if last_err_rad <= tol_rad:
                    stable_hits += 1
                    if stable_hits >= stable_cycles:
                        return True
                else:
                    stable_hits = 0

                progress_ok = True
                if start is not None:
                    progress_ok = (last_progress_rad or 0.0) >= min_progress_rad

                # Require observable movement before trusting "motion done",
                # to avoid false-done on drivers that report idle too early.
                if status_done and last_err_rad <= done_tol_rad and progress_ok:
                    return True

            time.sleep(0.05)

        if target is not None and last_err_rad is not None and last_err_rad <= done_tol_rad:
            if start is None:
                return True
            if last_progress_rad is not None and last_progress_rad >= min_progress_rad:
                return True
        return False

    def _wait_pose_done(
        self,
        timeout: float,
        target_pose: List[float],
        start_pose: Optional[List[float]] = None,
    ) -> bool:
        if not self.robot:
            return False

        reach_mm = float(self.cfg.get("move_pose_reach_tolerance_mm", 5.0))
        reach_deg = float(self.cfg.get("move_pose_reach_tolerance_deg", 5.0))
        done_mm = float(self.cfg.get("move_pose_done_tolerance_mm", max(8.0, reach_mm * 1.8)))
        done_deg = float(self.cfg.get("move_pose_done_tolerance_deg", max(8.0, reach_deg * 1.6)))
        min_progress_m = float(self.cfg.get("move_pose_min_progress_m", 0.002))
        min_progress_deg = float(self.cfg.get("move_pose_min_progress_deg", 1.0))
        stable_cycles = int(max(1, self.cfg.get("move_reach_stable_cycles", 1)))

        reach_m = max(0.0005, reach_mm / 1000.0)
        done_m = max(reach_m, done_mm / 1000.0)
        reach_rad = math.radians(max(0.1, reach_deg))
        done_rad = math.radians(max(reach_deg, done_deg))
        min_progress_rad = math.radians(max(0.0, min_progress_deg))

        deadline = time.time() + timeout
        stable_hits = 0
        last_pos_err_m: Optional[float] = None
        last_rot_err_rad: Optional[float] = None
        last_pos_progress_m: Optional[float] = None
        last_rot_progress_rad: Optional[float] = None

        while time.time() < deadline:
            status_done = False
            try:
                status = self.robot.get_arm_status()
                if status is not None and hasattr(status, "msg"):
                    motion_status = getattr(status.msg, "motion_status", None)
                    status_done = motion_status == 0
            except Exception:
                pass

            cur_pose = self.get_flange_pose()
            if cur_pose is not None and len(cur_pose) == 6:
                last_pos_err_m, last_rot_err_rad = self._pose_error(cur_pose, target_pose)
                if start_pose is not None and len(start_pose) == 6:
                    last_pos_progress_m, last_rot_progress_rad = self._pose_error(cur_pose, start_pose)

                if last_pos_err_m <= reach_m and last_rot_err_rad <= reach_rad:
                    stable_hits += 1
                    if stable_hits >= stable_cycles:
                        return True
                else:
                    stable_hits = 0

                progress_ok = True
                if start_pose is not None and len(start_pose) == 6:
                    pos_progress_ok = (last_pos_progress_m or 0.0) >= min_progress_m
                    rot_progress_ok = (last_rot_progress_rad or 0.0) >= min_progress_rad
                    progress_ok = pos_progress_ok or rot_progress_ok

                # Guard against false "done" by requiring pose convergence
                # (or at least observable progress near target).
                if status_done and last_pos_err_m <= done_m and last_rot_err_rad <= done_rad and progress_ok:
                    return True

            time.sleep(0.05)

        if last_pos_err_m is not None and last_rot_err_rad is not None:
            if last_pos_err_m <= done_m and last_rot_err_rad <= done_rad:
                if start_pose is None:
                    return True
                pos_progress_ok = (last_pos_progress_m or 0.0) >= min_progress_m
                rot_progress_ok = (last_rot_progress_rad or 0.0) >= min_progress_rad
                if pos_progress_ok or rot_progress_ok:
                    return True
        return False

    def get_joint_positions(self) -> Optional[List[float]]:
        if not self.robot:
            return None
        data = self.robot.get_joint_angles()
        if data is None or not hasattr(data, "msg"):
            return None
        joints = list(data.msg)
        if len(joints) != 7:
            return None
        self.last_joint_feedback_at = time.time()
        return [float(v) for v in joints]

    def get_flange_pose(self) -> Optional[List[float]]:
        if not self.robot:
            return None
        data = self.robot.get_flange_pose()
        if data is None or not hasattr(data, "msg"):
            return None
        pose = list(data.msg)
        if len(pose) != 6:
            return None
        return [float(v) for v in pose]

    def move_joints(self, joints: List[float], timeout: float) -> bool:
        if not self.robot:
            self.last_move_error = "robot_not_connected"
            self.logger.error("move_joints rejected: robot_not_connected")
            return False

        self.last_move_error = ""
        self.logger.debug("move_joints target_rad=%s timeout=%.2fs", [round(v, 6) for v in joints], timeout)
        start = self.get_joint_positions()
        try:
            self.robot.move_j(joints)
        except Exception as exc:
            self.last_move_error = f"move_j exception: {exc}"
            self.logger.exception("move_j exception for target_rad=%s", [round(v, 6) for v in joints])
            return False

        ok = self._wait_motion_done(timeout, target=joints, start=start)
        if not ok and not self.last_move_error:
            cur = self.get_joint_positions()
            if cur is not None:
                err_deg = math.degrees(self._max_joint_error(cur, joints))
                if start is not None:
                    prog_deg = math.degrees(self._max_joint_error(cur, start))
                    self.last_move_error = (
                        f"motion not settled within {timeout:.1f}s "
                        f"(max_err={err_deg:.2f}deg, progress={prog_deg:.2f}deg)"
                    )
                else:
                    self.last_move_error = f"motion not settled within {timeout:.1f}s (max_err={err_deg:.2f}deg)"
            else:
                self.last_move_error = f"motion not settled within {timeout:.1f}s (no joint feedback)"
        if not ok:
            self.logger.error("move_joints failed: %s", self.last_move_error)
        return ok

    def _move_pose_with_mode(self, pose: List[float], timeout: float, mode: str) -> bool:
        if not self.robot:
            self.last_move_error = "robot_not_connected"
            self.logger.error("move_pose_%s rejected: robot_not_connected", mode)
            return False
        if len(pose) != 6:
            self.last_move_error = f"invalid pose length={len(pose)} for move_{mode}"
            self.logger.error(self.last_move_error)
            return False

        move_fn_name = f"move_{mode}"
        if not hasattr(self.robot, move_fn_name):
            self.last_move_error = f"{move_fn_name} not supported by this SDK version"
            self.logger.error(self.last_move_error)
            return False

        self.last_move_error = ""
        self.logger.debug(
            "move_pose_%s target_m_rad=%s timeout=%.2fs",
            mode,
            [round(v, 6) for v in pose],
            timeout,
        )

        start_pose = self.get_flange_pose()
        try:
            getattr(self.robot, move_fn_name)(pose)
        except Exception as exc:
            self.last_move_error = f"{move_fn_name} exception: {exc}"
            self.logger.exception("%s exception for target_m_rad=%s", move_fn_name, [round(v, 6) for v in pose])
            return False

        ok = self._wait_pose_done(timeout, target_pose=pose, start_pose=start_pose)
        if not ok and not self.last_move_error:
            cur_pose = self.get_flange_pose()
            if cur_pose is not None and len(cur_pose) == 6:
                pos_err_m, rot_err_rad = self._pose_error(cur_pose, pose)
                self.last_move_error = (
                    f"{move_fn_name} motion not settled within {timeout:.1f}s "
                    f"(pos_err={pos_err_m * 1000.0:.1f}mm, rot_err={math.degrees(rot_err_rad):.2f}deg)"
                )
            else:
                self.last_move_error = f"{move_fn_name} motion not settled within {timeout:.1f}s (no flange feedback)"
        if not ok:
            self.logger.error("move_pose_%s failed: %s", mode, self.last_move_error)
        return ok

    def move_pose_p(self, pose: List[float], timeout: float) -> bool:
        return self._move_pose_with_mode(pose, timeout, mode="p")

    def move_pose_l(self, pose: List[float], timeout: float) -> bool:
        return self._move_pose_with_mode(pose, timeout, mode="l")

    def open_gripper(self, width: float, force: float) -> bool:
        if self.effector is None:
            self.logger.error("open_gripper rejected: effector not ready")
            return False
        try:
            self.effector.move_gripper(width=float(width), force=float(force))
            self.logger.info("gripper open request sent width=%.4f force=%.3f", float(width), float(force))
            return True
        except Exception:
            self.logger.exception("open_gripper failed width=%.4f force=%.3f", float(width), float(force))
            return False

    def close_gripper(self, width: float, force: float) -> bool:
        if self.effector is None:
            self.logger.error("close_gripper rejected: effector not ready")
            return False
        try:
            self.effector.move_gripper(width=float(width), force=float(force))
            self.logger.info("gripper close request sent width=%.4f force=%.3f", float(width), float(force))
            return True
        except Exception:
            self.logger.exception("close_gripper failed width=%.4f force=%.3f", float(width), float(force))
            return False

    def estop(self) -> bool:
        if not self.robot:
            return False
        try:
            if hasattr(self.robot, "electronic_emergency_stop"):
                self.robot.electronic_emergency_stop()
                return True
            if hasattr(self.robot, "disable"):
                self.robot.disable()
                return True
            return False
        except Exception:
            return False

    def diagnostics(self) -> Dict[str, str]:
        d: Dict[str, str] = {
            "connected": "yes" if self.connected else "no",
            "enable_ok": "yes" if self.last_enable_ok else "no",
            "speed_percent": str(self.speed_percent),
        }
        flags = self._get_joint_enable_flags()
        if flags is not None:
            d["joint_enable_flags"] = "".join("1" if v else "0" for v in flags)
        if self.last_joint_feedback_at > 0.0:
            age = time.time() - self.last_joint_feedback_at
            d["joint_feedback_alive"] = "yes" if age < 1.0 else f"stale({age:.2f}s)"
        else:
            d["joint_feedback_alive"] = "no_data"
        if self.effector is not None:
            d["effector"] = "ready"
        else:
            d["effector"] = f"not_ready({self.effector_init_error})"
        if self.last_connect_error:
            d["last_connect_error"] = self.last_connect_error
        if self.last_move_error:
            d["last_move_error"] = self.last_move_error
        return d


class NeroArmTester:
    def __init__(self, cfg_path: Path, backend_override: Optional[str]) -> None:
        self.cfg_path = cfg_path
        self.cfg = self._load_cfg(cfg_path)
        if backend_override:
            self.cfg["backend"] = backend_override

        self.logger = logging.getLogger("nero_test")
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self._log_handler: Optional[logging.Handler] = None
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        self._stdout_proxy: Optional[LogMirrorStream] = None
        self._stderr_proxy: Optional[LogMirrorStream] = None
        self.log_path: Optional[Path] = None
        self.log_commands = False
        self._init_logging()

        self.backend_key = str(self.cfg.get("backend", "real")).lower()
        if self.backend_key != "real":
            raise ValueError("This tester supports real backend only")

        self.safety_cfg = self.cfg.get("safety", {}) if isinstance(self.cfg.get("safety", {}), dict) else {}
        self.speed_percent = int(
            self.cfg.get("speed_percent", self.safety_cfg.get("speed_percent_default", 10))
        )
        self.joint_margin = float(self.cfg.get("joint_limits_margin", 0.03))
        self.enforce_joint_limit_clamp = bool(self.safety_cfg.get("enforce_joint_limit_clamp", True))
        self.warn_on_clamp = bool(self.safety_cfg.get("warn_on_clamp", True))
        self.max_joint_step_deg = float(self.safety_cfg.get("max_joint_step_deg", 12.0))
        self.max_joint_step_rad = math.radians(max(1.0, self.max_joint_step_deg))

        self.jog_step = float(self.cfg.get("jog_step_rad", 0.05))
        self.move_timeout = float(self.cfg.get("move_timeout_sec", 20.0))
        self.final_step_confirm = bool(self.cfg.get("final_step_confirm", True))
        self.in_step_prompt = False
        self.threepoint_cfg = self.cfg.get("threepoint", {}) if isinstance(self.cfg.get("threepoint", {}), dict) else {}
        self.strict_down_enabled = bool(self.threepoint_cfg.get("strict_down_enabled", False))
        self.lock_orientation_from = str(self.threepoint_cfg.get("lock_orientation_from", "pick")).strip().lower() or "pick"
        self.transfer_motion = str(self.threepoint_cfg.get("transfer_motion", "p")).strip().lower() or "p"
        self.approach_motion = str(self.threepoint_cfg.get("approach_motion", "l")).strip().lower() or "l"
        self.orientation_tolerance_deg = float(self.threepoint_cfg.get("orientation_tolerance_deg", 5.0))
        self.orientation_tolerance_rad = math.radians(max(0.1, self.orientation_tolerance_deg))
        self.min_joint7_deg = float(self.threepoint_cfg.get("min_joint7_deg", 30.0))
        self.min_joint7_rad = math.radians(self.min_joint7_deg)
        if self.lock_orientation_from in ("scan",):
            self.lock_orientation_from = "ready"
        if self.lock_orientation_from not in THREEPOINT_CONFIG_ROUTE:
            raise ValueError(f"threepoint.lock_orientation_from must be one of {THREEPOINT_CONFIG_ROUTE}")
        if self.transfer_motion not in ("p", "l"):
            raise ValueError("threepoint.transfer_motion must be 'p' or 'l'")
        if self.approach_motion not in ("p", "l"):
            raise ValueError("threepoint.approach_motion must be 'p' or 'l'")

        waypoint_file = Path(self.cfg.get("waypoint_file", cfg_path.parent / "waypoints.yaml"))
        self.store = WaypointStore(waypoint_file)
        self.store.load()

        self.backend = RealBackend(self.cfg, self.logger)
        self.threepoint_info: Dict[str, Dict[str, List[float]]] = {}
        self._sync_threepoint_points()
        self.feedback_csv_interval_sec = self._resolve_feedback_csv_interval()
        self.feedback_csv_path: Optional[Path] = None
        self._feedback_csv_file = None
        self._feedback_csv_writer = None
        self._feedback_csv_thread: Optional[threading.Thread] = None
        self._feedback_csv_stop = threading.Event()

    @staticmethod
    def _load_cfg(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        if isinstance(raw.get("robot_runtime"), dict):
            raw = dict(raw.get("robot_runtime"))

        def resolve_value(value):
            if not isinstance(value, str) or not value.strip():
                return value
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return str(candidate.resolve())
            return str((path.parent / candidate).resolve())

        for key in ("pyagxarm_repo", "waypoint_file"):
            if key in raw:
                raw[key] = resolve_value(raw[key])

        can_tools = raw.get("can_tools")
        if isinstance(can_tools, dict) and "scripts_dir" in can_tools:
            can_tools["scripts_dir"] = resolve_value(can_tools["scripts_dir"])

        return raw

    def _resolve_log_dir(self, raw_dir: str) -> Path:
        candidate = Path(str(raw_dir)).expanduser()
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parent / candidate

    def _resolve_feedback_csv_interval(self) -> float:
        cfg = self.cfg.get("logging", {}) if isinstance(self.cfg.get("logging", {}), dict) else {}
        raw = cfg.get("feedback_csv_interval_sec", 5.0)
        try:
            interval = float(raw)
        except Exception:
            interval = 5.0
        return max(1.0, interval)

    def _resolve_feedback_csv_path(self) -> Path:
        if self.log_path is not None:
            return self.log_path.with_name(f"{self.log_path.stem}_feedback.csv")

        cfg = self.cfg.get("logging", {}) if isinstance(self.cfg.get("logging", {}), dict) else {}
        log_dir = self._resolve_log_dir(str(cfg.get("dir", "./logs")))
        prefix = str(cfg.get("filename_prefix", "nero_test")).strip() or "nero_test"
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        return log_dir / f"{prefix}_{run_id}_feedback.csv"

    @staticmethod
    def _feedback_csv_header() -> List[str]:
        return [
            "timestamp_iso",
            "timestamp_unix_sec",
            "joint1_rad",
            "joint2_rad",
            "joint3_rad",
            "joint4_rad",
            "joint5_rad",
            "joint6_rad",
            "joint7_rad",
            "joint1_deg",
            "joint2_deg",
            "joint3_deg",
            "joint4_deg",
            "joint5_deg",
            "joint6_deg",
            "joint7_deg",
            "flange_x",
            "flange_y",
            "flange_z",
            "flange_rx",
            "flange_ry",
            "flange_rz",
            "connected",
            "joint_feedback_alive",
            "joint_enable_flags",
            "effector",
            "sample_error",
        ]

    @staticmethod
    def _format_float(value: Optional[float], precision: int = 6) -> str:
        if value is None:
            return ""
        return f"{float(value):.{precision}f}"

    def _collect_feedback_csv_row(self, ts: float) -> List[str]:
        joints: Optional[List[float]] = None
        pose: Optional[List[float]] = None
        sample_error = ""

        try:
            joints = self.backend.get_joint_positions()
        except Exception as exc:
            sample_error = f"joint_feedback_error:{exc}"

        try:
            pose = self.backend.get_flange_pose()
        except Exception as exc:
            if sample_error:
                sample_error += ";"
            sample_error += f"flange_feedback_error:{exc}"

        joint_rad = joints if joints is not None and len(joints) == 7 else [None] * 7
        joint_deg = [math.degrees(v) if v is not None else None for v in joint_rad]
        flange = pose if pose is not None and len(pose) == 6 else [None] * 6
        diag = self.backend.diagnostics()

        iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
        row = [
            iso,
            self._format_float(ts, precision=3),
            *(self._format_float(v) for v in joint_rad),
            *(self._format_float(v, precision=3) for v in joint_deg),
            *(self._format_float(v) for v in flange),
            str(diag.get("connected", "")),
            str(diag.get("joint_feedback_alive", "")),
            str(diag.get("joint_enable_flags", "")),
            str(diag.get("effector", "")),
            sample_error,
        ]
        return row

    def _feedback_csv_loop(self) -> None:
        while not self._feedback_csv_stop.wait(self.feedback_csv_interval_sec):
            if self._feedback_csv_writer is None or self._feedback_csv_file is None:
                continue
            try:
                self._feedback_csv_writer.writerow(self._collect_feedback_csv_row(time.time()))
                self._feedback_csv_file.flush()
            except Exception as exc:
                self.logger.warning("feedback csv write failed: %s", exc)

    def _start_feedback_csv_logger(self) -> None:
        if self._feedback_csv_thread is not None:
            return
        try:
            path = self._resolve_feedback_csv_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            need_header = (not path.exists()) or path.stat().st_size == 0

            self._feedback_csv_file = path.open("a", encoding="utf-8", newline="")
            self._feedback_csv_writer = csv.writer(self._feedback_csv_file)
            if need_header:
                self._feedback_csv_writer.writerow(self._feedback_csv_header())
            # Write one initial sample immediately, then continue on interval.
            self._feedback_csv_writer.writerow(self._collect_feedback_csv_row(time.time()))
            self._feedback_csv_file.flush()

            self.feedback_csv_path = path
            self._feedback_csv_stop.clear()
            self._feedback_csv_thread = threading.Thread(
                target=self._feedback_csv_loop,
                name="nero_feedback_csv",
                daemon=True,
            )
            self._feedback_csv_thread.start()
            self.logger.info(
                "feedback csv logger started file=%s interval_sec=%.1f",
                path,
                self.feedback_csv_interval_sec,
            )
        except Exception as exc:
            self.feedback_csv_path = None
            self._feedback_csv_thread = None
            self._feedback_csv_writer = None
            if self._feedback_csv_file is not None:
                try:
                    self._feedback_csv_file.close()
                except Exception:
                    pass
                self._feedback_csv_file = None
            self.logger.warning("failed to start feedback csv logger: %s", exc)

    def _stop_feedback_csv_logger(self) -> None:
        self._feedback_csv_stop.set()
        if self._feedback_csv_thread is not None:
            try:
                self._feedback_csv_thread.join(timeout=max(1.0, self.feedback_csv_interval_sec + 0.5))
            except Exception:
                pass
            self._feedback_csv_thread = None
        self._feedback_csv_writer = None
        if self._feedback_csv_file is not None:
            try:
                self._feedback_csv_file.flush()
                self._feedback_csv_file.close()
            except Exception:
                pass
            self._feedback_csv_file = None

    def _cleanup_old_logs(self, log_dir: Path, prefix: str, retain_runs: int) -> None:
        if retain_runs <= 0:
            return
        files = sorted(
            log_dir.glob(f"{prefix}_*.log"),
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        for stale in files[retain_runs:]:
            if self.log_path is not None and stale == self.log_path:
                continue
            try:
                stale.unlink()
                self.logger.info("removed stale log file: %s", stale)
            except Exception as exc:
                self.logger.warning("failed to remove stale log file %s: %s", stale, exc)

    def _init_logging(self) -> None:
        cfg = self.cfg.get("logging", {}) if isinstance(self.cfg.get("logging", {}), dict) else {}
        merged = dict(DEFAULT_LOGGING_CFG)
        merged.update(cfg)

        enabled = bool(merged.get("enabled", True))
        self.log_commands = bool(merged.get("log_commands", True))
        if not enabled:
            return

        try:
            level_name = str(merged.get("level", "INFO")).upper()
            level = getattr(logging, level_name, logging.INFO)
            self.logger.setLevel(level)

            log_dir = self._resolve_log_dir(str(merged.get("dir", "./logs")))
            log_dir.mkdir(parents=True, exist_ok=True)

            prefix = str(merged.get("filename_prefix", "nero_test")).strip() or "nero_test"
            run_id = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
            log_name = f"{prefix}_{run_id}.log" if bool(merged.get("per_run_file", True)) else f"{prefix}.log"
            self.log_path = log_dir / log_name

            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.handlers.clear()
            self.logger.addHandler(file_handler)
            self._log_handler = file_handler

            if bool(merged.get("capture_stdout", True)):
                self._stdout_proxy = LogMirrorStream(self._stdout_original, self.logger, logging.INFO)
                sys.stdout = self._stdout_proxy
            if bool(merged.get("capture_stderr", True)):
                self._stderr_proxy = LogMirrorStream(self._stderr_original, self.logger, logging.ERROR)
                sys.stderr = self._stderr_proxy

            self.logger.info("logging initialized file=%s level=%s", self.log_path, level_name)
            self._cleanup_old_logs(log_dir, prefix, int(merged.get("retain_runs", 30)))
        except Exception as exc:
            self.log_path = None
            self.log_commands = False
            self.logger.handlers.clear()
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
            try:
                self._stderr_original.write(f"[WARN] logging init failed: {exc}\n")
                self._stderr_original.flush()
            except Exception:
                pass

    def _teardown_logging(self) -> None:
        if self._stdout_proxy is not None:
            self._stdout_proxy.flush()
            sys.stdout = self._stdout_original
            self._stdout_proxy = None
        if self._stderr_proxy is not None:
            self._stderr_proxy.flush()
            sys.stderr = self._stderr_original
            self._stderr_proxy = None
        if self._log_handler is not None:
            self._log_handler.flush()
            self._log_handler.close()
            self.logger.removeHandler(self._log_handler)
            self._log_handler = None

    def shutdown(self) -> None:
        try:
            self.store.save()
        except Exception:
            pass
        try:
            self._stop_feedback_csv_logger()
        except Exception:
            pass
        try:
            self.backend.disconnect()
        except Exception:
            pass
        try:
            self.logger.info("tester shutdown")
        except Exception:
            pass
        try:
            self._teardown_logging()
        except Exception:
            pass

    def _joint_limit(self, idx: int) -> Tuple[float, float]:
        name = JOINT_NAMES[idx]
        limits = self.cfg.get("joint_limits", {})
        if name not in limits:
            raise KeyError(f"joint limit missing for {name}")
        low, high = limits[name]
        return float(low), float(high)

    def _soft_clamp_joint(self, idx: int, val: float) -> float:
        low, high = self._joint_limit(idx)
        return clamp(float(val), low + self.joint_margin, high - self.joint_margin)

    def _normalize_target_joints(self, joints: List[float], name: str) -> List[float]:
        if len(joints) != 7:
            raise ValueError("Expected 7 joints")
        out = list(joints)
        for i in range(7):
            if self.enforce_joint_limit_clamp:
                clamped = self._soft_clamp_joint(i, out[i])
                if self.warn_on_clamp and abs(clamped - out[i]) > 1e-9:
                    print(
                        f"[WARN] {name}: {JOINT_NAMES[i]} target {math.degrees(out[i]):.3f}deg "
                        f"clamped to {math.degrees(clamped):.3f}deg"
                    )
                out[i] = clamped
        self._check_joints_safe(out)
        return out

    def _check_joints_safe(self, joints: List[float]) -> None:
        if len(joints) != 7:
            raise ValueError("Expected 7 joints")
        for i, val in enumerate(joints):
            low, high = self._joint_limit(i)
            low_soft = low + self.joint_margin
            high_soft = high - self.joint_margin
            if val < low_soft or val > high_soft:
                raise ValueError(
                    f"{JOINT_NAMES[i]}={val:.4f} out of soft limit [{low_soft:.4f}, {high_soft:.4f}]"
                )

    def _joint_max_delta(self, a: List[float], b: List[float]) -> float:
        return max(abs(a[i] - b[i]) for i in range(7))

    def _interp(self, a: List[float], b: List[float], t: float) -> List[float]:
        return [a[i] + (b[i] - a[i]) * t for i in range(7)]

    def _move_joints_segmented(self, target: List[float], label: str) -> bool:
        cur = self.backend.get_joint_positions()
        if cur is None:
            print("[ERR] no joint feedback")
            return False

        try:
            tgt = self._normalize_target_joints(target, label)
        except Exception as exc:
            print(f"[ERR] unsafe target for {label}: {exc}")
            return False

        max_delta = self._joint_max_delta(cur, tgt)
        segments = max(1, int(math.ceil(max_delta / max(self.max_joint_step_rad, 1e-6))))

        if segments > 1:
            print(f"[RUN] {label} segmented move: {segments} steps")
        else:
            print(f"[RUN] {label}")

        for step in range(1, segments + 1):
            ratio = step / segments
            step_target = self._interp(cur, tgt, ratio)
            try:
                step_target = self._normalize_target_joints(step_target, f"{label}-seg{step}")
            except Exception as exc:
                print(f"[ERR] segment reject for {label}: {exc}")
                return False
            ok = self.backend.move_joints(step_target, self.move_timeout)
            if not ok:
                diag = self.backend.diagnostics()
                detail = diag.get("last_move_error", "")
                if detail:
                    print(f"[ERR] backend move detail: {detail}")
                print(f"[ERR] move failed at segment {step}/{segments}")
                self.logger.error("move failed label=%s step=%d/%d detail=%s", label, step, segments, detail)
                return False
        return True

    @staticmethod
    def _pose_mm_deg_to_m_rad(pose_mm_deg: List[float]) -> List[float]:
        if len(pose_mm_deg) != 6:
            raise ValueError("pose_mm_deg must contain 6 values")
        return [
            float(pose_mm_deg[0]) / 1000.0,
            float(pose_mm_deg[1]) / 1000.0,
            float(pose_mm_deg[2]) / 1000.0,
            math.radians(float(pose_mm_deg[3])),
            math.radians(float(pose_mm_deg[4])),
            math.radians(float(pose_mm_deg[5])),
        ]

    @staticmethod
    def _angle_abs_diff_rad(a: float, b: float) -> float:
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def _active_pose_key(self) -> str:
        return "locked_pose_m_rad" if self.strict_down_enabled else "pose_m_rad"

    def _check_joint7_target(self, joints: List[float], label: str) -> None:
        if len(joints) != 7:
            raise ValueError(f"{label}: expected 7 joints")
        j7 = float(joints[6])
        if j7 <= self.min_joint7_rad:
            raise ValueError(
                f"{label}: joint7={math.degrees(j7):.3f}deg must be > {self.min_joint7_deg:.3f}deg"
            )

    def _validate_runtime_joint7(self, label: str) -> bool:
        joints = self.backend.get_joint_positions()
        if joints is None or len(joints) != 7:
            print(f"[ERR] {label}: no joint feedback for J7 check")
            self.logger.error("runtime J7 check failed label=%s reason=no_joint_feedback", label)
            return False
        j7 = float(joints[6])
        if j7 <= self.min_joint7_rad:
            print(
                f"[ERR] {label}: joint7={math.degrees(j7):.3f}deg, "
                f"must be > {self.min_joint7_deg:.3f}deg"
            )
            self.logger.error(
                "runtime J7 check failed label=%s joint7_deg=%.3f min_deg=%.3f",
                label,
                math.degrees(j7),
                self.min_joint7_deg,
            )
            return False
        return True

    def _validate_locked_orientation(self, label: str) -> bool:
        if not self.strict_down_enabled:
            return True
        lock_pose = self.threepoint_info.get(self.lock_orientation_from, {}).get("locked_pose_m_rad")
        if not isinstance(lock_pose, list) or len(lock_pose) != 6:
            print(
                "[ERR] strict-down orientation baseline missing "
                f"({self.lock_orientation_from}.locked_pose_m_rad)"
            )
            self.logger.error("strict-down baseline missing")
            return False

        cur_pose = self.backend.get_flange_pose()
        if cur_pose is None:
            print(f"[ERR] {label}: no flange pose feedback for strict-down check")
            self.logger.error("strict-down check failed label=%s reason=no_flange_pose", label)
            return False

        diffs_rad = [
            self._angle_abs_diff_rad(float(cur_pose[3 + i]), float(lock_pose[3 + i]))
            for i in range(3)
        ]
        max_diff_rad = max(diffs_rad)
        if max_diff_rad > self.orientation_tolerance_rad:
            diffs_deg = [math.degrees(v) for v in diffs_rad]
            print(
                f"[ERR] {label}: strict-down orientation violation "
                f"(rx={diffs_deg[0]:.2f}deg ry={diffs_deg[1]:.2f}deg rz={diffs_deg[2]:.2f}deg, "
                f"tol={self.orientation_tolerance_deg:.2f}deg)"
            )
            self.logger.error(
                "strict-down violation label=%s diff_deg=(%.3f, %.3f, %.3f) tol_deg=%.3f",
                label,
                diffs_deg[0],
                diffs_deg[1],
                diffs_deg[2],
                self.orientation_tolerance_deg,
            )
            return False
        return True

    def _execute_pose_motion(self, pose_m_rad: List[float], label: str, motion: str) -> bool:
        if len(pose_m_rad) != 6:
            print(f"[ERR] {label}: pose must have 6 elements")
            return False
        if motion not in ("p", "l"):
            print(f"[ERR] {label}: unsupported pose motion type '{motion}'")
            return False

        print(f"[RUN] {label} (move_{motion})")
        if motion == "p":
            ok = self.backend.move_pose_p(pose_m_rad, self.move_timeout)
        else:
            ok = self.backend.move_pose_l(pose_m_rad, self.move_timeout)
        if not ok:
            diag = self.backend.diagnostics()
            detail = diag.get("last_move_error", "")
            if detail:
                print(f"[ERR] backend move_{motion} detail: {detail}")
            print(f"[ERR] pose move failed: {label}")
            self.logger.error("pose move failed label=%s mode=%s detail=%s", label, motion, detail)
            return False

        if not self._validate_locked_orientation(label):
            return False
        if not self._validate_runtime_joint7(label):
            return False
        return True

    def _move_threepoint_pose(self, name: str, label: str, motion: str) -> bool:
        info = self.threepoint_info.get(name, {})
        pose_key = self._active_pose_key()
        pose = info.get(pose_key)
        if not isinstance(pose, list) or len(pose) != 6:
            print(f"[ERR] threepoint.{name}.{pose_key} missing")
            self.logger.error("threepoint pose missing name=%s key=%s", name, pose_key)
            return False
        return self._execute_pose_motion(pose, label=label, motion=motion)

    def _to_rad_joints_from_deg(self, deg_values: List[float], name: str) -> List[float]:
        if not isinstance(deg_values, list) or len(deg_values) != 7:
            raise ValueError(f"threepoint.{name}.joints_deg must contain 7 values")
        rad = [math.radians(float(v)) for v in deg_values]
        return self._normalize_target_joints(rad, f"threepoint.{name}")

    def _pose_map_to_list(self, raw) -> Optional[List[float]]:
        if not isinstance(raw, dict):
            return None
        keys = ["x", "y", "z", "rx", "ry", "rz"]
        if not all(k in raw for k in keys):
            return None
        return [float(raw[k]) for k in keys]

    def _sync_threepoint_points(self) -> None:
        tp = self.threepoint_cfg
        ready_cfg = tp.get("ready", tp.get("scan", {}))
        pick_cfg = tp.get("pick", {})
        transport_cfg = tp.get("transport", tp.get("dump_pre", {}))
        dump_cfg = tp.get("dump", {})

        ready_deg = ready_cfg.get("joints_deg") if isinstance(ready_cfg, dict) else None
        pick_deg = pick_cfg.get("joints_deg") if isinstance(pick_cfg, dict) else None
        dump_deg = dump_cfg.get("joints_deg") if isinstance(dump_cfg, dict) else None
        if not ready_deg or not pick_deg or not dump_deg:
            raise ValueError("threepoint.ready/pick/dump.joints_deg are required")

        ready_pose_mm_deg = self._pose_map_to_list(ready_cfg.get("pose_mm_deg") if isinstance(ready_cfg, dict) else None)
        pick_pose_mm_deg = self._pose_map_to_list(pick_cfg.get("pose_mm_deg") if isinstance(pick_cfg, dict) else None)
        dump_pose_mm_deg = self._pose_map_to_list(dump_cfg.get("pose_mm_deg") if isinstance(dump_cfg, dict) else None)
        if ready_pose_mm_deg is None:
            raise ValueError("threepoint.ready.pose_mm_deg is required")
        if pick_pose_mm_deg is None:
            raise ValueError("threepoint.pick.pose_mm_deg is required")
        if dump_pose_mm_deg is None:
            raise ValueError("threepoint.dump.pose_mm_deg is required")

        ready_rad = self._to_rad_joints_from_deg(ready_deg, "ready")
        pick_rad = self._to_rad_joints_from_deg(pick_deg, "pick")
        dump_rad = self._to_rad_joints_from_deg(dump_deg, "dump")

        ready_pose_m_rad = self._pose_mm_deg_to_m_rad(ready_pose_mm_deg)
        pick_pose_m_rad = self._pose_mm_deg_to_m_rad(pick_pose_mm_deg)
        dump_pose_m_rad = self._pose_mm_deg_to_m_rad(dump_pose_mm_deg)
        prepick_pose_m_rad = list(pick_pose_m_rad)
        prepick_pose_m_rad[2] = float(prepick_pose_m_rad[2]) + PREPICK_OFFSET_M
        prepick_pose_mm_deg = list(pick_pose_mm_deg)
        prepick_pose_mm_deg[2] = float(prepick_pose_mm_deg[2]) + PREPICK_OFFSET_M * 1000.0

        transport_deg = transport_cfg.get("joints_deg") if isinstance(transport_cfg, dict) else None
        transport_pose_mm_deg = self._pose_map_to_list(
            transport_cfg.get("pose_mm_deg") if isinstance(transport_cfg, dict) else None
        )
        if transport_deg:
            transport_rad = self._to_rad_joints_from_deg(transport_deg, "transport")
        else:
            transport_ratio = clamp(float(tp.get("transport_ratio", tp.get("dump_pre_ratio", 0.50))), 0.05, 0.95)
            transport_rad = self._normalize_target_joints(
                self._interp(ready_rad, dump_rad, transport_ratio),
                "threepoint.transport",
            )
            transport_deg = [math.degrees(v) for v in transport_rad]

        if transport_pose_mm_deg is not None:
            transport_pose_m_rad = self._pose_mm_deg_to_m_rad(transport_pose_mm_deg)
        else:
            transport_ratio = clamp(float(tp.get("transport_ratio", tp.get("dump_pre_ratio", 0.50))), 0.05, 0.95)
            transport_xyz = [
                ready_pose_m_rad[i] + (dump_pose_m_rad[i] - ready_pose_m_rad[i]) * transport_ratio
                for i in range(3)
            ]
            transport_pose_m_rad = [transport_xyz[0], transport_xyz[1], transport_xyz[2], *ready_pose_m_rad[3:6]]
            transport_pose_mm_deg = [
                transport_pose_m_rad[0] * 1000.0,
                transport_pose_m_rad[1] * 1000.0,
                transport_pose_m_rad[2] * 1000.0,
                math.degrees(transport_pose_m_rad[3]),
                math.degrees(transport_pose_m_rad[4]),
                math.degrees(transport_pose_m_rad[5]),
            ]

        lock_pose_m_rad = {
            "ready": ready_pose_m_rad,
            "pick": pick_pose_m_rad,
            "transport": transport_pose_m_rad,
            "dump": dump_pose_m_rad,
        }[self.lock_orientation_from]
        locked_rpy = list(lock_pose_m_rad[3:6])

        def as_locked_pose(pose_m_rad: List[float]) -> List[float]:
            return [pose_m_rad[0], pose_m_rad[1], pose_m_rad[2], *locked_rpy]

        info: Dict[str, Dict[str, List[float]]] = {
            "ready": {
                "joints_rad": ready_rad,
                "joints_deg": [float(v) for v in ready_deg],
                "pose_mm_deg": ready_pose_mm_deg,
                "pose_m_rad": ready_pose_m_rad,
                "locked_pose_m_rad": as_locked_pose(ready_pose_m_rad),
            },
            "prepick": {
                "joints_rad": pick_rad,
                "joints_deg": [float(v) for v in pick_deg],
                "pose_mm_deg": prepick_pose_mm_deg,
                "pose_m_rad": prepick_pose_m_rad,
                "locked_pose_m_rad": as_locked_pose(prepick_pose_m_rad),
            },
            "pick": {
                "joints_rad": pick_rad,
                "joints_deg": [float(v) for v in pick_deg],
                "pose_mm_deg": pick_pose_mm_deg,
                "pose_m_rad": pick_pose_m_rad,
                "locked_pose_m_rad": as_locked_pose(pick_pose_m_rad),
            },
            "transport": {
                "joints_rad": transport_rad,
                "joints_deg": [float(v) for v in transport_deg],
                "pose_mm_deg": transport_pose_mm_deg,
                "pose_m_rad": transport_pose_m_rad,
                "locked_pose_m_rad": as_locked_pose(transport_pose_m_rad),
            },
            "dump": {
                "joints_rad": dump_rad,
                "joints_deg": [float(v) for v in dump_deg],
                "pose_mm_deg": dump_pose_mm_deg,
                "pose_m_rad": dump_pose_m_rad,
                "locked_pose_m_rad": as_locked_pose(dump_pose_m_rad),
            },
        }

        for name in THREEPOINT_CONFIG_ROUTE:
            self._check_joint7_target(info[name]["joints_rad"], f"threepoint.{name}")

        self.threepoint_info = info

        overwrite = bool(tp.get("overwrite_waypoints_on_start", True))
        if overwrite:
            for name in THREEPOINT_CONFIG_ROUTE:
                self.store.set(name, list(info[name]["joints_rad"]), None)
            self.store.save()

    def _print_diag(self) -> None:
        print("backend=real")
        diag = self.backend.diagnostics()
        if not diag:
            print("diag: no data")
            return
        for key in sorted(diag.keys()):
            print(f"{key}={diag[key]}")

    def _print_status(self) -> None:
        joints = self.backend.get_joint_positions()
        pose = self.backend.get_flange_pose()

        print("backend=real")
        print(
            f"speed_percent={self.speed_percent} max_joint_step_deg={self.max_joint_step_deg:.1f} "
            f"joint_margin={self.joint_margin:.3f} min_joint7_deg>{self.min_joint7_deg:.1f}"
        )

        if joints is None:
            print("joints: N/A")
        else:
            print("joints_rad:", " ".join(f"{JOINT_NAMES[i]}={joints[i]:+.3f}" for i in range(7)))
            print("joints_deg:", " ".join(f"{JOINT_NAMES[i]}={math.degrees(joints[i]):+.2f}" for i in range(7)))

        if pose is None:
            print("flange_pose: N/A")
        else:
            print("flange_pose(mm/deg):", " ".join(f"{v:+.3f}" for v in pose))

        names = self.store.names()
        print("waypoints:", ", ".join(names) if names else "none")

    def _show_points(self) -> None:
        if not self.threepoint_info:
            print("[ERR] threepoint config not loaded")
            return

        print("Five-point plan (prepick derived from pick +50mm Z):")
        for name in THREEPOINT_EXEC_ROUTE:
            info = self.threepoint_info[name]
            degs = info["joints_deg"]
            rads = info["joints_rad"]
            print(f"- {name}")
            print("  joints_deg:", "[" + ", ".join(f"{v:.3f}" for v in degs) + "]")
            print("  joints_rad:", "[" + ", ".join(f"{v:.6f}" for v in rads) + "]")
            pose = info.get("pose_mm_deg")
            if pose:
                print(
                    "  pose_mm_deg: "
                    f"x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}, "
                    f"rx={pose[3]:.3f}, ry={pose[4]:.3f}, rz={pose[5]:.3f}"
                )
            locked_pose = info.get("locked_pose_m_rad")
            if isinstance(locked_pose, list) and len(locked_pose) == 6:
                print(
                    "  locked_pose(m/rad): "
                    f"x={locked_pose[0]:.4f}, y={locked_pose[1]:.4f}, z={locked_pose[2]:.4f}, "
                    f"rx={locked_pose[3]:.4f}, ry={locked_pose[4]:.4f}, rz={locked_pose[5]:.4f}"
                )

    def _move_waypoint(self, name: str) -> bool:
        wp = self.store.get(name)
        if wp is None:
            print("[ERR] waypoint not found:", name)
            return False
        return self._move_joints_segmented(wp.joints, f"move {name}")

    def _teach_waypoint(self, name: str, force: bool = False) -> bool:
        del force
        joints = self.backend.get_joint_positions()
        if joints is None:
            print("[ERR] no joint feedback")
            return False
        try:
            joints = self._normalize_target_joints(joints, f"teach.{name}")
        except Exception as exc:
            print(f"[ERR] unsafe teach point: {exc}")
            return False

        pose = self.backend.get_flange_pose()
        self.store.set(name, joints, pose)
        self.store.save()
        print("[OK] waypoint saved:", name)
        return True

    def _jog_joint(self, joint_token: str, delta_token: str) -> bool:
        if not joint_token.startswith("j"):
            print("[ERR] joint format must be j1..j7")
            return False
        try:
            idx = int(joint_token[1:]) - 1
        except Exception:
            print("[ERR] joint format must be j1..j7")
            return False
        if idx < 0 or idx >= 7:
            print("[ERR] joint index out of range")
            return False

        if delta_token == "+":
            delta = self.jog_step
        elif delta_token == "-":
            delta = -self.jog_step
        else:
            try:
                delta = float(delta_token)
            except Exception:
                print("[ERR] delta must be +, -, or float")
                return False

        cur = self.backend.get_joint_positions()
        if cur is None:
            print("[ERR] no joint feedback")
            return False

        target = list(cur)
        target[idx] += delta
        return self._move_joints_segmented(target, f"jog {JOINT_NAMES[idx]}")

    def _run_steps(self, steps: List[Tuple[str, Callable[[], bool]]], step_mode: bool) -> bool:
        for label, action in steps:
            self.logger.info("step start: %s", label)
            if step_mode:
                self.in_step_prompt = True
                try:
                    input(f"[STEP] {label} -> press Enter")
                except KeyboardInterrupt:
                    print("\n[ABORT] interrupted")
                    self.logger.warning("step interrupted by user: %s", label)
                    return False
                finally:
                    self.in_step_prompt = False

            ok = action()
            if not ok:
                print("[ERR] step failed:", label)
                self.logger.error("step failed: %s", label)
                return False
            self.logger.info("step done: %s", label)
        return True

    def _save_runtime_state(self, name: str) -> bool:
        joints = self.backend.get_joint_positions()
        if joints is None:
            print("[ERR] cannot save state: no joint feedback")
            return False
        pose = self.backend.get_flange_pose()
        self.store.set(name, joints, pose)
        self.store.save()
        print(f"[OK] saved state waypoint: {name}")
        return True

    def _run_threepoint_sequence(self, step_mode: bool) -> bool:
        self.logger.info(
            "threepoint sequence start step_mode=%s strict_down=%s transfer=%s approach=%s",
            step_mode,
            self.strict_down_enabled,
            self.transfer_motion,
            self.approach_motion,
        )
        pose_key = self._active_pose_key()
        for name in THREEPOINT_EXEC_ROUTE:
            info = self.threepoint_info.get(name, {})
            pose = info.get(pose_key)
            if not isinstance(pose, list) or len(pose) != 6:
                print(f"[ERR] missing threepoint pose ({pose_key}): {name}")
                return False

        gc = self.cfg.get("gripper", {}) if isinstance(self.cfg.get("gripper", {}), dict) else {}
        open_width = float(gc.get("open_width", 0.05))
        close_width = float(gc.get("close_width", 0.0))
        force = float(gc.get("force", 1.0))

        task = self.cfg.get("task", {}) if isinstance(self.cfg.get("task", {}), dict) else {}
        save_name = str(task.get("save_closed_state_waypoint", "grip_closed_at_pick")).strip() or "grip_closed_at_pick"

        steps: List[Tuple[str, Callable[[], bool]]] = [
            ("move ready", lambda: self._move_threepoint_pose("ready", "move ready", self.transfer_motion)),
            ("open gripper (ready)", lambda: self.backend.open_gripper(open_width, force)),
            ("move prepick", lambda: self._move_threepoint_pose("prepick", "move prepick", "p")),
            ("move pick", lambda: self._move_threepoint_pose("pick", "move pick", "l")),
            ("close gripper", lambda: self.backend.close_gripper(close_width, force)),
            (f"save closed state {save_name}", lambda: self._save_runtime_state(save_name)),
            (
                "move ready (keep gripper closed)",
                lambda: self._move_threepoint_pose("ready", "move ready (keep gripper closed)", self.transfer_motion),
            ),
            (
                "move transport (keep gripper closed)",
                lambda: self._move_threepoint_pose("transport", "move transport (keep gripper closed)", self.transfer_motion),
            ),
            ("move dump", lambda: self._move_threepoint_pose("dump", "move dump", self.approach_motion)),
            ("open gripper (dump)", lambda: self.backend.open_gripper(open_width, force)),
            ("move transport (return)", lambda: self._move_threepoint_pose("transport", "move transport (return)", self.transfer_motion)),
            ("move ready (return)", lambda: self._move_threepoint_pose("ready", "move ready (return)", self.transfer_motion)),
        ]

        ok = self._run_steps(steps, step_mode)
        if ok:
            print("[OK] threepoint sequence completed")
            self.logger.info("threepoint sequence completed")
        else:
            self.logger.error("threepoint sequence failed")
        return ok

    def _test_joints(self) -> bool:
        cfg = self.cfg.get("joint_test", {}) if isinstance(self.cfg.get("joint_test", {}), dict) else {}
        delta_deg = float(cfg.get("delta_deg", 3.0))
        dwell_sec = float(cfg.get("dwell_sec", 0.3))
        joint_tokens = cfg.get("joints", ["j1", "j2", "j3", "j4", "j5", "j6"])

        indices: List[int] = []
        for token in joint_tokens:
            if isinstance(token, str) and re.fullmatch(r"j[1-7]", token.strip().lower()):
                indices.append(int(token.strip().lower()[1:]) - 1)
        if not indices:
            print("[ERR] joint_test.joints is empty")
            return False

        start = self.backend.get_joint_positions()
        if start is None:
            print("[ERR] no joint feedback")
            return False

        delta_rad = math.radians(delta_deg)
        print(f"[RUN] joint test, delta={delta_deg:.2f}deg")

        for idx in indices:
            center = self.backend.get_joint_positions()
            if center is None:
                print("[ERR] joint feedback lost")
                return False

            plus = list(center)
            plus[idx] += delta_rad
            if not self._move_joints_segmented(plus, f"joint test {JOINT_NAMES[idx]} +"):
                return False
            time.sleep(dwell_sec)

            minus = list(center)
            minus[idx] -= delta_rad
            if not self._move_joints_segmented(minus, f"joint test {JOINT_NAMES[idx]} -"):
                return False
            time.sleep(dwell_sec)

            if not self._move_joints_segmented(center, f"joint test {JOINT_NAMES[idx]} center"):
                return False
            time.sleep(dwell_sec)

        if not self._move_joints_segmented(start, "joint test return start"):
            return False

        print("[OK] joint test completed")
        return True

    def _test_gripper(self) -> bool:
        gc = self.cfg.get("gripper", {}) if isinstance(self.cfg.get("gripper", {}), dict) else {}
        open_width = float(gc.get("open_width", 0.05))
        close_width = float(gc.get("close_width", 0.0))
        force = float(gc.get("force", 1.0))
        dwell = float(gc.get("dwell_sec", 0.8))

        print("[RUN] gripper test: open -> close -> open")
        if not self.backend.open_gripper(open_width, force):
            print("[ERR] open gripper failed")
            return False
        time.sleep(dwell)

        if not self.backend.close_gripper(close_width, force):
            print("[ERR] close gripper failed")
            return False
        time.sleep(dwell)

        if not self.backend.open_gripper(open_width, force):
            print("[ERR] open gripper failed")
            return False
        time.sleep(dwell)

        print("[OK] gripper test completed")
        return True

    def _find_can_scripts_dir(self) -> Optional[Path]:
        can_tools = self.cfg.get("can_tools", {}) if isinstance(self.cfg.get("can_tools", {}), dict) else {}
        configured = can_tools.get("scripts_dir")

        candidates: List[Path] = []
        if configured:
            candidates.append(Path(str(configured)).expanduser())

        repo = Path(str(self.cfg.get("pyagxarm_repo", ""))).expanduser()
        if repo:
            candidates.extend(
                [
                    repo / "pyAgxArm" / "scripts" / "ubuntu",
                    repo / "pyAgxArm" / "scripts" / "linux",
                    repo / "scripts" / "ubuntu",
                    repo / "scripts" / "linux",
                ]
            )

        for cand in candidates:
            if cand.is_dir() and (cand / "find_all_can_port.sh").exists() and (cand / "can_activate.sh").exists():
                return cand
        return None

    @staticmethod
    def _run_cmd_output(args: List[str], timeout: float = 2.0) -> Tuple[int, str, str]:
        try:
            out = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return out.returncode, out.stdout.strip(), out.stderr.strip()
        except Exception as exc:
            return 1, "", str(exc)

    def _can_check(self) -> bool:
        script_dir = self._find_can_scripts_dir()
        channel = str(self.cfg.get("can_channel", "can0"))
        bitrate = int(self.cfg.get("can_bitrate", 1000000))

        print(f"can_channel={channel}")
        print(f"can_bitrate={bitrate}")
        if script_dir is None:
            print("[WARN] CAN script dir not found from config")
        else:
            print(f"can_scripts_dir={script_dir}")
            for script in ["find_all_can_port.sh", "can_activate.sh", "can_muti_activate.sh"]:
                print(f"{script}={'yes' if (script_dir / script).exists() else 'no'}")

        rc, out, err = self._run_cmd_output(["ip", "-details", "link", "show", channel], timeout=2.0)
        if rc == 0:
            print("[OK] ip link show", channel)
            for line in out.splitlines()[:8]:
                print("  " + line)
        else:
            print(f"[WARN] ip link show {channel} failed: {err or out or f'rc={rc}'}")

        return True

    def _can_hint_up(self, channel: Optional[str], bitrate: Optional[int], usb_port: Optional[str]) -> None:
        script_dir = self._find_can_scripts_dir()
        if script_dir is None:
            print("[ERR] CAN script dir not found, set can_tools.scripts_dir in config")
            return

        if_name = channel or str(self.cfg.get("can_channel", "can0"))
        bit = int(bitrate if bitrate is not None else int(self.cfg.get("can_bitrate", 1000000)))

        print("Use official scripts:")
        print(f"  cd {script_dir}")
        print("  bash find_all_can_port.sh")
        if usb_port:
            print(f"  bash can_activate.sh {if_name} {bit} \"{usb_port}\"")
        else:
            print(f"  bash can_activate.sh {if_name} {bit}")

    def _precheck(self) -> bool:
        ok = True
        pose_key = self._active_pose_key()

        for name in THREEPOINT_EXEC_ROUTE:
            info = self.threepoint_info.get(name, {})
            pose = info.get(pose_key)
            joints = info.get("joints_rad")
            if not isinstance(pose, list) or len(pose) != 6:
                print(f"[ERR] missing threepoint pose ({pose_key}):", name)
                ok = False
                continue
            if not isinstance(joints, list) or len(joints) != 7:
                print("[ERR] missing threepoint joints:", name)
                ok = False
                continue
            try:
                self._check_joints_safe(joints)
                self._check_joint7_target(joints, f"threepoint.{name}")
            except Exception as exc:
                print(f"[ERR] unsafe waypoint {name}: {exc}")
                ok = False

        diag = self.backend.diagnostics()
        flags = diag.get("joint_enable_flags", "")
        if isinstance(flags, str) and len(flags) == 7 and "0" in flags:
            print(f"[ERR] joint not fully enabled: {flags} (need 1111111)")
            ok = False

        if diag.get("effector", "").startswith("not_ready"):
            print("[WARN] gripper effector not ready")
            ok = False

        if self.in_step_prompt:
            print("[WARN] currently waiting for Enter in step mode")
            ok = False

        if self.final_step_confirm:
            print("[INFO] run threepoint defaults to step mode")

        if ok:
            print("[OK] precheck passed")
        else:
            print("[WARN] precheck found issues")
        return ok

    @staticmethod
    def _print_help() -> None:
        print("Commands:")
        print("  help")
        print("  status")
        print("  precheck")
        print("  diag")
        print("  enable")
        print("  home")
        print("  speed <1-100>")
        print("  jog jN <+|-|delta_rad>")
        print("  teach <name>")
        print("  list wp")
        print("  del wp <name>")
        print("  move <name>")
        print("  open")
        print("  close")
        print("  can check")
        print("  can hint-up [channel] [bitrate] [usb_port]")
        print("  test joints")
        print("  test gripper")
        print("  show points")
        print("  run threepoint [step|auto]")
        print("  run final [step|auto]")
        print("  estop")
        print("  save")
        print("  quit")

    @staticmethod
    def _suggest_command(raw: str, tokens: List[str]) -> Optional[str]:
        compact = re.sub(r"\s+", "", raw.lower())
        m = re.fullmatch(r"speed([0-9]{1,3})", compact)
        if m:
            return f"did you mean: speed {int(m.group(1))}"
        lower = [t.lower() for t in tokens]
        if lower and lower[0] == "jog" and len(tokens) != 3:
            return "jog usage: jog jN <+|-|delta_rad>"
        return None

    def run(self) -> int:
        if self.log_path is not None:
            print(f"log_file={self.log_path}")

        if not self.backend.connect():
            print("[ERR] backend connect failed")
            diag = self.backend.diagnostics()
            if "last_connect_error" in diag:
                print("[ERR]", diag["last_connect_error"])
            self.logger.error("backend connect failed: %s", diag.get("last_connect_error", "unknown"))
            return 1

        self.backend.set_speed_percent(self.speed_percent)
        self._start_feedback_csv_logger()

        print("NERO real-arm tester started")
        print("backend=real")
        print("config=", self.cfg_path)
        if self.feedback_csv_path is not None:
            print(f"feedback_csv_file={self.feedback_csv_path}")
        self._print_help()
        self.logger.info("cli started backend=real config=%s", self.cfg_path)

        while True:
            try:
                raw = input("nero-test> ")
                raw = re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                self.logger.info("cli exit by keyboard/eof")
                break

            if not raw:
                continue
            if self.log_commands:
                self.logger.info("CMD %s", raw)

            compact = re.sub(r"\s+", "", raw.lower())
            compact_speed = re.fullmatch(r"speed([0-9]{1,3})", compact)
            if compact_speed is not None:
                tokens = ["speed", compact_speed.group(1)]
            else:
                try:
                    tokens = shlex.split(raw)
                except Exception as exc:
                    print(f"[ERR] parse error: {exc}")
                    self.logger.warning("parse error for command=%r err=%s", raw, exc)
                    continue

            cmd = tokens[0].lower()

            try:
                if cmd == "help":
                    self._print_help()
                elif cmd == "status":
                    self._print_status()
                elif cmd == "precheck":
                    self._precheck()
                elif cmd == "diag":
                    self._print_diag()
                elif cmd == "enable":
                    print("[OK] enabled" if self.backend.enable() else "[ERR] enable failed")
                elif cmd == "home":
                    joints = [float(v) for v in self.cfg.get("home_joints", [0.0] * 7)]
                    self._move_joints_segmented(joints, "home")
                elif cmd == "speed" and len(tokens) == 2:
                    self.speed_percent = int(clamp(int(tokens[1]), 1, 100))
                    self.backend.set_speed_percent(self.speed_percent)
                    print("[OK] speed_percent=", self.speed_percent)
                elif cmd == "jog" and len(tokens) == 3:
                    self._jog_joint(tokens[1].lower(), tokens[2])
                elif cmd == "teach" and len(tokens) in (2, 3):
                    self._teach_waypoint(tokens[1], force=len(tokens) == 3)
                elif cmd == "list" and len(tokens) == 2 and tokens[1] == "wp":
                    for name in self.store.names():
                        print(f"- {name}")
                elif cmd == "del" and len(tokens) == 3 and tokens[1] == "wp":
                    print("[OK] deleted" if self.store.delete(tokens[2]) else "[ERR] waypoint not found")
                    self.store.save()
                elif cmd == "move" and len(tokens) == 2:
                    self._move_waypoint(tokens[1])
                elif cmd == "open":
                    gc = self.cfg.get("gripper", {})
                    ok = self.backend.open_gripper(float(gc.get("open_width", 0.05)), float(gc.get("force", 1.0)))
                    print("[OK] gripper open" if ok else "[ERR] gripper open failed")
                elif cmd == "close":
                    gc = self.cfg.get("gripper", {})
                    ok = self.backend.close_gripper(float(gc.get("close_width", 0.0)), float(gc.get("force", 1.0)))
                    print("[OK] gripper close" if ok else "[ERR] gripper close failed")
                elif cmd == "can" and len(tokens) >= 2 and tokens[1] == "check":
                    self._can_check()
                elif cmd == "can" and len(tokens) >= 2 and tokens[1] == "hint-up":
                    channel = tokens[2] if len(tokens) >= 3 else None
                    bitrate = int(tokens[3]) if len(tokens) >= 4 else None
                    usb_port = tokens[4] if len(tokens) >= 5 else None
                    self._can_hint_up(channel, bitrate, usb_port)
                elif cmd == "test" and len(tokens) == 2 and tokens[1] == "joints":
                    self._test_joints()
                elif cmd == "test" and len(tokens) == 2 and tokens[1] == "gripper":
                    self._test_gripper()
                elif cmd == "show" and len(tokens) == 2 and tokens[1] == "points":
                    self._show_points()
                elif cmd == "run" and len(tokens) >= 2 and tokens[1] in ("threepoint", "final"):
                    mode = tokens[2].lower() if len(tokens) >= 3 else ("step" if self.final_step_confirm else "auto")
                    step_mode = mode != "auto"
                    self._run_threepoint_sequence(step_mode)
                elif cmd == "estop":
                    print("[OK] estop sent" if self.backend.estop() else "[ERR] estop failed")
                elif cmd == "save":
                    self.store.save()
                    print("[OK] waypoints saved")
                elif cmd in ("quit", "exit"):
                    self.logger.info("cli exit command received: %s", cmd)
                    break
                else:
                    print("[ERR] unknown command, run help")
                    hint = self._suggest_command(raw, tokens)
                    if hint:
                        print(f"[HINT] {hint}")
            except KeyboardInterrupt:
                print("\n[ABORT] command interrupted by Ctrl+C")
                self.logger.warning("command interrupted by keyboard raw=%r tokens=%r", raw, tokens)
            except Exception as exc:
                print(f"[ERR] {exc}")
                self.logger.exception("command failed raw=%r tokens=%r", raw, tokens)

        return 0


def default_config_path() -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    unified = root_dir / "pipeline_config.yaml"
    if unified.is_file():
        return unified
    return Path(__file__).resolve().parent / "config" / "default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="NERO terminal tester")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--backend", choices=["real"], default=None)
    args = parser.parse_args()

    tester = NeroArmTester(Path(args.config).expanduser().resolve(), args.backend)
    try:
        return tester.run()
    finally:
        tester.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
