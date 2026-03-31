from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


ALLOWED_COMMANDS = {
    "status",
    "precheck",
    "enable",
    "enable_gravity_compensation",
    "disable_gravity_compensation",
    "disable",
    "reset_clear_errors",
    "set_speed_percent",
    "home",
    "open",
    "close",
    "show_points",
    "run_threepoint_auto",
    "run_threepoint_step_start",
    "run_threepoint_step_next",
    "run_threepoint_step_cancel",
    "apply_pick_override_from_vision",
    "clear_pick_override",
    "estop",
}

CTRL_MODE_LABELS: dict[int, str] = {
    0x00: "待机模式",
    0x01: "CAN 指令控制模式",
    0x02: "示教模式",
    0x03: "以太网控制模式",
    0x04: "Wi-Fi 控制模式",
    0x05: "遥控器控制模式",
    0x06: "联动示教输入模式",
    0x07: "离线轨迹模式",
    0x08: "TCP 控制模式",
}

GRAVITY_COMP_CTRL_MODE = 0x06
GRAVITY_INTERLOCKED_COMMANDS = {
    "home",
    "run_threepoint_auto",
    "run_threepoint_step_start",
    "run_threepoint_step_next",
}

DAEMON_INTERLOCKED_COMMANDS = {
    "home",
    "run_threepoint_auto",
    "run_threepoint_step_start",
    "run_threepoint_step_next",
}

ROBOT_EXCLUSIVE_ENV = "DABAI_ROBOT_EXCLUSIVE_CONTROL"
AUTO_ENABLE_SERVICE_NAME = "nero-auto-enable.service"


class RobotControlManager:
    """Web-safe robot control adapter built on top of NeroArmTester."""

    def __init__(
        self,
        arm_config_path: Path,
        backend_override: str | None,
        enabled: bool,
        tester_factory: Callable[[Path, str | None], Any] | None = None,
    ) -> None:
        self._arm_config_path = Path(arm_config_path).expanduser().resolve()
        self._backend_override = backend_override
        self._enabled = bool(enabled)
        self._tester_factory = tester_factory

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._tester: Any | None = None
        self._busy = False
        self._step_session: dict[str, Any] | None = None
        self._speed_percent: int | None = None
        self._pick_override: dict[str, Any] | None = None
        self._ctrl_mode_hint: int | None = None
        self._handeye_mode_required = "calibrated"
        (
            self._flange_to_optical,
            self._handeye_source_effective,
            self._handeye_error,
        ) = self._load_handeye_flange_to_optical()
        self._handeye_tcp_offset_m_rad: list[float] | None = self._load_handeye_tcp_offset_m_rad()
        self._pick_override_safety = self._load_pick_override_safety_config(self._arm_config_path)
        self._dynamic_stability_config = self._load_dynamic_stability_config(self._arm_config_path)
        self._last_result: dict[str, Any] = {
            "ok": False,
            "command": "",
            "message": "idle",
            "stdout_lines": [],
            "step_progress": self._render_step_progress_unlocked(),
            "error_code": None,
            "diag": {},
        }

    def shutdown(self) -> None:
        with self._state_lock:
            tester = self._tester
            self._tester = None
            self._step_session = None
            self._pick_override = None
        if tester is None:
            return
        with contextlib.suppress(Exception):
            tester.shutdown()

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            tester = self._tester
            busy = self._busy
            last_result = dict(self._last_result)
            step_progress = self._render_step_progress_unlocked()
            pick_override_meta = self._render_pick_override_meta_unlocked()
            speed_percent = self._speed_percent
            ctrl_mode_hint = self._ctrl_mode_hint

        joints: list[float] | None = None
        pose: list[float] | None = None
        diag: dict[str, str] = {}
        connected = False
        ctrl_mode: int | None = None
        ctrl_mode_inferred = False

        if tester is not None:
            with contextlib.suppress(Exception):
                diag = dict(tester.backend.diagnostics())
            with contextlib.suppress(Exception):
                joints = tester.backend.get_joint_positions()
            with contextlib.suppress(Exception):
                pose = tester.backend.get_flange_pose()
            with contextlib.suppress(Exception):
                speed_percent = int(getattr(tester, "speed_percent", 0))
            connected = bool(getattr(tester.backend, "connected", False))
            ctrl_mode_feedback = self._read_ctrl_mode_from_robot(tester)
            if ctrl_mode_feedback is None:
                if ctrl_mode_hint is not None:
                    ctrl_mode = int(ctrl_mode_hint)
                    ctrl_mode_inferred = True
            elif int(ctrl_mode_feedback) == GRAVITY_COMP_CTRL_MODE:
                ctrl_mode = int(ctrl_mode_feedback)
                with self._state_lock:
                    self._ctrl_mode_hint = int(ctrl_mode_feedback)
            elif ctrl_mode_hint == GRAVITY_COMP_CTRL_MODE:
                # Some firmware still reports CAN mode while leader drag is active.
                ctrl_mode = int(ctrl_mode_hint)
                ctrl_mode_inferred = True
            else:
                ctrl_mode = int(ctrl_mode_feedback)
                with self._state_lock:
                    self._ctrl_mode_hint = int(ctrl_mode_feedback)
        elif ctrl_mode_hint is not None:
            ctrl_mode = int(ctrl_mode_hint)
            ctrl_mode_inferred = True

        gravity_comp_active = bool(ctrl_mode == GRAVITY_COMP_CTRL_MODE)
        ctrl_mode_label = self._format_ctrl_mode_label(ctrl_mode, inferred=ctrl_mode_inferred)
        handeye_error = self._handeye_error
        if self._handeye_tcp_offset_m_rad is None:
            handeye_error = handeye_error or "handeye tcp offset unavailable"
        handeye_ready = self._flange_to_optical is not None and self._handeye_tcp_offset_m_rad is not None and handeye_error is None
        handeye_source_effective = str(self._handeye_source_effective if self._flange_to_optical is not None else "unavailable")

        return {
            "enabled": self._enabled,
            "connected": connected,
            "busy": busy,
            "speed_percent": speed_percent,
            "joint_positions_rad": joints,
            "flange_pose_m_rad": pose,
            "diag": diag,
            "step_progress": step_progress,
            "last_result": last_result,
            "pick_override_active": bool(pick_override_meta["active"]),
            "pick_override_mode": str(pick_override_meta["mode"]),
            "pick_override_close_width": pick_override_meta["close_width"],
            "pick_override_force": pick_override_meta["force"],
            "pick_override_smooth_segments": pick_override_meta["smooth_segments"],
            "dynamic_stability_config": {
                "window": int(self._dynamic_stability_config["window"]),
                "min_axis_quality": float(self._dynamic_stability_config["min_axis_quality"]),
                "mad_limit": {
                    "x": float(self._dynamic_stability_config["mad_limit"]["x"]),
                    "y": float(self._dynamic_stability_config["mad_limit"]["y"]),
                    "z": float(self._dynamic_stability_config["mad_limit"]["z"]),
                    "yaw": float(self._dynamic_stability_config["mad_limit"]["yaw"]),
                },
            },
            "gravity_compensation_active": gravity_comp_active,
            "ctrl_mode": ctrl_mode,
            "ctrl_mode_label": ctrl_mode_label,
            "handeye_mode_required": str(self._handeye_mode_required),
            "handeye_source_effective": handeye_source_effective,
            "handeye_ready": bool(handeye_ready),
            "handeye_error": handeye_error,
        }

    def execute_command(self, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        name = str(command).strip().lower()
        if name not in ALLOWED_COMMANDS:
            return self._build_result(False, name, "unsupported command", [], self._safe_diag(), error_code="invalid_params")

        if not self._enabled:
            return self._build_result(False, name, "robot web control disabled", [], self._safe_diag())

        if name == "estop":
            return self._execute_estop_anytime(name)
        if name == "disable":
            return self._execute_disable_anytime(name)
        if name == "reset_clear_errors":
            return self._execute_reset_clear_anytime(name)

        if not self._command_lock.acquire(blocking=False):
            return self._build_result(False, name, "robot is busy", [], self._safe_diag())

        with self._state_lock:
            self._busy = True

        try:
            result = self._execute_command_locked(name, params)
        finally:
            with self._state_lock:
                self._busy = False
            self._command_lock.release()

        with self._state_lock:
            self._last_result = dict(result)
        return result

    def _execute_estop_anytime(self, command: str) -> dict[str, Any]:
        # If another long command is running, we still try to send estop immediately.
        acquired = self._command_lock.acquire(blocking=False)
        if acquired:
            with self._state_lock:
                self._busy = True
            try:
                result = self._execute_estop_locked(command, forced=False)
            finally:
                with self._state_lock:
                    self._busy = False
                self._command_lock.release()
        else:
            result = self._execute_estop_locked(command, forced=True)

        with self._state_lock:
            self._last_result = dict(result)
        return result

    def _execute_disable_anytime(self, command: str) -> dict[str, Any]:
        acquired = self._command_lock.acquire(blocking=False)
        if acquired:
            with self._state_lock:
                self._busy = True
            try:
                result = self._execute_disable_locked(command, forced=False)
            finally:
                with self._state_lock:
                    self._busy = False
                self._command_lock.release()
        else:
            result = self._execute_disable_locked(command, forced=True)

        with self._state_lock:
            self._last_result = dict(result)
        return result

    def _execute_reset_clear_anytime(self, command: str) -> dict[str, Any]:
        acquired = self._command_lock.acquire(blocking=False)
        if acquired:
            with self._state_lock:
                self._busy = True
            try:
                result = self._execute_reset_clear_locked(command, forced=False)
            finally:
                with self._state_lock:
                    self._busy = False
                self._command_lock.release()
        else:
            result = self._execute_reset_clear_locked(command, forced=True)

        with self._state_lock:
            self._last_result = dict(result)
        return result

    def _execute_command_locked(self, command: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if command in DAEMON_INTERLOCKED_COMMANDS:
            conflict_message = self._get_daemon_conflict_message()
            if conflict_message is not None:
                return self._build_result(
                    False,
                    command,
                    conflict_message,
                    [],
                    self._safe_diag(),
                    error_code="daemon_conflict",
                )

        try:
            tester = self._ensure_tester()
        except Exception as exc:
            return self._build_result(False, command, f"connect failed: {exc}", [], self._safe_diag())

        if command in GRAVITY_INTERLOCKED_COMMANDS and self._is_gravity_compensation_active(tester):
            return self._build_result(
                False,
                command,
                "gravity compensation is active; disable it before motion commands",
                [],
                self._safe_diag(tester),
                error_code="gravity_interlock",
            )

        if command == "status":
            ok, lines = self._capture_output(lambda: tester._print_status())
            return self._build_result(ok, command, "status updated", lines, self._safe_diag(tester))

        if command == "precheck":
            ok, lines = self._capture_output(lambda: bool(tester._precheck()))
            message = "precheck passed" if ok else "precheck failed"
            return self._build_result(ok, command, message, lines, self._safe_diag(tester))

        if command == "enable":
            ok = bool(tester.backend.enable())
            message = "enabled" if ok else "enable failed"
            return self._build_result(ok, command, message, [], self._safe_diag(tester))

        if command == "enable_gravity_compensation":
            robot = getattr(tester.backend, "robot", None)
            setter = getattr(robot, "set_leader_mode", None) if robot is not None else None
            if not callable(setter):
                return self._build_result(
                    False,
                    command,
                    "gravity compensation unsupported by backend",
                    [],
                    self._safe_diag(tester),
                )
            try:
                setter()
            except Exception as exc:
                return self._build_result(
                    False,
                    command,
                    f"enable gravity compensation failed: {exc}",
                    [],
                    self._safe_diag(tester),
                )
            with self._state_lock:
                self._ctrl_mode_hint = GRAVITY_COMP_CTRL_MODE
            return self._build_result(True, command, "gravity compensation enabled", [], self._safe_diag(tester))

        if command == "disable_gravity_compensation":
            robot = getattr(tester.backend, "robot", None)
            setter = getattr(robot, "set_normal_mode", None) if robot is not None else None
            if not callable(setter):
                return self._build_result(
                    False,
                    command,
                    "normal mode restore unsupported by backend",
                    [],
                    self._safe_diag(tester),
                )
            try:
                setter()
            except Exception as exc:
                return self._build_result(
                    False,
                    command,
                    f"restore normal mode failed: {exc}",
                    [],
                    self._safe_diag(tester),
                )
            with self._state_lock:
                self._ctrl_mode_hint = 0x01
            return self._build_result(True, command, "normal mode restored", [], self._safe_diag(tester))

        if command == "set_speed_percent":
            parsed_speed, err = self._parse_speed_percent(params)
            if err is not None:
                return self._build_result(False, command, err, [], self._safe_diag(tester), error_code="invalid_params")
            tester.speed_percent = int(parsed_speed)
            with contextlib.suppress(Exception):
                tester.backend.set_speed_percent(int(parsed_speed))
            with self._state_lock:
                self._speed_percent = int(parsed_speed)
            return self._build_result(True, command, f"speed set to {parsed_speed}", [], self._safe_diag(tester))

        if command == "home":
            joints = [float(v) for v in tester.cfg.get("home_joints", [0.0] * 7)]
            ok, lines = self._capture_output(lambda: bool(tester._move_joints_segmented(joints, "home")))
            message = "home done" if ok else "home failed"
            return self._build_result(ok, command, message, lines, self._safe_diag(tester))

        if command == "open":
            gc = tester.cfg.get("gripper", {}) if isinstance(tester.cfg.get("gripper", {}), dict) else {}
            ok = bool(tester.backend.open_gripper(float(gc.get("open_width", 0.05)), float(gc.get("force", 1.0))))
            message = "gripper open" if ok else "gripper open failed"
            return self._build_result(ok, command, message, [], self._safe_diag(tester))

        if command == "close":
            gc = tester.cfg.get("gripper", {}) if isinstance(tester.cfg.get("gripper", {}), dict) else {}
            ok = bool(tester.backend.close_gripper(float(gc.get("close_width", 0.0)), float(gc.get("force", 1.0))))
            message = "gripper close" if ok else "gripper close failed"
            return self._build_result(ok, command, message, [], self._safe_diag(tester))

        if command == "show_points":
            ok, lines = self._capture_output(lambda: tester._show_points())
            return self._build_result(ok, command, "points listed", lines, self._safe_diag(tester))

        if command == "run_threepoint_auto":
            try:
                steps = self._build_threepoint_steps(tester)
            except Exception as exc:
                return self._build_result(
                    False,
                    command,
                    f"threepoint auto init failed: {exc}",
                    [],
                    self._safe_diag(tester),
                    error_code="invalid_params",
                )

            runner = getattr(tester, "_run_steps", None)
            if callable(runner):
                ok, lines = self._capture_output(lambda: bool(runner(steps, False)))
            else:
                ok, lines = self._capture_output(lambda: bool(tester._run_threepoint_sequence(step_mode=False)))
            with self._state_lock:
                self._step_session = None
                self._clear_pick_override_unlocked(tester)
            message = "threepoint auto completed" if ok else "threepoint auto failed"
            return self._build_result(ok, command, message, lines, self._safe_diag(tester))

        if command == "run_threepoint_step_start":
            session_active = False
            with self._state_lock:
                if self._step_session is not None and bool(self._step_session.get("active", False)):
                    session_active = True
            if session_active:
                return self._build_result(False, command, "step session already active", [], self._safe_diag(tester))

            try:
                steps = self._build_threepoint_steps(tester)
            except Exception as exc:
                return self._build_result(False, command, f"step session init failed: {exc}", [], self._safe_diag(tester))

            with self._state_lock:
                self._step_session = {
                    "active": True,
                    "index": 0,
                    "total": len(steps),
                    "last_label": None,
                    "steps": steps,
                    "created_at": time.time(),
                }
            return self._build_result(True, command, "step session started", [], self._safe_diag(tester))

        if command == "run_threepoint_step_next":
            with self._state_lock:
                session = self._step_session
            if session is None or not bool(session.get("active", False)):
                return self._build_result(False, command, "no active step session", [], self._safe_diag(tester))

            index = int(session.get("index", 0))
            steps = session.get("steps")
            if not isinstance(steps, list) or index < 0 or index >= len(steps):
                with self._state_lock:
                    self._step_session = None
                    self._clear_pick_override_unlocked(tester)
                return self._build_result(False, command, "step session is invalid", [], self._safe_diag(tester))

            label, action = steps[index]
            ok, lines = self._capture_output(lambda: bool(action()))
            clear_override = False

            with self._state_lock:
                live = self._step_session
                if live is not None:
                    live["last_label"] = label
                    if ok:
                        live["index"] = index + 1
                        if int(live["index"]) >= int(live.get("total", 0)):
                            live["active"] = False
                            clear_override = True
                    else:
                        live["active"] = False
                        clear_override = True
                if clear_override:
                    self._clear_pick_override_unlocked(tester)
            message = f"step done: {label}" if ok else f"step failed: {label}"
            return self._build_result(ok, command, message, lines, self._safe_diag(tester))

        if command == "run_threepoint_step_cancel":
            has_session = False
            with self._state_lock:
                has_session = self._step_session is not None
                if has_session:
                    self._step_session = None
                    self._clear_pick_override_unlocked(tester)
            if not has_session:
                return self._build_result(False, command, "no active step session", [], self._safe_diag(tester))
            return self._build_result(True, command, "step session cancelled", [], self._safe_diag(tester))

        if command == "apply_pick_override_from_vision":
            parsed, err = self._parse_pick_override_params(params)
            if err is not None:
                return self._build_result(False, command, err, [], self._safe_diag(tester), error_code="invalid_params")
            ok, message = self._apply_pick_override_from_vision(tester, parsed)
            return self._build_result(ok, command, message, [], self._safe_diag(tester))

        if command == "clear_pick_override":
            with self._state_lock:
                cleared = self._clear_pick_override_unlocked(tester)
            if not cleared:
                return self._build_result(False, command, "pick override is not active", [], self._safe_diag(tester))
            return self._build_result(True, command, "pick override cleared", [], self._safe_diag(tester))

        return self._build_result(False, command, "unsupported command", [], self._safe_diag(tester))

    def _execute_estop_locked(self, command: str, forced: bool) -> dict[str, Any]:
        tester = self._peek_tester()
        if tester is None:
            return self._build_result(False, command, "robot session not initialized", [], {})

        ok = False
        with contextlib.suppress(Exception):
            ok = bool(tester.backend.estop())

        with self._state_lock:
            self._step_session = None
            self._clear_pick_override_unlocked(tester)
            self._ctrl_mode_hint = None

        msg = "estop sent"
        if forced:
            msg = "estop sent (forced while busy)"
        if not ok:
            msg = "estop failed"
        return self._build_result(ok, command, msg, [], self._safe_diag(tester))

    def _execute_disable_locked(self, command: str, forced: bool) -> dict[str, Any]:
        tester = self._peek_tester()
        if tester is None:
            try:
                tester = self._ensure_tester()
            except Exception as exc:
                return self._build_result(False, command, f"connect failed: {exc}", [], self._safe_diag())

        ok = False
        robot = getattr(tester.backend, "robot", None)
        if robot is not None and hasattr(robot, "disable"):
            try:
                result = robot.disable(255)
                ok = bool(result) if result is not None else True
            except Exception:
                ok = False
        if not ok and hasattr(tester.backend, "estop"):
            with contextlib.suppress(Exception):
                ok = bool(tester.backend.estop())

        with self._state_lock:
            self._step_session = None
            self._clear_pick_override_unlocked(tester)
            self._ctrl_mode_hint = None

        msg = "robot disabled"
        if forced:
            msg = "robot disabled (forced while busy)"
        if not ok:
            msg = "robot disable failed"
        return self._build_result(ok, command, msg, [], self._safe_diag(tester))

    def _execute_reset_clear_locked(self, command: str, forced: bool) -> dict[str, Any]:
        tester = self._peek_tester()
        if tester is None:
            try:
                tester = self._ensure_tester()
            except Exception as exc:
                return self._build_result(False, command, f"connect failed: {exc}", [], self._safe_diag())

        lines: list[str] = []
        ok = True
        robot = getattr(tester.backend, "robot", None)
        reset_fn = getattr(robot, "reset", None) if robot is not None else None
        if callable(reset_fn):
            try:
                reset_fn()
                lines.append("reset command sent")
            except Exception as exc:
                ok = False
                lines.append(f"reset failed: {exc}")
        else:
            ok = False
            lines.append("reset unsupported by backend")

        if ok:
            time.sleep(0.15)
            try:
                enabled = bool(tester.backend.enable())
            except Exception as exc:
                enabled = False
                lines.append(f"re-enable failed: {exc}")
            if enabled:
                lines.append("re-enable success")
            else:
                ok = False
                if not any(line.startswith("re-enable failed:") for line in lines):
                    lines.append("re-enable failed")

        speed_percent = int(getattr(tester, "speed_percent", 0))
        if speed_percent < 1:
            speed_percent = 1
        with contextlib.suppress(Exception):
            tester.backend.set_speed_percent(speed_percent)
            lines.append(f"speed restored to {speed_percent}%")

        with self._state_lock:
            self._step_session = None
            self._clear_pick_override_unlocked(tester)
            self._speed_percent = speed_percent
            self._ctrl_mode_hint = None

        msg = "reset+clear completed" if ok else "reset+clear failed"
        if forced:
            msg = f"{msg} (forced while busy)"
        return self._build_result(ok, command, msg, lines, self._safe_diag(tester))

    def _capture_output(self, func: Callable[[], Any]) -> tuple[bool, list[str]]:
        buf = io.StringIO()
        ok = True
        with contextlib.redirect_stdout(buf):
            try:
                ret = func()
                if isinstance(ret, bool):
                    ok = ret
            except Exception as exc:
                ok = False
                print(f"[ERR] {exc}")
        lines = [line.rstrip() for line in buf.getvalue().splitlines() if line.strip()]
        return ok, lines

    def _ensure_tester(self) -> Any:
        with self._state_lock:
            tester = self._tester

        if tester is None:
            tester = self._create_tester(self._arm_config_path, self._backend_override)
            with self._state_lock:
                self._tester = tester

        connected = bool(getattr(tester.backend, "connected", False))
        if not connected and not bool(tester.backend.connect()):
            raise RuntimeError(tester.backend.diagnostics().get("last_connect_error", "backend connect failed"))

        with contextlib.suppress(Exception):
            tester.backend.set_speed_percent(int(tester.speed_percent))
        with contextlib.suppress(Exception):
            self._sync_backend_tcp_offset(tester)
        with self._state_lock:
            self._speed_percent = int(getattr(tester, "speed_percent", 0))
        ctrl_mode = self._read_ctrl_mode_from_robot(tester)
        if ctrl_mode is not None:
            with self._state_lock:
                if int(ctrl_mode) == GRAVITY_COMP_CTRL_MODE or self._ctrl_mode_hint != GRAVITY_COMP_CTRL_MODE:
                    self._ctrl_mode_hint = int(ctrl_mode)
        with contextlib.suppress(Exception):
            tester._start_feedback_csv_logger()
        return tester

    def _peek_tester(self) -> Any | None:
        with self._state_lock:
            return self._tester

    def _create_tester(self, cfg_path: Path, backend_override: str | None) -> Any:
        if self._tester_factory is not None:
            return self._tester_factory(cfg_path, backend_override)

        repo_root = Path(__file__).resolve().parents[3]
        cli_path = repo_root / "robot_runtime" / "nero_test_cli.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"nero_test_cli.py not found: {cli_path}")

        spec = importlib.util.spec_from_file_location("nero_test_cli_web", cli_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load module from {cli_path}")
        module = importlib.util.module_from_spec(spec)
        module_name = spec.name
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

        base_cls = module.NeroArmTester

        class WebNeroArmTester(base_cls):
            def _init_logging(self_inner) -> None:
                logging_cfg = self_inner.cfg.get("logging", {})
                if not isinstance(logging_cfg, dict):
                    logging_cfg = {}
                logging_cfg = dict(logging_cfg)
                logging_cfg["capture_stdout"] = False
                logging_cfg["capture_stderr"] = False
                self_inner.cfg["logging"] = logging_cfg
                super()._init_logging()

        return WebNeroArmTester(cfg_path, backend_override)

    def _build_threepoint_steps(self, tester: Any) -> list[tuple[str, Callable[[], bool]]]:
        with self._state_lock:
            override = dict(self._pick_override) if self._pick_override is not None else None
        self._sync_tester_override_context(tester, override)

        gc = tester.cfg.get("gripper", {}) if isinstance(tester.cfg.get("gripper", {}), dict) else {}
        open_width = float(gc.get("open_width", 0.05))
        close_width = float(gc.get("close_width", 0.0))
        force = float(gc.get("force", 1.0))

        if override is not None:
            close_width = float(override.get("close_width", close_width))
            force = float(override.get("force", force))
        smooth_segments = 1
        if override is not None:
            smooth_segments = int(max(1, min(10, int(override.get("smooth_segments", 1)))))

        task = tester.cfg.get("task", {}) if isinstance(tester.cfg.get("task", {}), dict) else {}
        save_name = str(task.get("save_closed_state_waypoint", "grip_closed_at_pick")).strip() or "grip_closed_at_pick"

        builder = getattr(tester, "_build_threepoint_steps", None)
        if callable(builder):
            steps_raw = builder()
            if not isinstance(steps_raw, list):
                raise ValueError("invalid threepoint step builder result")
            steps: list[tuple[str, Callable[[], bool]]] = []
            for item in steps_raw:
                if (
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not callable(item[1])
                ):
                    raise ValueError("invalid threepoint step item")
                steps.append((item[0], item[1]))
        else:
            pose_key = tester._active_pose_key()
            for name in getattr(tester, "threepoint_info", {}).keys() or ("ready", "pick", "transport", "dump"):
                info = tester.threepoint_info.get(name, {})
                pose = info.get(pose_key)
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError(f"missing threepoint pose ({pose_key}): {name}")

            def move_pose(name: str, label: str, motion: str, segmented: bool) -> bool:
                if segmented and smooth_segments > 1:
                    return self._move_threepoint_pose_segmented(
                        tester,
                        name=name,
                        label=label,
                        motion=motion,
                        segments=smooth_segments,
                    )
                return bool(tester._move_threepoint_pose(name, label, motion))

            steps = [
                ("move ready", lambda: tester._move_threepoint_pose("ready", "move ready", tester.transfer_motion)),
                ("open gripper (ready)", lambda: tester.backend.open_gripper(open_width, force)),
                ("move pick", lambda: move_pose("pick", "move pick", tester.approach_motion, segmented=True)),
                ("close gripper", lambda: tester.backend.close_gripper(close_width, force)),
                (f"save closed state {save_name}", lambda: tester._save_runtime_state(save_name)),
                (
                    "move ready (keep gripper closed)",
                    lambda: move_pose("ready", "move ready (keep gripper closed)", tester.transfer_motion, segmented=True),
                ),
                (
                    "move transport (keep gripper closed)",
                    lambda: tester._move_threepoint_pose(
                        "transport",
                        "move transport (keep gripper closed)",
                        tester.transfer_motion,
                    ),
                ),
                ("move dump", lambda: tester._move_threepoint_pose("dump", "move dump", tester.approach_motion)),
                ("open gripper (dump)", lambda: tester.backend.open_gripper(open_width, force)),
                (
                    "move transport (return)",
                    lambda: tester._move_threepoint_pose("transport", "move transport (return)", tester.transfer_motion),
                ),
                (
                    "move ready (return)",
                    lambda: tester._move_threepoint_pose("ready", "move ready (return)", tester.transfer_motion),
                ),
            ]

        profile = str(getattr(tester, "execution_profile", "pose_only")).strip().lower() or "pose_only"
        normalized_steps: list[tuple[str, Callable[[], bool]]] = []
        for label, action in steps:
            if label == "close gripper":
                normalized_steps.append(
                    (label, lambda close_w=close_width, grip_force=force: tester.backend.close_gripper(close_w, grip_force))
                )
                continue

            if smooth_segments > 1 and profile != "joint_first":
                if label == "move pick":
                    approach_motion = str(getattr(tester, "approach_motion", "p")).strip().lower() or "p"
                    normalized_steps.append(
                        (
                            label,
                            lambda segments=smooth_segments, move_mode=approach_motion: self._move_threepoint_pose_segmented(
                                tester,
                                name="pick",
                                label="move pick",
                                motion=move_mode,
                                segments=segments,
                            ),
                        )
                    )
                    continue
                if label == "move ready (keep gripper closed)":
                    transfer_motion = str(getattr(tester, "transfer_motion", "p")).strip().lower() or "p"
                    normalized_steps.append(
                        (
                            label,
                            lambda segments=smooth_segments, move_mode=transfer_motion: self._move_threepoint_pose_segmented(
                                tester,
                                name="ready",
                                label="move ready (keep gripper closed)",
                                motion=move_mode,
                                segments=segments,
                            ),
                        )
                    )
                    continue

            normalized_steps.append((label, action))
        return normalized_steps

    def _move_threepoint_pose_segmented(
        self,
        tester: Any,
        name: str,
        label: str,
        motion: str,
        segments: int,
    ) -> bool:
        motion_mode = str(motion).strip().lower() or "p"
        if motion_mode not in {"p", "l"}:
            print(f"[ERR] {label}: unsupported segmented motion '{motion_mode}'")
            return False

        if segments <= 1:
            return bool(tester._move_threepoint_pose(name, label, motion_mode))

        # Official SDK docs warn against continuously streaming move_l targets.
        # Keep smooth segmentation strictly on move_p; move_l stays single-shot.
        if motion_mode == "l":
            print(
                f"[WARN] {label}: smooth_segments={int(segments)} is not applied to move_l; "
                "fallback to single move_l"
            )
            return bool(tester._move_threepoint_pose(name, label, motion_mode))

        info = tester.threepoint_info.get(name, {})
        pose_key = tester._active_pose_key()
        target = info.get(pose_key)
        if not isinstance(target, list) or len(target) != 6:
            print(f"[ERR] threepoint.{name}.{pose_key} missing")
            return False

        current = tester.backend.get_flange_pose()
        if not isinstance(current, list) or len(current) != 6:
            return bool(tester._move_threepoint_pose(name, label, motion))

        executor = getattr(tester, "_execute_pose_motion", None)
        if not callable(executor):
            return bool(tester._move_threepoint_pose(name, label, motion))

        print(f"[RUN] {label} segmented move: {segments} steps")
        cur = [float(v) for v in current]
        tgt = [float(v) for v in target]
        for idx in range(1, segments + 1):
            ratio = float(idx) / float(segments)
            pose = [
                cur[i] + (tgt[i] - cur[i]) * ratio
                for i in range(6)
            ]
            step_label = label if idx == segments else f"{label} seg{idx}/{segments}"
            if not bool(executor(pose, step_label, motion_mode)):
                return False
        return True

    @staticmethod
    def _normalize_angle_rad(value: float) -> float:
        return math.atan2(math.sin(float(value)), math.cos(float(value)))

    @staticmethod
    def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(float(roll))
        sr = math.sin(float(roll))
        cp = math.cos(float(pitch))
        sp = math.sin(float(pitch))
        cy = math.cos(float(yaw))
        sy = math.sin(float(yaw))
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _pose_to_matrix(cls, pose_m_rad: list[float]) -> np.ndarray:
        if len(pose_m_rad) != 6:
            raise ValueError("pose must contain 6 values")
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = cls._rpy_to_matrix(pose_m_rad[3], pose_m_rad[4], pose_m_rad[5])
        out[:3, 3] = np.array([float(v) for v in pose_m_rad[:3]], dtype=np.float64)
        return out

    @classmethod
    def _transform_point(cls, matrix_4x4: np.ndarray, point_xyz: list[float]) -> np.ndarray:
        vec = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=np.float64)
        return (matrix_4x4 @ vec)[:3]

    @classmethod
    def _transform_direction(cls, matrix_4x4: np.ndarray, direction_xyz: list[float]) -> np.ndarray:
        vec = np.array([float(direction_xyz[0]), float(direction_xyz[1]), float(direction_xyz[2])], dtype=np.float64)
        out = matrix_4x4[:3, :3] @ vec
        norm = float(np.linalg.norm(out))
        if norm < 1e-9:
            raise ValueError("direction norm is too small")
        return out / norm

    @classmethod
    def _load_handeye_flange_to_optical(cls) -> tuple[np.ndarray | None, str, str | None]:
        handeye_path = cls._handeye_path()
        if not handeye_path.exists():
            return None, "unavailable", f"handeye extrinsics file not found: {handeye_path}"
        try:
            raw = yaml.safe_load(handeye_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                return None, "unavailable", "invalid handeye yaml format"
            nominal = raw.get("nominal_camera")
            if not isinstance(nominal, dict):
                return None, "unavailable", "nominal_camera section missing in handeye yaml"
            calibrated = raw.get("calibrated_camera")
            if not isinstance(calibrated, dict) or not bool(calibrated.get("enabled", False)):
                return (
                    None,
                    "unavailable",
                    "calibrated handeye unavailable: calibrated_camera.enabled=true is required",
                )
            camera_source = calibrated
            cam_xyz = camera_source.get("xyz_m", nominal.get("xyz_m", [0.0, 0.0, 0.0]))
            cam_rpy = camera_source.get("rpy_rad", nominal.get("rpy_rad", [0.0, 0.0, 0.0]))
            opt_xyz = nominal.get("optical_xyz_m", [0.0, 0.0, 0.0])
            opt_rpy = nominal.get("optical_rpy_rad", [0.0, 0.0, 0.0])
            flange_to_camera = cls._pose_to_matrix([float(v) for v in [*cam_xyz, *cam_rpy]])
            camera_to_optical = cls._pose_to_matrix([float(v) for v in [*opt_xyz, *opt_rpy]])
            return flange_to_camera @ camera_to_optical, "calibrated", None
        except Exception as exc:
            return None, "unavailable", f"calibrated handeye load failed: {exc}"

    @classmethod
    def _handeye_path(cls) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "模型文件" / "nero_description" / "config" / "handeye_extrinsics.yaml"

    @staticmethod
    def _as_float(raw: Any, default: float) -> float:
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(raw: Any, default: int) -> int:
        try:
            return int(raw)
        except Exception:
            return int(default)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    @classmethod
    def _load_yaml_mapping(cls, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _load_pick_override_safety_config(cls, arm_config_path: Path) -> dict[str, float]:
        defaults = {
            "prepick_offset_m": 0.05,
            "max_descent_m": 0.35,
            "min_safe_z_m": 0.10,
        }
        raw = cls._load_yaml_mapping(arm_config_path)
        grasp_cfg: dict[str, Any] = {}
        dynamic_grasp = raw.get("dynamic_grasp")
        if isinstance(dynamic_grasp, dict):
            grasp = dynamic_grasp.get("grasp")
            if isinstance(grasp, dict):
                grasp_cfg = grasp
        elif isinstance(raw.get("grasp"), dict):
            grasp_cfg = raw.get("grasp")
        return {
            "prepick_offset_m": max(0.0, cls._as_float(grasp_cfg.get("prepick_offset_m"), defaults["prepick_offset_m"])),
            "max_descent_m": max(0.01, cls._as_float(grasp_cfg.get("max_descent_m"), defaults["max_descent_m"])),
            # Preserve configured frame convention (some setups legitimately use negative Z in base frame).
            "min_safe_z_m": cls._as_float(grasp_cfg.get("min_safe_z_m"), defaults["min_safe_z_m"]),
        }

    @classmethod
    def _load_dynamic_stability_config(cls, arm_config_path: Path) -> dict[str, Any]:
        defaults = {
            "window": 18,
            "min_axis_quality": 0.85,
            "mad_limit": {"x": 1.5, "y": 1.5, "z": 2.0, "yaw": 1.2},
        }
        raw = cls._load_yaml_mapping(arm_config_path)
        source: dict[str, Any] = {}
        dynamic_grasp = raw.get("dynamic_grasp")
        if isinstance(dynamic_grasp, dict):
            vision = dynamic_grasp.get("vision")
            if isinstance(vision, dict):
                source = vision
        if not source and isinstance(raw.get("robot_runtime"), dict):
            rr = raw.get("robot_runtime")
            if isinstance(rr, dict):
                wds = rr.get("web_dynamic_stability")
                if isinstance(wds, dict):
                    source = wds

        nested = source.get("web_dynamic_stability")
        if isinstance(nested, dict):
            source = dict(source)
            source.update(nested)

        mad_source = source.get("mad_limit")
        if not isinstance(mad_source, dict):
            mad_source = {}
        x_limit = cls._as_float(mad_source.get("x", source.get("web_dynamic_mad_limit_x_mm")), defaults["mad_limit"]["x"])
        y_limit = cls._as_float(mad_source.get("y", source.get("web_dynamic_mad_limit_y_mm")), defaults["mad_limit"]["y"])
        z_limit = cls._as_float(mad_source.get("z", source.get("web_dynamic_mad_limit_z_mm")), defaults["mad_limit"]["z"])
        yaw_limit = cls._as_float(
            mad_source.get("yaw", source.get("web_dynamic_mad_limit_yaw_deg")),
            defaults["mad_limit"]["yaw"],
        )
        return {
            "window": max(3, min(120, cls._as_int(source.get("window", source.get("web_dynamic_window")), defaults["window"]))),
            "min_axis_quality": cls._clamp(
                cls._as_float(source.get("min_axis_quality", source.get("web_dynamic_min_axis_quality")), defaults["min_axis_quality"]),
                0.0,
                1.0,
            ),
            "mad_limit": {
                "x": max(0.1, float(x_limit)),
                "y": max(0.1, float(y_limit)),
                "z": max(0.1, float(z_limit)),
                "yaw": max(0.1, float(yaw_limit)),
            },
        }

    @classmethod
    def _matrix_to_pose_m_rad(cls, matrix_4x4: np.ndarray) -> list[float]:
        if matrix_4x4.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        rot = matrix_4x4[:3, :3]
        x = float(matrix_4x4[0, 3])
        y = float(matrix_4x4[1, 3])
        z = float(matrix_4x4[2, 3])

        sp = cls._clamp(-float(rot[2, 0]), -1.0, 1.0)
        pitch = math.asin(sp)
        cp = math.cos(pitch)
        if abs(cp) > 1e-8:
            roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
            yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        else:
            roll = 0.0
            yaw = math.atan2(float(-rot[0, 1]), float(rot[1, 1]))
        return [x, y, z, roll, pitch, yaw]

    @classmethod
    def _load_handeye_tcp_offset_m_rad(cls) -> list[float] | None:
        handeye_path = cls._handeye_path()
        if not handeye_path.exists():
            return None
        try:
            raw = yaml.safe_load(handeye_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                return None
            gripper = raw.get("gripper_nominal")
            if not isinstance(gripper, dict):
                return None
            flange_to_mount = cls._pose_to_matrix(
                [*gripper.get("mount_xyz_m", [0.0, 0.0, 0.0]), *gripper.get("mount_rpy_rad", [0.0, 0.0, 0.0])]
            )
            mount_to_base = cls._pose_to_matrix([*gripper.get("xyz_m", [0.0, 0.0, 0.0]), *gripper.get("rpy_rad", [0.0, 0.0, 0.0])])
            base_to_tcp = cls._pose_to_matrix(
                [*gripper.get("tcp_xyz_m", [0.0, 0.0, 0.0]), *gripper.get("tcp_rpy_rad", [0.0, 0.0, 0.0])]
            )
            tcp_offset = cls._matrix_to_pose_m_rad(flange_to_mount @ mount_to_base @ base_to_tcp)
            if len(tcp_offset) != 6:
                return None
            if not all(math.isfinite(float(v)) for v in tcp_offset):
                return None
            return [float(v) for v in tcp_offset]
        except Exception:
            return None

    @staticmethod
    def _sync_tester_override_context(tester: Any, override: dict[str, Any] | None) -> None:
        if override is None:
            if hasattr(tester, "_web_override_context"):
                with contextlib.suppress(Exception):
                    delattr(tester, "_web_override_context")
            return
        context = {
            "active": True,
            "mode": str(override.get("mode", "none")),
            "static_pick_pose_m_rad": list(override.get("original_pick_pose_m_rad", [])),
            "override_pick_pose_m_rad": list(override.get("applied_pick_pose_m_rad", [])),
            "override_pick_vs_static_pick_delta": list(override.get("pick_delta_m_rad", [])),
        }
        with contextlib.suppress(Exception):
            setattr(tester, "_web_override_context", context)

    def _sync_backend_tcp_offset(self, tester: Any) -> None:
        tcp_offset = self._handeye_tcp_offset_m_rad
        if tcp_offset is None:
            return
        backend = getattr(tester, "backend", None)
        robot = getattr(backend, "robot", None) if backend is not None else None
        if robot is None:
            return
        setter = getattr(robot, "set_tcp_offset", None)
        if callable(setter):
            setter([float(v) for v in tcp_offset])

    @staticmethod
    def _render_pick_override_meta(override: dict[str, Any] | None) -> dict[str, Any]:
        if override is None:
            return {
                "active": False,
                "mode": "none",
                "close_width": None,
                "force": None,
                "smooth_segments": None,
            }
        return {
            "active": True,
            "mode": str(override.get("mode", "none")),
            "close_width": float(override.get("close_width", 0.0)),
            "force": float(override.get("force", 0.0)),
            "smooth_segments": int(override.get("smooth_segments", 1)),
        }

    def _render_pick_override_meta_unlocked(self) -> dict[str, Any]:
        return self._render_pick_override_meta(self._pick_override)

    @staticmethod
    def _parse_speed_percent(params: dict[str, Any] | None) -> tuple[int | None, str | None]:
        if not isinstance(params, dict):
            return None, "invalid params: expect object with percent"
        raw = params.get("percent")
        try:
            value = int(raw)
        except Exception:
            return None, "invalid params: percent must be an integer"
        if value < 1 or value > 100:
            return None, "invalid params: percent must be between 1 and 100"
        return value, None

    @staticmethod
    def _parse_pick_override_params(params: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(params, dict):
            return None, "invalid params: expect object"

        required = ("x_mm", "y_mm", "z_mm", "yaw_deg", "close_width", "force")
        missing = [key for key in required if key not in params]
        if missing:
            return None, f"invalid params: missing {', '.join(missing)}"

        out: dict[str, Any] = {}
        for key in ("x_mm", "y_mm", "z_mm", "yaw_deg", "close_width", "force"):
            try:
                out[key] = float(params[key])
            except Exception:
                return None, f"invalid params: {key} must be a number"

        raw_segments = params.get("smooth_segments", 3)
        try:
            smooth_segments = int(raw_segments)
        except Exception:
            return None, "invalid params: smooth_segments must be an integer"
        if smooth_segments < 1 or smooth_segments > 10:
            return None, "invalid params: smooth_segments must be between 1 and 10"
        out["smooth_segments"] = smooth_segments
        segmented_motion = str(params.get("segmented_motion", "p")).strip().lower() or "p"
        if segmented_motion != "p":
            return None, "invalid params: segmented_motion must be 'p' (segmented move_l is unsafe)"
        out["segmented_motion"] = segmented_motion

        mode = str(params.get("mode", "random")).strip().lower()
        if mode not in {"random", "dynamic"}:
            return None, "invalid params: mode must be random or dynamic"
        out["mode"] = mode

        if out["close_width"] < 0.0:
            return None, "invalid params: close_width must be >= 0"
        if out["close_width"] > 0.1:
            return None, "invalid params: close_width must be <= 0.1"
        if out["force"] < 0.0:
            return None, "invalid params: force must be >= 0"
        if out["force"] > 3.0:
            return None, "invalid params: force must be <= 3.0"
        return out, None

    def _apply_pick_override_from_vision(self, tester: Any, params: dict[str, Any]) -> tuple[bool, str]:
        if self._flange_to_optical is None:
            detail = self._handeye_error or "calibrated handeye unavailable"
            return False, f"calibrated handeye unavailable: {detail}"
        if self._handeye_tcp_offset_m_rad is None:
            return False, "calibrated handeye unavailable: handeye tcp offset unavailable"

        current_flange = tester.backend.get_flange_pose()
        if not isinstance(current_flange, list) or len(current_flange) != 6:
            return False, "cannot apply pick override: no flange pose feedback"
        if not all(math.isfinite(float(v)) for v in current_flange):
            return False, "cannot apply pick override: flange pose has invalid values"

        pick_info = tester.threepoint_info.get("pick", {})
        prepick_info = tester.threepoint_info.get("prepick", {})
        pick_pose = pick_info.get("pose_m_rad")
        pick_locked = pick_info.get("locked_pose_m_rad")
        prepick_pose = prepick_info.get("pose_m_rad")
        prepick_locked = prepick_info.get("locked_pose_m_rad")
        ready_info = tester.threepoint_info.get("ready", {})
        ready_pose = ready_info.get("pose_m_rad")
        if not isinstance(pick_pose, list) or len(pick_pose) != 6:
            return False, "cannot apply pick override: pick pose missing"
        if not isinstance(pick_locked, list) or len(pick_locked) != 6:
            pick_locked = list(pick_pose)
        if not isinstance(prepick_pose, list) or len(prepick_pose) != 6:
            return False, "cannot apply pick override: prepick pose missing"
        if not isinstance(prepick_locked, list) or len(prepick_locked) != 6:
            prepick_locked = list(prepick_pose)
        if not isinstance(ready_pose, list) or len(ready_pose) != 6:
            return False, "cannot apply pick override: ready pose missing"

        base_to_flange = self._pose_to_matrix([float(v) for v in current_flange])
        base_to_optical = base_to_flange @ self._flange_to_optical

        point_optical_m = [float(params["x_mm"]) / 1000.0, float(params["y_mm"]) / 1000.0, float(params["z_mm"]) / 1000.0]
        point_base = self._transform_point(base_to_optical, point_optical_m)

        yaw_optical_rad = math.radians(float(params["yaw_deg"]))
        axis_optical = [math.cos(yaw_optical_rad), math.sin(yaw_optical_rad), 0.0]
        axis_base = self._transform_direction(base_to_optical, axis_optical)
        yaw_base = math.atan2(float(axis_base[1]), float(axis_base[0]))

        self._sync_backend_tcp_offset(tester)
        backend = getattr(tester, "backend", None)
        robot = getattr(backend, "robot", None) if backend is not None else None
        converter = getattr(robot, "get_tcp2flange_pose", None) if robot is not None else None
        if not callable(converter):
            return False, "cannot apply pick override: tcp->flange transform unavailable"

        prepick_offset_m = float(self._pick_override_safety["prepick_offset_m"])
        max_descent_m = float(self._pick_override_safety["max_descent_m"])
        min_safe_z_m = float(self._pick_override_safety["min_safe_z_m"])

        with self._state_lock:
            override = self._pick_override
            if override is None:
                override = {
                    "original_pick_pose_m_rad": [float(v) for v in pick_pose],
                    "original_pick_locked_pose_m_rad": [float(v) for v in pick_locked],
                    "original_prepick_pose_m_rad": [float(v) for v in prepick_pose],
                    "original_prepick_locked_pose_m_rad": [float(v) for v in prepick_locked],
                }
            original_pose = [float(v) for v in override["original_pick_pose_m_rad"]]
            ref_yaw = float(original_pose[5])
            alt_yaw = self._normalize_angle_rad(yaw_base + math.pi)
            main_err = abs(self._normalize_angle_rad(yaw_base - ref_yaw))
            alt_err = abs(self._normalize_angle_rad(alt_yaw - ref_yaw))
            chosen_yaw = alt_yaw if alt_err + 1e-6 < main_err else yaw_base

            target_tcp_pose = [
                float(point_base[0]),
                float(point_base[1]),
                float(point_base[2]),
                float(original_pose[3]),
                float(original_pose[4]),
                float(chosen_yaw),
            ]
            if target_tcp_pose[2] < min_safe_z_m:
                return (
                    False,
                    f"reject override: pick.z={target_tcp_pose[2]:.4f}m below min_safe_z={min_safe_z_m:.4f}m",
                )
            descent_m = float(ready_pose[2]) - float(target_tcp_pose[2])
            if descent_m > max_descent_m + 1e-9:
                return (
                    False,
                    f"reject override: descent={descent_m:.4f}m exceeds max_descent={max_descent_m:.4f}m",
                )

            try:
                converted = converter([float(v) for v in target_tcp_pose])
            except Exception as exc:
                return False, f"cannot apply pick override: tcp->flange transform failed ({exc})"
            if not isinstance(converted, (list, tuple)) or len(converted) != 6:
                return False, "cannot apply pick override: invalid tcp->flange result"
            applied_pick_pose = [float(v) for v in converted]
            if not all(math.isfinite(float(v)) for v in applied_pick_pose):
                return False, "cannot apply pick override: tcp->flange result has invalid values"

            applied_prepick_pose = list(applied_pick_pose)
            applied_prepick_pose[2] = float(applied_pick_pose[2]) + prepick_offset_m
            applied_pick_locked_pose = list(applied_pick_pose)
            applied_prepick_locked_pose = list(applied_prepick_pose)
            pick_info["pose_m_rad"] = list(applied_pick_pose)
            pick_info["locked_pose_m_rad"] = list(applied_pick_locked_pose)
            prepick_info["pose_m_rad"] = list(applied_prepick_pose)
            prepick_info["locked_pose_m_rad"] = list(applied_prepick_locked_pose)

            pick_delta = [float(applied_pick_pose[i]) - float(original_pose[i]) for i in range(6)]

            override.update(
                {
                    "mode": str(params["mode"]),
                    "close_width": float(params["close_width"]),
                    "force": float(params["force"]),
                    "smooth_segments": int(params["smooth_segments"]),
                    "applied_pick_target_tcp_pose_m_rad": list(target_tcp_pose),
                    "applied_pick_pose_m_rad": list(applied_pick_pose),
                    "applied_pick_locked_pose_m_rad": list(applied_pick_locked_pose),
                    "applied_prepick_pose_m_rad": list(applied_prepick_pose),
                    "applied_prepick_locked_pose_m_rad": list(applied_prepick_locked_pose),
                    "pick_delta_m_rad": pick_delta,
                    "prepick_offset_m": prepick_offset_m,
                    "max_descent_m": max_descent_m,
                    "min_safe_z_m": min_safe_z_m,
                    "source_optical_mm": [float(params["x_mm"]), float(params["y_mm"]), float(params["z_mm"])],
                    "source_yaw_deg": float(params["yaw_deg"]),
                    "applied_at": time.time(),
                }
            )
            self._pick_override = override

        message = "pick override applied"
        smooth_segments = int(params["smooth_segments"])
        approach_motion = str(getattr(tester, "approach_motion", "p")).strip().lower() or "p"
        if smooth_segments > 1 and approach_motion == "l":
            message = (
                f"{message}; notice: smooth_segments={smooth_segments} only applies to move_p, "
                "pick step will fallback to single move_l"
            )
        return True, message

    def _clear_pick_override_unlocked(self, tester: Any | None) -> bool:
        override = self._pick_override
        if override is None:
            return False

        if tester is not None:
            pick_info = getattr(tester, "threepoint_info", {}).get("pick", {})
            prepick_info = getattr(tester, "threepoint_info", {}).get("prepick", {})
            original_pose = override.get("original_pick_pose_m_rad")
            original_locked = override.get("original_pick_locked_pose_m_rad")
            original_prepick_pose = override.get("original_prepick_pose_m_rad")
            original_prepick_locked = override.get("original_prepick_locked_pose_m_rad")
            if isinstance(pick_info, dict):
                if isinstance(original_pose, list) and len(original_pose) == 6:
                    pick_info["pose_m_rad"] = [float(v) for v in original_pose]
                if isinstance(original_locked, list) and len(original_locked) == 6:
                    pick_info["locked_pose_m_rad"] = [float(v) for v in original_locked]
            if isinstance(prepick_info, dict):
                if isinstance(original_prepick_pose, list) and len(original_prepick_pose) == 6:
                    prepick_info["pose_m_rad"] = [float(v) for v in original_prepick_pose]
                if isinstance(original_prepick_locked, list) and len(original_prepick_locked) == 6:
                    prepick_info["locked_pose_m_rad"] = [float(v) for v in original_prepick_locked]
            if hasattr(tester, "_web_override_context"):
                with contextlib.suppress(Exception):
                    delattr(tester, "_web_override_context")

        self._pick_override = None
        return True

    def _safe_diag(self, tester: Any | None = None) -> dict[str, str]:
        live = tester if tester is not None else self._peek_tester()
        if live is None:
            return {}
        with contextlib.suppress(Exception):
            return dict(live.backend.diagnostics())
        return {}

    @staticmethod
    def _is_truthy_env(raw: str | None) -> bool:
        if raw is None:
            return False
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _is_robot_exclusive_control(cls) -> bool:
        return cls._is_truthy_env(os.environ.get(ROBOT_EXCLUSIVE_ENV))

    @staticmethod
    def _is_auto_enable_service_active() -> bool:
        if shutil.which("systemctl") is None:
            return False
        try:
            proc = subprocess.run(
                ["systemctl", "is-active", "--quiet", AUTO_ENABLE_SERVICE_NAME],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
        return proc.returncode == 0

    @classmethod
    def _get_daemon_conflict_message(cls) -> str | None:
        if cls._is_robot_exclusive_control():
            return None
        if not cls._is_auto_enable_service_active():
            return None
        return (
            "daemon conflict: nero-auto-enable.service is active; "
            "stop it or relaunch vision with exclusive robot control"
        )

    @staticmethod
    def _format_ctrl_mode_label(ctrl_mode: int | None, inferred: bool) -> str:
        if ctrl_mode is None:
            return "未知"
        label = CTRL_MODE_LABELS.get(int(ctrl_mode), f"未知模式({int(ctrl_mode)})")
        if inferred:
            return f"推断: {label}"
        return label

    @staticmethod
    def _read_ctrl_mode_from_robot(tester: Any | None) -> int | None:
        if tester is None:
            return None
        backend = getattr(tester, "backend", None)
        robot = getattr(backend, "robot", None) if backend is not None else None
        if robot is None:
            return None
        getter = getattr(robot, "get_arm_status", None)
        if not callable(getter):
            return None
        try:
            status = getter()
        except Exception:
            return None
        msg = getattr(status, "msg", None)
        if msg is None:
            return None
        raw = getattr(msg, "ctrl_mode", None)
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    def _is_gravity_compensation_active(self, tester: Any) -> bool:
        ctrl_mode = self._read_ctrl_mode_from_robot(tester)
        if ctrl_mode is not None and int(ctrl_mode) == GRAVITY_COMP_CTRL_MODE:
            with self._state_lock:
                self._ctrl_mode_hint = int(ctrl_mode)
            return True

        with self._state_lock:
            hint = self._ctrl_mode_hint

        if hint == GRAVITY_COMP_CTRL_MODE:
            return True

        if ctrl_mode is not None:
            with self._state_lock:
                self._ctrl_mode_hint = int(ctrl_mode)
        return False

    def _render_step_progress_unlocked(self) -> dict[str, Any]:
        session = self._step_session
        if session is None:
            return {
                "active": False,
                "index": 0,
                "total": 0,
                "next_label": None,
                "last_label": None,
            }

        steps = session.get("steps")
        index = int(session.get("index", 0))
        total = int(session.get("total", 0))
        next_label = None
        if isinstance(steps, list) and 0 <= index < len(steps):
            next_label = steps[index][0]

        return {
            "active": bool(session.get("active", False)),
            "index": index,
            "total": total,
            "next_label": next_label,
            "last_label": session.get("last_label"),
        }

    def _build_result(
        self,
        ok: bool,
        command: str,
        message: str,
        stdout_lines: list[str],
        diag: dict[str, str],
        error_code: str | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            step_progress = self._render_step_progress_unlocked()
        return {
            "ok": bool(ok),
            "command": command,
            "message": message,
            "stdout_lines": list(stdout_lines),
            "step_progress": step_progress,
            "error_code": error_code,
            "diag": dict(diag),
        }
