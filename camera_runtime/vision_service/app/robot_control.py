from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


ALLOWED_COMMANDS = {
    "status",
    "precheck",
    "enable",
    "home",
    "open",
    "close",
    "show_points",
    "run_threepoint_auto",
    "run_threepoint_step_start",
    "run_threepoint_step_next",
    "run_threepoint_step_cancel",
    "estop",
}


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
        self._last_result: dict[str, Any] = {
            "ok": False,
            "command": "",
            "message": "idle",
            "stdout_lines": [],
            "step_progress": self._render_step_progress_unlocked(),
            "diag": {},
        }

    def shutdown(self) -> None:
        with self._state_lock:
            tester = self._tester
            self._tester = None
            self._step_session = None
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

        joints: list[float] | None = None
        pose: list[float] | None = None
        diag: dict[str, str] = {}
        connected = False

        if tester is not None:
            with contextlib.suppress(Exception):
                diag = dict(tester.backend.diagnostics())
            with contextlib.suppress(Exception):
                joints = tester.backend.get_joint_positions()
            with contextlib.suppress(Exception):
                pose = tester.backend.get_flange_pose()
            connected = bool(getattr(tester.backend, "connected", False))

        return {
            "enabled": self._enabled,
            "connected": connected,
            "busy": busy,
            "joint_positions_rad": joints,
            "flange_pose_m_rad": pose,
            "diag": diag,
            "step_progress": step_progress,
            "last_result": last_result,
        }

    def execute_command(self, command: str) -> dict[str, Any]:
        name = str(command).strip().lower()
        if name not in ALLOWED_COMMANDS:
            return self._build_result(False, name, "unsupported command", [], self._safe_diag())

        if not self._enabled:
            return self._build_result(False, name, "robot web control disabled", [], self._safe_diag())

        if name == "estop":
            return self._execute_estop_anytime(name)

        if not self._command_lock.acquire(blocking=False):
            return self._build_result(False, name, "robot is busy", [], self._safe_diag())

        with self._state_lock:
            self._busy = True

        try:
            result = self._execute_command_locked(name)
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

    def _execute_command_locked(self, command: str) -> dict[str, Any]:
        try:
            tester = self._ensure_tester()
        except Exception as exc:
            return self._build_result(False, command, f"connect failed: {exc}", [], self._safe_diag())

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
            ok, lines = self._capture_output(lambda: bool(tester._run_threepoint_sequence(step_mode=False)))
            if ok:
                with self._state_lock:
                    self._step_session = None
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
                return self._build_result(False, command, "step session is invalid", [], self._safe_diag(tester))

            label, action = steps[index]
            ok, lines = self._capture_output(lambda: bool(action()))

            with self._state_lock:
                live = self._step_session
                if live is not None:
                    live["last_label"] = label
                    if ok:
                        live["index"] = index + 1
                        if int(live["index"]) >= int(live.get("total", 0)):
                            live["active"] = False
                    else:
                        live["active"] = False
            message = f"step done: {label}" if ok else f"step failed: {label}"
            return self._build_result(ok, command, message, lines, self._safe_diag(tester))

        if command == "run_threepoint_step_cancel":
            has_session = False
            with self._state_lock:
                has_session = self._step_session is not None
                if has_session:
                    self._step_session = None
            if not has_session:
                return self._build_result(False, command, "no active step session", [], self._safe_diag(tester))
            return self._build_result(True, command, "step session cancelled", [], self._safe_diag(tester))

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

        msg = "estop sent"
        if forced:
            msg = "estop sent (forced while busy)"
        if not ok:
            msg = "estop failed"
        return self._build_result(ok, command, msg, [], self._safe_diag(tester))

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
        pose_key = tester._active_pose_key()
        for name in getattr(tester, "threepoint_info", {}).keys() or ("ready", "pick", "transport", "dump"):
            info = tester.threepoint_info.get(name, {})
            pose = info.get(pose_key)
            if not isinstance(pose, list) or len(pose) != 6:
                raise ValueError(f"missing threepoint pose ({pose_key}): {name}")

        gc = tester.cfg.get("gripper", {}) if isinstance(tester.cfg.get("gripper", {}), dict) else {}
        open_width = float(gc.get("open_width", 0.05))
        close_width = float(gc.get("close_width", 0.0))
        force = float(gc.get("force", 1.0))

        task = tester.cfg.get("task", {}) if isinstance(tester.cfg.get("task", {}), dict) else {}
        save_name = str(task.get("save_closed_state_waypoint", "grip_closed_at_pick")).strip() or "grip_closed_at_pick"

        return [
            ("move ready", lambda: tester._move_threepoint_pose("ready", "move ready", tester.transfer_motion)),
            ("open gripper (ready)", lambda: tester.backend.open_gripper(open_width, force)),
            ("move pick", lambda: tester._move_threepoint_pose("pick", "move pick", tester.approach_motion)),
            ("close gripper", lambda: tester.backend.close_gripper(close_width, force)),
            (f"save closed state {save_name}", lambda: tester._save_runtime_state(save_name)),
            (
                "move ready (keep gripper closed)",
                lambda: tester._move_threepoint_pose("ready", "move ready (keep gripper closed)", tester.transfer_motion),
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

    def _safe_diag(self, tester: Any | None = None) -> dict[str, str]:
        live = tester if tester is not None else self._peek_tester()
        if live is None:
            return {}
        with contextlib.suppress(Exception):
            return dict(live.backend.diagnostics())
        return {}

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
    ) -> dict[str, Any]:
        with self._state_lock:
            step_progress = self._render_step_progress_unlocked()
        return {
            "ok": bool(ok),
            "command": command,
            "message": message,
            "stdout_lines": list(stdout_lines),
            "step_progress": step_progress,
            "diag": dict(diag),
        }
