from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

from vision_service.app.robot_control import RobotControlManager


class FakeBackend:
    def __init__(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.disable_calls = 0
                self.reset_calls = 0
                self.ctrl_mode = 0x01

            def disable(self, _joint_index: int = 255) -> bool:
                self.disable_calls += 1
                return True

            def reset(self) -> None:
                self.reset_calls += 1

            def set_leader_mode(self) -> None:
                self.ctrl_mode = 0x06

            def set_normal_mode(self) -> None:
                self.ctrl_mode = 0x01

            def get_arm_status(self):
                return types.SimpleNamespace(msg=types.SimpleNamespace(ctrl_mode=self.ctrl_mode))

        self.robot = FakeRobot()
        self.connected = False
        self.connect_calls = 0
        self.enable_calls = 0
        self.estop_calls = 0
        self.speed_calls: list[int] = []

    def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.speed_calls.append(int(percent))

    def diagnostics(self) -> dict[str, str]:
        return {
            "connected": "yes" if self.connected else "no",
            "enable_ok": "yes" if self.enable_calls > 0 else "no",
        }

    def get_joint_positions(self):
        return [0.0] * 7

    def get_flange_pose(self):
        return [0.0] * 6

    def enable(self) -> bool:
        self.enable_calls += 1
        return True

    def open_gripper(self, width: float, force: float) -> bool:
        _ = (width, force)
        return True

    def close_gripper(self, width: float, force: float) -> bool:
        _ = (width, force)
        return True

    def estop(self) -> bool:
        self.estop_calls += 1
        return True


class FakeTester:
    def __init__(self, cfg_path: Path, backend_override: str | None) -> None:
        _ = (cfg_path, backend_override)
        self.backend = FakeBackend()
        self.speed_percent = 30
        self.cfg = {
            "home_joints": [0.0] * 7,
            "gripper": {
                "open_width": 0.05,
                "close_width": 0.0,
                "force": 1.0,
            },
            "task": {
                "save_closed_state_waypoint": "grip_closed_at_pick",
            },
        }
        self.transfer_motion = "p"
        self.approach_motion = "p"
        self.pose_motion_labels: list[str] = []
        pose = [0.0] * 6
        self.threepoint_info = {
            "ready": {"pose_m_rad": list(pose), "locked_pose_m_rad": list(pose)},
            "pick": {"pose_m_rad": list(pose), "locked_pose_m_rad": list(pose)},
            "transport": {"pose_m_rad": list(pose), "locked_pose_m_rad": list(pose)},
            "dump": {"pose_m_rad": list(pose), "locked_pose_m_rad": list(pose)},
        }
        self.slow_event = threading.Event()

    def shutdown(self) -> None:
        self.backend.connected = False

    def _start_feedback_csv_logger(self) -> None:
        return None

    def _active_pose_key(self) -> str:
        return "pose_m_rad"

    def _print_status(self) -> None:
        print("backend=real")

    def _precheck(self) -> bool:
        print("[OK] precheck passed")
        return True

    def _move_joints_segmented(self, joints, label: str) -> bool:
        _ = joints
        print(f"[RUN] {label}")
        return True

    def _show_points(self) -> None:
        print("Four-point plan")

    def _run_threepoint_sequence(self, step_mode: bool) -> bool:
        _ = step_mode
        self.slow_event.set()
        time.sleep(0.25)
        print("[OK] threepoint sequence completed")
        return True

    def _move_threepoint_pose(self, name: str, label: str, motion: str) -> bool:
        _ = (name, motion)
        self.pose_motion_labels.append(label)
        print(label)
        return True

    def _execute_pose_motion(self, pose_m_rad, label: str, motion: str) -> bool:
        _ = (pose_m_rad, motion)
        self.pose_motion_labels.append(label)
        print(label)
        return True

    def _save_runtime_state(self, name: str) -> bool:
        print(f"saved {name}")
        return True


def make_manager() -> tuple[RobotControlManager, list[FakeTester]]:
    created: list[FakeTester] = []

    def factory(cfg_path: Path, backend_override: str | None) -> FakeTester:
        tester = FakeTester(cfg_path, backend_override)
        created.append(tester)
        return tester

    manager = RobotControlManager(
        arm_config_path=Path("/tmp/fake_robot.yaml"),
        backend_override="real",
        enabled=True,
        tester_factory=factory,
    )
    manager._flange_to_optical = np.eye(4, dtype=np.float64)  # type: ignore[attr-defined]
    return manager, created


def test_manager_connects_on_demand() -> None:
    manager, created = make_manager()

    state0 = manager.get_state()
    assert state0["connected"] is False

    result = manager.execute_command("status")
    assert result["ok"] is True
    assert created
    assert created[0].backend.connect_calls == 1


def test_manager_rejects_busy_command() -> None:
    manager, created = make_manager()

    holder: dict[str, object] = {}

    def worker() -> None:
        holder["auto"] = manager.execute_command("run_threepoint_auto")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if created and created[0].slow_event.is_set():
            break
        time.sleep(0.01)

    busy = manager.execute_command("status")
    assert busy["ok"] is False
    assert busy["message"] == "robot is busy"

    t.join(timeout=2.0)
    assert bool(holder.get("auto", {}).get("ok")) is True


def test_step_session_start_next_cancel() -> None:
    manager, _ = make_manager()

    start = manager.execute_command("run_threepoint_step_start")
    assert start["ok"] is True
    assert start["step_progress"]["active"] is True

    one = manager.execute_command("run_threepoint_step_next")
    assert one["ok"] is True
    assert one["step_progress"]["index"] == 1

    cancel = manager.execute_command("run_threepoint_step_cancel")
    assert cancel["ok"] is True
    state = manager.get_state()
    assert state["step_progress"]["active"] is False


def test_set_speed_percent_updates_backend_and_state() -> None:
    manager, created = make_manager()

    result = manager.execute_command("set_speed_percent", {"percent": 42})
    assert result["ok"] is True
    assert created
    assert created[0].backend.speed_calls[-1] == 42

    state = manager.get_state()
    assert state["speed_percent"] == 42


def test_set_speed_percent_rejects_invalid_params() -> None:
    manager, _ = make_manager()

    result = manager.execute_command("set_speed_percent", {"percent": 101})
    assert result["ok"] is False
    assert result["error_code"] == "invalid_params"


def test_gravity_compensation_enable_disable_updates_state() -> None:
    manager, _ = make_manager()

    enabled = manager.execute_command("enable_gravity_compensation")
    assert enabled["ok"] is True

    state = manager.get_state()
    assert state["gravity_compensation_active"] is True
    assert state["ctrl_mode"] == 0x06
    assert "联动示教输入模式" in str(state["ctrl_mode_label"])

    disabled = manager.execute_command("disable_gravity_compensation")
    assert disabled["ok"] is True

    state2 = manager.get_state()
    assert state2["gravity_compensation_active"] is False
    assert state2["ctrl_mode"] == 0x01


def test_gravity_compensation_rejects_interlocked_motion_commands() -> None:
    manager, _ = make_manager()
    manager.execute_command("enable_gravity_compensation")

    blocked = manager.execute_command("home")
    assert blocked["ok"] is False
    assert blocked["error_code"] == "gravity_interlock"

    blocked_step = manager.execute_command("run_threepoint_step_start")
    assert blocked_step["ok"] is False
    assert blocked_step["error_code"] == "gravity_interlock"


def test_gravity_compensation_keeps_inferred_active_when_ctrl_mode_feedback_stays_can() -> None:
    manager, created = make_manager()
    manager.execute_command("status")
    tester = created[0]

    def _no_op_leader_mode() -> None:
        return None

    tester.backend.robot.set_leader_mode = _no_op_leader_mode  # type: ignore[assignment]
    enabled = manager.execute_command("enable_gravity_compensation")
    assert enabled["ok"] is True

    state = manager.get_state()
    assert state["gravity_compensation_active"] is True
    assert state["ctrl_mode"] == 0x06
    assert "推断" in str(state["ctrl_mode_label"])

    blocked = manager.execute_command("home")
    assert blocked["ok"] is False
    assert blocked["error_code"] == "gravity_interlock"


def test_gravity_compensation_command_reports_unsupported_backend() -> None:
    manager, created = make_manager()
    manager.execute_command("status")
    tester = created[0]
    tester.backend.robot.set_leader_mode = None  # type: ignore[assignment]

    result = manager.execute_command("enable_gravity_compensation")
    assert result["ok"] is False
    assert "unsupported" in str(result["message"]).lower()


def test_apply_pick_override_updates_pick_only() -> None:
    manager, created = make_manager()

    manager.execute_command("status")
    tester = created[0]
    ready_before = list(tester.threepoint_info["ready"]["pose_m_rad"])
    pick_before = list(tester.threepoint_info["pick"]["pose_m_rad"])
    transport_before = list(tester.threepoint_info["transport"]["pose_m_rad"])
    dump_before = list(tester.threepoint_info["dump"]["pose_m_rad"])

    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 120.0,
            "y_mm": 30.0,
            "z_mm": 480.0,
            "yaw_deg": 25.0,
            "close_width": 0.01,
            "force": 1.2,
            "smooth_segments": 3,
            "mode": "random",
        },
    )
    assert result["ok"] is True

    pick_after = tester.threepoint_info["pick"]["pose_m_rad"]
    assert pick_after != pick_before
    assert tester.threepoint_info["ready"]["pose_m_rad"] == ready_before
    assert tester.threepoint_info["transport"]["pose_m_rad"] == transport_before
    assert tester.threepoint_info["dump"]["pose_m_rad"] == dump_before

    state = manager.get_state()
    assert state["pick_override_active"] is True
    assert state["pick_override_mode"] == "random"
    assert abs(float(state["pick_override_close_width"]) - 0.01) < 1e-6
    assert abs(float(state["pick_override_force"]) - 1.2) < 1e-6
    assert state["pick_override_smooth_segments"] == 3


def test_step_cancel_clears_pick_override() -> None:
    manager, _ = make_manager()

    manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 1.0,
            "smooth_segments": 3,
            "mode": "dynamic",
        },
    )
    start = manager.execute_command("run_threepoint_step_start")
    assert start["ok"] is True

    cancel = manager.execute_command("run_threepoint_step_cancel")
    assert cancel["ok"] is True

    state = manager.get_state()
    assert state["pick_override_active"] is False
    assert state["pick_override_mode"] == "none"


def test_smooth_segments_apply_only_pick_and_pick_return() -> None:
    manager, created = make_manager()
    manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 1.0,
            "smooth_segments": 3,
            "mode": "random",
        },
    )
    manager.execute_command("run_threepoint_step_start")
    for _ in range(6):
        result = manager.execute_command("run_threepoint_step_next")
        assert result["ok"] is True

    labels = created[0].pose_motion_labels
    assert any(label.startswith("move pick seg1/3") for label in labels)
    assert any(label.startswith("move ready (keep gripper closed) seg1/3") for label in labels)
    assert not any(label.startswith("move transport seg") for label in labels)


def test_disable_is_available_while_busy_and_clears_override() -> None:
    manager, created = make_manager()

    manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 1.0,
            "smooth_segments": 3,
            "mode": "random",
        },
    )

    def worker() -> None:
        manager.execute_command("run_threepoint_auto")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if created and created[0].slow_event.is_set():
            break
        time.sleep(0.01)

    disable = manager.execute_command("disable")
    assert disable["ok"] is True
    assert created[0].backend.robot.disable_calls >= 1

    state = manager.get_state()
    assert state["pick_override_active"] is False

    t.join(timeout=2.0)


def test_reset_clear_errors_calls_reset_and_reenables() -> None:
    manager, created = make_manager()

    manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 1.0,
            "smooth_segments": 3,
            "mode": "random",
        },
    )

    result = manager.execute_command("reset_clear_errors")
    assert result["ok"] is True
    assert created[0].backend.robot.reset_calls >= 1
    assert created[0].backend.enable_calls >= 1
    assert any("reset command sent" in line for line in result["stdout_lines"])
    assert any("re-enable success" in line for line in result["stdout_lines"])

    state = manager.get_state()
    assert state["pick_override_active"] is False


def test_estop_is_available_while_busy() -> None:
    manager, created = make_manager()

    def worker() -> None:
        manager.execute_command("run_threepoint_auto")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if created and created[0].slow_event.is_set():
            break
        time.sleep(0.01)

    estop = manager.execute_command("estop")
    assert estop["ok"] is True
    assert created[0].backend.estop_calls >= 1

    t.join(timeout=2.0)


def test_create_tester_registers_module_before_exec(monkeypatch) -> None:
    manager = RobotControlManager(
        arm_config_path=Path("/tmp/fake_robot.yaml"),
        backend_override="real",
        enabled=True,
    )

    class FakeLoader:
        def exec_module(self, module: types.ModuleType) -> None:
            # Mirrors the dataclass crash seen when module is not registered in sys.modules.
            if sys.modules.get(module.__name__) is not module:
                raise AttributeError("'NoneType' object has no attribute '__dict__'")

            class NeroArmTester:
                def __init__(self, cfg_path: Path, backend_override: str | None) -> None:
                    _ = (cfg_path, backend_override)
                    self.cfg = {"logging": {}}

            module.NeroArmTester = NeroArmTester

    class FakeSpec:
        def __init__(self, name: str) -> None:
            self.name = name
            self.loader = FakeLoader()

    fake_module = types.ModuleType("nero_test_cli_web")

    def fake_spec_from_file_location(name: str, _path: Path) -> FakeSpec:
        return FakeSpec(name)

    def fake_module_from_spec(spec: FakeSpec) -> types.ModuleType:
        fake_module.__name__ = spec.name
        return fake_module

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr(importlib.util, "module_from_spec", fake_module_from_spec)
    monkeypatch.delitem(sys.modules, "nero_test_cli_web", raising=False)

    tester = manager._create_tester(Path("/tmp/fake_robot.yaml"), "real")
    assert tester is not None
    assert sys.modules.get("nero_test_cli_web") is fake_module
