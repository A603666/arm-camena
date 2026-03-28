from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

from vision_service.app.robot_control import RobotControlManager


class FakeBackend:
    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.enable_calls = 0
        self.estop_calls = 0

    def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    def set_speed_percent(self, percent: int) -> None:
        _ = percent

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
        pose = [0.0] * 6
        self.threepoint_info = {
            "ready": {"pose_m_rad": pose, "locked_pose_m_rad": pose},
            "pick": {"pose_m_rad": pose, "locked_pose_m_rad": pose},
            "transport": {"pose_m_rad": pose, "locked_pose_m_rad": pose},
            "dump": {"pose_m_rad": pose, "locked_pose_m_rad": pose},
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
