from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest

from vision_service.app.robot_control import RobotControlManager


@pytest.fixture(autouse=True)
def _default_exclusive_control(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_ROBOT_EXCLUSIVE_CONTROL", "1")


class FakeBackend:
    def __init__(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.disable_calls = 0
                self.reset_calls = 0
                self.ctrl_mode = 0x01
                self.tcp_offset = [0.0] * 6
                self.last_tcp_pose: list[float] | None = None

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
                return types.SimpleNamespace(
                    msg=types.SimpleNamespace(
                        ctrl_mode=self.ctrl_mode,
                    )
                )

            def set_tcp_offset(self, pose_m_rad) -> None:
                self.tcp_offset = [float(v) for v in pose_m_rad]

            def get_tcp2flange_pose(self, tcp_pose):
                pose = [float(v) for v in tcp_pose]
                self.last_tcp_pose = list(pose)
                return [
                    pose[0] - float(self.tcp_offset[0]),
                    pose[1] - float(self.tcp_offset[1]),
                    pose[2] - float(self.tcp_offset[2]),
                    pose[3],
                    pose[4],
                    pose[5],
                ]

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
        ready_pose = [0.0, 0.0, 0.60, 0.0, 0.0, 0.0]
        pick_pose = [0.0, 0.0, 0.20, 0.0, 0.0, 0.0]
        prepick_pose = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]
        self.threepoint_info = {
            "ready": {"pose_m_rad": list(ready_pose), "locked_pose_m_rad": list(ready_pose)},
            "prepick": {"pose_m_rad": list(prepick_pose), "locked_pose_m_rad": list(prepick_pose)},
            "pick": {"pose_m_rad": list(pick_pose), "locked_pose_m_rad": list(pick_pose)},
            "transport": {"pose_m_rad": [0.2, 0.0, 0.3, 0.0, 0.0, 0.0], "locked_pose_m_rad": [0.2, 0.0, 0.3, 0.0, 0.0, 0.0]},
            "dump": {"pose_m_rad": [0.3, 0.0, 0.3, 0.0, 0.0, 0.0], "locked_pose_m_rad": [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]},
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
    manager._handeye_tcp_offset_m_rad = [0.0, 0.0, 0.10, 0.0, 0.0, 0.0]  # type: ignore[attr-defined]
    manager._pick_override_safety = {  # type: ignore[attr-defined]
        "prepick_offset_m": 0.05,
        "max_descent_m": 0.35,
        "min_safe_z_m": 0.10,
    }
    manager._handeye_source_effective = "calibrated"  # type: ignore[attr-defined]
    manager._handeye_error = None  # type: ignore[attr-defined]
    return manager, created


def test_manager_connects_on_demand() -> None:
    manager, created = make_manager()

    state0 = manager.get_state()
    assert state0["connected"] is False
    assert state0["dynamic_stability_config"]["window"] == 18
    assert state0["handeye_mode_required"] == "calibrated"
    assert state0["handeye_source_effective"] == "calibrated"
    assert state0["handeye_ready"] is True
    assert state0["handeye_error"] is None

    result = manager.execute_command("status")
    assert result["ok"] is True
    assert created
    assert created[0].backend.connect_calls == 1


def test_pick_override_safety_config_allows_negative_min_safe_z() -> None:
    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "pipeline_config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "dynamic_grasp:",
                    "  grasp:",
                    "    prepick_offset_m: 0.05",
                    "    max_descent_m: 0.35",
                    "    min_safe_z_m: -0.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        safety = RobotControlManager._load_pick_override_safety_config(cfg_path)
        assert float(safety["min_safe_z_m"]) == pytest.approx(-0.3, abs=1e-9)


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


def test_daemon_conflict_rejects_motion_commands(monkeypatch) -> None:
    monkeypatch.delenv("DABAI_ROBOT_EXCLUSIVE_CONTROL", raising=False)
    monkeypatch.setattr(RobotControlManager, "_is_auto_enable_service_active", staticmethod(lambda: True))
    manager, _ = make_manager()

    blocked = manager.execute_command("home")
    assert blocked["ok"] is False
    assert blocked["error_code"] == "daemon_conflict"

    non_motion = manager.execute_command("status")
    assert non_motion["ok"] is True


def test_daemon_conflict_allows_motion_when_exclusive_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_ROBOT_EXCLUSIVE_CONTROL", "1")
    monkeypatch.setattr(RobotControlManager, "_is_auto_enable_service_active", staticmethod(lambda: True))
    manager, _ = make_manager()

    result = manager.execute_command("home")
    assert result["ok"] is True


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


def test_apply_pick_override_updates_pick_and_prepick() -> None:
    manager, created = make_manager()

    manager.execute_command("status")
    tester = created[0]
    ready_before = list(tester.threepoint_info["ready"]["pose_m_rad"])
    prepick_before = list(tester.threepoint_info["prepick"]["pose_m_rad"])
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
    prepick_after = tester.threepoint_info["prepick"]["pose_m_rad"]
    assert pick_after != pick_before
    assert prepick_after != prepick_before
    assert abs(float(prepick_after[2]) - float(pick_after[2]) - 0.05) < 1e-6
    assert tester.threepoint_info["ready"]["pose_m_rad"] == ready_before
    assert tester.threepoint_info["transport"]["pose_m_rad"] == transport_before
    assert tester.threepoint_info["dump"]["pose_m_rad"] == dump_before

    state = manager.get_state()
    assert state["pick_override_active"] is True
    assert state["pick_override_mode"] == "random"
    assert abs(float(state["pick_override_close_width"]) - 0.01) < 1e-6
    assert abs(float(state["pick_override_force"]) - 1.2) < 1e-6
    assert state["pick_override_smooth_segments"] == 3


def test_clear_pick_override_restores_pick_and_prepick() -> None:
    manager, created = make_manager()
    manager.execute_command("status")
    tester = created[0]
    pick_before = list(tester.threepoint_info["pick"]["pose_m_rad"])
    prepick_before = list(tester.threepoint_info["prepick"]["pose_m_rad"])

    applied = manager.execute_command(
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
    assert applied["ok"] is True
    assert tester.threepoint_info["pick"]["pose_m_rad"] != pick_before
    assert tester.threepoint_info["prepick"]["pose_m_rad"] != prepick_before

    cleared = manager.execute_command("clear_pick_override")
    assert cleared["ok"] is True
    assert tester.threepoint_info["pick"]["pose_m_rad"] == pick_before
    assert tester.threepoint_info["prepick"]["pose_m_rad"] == prepick_before


def test_apply_pick_override_uses_tcp_to_flange_transform() -> None:
    manager, created = make_manager()
    manager._handeye_tcp_offset_m_rad = [0.0, 0.0, 0.20, 0.0, 0.0, 0.0]  # type: ignore[attr-defined]
    manager.execute_command("status")
    tester = created[0]

    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 500.0,
            "yaw_deg": 0.0,
            "close_width": 0.0,
            "force": 1.0,
            "smooth_segments": 1,
            "mode": "dynamic",
        },
    )
    assert result["ok"] is True
    assert tester.backend.robot.last_tcp_pose is not None
    assert abs(float(tester.backend.robot.last_tcp_pose[2]) - 0.5) < 1e-6
    pick_after = tester.threepoint_info["pick"]["pose_m_rad"]
    assert abs(float(pick_after[2]) - 0.3) < 1e-6


def test_apply_pick_override_rejects_if_pick_below_min_safe_z() -> None:
    manager, created = make_manager()
    manager._pick_override_safety = {  # type: ignore[attr-defined]
        "prepick_offset_m": 0.05,
        "max_descent_m": 0.35,
        "min_safe_z_m": 0.40,
    }
    manager.execute_command("status")
    tester = created[0]
    pick_before = list(tester.threepoint_info["pick"]["pose_m_rad"])

    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 300.0,
            "yaw_deg": 0.0,
            "close_width": 0.0,
            "force": 1.0,
            "smooth_segments": 1,
            "mode": "dynamic",
        },
    )
    assert result["ok"] is False
    assert "min_safe_z" in str(result["message"])
    assert tester.threepoint_info["pick"]["pose_m_rad"] == pick_before


def test_apply_pick_override_rejects_when_calibrated_handeye_unavailable() -> None:
    manager, created = make_manager()
    manager._flange_to_optical = None  # type: ignore[attr-defined]
    manager._handeye_source_effective = "unavailable"  # type: ignore[attr-defined]
    manager._handeye_error = "calibrated handeye unavailable: calibrated_camera.enabled=true is required"  # type: ignore[attr-defined]
    manager.execute_command("status")
    tester = created[0]
    pick_before = list(tester.threepoint_info["pick"]["pose_m_rad"])

    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 300.0,
            "yaw_deg": 0.0,
            "close_width": 0.0,
            "force": 1.0,
            "smooth_segments": 1,
            "mode": "dynamic",
        },
    )
    assert result["ok"] is False
    assert "calibrated handeye unavailable" in str(result["message"])
    assert tester.threepoint_info["pick"]["pose_m_rad"] == pick_before


def test_apply_pick_override_rejects_if_descent_exceeds_limit() -> None:
    manager, created = make_manager()
    manager._pick_override_safety = {  # type: ignore[attr-defined]
        "prepick_offset_m": 0.05,
        "max_descent_m": 0.05,
        "min_safe_z_m": 0.10,
    }
    manager.execute_command("status")
    tester = created[0]
    pick_before = list(tester.threepoint_info["pick"]["pose_m_rad"])

    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 480.0,
            "yaw_deg": 0.0,
            "close_width": 0.0,
            "force": 1.0,
            "smooth_segments": 1,
            "mode": "dynamic",
        },
    )
    assert result["ok"] is False
    assert "max_descent" in str(result["message"])
    assert tester.threepoint_info["pick"]["pose_m_rad"] == pick_before


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


def test_smooth_segments_degrade_to_single_move_l_with_notice() -> None:
    manager, created = make_manager()
    manager.execute_command("status")
    assert created
    created[0].approach_motion = "l"

    apply_result = manager.execute_command(
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
    assert apply_result["ok"] is True
    assert "fallback to single move_l" in str(apply_result["message"])

    start = manager.execute_command("run_threepoint_step_start")
    assert start["ok"] is True
    manager.execute_command("run_threepoint_step_next")  # move ready
    manager.execute_command("run_threepoint_step_next")  # open gripper
    pick_step = manager.execute_command("run_threepoint_step_next")  # move pick
    assert pick_step["ok"] is True
    assert any("fallback to single move_l" in str(line) for line in pick_step["stdout_lines"])

    labels = created[0].pose_motion_labels
    assert "move pick" in labels
    assert not any(label.startswith("move pick seg") for label in labels)


def test_apply_pick_override_rejects_segmented_motion_l_param() -> None:
    manager, _ = make_manager()
    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 1.0,
            "smooth_segments": 3,
            "segmented_motion": "l",
            "mode": "random",
        },
    )
    assert result["ok"] is False
    assert "segmented_motion must be 'p'" in str(result["message"])


def test_apply_pick_override_rejects_close_width_above_official_limit() -> None:
    manager, _ = make_manager()
    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.11,
            "force": 1.0,
            "smooth_segments": 3,
            "mode": "random",
        },
    )
    assert result["ok"] is False
    assert result["error_code"] == "invalid_params"
    assert "close_width must be <= 0.1" in str(result["message"])


def test_apply_pick_override_rejects_force_above_official_limit() -> None:
    manager, _ = make_manager()
    result = manager.execute_command(
        "apply_pick_override_from_vision",
        {
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 500.0,
            "yaw_deg": 10.0,
            "close_width": 0.01,
            "force": 3.1,
            "smooth_segments": 3,
            "mode": "random",
        },
    )
    assert result["ok"] is False
    assert result["error_code"] == "invalid_params"
    assert "force must be <= 3.0" in str(result["message"])


def test_auto_and_step_start_share_same_step_builder(monkeypatch) -> None:
    created: list[FakeTester] = []

    class StepAwareTester(FakeTester):
        def __init__(self, cfg_path: Path, backend_override: str | None) -> None:
            super().__init__(cfg_path, backend_override)
            self.executed_labels: list[tuple[str, bool]] = []

        def _run_steps(self, steps, step_mode: bool) -> bool:  # type: ignore[override]
            for label, action in steps:
                self.executed_labels.append((str(label), bool(step_mode)))
                if not bool(action()):
                    return False
            return True

    def factory(cfg_path: Path, backend_override: str | None) -> FakeTester:
        tester = StepAwareTester(cfg_path, backend_override)
        created.append(tester)
        return tester

    manager = RobotControlManager(
        arm_config_path=Path("/tmp/fake_robot.yaml"),
        backend_override="real",
        enabled=True,
        tester_factory=factory,
    )
    manager._flange_to_optical = np.eye(4, dtype=np.float64)  # type: ignore[attr-defined]

    build_calls: list[str] = []

    def fake_builder(_tester):
        build_calls.append("build")
        return [("mock-step", lambda: True)]

    monkeypatch.setattr(manager, "_build_threepoint_steps", fake_builder)

    auto = manager.execute_command("run_threepoint_auto")
    assert auto["ok"] is True

    start = manager.execute_command("run_threepoint_step_start")
    assert start["ok"] is True

    next_step = manager.execute_command("run_threepoint_step_next")
    assert next_step["ok"] is True
    assert next_step["message"] == "step done: mock-step"
    assert build_calls == ["build", "build"]
    assert created
    assert ("mock-step", False) in created[0].executed_labels


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
