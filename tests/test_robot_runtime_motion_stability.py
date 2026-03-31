from __future__ import annotations

import logging
import math
import sys
import time
import types

from robot_runtime.nero_test_cli import RealBackend


class _StatusRobot:
    def __init__(self, arm_status: int, motion_status: int = 1) -> None:
        self.arm_status = int(arm_status)
        self.motion_status = int(motion_status)
        self.mode_feedback = 0x01

    def get_arm_status(self):
        return types.SimpleNamespace(
            msg=types.SimpleNamespace(
                ctrl_mode=0x01,
                arm_status=self.arm_status,
                mode_feedback=self.mode_feedback,
                motion_status=self.motion_status,
            )
        )


class _ConnectRobot:
    def __init__(self, enable_results: list[bool], enable_flags: list[bool] | None) -> None:
        self._enable_results = list(enable_results)
        self._enable_flags = None if enable_flags is None else list(enable_flags)
        self.disconnect_calls = 0
        self.set_normal_mode_calls = 0
        self.speed_percent_calls: list[int] = []

    def connect(self) -> None:
        return None

    def set_normal_mode(self) -> None:
        self.set_normal_mode_calls += 1

    def enable(self) -> bool:
        if self._enable_results:
            return bool(self._enable_results.pop(0))
        return False

    def get_joints_enable_status_list(self):
        if self._enable_flags is None:
            return None
        return list(self._enable_flags)

    def set_speed_percent(self, percent: int) -> None:
        self.speed_percent_calls.append(int(percent))

    def get_joint_angles(self):
        return types.SimpleNamespace(msg=[0.0] * 7)

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def _make_backend_with_status(arm_status: int, motion_status: int = 1) -> RealBackend:
    backend = RealBackend.__new__(RealBackend)
    backend.cfg = {}
    backend.robot = _StatusRobot(arm_status=arm_status, motion_status=motion_status)
    backend.connected = True
    backend.speed_percent = 10
    backend.last_enable_ok = True
    backend.last_connect_error = ""
    backend.effector = None
    backend.effector_init_error = "not_initialized"
    backend.last_joint_feedback_at = 0.0
    backend.last_move_error = ""
    return backend


def test_pose_error_uses_so3_geodesic_instead_of_euler_component_max() -> None:
    ready_pose = [
        -319.854 / 1000.0,
        -2.171 / 1000.0,
        311.011 / 1000.0,
        math.radians(-144.969),
        math.radians(-81.967),
        math.radians(-125.617),
    ]
    pick_pose = [
        -391.144 / 1000.0,
        -0.064 / 1000.0,
        182.859 / 1000.0,
        math.radians(44.440),
        math.radians(-88.356),
        math.radians(43.407),
    ]

    _, so3_rot = RealBackend._pose_error(ready_pose, pick_pose)
    euler_component_max = max(
        abs(math.atan2(math.sin(ready_pose[i] - pick_pose[i]), math.cos(ready_pose[i] - pick_pose[i])))
        for i in range(3, 6)
    )

    assert math.degrees(euler_component_max) > 150.0
    assert 5.0 <= math.degrees(so3_rot) <= 20.0


def test_wait_motion_done_fails_fast_on_fatal_arm_status() -> None:
    backend = _make_backend_with_status(arm_status=3)

    start = time.time()
    ok = backend._wait_motion_done(timeout=5.0, target=[0.0] * 7, start=[0.0] * 7)
    elapsed = time.time() - start

    assert ok is False
    assert elapsed < 1.0
    assert "singularity" in backend.last_move_error


def test_wait_pose_done_fails_fast_on_fatal_arm_status() -> None:
    backend = _make_backend_with_status(arm_status=2)

    start = time.time()
    ok = backend._wait_pose_done(timeout=5.0, target_pose=[0.0] * 6, start_pose=[0.0] * 6)
    elapsed = time.time() - start

    assert ok is False
    assert elapsed < 1.0
    assert "no_solution" in backend.last_move_error


def test_connect_fails_when_enable_does_not_succeed(monkeypatch) -> None:
    import robot_runtime.nero_test_cli as cli

    robot = _ConnectRobot(enable_results=[False] * 3, enable_flags=[True] * 7)
    fake_pyagxarm = types.SimpleNamespace(
        AgxArmFactory=types.SimpleNamespace(create_arm=lambda _cfg: robot),
        create_agx_arm_config=lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setitem(sys.modules, "pyAgxArm", fake_pyagxarm)
    monkeypatch.setattr(cli.time, "sleep", lambda _sec: None)

    backend = RealBackend(cfg={"speed_percent": 20}, logger=logging.getLogger("connect_enable_guard"))
    ok = backend.connect()

    assert ok is False
    assert backend.connected is False
    assert backend.last_enable_ok is False
    assert "enable()" in backend.last_connect_error
    assert robot.disconnect_calls >= 1


def test_connect_fails_when_joint_enable_flags_are_not_all_true(monkeypatch) -> None:
    import robot_runtime.nero_test_cli as cli

    robot = _ConnectRobot(enable_results=[True], enable_flags=[True, True, True, True, False, True, True])
    fake_pyagxarm = types.SimpleNamespace(
        AgxArmFactory=types.SimpleNamespace(create_arm=lambda _cfg: robot),
        create_agx_arm_config=lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setitem(sys.modules, "pyAgxArm", fake_pyagxarm)
    monkeypatch.setattr(cli.time, "sleep", lambda _sec: None)

    backend = RealBackend(cfg={"speed_percent": 20}, logger=logging.getLogger("connect_flag_guard"))
    ok = backend.connect()

    assert ok is False
    assert backend.connected is False
    assert "joint_enable_flags" in backend.last_connect_error
    assert robot.disconnect_calls >= 1


def test_connect_passes_cfg_joint_limits_to_pyagxarm(monkeypatch) -> None:
    import robot_runtime.nero_test_cli as cli

    robot = _ConnectRobot(enable_results=[True], enable_flags=[True] * 7)
    captured: dict[str, object] = {}

    def fake_create_agx_arm_config(**kwargs):
        captured.update(kwargs)
        return dict(kwargs)

    fake_pyagxarm = types.SimpleNamespace(
        AgxArmFactory=types.SimpleNamespace(create_arm=lambda _cfg: robot),
        create_agx_arm_config=fake_create_agx_arm_config,
    )
    monkeypatch.setitem(sys.modules, "pyAgxArm", fake_pyagxarm)
    monkeypatch.setattr(cli.time, "sleep", lambda _sec: None)

    backend = RealBackend(
        cfg={
            "speed_percent": 20,
            "joint_limits": {
                "joint1": [-2.0, 2.0],
                "joint2": [-1.0, 1.0],
                "joint3": [-1.5, 1.5],
                "joint4": [-1.2, 1.2],
                "joint5": [-2.5, 2.5],
                "joint6": [-0.7, 0.9],
                "joint7": [-1.8, 1.8],
            },
        },
        logger=logging.getLogger("connect_joint_limit_override"),
    )
    ok = backend.connect()

    assert ok is True
    assert captured.get("joint_limits") == backend.cfg["joint_limits"]


def test_disconnect_calls_robot_disconnect_and_clears_runtime_state() -> None:
    backend = RealBackend(cfg={"speed_percent": 10}, logger=logging.getLogger("disconnect_guard"))
    robot = _ConnectRobot(enable_results=[True], enable_flags=[True] * 7)
    backend.robot = robot
    backend.effector = object()
    backend.effector_init_error = ""
    backend.connected = True
    backend.last_enable_ok = True

    backend.disconnect()

    assert robot.disconnect_calls == 1
    assert backend.robot is None
    assert backend.effector is None
    assert backend.connected is False
    assert backend.last_enable_ok is False
