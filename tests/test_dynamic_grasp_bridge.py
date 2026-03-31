from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_grasp.bridge import RobotBridge, _load_python_module, _resolve_nero_cli_module_path


class _FakeSdkRobot:
    def __init__(self) -> None:
        self.normal_mode_calls = 0
        self.tcp_offsets: list[tuple[float, ...]] = []

    def set_normal_mode(self) -> None:
        self.normal_mode_calls += 1

    def set_tcp_offset(self, pose) -> None:
        self.tcp_offsets.append(tuple(float(v) for v in pose))


class _FakeBackend:
    def __init__(self, flags: list[bool]) -> None:
        self.robot = _FakeSdkRobot()
        self.effector = None
        self.speed_percent = 35
        self._flags = list(flags)
        self.enable_calls = 0
        self.last_move_error = ""

    def _get_joint_enable_flags(self):
        return list(self._flags)

    def diagnostics(self) -> dict[str, str]:
        return {
            "connected": "yes",
            "joint_feedback_alive": "yes",
            "joint_enable_flags": "".join("1" if flag else "0" for flag in self._flags),
        }

    def enable(self) -> bool:
        self.enable_calls += 1
        self._flags = [True] * 7
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.speed_percent = int(percent)

    def move_pose_p(self, pose, timeout: float) -> bool:
        return True

    def move_pose_l(self, pose, timeout: float) -> bool:
        return True


def _make_bridge(flags: list[bool]) -> RobotBridge:
    bridge = RobotBridge.__new__(RobotBridge)
    bridge.arm_config_path = Path("/tmp/default.yaml")
    bridge.arm_cfg = {}
    bridge.logger = logging.getLogger(f"dynamic_grasp_bridge_{id(flags)}")
    bridge.logger.handlers.clear()
    bridge.logger.addHandler(logging.NullHandler())
    bridge._move_timeout_sec = 5.0
    bridge._tcp_offset_pose = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    bridge._module = SimpleNamespace()
    bridge._backend = _FakeBackend(flags)
    bridge._route = SimpleNamespace(
        ready_flange_pose=(0.0, 0.0, 0.4, 0.0, 0.0, 0.0),
        transport_flange_pose=(0.2, 0.0, 0.4, 0.0, 0.0, 0.0),
        dump_flange_pose=(0.25, 0.0, 0.35, 0.0, 0.0, 0.0),
    )
    return bridge


class RobotBridgeTests(unittest.TestCase):
    def test_motion_session_health_fails_when_joint_disabled(self) -> None:
        bridge = _make_bridge([True, True, False, True, True, True, True])
        healthy, reason = bridge.motion_session_health()
        self.assertFalse(healthy)
        self.assertIn("joint_enable_flags=1101111", reason)

    def test_move_flange_recovers_before_motion(self) -> None:
        bridge = _make_bridge([True, True, False, True, True, True, True])
        self.assertTrue(bridge.move_flange_p([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]))
        self.assertEqual(bridge._backend.enable_calls, 1)
        self.assertEqual(bridge._backend.robot.normal_mode_calls, 1)
        self.assertEqual(bridge._backend.robot.tcp_offsets[-1], (0.0, 0.0, 0.1, 0.0, 0.0, 0.0))

    def test_resolve_route_pose_supports_transport_and_legacy_dump_pre(self) -> None:
        bridge = _make_bridge([True] * 7)
        transport = bridge.resolve_route_pose("threepoint.transport")
        legacy = bridge.resolve_route_pose("threepoint.dump_pre")
        self.assertEqual(transport, legacy)

    def test_resolve_nero_cli_module_path_supports_unified_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unified_cfg = root / "pipeline_config.yaml"
            unified_cfg.write_text("dynamic_grasp: {}\n", encoding="utf-8")
            expected = root / "robot_runtime" / "nero_test_cli.py"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("# test cli\n", encoding="utf-8")

            resolved = _resolve_nero_cli_module_path(unified_cfg)
            self.assertEqual(resolved, expected.resolve())

    def test_resolve_nero_cli_module_path_supports_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_cfg = root / "robot_runtime" / "config" / "default.yaml"
            legacy_cfg.parent.mkdir(parents=True, exist_ok=True)
            legacy_cfg.write_text("backend: real\n", encoding="utf-8")
            expected = root / "robot_runtime" / "nero_test_cli.py"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("# test cli\n", encoding="utf-8")

            resolved = _resolve_nero_cli_module_path(legacy_cfg)
            self.assertEqual(resolved, expected.resolve())

    def test_load_python_module_registers_in_sys_modules_for_dataclass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "dataclass_mod.py"
            module_path.write_text(
                "\n".join(
                    [
                        "from dataclasses import dataclass",
                        "import sys",
                        "MODULE_VISIBLE = (__name__ in sys.modules)",
                        "@dataclass",
                        "class Payload:",
                        "    value: int = 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            module_name = "dynamic_grasp_bridge_dataclass_test"
            previous = sys.modules.pop(module_name, None)
            try:
                module = _load_python_module(module_name, module_path)
            finally:
                if previous is not None:
                    sys.modules[module_name] = previous
                else:
                    sys.modules.pop(module_name, None)

            self.assertTrue(module.MODULE_VISIBLE)
            self.assertEqual(module.Payload(3).value, 3)


if __name__ == "__main__":
    unittest.main()
