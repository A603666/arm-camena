from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime import nero_auto_enable_daemon as daemon  # noqa: E402


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += float(sec)


class _DelayedEnableRobot:
    def __init__(self) -> None:
        self.enable_calls = 0

    def get_joints_enable_status_list(self):
        if self.enable_calls < 4:
            return [False] * 7
        if self.enable_calls < 8:
            return [True, False, False, False, False, False, False]
        return [True] * 7

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7)

    def enable(self) -> bool:
        self.enable_calls += 1
        return False


class _NoFeedbackRobot:
    def get_joints_enable_status_list(self):
        raise RuntimeError("no feedback")

    def get_joint_angles(self):
        return None

    def enable(self) -> bool:
        return False


class _AllDisabledRobot:
    def get_joints_enable_status_list(self):
        return [False] * 7

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7)

    def enable(self) -> bool:
        return False


class _PartialNeverFullRobot:
    def get_joints_enable_status_list(self):
        return [True, False, False, False, False, False, False]

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7)

    def enable(self) -> bool:
        return False


class _SessionRobot:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.set_normal_mode_calls = 0
        self.reset_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def set_normal_mode(self) -> None:
        self.set_normal_mode_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1

    def get_joints_enable_status_list(self):
        return [False] * 7

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7)

    def enable(self) -> bool:
        return False


class AutoEnableDaemonEnableFlowTests(unittest.TestCase):
    def _run_with_fake_clock(self, fn):
        clock = _FakeClock()
        with mock.patch.object(daemon.time, "time", side_effect=clock.time):
            with mock.patch.object(daemon.time, "sleep", side_effect=clock.sleep):
                return fn()

    @staticmethod
    def _fake_cfg() -> daemon.DaemonConfig:
        return daemon.DaemonConfig(
            can_channel="can1",
            can_bitrate=1000000,
            can_interface="socketcan",
            usb_bus_info="1-2.1:1.0",
            can_scripts_dir=Path("/tmp"),
            retry_interval_sec=2.0,
            enable_timeout_sec=1.0,
            health_check_sec=1.0,
        )

    @staticmethod
    def _make_fake_pyagxarm(robot: _SessionRobot) -> ModuleType:
        fake_mod = ModuleType("pyAgxArm")

        class _Factory:
            @staticmethod
            def create_arm(_cfg):
                return robot

        def _create_agx_arm_config(**kwargs):
            return kwargs

        fake_mod.AgxArmFactory = _Factory  # type: ignore[attr-defined]
        fake_mod.create_agx_arm_config = _create_agx_arm_config  # type: ignore[attr-defined]
        return fake_mod

    def test_delayed_enable_succeeds_with_progress_grace(self) -> None:
        robot = _DelayedEnableRobot()

        def _run():
            return daemon.ensure_enabled(robot, timeout_sec=1.0, progress_grace_sec=0.8)

        enabled, flags, reason = self._run_with_fake_clock(_run)
        self.assertTrue(enabled)
        self.assertEqual(reason, "")
        self.assertEqual(flags, [True] * 7)

    def test_timeout_reason_no_feedback(self) -> None:
        robot = _NoFeedbackRobot()

        def _run():
            return daemon.ensure_enabled(robot, timeout_sec=0.6, progress_grace_sec=0.6)

        enabled, flags, reason = self._run_with_fake_clock(_run)
        self.assertFalse(enabled)
        self.assertIsNone(flags)
        self.assertEqual(reason, "no_feedback")

    def test_timeout_reason_all_disabled(self) -> None:
        robot = _AllDisabledRobot()

        def _run():
            return daemon.ensure_enabled(robot, timeout_sec=0.6, progress_grace_sec=0.6)

        enabled, flags, reason = self._run_with_fake_clock(_run)
        self.assertFalse(enabled)
        self.assertEqual(flags, [False] * 7)
        self.assertEqual(reason, "all_disabled")

    def test_timeout_reason_partial_enabled(self) -> None:
        robot = _PartialNeverFullRobot()

        def _run():
            return daemon.ensure_enabled(robot, timeout_sec=0.4, progress_grace_sec=0.4)

        enabled, flags, reason = self._run_with_fake_clock(_run)
        self.assertFalse(enabled)
        self.assertEqual(flags, [True, False, False, False, False, False, False])
        self.assertEqual(reason, "partial_enabled")

    def test_establish_session_retries_with_reset_after_all_disabled(self) -> None:
        cfg = self._fake_cfg()
        robot = _SessionRobot()
        logger = logging.getLogger("auto_enable_test_reset_success")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with mock.patch.dict(sys.modules, {"pyAgxArm": self._make_fake_pyagxarm(robot)}):
            with mock.patch.object(daemon, "find_can_iface_by_usb", return_value="can1"):
                with mock.patch.object(daemon, "activate_can", return_value=True):
                    with mock.patch.object(daemon, "wait_joint_ready", return_value=True):
                        with mock.patch.object(
                            daemon,
                            "ensure_enabled",
                            side_effect=[
                                (False, [False] * 7, "all_disabled"),
                                (True, [True] * 7, ""),
                            ],
                        ) as ensure_mock:
                            ok, session_robot, reason = daemon.establish_robot_session(cfg, logger)

        self.assertTrue(ok)
        self.assertIs(session_robot, robot)
        self.assertEqual(reason, "")
        self.assertEqual(ensure_mock.call_count, 2)
        self.assertEqual(robot.reset_calls, 1)
        self.assertEqual(robot.set_normal_mode_calls, 2)
        self.assertEqual(robot.disconnect_calls, 0)

    def test_establish_session_fails_when_reset_retry_still_all_disabled(self) -> None:
        cfg = self._fake_cfg()
        robot = _SessionRobot()
        logger = logging.getLogger("auto_enable_test_reset_fail")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with mock.patch.dict(sys.modules, {"pyAgxArm": self._make_fake_pyagxarm(robot)}):
            with mock.patch.object(daemon, "find_can_iface_by_usb", return_value="can1"):
                with mock.patch.object(daemon, "activate_can", return_value=True):
                    with mock.patch.object(daemon, "wait_joint_ready", return_value=True):
                        with mock.patch.object(
                            daemon,
                            "ensure_enabled",
                            side_effect=[
                                (False, [False] * 7, "all_disabled"),
                                (False, [False] * 7, "all_disabled"),
                            ],
                        ) as ensure_mock:
                            ok, session_robot, reason = daemon.establish_robot_session(cfg, logger)

        self.assertFalse(ok)
        self.assertIsNone(session_robot)
        self.assertEqual(reason, "joint enable timeout (all_disabled)")
        self.assertEqual(ensure_mock.call_count, 2)
        self.assertEqual(robot.reset_calls, 1)
        self.assertEqual(robot.disconnect_calls, 1)


if __name__ == "__main__":
    unittest.main()
