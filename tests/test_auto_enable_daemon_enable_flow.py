from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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


class AutoEnableDaemonEnableFlowTests(unittest.TestCase):
    def _run_with_fake_clock(self, fn):
        clock = _FakeClock()
        with mock.patch.object(daemon.time, "time", side_effect=clock.time):
            with mock.patch.object(daemon.time, "sleep", side_effect=clock.sleep):
                return fn()

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


if __name__ == "__main__":
    unittest.main()
