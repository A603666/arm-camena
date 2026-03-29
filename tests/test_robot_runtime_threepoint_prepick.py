from __future__ import annotations

import logging
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.nero_test_cli import (  # noqa: E402
    PREPICK_OFFSET_M,
    THREEPOINT_EXEC_ROUTE,
    NeroArmTester,
)


class _StoreStub:
    def __init__(self) -> None:
        self.saved: list[tuple[str, list[float] | None]] = []

    def set(self, name: str, joints, pose) -> None:
        del pose
        self.saved.append((name, list(joints) if isinstance(joints, list) else None))

    def save(self) -> None:
        return None


class _BackendStub:
    def __init__(self, events: list[tuple]) -> None:
        self._events = events

    def open_gripper(self, width: float, force: float) -> bool:
        self._events.append(("open", round(float(width), 4), round(float(force), 3)))
        return True

    def close_gripper(self, width: float, force: float) -> bool:
        self._events.append(("close", round(float(width), 4), round(float(force), 3)))
        return True


class RobotRuntimePrepickTests(unittest.TestCase):
    def test_sync_threepoint_points_derives_prepick_from_pick_pose(self) -> None:
        tester = NeroArmTester.__new__(NeroArmTester)
        tester.threepoint_cfg = {
            "ready": {
                "joints_deg": [0.0, 10.0, 0.0, 80.0, 0.0, 0.0, 60.0],
                "pose_mm_deg": {"x": -300.0, "y": 0.0, "z": 300.0, "rx": 0.0, "ry": -90.0, "rz": 0.0},
            },
            "pick": {
                "joints_deg": [0.0, 20.0, 0.0, 90.0, 0.0, 0.0, 60.0],
                "pose_mm_deg": {"x": -390.0, "y": 5.0, "z": 180.0, "rx": 45.0, "ry": -88.0, "rz": 43.0},
            },
            "transport": {
                "joints_deg": [60.0, 22.0, 0.0, 90.0, 0.0, 0.0, 60.0],
                "pose_mm_deg": {"x": -50.0, "y": -350.0, "z": 295.0, "rx": -130.0, "ry": -85.0, "rz": -45.0},
            },
            "dump": {
                "joints_deg": [140.0, 24.0, 10.0, 96.0, 1.0, -7.0, 60.0],
                "pose_mm_deg": {"x": 340.0, "y": -80.0, "z": 303.0, "rx": -6.0, "ry": -75.0, "rz": -89.0},
            },
            "overwrite_waypoints_on_start": False,
        }
        tester.lock_orientation_from = "pick"
        tester.min_joint7_deg = 10.0
        tester.min_joint7_rad = math.radians(tester.min_joint7_deg)
        tester.store = _StoreStub()
        tester._normalize_target_joints = lambda joints, name: [float(v) for v in joints]  # type: ignore[method-assign]

        NeroArmTester._sync_threepoint_points(tester)
        pick_pose = tester.threepoint_info["pick"]["pose_m_rad"]
        prepick_pose = tester.threepoint_info["prepick"]["pose_m_rad"]

        self.assertAlmostEqual(prepick_pose[0], pick_pose[0], places=6)
        self.assertAlmostEqual(prepick_pose[1], pick_pose[1], places=6)
        self.assertAlmostEqual(prepick_pose[2] - pick_pose[2], PREPICK_OFFSET_M, places=6)
        self.assertAlmostEqual(prepick_pose[3], pick_pose[3], places=6)
        self.assertAlmostEqual(prepick_pose[4], pick_pose[4], places=6)
        self.assertAlmostEqual(prepick_pose[5], pick_pose[5], places=6)

    def test_threepoint_sequence_inserts_prepick_with_fixed_p_then_l(self) -> None:
        events: list[tuple] = []

        tester = NeroArmTester.__new__(NeroArmTester)
        tester.logger = logging.getLogger(f"threepoint_prepick_{id(events)}")
        tester.logger.handlers.clear()
        tester.logger.addHandler(logging.NullHandler())
        tester.strict_down_enabled = False
        tester.transfer_motion = "p"
        tester.approach_motion = "p"
        tester.in_step_prompt = False
        tester.cfg = {
            "gripper": {"open_width": 0.05, "close_width": 0.0, "force": 1.0},
            "task": {"save_closed_state_waypoint": "grip_closed_at_pick"},
        }
        tester.backend = _BackendStub(events)
        tester.threepoint_info = {
            name: {"pose_m_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "locked_pose_m_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
            for name in THREEPOINT_EXEC_ROUTE
        }
        tester._move_threepoint_pose = lambda name, label, motion: events.append(("move", name, motion)) or True  # type: ignore[method-assign]
        tester._save_runtime_state = lambda name: events.append(("save", name)) or True  # type: ignore[method-assign]

        ok = NeroArmTester._run_threepoint_sequence(tester, step_mode=False)
        self.assertTrue(ok)

        self.assertEqual(events[0], ("move", "ready", "p"))
        self.assertEqual(events[1], ("open", 0.05, 1.0))
        self.assertEqual(events[2], ("move", "prepick", "p"))
        self.assertEqual(events[3], ("move", "pick", "l"))
        self.assertEqual(events[4], ("close", 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
