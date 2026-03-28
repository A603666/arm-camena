from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_grasp.math_utils import HandEyeModel, plan_scan_xy


EXTRINSICS_PATH = ROOT / "模型文件" / "nero_description" / "config" / "handeye_extrinsics.yaml"


class DynamicGraspMathTests(unittest.TestCase):
    def test_nominal_tcp_offset_matches_handeye_description(self) -> None:
        model = HandEyeModel.from_yaml(EXTRINSICS_PATH, mode="nominal")
        self.assertAlmostEqual(model.tcp_offset_pose[0], 0.0, places=6)
        self.assertAlmostEqual(model.tcp_offset_pose[1], 0.0, places=6)
        self.assertAlmostEqual(model.tcp_offset_pose[2], 0.12743559577416268, places=6)

    def test_tcp_offset_composes_mount_and_base_chain(self) -> None:
        config_text = """
nominal_camera:
  xyz_m: [0.0, 0.0, 0.0]
  rpy_rad: [0.0, 0.0, 0.0]
  optical_xyz_m: [0.0, 0.0, 0.0]
  optical_rpy_rad: [0.0, 0.0, 0.0]
gripper_nominal:
  mount_xyz_m: [0.0, 0.0, 0.0]
  mount_rpy_rad: [0.0, 0.0, 1.5707963267948966]
  xyz_m: [0.01, 0.0, 0.0]
  rpy_rad: [0.0, 0.0, 0.0]
  tcp_xyz_m: [0.0, 0.0, 0.02]
  tcp_rpy_rad: [0.0, 0.0, 0.0]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(config_text)
            temp_path = Path(handle.name)
        try:
            model = HandEyeModel.from_yaml(temp_path, mode="nominal")
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertAlmostEqual(model.tcp_offset_pose[0], 0.0, places=6)
        self.assertAlmostEqual(model.tcp_offset_pose[1], 0.01, places=6)
        self.assertAlmostEqual(model.tcp_offset_pose[2], 0.02, places=6)
        self.assertAlmostEqual(model.tcp_offset_pose[5], math.pi / 2.0, places=6)

    def test_point_optical_origin_matches_nominal_camera_origin(self) -> None:
        model = HandEyeModel.from_yaml(EXTRINSICS_PATH, mode="nominal")
        origin = model.point_optical_to_base([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(origin[0]), 0.0598795391, places=6)
        self.assertAlmostEqual(float(origin[1]), 0.0, places=6)
        self.assertAlmostEqual(float(origin[2]), -0.0288839165, places=6)

    def test_axis_yaw_honors_alignment_offset(self) -> None:
        model = HandEyeModel(
            extrinsics_path=Path("/tmp/identity.yaml"),
            mode="nominal",
            flange_to_camera_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            camera_to_optical_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            tcp_offset_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        yaw = model.axis_yaw_in_base([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], yaw_offset_deg=90.0)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=6)

    def test_plan_scan_xy_clamps_to_window(self) -> None:
        planned = plan_scan_xy(
            ready_xy=[0.0, 0.0],
            current_xy=[0.17, 0.0],
            delta_xy=[0.10, 0.0],
            max_step_m=0.10,
            max_offset_m=0.18,
        )
        self.assertAlmostEqual(float(planned[0]), 0.18, places=6)
        self.assertAlmostEqual(float(planned[1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
