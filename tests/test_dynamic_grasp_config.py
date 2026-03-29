from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_grasp.config import load_app_config


def _base_config_text(route_block: str) -> str:
    return f"""
vision:
  base_url: http://127.0.0.1:18000
handeye:
  mode: nominal
  extrinsics_path: ./模型文件/nero_description/config/handeye_extrinsics.yaml
scan:
  max_offset_xy_m: 0.18
grasp:
  open_width_m: 0.05
route:
{route_block}
runtime:
  arm_config_path: ./robot_runtime/config/default.yaml
"""


class DynamicGraspConfigTests(unittest.TestCase):
    def test_route_uses_transport_from_when_present(self) -> None:
        cfg_text = _base_config_text(
            """  ready_from: threepoint.ready
  transport_from: threepoint.transport
  dump_from: threepoint.dump
"""
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertEqual(cfg.route.ready_from, "threepoint.ready")
        self.assertEqual(cfg.route.transport_from, "threepoint.transport")
        self.assertEqual(cfg.route.dump_from, "threepoint.dump")

    def test_route_supports_legacy_dump_pre_from_alias(self) -> None:
        cfg_text = _base_config_text(
            """  ready_from: threepoint.scan
  dump_pre_from: threepoint.dump_pre
  dump_from: threepoint.dump
"""
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertEqual(cfg.route.transport_from, "threepoint.dump_pre")

    def test_vision_stability_defaults_and_overrides(self) -> None:
        cfg_text = _base_config_text(
            """  ready_from: threepoint.ready
  transport_from: threepoint.transport
  dump_from: threepoint.dump
"""
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertEqual(cfg.vision.plan_buffer_frames, 3)
        self.assertAlmostEqual(cfg.vision.max_replan_jump_mm, 25.0, places=6)
        self.assertAlmostEqual(cfg.vision.poll_hz, 12.0, places=6)
        self.assertAlmostEqual(cfg.vision.min_quality_score, 0.5, places=6)
        self.assertTrue(cfg.vision.reject_on_quality_drop)
        self.assertAlmostEqual(cfg.vision.stable_pos_tol_mm, 5.0, places=6)
        self.assertAlmostEqual(cfg.vision.stable_z_tol_mm, 5.0, places=6)
        self.assertAlmostEqual(cfg.vision.stable_yaw_tol_deg, 5.0, places=6)
        self.assertAlmostEqual(cfg.grasp.prepick_offset_m, 0.05, places=6)

    def test_grasp_prepick_offset_supports_legacy_hover_alias(self) -> None:
        cfg_text = """
vision:
  base_url: http://127.0.0.1:18000
handeye:
  mode: nominal
  extrinsics_path: ./模型文件/nero_description/config/handeye_extrinsics.yaml
scan:
  max_offset_xy_m: 0.18
grasp:
  open_width_m: 0.05
  hover_clearance_m: 0.06
route:
  ready_from: threepoint.ready
  transport_from: threepoint.transport
  dump_from: threepoint.dump
runtime:
  arm_config_path: ./robot_runtime/config/default.yaml
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertAlmostEqual(cfg.grasp.prepick_offset_m, 0.06, places=6)

    def test_grasp_prepick_offset_prefers_new_field_over_legacy_hover(self) -> None:
        cfg_text = """
vision:
  base_url: http://127.0.0.1:18000
handeye:
  mode: nominal
  extrinsics_path: ./模型文件/nero_description/config/handeye_extrinsics.yaml
scan:
  max_offset_xy_m: 0.18
grasp:
  open_width_m: 0.05
  prepick_offset_m: 0.05
  hover_clearance_m: 0.09
route:
  ready_from: threepoint.ready
  transport_from: threepoint.transport
  dump_from: threepoint.dump
runtime:
  arm_config_path: ./robot_runtime/config/default.yaml
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertAlmostEqual(cfg.grasp.prepick_offset_m, 0.05, places=6)

    def test_supports_unified_dynamic_grasp_root(self) -> None:
        cfg_text = """
dynamic_grasp:
  vision:
    base_url: http://127.0.0.1:18000
  handeye:
    mode: nominal
    extrinsics_path: ./模型文件/nero_description/config/handeye_extrinsics.yaml
  scan:
    max_offset_xy_m: 0.18
  grasp:
    open_width_m: 0.05
  route:
    ready_from: threepoint.ready
    transport_from: threepoint.transport
    dump_from: threepoint.dump
  runtime:
    speed_percent: 35
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(cfg_text)
            cfg_path = Path(handle.name)
        try:
            cfg = load_app_config(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)

        self.assertEqual(cfg.route.ready_from, "threepoint.ready")
        self.assertEqual(cfg.route.transport_from, "threepoint.transport")
        self.assertEqual(cfg.route.dump_from, "threepoint.dump")
        self.assertEqual(cfg.runtime.arm_config_path, cfg_path.resolve())


if __name__ == "__main__":
    unittest.main()
