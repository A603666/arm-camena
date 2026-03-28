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


if __name__ == "__main__":
    unittest.main()
