from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.nero_test_cli import NeroArmTester  # noqa: E402


class RobotRuntimeUnifiedConfigTests(unittest.TestCase):
    def test_load_cfg_supports_unified_root(self) -> None:
        cfg = NeroArmTester._load_cfg((ROOT / "pipeline_config.yaml").resolve())
        self.assertEqual(cfg.get("backend"), "real")
        self.assertIn("threepoint", cfg)
        self.assertIn("can_tools", cfg)

    def test_unified_config_uses_safe_default_min_joint7_deg(self) -> None:
        cfg = NeroArmTester._load_cfg((ROOT / "pipeline_config.yaml").resolve())
        threepoint = cfg.get("threepoint", {})
        self.assertIsInstance(threepoint, dict)
        self.assertEqual(float(threepoint.get("min_joint7_deg")), 10.0)


if __name__ == "__main__":
    unittest.main()
