from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unified_config_loader import (  # noqa: E402
    load_dynamic_grasp_section,
    load_robot_runtime_section,
    load_vision_env_map,
)


class UnifiedConfigLoaderTests(unittest.TestCase):
    def test_unified_config_sections_are_loadable(self) -> None:
        cfg_path = ROOT / "pipeline_config.yaml"

        dynamic_cfg, dynamic_unified = load_dynamic_grasp_section(cfg_path)
        robot_cfg, robot_unified = load_robot_runtime_section(cfg_path)

        self.assertTrue(dynamic_unified)
        self.assertTrue(robot_unified)
        self.assertIn("vision", dynamic_cfg)
        self.assertIn("threepoint", robot_cfg)
        self.assertEqual(robot_cfg.get("backend"), "real")

    def test_unified_config_builds_vision_env(self) -> None:
        cfg_path = (ROOT / "pipeline_config.yaml").resolve()
        env_map, is_unified = load_vision_env_map(cfg_path)

        self.assertTrue(is_unified)
        self.assertEqual(env_map.get("DABAI_ROBOT_CONFIG"), str(cfg_path))
        self.assertEqual(env_map.get("DABAI_YOLO_IMGSZ"), "512")
        self.assertEqual(env_map.get("DABAI_INFER_EVERY_N"), "2")
        self.assertEqual(env_map.get("DABAI_GEOM_BACKEND"), "cpu")
        self.assertEqual(env_map.get("DABAI_GROUND_FIT_FAST_ENABLED"), "1")

    def test_legacy_dynamic_grasp_config_is_supported(self) -> None:
        cfg_path = ROOT / "dynamic_grasp_config.yaml"
        dynamic_cfg, dynamic_unified = load_dynamic_grasp_section(cfg_path)
        env_map, env_unified = load_vision_env_map(cfg_path)

        self.assertFalse(dynamic_unified)
        self.assertIn("vision", dynamic_cfg)
        self.assertFalse(env_unified)
        self.assertEqual(env_map, {})


if __name__ == "__main__":
    unittest.main()
