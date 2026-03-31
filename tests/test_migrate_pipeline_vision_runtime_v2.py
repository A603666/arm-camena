from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_pipeline_vision_runtime_v2 import migrate_vision_runtime_section  # noqa: E402


class MigratePipelineVisionRuntimeV2Tests(unittest.TestCase):
    def test_migrate_moves_flat_keys_into_grouped_layout(self) -> None:
        section = {
            "web_port": 18000,
            "target_lock_hits": 2,
            "grasp_hold_frames": 7,
        }
        summary = migrate_vision_runtime_section(section)
        self.assertEqual(section["service"]["web_port"], 18000)
        self.assertEqual(section["tracking"]["target_lock_hits"], 2)
        self.assertEqual(section["stability"]["grasp_hold_frames"], 7)
        self.assertNotIn("web_port", section)
        self.assertNotIn("target_lock_hits", section)
        self.assertNotIn("grasp_hold_frames", section)
        self.assertIn("web_port", summary["moved"])
        self.assertIn("target_lock_hits", summary["moved"])
        self.assertIn("grasp_hold_frames", summary["moved"])

    def test_migrate_preserves_existing_grouped_values(self) -> None:
        section = {
            "service": {"web_port": 19000},
            "web_port": 18000,
        }
        summary = migrate_vision_runtime_section(section)
        self.assertEqual(section["service"]["web_port"], 19000)
        self.assertNotIn("web_port", section)
        self.assertIn("web_port", summary["skipped_existing"])
        self.assertIn("web_port", summary["removed_only"])


if __name__ == "__main__":
    unittest.main()
