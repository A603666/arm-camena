from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.nero_auto_enable_daemon import load_config  # noqa: E402


class AutoEnableUnifiedConfigTests(unittest.TestCase):
    def test_loads_auto_enable_from_unified_config(self) -> None:
        cfg = load_config(ROOT / "pipeline_config.yaml")
        self.assertEqual(cfg.can_channel, "can0")
        self.assertEqual(cfg.can_bitrate, 1000000)
        self.assertEqual(cfg.usb_bus_info, "1-2.2:1.0")

    def test_auto_enable_can_fields_can_fallback_to_robot_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "can_activate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            cfg_text = f"""
robot_runtime:
  can_channel: can0
  can_bitrate: 500000
  can_interface: socketcan
  can_tools:
    scripts_dir: {scripts_dir}
auto_enable:
  usb_bus_info: "1-2.4:1.0"
  retry_interval_sec: 2.0
  enable_timeout_sec: 8.0
  health_check_sec: 1.0
"""
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                handle.write(cfg_text)
                cfg_path = Path(handle.name)
            try:
                cfg = load_config(cfg_path)
            finally:
                cfg_path.unlink(missing_ok=True)

            self.assertEqual(cfg.can_channel, "can0")
            self.assertEqual(cfg.can_bitrate, 500000)
            self.assertEqual(cfg.can_interface, "socketcan")

    def test_accepts_can1_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "can_activate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            cfg_text = f"""
can_channel: can1
can_bitrate: 1000000
can_interface: socketcan
usb_bus_info: "1-2.4:1.0"
can_scripts_dir: {scripts_dir}
retry_interval_sec: 2.0
enable_timeout_sec: 8.0
health_check_sec: 1.0
"""
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                handle.write(cfg_text)
                cfg_path = Path(handle.name)
            try:
                cfg = load_config(cfg_path)
            finally:
                cfg_path.unlink(missing_ok=True)

            self.assertEqual(cfg.can_channel, "can1")

    def test_accepts_custom_can_channel_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "can_activate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            cfg_text = f"""
can_channel: can_piper
can_bitrate: 1000000
can_interface: socketcan
usb_bus_info: "1-2.4:1.0"
can_scripts_dir: {scripts_dir}
retry_interval_sec: 2.0
enable_timeout_sec: 8.0
health_check_sec: 1.0
"""
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                handle.write(cfg_text)
                cfg_path = Path(handle.name)
            try:
                cfg = load_config(cfg_path)
            finally:
                cfg_path.unlink(missing_ok=True)

            self.assertEqual(cfg.can_channel, "can_piper")

    def test_rejects_invalid_channel_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "can_activate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            cfg_text = f"""
can_channel: "can piper"
can_bitrate: 1000000
can_interface: socketcan
usb_bus_info: "1-2.4:1.0"
can_scripts_dir: {scripts_dir}
retry_interval_sec: 2.0
enable_timeout_sec: 8.0
health_check_sec: 1.0
"""
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                handle.write(cfg_text)
                cfg_path = Path(handle.name)
            try:
                with self.assertRaisesRegex(ValueError, "can_channel must be a valid interface name"):
                    load_config(cfg_path)
            finally:
                cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
