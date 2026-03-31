from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_grasp.vision_client import VisionClient, VisionClientError, VisionSnapshot


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


class DynamicGraspVisionTests(unittest.TestCase):
    def test_snapshot_parser_extracts_v2_fields(self) -> None:
        snapshot = VisionSnapshot.from_payload(
            {
                "schema_version": 2,
                "status": "ok",
                "target": {"bbox_xyxy": [10, 20, 30, 40]},
                "tracking": {"state": "locked", "confidence": 0.9, "reason": None},
                "depth": {"target_center_depth_mm": 800.0},
                "size": {"width_mm": 45.0},
                "quality": {"score": 0.72, "flags": []},
                "grasp": {
                    "x_mm": 10.0,
                    "y_mm": -20.0,
                    "z_mm": 800.0,
                    "axis_dir_cam": [1.0, 0.0, 0.0],
                },
            },
            api_version="v2",
        )
        self.assertTrue(snapshot.is_trackable)
        self.assertEqual(snapshot.schema_version, 2)
        self.assertEqual(snapshot.bbox_xyxy, (10, 20, 30, 40))
        self.assertEqual(snapshot.grasp_point_optical_m, (0.01, -0.02, 0.8))
        self.assertEqual(snapshot.axis_dir_optical, (1.0, 0.0, 0.0))

    def test_snapshot_without_grasp_is_not_trackable(self) -> None:
        snapshot = VisionSnapshot.from_payload(
            {"schema_version": 2, "status": "ok", "target": {"bbox_xyxy": [1, 2, 3, 4]}, "tracking": {"state": "locked"}},
            api_version="v2",
        )
        self.assertFalse(snapshot.is_trackable)

    def test_snapshot_parser_extracts_quality_fields(self) -> None:
        snapshot = VisionSnapshot.from_payload(
            {
                "schema_version": 2,
                "status": "ok",
                "target": {"bbox_xyxy": [10, 20, 30, 40]},
                "tracking": {"state": "locked", "confidence": 0.8},
                "grasp": {
                    "x_mm": 10.0,
                    "y_mm": -20.0,
                    "z_mm": 800.0,
                    "axis_dir_cam": [1.0, 0.0, 0.0],
                },
                "quality": {"score": 0.72, "flags": []},
            },
            api_version="v2",
        )
        self.assertAlmostEqual(snapshot.source_quality_score or 0.0, 0.72, places=6)
        self.assertEqual(snapshot.quality_flags, ())
        self.assertTrue(snapshot.quality_ok)

    def test_snapshot_parser_rejects_non_v2_api_version(self) -> None:
        with self.assertRaisesRegex(VisionClientError, "only supports api_version=v2"):
            VisionSnapshot.from_payload({"status": "ok"}, api_version="v1")

    def test_vision_client_rejects_non_v2_api_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports api_version=v2"):
            VisionClient("http://127.0.0.1:18000", timeout_sec=1.0, api_version="v1")

    @patch("dynamic_grasp.vision_client.urllib.request.urlopen")
    def test_health_request_parses_json(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            json.dumps(
                {
                    "service": "ok",
                    "stream_connected": True,
                    "seconds_since_last_frame": 0.1,
                    "seconds_since_last_update": 0.1,
                    "latest_status": "ok",
                }
            ).encode("utf-8")
        )
        client = VisionClient("http://127.0.0.1:18000", timeout_sec=1.0)
        health = client.get_health()
        self.assertTrue(health.stream_connected)
        self.assertEqual(health.service, "ok")

    @patch("dynamic_grasp.vision_client.urllib.request.urlopen")
    def test_bad_json_raises_client_error(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(b"{not-json")
        client = VisionClient("http://127.0.0.1:18000", timeout_sec=1.0)
        with self.assertRaises(VisionClientError):
            client.get_latest_target()


if __name__ == "__main__":
    unittest.main()
