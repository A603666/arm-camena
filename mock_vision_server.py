#!/usr/bin/env python3
"""Simple mock vision service for dynamic-grasp acceptance checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCE = [
    {
        "status": "no_target",
        "target": None,
        "depth": {"center_depth_mm": 820.0, "target_center_depth_mm": None},
        "size": None,
        "grasp": None,
        "timing": {"frame_id": 1, "ts_us": 1, "fps": 10.0},
    },
    {
        "status": "ok",
        "target": {"class_id": 0, "class_name": "mock", "conf": 0.95, "bbox_xyxy": [220, 160, 420, 360]},
        "depth": {"center_depth_mm": 820.0, "target_center_depth_mm": 760.0},
        "size": {"length_mm": 120.0, "width_mm": 35.0, "height_mm": 24.0},
        "grasp": {
            "x_mm": 32.0,
            "y_mm": -22.0,
            "z_mm": 760.0,
            "u": 320,
            "v": 240,
            "yaw_deg": 0.0,
            "axis_dir_cam": [1.0, 0.0, 0.0],
            "axis_quality": 0.95,
            "axis_state": "live",
        },
        "timing": {"frame_id": 2, "ts_us": 2, "fps": 10.0},
    },
    {
        "status": "ok",
        "target": {"class_id": 0, "class_name": "mock", "conf": 0.96, "bbox_xyxy": [240, 170, 400, 330]},
        "depth": {"center_depth_mm": 820.0, "target_center_depth_mm": 752.0},
        "size": {"length_mm": 118.0, "width_mm": 34.0, "height_mm": 24.0},
        "grasp": {
            "x_mm": 4.0,
            "y_mm": -3.0,
            "z_mm": 752.0,
            "u": 320,
            "v": 240,
            "yaw_deg": 0.0,
            "axis_dir_cam": [1.0, 0.0, 0.0],
            "axis_quality": 0.97,
            "axis_state": "live",
        },
        "timing": {"frame_id": 3, "ts_us": 3, "fps": 10.0},
    },
]


@dataclass
class ServerState:
    sequence: list[dict[str, Any]]
    loop: bool
    index: int = 0

    def latest_target(self) -> dict[str, Any]:
        if not self.sequence:
            raise RuntimeError("mock vision sequence is empty")
        payload = self.sequence[min(self.index, len(self.sequence) - 1)]
        if self.loop and self.sequence:
            self.index = (self.index + 1) % len(self.sequence)
        elif self.index < len(self.sequence) - 1:
            self.index += 1
        return payload


def load_sequence(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return list(DEFAULT_SEQUENCE)
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("mock scenario file must contain a JSON array")
    return [dict(item) for item in raw]


def build_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/health":
                self._send_json(
                    {
                        "service": "ok",
                        "stream_connected": True,
                        "seconds_since_last_frame": 0.02,
                        "seconds_since_last_update": 0.02,
                        "latest_status": state.sequence[min(state.index, len(state.sequence) - 1)].get("status", "unknown"),
                    }
                )
                return
            if self.path == "/api/latest-target":
                self._send_json(state.latest_target())
                return
            self.send_error(404, "Not Found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock vision service for dynamic grasp integration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--scenario", default=None, help="Optional JSON file that contains a target-result array")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop the scenario once it reaches the last frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = ServerState(sequence=load_sequence(args.scenario), loop=not args.no_loop)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"mock_vision_server=http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
