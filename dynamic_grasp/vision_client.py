from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class VisionClientError(RuntimeError):
    """Raised when the vision service cannot provide a valid JSON response."""


@dataclass(frozen=True)
class VisionHealth:
    service: str
    stream_connected: bool
    seconds_since_last_frame: float | None
    seconds_since_last_update: float | None
    latest_status: str


@dataclass(frozen=True)
class VisionSnapshot:
    status: str
    bbox_xyxy: tuple[int, int, int, int] | None
    grasp_point_optical_m: tuple[float, float, float] | None
    axis_dir_optical: tuple[float, float, float] | None
    target_center_depth_m: float | None
    width_m: float | None
    raw: dict[str, Any]

    @property
    def is_trackable(self) -> bool:
        return self.status == "ok" and self.grasp_point_optical_m is not None and self.axis_dir_optical is not None

    def signature(self) -> tuple[Any, ...]:
        point = None
        if self.grasp_point_optical_m is not None:
            point = tuple(round(v, 4) for v in self.grasp_point_optical_m)
        return (self.status, self.bbox_xyxy, point)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VisionSnapshot":
        if not isinstance(payload, dict):
            raise VisionClientError("vision payload must be a JSON object")

        target = payload.get("target")
        bbox_xyxy: tuple[int, int, int, int] | None = None
        if isinstance(target, dict):
            bbox = target.get("bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox_xyxy = tuple(int(v) for v in bbox)

        grasp_point: tuple[float, float, float] | None = None
        axis_dir: tuple[float, float, float] | None = None
        grasp = payload.get("grasp")
        if isinstance(grasp, dict):
            coords = [grasp.get("x_mm"), grasp.get("y_mm"), grasp.get("z_mm")]
            if all(v is not None for v in coords):
                grasp_point = tuple(float(v) / 1000.0 for v in coords)
            raw_axis = grasp.get("axis_dir_cam")
            if isinstance(raw_axis, list) and len(raw_axis) == 3:
                axis_dir = tuple(float(v) for v in raw_axis)

        target_center_depth_m: float | None = None
        depth = payload.get("depth")
        if isinstance(depth, dict) and depth.get("target_center_depth_mm") is not None:
            target_center_depth_m = float(depth["target_center_depth_mm"]) / 1000.0

        width_m: float | None = None
        size = payload.get("size")
        if isinstance(size, dict) and size.get("width_mm") is not None:
            width_m = float(size["width_mm"]) / 1000.0

        return cls(
            status=str(payload.get("status", "unknown")),
            bbox_xyxy=bbox_xyxy,
            grasp_point_optical_m=grasp_point,
            axis_dir_optical=axis_dir,
            target_center_depth_m=target_center_depth_m,
            width_m=width_m,
            raw=dict(payload),
        )


class VisionClient:
    def __init__(self, base_url: str, timeout_sec: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = float(timeout_sec)

    def _load_json(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                status = getattr(response, "status", 200)
                body = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise VisionClientError(f"vision request failed for {url}: {exc}") from exc

        if int(status) >= 400:
            raise VisionClientError(f"vision request failed for {url}: status={status}")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionClientError(f"vision response from {url} is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise VisionClientError(f"vision response from {url} must be a JSON object")
        return payload

    def get_health(self) -> VisionHealth:
        payload = self._load_json("/api/health")
        return VisionHealth(
            service=str(payload.get("service", "unknown")),
            stream_connected=bool(payload.get("stream_connected", False)),
            seconds_since_last_frame=(
                None if payload.get("seconds_since_last_frame") is None else float(payload["seconds_since_last_frame"])
            ),
            seconds_since_last_update=(
                None if payload.get("seconds_since_last_update") is None else float(payload["seconds_since_last_update"])
            ),
            latest_status=str(payload.get("latest_status", "unknown")),
        )

    def get_latest_target(self) -> VisionSnapshot:
        return VisionSnapshot.from_payload(self._load_json("/api/latest-target"))
