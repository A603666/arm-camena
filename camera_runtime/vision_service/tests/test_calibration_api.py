from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_service.app.calibration_api import CalibrationModeRequest, build_calibration_router


class StubCalibrationManager:
    def __init__(self) -> None:
        self.mode_active = False
        self.capture_calls = 0

    def get_status(self):
        return {
            "board": {"cols": 11, "rows": 8, "square_size_m": 0.015},
            "min_samples": 15,
            "sample_count": 0,
            "latest_detection": {"found": False, "method": "none", "frame_id": 0, "updated_at": None},
            "can_capture": False,
            "can_compute": False,
            "can_apply": False,
            "mode_active": self.mode_active,
            "latest_result_summary": None,
            "latest_result_path": None,
            "latest_backup_path": None,
            "handeye_yaml_path": "/tmp/handeye.yaml",
            "last_error": None,
        }

    def set_mode_active(self, active: bool):
        self.mode_active = bool(active)
        return {"ok": True, "mode_active": self.mode_active}

    def capture_sample(self, _pose):
        self.capture_calls += 1
        return {"ok": True, "sample_count": self.capture_calls}

    def reset_session(self):
        self.capture_calls = 0
        return {"ok": True, "sample_count": 0}

    def compute_calibration(self):
        raise ValueError("insufficient samples: need at least 15")

    def apply_calibration(self):
        raise ValueError("no calibration result to apply")

    def rollback_last_apply(self):
        raise ValueError("no backup available for rollback")


class StubRobotManager:
    def __init__(self, pose):
        self._pose = pose

    def get_state(self):
        return {
            "enabled": True,
            "connected": True,
            "busy": False,
            "flange_pose_m_rad": self._pose,
        }


def _fake_request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def _resolve_endpoint(router, path: str, method: str):
    for route in router.routes:
        route_path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if route_path == path and method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def test_calibration_mode_loopback_rejected() -> None:
    router = build_calibration_router(
        manager=StubCalibrationManager(),
        robot_manager=StubRobotManager([0, 0, 0, 0, 0, 0]),
        loopback_only=True,
    )
    endpoint = _resolve_endpoint(router, "/api/calibration/mode", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(CalibrationModeRequest(active=True), _fake_request("10.0.0.5"))
    assert "loopback" in str(exc.value).lower()


def test_calibration_capture_requires_robot_pose() -> None:
    router = build_calibration_router(
        manager=StubCalibrationManager(),
        robot_manager=StubRobotManager(None),
        loopback_only=False,
    )
    endpoint = _resolve_endpoint(router, "/api/calibration/capture", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(_fake_request("10.0.0.5"))
    assert "409" in str(exc.value)


def test_calibration_compute_returns_bad_request() -> None:
    router = build_calibration_router(
        manager=StubCalibrationManager(),
        robot_manager=StubRobotManager([0, 0, 0, 0, 0, 0]),
        loopback_only=False,
    )
    endpoint = _resolve_endpoint(router, "/api/calibration/compute", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(_fake_request("10.0.0.5"))
    assert "400" in str(exc.value)


def test_calibration_capture_success() -> None:
    manager = StubCalibrationManager()
    router = build_calibration_router(
        manager=manager,
        robot_manager=StubRobotManager([0, 0, 0, 0, 0, 0]),
        loopback_only=False,
    )
    endpoint = _resolve_endpoint(router, "/api/calibration/capture", "POST")

    payload = endpoint(_fake_request("10.0.0.5"))
    assert payload["ok"] is True
    assert payload["sample_count"] == 1
