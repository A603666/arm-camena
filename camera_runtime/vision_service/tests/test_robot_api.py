from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_service.app.robot_api import RobotCommandRequest, build_robot_router


class StubManager:
    def __init__(self) -> None:
        self.state = {
            "enabled": True,
            "connected": False,
            "busy": False,
            "speed_percent": 30,
            "joint_positions_rad": None,
            "flange_pose_m_rad": None,
            "diag": {},
            "step_progress": {
                "active": False,
                "index": 0,
                "total": 0,
                "next_label": None,
                "last_label": None,
            },
            "last_result": {
                "ok": False,
                "command": "",
                "message": "idle",
                "stdout_lines": [],
                "error_code": None,
                "step_progress": {
                    "active": False,
                    "index": 0,
                    "total": 0,
                    "next_label": None,
                    "last_label": None,
                },
                "diag": {},
            },
            "pick_override_active": False,
            "pick_override_mode": "none",
            "pick_override_close_width": None,
            "pick_override_force": None,
            "pick_override_smooth_segments": None,
            "dynamic_stability_config": {
                "window": 18,
                "min_axis_quality": 0.85,
                "mad_limit": {"x": 1.5, "y": 1.5, "z": 2.0, "yaw": 1.2},
            },
            "gravity_compensation_active": False,
            "ctrl_mode": None,
            "ctrl_mode_label": "未知",
            "handeye_mode_required": "calibrated",
            "handeye_source_effective": "calibrated",
            "handeye_ready": True,
            "handeye_error": None,
        }
        self.next_result = {
            "ok": True,
            "command": "status",
            "message": "ok",
            "stdout_lines": [],
            "error_code": None,
            "step_progress": {
                "active": False,
                "index": 0,
                "total": 0,
                "next_label": None,
                "last_label": None,
            },
            "diag": {},
        }

    def get_state(self):
        return dict(self.state)

    def execute_command(self, command: str, params=None):
        _ = params
        out = dict(self.next_result)
        out["command"] = command
        return out


def _fake_request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def _resolve_endpoint(router, path: str, method: str):
    for route in router.routes:
        route_path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if route_path == path and method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def test_robot_state_endpoint_returns_data() -> None:
    manager = StubManager()
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/state", "GET")

    payload = endpoint(_fake_request("10.0.0.5"))
    assert payload["enabled"] is True
    assert "allowed_commands" in payload
    assert payload["dynamic_stability_config"]["window"] == 18
    assert payload["dynamic_stability_config"]["mad_limit"]["yaw"] == pytest.approx(1.2)
    assert payload["handeye_mode_required"] == "calibrated"
    assert payload["handeye_source_effective"] == "calibrated"
    assert payload["handeye_ready"] is True


def test_robot_command_endpoint_allows_non_loopback_when_loopback_disabled() -> None:
    manager = StubManager()
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    payload = endpoint(RobotCommandRequest(command="status"), _fake_request("10.0.0.5"))
    assert payload["command"] == "status"
    assert payload["enabled"] is True


def test_robot_command_rejects_unsupported() -> None:
    manager = StubManager()
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(RobotCommandRequest(command="not_exists"), _fake_request("10.0.0.5"))
    assert "unsupported" in str(exc.value).lower()


def test_robot_command_busy_raises_conflict() -> None:
    manager = StubManager()
    manager.next_result = {
        "ok": False,
        "command": "status",
        "message": "robot is busy",
        "stdout_lines": [],
        "error_code": None,
        "step_progress": {
            "active": False,
            "index": 0,
            "total": 0,
            "next_label": None,
            "last_label": None,
        },
        "diag": {},
    }
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(RobotCommandRequest(command="status"), _fake_request("10.0.0.5"))
    assert "409" in str(exc.value)


def test_robot_command_gravity_interlock_raises_conflict() -> None:
    manager = StubManager()
    manager.next_result = {
        "ok": False,
        "command": "home",
        "message": "gravity compensation is active; disable it before motion commands",
        "stdout_lines": [],
        "error_code": "gravity_interlock",
        "step_progress": {
            "active": False,
            "index": 0,
            "total": 0,
            "next_label": None,
            "last_label": None,
        },
        "diag": {},
    }
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(RobotCommandRequest(command="home"), _fake_request("10.0.0.5"))
    assert "409" in str(exc.value)


def test_robot_command_daemon_conflict_raises_conflict() -> None:
    manager = StubManager()
    manager.next_result = {
        "ok": False,
        "command": "home",
        "message": "daemon conflict: nero-auto-enable.service is active; stop it or relaunch vision with exclusive robot control",
        "stdout_lines": [],
        "error_code": "daemon_conflict",
        "step_progress": {
            "active": False,
            "index": 0,
            "total": 0,
            "next_label": None,
            "last_label": None,
        },
        "diag": {},
    }
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(RobotCommandRequest(command="home"), _fake_request("10.0.0.5"))
    assert "409" in str(exc.value)


def test_robot_command_invalid_params_raises_bad_request() -> None:
    manager = StubManager()
    manager.next_result = {
        "ok": False,
        "command": "set_speed_percent",
        "message": "invalid params: percent must be between 1 and 100",
        "stdout_lines": [],
        "error_code": "invalid_params",
        "step_progress": {
            "active": False,
            "index": 0,
            "total": 0,
            "next_label": None,
            "last_label": None,
        },
        "diag": {},
    }
    router = build_robot_router(manager=manager, enabled=True, loopback_only=False)
    endpoint = _resolve_endpoint(router, "/api/robot/command", "POST")

    with pytest.raises(Exception) as exc:
        endpoint(RobotCommandRequest(command="set_speed_percent", params={"percent": 101}), _fake_request("10.0.0.5"))
    assert "400" in str(exc.value)


def test_robot_endpoint_loopback_only_rejects_non_loopback() -> None:
    manager = StubManager()
    router = build_robot_router(manager=manager, enabled=True, loopback_only=True)
    endpoint = _resolve_endpoint(router, "/api/robot/state", "GET")

    with pytest.raises(Exception) as exc:
        endpoint(_fake_request("10.0.0.5"))
    assert "loopback" in str(exc.value).lower()


def test_robot_endpoint_loopback_only_accepts_localhost() -> None:
    manager = StubManager()
    router = build_robot_router(manager=manager, enabled=True, loopback_only=True)
    endpoint = _resolve_endpoint(router, "/api/robot/state", "GET")

    payload = endpoint(_fake_request("127.0.0.1"))
    assert payload["enabled"] is True
