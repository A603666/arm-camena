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
                "step_progress": {
                    "active": False,
                    "index": 0,
                    "total": 0,
                    "next_label": None,
                    "last_label": None,
                },
                "diag": {},
            },
        }
        self.next_result = {
            "ok": True,
            "command": "status",
            "message": "ok",
            "stdout_lines": [],
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

    def execute_command(self, command: str):
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
