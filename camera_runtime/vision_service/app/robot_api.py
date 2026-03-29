from __future__ import annotations

import ipaddress
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .robot_control import ALLOWED_COMMANDS, RobotControlManager


class RobotCommandRequest(BaseModel):
    command: str
    params: dict[str, Any] | None = None


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enforce_access(request: Request, loopback_only: bool) -> None:
    if not loopback_only:
        return
    host = request.client.host if request.client is not None else None
    if not _is_loopback_host(host):
        raise HTTPException(status_code=403, detail="robot control is loopback-only")


def build_robot_router(manager: RobotControlManager, enabled: bool, loopback_only: bool) -> APIRouter:
    router = APIRouter()

    @router.get("/api/robot/state")
    def robot_state(request: Request) -> dict[str, Any]:
        _enforce_access(request, loopback_only=loopback_only)
        state = manager.get_state()
        state["allowed_commands"] = sorted(ALLOWED_COMMANDS)
        state["enabled"] = bool(enabled)
        return state

    @router.post("/api/robot/command")
    def robot_command(payload: RobotCommandRequest, request: Request) -> dict[str, Any]:
        _enforce_access(request, loopback_only=loopback_only)
        normalized = payload.command.strip().lower()
        if normalized not in ALLOWED_COMMANDS:
            raise HTTPException(status_code=400, detail="unsupported command")

        result = manager.execute_command(payload.command, payload.params)
        result["allowed_commands"] = sorted(ALLOWED_COMMANDS)
        result["enabled"] = bool(enabled)

        ok = bool(result.get("ok", False))
        message = str(result.get("message", ""))
        error_code = str(result.get("error_code", "") or "")
        if not ok and error_code == "invalid_params":
            raise HTTPException(status_code=400, detail=message or "invalid params")
        if not ok and error_code == "gravity_interlock":
            raise HTTPException(status_code=409, detail=message or "gravity compensation interlock active")
        if not ok and message == "robot is busy":
            raise HTTPException(status_code=409, detail=message)
        if not ok and message == "robot web control disabled":
            raise HTTPException(status_code=503, detail=message)

        return result

    return router
