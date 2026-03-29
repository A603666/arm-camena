from __future__ import annotations

import ipaddress
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .calibration import CalibrationManager
from .robot_control import RobotControlManager


class CalibrationModeRequest(BaseModel):
    active: bool


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enforce_loopback(request: Request, loopback_only: bool) -> None:
    if not loopback_only:
        return
    host = request.client.host if request.client is not None else None
    if not _is_loopback_host(host):
        raise HTTPException(status_code=403, detail="calibration control is loopback-only")


def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def build_calibration_router(
    manager: CalibrationManager,
    robot_manager: RobotControlManager,
    loopback_only: bool,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/calibration/status")
    def calibration_status() -> dict[str, Any]:
        return manager.get_status()

    @router.post("/api/calibration/mode")
    def calibration_mode(payload: CalibrationModeRequest, request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        return manager.set_mode_active(bool(payload.active))

    @router.post("/api/calibration/capture")
    def calibration_capture(request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        robot_state = robot_manager.get_state()
        flange_pose = robot_state.get("flange_pose_m_rad")
        if flange_pose is None:
            raise HTTPException(status_code=409, detail="robot flange pose unavailable")
        try:
            return manager.capture_sample(flange_pose)
        except Exception as exc:
            raise _to_http_error(exc) from exc

    @router.post("/api/calibration/reset")
    def calibration_reset(request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        return manager.reset_session()

    @router.post("/api/calibration/compute")
    def calibration_compute(request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        try:
            return manager.compute_calibration()
        except Exception as exc:
            raise _to_http_error(exc) from exc

    @router.post("/api/calibration/apply")
    def calibration_apply(request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        try:
            return manager.apply_calibration()
        except Exception as exc:
            raise _to_http_error(exc) from exc

    @router.post("/api/calibration/rollback")
    def calibration_rollback(request: Request) -> dict[str, Any]:
        _enforce_loopback(request, loopback_only=loopback_only)
        try:
            return manager.rollback_last_apply()
        except Exception as exc:
            raise _to_http_error(exc) from exc

    return router
