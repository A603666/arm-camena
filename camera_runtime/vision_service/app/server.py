from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from .calibration import CalibrationManager
from .calibration_api import build_calibration_router
from .config import AppConfig, load_config
from .processor import VisionProcessor
from .robot_api import build_robot_router
from .robot_control import RobotControlManager
from .state import SharedState

config: AppConfig = load_config()
state = SharedState()
calibration_manager = CalibrationManager()
robot_manager = RobotControlManager(
    arm_config_path=config.robot_arm_config_path,
    backend_override=config.robot_backend_override,
    enabled=config.robot_control_enabled,
)
processor = VisionProcessor(config=config, state=state, calibration_manager=calibration_manager)

app = FastAPI(title="DaBai Vision Service", version="1.0.0")
static_dir = Path(__file__).resolve().parents[1] / "static"
app.include_router(
    build_robot_router(
        manager=robot_manager,
        enabled=config.robot_control_enabled,
        loopback_only=config.robot_loopback_only,
    )
)
app.include_router(
    build_calibration_router(
        manager=calibration_manager,
        robot_manager=robot_manager,
        loopback_only=config.robot_loopback_only,
    )
)


@app.on_event("startup")
def on_startup() -> None:
    processor.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    processor.stop()
    robot_manager.shutdown()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/calibration")
def calibration_index() -> FileResponse:
    return FileResponse(static_dir / "calibration.html")


@app.get("/api/health")
def health() -> dict:
    return state.get_health(stale_frame_threshold_sec=config.stale_frame_threshold_sec)


@app.get("/api/latest-target")
def latest_target() -> dict:
    return state.get_latest_result()


async def _mjpeg_generator() -> AsyncGenerator[bytes, None]:
    last_ref = None
    while True:
        frame = state.get_latest_annotated_jpeg()
        if frame is None:
            await asyncio.sleep(0.03)
            continue
        # Avoid flooding with exact same bytes when frame rate is low.
        if frame is last_ref:
            await asyncio.sleep(0.01)
            continue
        last_ref = frame
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
        )
        yield header + frame + b"\r\n"
        await asyncio.sleep(0.005)


@app.get("/stream/annotated")
async def stream_annotated() -> StreamingResponse:
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


async def _calibration_mjpeg_generator() -> AsyncGenerator[bytes, None]:
    last_ref = None
    while True:
        frame = calibration_manager.get_latest_preview_jpeg()
        if frame is None:
            await asyncio.sleep(0.03)
            continue
        if frame is last_ref:
            await asyncio.sleep(0.01)
            continue
        last_ref = frame
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
        )
        yield header + frame + b"\r\n"
        await asyncio.sleep(0.005)


@app.get("/stream/calibration")
async def stream_calibration() -> StreamingResponse:
    return StreamingResponse(_calibration_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/vision")
async def ws_vision(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(state.get_latest_result())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
