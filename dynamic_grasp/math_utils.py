from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose_m_rad: Iterable[float]) -> np.ndarray:
    pose = [float(v) for v in pose_m_rad]
    if len(pose) != 6:
        raise ValueError(f"pose must contain 6 values, got {len(pose)}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rpy_to_matrix(pose[3], pose[4], pose[5])
    matrix[:3, 3] = np.array(pose[:3], dtype=np.float64)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    rot = matrix[:3, :3]
    pitch = -math.asin(max(-1.0, min(1.0, float(rot[2, 0]))))
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    return [
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        normalize_angle_rad(roll),
        normalize_angle_rad(pitch),
        normalize_angle_rad(yaw),
    ]


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def transform_point(matrix: np.ndarray, point_xyz: Iterable[float]) -> np.ndarray:
    point = np.array([float(v) for v in point_xyz] + [1.0], dtype=np.float64)
    if point.shape != (4,):
        raise ValueError("point_xyz must contain exactly 3 values")
    transformed = np.asarray(matrix, dtype=np.float64) @ point
    return transformed[:3]


def transform_direction(matrix: np.ndarray, direction_xyz: Iterable[float]) -> np.ndarray:
    direction = np.array([float(v) for v in direction_xyz], dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("direction_xyz must contain exactly 3 values")
    transformed = np.asarray(matrix, dtype=np.float64)[:3, :3] @ direction
    norm = float(np.linalg.norm(transformed))
    if norm < 1e-9:
        raise ValueError("direction norm is too small after transform")
    return transformed / norm


def clamp_xy_step(delta_xy: Iterable[float], max_step_m: float) -> np.ndarray:
    delta = np.array([float(v) for v in delta_xy], dtype=np.float64)
    if delta.shape != (2,):
        raise ValueError("delta_xy must contain exactly 2 values")
    norm = float(np.linalg.norm(delta))
    if norm <= max_step_m or norm <= 1e-9:
        return delta
    return delta * (max_step_m / norm)


def clamp_xy_window(ready_xy: Iterable[float], candidate_xy: Iterable[float], max_offset_m: float) -> np.ndarray:
    ready = np.array([float(v) for v in ready_xy], dtype=np.float64)
    candidate = np.array([float(v) for v in candidate_xy], dtype=np.float64)
    if ready.shape != (2,) or candidate.shape != (2,):
        raise ValueError("ready_xy and candidate_xy must contain exactly 2 values")
    offset = candidate - ready
    norm = float(np.linalg.norm(offset))
    if norm <= max_offset_m or norm <= 1e-9:
        return candidate
    return ready + offset * (max_offset_m / norm)


def plan_scan_xy(
    ready_xy: Iterable[float],
    current_xy: Iterable[float],
    delta_xy: Iterable[float],
    max_step_m: float,
    max_offset_m: float,
) -> np.ndarray:
    current = np.array([float(v) for v in current_xy], dtype=np.float64)
    step = clamp_xy_step(delta_xy, max_step_m=max_step_m)
    return clamp_xy_window(ready_xy=ready_xy, candidate_xy=current + step, max_offset_m=max_offset_m)


@dataclass(frozen=True)
class HandEyeModel:
    extrinsics_path: Path
    mode: str
    flange_to_camera_pose: tuple[float, float, float, float, float, float]
    camera_to_optical_pose: tuple[float, float, float, float, float, float]
    tcp_offset_pose: tuple[float, float, float, float, float, float]

    @classmethod
    def from_yaml(cls, path: str | Path, mode: str = "nominal") -> "HandEyeModel":
        extrinsics_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(extrinsics_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("hand-eye config must be a mapping")

        nominal = raw.get("nominal_camera")
        calibrated = raw.get("calibrated_camera")
        gripper = raw.get("gripper_nominal")
        if not isinstance(nominal, dict) or not isinstance(gripper, dict):
            raise ValueError("hand-eye config is missing nominal_camera or gripper_nominal")

        selected_mode = str(mode).strip().lower() or "nominal"
        camera_source = nominal
        if selected_mode == "calibrated" and isinstance(calibrated, dict) and bool(calibrated.get("enabled", False)):
            camera_source = calibrated

        flange_to_camera_pose = tuple(
            float(v) for v in [
                *camera_source.get("xyz_m", nominal.get("xyz_m", [0.0, 0.0, 0.0])),
                *camera_source.get("rpy_rad", nominal.get("rpy_rad", [0.0, 0.0, 0.0])),
            ]
        )
        camera_to_optical_pose = tuple(
            float(v) for v in [
                *nominal.get("optical_xyz_m", [0.0, 0.0, 0.0]),
                *nominal.get("optical_rpy_rad", [0.0, 0.0, 0.0]),
            ]
        )

        flange_to_gripper_mount = pose_to_matrix(
            [
                *gripper.get("mount_xyz_m", [0.0, 0.0, 0.0]),
                *gripper.get("mount_rpy_rad", [0.0, 0.0, 0.0]),
            ]
        )
        gripper_mount_to_base = pose_to_matrix(
            [
                *gripper.get("xyz_m", [0.0, 0.0, 0.0]),
                *gripper.get("rpy_rad", [0.0, 0.0, 0.0]),
            ]
        )
        gripper_base_to_tcp = pose_to_matrix(
            [
                *gripper.get("tcp_xyz_m", [0.0, 0.0, 0.0]),
                *gripper.get("tcp_rpy_rad", [0.0, 0.0, 0.0]),
            ]
        )
        tcp_offset_pose = tuple(
            float(v)
            for v in matrix_to_pose(
                flange_to_gripper_mount @ gripper_mount_to_base @ gripper_base_to_tcp
            )
        )

        return cls(
            extrinsics_path=extrinsics_path,
            mode=selected_mode,
            flange_to_camera_pose=flange_to_camera_pose,
            camera_to_optical_pose=camera_to_optical_pose,
            tcp_offset_pose=tcp_offset_pose,
        )

    @property
    def flange_to_optical_matrix(self) -> np.ndarray:
        return pose_to_matrix(self.flange_to_camera_pose) @ pose_to_matrix(self.camera_to_optical_pose)

    def base_to_optical_matrix(self, flange_pose_m_rad: Iterable[float]) -> np.ndarray:
        return pose_to_matrix(flange_pose_m_rad) @ self.flange_to_optical_matrix

    def point_optical_to_base(self, flange_pose_m_rad: Iterable[float], point_optical_m: Iterable[float]) -> np.ndarray:
        return transform_point(self.base_to_optical_matrix(flange_pose_m_rad), point_optical_m)

    def direction_optical_to_base(self, flange_pose_m_rad: Iterable[float], direction_optical: Iterable[float]) -> np.ndarray:
        return transform_direction(self.base_to_optical_matrix(flange_pose_m_rad), direction_optical)

    def scan_delta_xy_base(self, flange_pose_m_rad: Iterable[float], point_optical_m: Iterable[float]) -> np.ndarray:
        point = np.array([float(v) for v in point_optical_m], dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("point_optical_m must contain exactly 3 values")
        rotation = self.base_to_optical_matrix(flange_pose_m_rad)[:3, :3]
        delta = rotation @ np.array([point[0], point[1], 0.0], dtype=np.float64)
        return delta[:2]

    def axis_yaw_in_base(
        self,
        flange_pose_m_rad: Iterable[float],
        axis_direction_optical: Iterable[float],
        yaw_offset_deg: float = 0.0,
    ) -> float:
        axis_base = self.direction_optical_to_base(flange_pose_m_rad, axis_direction_optical)
        axis_xy = axis_base[:2]
        norm = float(np.linalg.norm(axis_xy))
        if norm < 1e-9:
            raise ValueError("axis projection on base XY plane is too small")
        yaw = math.atan2(float(axis_xy[1]), float(axis_xy[0])) + math.radians(float(yaw_offset_deg))
        return normalize_angle_rad(yaw)
