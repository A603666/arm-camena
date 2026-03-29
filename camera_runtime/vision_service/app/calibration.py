from __future__ import annotations

import json
import math
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class CalibrationBoard:
    cols: int = 11
    rows: int = 8
    square_size_m: float = 0.015


class CalibrationManager:
    def __init__(
        self,
        *,
        board: CalibrationBoard | None = None,
        min_samples: int = 15,
        preview_jpeg_quality: int = 85,
        handeye_yaml_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._board = board or CalibrationBoard()
        self._min_samples = max(1, int(min_samples))
        self._preview_jpeg_quality = max(50, min(95, int(preview_jpeg_quality)))

        vision_root = Path(__file__).resolve().parents[1]
        repo_root = Path(__file__).resolve().parents[3]
        self._handeye_yaml_path = (
            Path(handeye_yaml_path).expanduser().resolve()
            if handeye_yaml_path is not None
            else repo_root / "模型文件" / "nero_description" / "config" / "handeye_extrinsics.yaml"
        )
        self._output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else vision_root / "calibration_runs"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._samples: list[dict[str, Any]] = []
        self._latest_corners: np.ndarray | None = None
        self._latest_image_size: tuple[int, int] | None = None
        self._latest_meta: dict[str, Any] = {}
        self._latest_detection_found = False
        self._latest_detection_method = "none"
        self._latest_detection_time: float | None = None
        self._latest_preview_jpeg: bytes | None = None

        self._latest_result: dict[str, Any] | None = None
        self._latest_result_path: Path | None = None
        self._latest_backup_path: Path | None = None
        self._last_error: str | None = None
        self._mode_active = False

    @property
    def handeye_yaml_path(self) -> Path:
        return self._handeye_yaml_path

    def set_mode_active(self, active: bool) -> dict[str, Any]:
        with self._lock:
            self._mode_active = bool(active)
            return {"ok": True, "mode_active": self._mode_active}

    def is_mode_active(self) -> bool:
        with self._lock:
            return bool(self._mode_active)

    @staticmethod
    def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(float(roll))
        sr = math.sin(float(roll))
        cp = math.cos(float(pitch))
        sp = math.sin(float(pitch))
        cy = math.cos(float(yaw))
        sy = math.sin(float(yaw))
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _pose_to_matrix(cls, pose_m_rad: list[float] | tuple[float, ...]) -> np.ndarray:
        pose = [float(v) for v in pose_m_rad]
        if len(pose) != 6:
            raise ValueError("pose must contain 6 values")
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = cls._rpy_to_matrix(pose[3], pose[4], pose[5])
        out[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
        return out

    @staticmethod
    def _normalize_angle_rad(v: float) -> float:
        return math.atan2(math.sin(float(v)), math.cos(float(v)))

    @classmethod
    def _matrix_to_pose(cls, matrix_4x4: np.ndarray) -> list[float]:
        matrix = np.asarray(matrix_4x4, dtype=np.float64)
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
            cls._normalize_angle_rad(roll),
            cls._normalize_angle_rad(pitch),
            cls._normalize_angle_rad(yaw),
        ]

    @staticmethod
    def _invert_transform(matrix_4x4: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix_4x4, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
        rot = matrix[:3, :3]
        trans = matrix[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rot.T
        out[:3, 3] = -rot.T @ trans
        return out

    @staticmethod
    def _validate_pose(pose_m_rad: list[float] | tuple[float, ...]) -> list[float]:
        if not isinstance(pose_m_rad, (list, tuple)) or len(pose_m_rad) != 6:
            raise ValueError("robot flange pose must be a list[6] in meter/radian")
        out = [float(v) for v in pose_m_rad]
        if not all(math.isfinite(v) for v in out):
            raise ValueError("robot flange pose contains non-finite values")
        return out

    def _build_object_points_m(self) -> np.ndarray:
        obj = np.zeros((self._board.rows * self._board.cols, 3), dtype=np.float32)
        grid = np.mgrid[0 : self._board.cols, 0 : self._board.rows].T.reshape(-1, 2)
        obj[:, :2] = grid.astype(np.float32) * float(self._board.square_size_m)
        return obj

    def _detect_chessboard(self, rgb: np.ndarray) -> tuple[bool, np.ndarray | None, str]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        board_size = (self._board.cols, self._board.rows)

        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, board_size, flags=sb_flags)
        if found_sb and corners_sb is not None:
            corners = corners_sb.reshape(-1, 2).astype(np.float32)
            return True, corners, "findChessboardCornersSB"

        std_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found_std, corners_std = cv2.findChessboardCorners(gray, board_size, flags=std_flags)
        if not found_std or corners_std is None:
            return False, None, "none"

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-3,
        )
        refined = cv2.cornerSubPix(
            gray,
            corners_std,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        corners = refined.reshape(-1, 2).astype(np.float32)
        return True, corners, "findChessboardCorners"

    def ingest_frame(self, rgb: np.ndarray, meta: dict[str, Any]) -> None:
        if rgb is None or rgb.size == 0:
            return

        h, w = rgb.shape[:2]
        found, corners, method = self._detect_chessboard(rgb)

        preview = rgb.copy()
        if found and corners is not None:
            cv2.drawChessboardCorners(
                preview,
                (self._board.cols, self._board.rows),
                corners.reshape(-1, 1, 2),
                True,
            )

        with self._lock:
            sample_count = len(self._samples)

        text_color = (0, 220, 0) if found else (0, 120, 255)
        cv2.putText(
            preview,
            f"Calibration board: {'DETECTED' if found else 'NOT DETECTED'}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"Samples: {sample_count}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"Method: {method}",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, self._preview_jpeg_quality],
        )
        preview_jpeg = encoded.tobytes() if ok else None

        with self._lock:
            self._latest_corners = corners if found else None
            self._latest_image_size = (int(w), int(h))
            self._latest_meta = dict(meta)
            self._latest_detection_found = bool(found)
            self._latest_detection_method = method
            self._latest_detection_time = time.time()
            if preview_jpeg is not None:
                self._latest_preview_jpeg = preview_jpeg

    def get_latest_preview_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_preview_jpeg

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            result = self._latest_result
            summary = None
            if result is not None:
                summary = {
                    "sample_count": int(result.get("sample_count", 0)),
                    "rms": float(result.get("camera_intrinsics", {}).get("rms", 0.0)),
                    "mean_reproj_error_px": float(
                        result.get("camera_intrinsics", {})
                        .get("reprojection_error_px", {})
                        .get("mean", 0.0)
                    ),
                    "target_translation_std_mm": float(
                        result.get("handeye", {})
                        .get("target_consistency", {})
                        .get("translation_std_mm", 0.0)
                    ),
                    "target_rotation_std_deg": float(
                        result.get("handeye", {})
                        .get("target_consistency", {})
                        .get("rotation_std_deg", 0.0)
                    ),
                }

            return {
                "board": {
                    "cols": self._board.cols,
                    "rows": self._board.rows,
                    "square_size_m": self._board.square_size_m,
                },
                "min_samples": self._min_samples,
                "sample_count": len(self._samples),
                "latest_detection": {
                    "found": self._latest_detection_found,
                    "method": self._latest_detection_method,
                    "frame_id": int(self._latest_meta.get("frame_id", 0) or 0),
                    "updated_at": self._latest_detection_time,
                },
                "can_capture": bool(self._latest_detection_found),
                "can_compute": len(self._samples) >= self._min_samples,
                "can_apply": result is not None,
                "mode_active": bool(self._mode_active),
                "latest_result_summary": summary,
                "latest_result_path": str(self._latest_result_path) if self._latest_result_path is not None else None,
                "latest_backup_path": str(self._latest_backup_path) if self._latest_backup_path is not None else None,
                "handeye_yaml_path": str(self._handeye_yaml_path),
                "last_error": self._last_error,
            }

    def capture_sample(self, robot_flange_pose_m_rad: list[float] | tuple[float, ...]) -> dict[str, Any]:
        pose = self._validate_pose(robot_flange_pose_m_rad)
        with self._lock:
            if not self._latest_detection_found or self._latest_corners is None:
                raise RuntimeError("checkerboard not detected in latest frame")
            if self._latest_image_size is None:
                raise RuntimeError("no image frame available")

            sample = {
                "timestamp": time.time(),
                "frame_id": int(self._latest_meta.get("frame_id", 0) or 0),
                "robot_flange_pose_m_rad": pose,
                "corners_2d": self._latest_corners.astype(np.float32).tolist(),
                "image_size": [int(self._latest_image_size[0]), int(self._latest_image_size[1])],
                "meta_snapshot": dict(self._latest_meta),
            }
            self._samples.append(sample)
            self._latest_result = None
            self._latest_result_path = None
            self._last_error = None
            count = len(self._samples)

        return {
            "ok": True,
            "sample_count": count,
            "captured_frame_id": int(sample["frame_id"]),
        }

    def reset_session(self) -> dict[str, Any]:
        with self._lock:
            self._samples = []
            self._latest_result = None
            self._latest_result_path = None
            self._last_error = None
        return {"ok": True, "sample_count": 0}

    @staticmethod
    def _rotation_angle_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
        rel = np.asarray(lhs, dtype=np.float64).T @ np.asarray(rhs, dtype=np.float64)
        cos_theta = max(-1.0, min(1.0, float((np.trace(rel) - 1.0) * 0.5)))
        return math.degrees(math.acos(cos_theta))

    def _compute_target_consistency(
        self,
        t_gripper2base: list[np.ndarray],
        r_gripper2base: list[np.ndarray],
        t_cam2gripper: np.ndarray,
        r_cam2gripper: np.ndarray,
        t_target2cam: list[np.ndarray],
        r_target2cam: list[np.ndarray],
    ) -> dict[str, float]:
        target_translations: list[np.ndarray] = []
        target_rotations: list[np.ndarray] = []

        t_cg = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3, 1)
        r_cg = np.asarray(r_cam2gripper, dtype=np.float64)

        for i in range(len(t_gripper2base)):
            t_bg = np.asarray(t_gripper2base[i], dtype=np.float64).reshape(3, 1)
            r_bg = np.asarray(r_gripper2base[i], dtype=np.float64)
            t_ct = np.asarray(t_target2cam[i], dtype=np.float64).reshape(3, 1)
            r_ct = np.asarray(r_target2cam[i], dtype=np.float64)

            t_bt = (r_bg @ (r_cg @ t_ct + t_cg)) + t_bg
            r_bt = r_bg @ r_cg @ r_ct
            target_translations.append(t_bt.reshape(3))
            target_rotations.append(r_bt)

        xyz = np.asarray(target_translations, dtype=np.float64)
        std_xyz_mm = float(np.mean(np.std(xyz, axis=0)) * 1000.0)

        ref_r = target_rotations[0]
        rot_err_deg = [self._rotation_angle_deg(ref_r, r) for r in target_rotations]
        std_rot_deg = float(np.std(np.asarray(rot_err_deg, dtype=np.float64)))

        return {
            "translation_std_mm": round(std_xyz_mm, 4),
            "rotation_std_deg": round(std_rot_deg, 4),
            "rotation_max_deg": round(float(max(rot_err_deg)), 4),
        }

    def compute_calibration(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            raise ValueError("no samples captured")
        if len(samples) < self._min_samples:
            raise ValueError(f"insufficient samples: need at least {self._min_samples}")

        obj_template = self._build_object_points_m()
        image_size = tuple(int(v) for v in samples[0]["image_size"])

        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for sample in samples:
            corners = np.asarray(sample["corners_2d"], dtype=np.float32).reshape(-1, 2)
            if corners.shape[0] != obj_template.shape[0]:
                raise ValueError("sample corner count does not match board definition")
            object_points.append(obj_template.copy())
            image_points.append(corners.reshape(-1, 1, 2))

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )

        reproj_errors: list[float] = []
        for i, obj in enumerate(object_points):
            projected, _ = cv2.projectPoints(
                obj,
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs,
            )
            img = image_points[i].reshape(-1, 2)
            proj = projected.reshape(-1, 2)
            err = np.sqrt(np.mean(np.sum((img - proj) ** 2, axis=1)))
            reproj_errors.append(float(err))

        r_gripper2base: list[np.ndarray] = []
        t_gripper2base: list[np.ndarray] = []
        r_target2cam: list[np.ndarray] = []
        t_target2cam: list[np.ndarray] = []

        for i, sample in enumerate(samples):
            corners = image_points[i]
            ok, rvec, tvec = cv2.solvePnP(
                obj_template,
                corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise RuntimeError(f"solvePnP failed at sample index {i}")

            r_tc, _ = cv2.Rodrigues(rvec)
            r_target2cam.append(r_tc.astype(np.float64))
            t_target2cam.append(np.asarray(tvec, dtype=np.float64).reshape(3, 1))

            # Assumption per spec: flange_pose is gripper->base.
            t_bg = self._pose_to_matrix(sample["robot_flange_pose_m_rad"])
            r_gripper2base.append(t_bg[:3, :3].astype(np.float64))
            t_gripper2base.append(t_bg[:3, 3].astype(np.float64).reshape(3, 1))

        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        t_gcopt = np.eye(4, dtype=np.float64)
        t_gcopt[:3, :3] = np.asarray(r_cam2gripper, dtype=np.float64)
        t_gcopt[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
        flange_to_camera_optical_pose = self._matrix_to_pose(t_gcopt)

        consistency = self._compute_target_consistency(
            t_gripper2base=t_gripper2base,
            r_gripper2base=r_gripper2base,
            t_cam2gripper=t_cam2gripper,
            r_cam2gripper=r_cam2gripper,
            t_target2cam=t_target2cam,
            r_target2cam=r_target2cam,
        )

        result = {
            "generated_at": time.time(),
            "sample_count": len(samples),
            "board": {
                "cols": self._board.cols,
                "rows": self._board.rows,
                "square_size_m": self._board.square_size_m,
            },
            "camera_intrinsics": {
                "rms": float(rms),
                "camera_matrix": np.asarray(camera_matrix, dtype=np.float64).tolist(),
                "dist_coeffs": np.asarray(dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
                "reprojection_error_px": {
                    "mean": float(np.mean(reproj_errors)),
                    "max": float(np.max(reproj_errors)),
                    "min": float(np.min(reproj_errors)),
                    "per_sample": reproj_errors,
                },
            },
            "handeye": {
                "method": "tsai",
                "flange_to_camera_optical_pose_m_rad": flange_to_camera_optical_pose,
                "camera_optical_to_flange_rmat": np.asarray(r_cam2gripper, dtype=np.float64).tolist(),
                "camera_optical_to_flange_tvec_m": np.asarray(t_cam2gripper, dtype=np.float64).reshape(3).tolist(),
                "target_consistency": consistency,
            },
            "samples": [
                {
                    "timestamp": float(s["timestamp"]),
                    "frame_id": int(s["frame_id"]),
                    "robot_flange_pose_m_rad": [float(v) for v in s["robot_flange_pose_m_rad"]],
                }
                for s in samples
            ],
        }

        now = time.localtime()
        stamp = time.strftime("%Y%m%d_%H%M%S", now)
        ms = int((time.time() % 1.0) * 1000.0)
        run_path = self._output_dir / f"calibration_result_{stamp}_{ms:03d}.json"
        run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        with self._lock:
            self._latest_result = result
            self._latest_result_path = run_path
            self._last_error = None

        return result

    def _load_nominal_optical_pose(self, raw_yaml: dict[str, Any]) -> list[float]:
        nominal = raw_yaml.get("nominal_camera")
        if not isinstance(nominal, dict):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        opt_xyz = nominal.get("optical_xyz_m", [0.0, 0.0, 0.0])
        opt_rpy = nominal.get("optical_rpy_rad", [0.0, 0.0, 0.0])
        return [
            float(opt_xyz[0]),
            float(opt_xyz[1]),
            float(opt_xyz[2]),
            float(opt_rpy[0]),
            float(opt_rpy[1]),
            float(opt_rpy[2]),
        ]

    def _flange_to_camera_link_pose(
        self,
        flange_to_camera_optical_pose: list[float],
        camera_link_to_optical_pose: list[float],
    ) -> list[float]:
        t_f_copt = self._pose_to_matrix(flange_to_camera_optical_pose)
        t_clink_copt = self._pose_to_matrix(camera_link_to_optical_pose)
        t_f_clink = t_f_copt @ self._invert_transform(t_clink_copt)
        return self._matrix_to_pose(t_f_clink)

    def apply_calibration(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest_result
            latest_result_path = self._latest_result_path

        if latest is None:
            raise ValueError("no calibration result to apply")
        if not self._handeye_yaml_path.exists():
            raise FileNotFoundError(f"handeye yaml not found: {self._handeye_yaml_path}")

        raw = yaml.safe_load(self._handeye_yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise RuntimeError("invalid handeye yaml format")

        optical_pose = self._load_nominal_optical_pose(raw)
        flange_to_camera_optical_pose = [
            float(v)
            for v in latest.get("handeye", {}).get("flange_to_camera_optical_pose_m_rad", [0.0] * 6)
        ]
        if len(flange_to_camera_optical_pose) != 6:
            raise RuntimeError("invalid calibration output: flange_to_camera_optical_pose_m_rad")

        flange_to_camera_link_pose = self._flange_to_camera_link_pose(
            flange_to_camera_optical_pose=flange_to_camera_optical_pose,
            camera_link_to_optical_pose=optical_pose,
        )

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        ms = int((time.time() % 1.0) * 1000.0)
        backup_path = self._handeye_yaml_path.with_name(
            f"{self._handeye_yaml_path.stem}.backup.{ts}_{ms:03d}{self._handeye_yaml_path.suffix}"
        )
        shutil.copy2(self._handeye_yaml_path, backup_path)

        nominal = raw.get("nominal_camera") if isinstance(raw.get("nominal_camera"), dict) else {}
        calibrated = raw.get("calibrated_camera") if isinstance(raw.get("calibrated_camera"), dict) else {}
        calibrated["enabled"] = True
        calibrated["parent_frame"] = str(calibrated.get("parent_frame") or nominal.get("parent_frame") or "flange_frame")
        calibrated["child_frame"] = str(calibrated.get("child_frame") or "handeye_camera_calibrated")
        calibrated["xyz_m"] = [float(v) for v in flange_to_camera_link_pose[:3]]
        calibrated["rpy_rad"] = [float(v) for v in flange_to_camera_link_pose[3:6]]
        calibrated["source"] = {
            "updated_at": float(time.time()),
            "sample_count": int(latest.get("sample_count", 0)),
            "rms": float(latest.get("camera_intrinsics", {}).get("rms", 0.0)),
            "result_file": str(latest_result_path) if latest_result_path is not None else None,
        }
        raw["calibrated_camera"] = calibrated

        self._handeye_yaml_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        with self._lock:
            self._latest_backup_path = backup_path
            self._last_error = None

        return {
            "ok": True,
            "handeye_yaml_path": str(self._handeye_yaml_path),
            "backup_path": str(backup_path),
            "calibrated_camera": {
                "enabled": True,
                "xyz_m": calibrated["xyz_m"],
                "rpy_rad": calibrated["rpy_rad"],
            },
        }

    def rollback_last_apply(self) -> dict[str, Any]:
        with self._lock:
            backup_path = self._latest_backup_path

        if backup_path is None:
            raise ValueError("no backup available for rollback")
        if not backup_path.exists():
            raise FileNotFoundError(f"backup file missing: {backup_path}")

        shutil.copy2(backup_path, self._handeye_yaml_path)
        return {
            "ok": True,
            "handeye_yaml_path": str(self._handeye_yaml_path),
            "restored_from": str(backup_path),
        }
