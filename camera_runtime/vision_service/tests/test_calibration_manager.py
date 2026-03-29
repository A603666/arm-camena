from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from vision_service.app import calibration as calibration_module
from vision_service.app.calibration import CalibrationManager


def test_build_object_points_11x8_15mm(tmp_path: Path) -> None:
    manager = CalibrationManager(
        min_samples=1,
        handeye_yaml_path=tmp_path / "handeye_extrinsics.yaml",
        output_dir=tmp_path / "runs",
    )

    points = manager._build_object_points_m()
    assert points.shape == (88, 3)
    assert np.isclose(points[:, 0].min(), 0.0)
    assert np.isclose(points[:, 1].min(), 0.0)
    assert np.isclose(points[:, 0].max(), 10 * 0.015)
    assert np.isclose(points[:, 1].max(), 7 * 0.015)
    assert len(np.unique(points[:, 0])) == 11
    assert len(np.unique(points[:, 1])) == 8


def test_compute_calibration_with_patched_opencv(tmp_path: Path, monkeypatch) -> None:
    manager = CalibrationManager(
        min_samples=1,
        handeye_yaml_path=tmp_path / "handeye_extrinsics.yaml",
        output_dir=tmp_path / "runs",
    )

    corners = np.stack(
        [
            np.linspace(20.0, 220.0, 88),
            np.linspace(30.0, 180.0, 88),
        ],
        axis=1,
    ).astype(np.float32)

    def fake_detect(_rgb: np.ndarray):
        return True, corners, "mock"

    monkeypatch.setattr(manager, "_detect_chessboard", fake_detect)

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    manager.ingest_frame(rgb, {"frame_id": 7})
    manager.capture_sample([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def fake_calibrate_camera(object_points, image_points, image_size, _k, _d):
        _ = object_points, image_points, image_size
        k = np.array(
            [
                [320.0, 0.0, 160.0],
                [0.0, 320.0, 120.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        d = np.zeros((1, 5), dtype=np.float64)
        rvecs = [np.zeros((3, 1), dtype=np.float64)]
        tvecs = [np.array([[0.0], [0.0], [1.0]], dtype=np.float64)]
        return 0.21, k, d, rvecs, tvecs

    def fake_project_points(obj, _rvec, _tvec, _k, _d):
        xy = obj[:, :2] * 100.0 + np.array([20.0, 30.0], dtype=np.float32)
        return xy.reshape(-1, 1, 2).astype(np.float32), None

    def fake_solve_pnp(_obj, _img, _k, _d, flags=0):
        _ = flags
        return True, np.zeros((3, 1), dtype=np.float64), np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    def fake_handeye(_rgb_r, _rgb_t, _rtc_r, _rtc_t, method=0):
        _ = method
        return np.eye(3, dtype=np.float64), np.array([[0.01], [0.0], [0.2]], dtype=np.float64)

    monkeypatch.setattr(calibration_module.cv2, "calibrateCamera", fake_calibrate_camera)
    monkeypatch.setattr(calibration_module.cv2, "projectPoints", fake_project_points)
    monkeypatch.setattr(calibration_module.cv2, "solvePnP", fake_solve_pnp)
    monkeypatch.setattr(calibration_module.cv2, "calibrateHandEye", fake_handeye)

    result = manager.compute_calibration()
    assert result["sample_count"] == 1
    assert result["camera_intrinsics"]["rms"] == 0.21
    assert result["handeye"]["method"] == "tsai"

    status = manager.get_status()
    assert status["latest_result_path"] is not None
    assert Path(status["latest_result_path"]).exists()


def test_apply_and_rollback_handeye_yaml(tmp_path: Path) -> None:
    handeye_path = tmp_path / "handeye_extrinsics.yaml"
    original = {
        "nominal_camera": {
            "parent_frame": "flange_frame",
            "child_frame": "handeye_camera_link",
            "optical_xyz_m": [0.0, 0.0, 0.0],
            "optical_rpy_rad": [0.0, 0.0, -1.5707963267948966],
        },
        "calibrated_camera": {
            "enabled": False,
            "parent_frame": "flange_frame",
            "child_frame": "handeye_camera_calibrated",
            "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
    }
    handeye_path.write_text(yaml.safe_dump(original, sort_keys=False, allow_unicode=True), encoding="utf-8")

    out_dir = tmp_path / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "result.json"
    result_file.write_text("{}", encoding="utf-8")

    manager = CalibrationManager(
        min_samples=1,
        handeye_yaml_path=handeye_path,
        output_dir=out_dir,
    )
    manager._latest_result = {
        "sample_count": 3,
        "camera_intrinsics": {"rms": 0.33},
        "handeye": {"flange_to_camera_optical_pose_m_rad": [0.1, -0.02, 0.25, 0.0, 0.0, 0.0]},
    }
    manager._latest_result_path = result_file

    before_text = handeye_path.read_text(encoding="utf-8")
    apply_result = manager.apply_calibration()
    assert apply_result["ok"] is True
    assert Path(apply_result["backup_path"]).exists()

    applied_raw = yaml.safe_load(handeye_path.read_text(encoding="utf-8"))
    calibrated = applied_raw["calibrated_camera"]
    assert calibrated["enabled"] is True

    # Verify conversion: T_flange_clink * T_clink_optical == T_flange_optical
    t_flange_clink = manager._pose_to_matrix([*calibrated["xyz_m"], *calibrated["rpy_rad"]])
    t_clink_opt = manager._pose_to_matrix(manager._load_nominal_optical_pose(applied_raw))
    t_flange_opt = t_flange_clink @ t_clink_opt
    t_expected = manager._pose_to_matrix([0.1, -0.02, 0.25, 0.0, 0.0, 0.0])
    assert np.allclose(t_flange_opt, t_expected, atol=1e-6)

    rollback_result = manager.rollback_last_apply()
    assert rollback_result["ok"] is True
    assert handeye_path.read_text(encoding="utf-8") == before_text
