from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynamic_grasp.math_utils import HandEyeModel


def _write_handeye_yaml(path: Path, *, calibrated_enabled: bool) -> None:
    payload = {
        "nominal_camera": {
            "xyz_m": [0.01, 0.02, 0.03],
            "rpy_rad": [0.0, 0.0, 0.0],
            "optical_xyz_m": [0.0, 0.0, 0.0],
            "optical_rpy_rad": [0.0, 0.0, -1.5707963267948966],
        },
        "calibrated_camera": {
            "enabled": bool(calibrated_enabled),
            "xyz_m": [0.11, 0.12, 0.13],
            "rpy_rad": [0.1, 0.2, 0.3],
        },
        "gripper_nominal": {
            "mount_xyz_m": [0.0, 0.0, 0.0],
            "mount_rpy_rad": [0.0, 0.0, 0.0],
            "xyz_m": [0.0, 0.0, 0.02],
            "rpy_rad": [0.0, 0.0, 0.0],
            "tcp_xyz_m": [0.0, 0.0, 0.1],
            "tcp_rpy_rad": [0.0, 0.0, 0.0],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_handeye_calibrated_mode_loads_when_enabled(tmp_path: Path) -> None:
    cfg = tmp_path / "handeye.yaml"
    _write_handeye_yaml(cfg, calibrated_enabled=True)

    model = HandEyeModel.from_yaml(cfg, mode="calibrated")
    assert model.mode == "calibrated"
    assert model.flange_to_camera_pose[:3] == pytest.approx((0.11, 0.12, 0.13))


def test_handeye_calibrated_mode_fails_fast_when_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "handeye.yaml"
    _write_handeye_yaml(cfg, calibrated_enabled=False)

    with pytest.raises(ValueError, match="calibrated hand-eye is required"):
        HandEyeModel.from_yaml(cfg, mode="calibrated")
