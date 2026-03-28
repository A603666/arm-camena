from __future__ import annotations

from vision_service.app.config import load_config


def test_load_config_uses_yolo26n_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DABAI_YOLO_MODEL", raising=False)
    cfg = load_config()
    assert cfg.model_path.name == "yolo26n.pt"
    assert cfg.robot_arm_config_path.name == "default.yaml"
    assert cfg.robot_control_enabled is True
    assert cfg.robot_loopback_only is True
    assert cfg.yolo_precision == "fp32"
    assert cfg.yolo_warmup is True
    assert cfg.geom_backend == "auto"
    assert cfg.geom_parity_check is False
    assert cfg.geom_parity_every_n == 30


def test_load_config_reads_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOM_BACKEND", "torch")
    monkeypatch.setenv("DABAI_GEOM_PARITY_CHECK", "1")
    monkeypatch.setenv("DABAI_GEOM_PARITY_EVERY_N", "45")
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "2")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "30")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "0.8")
    monkeypatch.setenv("DABAI_OBJECT_HEIGHT_MIN_MM", "6.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "7")
    monkeypatch.setenv("DABAI_AXIS_EIG_RATIO_MIN", "1.6")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "9")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "0.4")

    cfg = load_config()
    assert cfg.geom_backend == "torch"
    assert cfg.geom_parity_check is True
    assert cfg.geom_parity_every_n == 45
    assert cfg.geometry_every_n == 2
    assert cfg.ransac_max_trials == 30
    assert abs(cfg.geometry_force_recalc_iou - 0.8) < 1e-6
    assert abs(cfg.object_height_min_mm - 6.5) < 1e-6
    assert cfg.support_close_px == 7
    assert abs(cfg.axis_eig_ratio_min - 1.6) < 1e-6
    assert cfg.axis_hold_frames == 9
    assert abs(cfg.axis_smooth_alpha - 0.4) < 1e-6


def test_load_config_clamps_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOM_BACKEND", "invalid")
    monkeypatch.setenv("DABAI_GEOM_PARITY_EVERY_N", "0")
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "99")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "-1")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "2.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "0")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "99")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "5")

    cfg = load_config()
    assert cfg.geom_backend == "auto"
    assert cfg.geom_parity_every_n == 1
    assert cfg.geometry_every_n == 6
    assert cfg.ransac_max_trials == 8
    assert abs(cfg.geometry_force_recalc_iou - 0.95) < 1e-6
    assert cfg.support_close_px == 1
    assert cfg.axis_hold_frames == 60
    assert abs(cfg.axis_smooth_alpha - 1.0) < 1e-6


def test_load_config_reads_robot_control_env(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "robot_web.yaml"
    cfg_path.write_text("backend: real\n", encoding="utf-8")

    monkeypatch.setenv("DABAI_ROBOT_CONTROL_ENABLED", "0")
    monkeypatch.setenv("DABAI_ROBOT_LOOPBACK_ONLY", "false")
    monkeypatch.setenv("DABAI_ROBOT_BACKEND_OVERRIDE", "real")
    monkeypatch.setenv("DABAI_ROBOT_CONFIG", str(cfg_path))

    cfg = load_config()
    assert cfg.robot_control_enabled is False
    assert cfg.robot_loopback_only is False
    assert cfg.robot_backend_override == "real"
    assert cfg.robot_arm_config_path == cfg_path.resolve()


def test_load_config_reads_yolo_precision_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_YOLO_PRECISION", "fp16")
    monkeypatch.setenv("DABAI_YOLO_WARMUP", "0")
    cfg = load_config()
    assert cfg.yolo_precision == "fp16"
    assert cfg.yolo_warmup is False
