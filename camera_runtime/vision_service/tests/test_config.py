from __future__ import annotations

from vision_service.app.config import load_config


def test_load_config_uses_yolo26n_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DABAI_YOLO_MODEL", raising=False)
    cfg = load_config()
    assert cfg.model_path.name == "yolo26n.pt"


def test_load_config_reads_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "2")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "30")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "0.8")
    monkeypatch.setenv("DABAI_OBJECT_HEIGHT_MIN_MM", "6.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "7")
    monkeypatch.setenv("DABAI_AXIS_EIG_RATIO_MIN", "1.6")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "9")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "0.4")

    cfg = load_config()
    assert cfg.geometry_every_n == 2
    assert cfg.ransac_max_trials == 30
    assert abs(cfg.geometry_force_recalc_iou - 0.8) < 1e-6
    assert abs(cfg.object_height_min_mm - 6.5) < 1e-6
    assert cfg.support_close_px == 7
    assert abs(cfg.axis_eig_ratio_min - 1.6) < 1e-6
    assert cfg.axis_hold_frames == 9
    assert abs(cfg.axis_smooth_alpha - 0.4) < 1e-6


def test_load_config_clamps_geometry_env(monkeypatch) -> None:
    monkeypatch.setenv("DABAI_GEOMETRY_EVERY_N", "99")
    monkeypatch.setenv("DABAI_RANSAC_MAX_TRIALS", "-1")
    monkeypatch.setenv("DABAI_GEOMETRY_FORCE_RECALC_IOU", "2.5")
    monkeypatch.setenv("DABAI_SUPPORT_CLOSE_PX", "0")
    monkeypatch.setenv("DABAI_AXIS_HOLD_FRAMES", "99")
    monkeypatch.setenv("DABAI_AXIS_SMOOTH_ALPHA", "5")

    cfg = load_config()
    assert cfg.geometry_every_n == 6
    assert cfg.ransac_max_trials == 8
    assert abs(cfg.geometry_force_recalc_iou - 0.95) < 1e-6
    assert cfg.support_close_px == 1
    assert cfg.axis_hold_frames == 60
    assert abs(cfg.axis_smooth_alpha - 1.0) < 1e-6
