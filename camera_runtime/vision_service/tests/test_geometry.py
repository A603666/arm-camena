from __future__ import annotations

import numpy as np

from vision_service.app.geometry import (
    estimate_object_geometry,
    extract_support_region,
    find_grasp_point,
    fit_ground_mask,
    fit_ground_plane,
    fit_ground_plane_conservative_fast,
)


def make_strip_points(length_mm: float, width_mm: float, z_mm: float, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = rng.uniform(-length_mm / 2, length_mm / 2, size=n)
    y = rng.uniform(-width_mm / 2, width_mm / 2, size=n)
    z = rng.normal(z_mm, 3.0, size=n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def test_geometry_estimation_returns_positive_dimensions() -> None:
    points = make_strip_points(length_mm=260, width_mm=70, z_mm=820, n=2000)
    geom = estimate_object_geometry(points)
    assert geom is not None
    assert geom["length_mm"] > 150
    assert 40 < geom["width_mm"] < 100
    assert geom["height_mm"] > 0


def test_grasp_point_found_when_local_region_is_narrow() -> None:
    rng = np.random.default_rng(123)
    x = rng.uniform(-180, 180, size=2800)
    # Wide edges, narrow middle section.
    y = np.where(
        np.abs(x) < 35,
        rng.uniform(-35, 35, size=x.size),
        rng.uniform(-80, 80, size=x.size),
    )
    z = rng.normal(900.0, 4.0, size=x.size)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    geom = estimate_object_geometry(points)
    assert geom is not None
    grasp = find_grasp_point(
        points=points,
        geometry=geom,
        width_limit_mm=100.0,
        window_length_mm=80.0,
        step_mm=5.0,
        min_points=100,
    )
    assert grasp["status"] == "ok"
    assert grasp["grasp_xyz_mm"] is not None


def test_grasp_point_rejected_when_object_too_wide() -> None:
    points = make_strip_points(length_mm=260, width_mm=150, z_mm=840, n=2600)
    geom = estimate_object_geometry(points)
    assert geom is not None
    grasp = find_grasp_point(
        points=points,
        geometry=geom,
        width_limit_mm=100.0,
        window_length_mm=80.0,
        step_mm=5.0,
        min_points=120,
    )
    assert grasp["status"] == "too_wide"
    assert grasp["grasp_xyz_mm"] is None


def test_ground_fit_marks_plane_as_inlier_majority() -> None:
    rng = np.random.default_rng(8)
    plane_x = rng.uniform(-200, 200, size=2500)
    plane_y = rng.uniform(-200, 200, size=2500)
    plane_z = 1000 + 0.02 * plane_x + 0.01 * plane_y + rng.normal(0, 2.0, size=2500)
    object_x = rng.uniform(-40, 40, size=500)
    object_y = rng.uniform(-30, 30, size=500)
    object_z = rng.normal(900, 3.0, size=500)

    points = np.concatenate(
        [
            np.stack([plane_x, plane_y, plane_z], axis=1),
            np.stack([object_x, object_y, object_z], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    mask = fit_ground_mask(points, residual_mm=8.0)
    assert mask.dtype == np.bool_
    assert mask.shape[0] == points.shape[0]
    assert np.mean(mask[:2500]) > 0.70


def test_ground_fit_accepts_custom_max_trials() -> None:
    rng = np.random.default_rng(2026)
    x = rng.uniform(-180, 180, size=1800)
    y = rng.uniform(-180, 180, size=1800)
    z = 950 + 0.03 * x + 0.01 * y + rng.normal(0, 1.5, size=1800)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    mask = fit_ground_mask(points, residual_mm=6.0, max_trials=12)
    assert mask.dtype == np.bool_
    assert mask.shape[0] == points.shape[0]


def test_ground_fit_fast_mode_returns_plane_on_clean_scene() -> None:
    rng = np.random.default_rng(1234)
    x = rng.uniform(-250, 250, size=2800)
    y = rng.uniform(-220, 220, size=2800)
    z = 980 + 0.02 * x + 0.015 * y + rng.normal(0, 1.8, size=2800)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    plane, mode = fit_ground_plane_conservative_fast(
        points=points,
        residual_mm=8.0,
        max_trials=120,
        fast_enabled=True,
        fast_sample_cap=1200,
        fast_max_trials=30,
        fast_min_inlier_ratio=0.4,
        fast_min_inliers=120,
    )
    assert plane is not None
    assert mode in {"fast", "fallback_full"}
    assert int(np.count_nonzero(plane.inlier_mask)) > 500


def test_ground_fit_fast_mode_falls_back_when_threshold_too_strict() -> None:
    rng = np.random.default_rng(4321)
    x = rng.uniform(-200, 200, size=2400)
    y = rng.uniform(-200, 200, size=2400)
    z = 1000 + 0.03 * x + 0.01 * y + rng.normal(0, 2.2, size=2400)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    plane, mode = fit_ground_plane_conservative_fast(
        points=points,
        residual_mm=8.0,
        max_trials=120,
        fast_enabled=True,
        fast_sample_cap=900,
        fast_max_trials=20,
        fast_min_inlier_ratio=0.98,  # force fallback path on purpose
        fast_min_inliers=5000,
    )
    assert plane is not None
    assert mode == "fallback_full"


def test_support_region_closing_merges_split_object_halves() -> None:
    bbox = (0, 0, 64, 32)
    ground_points = []
    ground_uv = []
    for v in range(bbox[1], bbox[3]):
        for u in range(bbox[0], bbox[2]):
            ground_points.append([float(u), float(v), 1000.0])
            ground_uv.append([float(u), float(v)])

    object_points = []
    object_uv = []
    for v in range(8, 24):
        for u in range(15, 19):
            object_points.append([float(u), float(v), 970.0])
            object_uv.append([float(u), float(v)])
        for u in range(28, 32):
            object_points.append([float(u), float(v), 970.0])
            object_uv.append([float(u), float(v)])

    points = np.asarray(ground_points + object_points, dtype=np.float32)
    uv = np.asarray(ground_uv + object_uv, dtype=np.float32)

    plane = fit_ground_plane(points, residual_mm=1.0)
    assert plane is not None

    support = extract_support_region(
        points=points,
        uv=uv,
        bbox_xyxy=bbox,
        ref_uv=(24.0, 16.0),
        plane=plane,
        height_min_mm=10.0,
        close_px=11,
        min_area_px=12,
    )
    assert support is not None
    selected_uv = support["selected_uv"]
    assert selected_uv.shape[0] == len(object_uv)
    assert float(np.min(selected_uv[:, 0])) <= 15.0
    assert float(np.max(selected_uv[:, 0])) >= 31.0


def test_plane_aware_geometry_tracks_tilted_object_axis() -> None:
    rng = np.random.default_rng(77)
    plane_x = rng.uniform(-220, 220, size=2500)
    plane_y = rng.uniform(-180, 180, size=2500)
    plane_z = 1000 + 0.05 * plane_x + 0.18 * plane_y
    ground_points = np.stack([plane_x, plane_y, plane_z], axis=1).astype(np.float32)

    plane = fit_ground_plane(ground_points, residual_mm=1.0)
    assert plane is not None

    theta = np.deg2rad(38.0)
    major_dir = (plane.basis_u_cam * np.cos(theta)) + (plane.basis_v_cam * np.sin(theta))
    minor_dir = (-plane.basis_u_cam * np.sin(theta)) + (plane.basis_v_cam * np.cos(theta))
    center = np.array([25.0, -15.0, 998.55], dtype=np.float32)

    major = rng.uniform(-80.0, 80.0, size=1800)
    minor = rng.uniform(-18.0, 18.0, size=1800)
    height = rng.uniform(8.0, 24.0, size=1800)
    object_points = (
        center[None, :]
        + np.outer(major, major_dir)
        + np.outer(minor, minor_dir)
        + np.outer(height, plane.normal_cam)
    ).astype(np.float32)

    geom = estimate_object_geometry(object_points, plane=plane)
    assert geom is not None
    axis = geom["major_axis_cam"]
    alignment = abs(float(np.dot(axis, major_dir)))
    assert alignment > 0.95
    assert geom["axis_eig_ratio"] > 2.0
