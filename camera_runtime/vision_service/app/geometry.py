from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor


@dataclass(frozen=True)
class GroundPlaneModel:
    inlier_mask: np.ndarray
    normal_cam: np.ndarray
    offset_mm: float
    basis_u_cam: np.ndarray
    basis_v_cam: np.ndarray
    coefficients: np.ndarray


def _normalize_vector(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return (vec / norm).astype(np.float32)


def _make_plane_basis(normal_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    basis_u = ref - np.dot(ref, normal_cam) * normal_cam
    if float(np.linalg.norm(basis_u)) < 1e-4:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        basis_u = ref - np.dot(ref, normal_cam) * normal_cam
    basis_u = _normalize_vector(basis_u)
    if basis_u is None:
        return None

    basis_v = np.cross(normal_cam, basis_u)
    basis_v = _normalize_vector(basis_v)
    if basis_v is None:
        return None
    return basis_u, basis_v


def median_depth_at(
    depth: np.ndarray,
    u: int,
    v: int,
    depth_scale: float,
    window: int,
    min_depth_mm: float,
    max_depth_mm: float,
) -> float | None:
    h, w = depth.shape[:2]
    half = max(0, window // 2)
    x1 = max(0, u - half)
    x2 = min(w, u + half + 1)
    y1 = max(0, v - half)
    y2 = min(h, v + half + 1)

    roi = depth[y1:y2, x1:x2].astype(np.float32) * depth_scale
    valid = (roi >= min_depth_mm) & (roi <= max_depth_mm)
    if not np.any(valid):
        return None
    return float(np.median(roi[valid]))


def depth_roi_to_points(
    depth: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    v_coords, u_coords = np.indices(roi.shape)
    u_abs = u_coords + x1
    v_abs = v_coords + y1

    z_mm = roi.astype(np.float32) * depth_scale
    valid = (z_mm >= min_depth_mm) & (z_mm <= max_depth_mm)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    z_valid = z_mm[valid]
    u_valid = u_abs[valid].astype(np.float32)
    v_valid = v_abs[valid].astype(np.float32)

    x_mm = (u_valid - cx) * z_valid / fx
    y_mm = (v_valid - cy) * z_valid / fy
    points = np.stack([x_mm, y_mm, z_valid], axis=1).astype(np.float32)
    uv = np.stack([u_valid, v_valid], axis=1).astype(np.float32)
    return points, uv


def fit_ground_plane(points: np.ndarray, residual_mm: float, max_trials: int = 120) -> GroundPlaneModel | None:
    if points.shape[0] < 100:
        return None

    max_trials = max(8, int(max_trials))
    xy = points[:, :2]
    z = points[:, 2]
    min_samples = max(30, min(80, int(points.shape[0] * 0.02)))
    try:
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=residual_mm,
                min_samples=min_samples,
                max_trials=max_trials,
                random_state=42,
            )
        except TypeError:
            ransac = RANSACRegressor(
                base_estimator=LinearRegression(),
                residual_threshold=residual_mm,
                min_samples=min_samples,
                max_trials=max_trials,
                random_state=42,
            )
        ransac.fit(xy, z)
        inlier_mask = ransac.inlier_mask_
        if inlier_mask is None:
            return None
        inlier_mask = inlier_mask.astype(bool)
        if int(np.count_nonzero(inlier_mask)) < 60:
            return None

        xy_inliers = xy[inlier_mask]
        z_inliers = z[inlier_mask]
        design = np.column_stack([xy_inliers, np.ones(xy_inliers.shape[0], dtype=np.float32)])
        coeffs, _, _, _ = np.linalg.lstsq(design, z_inliers, rcond=None)
        coeffs = coeffs.astype(np.float32)
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        raw_normal = np.array([a, b, -1.0], dtype=np.float32)
        normal_cam = _normalize_vector(raw_normal)
        if normal_cam is None:
            return None
        offset_mm = float(c / float(np.linalg.norm(raw_normal)))

        plane_basis = _make_plane_basis(normal_cam)
        if plane_basis is None:
            return None
        basis_u_cam, basis_v_cam = plane_basis

        return GroundPlaneModel(
            inlier_mask=inlier_mask,
            normal_cam=normal_cam,
            offset_mm=offset_mm,
            basis_u_cam=basis_u_cam,
            basis_v_cam=basis_v_cam,
            coefficients=coeffs,
        )
    except Exception:
        return None


def fit_ground_mask(points: np.ndarray, residual_mm: float, max_trials: int = 120) -> np.ndarray:
    plane = fit_ground_plane(points, residual_mm=residual_mm, max_trials=max_trials)
    if plane is None:
        return np.zeros(points.shape[0], dtype=bool)
    return plane.inlier_mask.astype(bool)


def point_heights_from_plane(points: np.ndarray, plane: GroundPlaneModel) -> np.ndarray:
    return (points @ plane.normal_cam) + plane.offset_mm


def extract_support_region(
    points: np.ndarray,
    uv: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    ref_uv: tuple[float, float],
    plane: GroundPlaneModel,
    height_min_mm: float,
    close_px: int,
    min_area_px: int = 16,
) -> dict[str, Any] | None:
    if points.shape[0] == 0:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = max(1, int(x2 - x1))
    bbox_h = max(1, int(y2 - y1))

    heights_mm = point_heights_from_plane(points, plane)
    elevated_mask = heights_mm >= float(height_min_mm)
    if int(np.count_nonzero(elevated_mask)) == 0:
        return None

    local_u = np.rint(uv[:, 0]).astype(np.int32) - int(x1)
    local_v = np.rint(uv[:, 1]).astype(np.int32) - int(y1)
    valid_uv = (local_u >= 0) & (local_u < bbox_w) & (local_v >= 0) & (local_v < bbox_h)
    support_mask = elevated_mask & valid_uv
    if int(np.count_nonzero(support_mask)) == 0:
        return None

    occupancy = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    occupancy[local_v[support_mask], local_u[support_mask]] = 255

    kernel_size = max(1, int(close_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(occupancy, connectivity=8)
    if num_labels <= 1:
        return None

    ref_u, ref_v = ref_uv
    best_label = -1
    best_dist = float("inf")
    best_area = -1
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        center_u = float(centroids[label][0]) + float(x1)
        center_v = float(centroids[label][1]) + float(y1)
        dist = math.hypot(center_u - ref_u, center_v - ref_v)
        if dist < best_dist or (abs(dist - best_dist) < 1e-6 and area > best_area):
            best_dist = dist
            best_area = area
            best_label = label

    if best_label < 0:
        return None

    selected_component = labels[local_v[support_mask], local_u[support_mask]] == best_label
    support_indices = np.flatnonzero(support_mask)
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    selected_mask[support_indices] = selected_component
    selected_count = int(np.count_nonzero(selected_mask))
    if selected_count == 0:
        return None

    area_px = int(stats[best_label, cv2.CC_STAT_AREA])
    bbox_area = max(1, bbox_w * bbox_h)
    density = float(selected_count / max(1, area_px))
    fill_ratio = float(area_px / bbox_area)

    return {
        "selected_mask": selected_mask,
        "selected_points": points[selected_mask],
        "selected_uv": uv[selected_mask],
        "selected_heights_mm": heights_mm[selected_mask].astype(np.float32),
        "area_px": area_px,
        "density": density,
        "fill_ratio": fill_ratio,
        "elevated_points": int(np.count_nonzero(elevated_mask)),
    }


def select_main_cluster(
    points: np.ndarray,
    uv: np.ndarray,
    ref_uv: tuple[float, float],
    eps_mm: float,
    min_samples: int,
) -> np.ndarray:
    if points.shape[0] < min_samples:
        return np.ones(points.shape[0], dtype=bool)

    labels = DBSCAN(eps=eps_mm, min_samples=min_samples).fit_predict(points)
    valid_labels = [label for label in np.unique(labels) if label >= 0]
    if not valid_labels:
        return np.ones(points.shape[0], dtype=bool)

    best_label = valid_labels[0]
    best_dist = float("inf")
    best_size = -1

    ref_u, ref_v = ref_uv
    for label in valid_labels:
        idx = labels == label
        cluster_uv = uv[idx]
        if cluster_uv.size == 0:
            continue
        center_u = float(np.mean(cluster_uv[:, 0]))
        center_v = float(np.mean(cluster_uv[:, 1]))
        dist = math.hypot(center_u - ref_u, center_v - ref_v)
        size = int(np.sum(idx))
        if dist < best_dist or (abs(dist - best_dist) < 1e-6 and size > best_size):
            best_dist = dist
            best_size = size
            best_label = label

    return labels == best_label


def estimate_object_geometry(
    points: np.ndarray,
    plane: GroundPlaneModel | None = None,
    trim_percentile: float = 98.0,
) -> dict[str, Any] | None:
    if points.shape[0] < 30:
        return None

    center = np.median(points, axis=0).astype(np.float32)
    if plane is None:
        basis_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        basis_v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        heights_mm = points[:, 2].astype(np.float32)
    else:
        basis_u = plane.basis_u_cam
        basis_v = plane.basis_v_cam
        heights_mm = point_heights_from_plane(points, plane).astype(np.float32)

    delta = points - center
    coords_2d = np.stack(
        [
            delta @ basis_u,
            delta @ basis_v,
        ],
        axis=1,
    ).astype(np.float32)

    trimmed_coords = coords_2d
    if points.shape[0] >= 60:
        radii = np.linalg.norm(coords_2d, axis=1)
        radius_limit = float(np.percentile(radii, max(50.0, min(99.5, trim_percentile))))
        trim_mask = radii <= radius_limit
        if int(np.count_nonzero(trim_mask)) >= 30:
            trimmed_coords = coords_2d[trim_mask]

    cov = np.cov(trimmed_coords.T)
    if cov.shape != (2, 2):
        return None

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    major_2d = eigvecs[:, order[0]]
    minor_2d = eigvecs[:, order[1]]

    major_cam = _normalize_vector((basis_u * float(major_2d[0])) + (basis_v * float(major_2d[1])))
    minor_cam = _normalize_vector((basis_u * float(minor_2d[0])) + (basis_v * float(minor_2d[1])))
    if major_cam is None or minor_cam is None:
        return None

    major_xy = _normalize_vector(major_cam[:2])
    minor_xy = _normalize_vector(minor_cam[:2])
    if major_xy is None:
        major_xy = np.array([1.0, 0.0], dtype=np.float32)
    if minor_xy is None:
        minor_xy = np.array([0.0, 1.0], dtype=np.float32)

    proj_major = coords_2d @ major_2d
    proj_minor = coords_2d @ minor_2d

    length_mm = float(np.percentile(proj_major, 95) - np.percentile(proj_major, 5))
    width_mm = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))
    height_mm = float(np.percentile(heights_mm, 95) - np.percentile(heights_mm, 5))
    axis_eig_ratio = float(eigvals[0] / max(eigvals[1], 1e-6))

    return {
        "center_xyz_mm": center,
        "major_axis_cam": major_cam.astype(np.float32),
        "minor_axis_cam": minor_cam.astype(np.float32),
        "major_axis_xy": major_xy.astype(np.float32),
        "minor_axis_xy": minor_xy.astype(np.float32),
        "basis_u_cam": basis_u.astype(np.float32),
        "basis_v_cam": basis_v.astype(np.float32),
        "plane_normal_cam": (np.array([0.0, 0.0, -1.0], dtype=np.float32) if plane is None else plane.normal_cam.astype(np.float32)),
        "proj_major": proj_major.astype(np.float32),
        "proj_minor": proj_minor.astype(np.float32),
        "length_mm": max(0.0, length_mm),
        "width_mm": max(0.0, width_mm),
        "height_mm": max(0.0, height_mm),
        "axis_eig_ratio": axis_eig_ratio,
        "trimmed_ratio": float(trimmed_coords.shape[0] / max(1, coords_2d.shape[0])),
    }


def axis_dir_to_yaw_deg(axis_dir_cam: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(axis_dir_cam[1]), float(axis_dir_cam[0]))))


def reproject_geometry_to_axis(
    points: np.ndarray,
    geometry: dict[str, Any],
    major_axis_cam: np.ndarray,
) -> dict[str, Any] | None:
    axis_cam = _normalize_vector(np.asarray(major_axis_cam, dtype=np.float32))
    if axis_cam is None:
        return None

    basis_u = np.asarray(geometry["basis_u_cam"], dtype=np.float32)
    basis_v = np.asarray(geometry["basis_v_cam"], dtype=np.float32)
    center = np.asarray(geometry["center_xyz_mm"], dtype=np.float32)

    major_2d = np.array(
        [
            float(np.dot(axis_cam, basis_u)),
            float(np.dot(axis_cam, basis_v)),
        ],
        dtype=np.float32,
    )
    major_2d = _normalize_vector(major_2d)
    if major_2d is None:
        return None
    minor_2d = np.array([-float(major_2d[1]), float(major_2d[0])], dtype=np.float32)

    major_cam = _normalize_vector((basis_u * float(major_2d[0])) + (basis_v * float(major_2d[1])))
    minor_cam = _normalize_vector((basis_u * float(minor_2d[0])) + (basis_v * float(minor_2d[1])))
    if major_cam is None or minor_cam is None:
        return None

    delta = points - center
    coords_2d = np.stack(
        [
            delta @ basis_u,
            delta @ basis_v,
        ],
        axis=1,
    ).astype(np.float32)
    proj_major = coords_2d @ major_2d
    proj_minor = coords_2d @ minor_2d
    major_xy = _normalize_vector(major_cam[:2])
    minor_xy = _normalize_vector(minor_cam[:2])
    if major_xy is None:
        major_xy = np.array([1.0, 0.0], dtype=np.float32)
    if minor_xy is None:
        minor_xy = np.array([0.0, 1.0], dtype=np.float32)

    updated = dict(geometry)
    updated["major_axis_cam"] = major_cam.astype(np.float32)
    updated["minor_axis_cam"] = minor_cam.astype(np.float32)
    updated["major_axis_xy"] = major_xy.astype(np.float32)
    updated["minor_axis_xy"] = minor_xy.astype(np.float32)
    updated["proj_major"] = proj_major.astype(np.float32)
    updated["proj_minor"] = proj_minor.astype(np.float32)
    updated["length_mm"] = max(0.0, float(np.percentile(proj_major, 95) - np.percentile(proj_major, 5)))
    updated["width_mm"] = max(0.0, float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5)))
    return updated


def find_grasp_point(
    points: np.ndarray,
    geometry: dict[str, Any],
    width_limit_mm: float,
    window_length_mm: float,
    step_mm: float,
    min_points: int,
) -> dict[str, Any]:
    proj_major = geometry["proj_major"]
    proj_minor = geometry["proj_minor"]
    center_xyz = geometry["center_xyz_mm"]
    major_axis_cam = geometry["major_axis_cam"]

    global_width = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))
    selected_mask = None

    if global_width <= width_limit_mm:
        selected_mask = np.ones(points.shape[0], dtype=bool)
    else:
        major_min = float(np.min(proj_major))
        major_max = float(np.max(proj_major))
        half_window = window_length_mm / 2.0
        best_score = -1e18

        c = major_min
        while c <= major_max:
            local_mask = np.abs(proj_major - c) <= half_window
            local_count = int(np.sum(local_mask))
            if local_count >= min_points:
                local_minor = proj_minor[local_mask]
                local_width = float(np.percentile(local_minor, 95) - np.percentile(local_minor, 5))
                if local_width <= width_limit_mm:
                    score = float(local_count) - local_width * 0.5
                    if score > best_score:
                        best_score = score
                        selected_mask = local_mask
            c += step_mm

    if selected_mask is None or int(np.sum(selected_mask)) < min_points:
        return {"status": "too_wide", "grasp_xyz_mm": None, "grasp_yaw_deg": None}

    selected_points = points[selected_mask]
    grasp_xyz = np.median(selected_points, axis=0)
    local_proj_minor = proj_minor[selected_mask]
    local_width = float(np.percentile(local_proj_minor, 95) - np.percentile(local_proj_minor, 5))

    return {
        "status": "ok",
        "grasp_xyz_mm": grasp_xyz.astype(np.float32),
        "grasp_yaw_deg": axis_dir_to_yaw_deg(major_axis_cam),
        "grasp_axis_dir_cam": major_axis_cam.astype(np.float32),
        "grasp_local_width_mm": max(0.0, local_width),
        "center_xyz_mm": center_xyz.astype(np.float32),
    }


def project_xyz_to_uv(xyz_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float] | None:
    z = float(xyz_mm[2])
    if z <= 1e-6:
        return None
    u = float((float(xyz_mm[0]) * fx / z) + cx)
    v = float((float(xyz_mm[1]) * fy / z) + cy)
    return u, v
