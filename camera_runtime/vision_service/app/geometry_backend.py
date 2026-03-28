from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import (
    depth_roi_to_points as cpu_depth_roi_to_points,
    estimate_object_geometry as cpu_estimate_object_geometry,
    find_grasp_point as cpu_find_grasp_point,
    reproject_geometry_to_axis as cpu_reproject_geometry_to_axis,
    select_main_cluster as cpu_select_main_cluster,
)

LOGGER = logging.getLogger(__name__)

_VALID_BACKENDS = {"auto", "cpu", "torch", "cuml"}


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def normalize_geom_backend(raw: str | None) -> str:
    value = (raw or "auto").strip().lower()
    return value if value in _VALID_BACKENDS else "auto"


def detect_torch_cuda_available() -> bool:
    torch = _try_import_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_cuml_available() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except Exception:
        return False


def resolve_geom_backend(requested: str | None, torch_cuda_available: bool, cuml_available: bool) -> str:
    normalized = normalize_geom_backend(requested)
    if normalized == "cpu":
        return "cpu"
    if normalized == "torch":
        return "torch" if torch_cuda_available else "cpu"
    if normalized == "cuml":
        if cuml_available:
            return "cuml"
        if torch_cuda_available:
            return "torch"
        return "cpu"

    if cuml_available:
        return "cuml"
    if torch_cuda_available:
        return "torch"
    return "cpu"


def _pick_torch_device(yolo_device: str | None) -> str:
    raw = (yolo_device or "").strip().lower()
    if raw.startswith("cuda"):
        return raw
    return "cuda:0"


@dataclass(frozen=True)
class GeometryBackendSelection:
    requested: str
    resolved: str
    torch_cuda_available: bool
    cuml_available: bool


class GeometryBackend:
    name = "cpu"
    is_gpu = False

    def depth_roi_to_points(
        self,
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
        raise NotImplementedError

    def select_main_cluster(
        self,
        points: np.ndarray,
        uv: np.ndarray,
        ref_uv: tuple[float, float],
        eps_mm: float,
        min_samples: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def estimate_object_geometry(
        self,
        points: np.ndarray,
        plane: Any | None = None,
        trim_percentile: float = 98.0,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def reproject_geometry_to_axis(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        major_axis_cam: np.ndarray,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def find_grasp_point(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        width_limit_mm: float,
        window_length_mm: float,
        step_mm: float,
        min_points: int,
    ) -> dict[str, Any]:
        raise NotImplementedError


class CPUReferenceBackend(GeometryBackend):
    name = "cpu"
    is_gpu = False

    def depth_roi_to_points(
        self,
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
        return cpu_depth_roi_to_points(
            depth=depth,
            bbox_xyxy=bbox_xyxy,
            depth_scale=depth_scale,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
        )

    def select_main_cluster(
        self,
        points: np.ndarray,
        uv: np.ndarray,
        ref_uv: tuple[float, float],
        eps_mm: float,
        min_samples: int,
    ) -> np.ndarray:
        return cpu_select_main_cluster(
            points=points,
            uv=uv,
            ref_uv=ref_uv,
            eps_mm=eps_mm,
            min_samples=min_samples,
        )

    def estimate_object_geometry(
        self,
        points: np.ndarray,
        plane: Any | None = None,
        trim_percentile: float = 98.0,
    ) -> dict[str, Any] | None:
        return cpu_estimate_object_geometry(points=points, plane=plane, trim_percentile=trim_percentile)

    def reproject_geometry_to_axis(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        major_axis_cam: np.ndarray,
    ) -> dict[str, Any] | None:
        return cpu_reproject_geometry_to_axis(points=points, geometry=geometry, major_axis_cam=major_axis_cam)

    def find_grasp_point(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        width_limit_mm: float,
        window_length_mm: float,
        step_mm: float,
        min_points: int,
    ) -> dict[str, Any]:
        return cpu_find_grasp_point(
            points=points,
            geometry=geometry,
            width_limit_mm=width_limit_mm,
            window_length_mm=window_length_mm,
            step_mm=step_mm,
            min_points=min_points,
        )


class TorchCudaBackend(CPUReferenceBackend):
    name = "torch"
    is_gpu = True

    def __init__(self, device: str = "cuda:0") -> None:
        torch = _try_import_torch()
        if torch is None:
            raise RuntimeError("torch is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")
        self._torch = torch
        self._device = torch.device(device)
        self._probe_cuda_linalg()

    def _probe_cuda_linalg(self) -> None:
        try:
            probe = self._torch.tensor([[1.0, 0.1], [0.1, 2.0]], dtype=self._torch.float32, device=self._device)
            self._torch.linalg.eigh(probe)
        except Exception as exc:
            raise RuntimeError(f"torch CUDA linalg probe failed: {exc}") from exc

    def _normalize_vector(self, vec):
        norm = self._torch.linalg.norm(vec)
        if float(norm.item()) < 1e-6:
            return None
        return (vec / norm).to(dtype=self._torch.float32)

    def _to_numpy_f32(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

    def depth_roi_to_points(
        self,
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

        torch = self._torch
        roi_t = torch.as_tensor(roi, dtype=torch.float32, device=self._device)
        roi_h, roi_w = roi.shape
        v_coords = torch.arange(y1, y2, dtype=torch.float32, device=self._device).unsqueeze(1).expand(roi_h, roi_w)
        u_coords = torch.arange(x1, x2, dtype=torch.float32, device=self._device).unsqueeze(0).expand(roi_h, roi_w)

        z_mm = roi_t * float(depth_scale)
        valid = (z_mm >= float(min_depth_mm)) & (z_mm <= float(max_depth_mm))
        if not bool(torch.any(valid)):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        z_valid = z_mm[valid]
        u_valid = u_coords[valid]
        v_valid = v_coords[valid]

        x_mm = (u_valid - float(cx)) * z_valid / float(fx)
        y_mm = (v_valid - float(cy)) * z_valid / float(fy)
        points = torch.stack([x_mm, y_mm, z_valid], dim=1)
        uv = torch.stack([u_valid, v_valid], dim=1)
        return self._to_numpy_f32(points), self._to_numpy_f32(uv)

    def estimate_object_geometry(
        self,
        points: np.ndarray,
        plane: Any | None = None,
        trim_percentile: float = 98.0,
    ) -> dict[str, Any] | None:
        if points.shape[0] < 30:
            return None

        torch = self._torch
        points_t = torch.as_tensor(points, dtype=torch.float32, device=self._device)
        center = torch.quantile(points_t, q=0.5, dim=0)

        if plane is None:
            basis_u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self._device)
            basis_v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self._device)
            heights_mm = points_t[:, 2]
        else:
            basis_u = torch.as_tensor(plane.basis_u_cam, dtype=torch.float32, device=self._device)
            basis_v = torch.as_tensor(plane.basis_v_cam, dtype=torch.float32, device=self._device)
            normal = torch.as_tensor(plane.normal_cam, dtype=torch.float32, device=self._device)
            heights_mm = (points_t @ normal) + float(plane.offset_mm)

        delta = points_t - center
        coords_2d = torch.stack([delta @ basis_u, delta @ basis_v], dim=1).to(dtype=torch.float32)
        trimmed_coords = coords_2d
        if points.shape[0] >= 60:
            radii = torch.linalg.norm(coords_2d, dim=1)
            trim_q = max(50.0, min(99.5, float(trim_percentile))) / 100.0
            radius_limit = torch.quantile(radii, q=trim_q)
            trim_mask = radii <= radius_limit
            if int(torch.count_nonzero(trim_mask).item()) >= 30:
                trimmed_coords = coords_2d[trim_mask]

        if int(trimmed_coords.shape[0]) < 2:
            return None

        centered_trim = trimmed_coords - torch.mean(trimmed_coords, dim=0, keepdim=True)
        denom = max(1, int(trimmed_coords.shape[0]) - 1)
        cov = (centered_trim.transpose(0, 1) @ centered_trim) / float(denom)
        if tuple(cov.shape) != (2, 2):
            return None

        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        major_2d = eigvecs[:, order[0]]
        minor_2d = eigvecs[:, order[1]]

        major_cam = self._normalize_vector((basis_u * major_2d[0]) + (basis_v * major_2d[1]))
        minor_cam = self._normalize_vector((basis_u * minor_2d[0]) + (basis_v * minor_2d[1]))
        if major_cam is None or minor_cam is None:
            return None

        major_xy = self._normalize_vector(major_cam[:2])
        minor_xy = self._normalize_vector(minor_cam[:2])
        if major_xy is None:
            major_xy = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self._device)
        if minor_xy is None:
            minor_xy = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self._device)

        proj_major = coords_2d @ major_2d
        proj_minor = coords_2d @ minor_2d
        length_mm = float(torch.quantile(proj_major, q=0.95).item() - torch.quantile(proj_major, q=0.05).item())
        width_mm = float(torch.quantile(proj_minor, q=0.95).item() - torch.quantile(proj_minor, q=0.05).item())
        height_mm = float(torch.quantile(heights_mm, q=0.95).item() - torch.quantile(heights_mm, q=0.05).item())
        axis_eig_ratio = float((eigvals[0] / torch.clamp(eigvals[1], min=1e-6)).item())

        return {
            "center_xyz_mm": self._to_numpy_f32(center),
            "major_axis_cam": self._to_numpy_f32(major_cam),
            "minor_axis_cam": self._to_numpy_f32(minor_cam),
            "major_axis_xy": self._to_numpy_f32(major_xy),
            "minor_axis_xy": self._to_numpy_f32(minor_xy),
            "basis_u_cam": self._to_numpy_f32(basis_u),
            "basis_v_cam": self._to_numpy_f32(basis_v),
            "plane_normal_cam": (
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
                if plane is None
                else np.asarray(plane.normal_cam, dtype=np.float32)
            ),
            "proj_major": self._to_numpy_f32(proj_major),
            "proj_minor": self._to_numpy_f32(proj_minor),
            "length_mm": max(0.0, float(length_mm)),
            "width_mm": max(0.0, float(width_mm)),
            "height_mm": max(0.0, float(height_mm)),
            "axis_eig_ratio": float(axis_eig_ratio),
            "trimmed_ratio": float(float(trimmed_coords.shape[0]) / max(1.0, float(coords_2d.shape[0]))),
        }

    def reproject_geometry_to_axis(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        major_axis_cam: np.ndarray,
    ) -> dict[str, Any] | None:
        torch = self._torch
        axis_cam = self._normalize_vector(torch.as_tensor(major_axis_cam, dtype=torch.float32, device=self._device))
        if axis_cam is None:
            return None

        basis_u = torch.as_tensor(geometry["basis_u_cam"], dtype=torch.float32, device=self._device)
        basis_v = torch.as_tensor(geometry["basis_v_cam"], dtype=torch.float32, device=self._device)
        center = torch.as_tensor(geometry["center_xyz_mm"], dtype=torch.float32, device=self._device)
        points_t = torch.as_tensor(points, dtype=torch.float32, device=self._device)

        major_2d = torch.tensor([float(torch.dot(axis_cam, basis_u)), float(torch.dot(axis_cam, basis_v))], dtype=torch.float32, device=self._device)
        major_2d = self._normalize_vector(major_2d)
        if major_2d is None:
            return None
        minor_2d = torch.tensor([-float(major_2d[1]), float(major_2d[0])], dtype=torch.float32, device=self._device)

        major_cam = self._normalize_vector((basis_u * major_2d[0]) + (basis_v * major_2d[1]))
        minor_cam = self._normalize_vector((basis_u * minor_2d[0]) + (basis_v * minor_2d[1]))
        if major_cam is None or minor_cam is None:
            return None

        delta = points_t - center
        coords_2d = torch.stack([delta @ basis_u, delta @ basis_v], dim=1).to(dtype=torch.float32)
        proj_major = coords_2d @ major_2d
        proj_minor = coords_2d @ minor_2d

        major_xy = self._normalize_vector(major_cam[:2])
        minor_xy = self._normalize_vector(minor_cam[:2])
        if major_xy is None:
            major_xy = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self._device)
        if minor_xy is None:
            minor_xy = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self._device)

        updated = dict(geometry)
        updated["major_axis_cam"] = self._to_numpy_f32(major_cam)
        updated["minor_axis_cam"] = self._to_numpy_f32(minor_cam)
        updated["major_axis_xy"] = self._to_numpy_f32(major_xy)
        updated["minor_axis_xy"] = self._to_numpy_f32(minor_xy)
        updated["proj_major"] = self._to_numpy_f32(proj_major)
        updated["proj_minor"] = self._to_numpy_f32(proj_minor)
        updated["length_mm"] = max(0.0, float(torch.quantile(proj_major, q=0.95).item() - torch.quantile(proj_major, q=0.05).item()))
        updated["width_mm"] = max(0.0, float(torch.quantile(proj_minor, q=0.95).item() - torch.quantile(proj_minor, q=0.05).item()))
        return updated

    def find_grasp_point(
        self,
        points: np.ndarray,
        geometry: dict[str, Any],
        width_limit_mm: float,
        window_length_mm: float,
        step_mm: float,
        min_points: int,
    ) -> dict[str, Any]:
        torch = self._torch
        proj_major = torch.as_tensor(geometry["proj_major"], dtype=torch.float32, device=self._device)
        proj_minor = torch.as_tensor(geometry["proj_minor"], dtype=torch.float32, device=self._device)
        points_t = torch.as_tensor(points, dtype=torch.float32, device=self._device)
        center_xyz = np.asarray(geometry["center_xyz_mm"], dtype=np.float32)
        major_axis_cam = np.asarray(geometry["major_axis_cam"], dtype=np.float32)

        global_width = float(torch.quantile(proj_minor, q=0.95).item() - torch.quantile(proj_minor, q=0.05).item())
        selected_mask = None
        if global_width <= float(width_limit_mm):
            selected_mask = torch.ones(proj_major.shape[0], dtype=torch.bool, device=self._device)
        else:
            major_min = float(torch.min(proj_major).item())
            major_max = float(torch.max(proj_major).item())
            half_window = float(window_length_mm) / 2.0
            best_score = -1e18

            c = major_min
            while c <= major_max:
                local_mask = torch.abs(proj_major - c) <= half_window
                local_count = int(torch.sum(local_mask).item())
                if local_count >= int(min_points):
                    local_minor = proj_minor[local_mask]
                    local_width = float(torch.quantile(local_minor, q=0.95).item() - torch.quantile(local_minor, q=0.05).item())
                    if local_width <= float(width_limit_mm):
                        score = float(local_count) - local_width * 0.5
                        if score > best_score:
                            best_score = score
                            selected_mask = local_mask
                c += float(step_mm)

        if selected_mask is None or int(torch.sum(selected_mask).item()) < int(min_points):
            return {"status": "too_wide", "grasp_xyz_mm": None, "grasp_yaw_deg": None}

        selected_points = points_t[selected_mask]
        grasp_xyz = torch.quantile(selected_points, q=0.5, dim=0)
        local_minor = proj_minor[selected_mask]
        local_width = float(torch.quantile(local_minor, q=0.95).item() - torch.quantile(local_minor, q=0.05).item())
        grasp_xyz_np = self._to_numpy_f32(grasp_xyz)

        yaw_deg = float(math.degrees(math.atan2(float(major_axis_cam[1]), float(major_axis_cam[0]))))
        return {
            "status": "ok",
            "grasp_xyz_mm": grasp_xyz_np,
            "grasp_yaw_deg": yaw_deg,
            "grasp_axis_dir_cam": major_axis_cam.astype(np.float32),
            "grasp_local_width_mm": max(0.0, local_width),
            "center_xyz_mm": center_xyz.astype(np.float32),
        }


class OptionalCumlBackend(CPUReferenceBackend):
    name = "cuml"
    is_gpu = True

    def _fit_predict_labels(self, points: np.ndarray, eps_mm: float, min_samples: int) -> np.ndarray:
        from cuml.cluster import DBSCAN as CuMlDBSCAN

        labels = CuMlDBSCAN(eps=float(eps_mm), min_samples=int(min_samples)).fit_predict(points.astype(np.float32, copy=False))
        if hasattr(labels, "to_numpy"):
            labels = labels.to_numpy()
        elif hasattr(labels, "values_host"):
            labels = labels.values_host
        elif hasattr(labels, "get"):
            labels = labels.get()
        return np.asarray(labels).reshape(-1).astype(np.int32, copy=False)

    def select_main_cluster(
        self,
        points: np.ndarray,
        uv: np.ndarray,
        ref_uv: tuple[float, float],
        eps_mm: float,
        min_samples: int,
    ) -> np.ndarray:
        if points.shape[0] < int(min_samples):
            return np.ones(points.shape[0], dtype=bool)

        try:
            labels = self._fit_predict_labels(points, eps_mm=eps_mm, min_samples=min_samples)
        except Exception as exc:
            LOGGER.warning("cuml cluster failed; fallback to CPU DBSCAN: %s", exc)
            return super().select_main_cluster(points=points, uv=uv, ref_uv=ref_uv, eps_mm=eps_mm, min_samples=min_samples)

        valid_labels = [int(label) for label in np.unique(labels) if int(label) >= 0]
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


def build_geometry_backend(
    requested_backend: str | None,
    yolo_device: str | None,
) -> tuple[GeometryBackend, GeometryBackendSelection]:
    torch_cuda_available = detect_torch_cuda_available()
    cuml_available = detect_cuml_available()
    requested = normalize_geom_backend(requested_backend)
    resolved = resolve_geom_backend(
        requested=requested,
        torch_cuda_available=torch_cuda_available,
        cuml_available=cuml_available,
    )

    selection = GeometryBackendSelection(
        requested=requested,
        resolved=resolved,
        torch_cuda_available=torch_cuda_available,
        cuml_available=cuml_available,
    )

    if resolved == "torch":
        try:
            return TorchCudaBackend(device=_pick_torch_device(yolo_device)), selection
        except Exception as exc:
            LOGGER.warning("torch backend unavailable, fallback to cpu: %s", exc)
            return CPUReferenceBackend(), GeometryBackendSelection(
                requested=requested,
                resolved="cpu",
                torch_cuda_available=torch_cuda_available,
                cuml_available=cuml_available,
            )

    if resolved == "cuml":
        try:
            return OptionalCumlBackend(), selection
        except Exception as exc:
            LOGGER.warning("cuml backend unavailable, fallback to cpu: %s", exc)
            return CPUReferenceBackend(), GeometryBackendSelection(
                requested=requested,
                resolved="cpu",
                torch_cuda_available=torch_cuda_available,
                cuml_available=cuml_available,
            )

    return CPUReferenceBackend(), selection
