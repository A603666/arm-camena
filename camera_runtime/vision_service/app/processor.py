from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

from .config import AppConfig
from .geometry import (
    axis_dir_to_yaw_deg,
    extract_support_region,
    fit_ground_plane,
    median_depth_at,
    project_xyz_to_uv,
)
from .geometry_backend import CPUReferenceBackend, GeometryBackend, build_geometry_backend
from .receiver import ZmqFrameReceiver
from .state import SharedState
from .types import FramePacket


class VisionProcessor:
    def __init__(self, config: AppConfig, state: SharedState) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._cfg = config
        self._state = state
        self._receiver = ZmqFrameReceiver(
            endpoint=config.zmq_endpoint,
            topic=config.zmq_topic,
            timeout_ms=config.zmq_timeout_ms,
        )
        self._model = YOLO(str(config.model_path))
        self._is_engine_model = config.model_path.suffix.lower() == ".engine"
        self._predict_supports_half = not self._is_engine_model
        if self._is_engine_model:
            self._logger.info("YOLO TensorRT engine detected: %s", config.model_path)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision-processor", daemon=True)
        self._fps = 0.0
        self._prev_frame_time: float | None = None
        self._frame_index = 0
        self._cached_detections: list[dict[str, Any]] = []
        self._cached_geometry_payload: dict[str, Any] | None = None
        self._cached_geometry_bbox: list[int] | None = None
        self._cached_geometry_class_id: int | None = None
        self._cached_geometry_backend: str | None = None
        self._axis_tracker: dict[str, Any] | None = None
        self._cpu_geometry_backend: GeometryBackend = CPUReferenceBackend()
        self._geometry_backend, selection = build_geometry_backend(
            requested_backend=self._cfg.geom_backend,
            yolo_device=self._cfg.yolo_device,
        )
        self._geometry_force_cpu = False
        self._geometry_fallback_count = 0
        self._geometry_parity_mismatch_count = 0

        self._enable_cuda_benchmark()
        self._warmup_yolo_if_needed()
        self._logger.info(
            "geometry backend requested=%s resolved=%s torch_cuda=%s cuml=%s",
            selection.requested,
            selection.resolved,
            selection.torch_cuda_available,
            selection.cuml_available,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._receiver.close()

    def _enable_cuda_benchmark(self) -> None:
        if "cuda" not in self._cfg.yolo_device.lower():
            return
        try:
            import torch

            torch.backends.cudnn.benchmark = True
        except Exception as exc:
            self._logger.warning("unable to enable cudnn benchmark: %s", exc)

    def _use_fp16_infer(self) -> bool:
        return self._cfg.yolo_precision == "fp16" and ("cuda" in self._cfg.yolo_device.lower())

    def _warmup_yolo_if_needed(self) -> None:
        if not self._cfg.yolo_warmup:
            return
        warmup_size = max(64, min(1280, int(self._cfg.yolo_imgsz)))
        warmup_frame = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
        base_kwargs: dict[str, Any] = {
            "source": warmup_frame,
            "conf": self._cfg.yolo_conf,
            "imgsz": self._cfg.yolo_imgsz,
            "device": self._cfg.yolo_device,
            "max_det": 1,
            "verbose": False,
        }
        try:
            if self._predict_supports_half:
                half_kwargs = dict(base_kwargs)
                half_kwargs["half"] = self._use_fp16_infer()
                try:
                    self._model.predict(**half_kwargs)
                    self._logger.info(
                        "YOLO warmup done (device=%s, precision=%s)",
                        self._cfg.yolo_device,
                        self._cfg.yolo_precision,
                    )
                except TypeError:
                    self._predict_supports_half = False
                    self._model.predict(**base_kwargs)
                    self._logger.info(
                        "YOLO warmup done (device=%s, precision=fp32, half-arg unsupported)",
                        self._cfg.yolo_device,
                    )
            else:
                self._model.predict(**base_kwargs)
                if self._is_engine_model:
                    self._logger.info("YOLO warmup done (device=%s, precision=engine)", self._cfg.yolo_device)
                else:
                    self._logger.info("YOLO warmup done (device=%s, precision=fp32)", self._cfg.yolo_device)
        except Exception as exc:
            self._logger.warning("YOLO warmup failed: %s", exc)

    def _runtime_geometry_backend(self) -> GeometryBackend:
        if self._geometry_force_cpu:
            return self._cpu_geometry_backend
        return self._geometry_backend

    def _update_fps(self, now: float) -> float:
        if self._prev_frame_time is None:
            self._prev_frame_time = now
            return self._fps
        dt = max(1e-4, now - self._prev_frame_time)
        instant = 1.0 / dt
        self._fps = instant if self._fps <= 0.0 else (self._fps * 0.85 + instant * 0.15)
        self._prev_frame_time = now
        return self._fps

    def _run(self) -> None:
        while not self._stop_event.is_set():
            packet = self._receiver.recv()
            if packet is None:
                continue
            now = time.time()
            fps = self._update_fps(now)
            result, annotated = self._process_packet(packet, fps)
            self._state.update(result=result, annotated_jpeg=annotated, frame_wall_time=now)

    def _process_packet(self, packet: FramePacket, fps: float) -> tuple[dict[str, Any], bytes | None]:
        meta = packet.meta
        rgb = packet.rgb
        depth = packet.depth
        h, w = rgb.shape[:2]

        fx = float(meta.get("fx", 0.0))
        fy = float(meta.get("fy", 0.0))
        cx = float(meta.get("cx", w / 2.0))
        cy = float(meta.get("cy", h / 2.0))
        depth_scale = float(meta.get("depth_scale", 1.0))
        if fx <= 1.0:
            fx = float(w)
        if fy <= 1.0:
            fy = float(h)

        center_depth_mm = median_depth_at(
            depth=depth,
            u=w // 2,
            v=h // 2,
            depth_scale=depth_scale,
            window=self._cfg.center_depth_window,
            min_depth_mm=self._cfg.min_depth_mm,
            max_depth_mm=self._cfg.max_depth_mm,
        )

        self._frame_index += 1
        run_infer = (self._frame_index % self._cfg.infer_every_n == 0) or (not self._cached_detections)
        if run_infer:
            infer_start = time.perf_counter()
            self._cached_detections = self._infer(rgb)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
        else:
            infer_ms = None
        detections = self._cached_detections
        selected = self._select_target(detections, w, h)

        result: dict[str, Any] = {
            "status": "no_target",
            "target": None,
            "depth": {"center_depth_mm": center_depth_mm},
            "size": None,
            "grasp": None,
            "segmentation": None,
            "timing": {
                "frame_id": int(meta.get("frame_id", 0)),
                "ts_us": int(meta.get("ts_us", 0)),
                "fps": round(float(fps), 2),
                "infer_ran": bool(run_infer),
                "geometry_ran": False,
                "geometry_reused": False,
                "infer_ms": None if infer_ms is None else round(float(infer_ms), 2),
                "geometry_backend": None,
                "geometry_total_ms": None,
                "geometry_depth_to_points_ms": None,
                "geometry_ground_fit_ms": None,
                "geometry_support_ms": None,
                "geometry_cluster_ms": None,
                "geometry_pca_ms": None,
                "geometry_grasp_ms": None,
                "geometry_fallback_count": int(self._geometry_fallback_count),
                "geometry_parity_mismatch_count": int(self._geometry_parity_mismatch_count),
            },
        }

        annotated = rgb.copy()
        cv2.circle(annotated, (w // 2, h // 2), 5, (255, 255, 0), -1)
        if center_depth_mm is not None:
            cv2.putText(
                annotated,
                f"CenterDepth: {center_depth_mm:.1f} mm",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if selected is None:
            self._clear_geometry_cache()
            self._clear_axis_tracker()
            cv2.putText(annotated, "No target", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 255), 2, cv2.LINE_AA)
            return result, self._encode_annotated(annotated)

        bbox = selected["bbox_xyxy"]
        class_id = int(selected["class_id"])
        class_name = selected["class_name"]
        conf = float(selected["conf"])
        u_center = int((bbox[0] + bbox[2]) / 2)
        v_center = int((bbox[1] + bbox[3]) / 2)
        target_center_depth_mm = median_depth_at(
            depth=depth,
            u=u_center,
            v=v_center,
            depth_scale=depth_scale,
            window=self._cfg.center_depth_window,
            min_depth_mm=self._cfg.min_depth_mm,
            max_depth_mm=self._cfg.max_depth_mm,
        )

        result["target"] = {
            "class_id": class_id,
            "class_name": class_name,
            "conf": round(conf, 4),
            "bbox_xyxy": bbox,
        }
        result["depth"]["target_center_depth_mm"] = target_center_depth_mm

        run_geometry = self._should_run_geometry(class_id=class_id, bbox=bbox)
        if (not run_geometry) and self._cached_geometry_payload is None:
            run_geometry = True
        result["timing"]["geometry_ran"] = bool(run_geometry)
        result["timing"]["geometry_reused"] = bool(not run_geometry)

        if run_geometry:
            geometry_payload, geometry_timing, backend_used = self._run_geometry_with_fallback(
                depth=depth,
                bbox=bbox,
                class_id=class_id,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ref_uv=(u_center, v_center),
            )
            self._cached_geometry_payload = geometry_payload
            self._cached_geometry_bbox = [int(v) for v in bbox]
            self._cached_geometry_class_id = class_id
            self._cached_geometry_backend = backend_used
            result["timing"]["geometry_backend"] = backend_used
            result["timing"]["geometry_total_ms"] = round(float(geometry_timing.get("total_ms", 0.0)), 2)
            result["timing"]["geometry_depth_to_points_ms"] = round(float(geometry_timing.get("depth_to_points_ms", 0.0)), 2)
            result["timing"]["geometry_ground_fit_ms"] = round(float(geometry_timing.get("ground_fit_ms", 0.0)), 2)
            result["timing"]["geometry_support_ms"] = round(float(geometry_timing.get("support_ms", 0.0)), 2)
            result["timing"]["geometry_cluster_ms"] = round(float(geometry_timing.get("cluster_ms", 0.0)), 2)
            result["timing"]["geometry_pca_ms"] = round(float(geometry_timing.get("pca_ms", 0.0)), 2)
            result["timing"]["geometry_grasp_ms"] = round(float(geometry_timing.get("grasp_ms", 0.0)), 2)
            result["timing"]["geometry_fallback_count"] = int(self._geometry_fallback_count)
            result["timing"]["geometry_parity_mismatch_count"] = int(self._geometry_parity_mismatch_count)
        else:
            geometry_payload = self._cached_geometry_payload or self._make_geometry_payload(status="invalid_depth", size=None, grasp=None)
            result["timing"]["geometry_backend"] = self._cached_geometry_backend

        result["status"] = str(geometry_payload["status"])
        result["size"] = self._clone_optional_dict(geometry_payload["size"])
        result["grasp"] = self._clone_optional_dict(geometry_payload["grasp"])
        result["segmentation"] = self._clone_optional_dict(geometry_payload.get("segmentation"))
        self._draw_segmentation_overlay(
            image=annotated,
            segmentation=result["segmentation"],
            visualization=self._clone_optional_dict(geometry_payload.get("visualization")),
        )
        self._draw_target_overlay(
            annotated,
            bbox,
            class_name,
            conf,
            result["status"],
            result["size"],
            result["grasp"],
        )
        return result, self._encode_annotated(annotated)

    def _run_geometry_with_fallback(
        self,
        depth: np.ndarray,
        bbox: list[int],
        class_id: int,
        depth_scale: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        ref_uv: tuple[int, int],
    ) -> tuple[dict[str, Any], dict[str, float], str]:
        backend = self._runtime_geometry_backend()
        backend_name = backend.name
        if self._geometry_force_cpu and backend_name == "cpu" and self._geometry_backend.name != "cpu":
            backend_name = "cpu_forced"

        axis_tracker_before = copy.deepcopy(self._axis_tracker)
        try:
            payload, timing = self._compute_geometry_payload(
                depth=depth,
                bbox=bbox,
                class_id=class_id,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ref_uv=ref_uv,
                backend=backend,
            )
        except Exception as exc:
            self._logger.warning("geometry backend %s failed, fallback to cpu: %s", backend_name, exc)
            self._axis_tracker = axis_tracker_before
            payload, timing = self._compute_geometry_payload(
                depth=depth,
                bbox=bbox,
                class_id=class_id,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ref_uv=ref_uv,
                backend=self._cpu_geometry_backend,
            )
            self._geometry_fallback_count += 1
            return payload, timing, "cpu_fallback"

        should_check_parity = (
            self._cfg.geom_parity_check
            and backend.name != "cpu"
            and (self._frame_index % max(1, self._cfg.geom_parity_every_n) == 0)
        )
        if not should_check_parity:
            return payload, timing, backend_name

        axis_tracker_after_primary = copy.deepcopy(self._axis_tracker)
        try:
            self._axis_tracker = copy.deepcopy(axis_tracker_before)
            cpu_payload, cpu_timing = self._compute_geometry_payload(
                depth=depth,
                bbox=bbox,
                class_id=class_id,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ref_uv=ref_uv,
                backend=self._cpu_geometry_backend,
            )
            axis_tracker_after_cpu = copy.deepcopy(self._axis_tracker)
        except Exception as exc:
            self._axis_tracker = axis_tracker_after_primary
            self._logger.warning("geometry parity CPU check failed: %s", exc)
            return payload, timing, backend_name

        parity_ok, reason = self._geometry_payload_within_threshold(payload, cpu_payload)
        if parity_ok:
            self._axis_tracker = axis_tracker_after_primary
            return payload, timing, backend_name

        self._geometry_parity_mismatch_count += 1
        self._geometry_fallback_count += 1
        self._geometry_force_cpu = True
        self._axis_tracker = axis_tracker_after_cpu
        self._logger.warning("geometry parity mismatch, force cpu backend: %s", reason)
        return cpu_payload, cpu_timing, "cpu_forced_parity"

    @staticmethod
    def _yaw_delta_deg(lhs: float, rhs: float) -> float:
        raw = abs(float(lhs) - float(rhs)) % 360.0
        return min(raw, 360.0 - raw)

    @classmethod
    def _geometry_payload_within_threshold(
        cls,
        lhs: dict[str, Any],
        rhs: dict[str, Any],
    ) -> tuple[bool, str]:
        lhs_status = str(lhs.get("status", ""))
        rhs_status = str(rhs.get("status", ""))
        if lhs_status != rhs_status:
            return False, f"status mismatch {lhs_status} != {rhs_status}"
        if lhs_status != "ok":
            return True, "status not ok"

        lhs_size = lhs.get("size")
        rhs_size = rhs.get("size")
        if bool(lhs_size) != bool(rhs_size):
            return False, "size presence mismatch"
        if lhs_size and rhs_size:
            for key in ("length_mm", "width_mm", "height_mm"):
                if abs(float(lhs_size.get(key, 0.0)) - float(rhs_size.get(key, 0.0))) > 3.0:
                    return False, f"size mismatch on {key}"

        lhs_grasp = lhs.get("grasp")
        rhs_grasp = rhs.get("grasp")
        if bool(lhs_grasp) != bool(rhs_grasp):
            return False, "grasp presence mismatch"
        if lhs_grasp and rhs_grasp:
            for key in ("x_mm", "y_mm", "z_mm"):
                if abs(float(lhs_grasp.get(key, 0.0)) - float(rhs_grasp.get(key, 0.0))) > 2.0:
                    return False, f"grasp mismatch on {key}"
            yaw_delta = cls._yaw_delta_deg(float(lhs_grasp.get("yaw_deg", 0.0)), float(rhs_grasp.get("yaw_deg", 0.0)))
            if yaw_delta > 2.0:
                return False, "grasp yaw mismatch"
        return True, "within-threshold"

    def _compute_geometry_payload(
        self,
        depth: np.ndarray,
        bbox: list[int],
        class_id: int,
        depth_scale: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        ref_uv: tuple[int, int],
        backend: GeometryBackend,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        overall_start = time.perf_counter()
        timings: dict[str, float] = {
            "depth_to_points_ms": 0.0,
            "ground_fit_ms": 0.0,
            "support_ms": 0.0,
            "cluster_ms": 0.0,
            "pca_ms": 0.0,
            "grasp_ms": 0.0,
            "total_ms": 0.0,
        }

        def finish(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
            timings["total_ms"] = (time.perf_counter() - overall_start) * 1000.0
            return payload, timings

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(bbox, w, h)
        t = time.perf_counter()
        points, uv = backend.depth_roi_to_points(
            depth=depth,
            bbox_xyxy=(x1, y1, x2, y2),
            depth_scale=depth_scale,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            min_depth_mm=self._cfg.min_depth_mm,
            max_depth_mm=self._cfg.max_depth_mm,
        )
        timings["depth_to_points_ms"] = (time.perf_counter() - t) * 1000.0

        segmentation: dict[str, Any] = {
            "input_points": int(points.shape[0]),
            "ground_points": 0,
            "non_ground_points": int(points.shape[0]),
            "main_cluster_points": 0,
            "ground_ratio": None,
            "support_points": 0,
            "support_area_px": None,
            "support_density": None,
            "support_fill_ratio": None,
            "axis_eig_ratio": None,
        }
        if points.shape[0] < self._cfg.min_object_points:
            return finish(
                self._make_geometry_payload(
                    status="invalid_depth",
                    size=None,
                    grasp=None,
                    segmentation=segmentation,
                    visualization=None,
                )
            )

        if points.shape[0] > self._cfg.max_points_for_geometry:
            stride = max(1, (points.shape[0] + self._cfg.max_points_for_geometry - 1) // self._cfg.max_points_for_geometry)
            points = points[::stride]
            uv = uv[::stride]
            segmentation["input_points"] = int(points.shape[0])
            segmentation["non_ground_points"] = int(points.shape[0])

        t = time.perf_counter()
        ground_plane = fit_ground_plane(
            points,
            residual_mm=self._cfg.ransac_residual_mm,
            max_trials=self._cfg.ransac_max_trials,
        )
        timings["ground_fit_ms"] = (time.perf_counter() - t) * 1000.0

        ground_mask = np.zeros(points.shape[0], dtype=bool) if ground_plane is None else ground_plane.inlier_mask.astype(bool)
        ground_count = int(np.count_nonzero(ground_mask))
        non_ground_count = int(points.shape[0] - ground_count)
        segmentation["ground_points"] = ground_count
        segmentation["non_ground_points"] = non_ground_count
        segmentation["ground_ratio"] = None if ground_plane is None else round(float(ground_count / max(1, points.shape[0])), 4)
        object_points = points[~ground_mask]
        object_uv = uv[~ground_mask]
        if object_points.shape[0] < self._cfg.min_object_points:
            object_points = points
            object_uv = uv

        visualization = {
            "ground_uv": self._sample_uv_points(uv[ground_mask], max_points=80),
        }
        main_points = object_points
        main_uv = object_uv
        support_source = "legacy_cluster"
        support_stats: dict[str, Any] = {
            "area_px": int(object_points.shape[0]),
            "density": 1.0,
            "fill_ratio": float(object_points.shape[0] / max(1, (x2 - x1) * (y2 - y1))),
            "elevated_points": int(object_points.shape[0]),
        }

        if ground_plane is not None:
            t = time.perf_counter()
            support_region = extract_support_region(
                points=points,
                uv=uv,
                bbox_xyxy=(x1, y1, x2, y2),
                ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
                plane=ground_plane,
                height_min_mm=self._cfg.object_height_min_mm,
                close_px=self._cfg.support_close_px,
                min_area_px=max(12, (self._cfg.support_close_px * self._cfg.support_close_px) // 2),
            )
            timings["support_ms"] = (time.perf_counter() - t) * 1000.0
            if support_region is not None and support_region["selected_points"].shape[0] >= self._cfg.min_object_points:
                main_points = support_region["selected_points"]
                main_uv = support_region["selected_uv"]
                support_stats = support_region
                support_source = "support_region"

        if support_source != "support_region":
            t = time.perf_counter()
            cluster_mask = backend.select_main_cluster(
                points=object_points,
                uv=object_uv,
                ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
                eps_mm=self._cfg.dbscan_eps_mm,
                min_samples=self._cfg.dbscan_min_samples,
            )
            timings["cluster_ms"] = (time.perf_counter() - t) * 1000.0
            main_points = object_points[cluster_mask]
            main_uv = object_uv[cluster_mask]
            support_stats = {
                "area_px": int(main_points.shape[0]),
                "density": 1.0,
                "fill_ratio": float(main_points.shape[0] / max(1, (x2 - x1) * (y2 - y1))),
                "elevated_points": int(main_points.shape[0]),
            }

        segmentation["main_cluster_points"] = int(main_points.shape[0])
        segmentation["support_points"] = int(main_points.shape[0])
        segmentation["support_area_px"] = int(support_stats.get("area_px", 0))
        segmentation["support_density"] = round(float(support_stats.get("density", 0.0)), 4)
        segmentation["support_fill_ratio"] = round(float(support_stats.get("fill_ratio", 0.0)), 4)
        visualization["main_uv"] = self._sample_uv_points(main_uv, max_points=80)
        visualization["support_uv"] = self._sample_uv_points(main_uv, max_points=80)
        visualization["support_source"] = support_source

        if main_points.shape[0] < self._cfg.min_object_points:
            return finish(
                self._make_geometry_payload(
                    status="invalid_depth",
                    size=None,
                    grasp=None,
                    segmentation=segmentation,
                    visualization=visualization,
                )
            )

        t = time.perf_counter()
        geometry = backend.estimate_object_geometry(main_points, plane=ground_plane)
        if geometry is None:
            geometry = backend.estimate_object_geometry(main_points)
        if geometry is None:
            timings["pca_ms"] = (time.perf_counter() - t) * 1000.0
            return finish(
                self._make_geometry_payload(
                    status="invalid_depth",
                    size=None,
                    grasp=None,
                    segmentation=segmentation,
                    visualization=visualization,
                )
            )

        axis_quality = self._score_axis_quality(
            axis_eig_ratio=float(geometry["axis_eig_ratio"]),
            point_count=int(main_points.shape[0]),
            density=float(support_stats.get("density", 0.0)),
        )
        axis_reliable = self._is_axis_reliable(
            axis_eig_ratio=float(geometry["axis_eig_ratio"]),
            point_count=int(main_points.shape[0]),
            area_px=int(support_stats.get("area_px", 0)),
            density=float(support_stats.get("density", 0.0)),
        )
        stable_axis_dir_cam, axis_state, axis_quality_out = self._stabilize_axis(
            axis_dir_cam=np.asarray(geometry["major_axis_cam"], dtype=np.float32),
            reliable=axis_reliable,
            axis_quality=axis_quality,
            class_id=int(class_id),
            bbox=[int(v) for v in bbox],
        )
        if stable_axis_dir_cam is None:
            stable_axis_dir_cam = np.asarray(geometry["major_axis_cam"], dtype=np.float32)
        geometry_for_grasp = backend.reproject_geometry_to_axis(main_points, geometry, stable_axis_dir_cam) or geometry
        timings["pca_ms"] = (time.perf_counter() - t) * 1000.0
        segmentation["axis_eig_ratio"] = round(float(geometry["axis_eig_ratio"]), 4)

        center_xyz = geometry["center_xyz_mm"]
        major_axis_cam = np.asarray(geometry_for_grasp["major_axis_cam"], dtype=np.float32)
        half_major_mm = max(10.0, float(geometry_for_grasp["length_mm"]) * 0.5)
        axis_p1_xyz = np.array(
            [
                float(center_xyz[0]) - float(major_axis_cam[0]) * half_major_mm,
                float(center_xyz[1]) - float(major_axis_cam[1]) * half_major_mm,
                float(center_xyz[2]) - float(major_axis_cam[2]) * half_major_mm,
            ],
            dtype=np.float32,
        )
        axis_p2_xyz = np.array(
            [
                float(center_xyz[0]) + float(major_axis_cam[0]) * half_major_mm,
                float(center_xyz[1]) + float(major_axis_cam[1]) * half_major_mm,
                float(center_xyz[2]) + float(major_axis_cam[2]) * half_major_mm,
            ],
            dtype=np.float32,
        )
        axis_p1_uv = project_xyz_to_uv(axis_p1_xyz, fx=fx, fy=fy, cx=cx, cy=cy)
        axis_p2_uv = project_xyz_to_uv(axis_p2_xyz, fx=fx, fy=fy, cx=cx, cy=cy)
        if axis_p1_uv is not None and axis_p2_uv is not None:
            visualization["major_axis_uv"] = [
                [int(axis_p1_uv[0]), int(axis_p1_uv[1])],
                [int(axis_p2_uv[0]), int(axis_p2_uv[1])],
            ]

        size = {
            "length_mm": round(float(geometry_for_grasp["length_mm"]), 2),
            "width_mm": round(float(geometry_for_grasp["width_mm"]), 2),
            "height_mm": round(float(geometry["height_mm"]), 2),
        }

        t = time.perf_counter()
        grasp = backend.find_grasp_point(
            points=main_points,
            geometry=geometry_for_grasp,
            width_limit_mm=self._cfg.gripper_width_limit_mm,
            window_length_mm=self._cfg.grasp_window_length_mm,
            step_mm=self._cfg.grasp_window_step_mm,
            min_points=self._cfg.grasp_window_min_points,
        )
        timings["grasp_ms"] = (time.perf_counter() - t) * 1000.0
        if grasp["status"] != "ok":
            return finish(
                self._make_geometry_payload(
                    status="too_wide",
                    size=size,
                    grasp=None,
                    segmentation=segmentation,
                    visualization=visualization,
                )
            )

        grasp_xyz = grasp["grasp_xyz_mm"]
        grasp_uv = project_xyz_to_uv(grasp_xyz, fx=fx, fy=fy, cx=cx, cy=cy)
        grasp_uv_int = None if grasp_uv is None else [int(grasp_uv[0]), int(grasp_uv[1])]
        grasp_result = {
            "x_mm": round(float(grasp_xyz[0]), 2),
            "y_mm": round(float(grasp_xyz[1]), 2),
            "z_mm": round(float(grasp_xyz[2]), 2),
            "u": grasp_uv_int[0] if grasp_uv_int else None,
            "v": grasp_uv_int[1] if grasp_uv_int else None,
            "yaw_deg": round(float(axis_dir_to_yaw_deg(stable_axis_dir_cam)), 2),
            "axis_dir_cam": [round(float(v), 4) for v in stable_axis_dir_cam.tolist()],
            "axis_quality": round(float(axis_quality_out), 3),
            "axis_state": axis_state,
        }
        return finish(
            self._make_geometry_payload(
                status="ok",
                size=size,
                grasp=grasp_result,
                segmentation=segmentation,
                visualization=visualization,
            )
        )

    def _should_run_geometry(self, class_id: int, bbox: list[int]) -> bool:
        if (
            self._cached_geometry_payload is None
            or self._cached_geometry_bbox is None
            or self._cached_geometry_class_id is None
        ):
            return True
        if self._cached_geometry_class_id != int(class_id):
            return True
        if self._frame_index % self._cfg.geometry_every_n == 0:
            return True
        return self._bbox_iou(self._cached_geometry_bbox, bbox) < self._cfg.geometry_force_recalc_iou

    def _clear_geometry_cache(self) -> None:
        self._cached_geometry_payload = None
        self._cached_geometry_bbox = None
        self._cached_geometry_class_id = None
        self._cached_geometry_backend = None

    def _clear_axis_tracker(self) -> None:
        self._axis_tracker = None

    def _score_axis_quality(self, axis_eig_ratio: float, point_count: int, density: float) -> float:
        eig_score = max(0.0, min(1.0, (float(axis_eig_ratio) - 1.0) / 2.0))
        point_score = max(0.0, min(1.0, float(point_count) / max(1.0, float(self._cfg.min_object_points * 2))))
        density_score = max(0.0, min(1.0, float(density) / 0.35))
        return max(0.0, min(1.0, 0.55 * eig_score + 0.25 * point_score + 0.20 * density_score))

    def _is_axis_reliable(self, axis_eig_ratio: float, point_count: int, area_px: int, density: float) -> bool:
        min_area_px = max(12, (self._cfg.support_close_px * self._cfg.support_close_px) // 2)
        return (
            float(axis_eig_ratio) >= float(self._cfg.axis_eig_ratio_min)
            and int(point_count) >= int(self._cfg.min_object_points)
            and int(area_px) >= min_area_px
            and float(density) >= 0.12
        )

    def _stabilize_axis(
        self,
        axis_dir_cam: np.ndarray | None,
        reliable: bool,
        axis_quality: float,
        class_id: int,
        bbox: list[int],
    ) -> tuple[np.ndarray | None, str, float]:
        raw_axis = None if axis_dir_cam is None else np.asarray(axis_dir_cam, dtype=np.float32)
        if raw_axis is not None:
            raw_norm = float(np.linalg.norm(raw_axis))
            raw_axis = None if raw_norm < 1e-6 else (raw_axis / raw_norm).astype(np.float32)

        tracker = self._axis_tracker
        continuity = False
        prev_axis = None if tracker is None else tracker.get("accepted_dir_cam")
        if (
            tracker is not None
            and prev_axis is not None
            and tracker.get("class_id") == int(class_id)
            and tracker.get("bbox") is not None
        ):
            continuity = self._bbox_iou(tracker["bbox"], bbox) >= self._cfg.geometry_force_recalc_iou

        aligned_axis = raw_axis
        if continuity and raw_axis is not None:
            prev_axis_arr = np.asarray(prev_axis, dtype=np.float32)
            if float(np.dot(prev_axis_arr, raw_axis)) < 0.0:
                aligned_axis = (-raw_axis).astype(np.float32)

        if reliable and aligned_axis is not None:
            smoothed_axis = aligned_axis
            if continuity and prev_axis is not None:
                prev_axis_arr = np.asarray(prev_axis, dtype=np.float32)
                alpha = float(self._cfg.axis_smooth_alpha)
                blended = ((1.0 - alpha) * prev_axis_arr) + (alpha * aligned_axis)
                blend_norm = float(np.linalg.norm(blended))
                if blend_norm >= 1e-6:
                    smoothed_axis = (blended / blend_norm).astype(np.float32)

            self._axis_tracker = {
                "class_id": int(class_id),
                "bbox": [int(v) for v in bbox],
                "accepted_dir_cam": smoothed_axis,
                "accepted_quality": float(axis_quality),
                "bad_streak": 0,
            }
            return smoothed_axis, "live", float(axis_quality)

        if continuity and prev_axis is not None:
            bad_streak = int(tracker.get("bad_streak", 0)) + 1
            tracker["bbox"] = [int(v) for v in bbox]
            tracker["bad_streak"] = bad_streak
            if bad_streak <= int(self._cfg.axis_hold_frames):
                held_quality = max(float(axis_quality), min(1.0, float(tracker.get("accepted_quality", 0.0)) * 0.85))
                return np.asarray(prev_axis, dtype=np.float32), "held", held_quality

        if aligned_axis is not None:
            return aligned_axis, "unreliable", float(axis_quality)
        if continuity and prev_axis is not None:
            return np.asarray(prev_axis, dtype=np.float32), "unreliable", float(axis_quality)
        return None, "unreliable", float(axis_quality)

    @staticmethod
    def _make_geometry_payload(
        status: str,
        size: dict[str, Any] | None,
        grasp: dict[str, Any] | None,
        segmentation: dict[str, Any] | None = None,
        visualization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "size": size,
            "grasp": grasp,
            "segmentation": segmentation,
            "visualization": visualization,
        }

    @staticmethod
    def _clone_optional_dict(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if payload is None else dict(payload)

    @staticmethod
    def _sample_uv_points(uv: np.ndarray, max_points: int = 80) -> list[list[int]]:
        if uv.size == 0:
            return []
        step = max(1, (uv.shape[0] + max_points - 1) // max_points)
        sampled = uv[::step][:max_points]
        return [[int(p[0]), int(p[1])] for p in sampled]

    @staticmethod
    def _draw_segmentation_overlay(
        image: np.ndarray,
        segmentation: dict[str, Any] | None,
        visualization: dict[str, Any] | None,
    ) -> None:
        if segmentation:
            g = int(segmentation.get("ground_points", 0))
            ng = int(segmentation.get("non_ground_points", 0))
            sp = int(segmentation.get("support_points", segmentation.get("main_cluster_points", 0)))
            ratio = segmentation.get("ground_ratio")
            ratio_text = "-" if ratio is None else f"{float(ratio) * 100:.1f}%"
            cv2.putText(
                image,
                f"Seg G/NG/S: {g}/{ng}/{sp} ({ratio_text})",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 80),
                2,
                cv2.LINE_AA,
            )
        if not visualization:
            return

        for point in visualization.get("ground_uv", []):
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        for point in visualization.get("support_uv", visualization.get("main_uv", [])):
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 255), -1)
        major_axis = visualization.get("major_axis_uv")
        if isinstance(major_axis, list) and len(major_axis) == 2:
            p1 = (int(major_axis[0][0]), int(major_axis[0][1]))
            p2 = (int(major_axis[1][0]), int(major_axis[1][1]))
            cv2.line(image, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image, p1, 4, (0, 255, 255), -1)
            cv2.circle(image, p2, 4, (0, 255, 255), -1)

    def _infer(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        predict_kwargs: dict[str, Any] = {
            "source": image_bgr,
            "conf": self._cfg.yolo_conf,
            "imgsz": self._cfg.yolo_imgsz,
            "device": self._cfg.yolo_device,
            "max_det": 20,
            "verbose": False,
        }
        if self._predict_supports_half:
            predict_kwargs_with_half = dict(predict_kwargs)
            predict_kwargs_with_half["half"] = self._use_fp16_infer()
            try:
                pred = self._model.predict(**predict_kwargs_with_half)[0]
            except TypeError:
                self._predict_supports_half = False
                pred = self._model.predict(**predict_kwargs)[0]
        else:
            pred = self._model.predict(**predict_kwargs)[0]

        detections: list[dict[str, Any]] = []
        names = pred.names if isinstance(pred.names, dict) else {}
        for box in pred.boxes:
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": str(names.get(cls_id, f"class_{cls_id}")),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return detections

    @staticmethod
    def _select_target(detections: list[dict[str, Any]], width: int, height: int) -> dict[str, Any] | None:
        if not detections:
            return None
        cx = width / 2.0
        cy = height / 2.0

        def center_distance(det: dict[str, Any]) -> float:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            ux = (x1 + x2) / 2.0
            vy = (y1 + y2) / 2.0
            return float(np.hypot(ux - cx, vy - cy))

        detections.sort(key=center_distance)
        return detections[0]

    @staticmethod
    def _clamp_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(bbox[0])))
        y1 = max(0, min(height - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(width, int(bbox[2])))
        y2 = max(y1 + 1, min(height, int(bbox[3])))
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_iou(lhs: list[int], rhs: list[int]) -> float:
        x_left = max(int(lhs[0]), int(rhs[0]))
        y_top = max(int(lhs[1]), int(rhs[1]))
        x_right = min(int(lhs[2]), int(rhs[2]))
        y_bottom = min(int(lhs[3]), int(rhs[3]))

        inter_w = max(0, x_right - x_left)
        inter_h = max(0, y_bottom - y_top)
        inter_area = float(inter_w * inter_h)
        if inter_area <= 0.0:
            return 0.0

        lhs_area = float(max(0, int(lhs[2]) - int(lhs[0])) * max(0, int(lhs[3]) - int(lhs[1])))
        rhs_area = float(max(0, int(rhs[2]) - int(rhs[0])) * max(0, int(rhs[3]) - int(rhs[1])))
        union_area = lhs_area + rhs_area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def _encode_annotated(self, annotated: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self._cfg.annotated_jpeg_quality],
        )
        if not ok:
            return None
        return encoded.tobytes()

    @staticmethod
    def _draw_target_overlay(
        image: np.ndarray,
        bbox: list[int],
        class_name: str,
        conf: float,
        status: str,
        size: dict[str, Any] | None,
        grasp: dict[str, Any] | None,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if status == "ok" else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{class_name} {conf:.2f} {status}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        if size:
            cv2.putText(
                image,
                f"L/W/H: {size['length_mm']:.1f}/{size['width_mm']:.1f}/{size['height_mm']:.1f} mm",
                (x1, min(image.shape[0] - 8, y2 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if grasp and grasp.get("u") is not None and grasp.get("v") is not None:
            gu, gv = int(grasp["u"]), int(grasp["v"])
            cv2.circle(image, (gu, gv), 6, (0, 0, 255), -1)
            axis_state = str(grasp.get("axis_state", "live"))
            axis_quality = grasp.get("axis_quality")
            quality_text = "-" if axis_quality is None else f"{float(axis_quality):.2f}"
            cv2.putText(
                image,
                f"G({grasp['x_mm']:.1f},{grasp['y_mm']:.1f},{grasp['z_mm']:.1f}) yaw={grasp['yaw_deg']:.1f} {axis_state}/{quality_text}",
                (max(8, gu - 180), max(18, gv - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
