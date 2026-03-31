from __future__ import annotations

import copy
import logging
import re
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from .calibration import CalibrationManager
from .config import AppConfig
from .geometry import (
    axis_dir_to_yaw_deg,
    extract_support_region,
    fit_ground_plane_conservative_fast,
    median_depth_at,
    project_xyz_to_uv,
)
from .geometry_backend import CPUReferenceBackend, GeometryBackend, build_geometry_backend
from .receiver import ZmqFrameReceiver
from .state import SharedState
from .types import FramePacket


class VisionProcessor:
    def __init__(
        self,
        config: AppConfig,
        state: SharedState,
        calibration_manager: CalibrationManager | None = None,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._cfg = config
        self._state = state
        self._calibration_manager = calibration_manager
        self._receiver = ZmqFrameReceiver(
            endpoint=config.zmq_endpoint,
            topic=config.zmq_topic,
            timeout_ms=config.zmq_timeout_ms,
        )
        try:
            from ultralytics.models.yolo.model import YOLO as UltralyticsYOLO
        except Exception as exc:
            raise RuntimeError("failed to import ultralytics YOLO runtime") from exc
        try:
            self._model = UltralyticsYOLO(str(config.model_path), task="detect")
        except TypeError:
            self._model = UltralyticsYOLO(str(config.model_path))
        self._is_engine_model = config.model_path.suffix.lower() == ".engine"
        self._predict_supports_half = not self._is_engine_model
        self._infer_imgsz = int(config.yolo_imgsz)
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
        self._grasp_tracker: dict[str, Any] | None = None
        self._target_tracker: dict[str, Any] | None = None
        self._support_source_tracker: dict[str, Any] | None = None
        self._cpu_geometry_backend: GeometryBackend = CPUReferenceBackend()
        self._geometry_backend, selection = build_geometry_backend(
            requested_backend=self._cfg.geom_backend,
            yolo_device=self._cfg.yolo_device,
        )
        self._geometry_force_cpu = False
        self._geometry_fallback_count = 0
        self._geometry_parity_mismatch_count = 0
        self._process_metrics_cache: dict[str, int | float | None] = {
            "process_rss_mb": None,
            "process_rss_peak_mb": None,
            "process_threads": None,
        }
        self._process_metrics_cache_ts = 0.0
        self._window_metrics_history: deque[dict[str, Any]] = deque(maxlen=max(10, int(self._cfg.metrics_window_size)))

        self._enable_cuda_benchmark()
        self._warmup_yolo_if_needed()
        self._logger.info(
            "geometry backend requested=%s resolved=%s torch_cuda=%s cuml=%s",
            selection.requested,
            selection.resolved,
            selection.torch_cuda_available,
            selection.cuml_available,
        )

    def _read_process_metrics(self) -> dict[str, int | float | None]:
        if not hasattr(self, "_process_metrics_cache"):
            self._process_metrics_cache = {
                "process_rss_mb": None,
                "process_rss_peak_mb": None,
                "process_threads": None,
            }
            self._process_metrics_cache_ts = 0.0

        now = time.time()
        if (now - self._process_metrics_cache_ts) < 0.5:
            return dict(self._process_metrics_cache)

        rss_mb: float | None = None
        rss_peak_mb: float | None = None
        threads: int | None = None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.startswith("VmRSS:"):
                        rss_mb = round(float(int(line.split()[1])) / 1024.0, 2)
                    elif line.startswith("VmHWM:"):
                        rss_peak_mb = round(float(int(line.split()[1])) / 1024.0, 2)
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
        except Exception:
            pass

        self._process_metrics_cache = {
            "process_rss_mb": rss_mb,
            "process_rss_peak_mb": rss_peak_mb,
            "process_threads": threads,
        }
        self._process_metrics_cache_ts = now
        return dict(self._process_metrics_cache)

    def _make_timing_payload(
        self,
        meta: dict[str, Any],
        fps: float,
        infer_ran: bool,
        infer_ms: float | None,
    ) -> dict[str, Any]:
        timing = {
            "frame_id": int(meta.get("frame_id", 0)),
            "ts_us": int(meta.get("ts_us", 0)),
            "fps": round(float(fps), 2),
            "infer_ran": bool(infer_ran),
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
            "geometry_ground_fit_mode": None,
            "geometry_points_input": None,
            "geometry_points_sampled": None,
            "geometry_ground_inlier_count": None,
            "geometry_ground_inlier_ratio": None,
        }
        timing.update(self._read_process_metrics())
        return timing

    def _pipeline_mode(self) -> str:
        raw_mode = getattr(self._cfg, "vision_pipeline", "v1")
        mode = str(raw_mode).strip().lower()
        return mode if mode in {"v1", "v2"} else "v1"

    def _metrics_slow_frame_threshold_ms(self) -> float:
        raw_threshold = getattr(self._cfg, "metrics_slow_frame_ms", 500.0)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = 500.0
        return max(10.0, threshold)

    def _shadow_compare_enabled(self) -> bool:
        raw_enabled = getattr(self._cfg, "shadow_compare", False)
        if isinstance(raw_enabled, bool):
            return raw_enabled
        return str(raw_enabled).strip().lower() in {"1", "true", "yes", "on"}

    def _window_metrics_rows(self) -> deque[dict[str, Any]]:
        history = getattr(self, "_window_metrics_history", None)
        if isinstance(history, deque):
            return history
        raw_size = getattr(self._cfg, "metrics_window_size", 120)
        try:
            maxlen = max(10, int(raw_size))
        except (TypeError, ValueError):
            maxlen = 120
        history = deque(maxlen=maxlen)
        self._window_metrics_history = history
        return history

    def _record_window_metrics(self, *, status: str, total_ms: float | None) -> dict[str, Any]:
        normalized_status = str(status).strip().lower() or "unknown"
        row = {
            "status": normalized_status,
            "is_loss": normalized_status in {"lost"},
            "is_invalid_depth": normalized_status in {"invalid_depth", "unstable"},
            "total_ms": None if total_ms is None else float(total_ms),
        }
        history = self._window_metrics_rows()
        history.append(row)

        rows = list(history)
        count = len(rows)
        if count <= 0:
            return {
                "samples": 0,
                "loss_rate": None,
                "invalid_depth_rate": None,
                "slow_frame_ratio": None,
                "latency_p95_ms": None,
            }

        loss_count = sum(1 for item in rows if bool(item.get("is_loss")))
        invalid_depth_count = sum(1 for item in rows if bool(item.get("is_invalid_depth")))
        latencies = [float(item["total_ms"]) for item in rows if item.get("total_ms") is not None]
        slow_threshold = self._metrics_slow_frame_threshold_ms()
        slow_count = sum(1 for value in latencies if float(value) >= slow_threshold)
        latency_p95_ms = None
        if latencies:
            latency_p95_ms = float(np.percentile(np.asarray(latencies, dtype=np.float32), 95))

        return {
            "samples": int(count),
            "loss_rate": round(float(loss_count / count), 4),
            "invalid_depth_rate": round(float(invalid_depth_count / count), 4),
            "slow_frame_ratio": None if not latencies else round(float(slow_count / len(latencies)), 4),
            "latency_p95_ms": None if latency_p95_ms is None else round(float(latency_p95_ms), 2),
        }

    @staticmethod
    def _normalize_status_v2(raw_status: str, tracking_state: str | None) -> str:
        status = str(raw_status).strip().lower()
        if status == "ok":
            return "ok"
        if status in {"invalid_depth"}:
            return "invalid_depth"
        if status in {"too_wide"}:
            return "too_wide"
        if status in {"calibration_mode"}:
            return "calibration_mode"
        if status in {"no_target"}:
            if str(tracking_state or "").strip().lower() in {"acquire"}:
                return "unstable"
            return "lost"
        if status in {"unstable", "lost"}:
            return status
        return "unstable"

    def _to_v2_result(self, result: dict[str, Any]) -> dict[str, Any]:
        target_raw = result.get("target")
        target = target_raw if isinstance(target_raw, dict) else {}
        tracking_state = None if target.get("tracker_state") is None else str(target.get("tracker_state"))
        normalized_status = self._normalize_status_v2(str(result.get("status", "unknown")), tracking_state)

        quality_score = None
        quality_flags: list[str] = []
        segmentation = result.get("segmentation")
        if isinstance(segmentation, dict):
            raw_score = segmentation.get("quality_score")
            if raw_score is not None:
                try:
                    quality_score = float(raw_score)
                except (TypeError, ValueError):
                    quality_score = None
            raw_flags = segmentation.get("quality_flags")
            if isinstance(raw_flags, list):
                quality_flags = [str(flag) for flag in raw_flags if str(flag)]

        timing = result.get("timing")
        timing_map = timing if isinstance(timing, dict) else {}
        infer_ms = timing_map.get("infer_ms")
        if infer_ms is not None:
            try:
                infer_ms = float(infer_ms)
            except (TypeError, ValueError):
                infer_ms = None
        geometry_total_ms = timing_map.get("geometry_total_ms")
        if geometry_total_ms is not None:
            try:
                geometry_total_ms = float(geometry_total_ms)
            except (TypeError, ValueError):
                geometry_total_ms = None
        total_ms = None
        if infer_ms is not None or geometry_total_ms is not None:
            total_ms = float((infer_ms or 0.0) + (geometry_total_ms or 0.0))

        window_metrics = self._record_window_metrics(status=normalized_status, total_ms=total_ms)
        metrics = {
            "frame": {
                "total_ms": None if total_ms is None else round(float(total_ms), 2),
                "infer_ms": None if infer_ms is None else round(float(infer_ms), 2),
                "geometry_total_ms": None if geometry_total_ms is None else round(float(geometry_total_ms), 2),
                "slow_frame": None if total_ms is None else bool(float(total_ms) >= self._metrics_slow_frame_threshold_ms()),
            },
            "window": window_metrics,
        }

        tracking = {
            "state": tracking_state,
            "confidence": None if target.get("tracker_score") is None else float(target.get("tracker_score")),
            "reason": None if target.get("rejected_reason") is None else str(target.get("rejected_reason")),
        }
        if normalized_status == "lost" and tracking["state"] is None:
            tracking["state"] = "lost"

        return {
            "schema_version": 2,
            "status": normalized_status,
            "target": {
                "class_id": target.get("class_id"),
                "class_name": target.get("class_name"),
                "conf": target.get("conf"),
                "bbox_xyxy": target.get("bbox_xyxy"),
            },
            "tracking": tracking,
            "depth": result.get("depth"),
            "size": result.get("size"),
            "grasp": result.get("grasp"),
            "segmentation": result.get("segmentation"),
            "quality": {
                "score": None if quality_score is None else round(float(quality_score), 3),
                "flags": quality_flags,
            },
            "metrics": metrics,
            "timing": timing_map,
        }

    @staticmethod
    def _build_shadow_compare_summary(
        *,
        primary_pipeline: str,
        v1_result: dict[str, Any],
        v2_result: dict[str, Any],
    ) -> dict[str, Any]:
        primary_mode = str(primary_pipeline).strip().lower()
        if primary_mode not in {"v1", "v2"}:
            primary_mode = "v1"
        v1_status = str(v1_result.get("status", "unknown")).strip().lower() or "unknown"
        v2_status = str(v2_result.get("status", "unknown")).strip().lower() or "unknown"
        tracking_v2 = v2_result.get("tracking")
        quality_v2 = v2_result.get("quality")
        metrics_v2 = v2_result.get("metrics")
        return {
            "enabled": True,
            "primary_pipeline": primary_mode,
            "v1": {
                "status": v1_status,
            },
            "v2": {
                "status": v2_status,
                "tracking": tracking_v2 if isinstance(tracking_v2, dict) else None,
                "quality": quality_v2 if isinstance(quality_v2, dict) else None,
                "metrics": metrics_v2 if isinstance(metrics_v2, dict) else None,
            },
            "status_match": bool(v1_status == v2_status),
        }

    def _finalize_packet_output(self, result: dict[str, Any], annotated: np.ndarray) -> tuple[dict[str, Any], bytes | None]:
        encoded = self._encode_annotated(annotated)
        pipeline_mode = self._pipeline_mode()
        shadow_enabled = self._shadow_compare_enabled()
        if pipeline_mode == "v2":
            v2_result = self._to_v2_result(result)
            if shadow_enabled:
                v2_result["shadow_compare"] = self._build_shadow_compare_summary(
                    primary_pipeline="v2",
                    v1_result=result,
                    v2_result=v2_result,
                )
            return v2_result, encoded

        if shadow_enabled:
            v2_shadow = self._to_v2_result(result)
            result_with_shadow = dict(result)
            result_with_shadow["shadow_compare"] = self._build_shadow_compare_summary(
                primary_pipeline="v1",
                v1_result=result,
                v2_result=v2_shadow,
            )
            return result_with_shadow, encoded
        return result, encoded

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
        warmup_size = max(64, min(1280, int(self._infer_imgsz)))
        warmup_frame = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
        base_kwargs: dict[str, Any] = {
            "source": warmup_frame,
            "conf": self._cfg.yolo_conf,
            "imgsz": self._infer_imgsz,
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

    def _adapt_engine_imgsz_from_assertion(self, exc: AssertionError) -> bool:
        if not self._is_engine_model:
            return False
        match = re.search(r"max model size \(\d+,\s*\d+,\s*(\d+),\s*(\d+)\)", str(exc))
        if not match:
            return False
        expected_h = int(match.group(1))
        expected_w = int(match.group(2))
        if expected_h != expected_w:
            return False
        expected = max(64, min(2048, expected_h))
        if expected == self._infer_imgsz:
            return False
        self._logger.warning(
            "YOLO TensorRT imgsz mismatch detected (requested=%s, expected=%s), auto switching.",
            self._infer_imgsz,
            expected,
        )
        self._infer_imgsz = expected
        return True

    def _predict_once(self, predict_kwargs: dict[str, Any]) -> Any:
        if self._predict_supports_half:
            predict_kwargs_with_half = dict(predict_kwargs)
            predict_kwargs_with_half["half"] = self._use_fp16_infer()
            try:
                return self._model.predict(**predict_kwargs_with_half)[0]
            except TypeError:
                self._predict_supports_half = False
        return self._model.predict(**predict_kwargs)[0]

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
        depth_h, depth_w = depth.shape[:2]

        calibration_mode_active = False
        if self._calibration_manager is not None:
            try:
                calibration_mode_active = self._calibration_manager.is_mode_active()
                if calibration_mode_active:
                    self._calibration_manager.ingest_frame(rgb=rgb, meta=meta)
            except Exception as exc:
                self._logger.debug("calibration manager mode/ingest failed: %s", exc)

        fx = float(meta.get("fx", 0.0))
        fy = float(meta.get("fy", 0.0))
        cx = float(meta.get("cx", w / 2.0))
        cy = float(meta.get("cy", h / 2.0))
        depth_scale = float(meta.get("depth_scale", 1.0))
        if fx <= 1.0:
            fx = float(w)
        if fy <= 1.0:
            fy = float(h)

        frame_center_depth_u, frame_center_depth_v = self._map_uv_between_frames(
            u=w // 2,
            v=h // 2,
            src_width=w,
            src_height=h,
            dst_width=depth_w,
            dst_height=depth_h,
        )
        center_depth_mm = median_depth_at(
            depth=depth,
            u=frame_center_depth_u,
            v=frame_center_depth_v,
            depth_scale=depth_scale,
            window=self._cfg.center_depth_window,
            min_depth_mm=self._cfg.min_depth_mm,
            max_depth_mm=self._cfg.max_depth_mm,
        )

        if calibration_mode_active:
            self._cached_detections = []
            self._clear_geometry_cache()
            self._clear_axis_tracker()
            self._clear_grasp_tracker()
            self._clear_target_tracker()
            self._clear_support_source_tracker()

            annotated = rgb.copy()
            cv2.putText(
                annotated,
                "Calibration Mode: YOLO/Geometry Paused",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
            calibration_result = {
                "status": "calibration_mode",
                "target": None,
                "depth": {"center_depth_mm": center_depth_mm},
                "size": None,
                "grasp": None,
                "segmentation": None,
                "timing": self._make_timing_payload(meta=meta, fps=fps, infer_ran=False, infer_ms=None),
            }
            return self._finalize_packet_output(calibration_result, annotated)

        self._frame_index += 1
        run_infer = (self._frame_index % self._cfg.infer_every_n == 0) or (not self._cached_detections)
        if run_infer:
            infer_start = time.perf_counter()
            self._cached_detections = self._infer(rgb)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
        else:
            infer_ms = None
        detections = self._cached_detections
        selected, tracker_state, tracker_score, rejected_reason = self._select_target(
            detections=detections,
            width=w,
            height=h,
            depth=depth,
            depth_scale=depth_scale,
            depth_width=depth_w,
            depth_height=depth_h,
        )

        result: dict[str, Any] = {
            "status": "no_target",
            "target": None,
            "depth": {"center_depth_mm": center_depth_mm},
            "size": None,
            "grasp": None,
            "segmentation": None,
            "timing": self._make_timing_payload(meta=meta, fps=fps, infer_ran=bool(run_infer), infer_ms=infer_ms),
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
            self._clear_grasp_tracker()
            self._clear_support_source_tracker()
            if tracker_state is not None or rejected_reason is not None:
                result["target"] = {
                    "class_id": None,
                    "class_name": None,
                    "conf": None,
                    "bbox_xyxy": None,
                    "tracker_state": tracker_state,
                    "tracker_score": None if tracker_score is None else round(float(tracker_score), 3),
                    "rejected_reason": rejected_reason,
                }
            label = "No target" if not rejected_reason else f"No target ({rejected_reason})"
            cv2.putText(annotated, label, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 255), 2, cv2.LINE_AA)
            return self._finalize_packet_output(result, annotated)

        bbox = selected["bbox_xyxy"]
        class_id = int(selected["class_id"])
        class_name = selected["class_name"]
        conf = float(selected["conf"])
        u_center = int((bbox[0] + bbox[2]) / 2)
        v_center = int((bbox[1] + bbox[3]) / 2)
        u_center_depth, v_center_depth = self._map_uv_between_frames(
            u=u_center,
            v=v_center,
            src_width=w,
            src_height=h,
            dst_width=depth_w,
            dst_height=depth_h,
        )
        target_center_depth_mm = selected.get("center_depth_mm")
        if target_center_depth_mm is None:
            target_center_depth_mm = median_depth_at(
                depth=depth,
                u=u_center_depth,
                v=v_center_depth,
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
            "tracker_state": tracker_state,
            "tracker_score": None if tracker_score is None else round(float(tracker_score), 3),
            "rejected_reason": rejected_reason,
        }
        result["depth"]["target_center_depth_mm"] = target_center_depth_mm

        run_geometry = self._should_run_geometry(class_id=class_id, bbox=bbox)
        if (not run_geometry) and self._cached_geometry_payload is None:
            run_geometry = True
        result["timing"]["geometry_ran"] = bool(run_geometry)
        result["timing"]["geometry_reused"] = bool(not run_geometry)

        if run_geometry:
            depth_bbox = list(
                self._map_bbox_between_frames(
                    bbox=bbox,
                    src_width=w,
                    src_height=h,
                    dst_width=depth_w,
                    dst_height=depth_h,
                )
            )
            geometry_payload, geometry_timing, backend_used = self._run_geometry_with_fallback(
                depth=depth,
                bbox=depth_bbox,
                class_id=class_id,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ref_uv=(u_center_depth, v_center_depth),
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
            result["timing"]["geometry_ground_fit_mode"] = geometry_timing.get("ground_fit_mode")
            result["timing"]["geometry_points_input"] = geometry_timing.get("points_input")
            result["timing"]["geometry_points_sampled"] = geometry_timing.get("points_sampled")
            result["timing"]["geometry_ground_inlier_count"] = geometry_timing.get("ground_inlier_count")
            ground_inlier_ratio = geometry_timing.get("ground_inlier_ratio")
            result["timing"]["geometry_ground_inlier_ratio"] = (
                None if ground_inlier_ratio is None else round(float(ground_inlier_ratio), 4)
            )
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
        return self._finalize_packet_output(result, annotated)

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
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        backend = self._runtime_geometry_backend()
        backend_name = backend.name
        if self._geometry_force_cpu and backend_name == "cpu" and self._geometry_backend.name != "cpu":
            backend_name = "cpu_forced"

        axis_tracker_before = copy.deepcopy(self._axis_tracker)
        grasp_tracker_before = copy.deepcopy(self._grasp_tracker)
        support_tracker_before = copy.deepcopy(self._support_source_tracker)
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
            self._grasp_tracker = grasp_tracker_before
            self._support_source_tracker = support_tracker_before
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
        grasp_tracker_after_primary = copy.deepcopy(self._grasp_tracker)
        support_tracker_after_primary = copy.deepcopy(self._support_source_tracker)
        try:
            self._axis_tracker = copy.deepcopy(axis_tracker_before)
            self._grasp_tracker = copy.deepcopy(grasp_tracker_before)
            self._support_source_tracker = copy.deepcopy(support_tracker_before)
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
            grasp_tracker_after_cpu = copy.deepcopy(self._grasp_tracker)
            support_tracker_after_cpu = copy.deepcopy(self._support_source_tracker)
        except Exception as exc:
            self._axis_tracker = axis_tracker_after_primary
            self._grasp_tracker = grasp_tracker_after_primary
            self._support_source_tracker = support_tracker_after_primary
            self._logger.warning("geometry parity CPU check failed: %s", exc)
            return payload, timing, backend_name

        parity_ok, reason = self._geometry_payload_within_threshold(payload, cpu_payload)
        if parity_ok:
            self._axis_tracker = axis_tracker_after_primary
            self._grasp_tracker = grasp_tracker_after_primary
            self._support_source_tracker = support_tracker_after_primary
            return payload, timing, backend_name

        self._geometry_parity_mismatch_count += 1
        self._geometry_fallback_count += 1
        self._geometry_force_cpu = True
        self._axis_tracker = axis_tracker_after_cpu
        self._grasp_tracker = grasp_tracker_after_cpu
        self._support_source_tracker = support_tracker_after_cpu
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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        overall_start = time.perf_counter()
        timings: dict[str, Any] = {
            "depth_to_points_ms": 0.0,
            "ground_fit_ms": 0.0,
            "support_ms": 0.0,
            "cluster_ms": 0.0,
            "pca_ms": 0.0,
            "grasp_ms": 0.0,
            "total_ms": 0.0,
        }
        timings["ground_fit_mode"] = "full"
        timings["points_input"] = 0
        timings["points_sampled"] = 0
        timings["ground_inlier_count"] = 0
        timings["ground_inlier_ratio"] = 0.0

        def finish(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            timings["total_ms"] = (time.perf_counter() - overall_start) * 1000.0
            return payload, timings

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(bbox, w, h)
        bbox_area_px = max(1, (x2 - x1) * (y2 - y1))
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

        raw_input_points = int(points.shape[0])
        timings["points_input"] = raw_input_points
        depth_valid_ratio = float(raw_input_points / max(1, bbox_area_px))
        segmentation: dict[str, Any] = {
            "input_points": raw_input_points,
            "ground_points": 0,
            "non_ground_points": raw_input_points,
            "main_cluster_points": 0,
            "depth_valid_ratio": round(depth_valid_ratio, 4),
            "ground_ratio": None,
            "support_points": 0,
            "support_area_px": None,
            "support_density": None,
            "support_fill_ratio": None,
            "axis_eig_ratio": None,
            "support_source": None,
            "support_switch_count": 0,
            "quality_score": None,
            "quality_ok": False,
            "quality_flags": [],
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
            segmentation["non_ground_points"] = int(points.shape[0])
            segmentation["sampled_points"] = int(points.shape[0])
        timings["points_sampled"] = int(points.shape[0])

        t = time.perf_counter()
        ground_plane, ground_fit_mode = fit_ground_plane_conservative_fast(
            points,
            residual_mm=self._cfg.ransac_residual_mm,
            max_trials=self._cfg.ransac_max_trials,
            fast_enabled=self._cfg.ground_fit_fast_enabled,
            fast_sample_cap=self._cfg.ground_fit_fast_sample_cap,
            fast_max_trials=self._cfg.ground_fit_fast_max_trials,
            fast_min_inlier_ratio=self._cfg.ground_fit_fast_min_inlier_ratio,
            fast_min_inliers=self._cfg.ground_fit_fast_min_inliers,
        )
        timings["ground_fit_ms"] = (time.perf_counter() - t) * 1000.0
        timings["ground_fit_mode"] = str(ground_fit_mode)

        ground_mask = np.zeros(points.shape[0], dtype=bool) if ground_plane is None else ground_plane.inlier_mask.astype(bool)
        ground_count = int(np.count_nonzero(ground_mask))
        non_ground_count = int(points.shape[0] - ground_count)
        timings["ground_inlier_count"] = ground_count
        timings["ground_inlier_ratio"] = float(ground_count / max(1, points.shape[0]))
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
        support_region_data: dict[str, Any] | None = None
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
                support_region_data = {
                    "points": support_region["selected_points"],
                    "uv": support_region["selected_uv"],
                    "stats": support_region,
                }

        seed_region_data: dict[str, Any] | None = None
        dbscan_region_data: dict[str, Any] | None = None
        t = time.perf_counter()
        if self._pipeline_mode() == "v2":
            seed_region_data = self._build_seed_region_source(
                points=object_points,
                uv=object_uv,
                bbox_xyxy=(x1, y1, x2, y2),
                ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
            )
            if bool(self._cfg.segmentation_debug_dbscan):
                cluster_mask = backend.select_main_cluster(
                    points=object_points,
                    uv=object_uv,
                    ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
                    eps_mm=self._cfg.dbscan_eps_mm,
                    min_samples=self._cfg.dbscan_min_samples,
                )
                dbscan_points = object_points[cluster_mask]
                dbscan_uv = object_uv[cluster_mask]
                dbscan_region_data = {
                    "points": dbscan_points,
                    "uv": dbscan_uv,
                    "stats": {
                        "area_px": int(dbscan_points.shape[0]),
                        "density": 1.0,
                        "fill_ratio": float(dbscan_points.shape[0] / max(1, (x2 - x1) * (y2 - y1))),
                        "elevated_points": int(dbscan_points.shape[0]),
                    },
                }
        else:
            cluster_mask = backend.select_main_cluster(
                points=object_points,
                uv=object_uv,
                ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
                eps_mm=self._cfg.dbscan_eps_mm,
                min_samples=self._cfg.dbscan_min_samples,
            )
            cluster_points = object_points[cluster_mask]
            cluster_uv = object_uv[cluster_mask]
            dbscan_region_data = {
                "points": cluster_points,
                "uv": cluster_uv,
                "stats": {
                    "area_px": int(cluster_points.shape[0]),
                    "density": 1.0,
                    "fill_ratio": float(cluster_points.shape[0] / max(1, (x2 - x1) * (y2 - y1))),
                    "elevated_points": int(cluster_points.shape[0]),
                },
            }
        timings["cluster_ms"] = (time.perf_counter() - t) * 1000.0

        source_options: dict[str, dict[str, Any] | None] = {
            "support_region": support_region_data,
            "seed_region": seed_region_data,
            "dbscan_debug": dbscan_region_data,
        }
        if support_region_data is not None:
            candidate_source = "support_region"
        elif seed_region_data is not None:
            candidate_source = "seed_region"
        else:
            candidate_source = "dbscan_debug"
        support_source, support_switch_count = self._stabilize_support_source(
            candidate_source=candidate_source,
            source_options=source_options,
        )
        selected_source = source_options.get(support_source) or source_options.get(candidate_source)
        if selected_source is None:
            selected_source = source_options.get("seed_region") or source_options.get("dbscan_debug")
        if selected_source is None:
            selected_source = {
                "points": object_points,
                "uv": object_uv,
                "stats": {
                    "area_px": int(object_points.shape[0]),
                    "density": 1.0,
                    "fill_ratio": float(object_points.shape[0] / max(1, (x2 - x1) * (y2 - y1))),
                    "elevated_points": int(object_points.shape[0]),
                },
            }
        assert selected_source is not None
        main_points = selected_source["points"]
        main_uv = selected_source["uv"]
        support_stats = selected_source["stats"]

        segmentation["main_cluster_points"] = int(main_points.shape[0])
        segmentation["support_points"] = int(main_points.shape[0])
        segmentation["support_area_px"] = int(support_stats.get("area_px", 0))
        segmentation["support_density"] = round(float(support_stats.get("density", 0.0)), 4)
        segmentation["support_fill_ratio"] = round(float(support_stats.get("fill_ratio", 0.0)), 4)
        segmentation["support_source"] = support_source
        segmentation["support_switch_count"] = int(support_switch_count)
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

        pre_quality_ok, pre_quality_score, pre_quality_flags = self._evaluate_geometry_quality(
            segmentation=segmentation,
            include_axis=False,
        )
        if not pre_quality_ok:
            segmentation["quality_ok"] = False
            segmentation["quality_score"] = round(float(pre_quality_score), 3)
            segmentation["quality_flags"] = pre_quality_flags
            return finish(
                self._make_geometry_payload(
                    status="unstable",
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

        quality_ok, quality_score, quality_flags = self._evaluate_geometry_quality(segmentation=segmentation, include_axis=True)
        segmentation["quality_ok"] = bool(quality_ok)
        segmentation["quality_score"] = round(float(quality_score), 3)
        segmentation["quality_flags"] = quality_flags
        if not quality_ok:
            return finish(
                self._make_geometry_payload(
                    status="unstable",
                    size=size,
                    grasp=None,
                    segmentation=segmentation,
                    visualization=visualization,
                )
            )

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

        raw_grasp_xyz = np.asarray(grasp["grasp_xyz_mm"], dtype=np.float32)
        raw_yaw_deg = float(axis_dir_to_yaw_deg(stable_axis_dir_cam))
        stable_grasp_xyz, stable_grasp_yaw_deg, grasp_stability_state, grasp_stability_score = self._stabilize_grasp(
            raw_grasp_xyz_mm=raw_grasp_xyz,
            raw_yaw_deg=raw_yaw_deg,
            class_id=int(class_id),
            bbox=[int(v) for v in bbox],
            axis_quality=float(axis_quality_out),
        )
        grasp_xyz = raw_grasp_xyz if stable_grasp_xyz is None else stable_grasp_xyz
        grasp_yaw_deg = raw_yaw_deg if stable_grasp_yaw_deg is None else stable_grasp_yaw_deg
        grasp_uv = project_xyz_to_uv(grasp_xyz, fx=fx, fy=fy, cx=cx, cy=cy)
        grasp_uv_int = None if grasp_uv is None else [int(grasp_uv[0]), int(grasp_uv[1])]
        grasp_result = {
            "x_mm": round(float(grasp_xyz[0]), 2),
            "y_mm": round(float(grasp_xyz[1]), 2),
            "z_mm": round(float(grasp_xyz[2]), 2),
            "raw_x": round(float(raw_grasp_xyz[0]), 2),
            "raw_y": round(float(raw_grasp_xyz[1]), 2),
            "raw_z": round(float(raw_grasp_xyz[2]), 2),
            "u": grasp_uv_int[0] if grasp_uv_int else None,
            "v": grasp_uv_int[1] if grasp_uv_int else None,
            "yaw_deg": round(float(grasp_yaw_deg), 2),
            "raw_yaw": round(float(raw_yaw_deg), 2),
            "axis_dir_cam": [round(float(v), 4) for v in stable_axis_dir_cam.tolist()],
            "axis_quality": round(float(axis_quality_out), 3),
            "axis_state": axis_state,
            "stability_state": grasp_stability_state,
            "stability_score": round(float(grasp_stability_score), 3),
            "source_quality_score": round(float(quality_score), 3),
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

    def _clear_grasp_tracker(self) -> None:
        self._grasp_tracker = None

    def _clear_target_tracker(self) -> None:
        self._target_tracker = None

    def _clear_support_source_tracker(self) -> None:
        self._support_source_tracker = None

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)

    @classmethod
    def _blend_angle_deg(cls, start_deg: float, target_deg: float, alpha: float) -> float:
        delta = cls._normalize_angle_deg(float(target_deg) - float(start_deg))
        return cls._normalize_angle_deg(float(start_deg) + max(0.0, min(1.0, float(alpha))) * delta)

    def _stabilize_grasp(
        self,
        raw_grasp_xyz_mm: np.ndarray | None,
        raw_yaw_deg: float | None,
        class_id: int,
        bbox: list[int],
        axis_quality: float,
    ) -> tuple[np.ndarray | None, float | None, str, float]:
        raw_xyz = None if raw_grasp_xyz_mm is None else np.asarray(raw_grasp_xyz_mm, dtype=np.float32)
        raw_yaw = None if raw_yaw_deg is None else self._normalize_angle_deg(raw_yaw_deg)
        tracker = self._grasp_tracker

        continuity = False
        prev_xyz = None if tracker is None else tracker.get("accepted_xyz_mm")
        prev_yaw = None if tracker is None else tracker.get("accepted_yaw_deg")
        if (
            tracker is not None
            and prev_xyz is not None
            and prev_yaw is not None
            and tracker.get("bbox") is not None
        ):
            continuity = self._bbox_iou(tracker["bbox"], bbox) >= self._cfg.geometry_force_recalc_iou

        if raw_xyz is None or raw_yaw is None:
            if continuity and prev_xyz is not None and prev_yaw is not None:
                bad_streak = int(tracker.get("bad_streak", 0)) + 1
                tracker["bbox"] = [int(v) for v in bbox]
                tracker["bad_streak"] = bad_streak
                if bad_streak <= int(self._cfg.grasp_hold_frames):
                    held_score = max(0.05, min(1.0, float(tracker.get("accepted_quality", axis_quality)) * 0.85))
                    return np.asarray(prev_xyz, dtype=np.float32), float(prev_yaw), "held", float(held_score)
            return raw_xyz, raw_yaw, "unreliable", max(0.05, min(1.0, float(axis_quality) * 0.6))

        if not continuity or prev_xyz is None or prev_yaw is None:
            self._grasp_tracker = {
                "class_id": int(class_id),
                "bbox": [int(v) for v in bbox],
                "accepted_xyz_mm": raw_xyz.astype(np.float32),
                "accepted_yaw_deg": float(raw_yaw),
                "accepted_quality": float(axis_quality),
                "bad_streak": 0,
            }
            score = max(0.05, min(1.0, 0.45 + float(axis_quality) * 0.55))
            return raw_xyz, float(raw_yaw), "live", float(score)

        prev_xyz_arr = np.asarray(prev_xyz, dtype=np.float32)
        prev_yaw_f = float(prev_yaw)
        jump_xy = float(np.linalg.norm(raw_xyz[:2] - prev_xyz_arr[:2]))
        jump_z = abs(float(raw_xyz[2]) - float(prev_xyz_arr[2]))
        jump_yaw = self._yaw_delta_deg(float(raw_yaw), prev_yaw_f)
        is_jump = (
            jump_xy > float(self._cfg.grasp_jump_xy_mm)
            or jump_z > float(self._cfg.grasp_jump_z_mm)
            or jump_yaw > float(self._cfg.grasp_jump_yaw_deg)
        )
        if is_jump:
            bad_streak = int(tracker.get("bad_streak", 0)) + 1
            tracker["bbox"] = [int(v) for v in bbox]
            tracker["bad_streak"] = bad_streak
            if bad_streak <= int(self._cfg.grasp_hold_frames):
                held_score = max(0.05, min(1.0, float(tracker.get("accepted_quality", axis_quality)) * 0.85))
                return prev_xyz_arr, prev_yaw_f, "held", float(held_score)

            # Hold window exhausted; re-acquire around the new measurement.
            self._grasp_tracker = {
                "class_id": int(class_id),
                "bbox": [int(v) for v in bbox],
                "accepted_xyz_mm": raw_xyz.astype(np.float32),
                "accepted_yaw_deg": float(raw_yaw),
                "accepted_quality": float(axis_quality) * 0.8,
                "bad_streak": 0,
            }
            score = max(0.05, min(1.0, float(axis_quality) * 0.5))
            return raw_xyz, float(raw_yaw), "jump_reacquire", float(score)

        point_alpha = float(self._cfg.grasp_point_smooth_alpha)
        yaw_alpha = float(self._cfg.grasp_yaw_smooth_alpha)
        smoothed_xyz = ((1.0 - point_alpha) * prev_xyz_arr) + (point_alpha * raw_xyz)
        smoothed_yaw = self._blend_angle_deg(prev_yaw_f, float(raw_yaw), yaw_alpha)

        self._grasp_tracker = {
            "class_id": int(class_id),
            "bbox": [int(v) for v in bbox],
            "accepted_xyz_mm": smoothed_xyz.astype(np.float32),
            "accepted_yaw_deg": float(smoothed_yaw),
            "accepted_quality": float(axis_quality),
            "bad_streak": 0,
        }

        xy_score = 1.0 - min(1.0, jump_xy / max(1.0, float(self._cfg.grasp_jump_xy_mm)))
        z_score = 1.0 - min(1.0, jump_z / max(1.0, float(self._cfg.grasp_jump_z_mm)))
        yaw_score = 1.0 - min(1.0, jump_yaw / max(1.0, float(self._cfg.grasp_jump_yaw_deg)))
        score = max(0.05, min(1.0, 0.40 * float(axis_quality) + 0.30 * xy_score + 0.20 * z_score + 0.10 * yaw_score))
        return smoothed_xyz.astype(np.float32), float(smoothed_yaw), "live", float(score)

    def _score_axis_quality(self, axis_eig_ratio: float, point_count: int, density: float) -> float:
        eig_score = max(0.0, min(1.0, (float(axis_eig_ratio) - 1.0) / 2.0))
        point_score = max(0.0, min(1.0, float(point_count) / max(1.0, float(self._cfg.min_object_points * 2))))
        density_score = max(0.0, min(1.0, float(density) / 0.35))
        return max(0.0, min(1.0, 0.55 * eig_score + 0.25 * point_score + 0.20 * density_score))

    def _stabilize_support_source(
        self,
        candidate_source: str,
        source_options: dict[str, dict[str, Any] | None],
    ) -> tuple[str, int]:
        tracker = self._support_source_tracker
        if tracker is None:
            self._support_source_tracker = {
                "source": candidate_source,
                "pending_source": None,
                "pending_count": 0,
                "switch_count": 0,
            }
            return candidate_source, 0

        current_source = str(tracker.get("source", candidate_source))
        pending_source = tracker.get("pending_source")
        pending_count = int(tracker.get("pending_count", 0))
        switch_count = int(tracker.get("switch_count", 0))
        current_available = source_options.get(current_source) is not None
        candidate_available = source_options.get(candidate_source) is not None

        if current_source == candidate_source:
            pending_source = None
            pending_count = 0
        elif not current_available and candidate_available:
            current_source = candidate_source
            pending_source = None
            pending_count = 0
            switch_count += 1
        else:
            if pending_source != candidate_source:
                pending_source = candidate_source
                pending_count = 1
            else:
                pending_count += 1
            if pending_count >= int(self._cfg.support_switch_hold_frames):
                current_source = candidate_source
                pending_source = None
                pending_count = 0
                switch_count += 1

        self._support_source_tracker = {
            "source": current_source,
            "pending_source": pending_source,
            "pending_count": pending_count,
            "switch_count": switch_count,
        }
        return current_source, switch_count

    def _build_seed_region_source(
        self,
        points: np.ndarray,
        uv: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        ref_uv: tuple[float, float],
    ) -> dict[str, Any] | None:
        if points.shape[0] < int(self._cfg.min_object_points):
            return None

        x1, y1, x2, y2 = bbox_xyxy
        bbox_area = max(1, int((x2 - x1) * (y2 - y1)))
        ref_u, ref_v = float(ref_uv[0]), float(ref_uv[1])
        du = uv[:, 0] - ref_u
        dv = uv[:, 1] - ref_v
        dist2 = (du * du) + (dv * dv)

        min_r = max(1, int(self._cfg.seed_region_min_radius_px))
        max_r = max(min_r, int(self._cfg.seed_region_max_radius_px))
        step_r = max(1, int(self._cfg.seed_region_radius_step_px))
        best_mask: np.ndarray | None = None
        for radius in range(min_r, max_r + 1, step_r):
            radius2 = float(radius * radius)
            local_mask = dist2 <= radius2
            if int(np.count_nonzero(local_mask)) >= int(self._cfg.min_object_points):
                best_mask = local_mask
                break

        if best_mask is None:
            keep_n = min(points.shape[0], max(int(self._cfg.min_object_points), int(self._cfg.support_points_min)))
            if keep_n <= 0:
                return None
            order = np.argpartition(dist2, keep_n - 1)[:keep_n]
            best_mask = np.zeros(points.shape[0], dtype=bool)
            best_mask[order] = True

        selected_points = points[best_mask]
        selected_uv = uv[best_mask]
        if selected_points.shape[0] < int(self._cfg.min_object_points):
            return None

        u_vals = selected_uv[:, 0]
        v_vals = selected_uv[:, 1]
        u_min = int(np.floor(np.min(u_vals)))
        u_max = int(np.ceil(np.max(u_vals)))
        v_min = int(np.floor(np.min(v_vals)))
        v_max = int(np.ceil(np.max(v_vals)))
        area_px = max(1, (u_max - u_min + 1) * (v_max - v_min + 1))
        density = float(selected_points.shape[0] / max(1, area_px))
        fill_ratio = float(area_px / max(1, bbox_area))

        return {
            "points": selected_points,
            "uv": selected_uv,
            "stats": {
                "area_px": int(area_px),
                "density": float(density),
                "fill_ratio": float(fill_ratio),
                "elevated_points": int(selected_points.shape[0]),
            },
        }

    def _evaluate_geometry_quality(self, segmentation: dict[str, Any], *, include_axis: bool = True) -> tuple[bool, float, list[str]]:
        depth_valid_ratio = float(segmentation.get("depth_valid_ratio", 0.0) or 0.0)
        support_points = int(segmentation.get("support_points", 0) or 0)
        support_fill_ratio = float(segmentation.get("support_fill_ratio", 0.0) or 0.0)
        ground_ratio_raw = segmentation.get("ground_ratio")
        ground_ratio = 1.0 if ground_ratio_raw is None else float(ground_ratio_raw)
        axis_eig_ratio = float(segmentation.get("axis_eig_ratio", 0.0) or 0.0)

        quality_flags: list[str] = []
        if depth_valid_ratio < float(self._cfg.depth_valid_ratio_min):
            quality_flags.append("depth_valid_ratio")
        if support_points < int(self._cfg.support_points_min):
            quality_flags.append("support_points")
        if support_fill_ratio < float(self._cfg.support_fill_ratio_min):
            quality_flags.append("support_fill_ratio")
        if ground_ratio > float(self._cfg.ground_ratio_max):
            quality_flags.append("ground_ratio")
        if include_axis and axis_eig_ratio < float(self._cfg.axis_eig_ratio_min):
            quality_flags.append("axis_eig_ratio")

        depth_score = min(1.0, depth_valid_ratio / max(1e-6, float(self._cfg.depth_valid_ratio_min)))
        support_score = min(1.0, float(support_points) / max(1.0, float(self._cfg.support_points_min)))
        fill_score = min(1.0, support_fill_ratio / max(1e-6, float(self._cfg.support_fill_ratio_min)))
        ground_score = 1.0
        if ground_ratio > float(self._cfg.ground_ratio_max):
            ground_score = max(
                0.0,
                1.0
                - (ground_ratio - float(self._cfg.ground_ratio_max))
                / max(0.05, 1.0 - float(self._cfg.ground_ratio_max)),
            )
        axis_score = 1.0
        if include_axis:
            axis_score = min(
                1.0,
                max(
                    0.0,
                    (axis_eig_ratio - 1.0) / max(1e-6, float(self._cfg.axis_eig_ratio_min) - 1.0),
                ),
            )
        quality_score = max(
            0.0,
            min(1.0, 0.20 * depth_score + 0.20 * support_score + 0.20 * fill_score + 0.20 * ground_score + 0.20 * axis_score),
        )
        if quality_score < float(self._cfg.quality_score_min):
            quality_flags.append("quality_score")
        quality_ok = (not quality_flags) and (quality_score >= float(self._cfg.quality_score_min))
        return quality_ok, quality_score, quality_flags

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
            "imgsz": self._infer_imgsz,
            "device": self._cfg.yolo_device,
            "max_det": 20,
            "verbose": False,
        }
        try:
            pred = self._predict_once(predict_kwargs)
        except AssertionError as exc:
            if not self._adapt_engine_imgsz_from_assertion(exc):
                raise
            predict_kwargs["imgsz"] = self._infer_imgsz
            pred = self._predict_once(predict_kwargs)

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

    def _mark_target_miss(self, reason: str) -> tuple[None, str | None, float | None, str | None]:
        tracker = self._target_tracker
        if tracker is None:
            return None, None, None, reason
        lost_count = int(tracker.get("lost_count", 0)) + 1
        if lost_count > int(self._cfg.target_lost_hold_frames):
            self._clear_target_tracker()
            return None, "lost", None, reason
        tracker["lost_count"] = lost_count
        tracker["state"] = "lost"
        return None, "lost", float(tracker.get("score", 0.0)), reason

    def _select_target(
        self,
        detections: list[dict[str, Any]],
        width: int,
        height: int,
        depth: np.ndarray,
        depth_scale: float,
        depth_width: int | None = None,
        depth_height: int | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, float | None, str | None]:
        if not detections:
            return self._mark_target_miss(reason="no_detection")

        if depth_width is None or depth_width <= 0:
            depth_width = int(depth.shape[1])
        if depth_height is None or depth_height <= 0:
            depth_height = int(depth.shape[0])

        cx = width / 2.0
        cy = height / 2.0
        candidates: list[dict[str, Any]] = []
        for det in detections:
            bbox = [int(v) for v in det["bbox_xyxy"]]
            ux = float((bbox[0] + bbox[2]) / 2.0)
            vy = float((bbox[1] + bbox[3]) / 2.0)
            ux_depth, vy_depth = self._map_uv_between_frames(
                u=int(ux),
                v=int(vy),
                src_width=width,
                src_height=height,
                dst_width=depth_width,
                dst_height=depth_height,
            )
            center_depth_mm = median_depth_at(
                depth=depth,
                u=ux_depth,
                v=vy_depth,
                depth_scale=depth_scale,
                window=self._cfg.center_depth_window,
                min_depth_mm=self._cfg.min_depth_mm,
                max_depth_mm=self._cfg.max_depth_mm,
            )
            candidate = dict(det)
            candidate["bbox_xyxy"] = bbox
            candidate["_center_uv"] = (ux, vy)
            candidate["_center_dist_px"] = float(np.hypot(ux - cx, vy - cy))
            candidate["center_depth_mm"] = center_depth_mm
            candidates.append(candidate)

        tracker = self._target_tracker
        if tracker is None:
            seed = sorted(candidates, key=lambda det: (float(det["_center_dist_px"]), -float(det.get("conf", 0.0))))[0]
            lock_hits = 1
            state = "locked" if lock_hits >= int(self._cfg.target_lock_hits) else "acquire"
            tracker_score = max(0.0, min(1.0, 0.5 + 0.5 * float(seed.get("conf", 0.0))))
            self._target_tracker = {
                "bbox": [int(v) for v in seed["bbox_xyxy"]],
                "center_uv": tuple(seed["_center_uv"]),
                "depth_mm": seed.get("center_depth_mm"),
                "state": state,
                "lock_hits": lock_hits,
                "lost_count": 0,
                "score": tracker_score,
            }
            if state != "locked":
                return None, state, tracker_score, "acquiring_lock"
            return seed, state, tracker_score, None

        prev_bbox = [int(v) for v in tracker.get("bbox", [0, 0, 1, 1])]
        prev_center = tracker.get("center_uv")
        prev_depth = tracker.get("depth_mm")
        if not isinstance(prev_center, tuple) or len(prev_center) != 2:
            x1, y1, x2, y2 = prev_bbox
            prev_center = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

        best: dict[str, Any] | None = None
        best_score = -1e9
        best_iou = 0.0
        best_center_jump = 1e9
        best_depth_jump = 0.0
        for det in candidates:
            iou = self._bbox_iou(prev_bbox, det["bbox_xyxy"])
            center = det["_center_uv"]
            center_jump = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
            center_score = 1.0 - min(1.0, center_jump / max(1.0, float(self._cfg.target_max_center_jump_px)))
            depth_jump = 0.0
            if prev_depth is not None and det.get("center_depth_mm") is not None:
                depth_jump = abs(float(det["center_depth_mm"]) - float(prev_depth))
            depth_score = (
                1.0
                if (prev_depth is None or det.get("center_depth_mm") is None)
                else 1.0 - min(1.0, depth_jump / max(1.0, float(self._cfg.target_max_depth_jump_mm)))
            )
            conf = max(0.0, min(1.0, float(det.get("conf", 0.0))))
            score = (0.50 * iou) + (0.25 * center_score) + (0.15 * depth_score) + (0.10 * conf)
            if score > best_score:
                best = det
                best_score = score
                best_iou = float(iou)
                best_center_jump = center_jump
                best_depth_jump = depth_jump

        if best is None:
            return self._mark_target_miss(reason="no_candidate")

        reject_reason: str | None = None
        if best_iou < float(self._cfg.target_lock_iou_min):
            reject_reason = "iou_break"
        elif best_center_jump > float(self._cfg.target_max_center_jump_px):
            reject_reason = "center_jump"
        elif (
            prev_depth is not None
            and best.get("center_depth_mm") is not None
            and best_depth_jump > float(self._cfg.target_max_depth_jump_mm)
        ):
            reject_reason = "depth_jump"
        if reject_reason is not None:
            return self._mark_target_miss(reason=reject_reason)

        lock_hits = int(tracker.get("lock_hits", 1))
        if str(tracker.get("state", "acquire")) != "locked":
            lock_hits += 1
        state = "locked" if lock_hits >= int(self._cfg.target_lock_hits) else "acquire"
        tracker_score = max(0.0, min(1.0, float(best_score)))
        self._target_tracker = {
            "bbox": [int(v) for v in best["bbox_xyxy"]],
            "center_uv": tuple(best["_center_uv"]),
            "depth_mm": best.get("center_depth_mm"),
            "state": state,
            "lock_hits": lock_hits,
            "lost_count": 0,
            "score": tracker_score,
        }
        if state != "locked":
            return None, state, tracker_score, "acquiring_lock"
        return best, state, tracker_score, None

    @staticmethod
    def _map_uv_between_frames(
        u: int,
        v: int,
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
    ) -> tuple[int, int]:
        if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
            return int(u), int(v)

        mapped_u = int(round(((float(u) + 0.5) * float(dst_width) / float(src_width)) - 0.5))
        mapped_v = int(round(((float(v) + 0.5) * float(dst_height) / float(src_height)) - 0.5))
        mapped_u = max(0, min(int(dst_width) - 1, mapped_u))
        mapped_v = max(0, min(int(dst_height) - 1, mapped_v))
        return mapped_u, mapped_v

    @classmethod
    def _map_bbox_between_frames(
        cls,
        bbox: list[int],
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = cls._clamp_bbox(bbox, src_width, src_height)
        mx1, my1 = cls._map_uv_between_frames(x1, y1, src_width, src_height, dst_width, dst_height)
        # Use (x2-1, y2-1) so max-edge maps inside frame, then restore exclusive bound.
        mx2_inclusive, my2_inclusive = cls._map_uv_between_frames(
            max(x1, x2 - 1),
            max(y1, y2 - 1),
            src_width,
            src_height,
            dst_width,
            dst_height,
        )
        mx2 = min(dst_width, mx2_inclusive + 1)
        my2 = min(dst_height, my2_inclusive + 1)
        if mx2 <= mx1:
            mx2 = min(dst_width, mx1 + 1)
        if my2 <= my1:
            my2 = min(dst_height, my1 + 1)
        return mx1, my1, mx2, my2

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
            axis_quality_text = "-" if axis_quality is None else f"{float(axis_quality):.2f}"
            stability_state = str(grasp.get("stability_state", "live"))
            stability_score = grasp.get("stability_score")
            stability_text = "-" if stability_score is None else f"{float(stability_score):.2f}"
            cv2.putText(
                image,
                (
                    f"G({grasp['x_mm']:.1f},{grasp['y_mm']:.1f},{grasp['z_mm']:.1f}) "
                    f"yaw={grasp['yaw_deg']:.1f} axis={axis_state}/{axis_quality_text} "
                    f"stab={stability_state}/{stability_text}"
                ),
                (max(8, gu - 180), max(18, gv - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
