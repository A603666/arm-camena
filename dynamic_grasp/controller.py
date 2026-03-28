from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bridge import RobotBridge
from .config import AppConfig
from .math_utils import HandEyeModel, matrix_to_pose, plan_scan_xy, pose_to_matrix
from .vision_client import VisionClient, VisionClientError, VisionSnapshot

SCAN_CONVERGED_STABLE_HITS = 3


@dataclass(frozen=True)
class TargetPlan:
    snapshot: VisionSnapshot
    target_point_base_m: tuple[float, float, float]
    target_yaw_rad: float
    hover_tcp_pose: tuple[float, float, float, float, float, float]
    final_tcp_pose: tuple[float, float, float, float, float, float]


class DynamicGraspController:
    def __init__(
        self,
        config: AppConfig,
        robot: RobotBridge,
        vision_client: VisionClient,
        handeye: HandEyeModel,
        logger: logging.Logger,
        mode: str = "step",
    ) -> None:
        self.config = config
        self.robot = robot
        self.vision_client = vision_client
        self.handeye = handeye
        self.logger = logger
        self.mode = str(mode).strip().lower() or "step"
        self._poll_sleep_sec = 1.0 / max(0.5, self.config.vision.poll_hz)

        self.ready_flange_pose = self.robot.resolve_route_pose(self.config.route.ready_from)
        self.transport_flange_pose = self.robot.resolve_route_pose(self.config.route.transport_from)
        self.dump_flange_pose = self.robot.resolve_route_pose(self.config.route.dump_from)
        arm_cfg = getattr(self.robot, "arm_cfg", {})
        threepoint_cfg = arm_cfg.get("threepoint", {}) if isinstance(arm_cfg, dict) else {}
        min_joint7_deg = float(threepoint_cfg.get("min_joint7_deg", 30.0))
        self._min_joint7_deg = min_joint7_deg
        self._min_joint7_rad = math.radians(min_joint7_deg)
        self.ready_tcp_pose: list[float] | None = None

    def shutdown(self) -> None:
        self.robot.shutdown()

    def _prompt(self, label: str) -> None:
        if self.mode != "step":
            return
        input(f"[STEP] {label} -> press Enter to continue (Ctrl+C to abort) ")

    def _fallback_ready_tcp_pose(self) -> list[float]:
        return matrix_to_pose(pose_to_matrix(self.ready_flange_pose) @ pose_to_matrix(self.handeye.tcp_offset_pose))

    def _refresh_ready_tcp_pose(self) -> None:
        current = self.robot.get_tcp_pose()
        self.ready_tcp_pose = current if current is not None else self._fallback_ready_tcp_pose()

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))

    def _prefer_equivalent_grasp_yaw(self, yaw_rad: float, reference_yaw_rad: float | None) -> float:
        """Pick yaw or yaw+pi, whichever is closer to the reference orientation.

        For a two-finger gripper, rotating 180deg around TCP Z can still represent
        the same grasp axis. Prefer the closer equivalent to reduce wrist flips.
        """
        base = self._normalize_angle(yaw_rad)
        if reference_yaw_rad is None:
            return base

        ref = self._normalize_angle(reference_yaw_rad)
        alt = self._normalize_angle(base + math.pi)
        err_base = abs(self._normalize_angle(base - ref))
        err_alt = abs(self._normalize_angle(alt - ref))
        return alt if (err_alt + 1e-6) < err_base else base

    def _move_ready(self, reason: str) -> bool:
        self.logger.info("move ready: %s", reason)
        ok = self.robot.move_flange_p(self.ready_flange_pose)
        if ok:
            if not self._joint7_is_safe(f"move ready ({reason})"):
                return False
            self._refresh_ready_tcp_pose()
        return ok

    def _joint7_is_safe(self, label: str) -> bool:
        joints = self.robot.get_joint_positions()
        if joints is None:
            self.logger.error("%s failed: no joint feedback for J7 guard", label)
            return False
        if len(joints) != 7:
            self.logger.error("%s failed: invalid joint feedback length=%d", label, len(joints))
            return False
        joint7_rad = float(joints[6])
        if joint7_rad <= self._min_joint7_rad:
            self.logger.error(
                "%s failed: J7=%.3fdeg must be > %.3fdeg",
                label,
                math.degrees(joint7_rad),
                self._min_joint7_deg,
            )
            return False
        return True

    def _gripper_open(self) -> bool:
        ok = self.robot.open_gripper(self.config.grasp.open_width_m, self.config.grasp.force_n)
        if ok and self.config.runtime.gripper_dwell_sec > 0.0:
            time.sleep(self.config.runtime.gripper_dwell_sec)
        return ok

    def _gripper_close(self) -> bool:
        ok = self.robot.close_gripper(self.config.grasp.close_width_m, self.config.grasp.force_n)
        if ok and self.config.runtime.gripper_dwell_sec > 0.0:
            time.sleep(self.config.runtime.gripper_dwell_sec)
        return ok

    def _wait_for_trackable_target(self, timeout_sec: float | None, stable_required: int | None = None) -> VisionSnapshot | None:
        deadline = None if timeout_sec is None else (time.time() + float(timeout_sec))
        stable_hits = 0
        last_signature: tuple[Any, ...] | None = None
        needed = max(1, stable_required or self.config.vision.target_stable_frames)

        while True:
            try:
                health = self.vision_client.get_health()
                snapshot = self.vision_client.get_latest_target()
            except VisionClientError as exc:
                self.logger.warning("vision poll failed: %s", exc)
                stable_hits = 0
                last_signature = None
                snapshot = None
                health_stream = False
            else:
                health_stream = health.stream_connected

            if snapshot is not None and health_stream and snapshot.is_trackable:
                signature = snapshot.signature()
                stable_hits = stable_hits + 1 if signature == last_signature else 1
                last_signature = signature
                if stable_hits >= needed:
                    return snapshot
            else:
                stable_hits = 0
                last_signature = None

            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(self._poll_sleep_sec)

    def _scan_to_center(self, initial_snapshot: VisionSnapshot) -> VisionSnapshot | None:
        snapshot = initial_snapshot
        stable_hits = 0

        while True:
            if not snapshot.is_trackable:
                return None
            grasp = snapshot.grasp_point_optical_m
            if grasp is None:
                return None

            if abs(grasp[0] * 1000.0) <= self.config.scan.converge_tol_mm and abs(grasp[1] * 1000.0) <= self.config.scan.converge_tol_mm:
                stable_hits += 1
                if stable_hits >= SCAN_CONVERGED_STABLE_HITS:
                    return snapshot
                snapshot = self._wait_for_trackable_target(
                    timeout_sec=self.config.scan.lost_target_timeout_sec,
                    stable_required=1,
                )
                if snapshot is None:
                    return None
                continue

            stable_hits = 0
            current_flange = self.robot.get_flange_pose()
            current_tcp = self.robot.get_tcp_pose()
            if current_flange is None or current_tcp is None or self.ready_tcp_pose is None:
                return None

            delta_xy = self.handeye.scan_delta_xy_base(current_flange, grasp)
            planned_xy = plan_scan_xy(
                ready_xy=self.ready_tcp_pose[:2],
                current_xy=current_tcp[:2],
                delta_xy=delta_xy,
                max_step_m=self.config.scan.max_step_xy_m,
                max_offset_m=self.config.scan.max_offset_xy_m,
            )

            if np.allclose(planned_xy, np.array(current_tcp[:2], dtype=np.float64), atol=1e-6):
                self.logger.warning("scan window reached before visual convergence")
                return None

            target_tcp = list(current_tcp)
            target_tcp[0] = float(planned_xy[0])
            target_tcp[1] = float(planned_xy[1])
            target_tcp[2] = float(self.ready_tcp_pose[2])
            target_tcp[5] = float(current_tcp[5])

            if not self.robot.move_tcp_p(target_tcp):
                return None
            if not self._joint7_is_safe("scan recenter"):
                return None

            snapshot = self._wait_for_trackable_target(
                timeout_sec=self.config.scan.lost_target_timeout_sec,
                stable_required=1,
            )
            if snapshot is None:
                return None

    def _align_yaw(self, snapshot: VisionSnapshot) -> VisionSnapshot | None:
        current_flange = self.robot.get_flange_pose()
        current_tcp = self.robot.get_tcp_pose()
        if current_flange is None or current_tcp is None or snapshot.axis_dir_optical is None:
            return None

        target_yaw = self.handeye.axis_yaw_in_base(
            current_flange,
            snapshot.axis_dir_optical,
            yaw_offset_deg=self.config.grasp.yaw_alignment_offset_deg,
        )
        target_yaw = self._prefer_equivalent_grasp_yaw(target_yaw, current_tcp[5])
        if abs(target_yaw - current_tcp[5]) > 1e-4:
            yaw_aligned_tcp = list(current_tcp)
            yaw_aligned_tcp[5] = target_yaw
            if not self.robot.move_tcp_p(yaw_aligned_tcp):
                return None
            if not self._joint7_is_safe("yaw align"):
                return None

        return self._wait_for_trackable_target(
            timeout_sec=self.config.scan.lost_target_timeout_sec,
            stable_required=1,
        )

    def _build_target_plan(self, snapshot: VisionSnapshot) -> TargetPlan | None:
        current_flange = self.robot.get_flange_pose()
        current_tcp = self.robot.get_tcp_pose()
        if current_flange is None or snapshot.grasp_point_optical_m is None or snapshot.axis_dir_optical is None:
            return None
        if current_tcp is None:
            return None
        if self.ready_tcp_pose is None:
            return None

        target_point_base = self.handeye.point_optical_to_base(current_flange, snapshot.grasp_point_optical_m)
        target_yaw = self.handeye.axis_yaw_in_base(
            current_flange,
            snapshot.axis_dir_optical,
            yaw_offset_deg=self.config.grasp.yaw_alignment_offset_deg,
        )
        reference_yaw = current_tcp[5]
        target_yaw = self._prefer_equivalent_grasp_yaw(target_yaw, reference_yaw)

        final_z = float(target_point_base[2]) + self.config.grasp.final_z_offset_m
        if final_z < self.config.grasp.min_safe_z_m:
            self.logger.warning("reject target: final z %.4f below min_safe_z %.4f", final_z, self.config.grasp.min_safe_z_m)
            return None

        descent = float(self.ready_tcp_pose[2]) - final_z
        if descent > self.config.grasp.max_descent_m:
            self.logger.warning(
                "reject target: descent %.4f exceeds max_descent %.4f",
                descent,
                self.config.grasp.max_descent_m,
            )
            return None

        hover_z = max(
            float(target_point_base[2]) + self.config.grasp.hover_clearance_m,
            final_z + max(0.005, self.config.grasp.hover_clearance_m * 0.25),
        )
        hover_tcp_pose = (
            float(target_point_base[0]),
            float(target_point_base[1]),
            hover_z,
            float(current_tcp[3]),
            float(current_tcp[4]),
            target_yaw,
        )
        final_tcp_pose = (
            float(target_point_base[0]),
            float(target_point_base[1]),
            final_z,
            float(current_tcp[3]),
            float(current_tcp[4]),
            target_yaw,
        )
        return TargetPlan(
            snapshot=snapshot,
            target_point_base_m=(
                float(target_point_base[0]),
                float(target_point_base[1]),
                float(target_point_base[2]),
            ),
            target_yaw_rad=target_yaw,
            hover_tcp_pose=hover_tcp_pose,
            final_tcp_pose=final_tcp_pose,
        )

    def _grasp_verified(self, status: Any, ctrl: Any) -> bool:
        if status is None or ctrl is None:
            return False
        if not self.robot.gripper_is_ok() or self.robot.gripper_fps() <= 0.0:
            return False
        if not hasattr(status, "msg") or not hasattr(ctrl, "msg"):
            return False

        foc = getattr(status.msg, "foc_status", None)
        if foc is None:
            return False
        if not bool(getattr(foc, "driver_enable_status", False)):
            return False

        faults = [
            bool(getattr(foc, "voltage_too_low", False)),
            bool(getattr(foc, "motor_overheating", False)),
            bool(getattr(foc, "driver_overcurrent", False)),
            bool(getattr(foc, "driver_overheating", False)),
            bool(getattr(foc, "sensor_status", False)),
            bool(getattr(foc, "driver_error_status", False)),
        ]
        if any(faults):
            return False

        width = float(getattr(status.msg, "width", -1.0))
        width_min, width_max = self.config.grasp.verify_width_range_m
        return width_min <= width <= width_max

    def _verify_grasp(self) -> bool:
        deadline = time.time() + self.config.grasp.verify_timeout_sec
        while time.time() < deadline:
            status = self.robot.get_gripper_status()
            ctrl = self.robot.get_gripper_ctrl_states()
            if self._grasp_verified(status, ctrl):
                self.logger.info("grasp verification passed")
                return True
            time.sleep(0.05)
        self.logger.warning("grasp verification timed out")
        return False

    def _recover_to_ready(self, reason: str) -> bool:
        self.logger.warning("recover to ready: %s", reason)
        current_tcp = self.robot.get_tcp_pose()
        if current_tcp is not None and self.ready_tcp_pose is not None and current_tcp[2] < self.ready_tcp_pose[2] - 1e-6:
            lift_tcp = list(current_tcp)
            lift_tcp[2] = float(self.ready_tcp_pose[2])
            self.robot.move_tcp_l(lift_tcp)
        return self._move_ready(reason)

    def _move_dump_route(self) -> bool:
        if not self.robot.move_flange_p(self.transport_flange_pose):
            return False
        if not self._joint7_is_safe("move transport"):
            return False
        if not self.robot.move_flange_l(self.dump_flange_pose):
            return False
        if not self._joint7_is_safe("move dump"):
            return False
        if not self._gripper_open():
            return False
        if not self.robot.move_flange_p(self.transport_flange_pose):
            return False
        if not self._joint7_is_safe("move transport (return)"):
            return False
        return self._move_ready("dump complete")

    def execute_cycle(self, initial_snapshot: VisionSnapshot) -> bool:
        self.logger.info("dynamic grasp cycle start")

        self._prompt("Move ready and open gripper")
        if not self._move_ready("cycle start"):
            return False
        if not self._gripper_open():
            return False

        self._prompt("Start scan phase")
        centered_snapshot = self._scan_to_center(initial_snapshot)
        if centered_snapshot is None:
            self._recover_to_ready("scan phase failed")
            return False

        aligned_snapshot = self._align_yaw(centered_snapshot)
        if aligned_snapshot is None:
            self._recover_to_ready("yaw align failed")
            return False

        self._prompt("Start descend phase")
        plan = self._build_target_plan(aligned_snapshot)
        if plan is None:
            self._recover_to_ready("invalid descend plan")
            return False

        if not self.robot.move_tcp_p(plan.hover_tcp_pose):
            self._recover_to_ready("hover move failed")
            return False
        if not self._joint7_is_safe("hover move"):
            self._recover_to_ready("hover move failed J7 check")
            return False

        refreshed_snapshot = self._wait_for_trackable_target(
            timeout_sec=self.config.scan.lost_target_timeout_sec,
            stable_required=1,
        )
        if refreshed_snapshot is None:
            self._recover_to_ready("target lost before final descend")
            return False

        plan = self._build_target_plan(refreshed_snapshot)
        if plan is None:
            self._recover_to_ready("refreshed descend plan invalid")
            return False

        if not self.robot.move_tcp_l(plan.final_tcp_pose):
            self._recover_to_ready("final descend failed")
            return False
        if not self._joint7_is_safe("final descend"):
            self._recover_to_ready("final descend failed J7 check")
            return False

        self._prompt("Close gripper")
        if not self._gripper_close():
            self._gripper_open()
            self._recover_to_ready("close gripper failed")
            return False

        if not self._verify_grasp():
            self._gripper_open()
            self._recover_to_ready("grasp verification failed")
            return False

        if not self.robot.move_tcp_l(plan.hover_tcp_pose):
            self._recover_to_ready("lift after grasp failed")
            return False
        if not self._joint7_is_safe("lift after grasp"):
            self._recover_to_ready("lift after grasp failed J7 check")
            return False

        if not self._move_ready("post-grasp ready"):
            return False

        self._prompt("Enter recovery route")
        if not self._move_dump_route():
            return False

        self.logger.info("dynamic grasp cycle completed")
        return True

    def run(self, once: bool) -> int:
        if not self.robot.connect():
            self.logger.error("robot backend connect failed: %s", self.robot.diagnostics())
            return 1

        self.robot.set_speed_percent(self.config.runtime.speed_percent)
        self.robot.set_tcp_offset(list(self.handeye.tcp_offset_pose))

        if not self._move_ready("startup"):
            self.logger.error("failed to move robot to ready pose")
            return 1

        if not self._gripper_open():
            self.logger.error("failed to open gripper at startup")
            return 1

        try:
            while True:
                snapshot = self._wait_for_trackable_target(timeout_sec=None)
                if snapshot is None:
                    continue
                success = self.execute_cycle(snapshot)
                if once:
                    return 0 if success else 1
        except KeyboardInterrupt:
            self.logger.warning("keyboard interrupt received, sending estop/recover")
            estop_ok = False
            recover_ok = False
            try:
                estop_ok = self.robot.estop()
                if estop_ok:
                    recover_ok = self.robot.recover_after_estop()
                else:
                    self.logger.warning("keyboard interrupt estop returned false")
            except Exception:
                self.logger.exception("keyboard interrupt estop/recover failed")
            finally:
                self.logger.warning(
                    "keyboard interrupt handled: estop_ok=%s recover_ok=%s",
                    estop_ok,
                    recover_ok,
                )
                return 130
