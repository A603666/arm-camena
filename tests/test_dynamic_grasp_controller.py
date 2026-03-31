from __future__ import annotations

import logging
import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_grasp.config import (
    AppConfig,
    GraspConfig,
    HandEyeConfig,
    RouteConfig,
    RuntimeConfig,
    ScanConfig,
    VisionConfig,
)
from dynamic_grasp.controller import DynamicGraspController
from dynamic_grasp.math_utils import HandEyeModel
from dynamic_grasp.vision_client import VisionHealth, VisionSnapshot


def make_snapshot(
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float = 760.0,
    axis_dir: tuple[float, float, float] = (1.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
    status: str = "ok",
    quality_score: float = 0.8,
    quality_flags: tuple[str, ...] = (),
) -> VisionSnapshot:
    grasp_point = None if status != "ok" else (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)
    axis = None if status != "ok" else axis_dir
    return VisionSnapshot(
        status=status,
        schema_version=2 if status == "ok" else 2,
        bbox_xyxy=(100, 100, 200, 200) if status == "ok" else None,
        grasp_point_optical_m=grasp_point,
        axis_dir_optical=axis,
        target_center_depth_m=z_mm / 1000.0 if status == "ok" else None,
        width_m=0.03 if status == "ok" else None,
        tracking_state="locked" if status == "ok" else "lost",
        tracking_confidence=0.8 if status == "ok" else 0.0,
        tracking_reason=None if status == "ok" else "no_target",
        source_quality_score=quality_score if status == "ok" else None,
        quality_flags=quality_flags if status == "ok" else (),
        raw=(
            {
                "status": status,
                "grasp": {
                    "yaw_deg": yaw_deg,
                },
                "segmentation": {
                    "quality_score": quality_score,
                    "quality_flags": list(quality_flags),
                },
            }
            if status == "ok"
            else {"status": status}
        ),
    )


class FakeVisionClient:
    def __init__(self, snapshots: list[VisionSnapshot], stream_sequence: list[bool] | None = None) -> None:
        self.snapshots = list(snapshots)
        self.stream_sequence = list(stream_sequence) if stream_sequence is not None else None
        self.last_snapshot = self.snapshots[-1] if self.snapshots else make_snapshot(x_mm=0.0, y_mm=0.0, status="no_target")

    def get_health(self) -> VisionHealth:
        stream_connected = True
        if self.stream_sequence is not None and self.stream_sequence:
            stream_connected = self.stream_sequence.pop(0)
        return VisionHealth("ok", stream_connected, 0.01, 0.01, "ok")

    def get_latest_target(self) -> VisionSnapshot:
        if self.snapshots:
            self.last_snapshot = self.snapshots.pop(0)
        return self.last_snapshot


class FakeRobot:
    def __init__(self, verified_width: float = 0.02, recover_success: bool = True, joint7_deg: float = 60.0) -> None:
        self.route = SimpleNamespace(baseline_rpy=(0.0, 0.0, 0.0))
        self.ready_pose = [0.0, 0.0, 0.40, 0.0, 0.0, 0.0]
        self.transport_pose = [0.20, 0.00, 0.40, 0.0, 0.0, 0.0]
        self.dump_pose = [0.25, 0.00, 0.34, 0.0, 0.0, 0.0]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(joint7_deg)]
        self.current_flange = list(self.ready_pose)
        self.current_tcp = list(self.ready_pose)
        self.verified_width = verified_width
        self.recover_success = recover_success
        self.actions: list[tuple] = []

    def resolve_route_pose(self, reference: str) -> list[float]:
        mapping = {
            "threepoint.ready": self.ready_pose,
            "threepoint.scan": self.ready_pose,
            "threepoint.transport": self.transport_pose,
            "threepoint.dump_pre": self.transport_pose,
            "threepoint.dump": self.dump_pose,
        }
        return list(mapping[reference])

    def connect(self) -> bool:
        self.actions.append(("connect",))
        return True

    def shutdown(self) -> None:
        self.actions.append(("shutdown",))

    def set_speed_percent(self, percent: int) -> None:
        self.actions.append(("speed", percent))

    def set_tcp_offset(self, pose) -> None:
        self.actions.append(("tcp_offset", tuple(pose)))

    def move_flange_p(self, pose) -> bool:
        self.current_flange = list(pose)
        self.current_tcp = list(pose)
        self.actions.append(("move_flange_p", tuple(round(v, 4) for v in pose)))
        return True

    def move_flange_l(self, pose) -> bool:
        self.current_flange = list(pose)
        self.current_tcp = list(pose)
        self.actions.append(("move_flange_l", tuple(round(v, 4) for v in pose)))
        return True

    def move_tcp_p(self, pose) -> bool:
        self.current_tcp = list(pose)
        self.current_flange = list(pose)
        self.actions.append(("move_tcp_p", tuple(round(v, 4) for v in pose)))
        return True

    def move_tcp_l(self, pose) -> bool:
        self.current_tcp = list(pose)
        self.current_flange = list(pose)
        self.actions.append(("move_tcp_l", tuple(round(v, 4) for v in pose)))
        return True

    def get_flange_pose(self):
        return list(self.current_flange)

    def get_tcp_pose(self):
        return list(self.current_tcp)

    def get_joint_positions(self):
        return list(self.joint_positions)

    def open_gripper(self, width_m: float, force_n: float) -> bool:
        self.actions.append(("open", round(width_m, 4), round(force_n, 3)))
        return True

    def close_gripper(self, width_m: float, force_n: float) -> bool:
        self.actions.append(("close", round(width_m, 4), round(force_n, 3)))
        return True

    def get_gripper_status(self):
        foc = SimpleNamespace(
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=False,
            driver_overheating=False,
            sensor_status=False,
            driver_error_status=False,
            driver_enable_status=True,
        )
        return SimpleNamespace(msg=SimpleNamespace(width=self.verified_width, force=1.0, foc_status=foc))

    def get_gripper_ctrl_states(self):
        return SimpleNamespace(msg=SimpleNamespace(width=self.verified_width, force=1.0, status_code=1, set_zero=0))

    def gripper_is_ok(self) -> bool:
        return True

    def gripper_fps(self) -> float:
        return 30.0

    def diagnostics(self) -> dict[str, str]:
        return {}

    def estop(self) -> bool:
        self.actions.append(("estop",))
        return True

    def recover_after_estop(self) -> bool:
        self.actions.append(("recover_after_estop",))
        return self.recover_success


def make_config(*, plan_buffer_frames: int = 3, max_replan_jump_mm: float = 2000.0) -> AppConfig:
    return AppConfig(
        config_path=Path("/tmp/dynamic_grasp_test.yaml"),
        vision=VisionConfig(
            base_url="http://mock",
            api_version="v2",
            poll_hz=50.0,
            health_timeout_sec=0.1,
            target_stable_frames=1,
            stable_pos_tol_mm=5.0,
            stable_z_tol_mm=5.0,
            stable_yaw_tol_deg=5.0,
            plan_buffer_frames=plan_buffer_frames,
            max_replan_jump_mm=max_replan_jump_mm,
            min_quality_score=0.5,
            reject_on_quality_drop=True,
        ),
        handeye=HandEyeConfig(mode="nominal", extrinsics_path=Path("/tmp/handeye.yaml")),
        scan=ScanConfig(max_offset_xy_m=0.18, max_step_xy_m=0.05, converge_tol_mm=5.0, lost_target_timeout_sec=0.03),
        grasp=GraspConfig(
            open_width_m=0.05,
            close_width_m=0.0,
            force_n=1.0,
            verify_enabled=False,
            prepick_offset_m=0.05,
            final_z_offset_m=0.0,
            max_descent_m=0.35,
            min_safe_z_m=0.10,
            yaw_alignment_offset_deg=0.0,
            verify_width_range_m=(0.003, 0.08),
            verify_timeout_sec=0.03,
        ),
        route=RouteConfig(
            ready_from="threepoint.ready",
            transport_from="threepoint.transport",
            dump_from="threepoint.dump",
        ),
        runtime=RuntimeConfig(
            arm_config_path=Path("/tmp/arm_default.yaml"),
            speed_percent=20,
            move_timeout_sec=5.0,
            gripper_dwell_sec=0.0,
            log_dir=Path("/tmp"),
        ),
    )


def make_identity_handeye() -> HandEyeModel:
    return HandEyeModel(
        extrinsics_path=Path("/tmp/identity.yaml"),
        mode="nominal",
        flange_to_camera_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        camera_to_optical_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tcp_offset_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


class DynamicGraspControllerTests(unittest.TestCase):
    def _make_controller(self, robot: FakeRobot, vision: FakeVisionClient, config: AppConfig | None = None) -> DynamicGraspController:
        logger = logging.getLogger(f"dynamic_grasp_test_{id(robot)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        controller = DynamicGraspController(
            config=config or make_config(),
            robot=robot,
            vision_client=vision,
            handeye=make_identity_handeye(),
            logger=logger,
            mode="auto",
        )
        controller.ready_tcp_pose = list(robot.current_tcp)
        return controller

    def test_success_path_reaches_dump_route(self) -> None:
        robot = FakeRobot(verified_width=0.02)
        centered = make_snapshot(x_mm=1.0, y_mm=1.0)
        vision = FakeVisionClient(
            [
                *([centered] * 12),
            ]
        )
        controller = self._make_controller(robot, vision)
        success = controller.execute_cycle(make_snapshot(x_mm=20.0, y_mm=15.0))
        self.assertTrue(success)
        self.assertIn(("close", 0.0, 1.0), robot.actions)
        self.assertIn(("move_flange_p", tuple(round(v, 4) for v in robot.transport_pose)), robot.actions)
        self.assertIn(("move_flange_l", tuple(round(v, 4) for v in robot.dump_pose)), robot.actions)
        transport_moves = [a for a in robot.actions if a == ("move_flange_p", tuple(round(v, 4) for v in robot.transport_pose))]
        self.assertGreaterEqual(len(transport_moves), 2)
        self.assertIn(("move_flange_p", tuple(round(v, 4) for v in robot.ready_pose)), robot.actions)

    def test_failed_grasp_without_verification_still_enters_dump_route(self) -> None:
        robot = FakeRobot(verified_width=0.0)
        centered = make_snapshot(x_mm=0.5, y_mm=0.5)
        vision = FakeVisionClient(
            [
                *([centered] * 12),
            ]
        )
        controller = self._make_controller(robot, vision)
        success = controller.execute_cycle(make_snapshot(x_mm=18.0, y_mm=10.0))
        self.assertTrue(success)
        self.assertIn(("close", 0.0, 1.0), robot.actions)
        self.assertIn(("move_flange_p", tuple(round(v, 4) for v in robot.transport_pose)), robot.actions)

    def test_failed_grasp_with_verification_enabled_does_not_enter_dump_route(self) -> None:
        robot = FakeRobot(verified_width=0.0)
        centered = make_snapshot(x_mm=0.5, y_mm=0.5)
        vision = FakeVisionClient([*([centered] * 12)])
        config = make_config()
        config = AppConfig(
            config_path=config.config_path,
            vision=config.vision,
            handeye=config.handeye,
            scan=config.scan,
            grasp=GraspConfig(
                open_width_m=config.grasp.open_width_m,
                close_width_m=config.grasp.close_width_m,
                force_n=config.grasp.force_n,
                verify_enabled=True,
                prepick_offset_m=config.grasp.prepick_offset_m,
                final_z_offset_m=config.grasp.final_z_offset_m,
                max_descent_m=config.grasp.max_descent_m,
                min_safe_z_m=config.grasp.min_safe_z_m,
                yaw_alignment_offset_deg=config.grasp.yaw_alignment_offset_deg,
                verify_width_range_m=config.grasp.verify_width_range_m,
                verify_timeout_sec=config.grasp.verify_timeout_sec,
            ),
            route=config.route,
            runtime=config.runtime,
        )
        controller = self._make_controller(robot, vision, config=config)
        success = controller.execute_cycle(make_snapshot(x_mm=18.0, y_mm=10.0))
        self.assertFalse(success)
        self.assertIn(("close", 0.0, 1.0), robot.actions)
        self.assertNotIn(("move_flange_p", tuple(round(v, 4) for v in robot.transport_pose)), robot.actions)

    def test_lost_target_returns_to_ready_without_grasp(self) -> None:
        robot = FakeRobot(verified_width=0.02)
        no_target = make_snapshot(x_mm=0.0, y_mm=0.0, status="no_target")
        vision = FakeVisionClient([no_target], stream_sequence=[True, True, True, True])
        controller = self._make_controller(robot, vision)
        success = controller.execute_cycle(make_snapshot(x_mm=22.0, y_mm=-18.0))
        self.assertFalse(success)
        self.assertNotIn(("close", 0.0, 1.0), robot.actions)
        self.assertIn(("move_flange_p", tuple(round(v, 4) for v in robot.ready_pose)), robot.actions)

    def test_align_yaw_prefers_equivalent_solution_near_current_pose(self) -> None:
        robot = FakeRobot(verified_width=0.02)
        robot.current_tcp[5] = 3.0
        robot.current_flange[5] = 0.0
        snapshot = make_snapshot(x_mm=1.0, y_mm=1.0, axis_dir=(1.0, 0.0, 0.0))
        vision = FakeVisionClient([snapshot], stream_sequence=[True])
        controller = self._make_controller(robot, vision)
        controller.ready_tcp_pose = list(robot.current_tcp)

        aligned = controller._align_yaw(snapshot)
        self.assertIsNotNone(aligned)

        move_actions = [action for action in robot.actions if action[0] == "move_tcp_p"]
        self.assertTrue(move_actions)
        chosen_yaw = move_actions[-1][1][5]
        self.assertAlmostEqual(chosen_yaw, 3.1416, places=3)

    def test_snapshot_within_tolerance_uses_position_and_yaw_threshold(self) -> None:
        robot = FakeRobot()
        vision = FakeVisionClient([])
        controller = self._make_controller(robot, vision)

        base = make_snapshot(x_mm=10.0, y_mm=10.0, z_mm=760.0, yaw_deg=0.0)
        near = make_snapshot(x_mm=12.0, y_mm=11.0, z_mm=763.0, yaw_deg=3.0)
        far = make_snapshot(x_mm=20.0, y_mm=10.0, z_mm=760.0, yaw_deg=0.0)
        yaw_far = make_snapshot(x_mm=12.0, y_mm=11.0, z_mm=763.0, yaw_deg=12.0)

        self.assertTrue(controller._snapshot_within_tolerance(base, near))
        self.assertFalse(controller._snapshot_within_tolerance(base, far))
        self.assertFalse(controller._snapshot_within_tolerance(base, yaw_far))

    def test_wait_for_trackable_target_rejects_low_quality_snapshots(self) -> None:
        robot = FakeRobot()
        low = make_snapshot(x_mm=10.0, y_mm=10.0, quality_score=0.2, quality_flags=("depth_valid_ratio",))
        good = make_snapshot(x_mm=10.5, y_mm=9.8, quality_score=0.8)
        vision = FakeVisionClient([low, low, good])
        controller = self._make_controller(robot, vision)

        snapshot = controller._wait_for_trackable_target(timeout_sec=0.2, stable_required=1)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertGreaterEqual(snapshot.source_quality_score or 0.0, 0.5)

    def test_buffered_target_plan_fuses_samples_and_rejects_large_replan_jump(self) -> None:
        robot = FakeRobot()
        vision = FakeVisionClient([])
        config = make_config(plan_buffer_frames=3, max_replan_jump_mm=15.0)
        controller = self._make_controller(robot, vision, config=config)
        controller.ready_tcp_pose = list(robot.current_tcp)

        seed = make_snapshot(x_mm=10.0, y_mm=8.0, z_mm=760.0, yaw_deg=0.0)
        samples = iter(
            [
                make_snapshot(x_mm=10.5, y_mm=8.3, z_mm=760.0, yaw_deg=1.0),
                make_snapshot(x_mm=9.7, y_mm=7.9, z_mm=760.0, yaw_deg=-1.0),
            ]
        )
        controller._wait_for_trackable_target = lambda timeout_sec, stable_required=1: next(samples, None)  # type: ignore[method-assign]
        fused = controller._build_buffered_target_plan(seed)
        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertAlmostEqual(fused.target_point_base_m[0], 0.0100, places=3)
        self.assertAlmostEqual(fused.target_point_base_m[1], 0.0080, places=3)

        reference = controller._build_target_plan(seed)
        self.assertIsNotNone(reference)
        far_seed = make_snapshot(x_mm=80.0, y_mm=70.0, z_mm=760.0, yaw_deg=0.0)
        rejected = controller._build_buffered_target_plan(far_seed, reference_plan=reference)
        self.assertIsNone(rejected)

    def test_target_plan_prepick_is_exactly_5cm_above_pick(self) -> None:
        robot = FakeRobot()
        vision = FakeVisionClient([])
        controller = self._make_controller(robot, vision)
        snapshot = make_snapshot(x_mm=15.0, y_mm=10.0, z_mm=760.0, yaw_deg=0.0)

        plan = controller._build_target_plan(snapshot)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertAlmostEqual(plan.prepick_tcp_pose[2] - plan.pick_tcp_pose[2], 0.05, places=6)

    def test_success_path_uses_prepick_then_pick_then_lift_to_prepick(self) -> None:
        robot = FakeRobot(verified_width=0.02)
        centered = make_snapshot(x_mm=1.0, y_mm=1.0)
        vision = FakeVisionClient([*([centered] * 12)])
        controller = self._make_controller(robot, vision)

        success = controller.execute_cycle(make_snapshot(x_mm=20.0, y_mm=15.0))
        self.assertTrue(success)
        linear_moves = [action for action in robot.actions if action[0] == "move_tcp_l"]
        self.assertGreaterEqual(len(linear_moves), 2)

        pick_move = linear_moves[0][1]
        lift_move = linear_moves[-1][1]
        self.assertAlmostEqual(float(lift_move[2]) - float(pick_move[2]), 0.05, places=3)
        self.assertEqual(pick_move[0], lift_move[0])
        self.assertEqual(pick_move[1], lift_move[1])
        self.assertEqual(pick_move[3:], lift_move[3:])

    def test_keyboard_interrupt_runs_estop_and_recover(self) -> None:
        class InterruptVisionClient:
            def get_health(self):
                raise KeyboardInterrupt()

            def get_latest_target(self):
                raise AssertionError("get_latest_target should not be called after interrupt")

        robot = FakeRobot()
        logger = logging.getLogger(f"dynamic_grasp_interrupt_{id(robot)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        controller = DynamicGraspController(
            config=make_config(),
            robot=robot,
            vision_client=InterruptVisionClient(),
            handeye=make_identity_handeye(),
            logger=logger,
            mode="auto",
        )
        exit_code = controller.run(once=False)
        self.assertEqual(exit_code, 130)
        self.assertIn(("estop",), robot.actions)
        self.assertIn(("recover_after_estop",), robot.actions)

    def test_joint7_must_be_strictly_greater_than_30_deg(self) -> None:
        robot = FakeRobot(verified_width=0.02, joint7_deg=30.0)
        centered = make_snapshot(x_mm=0.5, y_mm=0.5)
        vision = FakeVisionClient([centered, centered, centered, centered, centered])
        controller = self._make_controller(robot, vision)

        success = controller.execute_cycle(make_snapshot(x_mm=12.0, y_mm=8.0))
        self.assertFalse(success)
        self.assertNotIn(("move_flange_l", tuple(round(v, 4) for v in robot.dump_pose)), robot.actions)

    def test_recover_to_ready_returns_false_when_linear_lift_fails(self) -> None:
        robot = FakeRobot()
        vision = FakeVisionClient([])
        controller = self._make_controller(robot, vision)
        robot.current_tcp = [0.0, 0.0, 0.20, 0.0, 0.0, 0.0]
        controller.ready_tcp_pose = [0.0, 0.0, 0.40, 0.0, 0.0, 0.0]

        def _fail_lift(pose) -> bool:
            robot.current_tcp = list(pose)
            robot.current_flange = list(pose)
            robot.actions.append(("move_tcp_l", tuple(round(v, 4) for v in pose)))
            return False

        robot.move_tcp_l = _fail_lift  # type: ignore[assignment]
        ok = controller._recover_to_ready("unit test failure path")
        self.assertFalse(ok)
        self.assertFalse(any(action[0] == "move_flange_p" for action in robot.actions))


if __name__ == "__main__":
    unittest.main()
