#!/usr/bin/env python3
"""Local validation for the NERO hand-eye/gripper frame setup."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


REPO_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = REPO_DIR / "模型文件" / "nero_description"
CONFIG_PATH = PACKAGE_DIR / "config" / "handeye_extrinsics.yaml"
SCRIPT_PATH = PACKAGE_DIR / "urdf" / "compute_nominal_extrinsics.py"
XACRO_PATH = PACKAGE_DIR / "urdf" / "nero_with_handeye_camera.xacro"
LAUNCH_PATH = PACKAGE_DIR / "launch" / "display_handeye_camera.launch.py"
TMP_URDF = Path("/tmp/nero_with_handeye_camera.urdf")

REQUIRED_LINKS = {
    "flange_frame",
    "camera_mount_frame",
    "handeye_camera_link",
    "handeye_camera_optical_frame",
    "gripper_mount_frame",
    "gripper_base_frame",
    "gripper_tcp_nominal",
}
FORBIDDEN_LINKS = {"handeye_camera_calibrated"}


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_compute_module():
    return _load_module("compute_nominal_extrinsics", SCRIPT_PATH)


def _load_launch_module():
    return _load_module("display_handeye_camera_launch", LAUNCH_PATH)


def _parse_triplet(raw: str) -> list[float]:
    return [float(value) for value in raw.split()]


def _assert_close_triplet(name: str, actual: list[float], expected: list[float], *, tol: float = 1e-12) -> None:
    if len(actual) != len(expected):
        raise SystemExit(f"{name} length mismatch: {actual} != {expected}")
    for idx, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if abs(actual_value - expected_value) > tol:
            raise SystemExit(
                f"{name}[{idx}] mismatch: actual={actual_value:.16f} expected={expected_value:.16f}"
            )


def _joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {
        joint.attrib["name"]: joint
        for joint in root.findall("joint")
        if "name" in joint.attrib
    }


def _assert_joint_transform(
    joints: dict[str, ET.Element],
    *,
    joint_name: str,
    parent_link: str,
    child_link: str,
    xyz: list[float],
    rpy: list[float],
) -> None:
    joint = joints.get(joint_name)
    if joint is None:
        raise SystemExit(f"Missing expected joint: {joint_name}")

    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    if parent is None or child is None or origin is None:
        raise SystemExit(f"Joint {joint_name} is missing parent/child/origin tags")

    if parent.attrib.get("link") != parent_link:
        raise SystemExit(
            f"Joint {joint_name} parent mismatch: {parent.attrib.get('link')} != {parent_link}"
        )
    if child.attrib.get("link") != child_link:
        raise SystemExit(
            f"Joint {joint_name} child mismatch: {child.attrib.get('link')} != {child_link}"
        )

    _assert_close_triplet(
        f"{joint_name}.xyz",
        _parse_triplet(origin.attrib.get("xyz", "")),
        xyz,
    )
    _assert_close_triplet(
        f"{joint_name}.rpy",
        _parse_triplet(origin.attrib.get("rpy", "")),
        rpy,
    )


def main() -> int:
    compute_module = _load_compute_module()
    expected = compute_module.build_payload(SCRIPT_PATH)
    actual = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if expected != actual:
        raise SystemExit("Config mismatch: handeye_extrinsics.yaml is not in sync with compute_nominal_extrinsics.py")

    urdf_text = _run(
        [
            "xacro",
            str(XACRO_PATH),
            f"extrinsics_file:={CONFIG_PATH}",
        ]
    )
    TMP_URDF.write_text(urdf_text, encoding="utf-8")
    root = ET.fromstring(urdf_text)
    links = {
        link.attrib["name"]
        for link in root.findall("link")
        if "name" in link.attrib
    }

    missing_links = sorted(REQUIRED_LINKS - links)
    if missing_links:
        raise SystemExit(f"Missing required links in generated URDF: {missing_links}")

    unexpected_links = sorted(FORBIDDEN_LINKS & links)
    if unexpected_links:
        raise SystemExit(f"Unexpected runtime-only links found in URDF: {unexpected_links}")

    nominal_camera = actual["nominal_camera"]
    gripper_nominal = actual["gripper_nominal"]
    joints = _joint_map(root)
    _assert_joint_transform(
        joints,
        joint_name="flange_frame_joint",
        parent_link="end_effector",
        child_link="flange_frame",
        xyz=[0.0, 0.0, 0.0],
        rpy=[0.0, 0.0, 0.0],
    )
    _assert_joint_transform(
        joints,
        joint_name="camera_mount_frame_joint",
        parent_link=nominal_camera["parent_frame"],
        child_link=nominal_camera["mount_frame"],
        xyz=nominal_camera["mount_xyz_m"],
        rpy=nominal_camera["mount_rpy_rad"],
    )
    _assert_joint_transform(
        joints,
        joint_name="handeye_camera_link_joint",
        parent_link=nominal_camera["parent_frame"],
        child_link=nominal_camera["child_frame"],
        xyz=nominal_camera["xyz_m"],
        rpy=nominal_camera["rpy_rad"],
    )
    _assert_joint_transform(
        joints,
        joint_name="handeye_camera_optical_frame_joint",
        parent_link=nominal_camera["child_frame"],
        child_link=nominal_camera["optical_frame"],
        xyz=nominal_camera["optical_xyz_m"],
        rpy=nominal_camera["optical_rpy_rad"],
    )
    _assert_joint_transform(
        joints,
        joint_name="gripper_mount_frame_joint",
        parent_link=gripper_nominal["parent_frame"],
        child_link=gripper_nominal["mount_frame"],
        xyz=gripper_nominal["mount_xyz_m"],
        rpy=gripper_nominal["mount_rpy_rad"],
    )
    _assert_joint_transform(
        joints,
        joint_name="gripper_base_frame_joint",
        parent_link=gripper_nominal["mount_frame"],
        child_link=gripper_nominal["base_frame"],
        xyz=gripper_nominal["xyz_m"],
        rpy=gripper_nominal["rpy_rad"],
    )
    _assert_joint_transform(
        joints,
        joint_name="gripper_tcp_nominal_joint",
        parent_link=gripper_nominal["base_frame"],
        child_link=gripper_nominal["tcp_frame"],
        xyz=gripper_nominal["tcp_xyz_m"],
        rpy=gripper_nominal["tcp_rpy_rad"],
    )

    check_output = _run(["check_urdf", str(TMP_URDF)])
    launch_entities = None
    try:
        launch_module = _load_launch_module()
        launch_description = launch_module.generate_launch_description()
        launch_entities = len(launch_description.entities)
    except ModuleNotFoundError as exc:
        print(f"skip_launch_validation={exc}")

    print(check_output.strip())
    print(f"validated_links={sorted(REQUIRED_LINKS)}")
    print("validated_joint_origins=ok")
    if launch_entities is not None:
        print(f"validated_launch_entities={launch_entities}")
    print(f"validated_config={CONFIG_PATH}")
    print(f"validated_xacro={XACRO_PATH}")
    print(f"generated_urdf={TMP_URDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
