#!/usr/bin/env python3
"""NERO auto-enable daemon.

State machine:
1) wait CAN device by USB bus-info
2) activate CAN via official can_activate.sh
3) connect pyAgxArm (NERO/CAN)
4) set_normal_mode() to enable CAN push
5) enable all joints
6) health check loop; recover on any fault

This daemon intentionally does NOT send any motion commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent

TAG_CAN_UP = "CAN_UP"
TAG_SET_NORMAL_MODE_OK = "SET_NORMAL_MODE_OK"
TAG_ENABLE_OK = "ENABLE_OK"
TAG_RECOVERING = "RECOVERING"
TAG_RECOVERED = "RECOVERED"
TAG_ENABLE_TIMEOUT = "ENABLE_TIMEOUT"

REQUIRED_KEYS = [
    "can_channel",
    "can_bitrate",
    "can_interface",
    "usb_bus_info",
    "can_scripts_dir",
    "retry_interval_sec",
    "enable_timeout_sec",
    "health_check_sec",
]


@dataclass
class DaemonConfig:
    can_channel: str
    can_bitrate: int
    can_interface: str
    usb_bus_info: str
    can_scripts_dir: Path
    retry_interval_sec: float
    enable_timeout_sec: float
    health_check_sec: float


def log_tag(logger: logging.Logger, tag: str, message: str, level: int = logging.INFO) -> None:
    logger.log(level, "[%s] %s", tag, message)


def run_cmd(cmd: list[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def load_config(path: Path) -> DaemonConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")

    can_channel = str(raw["can_channel"]).strip()
    can_bitrate = int(raw["can_bitrate"])
    can_interface = str(raw["can_interface"]).strip()
    usb_bus_info = str(raw["usb_bus_info"]).strip()
    can_scripts_dir = Path(str(raw["can_scripts_dir"])).expanduser()
    if not can_scripts_dir.is_absolute():
        can_scripts_dir = (path.parent / can_scripts_dir).resolve()
    else:
        can_scripts_dir = can_scripts_dir.resolve()
    retry_interval_sec = float(raw["retry_interval_sec"])
    enable_timeout_sec = float(raw["enable_timeout_sec"])
    health_check_sec = float(raw["health_check_sec"])

    if not can_channel:
        raise ValueError("can_channel must not be empty")
    if can_bitrate <= 0:
        raise ValueError("can_bitrate must be > 0")
    if not can_interface:
        raise ValueError("can_interface must not be empty")
    if not usb_bus_info:
        raise ValueError("usb_bus_info must not be empty")
    if retry_interval_sec <= 0:
        raise ValueError("retry_interval_sec must be > 0")
    if enable_timeout_sec <= 0:
        raise ValueError("enable_timeout_sec must be > 0")
    if health_check_sec <= 0:
        raise ValueError("health_check_sec must be > 0")

    can_activate = can_scripts_dir / "can_activate.sh"
    if not can_activate.is_file():
        raise ValueError(f"can_activate.sh not found: {can_activate}")

    return DaemonConfig(
        can_channel=can_channel,
        can_bitrate=can_bitrate,
        can_interface=can_interface,
        usb_bus_info=usb_bus_info,
        can_scripts_dir=can_scripts_dir,
        retry_interval_sec=retry_interval_sec,
        enable_timeout_sec=enable_timeout_sec,
        health_check_sec=health_check_sec,
    )


def inject_pyagxarm_path(cfg: DaemonConfig, logger: logging.Logger) -> None:
    candidates: list[Path] = []

    env_repo = os.environ.get("PYAGXARM_REPO", "").strip()
    if env_repo:
        repo = Path(env_repo).expanduser().resolve()
        candidates.extend([repo, repo / "pyAgxArm"])

    for parent in cfg.can_scripts_dir.parents:
        if (parent / "pyAgxArm").is_dir():
            candidates.extend([parent, parent / "pyAgxArm"])
            break

    added: list[str] = []
    for cand in candidates:
        if cand.is_dir():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
                added.append(s)

    if added:
        logger.info("pyAgxArm sys.path injected: %s", ", ".join(added))


def get_can_interfaces() -> list[str]:
    rc, out, _ = run_cmd(["ip", "-br", "link", "show", "type", "can"])
    if rc != 0 or not out:
        return []
    names: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def get_bus_info(iface: str) -> Optional[str]:
    rc, out, _ = run_cmd(["ethtool", "-i", iface])
    if rc != 0:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("bus-info:"):
            return line.split(":", 1)[1].strip()
    return None


def find_can_iface_by_usb(usb_bus_info: str) -> Optional[str]:
    for iface in get_can_interfaces():
        bus = get_bus_info(iface)
        if bus == usb_bus_info:
            return iface
    return None


def is_can_up(channel: str, expect_bitrate: int) -> bool:
    rc, out, _ = run_cmd(["ip", "-br", "link", "show", channel])
    if rc != 0:
        return False
    if " UP " not in f" {out} ":
        return False

    rc2, out2, _ = run_cmd(["ip", "-details", "link", "show", channel])
    if rc2 != 0:
        return False

    m = re.search(r"\bbitrate\s+(\d+)\b", out2)
    if not m:
        return False
    return int(m.group(1)) == int(expect_bitrate)


def activate_can(cfg: DaemonConfig, logger: logging.Logger) -> bool:
    can_activate = cfg.can_scripts_dir / "can_activate.sh"
    cmd = [
        "bash",
        str(can_activate),
        cfg.can_channel,
        str(cfg.can_bitrate),
        cfg.usb_bus_info,
    ]
    rc, out, err = run_cmd(cmd, timeout=40.0)
    if rc != 0:
        logger.error("can_activate failed rc=%s stdout=%s stderr=%s", rc, out, err)
        return False

    if not is_can_up(cfg.can_channel, cfg.can_bitrate):
        logger.error(
            "can link check failed after activate: channel=%s bitrate=%s",
            cfg.can_channel,
            cfg.can_bitrate,
        )
        return False

    log_tag(
        logger,
        TAG_CAN_UP,
        f"channel={cfg.can_channel} bitrate={cfg.can_bitrate} usb={cfg.usb_bus_info}",
    )
    return True


def safe_disconnect(robot: Any, logger: logging.Logger) -> None:
    if robot is None:
        return
    if hasattr(robot, "disconnect"):
        try:
            robot.disconnect()
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("robot.disconnect() failed: %s", exc)


def get_joint_flags(robot: Any) -> Optional[list[bool]]:
    if not hasattr(robot, "get_joints_enable_status_list"):
        return None
    try:
        raw = robot.get_joints_enable_status_list()
    except Exception:
        return None
    if isinstance(raw, list) and len(raw) >= 7:
        return [bool(v) for v in raw[:7]]
    return None


def ensure_enabled(robot: Any, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        flags = get_joint_flags(robot)
        if flags is not None and all(flags):
            return True
        try:
            if hasattr(robot, "enable") and robot.enable():
                return True
        except Exception:
            pass
        time.sleep(0.2)
    flags = get_joint_flags(robot)
    return bool(flags is not None and all(flags))


def establish_robot_session(cfg: DaemonConfig, logger: logging.Logger) -> Tuple[bool, Optional[Any], str]:
    iface = find_can_iface_by_usb(cfg.usb_bus_info)
    if iface is None:
        return False, None, f"target USB CAN not found: usb_bus_info={cfg.usb_bus_info}"

    if not activate_can(cfg, logger):
        return False, None, "failed to activate CAN"

    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
    except Exception as exc:
        return False, None, f"pyAgxArm import failed: {exc}"

    try:
        arm_cfg = create_agx_arm_config(
            robot="nero",
            comm="can",
            channel=cfg.can_channel,
            interface=cfg.can_interface,
            bitrate=cfg.can_bitrate,
        )
        robot = AgxArmFactory.create_arm(arm_cfg)
        robot.connect()
    except Exception as exc:
        return False, None, f"robot connect failed: {exc}"

    if not hasattr(robot, "set_normal_mode"):
        safe_disconnect(robot, logger)
        return False, None, "set_normal_mode() not available in this SDK"

    try:
        robot.set_normal_mode()
        log_tag(logger, TAG_SET_NORMAL_MODE_OK, "set_normal_mode() completed")
    except Exception as exc:
        safe_disconnect(robot, logger)
        return False, None, f"set_normal_mode failed: {exc}"

    if not ensure_enabled(robot, cfg.enable_timeout_sec):
        log_tag(
            logger,
            TAG_ENABLE_TIMEOUT,
            f"joint enable timeout after {cfg.enable_timeout_sec:.1f}s",
            level=logging.ERROR,
        )
        safe_disconnect(robot, logger)
        return False, None, "joint enable timeout"

    log_tag(logger, TAG_ENABLE_OK, "all joints enabled")
    return True, robot, ""


def health_check(robot: Any, cfg: DaemonConfig, logger: logging.Logger) -> Tuple[bool, str]:
    if not is_can_up(cfg.can_channel, cfg.can_bitrate):
        return False, "can link is down or bitrate mismatch"

    try:
        joints = robot.get_joint_angles() if hasattr(robot, "get_joint_angles") else None
        if joints is None or not hasattr(joints, "msg"):
            return False, "no joint feedback"
    except Exception as exc:
        return False, f"joint feedback exception: {exc}"

    flags = get_joint_flags(robot)
    if flags is not None and not all(flags):
        if ensure_enabled(robot, min(cfg.enable_timeout_sec, 3.0)):
            log_tag(logger, TAG_ENABLE_OK, "re-enable success during health check")
            return True, ""
        return False, "joint motors not fully enabled"

    return True, ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NERO auto CAN push + auto-enable daemon")
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "config" / "auto_enable.yaml"),
        help="Path to daemon config yaml",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one establish cycle and exit (for validation/troubleshooting)",
    )
    return parser


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("nero_auto_enable")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logging()

    cfg_path = Path(args.config).expanduser().resolve()
    try:
        cfg = load_config(cfg_path)
    except Exception as exc:
        logger.error("load config failed: %s", exc)
        return 2

    inject_pyagxarm_path(cfg, logger)

    logger.info(
        "daemon start once=%s channel=%s bitrate=%s usb=%s",
        args.once,
        cfg.can_channel,
        cfg.can_bitrate,
        cfg.usb_bus_info,
    )

    recovering = False

    while True:
        ok, robot, reason = establish_robot_session(cfg, logger)
        if not ok:
            log_tag(logger, TAG_RECOVERING, reason, level=logging.WARNING)
            if args.once:
                return 1
            time.sleep(cfg.retry_interval_sec)
            recovering = True
            continue

        if recovering:
            log_tag(logger, TAG_RECOVERED, "session restored")
            recovering = False

        if args.once:
            safe_disconnect(robot, logger)
            return 0

        while True:
            healthy, reason = health_check(robot, cfg, logger)
            if not healthy:
                safe_disconnect(robot, logger)
                log_tag(logger, TAG_RECOVERING, reason, level=logging.WARNING)
                recovering = True
                break
            time.sleep(cfg.health_check_sec)


if __name__ == "__main__":
    raise SystemExit(main())
