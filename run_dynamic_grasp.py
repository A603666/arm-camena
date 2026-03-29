#!/usr/bin/env python3
"""Dynamic grasp integration entrypoint for the 集成测试 workspace."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dynamic_grasp.bridge import RobotBridge
from dynamic_grasp.config import AppConfig, load_app_config
from dynamic_grasp.controller import DynamicGraspController
from dynamic_grasp.math_utils import HandEyeModel
from dynamic_grasp.vision_client import VisionClient


def default_config_path() -> Path:
    unified = THIS_DIR / "pipeline_config.yaml"
    if unified.is_file():
        return unified
    return THIS_DIR / "dynamic_grasp_config.yaml"


def setup_logger(config: AppConfig) -> tuple[logging.Logger, Path]:
    log_dir = config.runtime.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dynamic_grasp_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("dynamic_grasp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic grasp integration runner")
    parser.add_argument("--config", default=str(default_config_path()), help="Path to the dynamic grasp YAML config")
    parser.add_argument("--mode", choices=["step", "auto"], default="step", help="Run with gated prompts or fully automatic")
    parser.add_argument("--once", action="store_true", help="Run a single detect-grasp-recover cycle, then exit")
    parser.add_argument("--vision-url", default=None, help="Override the vision service base URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_app_config(args.config)
    if args.vision_url:
        config = replace(config, vision=replace(config.vision, base_url=str(args.vision_url).rstrip("/")))

    logger, log_path = setup_logger(config)
    print(f"log_file={log_path}")
    logger.info("loaded config=%s mode=%s once=%s vision_url=%s", config.config_path, args.mode, args.once, config.vision.base_url)

    handeye = HandEyeModel.from_yaml(config.handeye.extrinsics_path, mode=config.handeye.mode)
    robot = RobotBridge(config.runtime.arm_config_path, config.runtime.move_timeout_sec, logger)
    vision = VisionClient(config.vision.base_url, timeout_sec=config.vision.health_timeout_sec)
    controller = DynamicGraspController(
        config=config,
        robot=robot,
        vision_client=vision,
        handeye=handeye,
        logger=logger,
        mode=args.mode,
    )

    try:
        return controller.run(once=args.once)
    finally:
        controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
