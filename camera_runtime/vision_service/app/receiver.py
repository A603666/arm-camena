from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import zmq

from .types import FramePacket


class ZmqFrameReceiver:
    def __init__(self, endpoint: str, topic: str, timeout_ms: int) -> None:
        self._topic = topic
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.connect(endpoint)

    def close(self) -> None:
        self._socket.close(0)

    def recv(self) -> FramePacket | None:
        try:
            parts = self._socket.recv_multipart()
        except zmq.Again:
            return None

        if len(parts) != 4:
            return None

        topic_bytes, meta_bytes, rgb_payload, depth_payload = parts
        if topic_bytes.decode("utf-8", errors="ignore") != self._topic:
            return None

        try:
            meta: dict[str, Any] = json.loads(meta_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            return None

        rgb = cv2.imdecode(np.frombuffer(rgb_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            return None

        depth_h = int(meta.get("depth_h", 0))
        depth_w = int(meta.get("depth_w", 0))
        if depth_h <= 0 or depth_w <= 0:
            return None

        depth = np.frombuffer(depth_payload, dtype=np.uint16)
        expected = depth_h * depth_w
        if depth.size != expected:
            return None
        depth = depth.reshape((depth_h, depth_w))

        return FramePacket(meta=meta, rgb=rgb, depth=depth)
