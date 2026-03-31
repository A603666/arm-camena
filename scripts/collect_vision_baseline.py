#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Sample:
    status: str
    latency_ms: float | None


def _get_json(base_url: str, path: str, timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"response from {path} is not a JSON object")
    return payload


def _latency_from_payload(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        frame = metrics.get("frame")
        if isinstance(frame, dict) and frame.get("total_ms") is not None:
            try:
                return float(frame.get("total_ms"))
            except (TypeError, ValueError):
                return None

    timing = payload.get("timing")
    if isinstance(timing, dict):
        infer_ms = timing.get("infer_ms")
        geometry_ms = timing.get("geometry_total_ms")
        total = 0.0
        used = False
        if infer_ms is not None:
            try:
                total += float(infer_ms)
                used = True
            except (TypeError, ValueError):
                pass
        if geometry_ms is not None:
            try:
                total += float(geometry_ms)
                used = True
            except (TypeError, ValueError):
                pass
        if used:
            return float(total)
    return None


def _compute_report(samples: list[Sample], *, slow_frame_ms: float) -> dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "loss_rate": None,
            "invalid_depth_rate": None,
            "slow_frame_ratio": None,
            "latency_p95_ms": None,
            "status_counts": {},
        }

    statuses = [item.status for item in samples]
    status_counts: dict[str, int] = {}
    for status in statuses:
        status_counts[status] = status_counts.get(status, 0) + 1

    loss_count = sum(1 for item in samples if item.status in {"lost", "no_target"})
    invalid_depth_count = sum(1 for item in samples if item.status in {"invalid_depth", "unstable"})
    latencies = [float(item.latency_ms) for item in samples if item.latency_ms is not None]
    slow_count = sum(1 for value in latencies if value >= float(slow_frame_ms))
    latency_p95 = None
    if latencies:
        latency_p95 = float(np.percentile(np.asarray(latencies, dtype=np.float32), 95))

    return {
        "samples": len(samples),
        "loss_rate": round(float(loss_count / len(samples)), 4),
        "invalid_depth_rate": round(float(invalid_depth_count / len(samples)), 4),
        "slow_frame_ratio": None if not latencies else round(float(slow_count / len(latencies)), 4),
        "latency_p95_ms": None if latency_p95 is None else round(float(latency_p95), 2),
        "status_counts": status_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect baseline metrics from vision service /api/latest-target")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000", help="Vision service base URL")
    parser.add_argument("--duration-sec", type=float, default=30.0, help="Sampling duration in seconds")
    parser.add_argument("--interval-sec", type=float, default=0.1, help="Sampling interval in seconds")
    parser.add_argument("--timeout-sec", type=float, default=1.0, help="HTTP timeout in seconds")
    parser.add_argument("--slow-frame-ms", type=float, default=500.0, help="Threshold for slow frame ratio")
    parser.add_argument("--output", default="", help="Optional JSON output file path")
    args = parser.parse_args()

    duration_sec = max(1.0, float(args.duration_sec))
    interval_sec = max(0.02, float(args.interval_sec))
    timeout_sec = max(0.1, float(args.timeout_sec))

    deadline = time.time() + duration_sec
    samples: list[Sample] = []
    health_errors = 0
    latest_errors = 0
    while time.time() < deadline:
        try:
            _get_json(args.base_url, "/api/health", timeout_sec=timeout_sec)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            health_errors += 1

        try:
            payload = _get_json(args.base_url, "/api/latest-target", timeout_sec=timeout_sec)
            status = str(payload.get("status", "unknown")).strip().lower() or "unknown"
            samples.append(Sample(status=status, latency_ms=_latency_from_payload(payload)))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            latest_errors += 1
        time.sleep(interval_sec)

    report = _compute_report(samples, slow_frame_ms=float(args.slow_frame_ms))
    report["base_url"] = args.base_url.rstrip("/")
    report["duration_sec"] = duration_sec
    report["interval_sec"] = interval_sec
    report["health_errors"] = int(health_errors)
    report["latest_target_errors"] = int(latest_errors)
    report["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            fp.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
