#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLISHER_DIR="${ROOT_DIR}/camera_runtime/publisher"
BUILD_DIR="${PUBLISHER_DIR}/build"
VISION_DIR="${ROOT_DIR}/camera_runtime/vision_service"
SDK_LIB_DIR="${ROOT_DIR}/vendor/OrbbecSDK/lib/arm64"
MODEL_PATH="${ROOT_DIR}/camera_runtime/yolo26n.pt"

publisher_pid=""
service_pid=""

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[vision] missing command: ${cmd}" >&2
        exit 1
    fi
}

cleanup() {
    local code="${1:-0}"
    trap - INT TERM

    for pid in "${service_pid}" "${publisher_pid}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${service_pid}" "${publisher_pid}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done

    exit "${code}"
}

check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
    shift
fi

trap 'echo "[vision] signal received, stopping processes..."; cleanup 130' INT TERM

require_command cmake
require_command pkg-config
require_command python3

arch="$(uname -m)"
if [[ "${arch}" != "aarch64" && "${arch}" != "arm64" ]]; then
    echo "[vision] this launcher targets Linux ARM64/Jetson. Current arch: ${arch}" >&2
    exit 1
fi

if [[ ! -d "${SDK_LIB_DIR}" ]]; then
    echo "[vision] Orbbec SDK ARM64 libs not found: ${SDK_LIB_DIR}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[vision] YOLO model not found: ${MODEL_PATH}" >&2
    exit 1
fi

if ! pkg-config --exists opencv4; then
    echo "[vision] OpenCV development files not found. Install: sudo apt install libopencv-dev" >&2
    exit 1
fi

if ! pkg-config --exists libzmq; then
    echo "[vision] ZeroMQ development files not found. Install: sudo apt install libzmq3-dev" >&2
    exit 1
fi

if ! python3 - <<'PY'
import importlib.util
import sys

modules = ["fastapi", "uvicorn", "numpy", "cv2", "zmq", "sklearn", "torch", "ultralytics"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    print("[vision] missing Python modules: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
then
    echo "[vision] install camera_runtime/vision_service/requirements.txt and prepare Jetson torch first." >&2
    exit 1
fi

export DABAI_YOLO_MODEL="${DABAI_YOLO_MODEL:-${MODEL_PATH}}"
export DABAI_WEB_PORT="${DABAI_WEB_PORT:-18000}"
export DABAI_YOLO_DEVICE="${DABAI_YOLO_DEVICE:-cuda:0}"
export LD_LIBRARY_PATH="${SDK_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/camera_runtime:${PYTHONPATH:-}"

if [[ "${check_only}" == "true" ]]; then
    echo "[vision] check passed"
    echo "  root=${ROOT_DIR}"
    echo "  model=${DABAI_YOLO_MODEL}"
    echo "  sdk_lib=${SDK_LIB_DIR}"
    echo "  web_port=${DABAI_WEB_PORT}"
    exit 0
fi

build_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo "[vision] building ARM64 publisher..."
cmake -S "${PUBLISHER_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release -j"${build_jobs}"

publisher_bin="${BUILD_DIR}/dabai_frame_publisher"
if [[ ! -x "${publisher_bin}" ]]; then
    echo "[vision] publisher executable not found: ${publisher_bin}" >&2
    exit 1
fi

echo "[vision] launching publisher: ${publisher_bin}"
"${publisher_bin}" &
publisher_pid=$!

echo "[vision] launching Python vision service..."
(
    cd "${VISION_DIR}"
    python3 run_service.py
) &
service_pid=$!

echo
echo "Publisher PID: ${publisher_pid}"
echo "Vision Service PID: ${service_pid}"
echo "Dashboard: http://127.0.0.1:${DABAI_WEB_PORT}/"

set +e
wait -n "${publisher_pid}" "${service_pid}"
status=$?
set -e

echo "[vision] one process exited, stopping the other..."
cleanup "${status}"
