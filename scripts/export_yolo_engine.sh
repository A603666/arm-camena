#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${ROOT_DIR}/camera_runtime/yolo26n.pt"
ENDPOINT="${DABAI_SUB_ENDPOINT:-tcp://127.0.0.1:5557}"
TOPIC="${DABAI_SUB_TOPIC:-frames.rgbd.v1}"
PROBE_TIMEOUT_SEC="3.0"
IMGSZ="auto"
BATCH="1"
HALF="true"
DYNAMIC="false"
DEVICE="0"
PROBE_ONLY="false"
SKIP_PROBE="false"

usage() {
    cat <<'EOF'
用法:
  ./scripts/export_yolo_engine.sh [选项]

功能:
  1) 查询相机数据流 meta 参数（rgb/depth 分辨率、内参、depth_scale）
  2) 按参数自动推导导出 imgsz（可手工覆盖）
  3) 导出 YOLO .pt -> TensorRT .engine

选项:
  --model PATH            .pt 模型路径（默认: camera_runtime/yolo26n.pt）
  --endpoint URL          ZMQ 订阅端点（默认: tcp://127.0.0.1:5557）
  --topic NAME            ZMQ 订阅主题（默认: frames.rgbd.v1）
  --probe-timeout SEC     探测超时秒（默认: 3.0）
  --imgsz N|auto          导出 imgsz（默认: auto）
  --batch N               导出 batch（默认: 1）
  --half true|false       是否 FP16（默认: true）
  --dynamic true|false    是否动态尺寸（默认: false）
  --device ID             导出设备（默认: 0）
  --probe-only            仅探测并打印流参数，不导出
  --skip-probe            跳过流探测（仅在 --imgsz 非 auto 时建议使用）
  -h, --help              显示帮助

示例:
  ./scripts/export_yolo_engine.sh --probe-only
  ./scripts/export_yolo_engine.sh
  ./scripts/export_yolo_engine.sh --imgsz 512 --half true --dynamic false --batch 1
EOF
}

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[export] missing command: ${cmd}" >&2
        exit 1
    fi
}

to_bool() {
    local raw="${1:-}"
    local lowered
    lowered="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
    case "${lowered}" in
        1|true|yes|on) echo "true" ;;
        0|false|no|off) echo "false" ;;
        *)
            echo "[export] invalid boolean: ${raw}" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --probe-timeout)
            PROBE_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --half)
            HALF="$(to_bool "$2")"
            shift 2
            ;;
        --dynamic)
            DYNAMIC="$(to_bool "$2")"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --probe-only)
            PROBE_ONLY="true"
            shift
            ;;
        --skip-probe)
            SKIP_PROBE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[export] unknown arg: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_command python3
require_command yolo

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[export] model not found: ${MODEL_PATH}" >&2
    exit 1
fi

probe_meta_json() {
    python3 - "$ENDPOINT" "$TOPIC" "$PROBE_TIMEOUT_SEC" <<'PY'
import json
import sys
import time

try:
    import zmq
except Exception as exc:
    print(f"PROBE_ERROR: import zmq failed: {exc}")
    raise SystemExit(0)

endpoint = sys.argv[1]
topic = sys.argv[2]
timeout_sec = float(sys.argv[3])

ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.RCVHWM, 1)
sub.setsockopt_string(zmq.SUBSCRIBE, topic)
sub.connect(endpoint)
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)

deadline = time.time() + max(0.1, timeout_sec)
meta = None
while time.time() < deadline:
    events = dict(poller.poll(200))
    if sub not in events:
        continue
    parts = sub.recv_multipart()
    if len(parts) < 2:
        continue
    try:
        meta = json.loads(parts[1].decode("utf-8"))
        break
    except Exception:
        continue

sub.close(0)
if meta is None:
    print("NO_STREAM_DATA")
else:
    print(json.dumps(meta, ensure_ascii=False))
PY
}

calc_auto_imgsz() {
    local meta_json="$1"
    python3 - "$meta_json" <<'PY'
import json
import math
import sys

meta = json.loads(sys.argv[1])
rgb_w = int(meta.get("rgb_w", 0) or 0)
rgb_h = int(meta.get("rgb_h", 0) or 0)

if rgb_w <= 0 or rgb_h <= 0:
    print("512")
    raise SystemExit(0)

base = min(rgb_w, rgb_h)
aligned = max(320, min(1280, (base // 32) * 32))
if aligned < 320:
    aligned = 320
print(str(aligned))
PY
}

validate_imgsz() {
    local imgsz="$1"
    if ! [[ "${imgsz}" =~ ^[0-9]+$ ]]; then
        echo "[export] imgsz must be integer or auto, got: ${imgsz}" >&2
        exit 1
    fi
    if (( imgsz < 320 || imgsz > 1280 )); then
        echo "[export] imgsz out of range [320, 1280]: ${imgsz}" >&2
        exit 1
    fi
}

validate_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[export] ${name} must be integer, got: ${value}" >&2
        exit 1
    fi
}

validate_int "batch" "${BATCH}"

probe_result=""
stream_meta=""
if [[ "${SKIP_PROBE}" != "true" ]]; then
    probe_result="$(probe_meta_json)"
    if [[ "${probe_result}" == PROBE_ERROR:* ]]; then
        echo "[export] ${probe_result}" >&2
    elif [[ "${probe_result}" == "NO_STREAM_DATA" ]]; then
        echo "[export] no live stream data from ${ENDPOINT} topic=${TOPIC}" >&2
    else
        stream_meta="${probe_result}"
        echo "[export] stream meta: ${stream_meta}"
    fi
fi

if [[ "${PROBE_ONLY}" == "true" ]]; then
    if [[ -n "${stream_meta}" ]]; then
        auto_imgsz="$(calc_auto_imgsz "${stream_meta}")"
        echo "[export] recommended imgsz=${auto_imgsz} (from stream rgb size)"
        exit 0
    fi
    echo "[export] probe-only requested but no stream data received." >&2
    exit 2
fi

if [[ "${IMGSZ}" == "auto" ]]; then
    if [[ -n "${stream_meta}" ]]; then
        EXPORT_IMGSZ="$(calc_auto_imgsz "${stream_meta}")"
        echo "[export] auto imgsz=${EXPORT_IMGSZ}"
    else
        EXPORT_IMGSZ="512"
        echo "[export] auto imgsz fallback to ${EXPORT_IMGSZ} (no stream meta)"
    fi
else
    validate_imgsz "${IMGSZ}"
    EXPORT_IMGSZ="${IMGSZ}"
fi

export_dir="$(cd "$(dirname "${MODEL_PATH}")" && pwd)"
model_name="$(basename "${MODEL_PATH}")"
model_stem="${model_name%.*}"
expected_engine="${export_dir}/${model_stem}.engine"

cmd=(
    yolo export
    "model=${MODEL_PATH}"
    format=engine
    "half=${HALF}"
    "dynamic=${DYNAMIC}"
    "batch=${BATCH}"
    "imgsz=${EXPORT_IMGSZ}"
    "device=${DEVICE}"
)

echo "[export] running: ${cmd[*]}"
"${cmd[@]}"

if [[ -f "${expected_engine}" ]]; then
    echo "[export] done: ${expected_engine}"
else
    candidate="$(ls -1t "${export_dir}"/*.engine 2>/dev/null | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        echo "[export] done: ${candidate}"
    else
        echo "[export] export finished but .engine not found under ${export_dir}" >&2
        exit 1
    fi
fi

