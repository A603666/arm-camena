#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLISHER_DIR="${ROOT_DIR}/camera_runtime/publisher"
BUILD_DIR="${PUBLISHER_DIR}/build"
VISION_DIR="${ROOT_DIR}/camera_runtime/vision_service"
SDK_LIB_DIR="${ROOT_DIR}/vendor/OrbbecSDK/lib/arm64"
MODEL_DIR="${ROOT_DIR}/camera_runtime"
DEFAULT_PT_MODEL="${MODEL_DIR}/yolo26n.pt"
CUSPARSELT_LIB_DIR=""
TORCH_LIB_DIR=""
DEFAULT_UNIFIED_CONFIG="${ROOT_DIR}/pipeline_config.yaml"

publisher_pid=""
service_pid=""
cleanup_done=0
using_unified_config=false
auto_enable_service_name="nero-auto-enable.service"
auto_enable_service_was_active=0
auto_enable_service_paused=0

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[vision] missing command: ${cmd}" >&2
        exit 1
    fi
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

ensure_sudo_session() {
    if [[ "${EUID}" -eq 0 ]]; then
        return 0
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "[vision] missing command: sudo" >&2
        return 1
    fi
    if ! sudo -v; then
        echo "[vision] sudo auth failed" >&2
        return 1
    fi
    return 0
}

run_systemctl_with_privilege() {
    local action="$1"
    local unit="$2"
    if [[ "${EUID}" -eq 0 ]]; then
        systemctl "${action}" "${unit}"
    else
        ensure_sudo_session || return 1
        sudo systemctl "${action}" "${unit}"
    fi
}

load_unified_vision_env_if_needed() {
    local config_path="$1"
    local loader_path="${ROOT_DIR}/scripts/unified_config_loader.py"
    local detected=""
    local line key value

    if [[ ! -f "${config_path}" ]]; then
        return 0
    fi
    if [[ ! -f "${loader_path}" ]]; then
        return 0
    fi

    detected="$(python3 "${loader_path}" --config "${config_path}" --mode detect 2>/dev/null || true)"
    if [[ "${detected}" != "unified" ]]; then
        return 0
    fi

    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        if [[ "${line}" != *=* ]]; then
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        if [[ -z "${key}" ]]; then
            continue
        fi
        if [[ -z "${!key+x}" ]]; then
            export "${key}=${value}"
        fi
    done < <(python3 "${loader_path}" --config "${config_path}" --mode vision-env)

    using_unified_config=true
    echo "[vision] loaded unified config: ${config_path}"
}

resolve_auto_enable_config_for_daemon() {
    local requested_config="$1"
    local loader_path="${ROOT_DIR}/scripts/unified_config_loader.py"
    local fallback_config="${ROOT_DIR}/robot_runtime/config/auto_enable.yaml"

    if [[ -f "${requested_config}" && -f "${loader_path}" ]]; then
        if python3 "${loader_path}" --config "${requested_config}" --mode component --component auto_enable >/dev/null 2>&1; then
            printf '%s\n' "${requested_config}"
            return 0
        fi
    fi

    printf '%s\n' "${fallback_config}"
}

pause_auto_enable_daemon_for_exclusive_control() {
    if [[ "${DABAI_ROBOT_CONTROL_ENABLED}" != "1" ]]; then
        export DABAI_ROBOT_EXCLUSIVE_CONTROL="${DABAI_ROBOT_EXCLUSIVE_CONTROL:-0}"
        return 0
    fi

    export DABAI_ROBOT_EXCLUSIVE_CONTROL="0"

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "[vision] systemctl not available; skip daemon isolation and keep non-exclusive robot control."
        return 0
    fi

    if ! systemctl is-active --quiet "${auto_enable_service_name}"; then
        export DABAI_ROBOT_EXCLUSIVE_CONTROL="1"
        return 0
    fi

    auto_enable_service_was_active=1
    echo "[vision] ${auto_enable_service_name} is active; stopping for exclusive robot control..."
    if ! run_systemctl_with_privilege stop "${auto_enable_service_name}"; then
        echo "[vision] failed to stop ${auto_enable_service_name}; abort to avoid control conflict." >&2
        exit 1
    fi
    if systemctl is-active --quiet "${auto_enable_service_name}"; then
        echo "[vision] ${auto_enable_service_name} is still active after stop; abort to avoid control conflict." >&2
        exit 1
    fi

    auto_enable_service_paused=1
    export DABAI_ROBOT_EXCLUSIVE_CONTROL="1"
    echo "[vision] ${auto_enable_service_name} paused; exclusive robot control enabled for this session."
}

restore_auto_enable_daemon_if_needed() {
    if [[ "${auto_enable_service_paused}" != "1" ]]; then
        return 0
    fi
    auto_enable_service_paused=0

    if [[ "${auto_enable_service_was_active}" != "1" ]]; then
        return 0
    fi
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "[vision] WARNING: cannot restore ${auto_enable_service_name} (systemctl missing)." >&2
        return 0
    fi

    echo "[vision] restoring ${auto_enable_service_name}..."
    if run_systemctl_with_privilege start "${auto_enable_service_name}"; then
        echo "[vision] restored ${auto_enable_service_name}."
    else
        echo "[vision] WARNING: failed to restore ${auto_enable_service_name}; please start it manually." >&2
    fi
}

prepare_robot_can() {
    local daemon_path="${ROOT_DIR}/robot_runtime/nero_auto_enable_daemon.py"
    local daemon_config=""
    local rc=0
    local max_attempts="${DABAI_AUTO_ENABLE_ONCE_MAX_ATTEMPTS:-3}"
    local retry_delay_sec="${DABAI_AUTO_ENABLE_ONCE_RETRY_DELAY_SEC:-2.0}"
    local attempt=1

    if [[ "${DABAI_ROBOT_CONTROL_ENABLED}" != "1" ]]; then
        echo "[vision] robot control disabled, skip robot CAN preparation."
        return 0
    fi

    if [[ ! -f "${daemon_path}" ]]; then
        echo "[vision] auto-enable daemon not found: ${daemon_path}" >&2
        exit 1
    fi

    daemon_config="$(resolve_auto_enable_config_for_daemon "${config_path}")"
    daemon_config="$(realpath -m "${daemon_config}")"
    if [[ ! -f "${daemon_config}" ]]; then
        echo "[vision] auto-enable config not found: ${daemon_config}" >&2
        exit 1
    fi

    echo "[vision] prepare robot can..."
    echo "[vision] auto-enable config: ${daemon_config}"

    if ! is_positive_int "${max_attempts}"; then
        echo "[vision] invalid DABAI_AUTO_ENABLE_ONCE_MAX_ATTEMPTS=${max_attempts} (must be >=1 integer)" >&2
        exit 2
    fi
    if ! is_nonnegative_number "${retry_delay_sec}"; then
        echo "[vision] invalid DABAI_AUTO_ENABLE_ONCE_RETRY_DELAY_SEC=${retry_delay_sec} (must be >=0 number)" >&2
        exit 2
    fi

    if command -v systemctl >/dev/null 2>&1; then
        if systemctl is-active --quiet "${auto_enable_service_name}"; then
            if [[ "${DABAI_ROBOT_EXCLUSIVE_CONTROL:-0}" == "1" ]]; then
                echo "[vision] ${auto_enable_service_name} is active while exclusive robot control is requested; abort." >&2
                exit 1
            fi
            echo "[vision] ${auto_enable_service_name} is active; skip local --once auto-enable to avoid CAN controller conflict."
            return 0
        fi
    fi

    if ! ensure_sudo_session; then
        echo "[vision] auto-enable once failed: sudo unavailable" >&2
        exit 1
    fi

    while (( attempt <= max_attempts )); do
        set +e
        python3 "${daemon_path}" --config "${daemon_config}" --once
        rc=$?
        set -e

        if [[ "${rc}" -eq 0 ]]; then
            if (( max_attempts > 1 )); then
                echo "[vision] auto-enable once passed (attempt ${attempt}/${max_attempts})"
            else
                echo "[vision] auto-enable once passed"
            fi
            return 0
        fi

        if (( attempt < max_attempts )); then
            echo "[vision] auto-enable once attempt ${attempt}/${max_attempts} failed (rc=${rc}), retry in ${retry_delay_sec}s..."
            sleep "${retry_delay_sec}"
        fi
        ((attempt++))
    done

    echo "[vision] auto-enable once failed after ${max_attempts} attempts (last rc=${rc})" >&2
    exit "${rc}"
}

resolve_default_model_path() {
    local model_dir="$1"
    local default_pt="$2"
    local engines=()

    shopt -s nullglob
    engines=("${model_dir}"/*.engine)
    shopt -u nullglob

    if (( ${#engines[@]} > 0 )); then
        printf '%s\n' "${engines[0]}"
        return
    fi
    printf '%s\n' "${default_pt}"
}

wait_for_pid_exit() {
    local pid="$1"
    local retries="${2:-30}"
    local interval_sec="${3:-0.1}"
    local i
    for ((i = 0; i < retries; ++i)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
        sleep "${interval_sec}"
    done
    return 1
}

terminate_pid_gracefully() {
    local pid="$1"
    local label="$2"
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi

    kill -TERM "${pid}" 2>/dev/null || true
    if ! wait_for_pid_exit "${pid}" 30 0.1; then
        echo "[vision] ${label} pid=${pid} did not exit after TERM, forcing KILL..."
        kill -KILL "${pid}" 2>/dev/null || true
        wait_for_pid_exit "${pid}" 20 0.1 || true
    fi
}

cleanup_children() {
    if [[ "${cleanup_done}" == "1" ]]; then
        return 0
    fi
    cleanup_done=1

    terminate_pid_gracefully "${service_pid}" "vision service"
    terminate_pid_gracefully "${publisher_pid}" "publisher"

    if [[ -n "${service_pid}" ]]; then
        wait "${service_pid}" 2>/dev/null || true
    fi
    if [[ -n "${publisher_pid}" ]]; then
        wait "${publisher_pid}" 2>/dev/null || true
    fi
}

on_exit() {
    local code="$?"
    trap - EXIT INT TERM HUP QUIT
    cleanup_children
    restore_auto_enable_daemon_if_needed || true
    return "${code}"
}

on_signal() {
    local signame="$1"
    echo "[vision] signal ${signame} received, stopping processes..."
    exit 130
}

find_project_service_pids() {
    local proc pid cwd cmdline
    for proc in /proc/[0-9]*; do
        pid="${proc##*/}"
        [[ "${pid}" == "$$" ]] && continue

        if [[ ! -r "${proc}/cmdline" ]]; then
            continue
        fi
        cmdline="$(cat "${proc}/cmdline" 2>/dev/null | tr '\0' ' ' || true)"
        [[ -z "${cmdline}" ]] && continue
        if [[ "${cmdline}" != *"run_service.py"* ]]; then
            continue
        fi

        cwd="$(readlink -f "${proc}/cwd" 2>/dev/null || true)"
        if [[ "${cwd}" != "${VISION_DIR}" && "${cmdline}" != *"${VISION_DIR}/run_service.py"* ]]; then
            continue
        fi

        printf '%s\n' "${pid}"
    done
}

find_project_publisher_pids() {
    local proc pid exe
    for proc in /proc/[0-9]*; do
        pid="${proc##*/}"
        [[ "${pid}" == "$$" ]] && continue

        exe="$(readlink -f "${proc}/exe" 2>/dev/null || true)"
        if [[ "${exe}" == "${BUILD_DIR}/dabai_frame_publisher" ]]; then
            printf '%s\n' "${pid}"
        fi
    done
}

publisher_needs_rebuild() {
    local publisher_bin="$1"
    if [[ ! -x "${publisher_bin}" ]]; then
        return 0
    fi
    if [[ ! -f "${BUILD_DIR}/Makefile" ]]; then
        return 0
    fi
    if find "${PUBLISHER_DIR}/src" "${PUBLISHER_DIR}/CMakeLists.txt" -type f -newer "${publisher_bin}" -print -quit | grep -q .; then
        return 0
    fi
    return 1
}

terminate_stale_processes() {
    local stale_service_pids=()
    local stale_publisher_pids=()
    local pid

    mapfile -t stale_service_pids < <(find_project_service_pids)
    mapfile -t stale_publisher_pids < <(find_project_publisher_pids)

    if (( ${#stale_service_pids[@]} == 0 && ${#stale_publisher_pids[@]} == 0 )); then
        return 0
    fi

    echo "[vision] found stale project processes, cleaning up..."
    for pid in "${stale_service_pids[@]}"; do
        echo "[vision] stopping stale vision service pid=${pid}"
        terminate_pid_gracefully "${pid}" "stale vision service"
    done
    for pid in "${stale_publisher_pids[@]}"; do
        echo "[vision] stopping stale publisher pid=${pid}"
        terminate_pid_gracefully "${pid}" "stale publisher"
    done

    for pid in "${stale_service_pids[@]}" "${stale_publisher_pids[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "[vision] failed to stop stale pid=${pid}" >&2
            exit 1
        fi
    done
}

check_only=false
allow_lan_robot_control=false
shadow_compare=false
pipeline_override=""
config_path="${DEFAULT_UNIFIED_CONFIG}"
config_provided=false
while (( "$#" > 0 )); do
    case "$1" in
        --check)
            check_only=true
            ;;
        --pipeline)
            if (( "$#" < 2 )); then
                echo "[vision] --pipeline requires v1 or v2" >&2
                exit 2
            fi
            pipeline_override="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            if [[ "${pipeline_override}" != "v1" && "${pipeline_override}" != "v2" ]]; then
                echo "[vision] --pipeline must be v1 or v2 (got: $2)" >&2
                exit 2
            fi
            shift
            ;;
        --shadow-compare)
            shadow_compare=true
            ;;
        --allow-lan-robot-control)
            allow_lan_robot_control=true
            ;;
        --config)
            if (( "$#" < 2 )); then
                echo "[vision] --config requires a path argument" >&2
                exit 2
            fi
            config_path="$2"
            config_provided=true
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./scripts/start_vision_arm64.sh [--check] [--pipeline v1|v2] [--shadow-compare] [--allow-lan-robot-control] [--config PATH]

  --check                      Check dependencies and resolved runtime config only.
  --pipeline v1|v2             Select vision pipeline version (env: DABAI_VISION_PIPELINE).
  --shadow-compare             Enable shadow compare mode for pipeline diagnostics.
  --allow-lan-robot-control    Allow robot control API access from LAN clients.
  --config PATH                Config path (default: ./pipeline_config.yaml if present).
EOF
            exit 0
            ;;
        *)
            echo "[vision] unknown argument: $1" >&2
            echo "[vision] supported arguments: --check, --pipeline v1|v2, --shadow-compare, --allow-lan-robot-control, --config PATH" >&2
            exit 2
            ;;
    esac
    shift
done

config_path="$(realpath -m "${config_path}")"
if [[ "${config_provided}" == "true" && ! -f "${config_path}" ]]; then
    echo "[vision] config file not found: ${config_path}" >&2
    exit 2
fi

trap on_exit EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap 'on_signal QUIT' QUIT

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

while IFS= read -r py_site_path; do
    [[ -z "${py_site_path}" ]] && continue
    candidate_cusparselt="${py_site_path}/nvidia/cusparselt/lib"
    if [[ -d "${candidate_cusparselt}" ]]; then
        CUSPARSELT_LIB_DIR="${candidate_cusparselt}"
    fi

    if [[ -z "${TORCH_LIB_DIR}" ]]; then
        candidate_torch_lib="${py_site_path}/torch/lib"
        if [[ -d "${candidate_torch_lib}" ]]; then
            TORCH_LIB_DIR="${candidate_torch_lib}"
        fi
    fi
done < <(python3 - <<'PY'
import site
import sysconfig

seen = set()
for p in (site.getusersitepackages(), sysconfig.get_paths().get("purelib", "")):
    if p and p not in seen:
        seen.add(p)
        print(p)
PY
)

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

ws_backends = ["websockets", "wsproto"]
if not any(importlib.util.find_spec(name) is not None for name in ws_backends):
    print(
        "[vision] missing WebSocket backend: install one of " + ", ".join(ws_backends),
        file=sys.stderr,
    )
    sys.exit(1)
PY
then
    echo "[vision] install camera_runtime/vision_service/requirements.txt and prepare Jetson torch first." >&2
    exit 1
fi

load_unified_vision_env_if_needed "${config_path}"
if [[ -n "${pipeline_override}" ]]; then
    export DABAI_VISION_PIPELINE="${pipeline_override}"
fi
if [[ "${shadow_compare}" == "true" ]]; then
    export DABAI_SHADOW_COMPARE="1"
fi

selected_model="${DABAI_YOLO_MODEL:-}"
if [[ -z "${selected_model}" ]]; then
    selected_model="$(resolve_default_model_path "${MODEL_DIR}" "${DEFAULT_PT_MODEL}")"
    if [[ "${selected_model}" == *.engine ]]; then
        echo "[vision] auto selected TensorRT engine: ${selected_model}"
    else
        echo "[vision] no .engine found, fallback to PyTorch model: ${selected_model}"
    fi
fi

if [[ ! -f "${selected_model}" ]]; then
    echo "[vision] YOLO model not found: ${selected_model}" >&2
    exit 1
fi

export DABAI_YOLO_MODEL="${selected_model}"
export DABAI_WEB_PORT="${DABAI_WEB_PORT:-18000}"
export DABAI_YOLO_DEVICE="${DABAI_YOLO_DEVICE:-cuda:0}"
export DABAI_YOLO_IMGSZ="${DABAI_YOLO_IMGSZ:-512}"
export DABAI_YOLO_PRECISION="${DABAI_YOLO_PRECISION:-fp32}"
export DABAI_YOLO_WARMUP="${DABAI_YOLO_WARMUP:-1}"
export DABAI_GEOM_BACKEND="${DABAI_GEOM_BACKEND:-cpu}"
export DABAI_VISION_PIPELINE="${DABAI_VISION_PIPELINE:-v2}"
export DABAI_SHADOW_COMPARE="${DABAI_SHADOW_COMPARE:-0}"
export DABAI_GEOM_PARITY_CHECK="${DABAI_GEOM_PARITY_CHECK:-0}"
export DABAI_GEOM_PARITY_EVERY_N="${DABAI_GEOM_PARITY_EVERY_N:-30}"
export DABAI_INFER_EVERY_N="${DABAI_INFER_EVERY_N:-2}"
export DABAI_GEOMETRY_EVERY_N="${DABAI_GEOMETRY_EVERY_N:-1}"
export DABAI_MAX_POINTS_FOR_GEOMETRY="${DABAI_MAX_POINTS_FOR_GEOMETRY:-12000}"
export DABAI_GROUND_FIT_FAST_ENABLED="${DABAI_GROUND_FIT_FAST_ENABLED:-1}"
export DABAI_GROUND_FIT_FAST_SAMPLE_CAP="${DABAI_GROUND_FIT_FAST_SAMPLE_CAP:-2500}"
export DABAI_GROUND_FIT_FAST_MAX_TRIALS="${DABAI_GROUND_FIT_FAST_MAX_TRIALS:-40}"
export DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO="${DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO:-0.55}"
export DABAI_GROUND_FIT_FAST_MIN_INLIERS="${DABAI_GROUND_FIT_FAST_MIN_INLIERS:-80}"
export DABAI_COLOR_FPS="${DABAI_COLOR_FPS:-15}"
export DABAI_DEPTH_FPS="${DABAI_DEPTH_FPS:-30}"
export DABAI_ALIGN_MODE="${DABAI_ALIGN_MODE:-auto}"
export DABAI_FRAME_SYNC="${DABAI_FRAME_SYNC:-0}"
export DABAI_OB_LOG_LEVEL="${DABAI_OB_LOG_LEVEL:-error}"
export DABAI_UVICORN_LOG_LEVEL="${DABAI_UVICORN_LOG_LEVEL:-warning}"
export DABAI_UVICORN_ACCESS_LOG="${DABAI_UVICORN_ACCESS_LOG:-0}"
export DABAI_SKIP_BUILD="${DABAI_SKIP_BUILD:-1}"
export DABAI_FORCE_REBUILD="${DABAI_FORCE_REBUILD:-0}"
export DABAI_TARGET_LOCK_IOU_MIN="${DABAI_TARGET_LOCK_IOU_MIN:-0.45}"
export DABAI_TARGET_LOCK_HITS="${DABAI_TARGET_LOCK_HITS:-2}"
export DABAI_TARGET_LOST_HOLD_FRAMES="${DABAI_TARGET_LOST_HOLD_FRAMES:-6}"
export DABAI_TARGET_MAX_CENTER_JUMP_PX="${DABAI_TARGET_MAX_CENTER_JUMP_PX:-80}"
export DABAI_TARGET_MAX_DEPTH_JUMP_MM="${DABAI_TARGET_MAX_DEPTH_JUMP_MM:-80}"
export DABAI_SUPPORT_SWITCH_HOLD_FRAMES="${DABAI_SUPPORT_SWITCH_HOLD_FRAMES:-3}"
export DABAI_DEPTH_VALID_RATIO_MIN="${DABAI_DEPTH_VALID_RATIO_MIN:-0.03}"
export DABAI_SUPPORT_POINTS_MIN="${DABAI_SUPPORT_POINTS_MIN:-180}"
export DABAI_SUPPORT_FILL_RATIO_MIN="${DABAI_SUPPORT_FILL_RATIO_MIN:-0.02}"
export DABAI_GROUND_RATIO_MAX="${DABAI_GROUND_RATIO_MAX:-0.96}"
export DABAI_QUALITY_SCORE_MIN="${DABAI_QUALITY_SCORE_MIN:-0.50}"
export DABAI_GRASP_POINT_SMOOTH_ALPHA="${DABAI_GRASP_POINT_SMOOTH_ALPHA:-0.20}"
export DABAI_GRASP_YAW_SMOOTH_ALPHA="${DABAI_GRASP_YAW_SMOOTH_ALPHA:-0.20}"
export DABAI_GRASP_HOLD_FRAMES="${DABAI_GRASP_HOLD_FRAMES:-7}"
export DABAI_GRASP_JUMP_XY_MM="${DABAI_GRASP_JUMP_XY_MM:-22}"
export DABAI_GRASP_JUMP_Z_MM="${DABAI_GRASP_JUMP_Z_MM:-22}"
export DABAI_GRASP_JUMP_YAW_DEG="${DABAI_GRASP_JUMP_YAW_DEG:-20}"
export DABAI_ROBOT_CONTROL_ENABLED="${DABAI_ROBOT_CONTROL_ENABLED:-1}"
export DABAI_AUTO_ENABLE_ONCE_MAX_ATTEMPTS="${DABAI_AUTO_ENABLE_ONCE_MAX_ATTEMPTS:-3}"
export DABAI_AUTO_ENABLE_ONCE_RETRY_DELAY_SEC="${DABAI_AUTO_ENABLE_ONCE_RETRY_DELAY_SEC:-2.0}"
export DABAI_ROBOT_EXCLUSIVE_CONTROL="${DABAI_ROBOT_EXCLUSIVE_CONTROL:-0}"
robot_loopback_only="${DABAI_ROBOT_LOOPBACK_ONLY:-1}"
if [[ "${allow_lan_robot_control}" == "true" ]]; then
    robot_loopback_only="0"
fi
export DABAI_ROBOT_LOOPBACK_ONLY="${robot_loopback_only}"
export DABAI_ROBOT_CONFIG="${DABAI_ROBOT_CONFIG:-${ROOT_DIR}/robot_runtime/config/default.yaml}"
ld_library_parts=()
if [[ -n "${TORCH_LIB_DIR}" ]]; then
    ld_library_parts+=("${TORCH_LIB_DIR}")
fi
if [[ -n "${CUSPARSELT_LIB_DIR}" ]]; then
    ld_library_parts+=("${CUSPARSELT_LIB_DIR}")
fi
ld_library_parts+=("${SDK_LIB_DIR}")
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    ld_library_parts+=("${LD_LIBRARY_PATH}")
fi
export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_library_parts[*]}")"
export PYTHONPATH="${ROOT_DIR}/camera_runtime:${PYTHONPATH:-}"

if [[ "${allow_lan_robot_control}" == "true" ]]; then
    echo "[vision] WARNING: LAN robot control enabled by --allow-lan-robot-control."
fi
if [[ "${DABAI_ROBOT_LOOPBACK_ONLY}" == "0" ]]; then
    echo "[vision] WARNING: robot control API is not loopback-only; LAN clients can send robot commands."
fi

config_mode="legacy"
if [[ "${using_unified_config}" == "true" ]]; then
    config_mode="unified"
fi

if [[ "${check_only}" == "true" ]]; then
    echo "[vision] check passed"
    echo "  root=${ROOT_DIR}"
    echo "  config=${config_path}"
    echo "  config_mode=${config_mode}"
    echo "  model=${DABAI_YOLO_MODEL}"
    echo "  sdk_lib=${SDK_LIB_DIR}"
    echo "  web_port=${DABAI_WEB_PORT}"
    echo "  yolo_device=${DABAI_YOLO_DEVICE}"
    echo "  yolo_imgsz=${DABAI_YOLO_IMGSZ}"
    echo "  yolo_precision=${DABAI_YOLO_PRECISION}"
    echo "  geom_backend=${DABAI_GEOM_BACKEND}"
    echo "  vision_pipeline=${DABAI_VISION_PIPELINE}"
    echo "  shadow_compare=${DABAI_SHADOW_COMPARE}"
    echo "  geom_parity_check=${DABAI_GEOM_PARITY_CHECK}"
    echo "  geom_parity_every_n=${DABAI_GEOM_PARITY_EVERY_N}"
    echo "  infer_every_n=${DABAI_INFER_EVERY_N}"
    echo "  geometry_every_n=${DABAI_GEOMETRY_EVERY_N}"
    echo "  max_points_for_geometry=${DABAI_MAX_POINTS_FOR_GEOMETRY}"
    echo "  ground_fit_fast_enabled=${DABAI_GROUND_FIT_FAST_ENABLED}"
    echo "  ground_fit_fast_sample_cap=${DABAI_GROUND_FIT_FAST_SAMPLE_CAP}"
    echo "  ground_fit_fast_max_trials=${DABAI_GROUND_FIT_FAST_MAX_TRIALS}"
    echo "  ground_fit_fast_min_inlier_ratio=${DABAI_GROUND_FIT_FAST_MIN_INLIER_RATIO}"
    echo "  ground_fit_fast_min_inliers=${DABAI_GROUND_FIT_FAST_MIN_INLIERS}"
    echo "  color_fps=${DABAI_COLOR_FPS}"
    echo "  depth_fps=${DABAI_DEPTH_FPS}"
    echo "  align_mode=${DABAI_ALIGN_MODE}"
    echo "  frame_sync=${DABAI_FRAME_SYNC}"
    echo "  ob_log_level=${DABAI_OB_LOG_LEVEL}"
    echo "  uvicorn_log_level=${DABAI_UVICORN_LOG_LEVEL}"
    echo "  uvicorn_access_log=${DABAI_UVICORN_ACCESS_LOG}"
    echo "  skip_build=${DABAI_SKIP_BUILD}"
    echo "  force_rebuild=${DABAI_FORCE_REBUILD}"
    echo "  target_lock_iou_min=${DABAI_TARGET_LOCK_IOU_MIN}"
    echo "  target_lock_hits=${DABAI_TARGET_LOCK_HITS}"
    echo "  target_lost_hold_frames=${DABAI_TARGET_LOST_HOLD_FRAMES}"
    echo "  target_max_center_jump_px=${DABAI_TARGET_MAX_CENTER_JUMP_PX}"
    echo "  target_max_depth_jump_mm=${DABAI_TARGET_MAX_DEPTH_JUMP_MM}"
    echo "  support_switch_hold_frames=${DABAI_SUPPORT_SWITCH_HOLD_FRAMES}"
    echo "  depth_valid_ratio_min=${DABAI_DEPTH_VALID_RATIO_MIN}"
    echo "  support_points_min=${DABAI_SUPPORT_POINTS_MIN}"
    echo "  support_fill_ratio_min=${DABAI_SUPPORT_FILL_RATIO_MIN}"
    echo "  ground_ratio_max=${DABAI_GROUND_RATIO_MAX}"
    echo "  quality_score_min=${DABAI_QUALITY_SCORE_MIN}"
    echo "  grasp_point_smooth_alpha=${DABAI_GRASP_POINT_SMOOTH_ALPHA}"
    echo "  grasp_yaw_smooth_alpha=${DABAI_GRASP_YAW_SMOOTH_ALPHA}"
    echo "  grasp_hold_frames=${DABAI_GRASP_HOLD_FRAMES}"
    echo "  grasp_jump_xy_mm=${DABAI_GRASP_JUMP_XY_MM}"
    echo "  grasp_jump_z_mm=${DABAI_GRASP_JUMP_Z_MM}"
    echo "  grasp_jump_yaw_deg=${DABAI_GRASP_JUMP_YAW_DEG}"
    echo "  robot_control_enabled=${DABAI_ROBOT_CONTROL_ENABLED}"
    echo "  auto_enable_once_max_attempts=${DABAI_AUTO_ENABLE_ONCE_MAX_ATTEMPTS}"
    echo "  auto_enable_once_retry_delay_sec=${DABAI_AUTO_ENABLE_ONCE_RETRY_DELAY_SEC}"
    echo "  robot_loopback_only=${DABAI_ROBOT_LOOPBACK_ONLY}"
    echo "  robot_exclusive_control=${DABAI_ROBOT_EXCLUSIVE_CONTROL}"
    echo "  robot_config=${DABAI_ROBOT_CONFIG}"
    exit 0
fi

terminate_stale_processes
pause_auto_enable_daemon_for_exclusive_control
prepare_robot_can

build_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
publisher_bin="${BUILD_DIR}/dabai_frame_publisher"

should_build=0
if [[ "${DABAI_FORCE_REBUILD}" == "1" ]]; then
    should_build=1
elif [[ "${DABAI_SKIP_BUILD}" != "1" ]]; then
    should_build=1
elif publisher_needs_rebuild "${publisher_bin}"; then
    should_build=1
fi

if [[ "${should_build}" == "1" ]]; then
    echo "[vision] building ARM64 publisher..."
    cmake -S "${PUBLISHER_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --config Release -j"${build_jobs}"
else
    echo "[vision] publisher binary is up-to-date, skip build."
fi

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
exit "${status}"
