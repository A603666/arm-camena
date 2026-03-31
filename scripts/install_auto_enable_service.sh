#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="nero-auto-enable.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
TMP_FILE="$(mktemp)"
DEFAULT_UNIFIED_CONFIG="${ROOT_DIR}/pipeline_config.yaml"
LEGACY_AUTO_ENABLE_CONFIG="${ROOT_DIR}/robot_runtime/config/auto_enable.yaml"
LOADER_PATH="${ROOT_DIR}/scripts/unified_config_loader.py"

if [[ -f "${DEFAULT_UNIFIED_CONFIG}" ]]; then
    CONFIG_PATH="${DEFAULT_UNIFIED_CONFIG}"
else
    CONFIG_PATH="${LEGACY_AUTO_ENABLE_CONFIG}"
fi

while (( "$#" > 0 )); do
    case "$1" in
        --config)
            if (( "$#" < 2 )); then
                echo "[install] --config requires a path argument" >&2
                exit 2
            fi
            CONFIG_PATH="$2"
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./scripts/install_auto_enable_service.sh [--config PATH]

  --config PATH   Config path for nero_auto_enable_daemon.py
EOF
            exit 0
            ;;
        *)
            echo "[install] unknown argument: $1" >&2
            echo "[install] supported arguments: --config PATH" >&2
            exit 2
            ;;
    esac
    shift
done

CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[install] config not found: ${CONFIG_PATH}" >&2
    exit 2
fi

if [[ ! -f "${LOADER_PATH}" ]]; then
    echo "[install] unified config loader not found: ${LOADER_PATH}" >&2
    exit 2
fi

auto_enable_yaml="$(python3 "${LOADER_PATH}" --config "${CONFIG_PATH}" --mode component --component auto_enable 2>/dev/null || true)"
if [[ -z "${auto_enable_yaml}" ]]; then
    echo "[install] failed to parse auto_enable section from config: ${CONFIG_PATH}" >&2
    exit 2
fi

mapfile -t auto_enable_info < <(
    printf '%s\n' "${auto_enable_yaml}" | python3 -c 'import sys, yaml; cfg = yaml.safe_load(sys.stdin.read()) or {}; print(str(cfg.get("can_channel", "")).strip()); print(str(cfg.get("usb_bus_info", "")).strip())'
)
AUTO_ENABLE_CHANNEL="${auto_enable_info[0]:-}"
AUTO_ENABLE_USB_BUS_INFO="${auto_enable_info[1]:-}"
if [[ -z "${AUTO_ENABLE_CHANNEL}" || -z "${AUTO_ENABLE_USB_BUS_INFO}" ]]; then
    echo "[install] invalid auto_enable section in config: ${CONFIG_PATH}" >&2
    exit 2
fi
echo "[install] auto_enable channel=${AUTO_ENABLE_CHANNEL} usb_bus_info=${AUTO_ENABLE_USB_BUS_INFO}"
if [[ ! "${AUTO_ENABLE_CHANNEL}" =~ ^can[0-9]+$ ]]; then
    echo "[install] invalid auto_enable.can_channel=${AUTO_ENABLE_CHANNEL}; expected format can<index> (e.g. can0/can1)." >&2
    exit 2
fi

cleanup() {
    rm -f "${TMP_FILE}"
}
trap cleanup EXIT

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

cat >"${TMP_FILE}" <<EOF
[Unit]
Description=NERO Auto CAN Push and Auto Enable Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${ROOT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 ${ROOT_DIR}/robot_runtime/nero_auto_enable_daemon.py --config ${CONFIG_PATH}
Restart=always
RestartSec=2
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

${SUDO} install -m 644 "${TMP_FILE}" "${SERVICE_PATH}"
${SUDO} systemctl daemon-reload
${SUDO} systemctl enable --now "${SERVICE_NAME}"
${SUDO} systemctl status "${SERVICE_NAME}" --no-pager
echo "[install] service config=${CONFIG_PATH}"
