#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="nero-auto-enable.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
TMP_FILE="$(mktemp)"

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
ExecStart=/usr/bin/python3 ${ROOT_DIR}/robot_runtime/nero_auto_enable_daemon.py --config ${ROOT_DIR}/robot_runtime/config/auto_enable.yaml
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
