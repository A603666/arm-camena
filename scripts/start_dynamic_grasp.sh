#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIFIED_CONFIG_PATH="${ROOT_DIR}/pipeline_config.yaml"
LEGACY_CONFIG_PATH="${ROOT_DIR}/dynamic_grasp_config.yaml"

if [[ -f "${UNIFIED_CONFIG_PATH}" ]]; then
    CONFIG_PATH="${UNIFIED_CONFIG_PATH}"
else
    CONFIG_PATH="${LEGACY_CONFIG_PATH}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

exec python3 "${ROOT_DIR}/run_dynamic_grasp.py" --config "${CONFIG_PATH}" "$@"
