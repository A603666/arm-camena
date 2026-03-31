#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIFIED_CONFIG_PATH="${ROOT_DIR}/pipeline_config.yaml"
LEGACY_CONFIG_PATH="${ROOT_DIR}/dynamic_grasp_config.yaml"
LOADER_PATH="${ROOT_DIR}/scripts/unified_config_loader.py"

if [[ -f "${UNIFIED_CONFIG_PATH}" ]]; then
    CONFIG_PATH="${UNIFIED_CONFIG_PATH}"
else
    CONFIG_PATH="${LEGACY_CONFIG_PATH}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -f "${LOADER_PATH}" ]]; then
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        if [[ "${line}" == DABAI_DYNAMIC_GRASP_API=* ]]; then
            export DABAI_DYNAMIC_GRASP_API="${line#*=}"
        fi
    done < <(python3 "${LOADER_PATH}" --config "${CONFIG_PATH}" --mode vision-env 2>/dev/null || true)
fi

exec python3 "${ROOT_DIR}/run_dynamic_grasp.py" --config "${CONFIG_PATH}" "$@"
