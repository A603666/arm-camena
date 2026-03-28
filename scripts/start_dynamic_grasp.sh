#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/dynamic_grasp_config.yaml"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

exec python3 "${ROOT_DIR}/run_dynamic_grasp.py" --config "${CONFIG_PATH}" "$@"
