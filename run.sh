#!/usr/bin/env bash
set -Eeuo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export ORCATCA_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 LOGURU_DIAGNOSE=NO
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
umask 077
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$HOME/miniforge3/envs/orcatca-npu/bin/python" ]]; then
  PY="$HOME/miniforge3/envs/orcatca-npu/bin/python"
else
  PY="$(command -v python)"
fi
# Reuse the already working CANN installation; do not install or upgrade drivers.
CANN_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/cann-8.5.1/set_env.sh}"
if [[ -r "$CANN_SCRIPT" ]]; then
  set +u
  source "$CANN_SCRIPT"
  set -u
fi
cd "$ROOT"
exec "$PY" -m orcatca "$@"
