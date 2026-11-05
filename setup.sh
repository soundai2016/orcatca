#!/usr/bin/env bash
# Install project dependencies only, into the existing working NPU environment.
set -Eeuo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${PYTHON_BIN:-}" ]]; then PY="$PYTHON_BIN"
elif [[ -x "$HOME/miniforge3/envs/orcatca-npu/bin/python" ]]; then PY="$HOME/miniforge3/envs/orcatca-npu/bin/python"
else PY="$(command -v python)"; fi
"$PY" -c 'import torch; print("Keeping PyTorch", torch.__version__)'
FLAGS=()
if [[ -n "${WHEELHOUSE:-}" ]]; then FLAGS+=(--no-index --find-links "$WHEELHOUSE"); fi
"$PY" -m pip install --no-build-isolation "${FLAGS[@]}" -e "$ROOT"
"$PY" -c 'import loguru; print("Loguru",loguru.__version__)'
echo 'Project installed. No CANN/driver/PyTorch installation was requested.'
