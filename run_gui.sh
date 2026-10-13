#!/bin/bash
# Launch the Kikuchi Pattern Processor GUI on a Linux workstation or an HPC desktop
# session (e.g. an Open OnDemand interactive desktop, which is a Slurm job: pick the
# core count / memory / GPU there; the app sizes its thread pools and RAM budgets from
# the allocation automatically).
#
#   bash /path/to/kikuchi-pattern-processor/run_gui.sh
#
# Site-specific settings live in local/env.sh next to this script (git-ignored), e.g.:
#   PYTHON_MODULE=python/3.12      # `module load` name, or empty if no modules
#   CUDA_MODULE=cuda/12.9.1        # provides the OpenCL / CUDA runtime, or empty
#   VENV=~/venvs/kikuchi           # virtualenv with requirements.txt installed
# Without that file the defaults below are used.
#
# Optional: KPP_THREADS=<n> to use fewer threads than allocated.
# Optional: KPP_GPU=0 to force CPU mode (pyebsdindex numba NLPAR, NumPy pipeline).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MODULE=${PYTHON_MODULE:-python/3.12}
CUDA_MODULE=${CUDA_MODULE:-cuda}
VENV=${VENV:-$HOME/venvs/kikuchi}
[ -f "$HERE/local/env.sh" ] && source "$HERE/local/env.sh"

if command -v module >/dev/null 2>&1; then
    [ -n "$PYTHON_MODULE" ] && module load "$PYTHON_MODULE"
    [ -n "$CUDA_MODULE" ] && { module load "$CUDA_MODULE" 2>/dev/null || true; }
fi
if [ "${KPP_GPU:-1}" = "0" ]; then
    export KPP_GPU=0
    export CUDA_VISIBLE_DEVICES=""   # hide GPUs from CuPy so CPU mode is used
    echo "KPP_GPU=0: CPU mode forced"
elif nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | head -1
else
    echo "no GPU in this job: CPU mode"
fi
VENV="${VENV/#\~/$HOME}"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    echo "warning: virtualenv $VENV not found; using the current python ($(command -v python))"
fi
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
cd "$HERE"
python gpu_pattern_viewer.py "$@"
