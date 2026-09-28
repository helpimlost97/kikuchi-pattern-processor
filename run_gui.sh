#!/bin/bash
# Launch the Kikuchi Pattern Processor GUI on OSC.
#
# Run this from a terminal INSIDE an OSC OnDemand "Cardinal Desktop" (Interactive
# Apps -> Desktops) session. The desktop is a SLURM job: pick the core count and
# memory there (e.g. 48-96 cores). The app sizes its thread pools and RAM budgets
# from that allocation automatically.
#
#   bash ~/EBSD/kikuchi-pattern-processor/run_gui.sh
#
# Optional: KPP_THREADS=<n> to use fewer threads than allocated.
# Optional: KPP_GPU=0 to force CPU mode even when the desktop has a GPU.
# GPU mode needs a desktop session with a GPU requested; on a CPU-only desktop the
# app falls back to CPU automatically.
module load python/3.12
module load cuda/12.9.1 2>/dev/null || true
if [ "${KPP_GPU:-1}" = "0" ]; then
    export KPP_GPU=0
    export CUDA_VISIBLE_DEVICES=""   # hide GPUs from CuPy so CPU mode is used
    echo "KPP_GPU=0: CPU mode forced"
elif nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | head -1
else
    echo "no GPU in this job: CPU mode"
fi
source ~/EBSD/venvs/kikuchi/bin/activate
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
cd "$(dirname "${BASH_SOURCE[0]}")"
python gpu_pattern_viewer.py "$@"
