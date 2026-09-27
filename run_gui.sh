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
module load python/3.12
source ~/EBSD/venvs/kikuchi/bin/activate
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
cd "$(dirname "${BASH_SOURCE[0]}")"
python gpu_pattern_viewer.py "$@"
