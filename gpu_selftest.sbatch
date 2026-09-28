#!/bin/bash
# GPU-vs-CPU self-test for the Kikuchi Pattern Processor's NLPAR / sigma-map / Apply&&Save
# code paths (osc_gpu_selftest.py). Submit with:
#
#   cd /fs/scratch/PAS2325/vegamichalak1/kikuchi_test
#   sbatch gpu_selftest.sbatch
#
# Cost: 1 GPU, 24 CPU cores, up to 1 hour wall time -> at most 1 GPU-hour against the
# pas2325 allocation (the 20-row subset finishes in a few minutes; the 1-hour cap is
# just headroom). Validate the header alone first with `sbatch --test-only gpu_selftest.sbatch`
# from that same directory -- that does NOT submit or run anything.
#SBATCH --account=pas2325
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=kpp_gpu_selftest
#SBATCH --output=%x_%j.out
set -euo pipefail

cd /fs/scratch/PAS2325/vegamichalak1/kikuchi_test

source /etc/profile.d/lmod.sh
module load python/3.12
module load cuda/12.9.1   # newest CUDA 12.x module (checked with: module -t avail cuda)

source ~/EBSD/venvs/kikuchi/bin/activate
export QT_QPA_PLATFORM=offscreen

echo "== nvidia-smi -L"
nvidia-smi -L

echo "== running osc_gpu_selftest.py"
python ~/EBSD/kikuchi-pattern-processor/osc_gpu_selftest.py subset_20rows.up1
