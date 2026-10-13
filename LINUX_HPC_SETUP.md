# Running the Kikuchi Pattern Processor on Linux / an HPC cluster

Read `README.md` first for what the tool does. This file covers only what changes
when you leave a Windows workstation: a Linux desktop, or a Slurm cluster with an
Open OnDemand style interactive desktop.

## TL;DR

- The app is one file, `gpu_pattern_viewer.py`, a PyQt5 GUI. Nothing in it is
  Windows-specific except the `.bat` launcher.
- It needs a display. On a cluster run it inside an interactive desktop session
  (OnDemand "Desktop"), not inside a browser code editor without an X server.
- The NLPAR batch runs through pyebsdindex: its OpenCL kernels when `pyopencl` can
  see a GPU, its numba CPU code otherwise (multi-threaded, uses every core the job
  owns). CuPy is optional and only accelerates the load statistics, previews and the
  `Apply && Save` pipeline.
- Thread count comes from the Slurm cpuset; RAM budgets respect the job's cgroup
  limit. `KPP_THREADS=<n>` and `KPP_GPU=0` override.

## Install

```bash
module load python/3.12            # or any recent Python module / system Python 3.11+
module load cuda/12.x              # provides the OpenCL and CUDA runtimes (GPU nodes)
python -m venv ~/venvs/kikuchi
source ~/venvs/kikuchi/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyopencl               # NLPAR on the GPU through pyebsdindex (optional)
pip install cupy-cuda12x           # previews / Apply && Save on the GPU (optional)
```

Check what the app will use:

```bash
python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"
python gpu_pattern_viewer.py       # prints [NLPAR] engine: ... and [GPU] ... at startup
```

If PyQt5 complains about `xcb` at launch, the X11 client libraries are missing
(`libxcb-xinerama0`, `libxkbcommon-x11-0` on Debian-style systems; present inside
OnDemand desktop images). `QT_QPA_PLATFORM=xcb` is the default and correct value.

## Run

```bash
bash run_gui.sh                    # loads modules, activates the venv, starts the GUI
```

`run_gui.sh` reads site-specific settings from `local/env.sh` (git-ignored):

```bash
PYTHON_MODULE=python/3.12
CUDA_MODULE=cuda/12.9.1
VENV=~/venvs/kikuchi
```

`Kikuchi Pattern Processor.desktop` is a launcher for XFCE/GNOME desktops: edit the
path inside it and copy it to your desktop.

For a headless check or benchmark, `selftest.py <scan.up1>` runs load, sigma map,
NLPAR batch and `Apply && Save` with timings and no display (`QT_QPA_PLATFORM=offscreen`).

## Where the GPU matters

| Stage | GPU | CPU | Notes |
|---|---|---|---|
| Load statistics (IQ map, mean pattern) | CuPy | threaded NumPy | fine either way |
| Sigma map, Optimize Lambda, NLPAR batch | pyebsdindex OpenCL | pyebsdindex numba | one shared object; sigma computed once |
| Single-pixel NLPAR/NPAR preview | n/a | NumPy | 0.2-0.4 s per click on 479x479 patterns |
| `Apply && Save` pipeline | chunked CuPy | threaded NumPy / scikit-image | the GPU helps most here |

For large patterns (e.g. 479x479) the OpenCL kernels run few work-items per tile and a
many-core CPU node can be faster than a small GPU; measure both with `selftest.py` and
pick with `KPP_GPU=0`.

## Memory and chunking

All chunk sizes are computed at runtime from available RAM (psutil, capped by the
Slurm cgroup limit) and, with CuPy, from free VRAM. The dataset is held in RAM when
it fits in 60 % of what the job may allocate, otherwise it is streamed from disk on
every pass; `KPP_IN_MEMORY=1/0` forces either. The NLPAR result is kept in RAM when a
second copy fits (`KPP_NLPAR_IN_MEMORY=0` forces the temp file).

## Temp files

Written next to the input file and deleted on clean exit or on the next load:

- `_nlpar_result.up1` / `.up2` (NLPAR batch result when it does not fit in RAM)
- pyebsdindex fallback name `<stem>_NLPAR_l<lam>sr<sr>.up1`

Clean up manually if the session is killed.

## Sanity test

Use your smallest scan first:

1. Load it, confirm the IQ map draws and clicking shows a pattern.
2. Aperture as appropriate, then `Optimize Lambda`, then `Apply NLPAR to Full Dataset`.
3. `Apply && Save` with defaults, then reopen the output in the app to confirm the
   header round-trips.
