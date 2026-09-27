# Running the Kikuchi Pattern Processor on OSC (Linux)

Starting point for whoever (human or Claude) is setting this up on the Ohio
Supercomputer Center. Read `README.md` first for what the tool does and the
feature list. This file covers only what changes when you leave Windows.

## TL;DR

- The app is one file, `gpu_pattern_viewer.py`, a PyQt5 GUI. Nothing in it is
  Windows-specific except the `.bat` launcher, which you do not need.
- It should run on Linux without code changes. It has NOT been tested there.
- It needs a display. Run it inside an OSC OnDemand **Interactive Desktop**
  session, not inside Code Server / VS Code in the browser (no X server there).
- GPU is optional. Without CuPy it falls back to CPU paths that give identical
  results. NLPAR batch on CPU is multi-threaded through pyebsdindex/numba and
  will use every core on the node. The `Apply && Save` pipeline on CPU is a
  per-pattern Python loop and is slow on big scans. See "Where the GPU matters".

## Files in this transfer

| File | Purpose |
|---|---|
| `gpu_pattern_viewer.py` | The whole application (~1700 lines, single file) |
| `README.md` | Feature description, dependencies, citations |
| `OSC_SETUP.md` | This file |
| `requirements.txt` | Pinned-ish package list matching the dev machine |
| `run_nlpar.py` | Headless NLPAR runner (opt_lambda + calcnlpar), no GUI needed |
| `interactive_compare.py` | Clickable IQ map, original vs NLPAR side by side |
| `compare_patterns.py` | Static plot of one original vs NLPAR pattern |
| `_check_nlpar.py` | Prints pyebsdindex UPFile source, introspection only |
| `LICENSE`, `.gitignore` | MIT, and the ignore list (UP1 files are gitignored) |

Not needed: `gpu_pattern_viewer.bat`, `__pycache__/`, `.vscode/`,
`test_kikuchipy.py` (empty).

Data files (`.up1`) are 0.7 to 32 GB each. Move them with Globus or `sftp`
(home is fine; scratch is not required), not through the chat.

## Dev machine environment (what it is known to work with)

```
Python 3.14.0 (Windows)      pyebsdindex 0.3.9.1     numba 0.64.0
numpy 2.3.5                  scipy 1.16.3            scikit-image 0.26.0
PyQt5 / Qt 5.15.2            cupy 14.0.1 (CUDA 12)   psutil 7.2.2
matplotlib (any recent)
```

Python 3.11+ should be fine. If 3.14 wheels are missing for something on OSC,
use 3.12.

## Install on OSC

```bash
module load python/3.12        # or whatever recent python module exists
module load cuda/12.4          # only if you want the GPU path; match cupy variant
python -m venv ~/venvs/kikuchi
source ~/venvs/kikuchi/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install cupy-cuda12x       # skip for CPU-only
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

Do not install `pyebsdindex[gpu]` or `pyopencl`. The app uses only the numba
CPU code in pyebsdindex plus its file I/O. OpenCL is not needed.

If PyQt5 complains about `xcb` at launch, install/load the X11 client libs
(`libxcb-xinerama0`, `libxkbcommon-x11-0` on Debian-style; on OSC they are
already present inside the OnDemand Desktop image). `QT_QPA_PLATFORM=xcb` is
the default and correct value.

## Run

Inside an OnDemand Interactive Desktop session (request a GPU node if you want
the GPU path, otherwise any node with lots of RAM and cores):

```bash
source ~/venvs/kikuchi/bin/activate
cd /path/to/kikuchi-pattern-processor
python gpu_pattern_viewer.py
```

The console prints `[GPU] CuPy detected: <name>` or `[GPU] CuPy not available`
on startup so you know which path you are on.

For a batch job without a display, `run_nlpar.py` does opt_lambda + NLPAR on
one file. Edit the three constants at the top and submit it with `sbatch`.
There is currently no headless CLI for the `Apply && Save` processing
pipeline; that is the first thing worth adding if you want to run full-scan
background/CLAHE processing as a Slurm job.

## Where the GPU matters

Each of these has a CuPy path and a CPU fallback selected by the `HAS_GPU`
flag at import time:

| Stage | GPU | CPU fallback | CPU speed |
|---|---|---|---|
| Mean pattern + IQ map on load | chunked CuPy | numpy | fine |
| Sigma-squared map for NLPAR | shift-and-compare on GPU | numpy loop | slow but one-time per aperture |
| NLPAR batch (full scan) | own CuPy implementation | pyebsdindex `calcnlpar` (numba, parallel) | good on a many-core node |
| On-demand NLPAR/NPAR preview (one pixel) | n/a | numpy | instant |
| `Apply && Save` pipeline | chunked CuPy | per-pattern `scipy.ndimage` + `skimage` CLAHE loop | slow, single-threaded |

So on a 48 to 96 core CPU node the NLPAR side is fine. The full-scan
background/CLAHE export is the one that really wants a GPU or a rewrite to
`joblib`/`multiprocessing` over patterns.

Set `NUMBA_NUM_THREADS` if you want to limit pyebsdindex, otherwise it takes
all cores it sees.

## Memory and chunking

All chunk sizes are computed at runtime from `psutil.virtual_memory().available`
and, on GPU, from `cp.cuda.Device(0).mem_info`. There are no hardcoded row
counts. Request enough RAM in the job (32 GB is comfortable for a 480x480 px
scan; the 8 GB TruePix scans are ~99 MB per row as uint8) and the app will
size chunks accordingly. If you see `_ArrayMemoryError`, the job has less
available RAM than psutil thinks or another process is competing.

## Temp files

The app writes intermediate UP1 files next to the INPUT file and deletes them
on clean exit:

- `_gpu_nlpar_result.up1`
- `_cpu_nlpar_result.up1`
- pyebsdindex fallback name `<stem>_NLPAR_l<lam>sr<sr>.up1`

They land next to the input; clean up manually if the session is killed.

## Things that were only ever tested on Windows and might bite

- `sys.stdout.reconfigure(errors='replace')` at the top is a Windows cp1252
  workaround. Harmless on Linux.
- File dialogs default to the input file's directory. Fine on Linux.
- The v4 UP1 header patch (copying bytes 42..filePos from the source file
  after the first write) is pure byte I/O, no platform dependence.
- Line endings: files may arrive with CRLF. Python does not care.
- Font rendering of the sigma/lambda characters in Qt labels depends on the
  desktop's fonts. Cosmetic only.

## Sanity test

Use the smallest scan you have (the FGHAZ 25 map, ~670 MB) before touching the
8 GB ones:

1. Load it, confirm the IQ map draws and clicking shows a pattern.
2. Aperture: `None`, then `Optimize lambda`, then `NLPAR batch`. Compare wall
   time against the Windows laptop numbers in the status bar.
3. `Apply && Save` with defaults, then reopen the output in the app to confirm
   the header round-trips.
