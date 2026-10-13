# Getting started (Windows, PyCharm)

A step-by-step for someone who has PyCharm and a recent Python installed and wants to
run the Kikuchi Pattern Processor on their own UP1/UP2 scans. For a Linux desktop or a
cluster, read this first and then `LINUX_HPC_SETUP.md`.

## 1. What you need

| Item | Notes |
|---|---|
| Windows 10/11, 64-bit | Linux works too, see `LINUX_HPC_SETUP.md` |
| Python 3.11 to 3.14, 64-bit | the version PyCharm shows under *Python Interpreter*; developed on 3.14 |
| PyCharm | Community edition is fine |
| Git | optional; "Download ZIP" on GitHub works just as well |
| RAM | at least 16 GB. A scan is held in RAM when it is smaller than ~60 % of the free RAM, otherwise it is streamed from disk on every pass (slower, but works) |
| Disk | free space of about the scan size next to the input file, for the output and a temporary NLPAR file |
| GPU | optional. NVIDIA with a current driver gives NLPAR on the GPU (OpenCL) and, with CuPy, GPU previews and export. Without one everything runs on the CPU with the same results |

Keep the scan files on a local drive. Reading a 10 GB file through OneDrive, a network
share or an external USB disk works but makes every pass several times slower.

## 2. Get the code

Either

```
git clone https://github.com/helpimlost97/kikuchi-pattern-processor.git
```

or open the repository page on GitHub, **Code → Download ZIP**, and unzip it somewhere
without spaces in the path if you can (spaces are fine, they just make command lines
longer).

## 3. Open it in PyCharm and create the environment

1. **File → Open…** and pick the `kikuchi-pattern-processor` folder.
2. **File → Settings → Project: kikuchi-pattern-processor → Python Interpreter →
   Add Interpreter → Add Local Interpreter**.
3. Choose **Virtualenv**, *New*, location inside the project (the default `.venv`),
   base interpreter = your installed Python 3.11+. OK.
4. Open the **Terminal** tab at the bottom of PyCharm. The prompt should show `(.venv)`.
   If it does not, close and reopen the terminal.
5. Install the packages:

   ```
   pip install -r requirements.txt
   ```

   That gives you PyQt5, pyebsdindex (file I/O, NLPAR, lambda optimisation), NumPy,
   SciPy, scikit-image, matplotlib and psutil. First install takes a few minutes.

6. Optional, GPU:

   ```
   pip install pyopencl
   ```

   With this, pyebsdindex runs NLPAR on the GPU through OpenCL. The NVIDIA driver
   already provides the OpenCL runtime; nothing else to install. Integrated Intel
   graphics are also seen but the app prefers a discrete GPU.

   ```
   pip install cupy-cuda12x
   ```

   CuPy moves the preview, the load statistics and the `Apply && Save` export to the
   GPU. It needs the CUDA 12 runtime libraries; if `import cupy` fails the app simply
   prints `[GPU] CPU mode` and carries on. Skip it if you do not want to deal with CUDA.

## 4. Run it

Right-click `gpu_pattern_viewer.py` in the project tree → **Run**. Or, outside PyCharm,
double-click `gpu_pattern_viewer.bat` (it calls whatever `python` is on your PATH, so
either make that the venv's Python or edit the `.bat`).

The first lines in the console tell you what you got:

```
[GPU] NVIDIA GeForce RTX ..., 8.5 GB VRAM            or   [GPU] CPU mode (CuPy not available)
[NLPAR] engine: pyebsdindex OpenCL; OpenCL kernel patch: full (clnlpar_fixed.cl)
                                                     or   [NLPAR] engine: pyebsdindex CPU (numba)
[CPU] 24 worker threads, 17.8 GB RAM available to this process
```

The `kernel patch: full` line matters: pyebsdindex's shipped OpenCL kernel crashes on
pattern sizes that are not a multiple of 16 pixels (479x479, for example) and has a
weighting bug; the app loads a corrected copy from `patches/` automatically. If the
line says anything else, set `KPP_GPU=0` (see section 7) and use the CPU engine.

## 5. A first session

1. **Load UP1/UP2.** The IQ map draws once the whole file has been read once. The
   file label says whether the scan is held in RAM or streamed from disk.
2. **Click a point on the map** to see its pattern. Ctrl + scroll zooms the map.
3. **Aperture.** `None` for a square detector with no dead area; `Circular (EBSD)` for
   a conventional phosphor camera; `Auto-detect` for direct electron detectors with chip
   gaps and dead corners. The aperture only decides which pixels enter the NLPAR
   statistics; it never alters the saved patterns. **Show detected mask** lets you check
   the auto-detect result.
4. **IQ range.** The two sliders exclude patterns outside the range from everything
   downstream: they are neither averaged nor used as neighbours, and they come out as
   zeros. Useful when most of the map is empty (e.g. particles on an extraction replica).
5. **NLPAR preview.** Tick *Enable Preview*, pick a *Search Radius* (3 for a first
   look, up to 10) and click around. The preview is instant and shows exactly what the
   full batch will produce for that point.
6. **Optimize Lambda.** Required before the batch. It computes the noise map (σ) and
   fits λ; both are kept for the batch. A few seconds to a minute depending on scan size.
   Leave *Saturation protect* on and *Distance threshold* at 0 unless you know why not.
7. **Apply NLPAR to Full Dataset.** Status bar shows percent and ETA. The result stays
   in RAM when a second copy of the scan fits, otherwise it goes to a temporary file
   next to the input. Typical: a 6.5 GB scan of 479x479 patterns at search radius 6
   takes about 6 minutes on an 8 GB laptop GPU, about 4 minutes on a 24-core CPU.
8. **Processing options** (static background subtract/divide, dynamic background,
   normalise, brightness/contrast, CLAHE) apply to the preview immediately so you can
   tune them on a few points.
9. **Apply && Save.** Writes a new UP1/UP2 with the same header, bit depth and grid as
   the input, readable by OIM Analysis. Never overwrite the input.

The app refuses to run the batch before Optimize Lambda, and refuses to save while the
NLPAR preview is on but no full-dataset result exists. Those are deliberate guards.

Temporary files (`_nlpar_result.up1/.up2` next to the input) are deleted when you load
another file or close the app. If the app is killed, delete them by hand.

## 6. Headless check

`selftest.py` runs load → σ map → NLPAR batch → `Apply && Save` on a scan with no
window and prints timings. Good for checking an install or comparing machines:

```
python selftest.py path\to\small_scan.up1
```

## 7. Switches (environment variables)

Set them in PyCharm under **Run → Edit Configurations → Environment variables**, or in
the terminal before launching (`set KPP_GPU=0` on Windows cmd, `$env:KPP_GPU=0` in
PowerShell).

| Variable | Effect |
|---|---|
| `KPP_GPU=0` | CPU only: pyebsdindex numba for NLPAR, NumPy for the rest. On large patterns a many-core CPU can beat a small GPU |
| `KPP_THREADS=<n>` | thread count for the CPU paths (default: all cores the process may use) |
| `KPP_IN_MEMORY=1` / `0` | force the scan into RAM / force streaming from disk |
| `KPP_NLPAR_IN_MEMORY=0` | always write the NLPAR result to the temp file |
| `KPP_NLPAR_KERNEL_PATCH=0` | use pyebsdindex's OpenCL kernel as shipped (for comparison only) |

## 8. When something goes wrong

- **`No module named PyQt5`** (or pyebsdindex): PyCharm is running the script with the
  wrong interpreter. Check the interpreter in the bottom-right corner of the window.
- **Console says `[NLPAR] engine: pyebsdindex CPU` although you have a GPU**: pyopencl
  is not installed in this venv, or the driver's OpenCL runtime is missing. `pip
  install pyopencl`, then `python -c "import pyopencl as cl; print(cl.get_platforms())"`.
- **`clFinish failed: INVALID_COMMAND_QUEUE`**: the OpenCL kernel patch did not apply
  (the startup line will say so). Run with `KPP_GPU=0`.
- **Everything is slow**: the scan is streamed from disk (file label says so) or lives
  on a network/cloud drive. Move it to a local SSD, close other memory-hungry programs.
- **`_ArrayMemoryError` / out of memory**: another program holds the RAM, or the scan
  is far bigger than the machine. `KPP_IN_MEMORY=0` forces streaming.
- **Saved file opens dark or wrong in OIM**: the output keeps the input's extension on
  purpose (`.up1` = 8-bit, `.up2` = 16-bit). Do not rename across extensions.
- **A dialog says "NLPAR is running"**: a batch or lambda fit is in progress; wait for
  the status bar to report completion before clicking again.

For anything else, the console output from startup to the error is what to send along
with the question.
