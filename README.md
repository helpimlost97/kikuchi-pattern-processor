# Kikuchi Pattern Processor (NPAR/NLPAR)

A GPU-accelerated GUI for viewing and processing EBSD / TKD / RKD Kikuchi pattern
files (EDAX UP1/UP2 format). Provides on-demand and full-dataset NPAR and NLPAR
denoising, plus the standard pattern-processing pipeline (static / dynamic
background, brightness, contrast, CLAHE), wrapped in a PyQt5 interface that runs
sub-second previews on a GPU and handles full-scan batch processing in
chunks sized to the available VRAM.

---

## ⚠️ This project is 100% vibe coded.

Every line was written through iterative LLM collaboration by someone who is not
a software engineer. It is a personal research tool, not production software. It
works on the author's data and machine. If it works on yours, great. If it
breaks in some interesting new way, congratulations — you now know roughly as
much about the internals as the author does. In fact, he didn't even write this readme.

No warranties, no roadmap, no test suite, no design document. Just a tool that
turned out to be useful enough to publish.

---

## What it does

- Loads EDAX **UP1 / UP2** pattern files (v3 and v4 headers, hex-grid flag
  preserved end-to-end for OIM round-trip compatibility)
- **8-bit and 16-bit patterns** (v1.5.0): a `.up2` is processed as uint16 all the
  way through. The raw/absolute-intensity preview and the brightness/contrast
  midpoint use the data's effective full scale (255 for `.up1`; for `.up2` the
  smallest 2^k-1 that holds the dataset max, e.g. 4095 for a 12-bit camera),
  the normalized export fills the full dtype range (255 / 65535), and every
  output (NLPAR temp files, `Apply && Save`) keeps the source extension
  because pyebsdindex and OIM take the bit depth from it.
- Interactive **IQ map** with click-to-preview, Ctrl + scroll zoom, dual-slider
  IQ range, semi-transparent mask overlay for the patterns outside it
- **NLPAR** using the canonical Brewick et al. 2019 formulation with
  dimensionless λ and a σ²-normalized weight: `w(i,j) = exp(-max(d̃, 0) / λ²)`
  - Full-dataset batch runs through **pyebsdindex's own NLPAR** (v1.6): its
    OpenCL kernels when `pyopencl` sees a GPU, numba on the CPU otherwise. The
    σ map from Optimize λ is reused by the batch. The viewer patches one line of
    pyebsdindex's OpenCL kernel at startup (neighbour σ indexed as the centre σ;
    see `PLAN_NLPAR_PYEBSDINDEX.md`); `KPP_NLPAR_KERNEL_PATCH=0` disables that.
  - On-demand single-pixel preview before committing to the full pass (our own
    NumPy code, kept for instant response; it uses pyebsdindex's σ map and
    saturation rule so interior pixels match the batch, and zero-pads at the
    scan border where pyebsdindex shifts or mirrors the window)
  - σ map and **Optimize λ** come from `pyebsdindex.nlpar.NLPAR.calcsigma` /
    `opt_lambda` on one shared object, so σ is computed once per aperture
  - **IQ-range exclusion** carries into the batch: patterns outside the IQ range
    are neither averaged nor used as neighbours (served to pyebsdindex as flat
    patterns with σ = 0, which gives them zero weight). They are not removed: the
    output holds them as they were recorded, unprocessed, so the data set stays
    complete and they can still be indexed (v1.7.0; earlier versions wrote zeros).
    Originally built for identifying carbides by TKD on extraction replicas,
    where most of the map is empty film and only the particles carry patterns
  - pyebsdindex's `saturation_protect` and `dthresh` are exposed in the panel
    with their defaults (on, 0) and apply to σ, λ, batch and preview alike
  - Scan borders follow pyebsdindex: the CPU kernel shifts the window inward at
    edges, the OpenCL kernel mirrors indices; the preview zero-pads
  - Engine choice: pyebsdindex uses OpenCL when `pyopencl` sees a GPU, else
    numba on all cores; `KPP_GPU=0` forces the CPU engine (on 479x479 patterns a
    24-core CPU was faster than an 8 GB laptop GPU)
- **NPAR** as a uniform-weight special case (single-pixel preview only;
  use NLPAR with high λ for the equivalent batch behavior)
- **Aperture masking** for the σ² estimate and the NLPAR distance term — three
  modes covering the geometries this tool was actually used on:
  - `None` — every pixel counts (default)
  - `Circular (EBSD)` — `pyebsdindex.makeautomask`-compatible inscribed
    circle, for conventional phosphor-camera EBSD
  - `Auto-detect` — variance-based mask from the per-pixel std across the
    dataset, intended for direct electron detectors with non-circular active
    area (e.g. annular RKD and other direct electron detectors, where chip gaps
    and dead corners need excluding)
- **Processing pipeline** applied to preview and full-dataset save: static
  background (subtract or divide) → dynamic background (Gaussian high-pass) →
  per-pattern min-max normalize → brightness / contrast → CLAHE → clip. The
  full-dataset version (`Apply && Save`) batches the chain on GPU with chunked
  CuPy ops. With the per-pattern stretch off (absolute intensity preserved), the
  static background is applied mean-free (subtract) or mean-one (divide), as
  pyebsdindex's `backsub` does, so patterns keep their own brightness instead of
  collapsing to black at the clip (v1.6.1).
- The control column on the left sits in a splitter: drag the divider to resize it.
- `MainWindow.unload()` drops the loaded data set and everything derived from it (used by
  the spherical indexing workbench so that only one data set is in memory at a time).
- **Montage batch API** (v1.7.0, no dialogs, for a montage whose tiles are separate
  pattern files; the workbench drives it with a Cancel button). Everything runs in the
  GUI thread and keeps the status bar alive, like `Apply && Save`:
  - `MainWindow.compute_shared_background(paths, progress=None)` returns the mean
    pattern (float64) over all patterns of all the files, streamed in chunks without
    loading any of them, and keeps it as `self.shared_background`. All files must have
    the same pattern size and bit depth (else `ValueError` naming the file).
  - `MainWindow.save_processed(path, progress=None, cancel=None)` is `Apply && Save`
    without the file dialog: the loaded data set, current settings, source header copied
    byte for byte. Returns `dict(path, n_saved, n_total, seconds, source_label)`. When
    `cancel()` returns True between chunks, the partial file is deleted and
    `BatchCancelled` is raised.
  - `MainWindow.batch_process(jobs, shared_background=None, progress=None, cancel=None)`
    takes `[(source, output), ...]` and, per file, loads it, puts in the shared
    background, runs the full NLPAR if `NLPAR: Enable Preview` is checked (SR and λ from
    the panel: the λ optimized on the preview tile is reused), saves it, and unloads it
    so the next file starts from empty RAM. The IQ-range sliders apply per tile, as
    fractions of that tile's IQ range. A failed file is recorded (`ok: False`, `error`)
    and the batch goes on. The per-file results (`source, output, ok, error, n_saved,
    n_total, seconds, n_excluded`) are returned and kept in `self.batch_results`;
    `n_excluded` patterns were outside the IQ range and are in the output unprocessed.
    pyebsdindex's CPU NLPAR crashes on scans with fewer than 2·SR+1 rows, so the batch
    fails such a tile with an error instead (e.g. 3-row tiles need SR = 1).
- **OIM-compatible UP1/UP2 output**: pyebsdindex's `write_data` only emits the
  42-byte standard header, but v4 UP1 files have additional bytes between byte
  42 and `filePos`. This tool copies those bytes verbatim from the source file
  after the first write, so the output round-trips cleanly through OIM Analysis.

---

## Files

```
gpu_pattern_viewer.py     Main application (single-file PyQt5 GUI)
gpu_pattern_viewer.bat    Windows launcher (calls `python` from PATH — edit
                          if your interpreter is named differently)
run_gui.sh                Linux / HPC launcher (modules + venv from local/env.sh)
Kikuchi Pattern Processor.desktop   Desktop launcher for Linux desktops (edit the path)
LINUX_HPC_SETUP.md        Running on Linux or a Slurm cluster
selftest.py               Headless load / sigma / NLPAR / Apply && Save check with timings
patches/clnlpar_fixed.cl  Corrected copy of pyebsdindex's OpenCL NLPAR kernel (see below)
PLAN_NLPAR_PYEBSDINDEX.md Design notes for the pyebsdindex-based NLPAR path
UPSTREAM_pyebsdindex_opencl_sigma_bug.md   Draft bug report for pyebsdindex
compare_patterns.py       Static side-by-side plot of one original vs NLPAR pattern
interactive_compare.py    Clickable IQ map → side-by-side original / NLPAR
run_nlpar.py              Headless NLPAR runner (opt_lambda + calcnlpar)
_check_nlpar.py           Prints pyebsdindex UPFile source for introspection
local/                    Git-ignored: your site-specific launch settings and notes
```

---

## Requirements

- **Windows or Linux.** Developed on a Windows laptop; run on a Slurm cluster
  through an OnDemand interactive desktop (see `LINUX_HPC_SETUP.md`). The
  pyebsdindex OpenCL path of v1.6 has so far only been exercised on Windows;
  on Linux it needs `pyopencl` plus the vendor OpenCL runtime (CUDA module)
- **Python 3.11+** (developed on 3.14)
- **NVIDIA GPU with CUDA** (optional — falls back to CPU paths if CuPy is not
  available, slower but functionally identical)
- Python packages:
  - `PyQt5`
  - `pyebsdindex` (UP1/UP2 I/O, NLPAR batch and σ estimate, λ optimization)
  - `numpy`, `scipy`, `scikit-image`, `matplotlib`
  - `pyopencl` (optional: NLPAR batch on the GPU through pyebsdindex's OpenCL
    kernels; the NVIDIA driver provides the OpenCL runtime)
  - `cupy-cuda12x` (optional: GPU previews, Apply && Save and load statistics;
    pick the variant matching your CUDA toolkit)

### Install

```bash
pip install pyqt5 pyebsdindex numpy scipy scikit-image matplotlib psutil
pip install pyopencl       # NLPAR batch on the GPU (pyebsdindex OpenCL)
pip install cupy-cuda12x   # only if you have CUDA; pick the matching variant
```

### Run

```bash
python gpu_pattern_viewer.py
```

or, on Windows, double-click `gpu_pattern_viewer.bat`.

---

## Citations

If you use NLPAR or NPAR in published work, **please cite the methods papers**
— not this repository. The methodology is the contribution; the GUI is just
a wrapper that calls the existing implementations and adds a GPU batch path.

**NLPAR**

> Brewick, P.T., Wright, S.I., Rowenhorst, D.J. (2019). NLPAR: Non-local
> smoothing for enhanced EBSD pattern indexing. *Ultramicroscopy* **200**,
> 50–61. https://doi.org/10.1016/j.ultramic.2019.02.013

**NPAR**

> Wright, S.I., Nowell, M.M., Lindeman, S.P., Camus, P.P., De Graef, M.,
> Jackson, M.A. (2015). Introduction and comparison of new EBSD post-processing
> methodologies. *Ultramicroscopy* **159**, 81–94.
> https://doi.org/10.1016/j.ultramic.2015.08.001

---

## Acknowledgments

This tool would not exist without the libraries it stands on:

- **[pyebsdindex](https://github.com/USNavalResearchLaboratory/pyebsdindex)** —
  David Rowenhorst and team at the U.S. Naval Research Laboratory.
  All UP1/UP2 I/O, the NLPAR batch (OpenCL and numba), the σ estimate and the
  λ optimization routine are pyebsdindex. The single-pixel NLPAR preview is a
  reimplementation of the same Brewick et al. 2019 math; the result files are
  written through pyebsdindex's `UPFile.write_data` and post-patched for v4
  header compatibility.
- **[kikuchipy](https://github.com/pyxem/kikuchipy)** — the broader EBSD Python
  ecosystem (Håkon Wiik Ånes and contributors). Not currently a runtime
  dependency of this tool, but the obvious companion library for anything
  beyond what this does — dictionary indexing, dynamical simulation,
  visualization, master pattern handling, the works.
- **[CuPy](https://cupy.dev/)** — GPU acceleration.
- **[PyQt5](https://riverbankcomputing.com/software/pyqt/)** — GUI framework.
- **[scikit-image](https://scikit-image.org/)** — CLAHE implementation.
- **[SciPy](https://scipy.org/) / [NumPy](https://numpy.org/) /
  [matplotlib](https://matplotlib.org/)** — everything else.

---

## Author

**N. Max Vega Michalak, PhD**
Welding Engineering — The Ohio State University

---

## License

MIT. See [LICENSE](LICENSE).
