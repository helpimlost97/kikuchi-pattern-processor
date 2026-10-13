# Kikuchi Pattern Processor (NPAR/NLPAR)

A GPU-accelerated GUI for viewing and processing EBSD / TKD / RKD Kikuchi pattern
files (EDAX UP1/UP2 format). Provides on-demand and full-dataset NPAR and NLPAR
denoising, plus the standard pattern-processing pipeline (static / dynamic
background, brightness, contrast, CLAHE), wrapped in a PyQt5 interface that runs
sub-second previews on a laptop GPU and handles full-scan batch processing in
chunks sized to the available VRAM.

---

## ⚠️ This project is 100% vibe coded.

Every line was written through iterative LLM collaboration by someone who is not
a software engineer. It is a personal research tool, not production software. It
works on the author's data and machine. If it works on yours, great. If it
breaks in some interesting new way, congratulations — you now know roughly as
much about the internals as the author does.

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
  IQ range filtering, semi-transparent mask overlay for excluded patterns
- **NLPAR** using the canonical Brewick et al. 2019 formulation with
  dimensionless λ and a σ²-normalized weight: `w(i,j) = exp(-max(d̃, 0) / λ²)`
  - Full-dataset batch runs through **pyebsdindex's own NLPAR** (v1.6): its
    OpenCL kernels when `pyopencl` sees a GPU, numba on the CPU otherwise. The
    σ map from Optimize λ is reused by the batch. The viewer patches one line of
    pyebsdindex's OpenCL kernel at startup (neighbour σ indexed as the centre σ;
    see `PLAN_NLPAR_PYEBSDINDEX.md`); `KPP_NLPAR_KERNEL_PATCH=0` disables that.
  - On-demand single-pixel preview before committing to the full pass
  - **Optimize λ** delegates to `pyebsdindex.nlpar.NLPAR.opt_lambda`
  - The pre-1.6 CuPy shift-and-compare batch is still in the file behind
    `KPP_NLPAR_ENGINE=legacy` for comparison and will be removed
- **NPAR** as a uniform-weight special case (single-pixel preview only;
  use NLPAR with high λ for the equivalent batch behavior)
- **Aperture masking** for the σ² estimate and the NLPAR distance term — three
  modes covering the geometries this tool was actually used on:
  - `None` — every pixel counts (default)
  - `Circular (EBSD)` — `pyebsdindex.makeautomask`-compatible inscribed
    circle, for conventional phosphor-camera EBSD
  - `Auto-detect` — variance-based mask from the per-pixel std across the
    dataset, intended for direct electron detectors with non-circular active
    area (e.g. Thermo Fisher TruePix RKD, where chip gaps and dead corners
    need excluding)
- **Processing pipeline** applied to preview and full-dataset save: static
  background (subtract or divide) → dynamic background (Gaussian high-pass) →
  per-pattern min-max normalize → brightness / contrast → CLAHE → clip. The
  full-dataset version (`Apply && Save`) batches the chain on GPU with chunked
  CuPy ops.
- **OIM-compatible UP1/UP2 output**: pyebsdindex's `write_data` only emits the
  42-byte standard header, but v4 UP1 files have additional bytes between byte
  42 and `filePos`. This tool copies those bytes verbatim from the source file
  after the first write, so the output round-trips cleanly through OIM Analysis.

---

## Files

```
gpu_pattern_viewer.py     Main application (~1200 lines, single-file PyQt5 GUI)
gpu_pattern_viewer.bat    Windows launcher (calls `python` from PATH — edit
                          if your interpreter is named differently)
compare_patterns.py       Static side-by-side plot of one original vs NLPAR pattern
interactive_compare.py    Clickable IQ map → side-by-side original / NLPAR
run_nlpar.py              Headless NLPAR runner (opt_lambda + calcnlpar)
_check_nlpar.py           Prints pyebsdindex UPFile source for introspection
```

---

## Requirements

- **Windows** (only platform tested)
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

**N. Max Vega Michalak**
PhD Candidate, Welding Engineering — The Ohio State University

---

## License

MIT. See [LICENSE](LICENSE).
