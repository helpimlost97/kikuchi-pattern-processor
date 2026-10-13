# Plan: NLPAR batch through pyebsdindex (decided 2026-10-13)

Status of each phase is tracked at the bottom. Read this before touching the NLPAR
code in `gpu_pattern_viewer.py`.

## Decision

The full-dataset NLPAR batch moves from the self-written CuPy shift-and-compare kernel
(`_apply_nlpar_gpu`) to pyebsdindex's `nlpar` module: OpenCL kernels (`pyebsdindex/opencl/
clnlpar.cl`) when pyopencl sees a GPU, the numba CPU implementation otherwise. Lambda
optimisation already used pyebsdindex. The motive is to run the accepted, publicly used
NRL code (Brewick, Wright, Rowenhorst, Ultramicroscopy 2019; PyEBSDIndex 0.3.x) rather than
a private re-implementation. Speed only has to be "not way slower".

Source-level estimate (no benchmark run yet): on 479x479 RKD patterns with an 8 GB NVIDIA
card the two are roughly a wash (pyebsdindex tiles are ~270 OpenCL work-items there because
NVIDIA reports max_mem_alloc = VRAM/4 and pyebsdindex uses a quarter of that per tile); on
small EBSD patterns pyebsdindex should be several times faster; on OSC's H100 the tiles are
~8x larger. Worst plausible case is ~1.5x slower than today on the laptop.

## What already exists (from the OSC port, v1.0-v1.5)

- `_apply_nlpar_cpu` runs `nlpar.NLPAR(...).calcnlpar(...)` with the aperture passed as
  `nlobj.mask`, `automask=False`, truncated-row cap via `nlobj.nrows`.
- `RamUPFile` / `RamOutUPFile` serve pyebsdindex's `read_data` from the in-RAM dataset and
  catch its `write_data` into the in-RAM result (`_serve_nlpar_from_ram`,
  `_nlpar_result_to_ram`). The OpenCL class uses the same rectangle reads and the same
  `write_data`, so it needs no new plumbing.
- Workflow gates: Optimize Lambda -> Apply NLPAR to Full Dataset -> Apply && Save.

## Known semantic differences vs the CuPy path

| Topic | CuPy path (ours) | pyebsdindex |
|---|---|---|
| Scan borders | zero-pad, missing neighbours dropped | mirrored indices |
| Saturation | none | `saturation_protect`: pixels >= 0.9961*max in either pattern excluded per pair, pixel count N varies per pair |
| Pixel count in d-tilde | fixed aperture count N_p | per-pair count n |
| Distance threshold | none | `dthresh` (default 0) |
| Per-pattern exclusion (IQ mask) | excluded as outputs AND as neighbours | none |
| Search radius | any | GPU kernel clips SR <= 10 (sigma SR <= 7) |
| Chunking | rows, VRAM-budgeted, host RAM capped | tiles, `min(max_mem_alloc/4, 4 GB)`; `chunksize` kwarg ignored on GPU |
| Output | file or RAM | file or RAM (via RamOutUPFile) |

## Phases

### Phase 0: environment  (DONE on laptop 2026-10-13; OSC pending)
- pyopencl installed into the app's Python (laptop: pythoncore-3.14, pyopencl 2026.1.4).
  The hyperspy bundle used by the spherical-indexing project has its own copy; the pattern
  processor does not see that environment.
- `KPP_GPU=0` now also imports `pyebsdindex.nlpar_cpu` so CPU-only runs stay CPU
  (OpenCL would otherwise still find the device).
- requirements.txt documents pyopencl as the optional NLPAR GPU extra.
- OSC: `pip install pyopencl` in `~/EBSD/venvs/kikuchi`; verify on a GPU node that
  `module load cuda/12.9.1` exposes the NVIDIA OpenCL ICD (`python -c "import pyopencl as
  cl; print(cl.get_platforms())"`). This is the main unknown.

### Phase 1: route the batch to pyebsdindex
- `apply_nlpar_batch` always calls the pyebsdindex path (rename `_apply_nlpar_cpu` to
  `_apply_nlpar_pyebsdindex`); pyebsdindex decides GPU vs CPU.
- One `NLPAR` object per (dataset, aperture, SR) kept on the window so the sigma computed
  during Optimize Lambda is reused by the batch; invalidate on file load, aperture change,
  SR change (same events that clear `_lambda_optimized`).
- Progress: wrap the output file object's `write_data` (fires once per finished tile) to
  drive the status bar / ETA for both RAM and temp-file outputs.
- Cap the SR spin box at 10.
- Keep the CuPy batch reachable only via `KPP_NLPAR_ENGINE=legacy` until Phase 4.

### Guiding rule (Max, 2026-10-13)
Use pyebsdindex modules wherever they exist; homebrew code only for (a) features
pyebsdindex does not have, which Phase 3 adds on top of it, and (b) speedups that are
clearly worth it, to be discussed later. Homebrew inventory in the viewer and its fate:

| Homebrew piece | pyebsdindex replacement | Phase |
|---|---|---|
| single-pixel NLPAR / NPAR preview (`_compute_nlpar_single`, NPAR average) | `calcnlpar` on a (2SR+1)^2 sub-scan around the pixel, served from RAM, full-scan sigma preset on the object; NPAR = same call with a very large lambda | 2 |
| sigma map (`_compute_sigma_map_gpu/_cpu`) | `calcsigma` on the shared NLPAR object | 2 |
| circular aperture (copy of `makeautomask`) | call `nlpar.NLPAR.makeautomask` | 2 |
| CuPy NLPAR batch (`_apply_nlpar_gpu`) | `calcnlpar` (done in Phase 1) | delete after Phase 4 run |
| static/dynamic background, normalize, CLAHE, B/C, IQ map, global stretch | not in pyebsdindex (only a Gaussian-surface `backsub`); kikuchipy has all of them | later discussion |
| parallel UP reader, RAM serving (`read_patterns_parallel`, `RamUPFile`) | pyebsdindex reads row by row (slow); speed item | later discussion |

### Phase 2: sigma, preview and aperture through pyebsdindex
- Sigma map from `nlobj.calcsigma()` (returns sigma, dist, countnn) on the shared object;
  store sigma^2 per aperture for display/diagnostics; delete `_compute_sigma_map_gpu/_cpu`.
- Preview: build a (2SR+1) x (2SR+1) sub-scan around the clicked pixel (RamUPFile view
  when in RAM, otherwise read the rows), preset `nlobj.sigma` to the matching slice of the
  full-scan sigma, run `calcnlpar` with the current engine, take the centre pattern.
  Interior pixels then equal the batch exactly; at scan edges the sub-scan border rule
  (CPU shifts inward, GPU mirrors) is applied to a smaller block, so a small difference
  is accepted and documented. NPAR preview = same call, lambda = 1e6.
- Circular aperture via `nlpar.NLPAR.makeautomask`; delete the local copy.
- Acceptance: interior preview == batch output at that pixel within integer rounding, on
  both engines.

### Phase 3: features pyebsdindex lacks
- IQ-range exclusion: short term, zero excluded patterns in the output after the batch
  (they still contribute as neighbours). Proper fix: upstream patch adding an optional
  per-pattern weight array to the pyebsdindex kernels (public domain, NRL); separate step.
- Borders: accept mirroring (published behaviour).
- Expose `saturation_protect` and `dthresh` as controls with pyebsdindex defaults.

### Phase 4: validation and docs
- **OPEN DECISION: which pattern set to validate on is undecided.** Candidates on the
  laptop: `FGHAZ 25/map20250109115929899.up1` (0.67 GB), `FGHAZ 25/ICHAZ25_20000_50nm.up1`
  (6.5 GB), `Base Metal/Area 1/map20260308181623025.up1` (7.8 GB); on OSC the 32 GB Ni RKD
  scan. Ask Max before running anything (no unrequested benchmarks on the laptop).
- Compare pyebsdindex GPU vs pyebsdindex CPU (expect near-identical), then vs the legacy
  CuPy path (expect differences only at borders / saturated pixels). Time both.
- Rewrite `osc_gpu_selftest.py` to compare pyebsdindex GPU vs CPU; run once on Cardinal GPU.
- Delete the CuPy NLPAR batch and sigma code; CuPy stays for Apply && Save, load stats.
- Update README, OSC_HANDOFF.md, requirements; bump to 1.6.0.

## Finding during Phase 1: pyebsdindex OpenCL kernel bug (2026-10-13)

`pyebsdindex/opencl/clnlpar.cl`, kernel `calcnlpar`, "calculate the weights" loop:

    sigma_ij = sigma[indx_xy];     // centre's sigma, for every neighbour
                                   // meant: sigma_ij = sigma[indx_ij];

so the GPU weight normalises with 2*sigma_i^2 instead of sigma_i^2 + sigma_j^2. The
`normd` kernel and the numba CPU kernel index correctly. Present in 0.3.9.1 (installed)
and in upstream `main` on 2026-10-13; no issue filed upstream for it. Measured on a
synthetic 24x16 scan with noise sigma 6 (left half) / 35 (right half), SR 3:

| comparison | interior mean abs diff | interior max | noise-boundary cols |
|---|---|---|---|
| CPU vs GPU as shipped | 2.8 | 119 | 14.1 |
| CPU vs GPU with the one-line fix | 0.000 | 1 | 0.000 |

The viewer applies the fix at runtime (`_patch_pyebsdindex_nlpar_kernel`: patched copy
of the kernel in the temp dir, pyebsdindex's `get_context` redirected to it; no-op once
upstream fixes it; `KPP_NLPAR_KERNEL_PATCH=0` disables). **To do: report / PR upstream
(Max decides).** Phase 4's expectation "GPU vs CPU near-identical" holds only with the
fix and only in the interior: borders differ by design (CPU shifts the window inward,
GPU mirrors) and the saturation threshold differs (0.999 vs 0.9961 of max).

## Status
- 2026-10-13: decision taken; Phase 0 done on the laptop (pyopencl 2026.1.4 installed,
  KPP_GPU=0 import guard, requirements note). OSC side of Phase 0 not started.
- 2026-10-13: Phase 1 done on the laptop (v1.6.0.dev1): batch routed to pyebsdindex,
  shared NLPAR object with sigma reuse, progress hook, legacy engine behind
  KPP_NLPAR_ENGINE=legacy, kernel bug work-around. Verified on a synthetic scan only
  (see OSC_HANDOFF.md v1.6.0.dev1). Not yet run on real data, not yet pushed/run on OSC.
