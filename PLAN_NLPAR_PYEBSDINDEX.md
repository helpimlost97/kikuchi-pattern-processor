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
small EBSD patterns pyebsdindex should be several times faster; on an H100 the tiles are
~8x larger. Worst plausible case is ~1.5x slower than today on the laptop.

## What already exists (from the HPC port, v1.0-v1.5)

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

### Phase 0: environment  (DONE on laptop 2026-10-13; HPC pending)
- pyopencl installed into the app's Python (laptop: pythoncore-3.14, pyopencl 2026.1.4).
  The hyperspy bundle used by the spherical-indexing project has its own copy; the pattern
  processor does not see that environment.
- `KPP_GPU=0` now also imports `pyebsdindex.nlpar_cpu` so CPU-only runs stay CPU
  (OpenCL would otherwise still find the device).
- requirements.txt documents pyopencl as the optional NLPAR GPU extra.
- HPC cluster: `pip install pyopencl` in the venv; verify on a GPU node that
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

### Guiding rule (maintainer decision, 2026-10-13)
Use pyebsdindex modules wherever they exist; homebrew code only for (a) features
pyebsdindex does not have, which Phase 3 adds on top of it, and (b) speedups that are
clearly worth it, to be discussed later. Homebrew inventory in the viewer and its fate:

| Homebrew piece | pyebsdindex replacement | Phase |
|---|---|---|
| single-pixel NLPAR / NPAR preview (`_compute_nlpar_single`, NPAR average) | **KEPT by maintainer decision (2026-10-13), deliberate exception to the rule**: it is the fast interactive tool he likes. It gets pyebsdindex's sigma and pyebsdindex's saturation rule so interior values match the batch; it keeps its own border behaviour (zero-pad, fewer neighbours at scan edges). Do not "clean it up" into a pyebsdindex call. | 2 |
| sigma map (`_compute_sigma_map_gpu/_cpu`) | `calcsigma` on the shared NLPAR object | 2 |
| circular aperture (copy of `makeautomask`) | call `nlpar.NLPAR.makeautomask` | 2 |
| CuPy NLPAR batch (`_apply_nlpar_gpu`) | `calcnlpar` (done in Phase 1) | deleted 2026-10-13 after the Phase 4 run |
| static/dynamic background, normalize, CLAHE, B/C, IQ map, global stretch | not in pyebsdindex (only a Gaussian-surface `backsub`); kikuchipy has all of them | later discussion |
| parallel UP reader, RAM serving (`read_patterns_parallel`, `RamUPFile`) | pyebsdindex reads row by row (slow); speed item | later discussion |

### Phase 2: sigma and aperture through pyebsdindex; preview aligned, not replaced
- Sigma map from `nlobj.calcsigma()` on the shared NLPAR object (the same call Optimize
  Lambda makes, so after Optimize Lambda the map is already there and the first click is
  instant; otherwise the first click pays the one-time sigma pass, as today). Store
  sigma^2 per aperture (`sigma_sq_map` = sigma**2); delete `_compute_sigma_map_gpu/_cpu`
  and the median patch of zero sigma (pyebsdindex handles degenerate pairs itself).
- Preview (`_compute_nlpar_single`, kept, see table): per click it stays the NumPy loop
  over the (2SR+1)^2 neighbourhood, so response time is unchanged. Two alignments with
  the batch kernel: (1) pixel pairs where either value is >= the saturation threshold are
  dropped from d and the pixel count n is per pair (pyebsdindex `saturation_protect`,
  CPU threshold 0.999*max, GPU 0.9961*max; use the current engine's); (2) sigma from the
  pyebsdindex map. Border behaviour stays ours (zero-pad, fewer neighbours at edges).
  Expose `dthresh` only if it is ever exposed for the batch (default 0 = no effect).
- NPAR preview: unchanged (uniform weights over the same neighbourhood).
- Circular aperture via `nlpar.NLPAR.makeautomask`; delete the local copy.
- Acceptance: for interior pixels away from saturation the preview equals the batch
  output within integer rounding on both engines; near saturated pixels it matches the
  engine it was aligned with; edge pixels may differ (documented).

### Phase 3: features pyebsdindex lacks  (DONE 2026-10-13, v1.6.0.dev3)
- IQ-range exclusion, implemented without changing pyebsdindex and with the legacy
  semantics (excluded patterns neither averaged nor used as neighbours): the input file
  object serves excluded patterns as zeros (`exclude` flag on RamUPFile / MaskedUPFile)
  and their sigma is 0 in the array handed to calcnlpar, so their weight is exp(-huge) = 0;
  outputs at excluded positions are zeroed explicitly. No upstream patch needed for this.
- Borders: pyebsdindex behaviour accepted and documented (numba shifts inward, OpenCL
  mirrors). Preview keeps zero-pad.
- `saturation_protect` and `dthresh` exposed in the NLPAR panel with pyebsdindex
  defaults, wired to lambda, sigma, batch and preview.
- Upstream bug report drafted: UPSTREAM_pyebsdindex_opencl_sigma_bug.md (not posted).

### Phase 4: validation and docs
- **Dataset decided (maintainer decision, 2026-10-13): `FG10 3 Disc RKD.up1`** (207x138, 479x479 uint8, 6.55 GB). Headless checks on the first 24 rows done
  (see the maintainer's local notes); The full scan was then run in the GUI on the laptop without issues (2026-10-13). Results are summarised in the maintainer's local notes.
- Compare pyebsdindex GPU vs pyebsdindex CPU (expect near-identical), then vs the legacy
  CuPy path (expect differences only at borders / saturated pixels). Time both.
- `osc_gpu_selftest.py` (compared the deleted CuPy code) removed; `selftest.py` is the headless check. Run it once on a cluster GPU node.
- CuPy NLPAR batch and sigma code deleted (2026-10-13); CuPy stays for Apply && Save, load stats.
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
(maintainer decides).** Phase 4's expectation "GPU vs CPU near-identical" holds only with the
fix and only in the interior: borders differ by design (CPU shifts the window inward,
GPU mirrors) and the saturation threshold differs (0.999 vs 0.9961 of max).

## Finding during Phase 4: pyebsdindex OpenCL kernels fault on pattern sizes that are not a multiple of 16 (2026-10-13)

`*(__global float16*)` casts in `calcsigma` / `calcnlpar` (misaligned when npix % 4 != 0,
and an unguarded partial last vector that overruns into the next pattern with a += race).
479x479 (229441 px) crashes with INVALID_COMMAND_QUEUE within a second. Fixed in
`patches/clnlpar_fixed.cl` (vload16/vstore16, element-wise tail, register accumulation),
used by the viewer when the installed kernel's sha1 is known; otherwise numba is used for
such datasets. Interior output equals numba within 1 grey level on all tested sizes.
Also in the upstream report draft.

## Finding during Phase 4: sigma and distance must use the same pixel set (2026-10-13)

The NLPAR weight is a z-score with sqrt(2N) ~ 661 in the denominator for 479x479 patterns,
so a few % bias between the sigma estimate and the distance (e.g. saturated pixel pairs
dropped from one but not the other) shifts it by >10 and zeroes every neighbour. This is
why the legacy CuPy batch, fed pyebsdindex's saturation-protected sigma since Phase 2,
returned the input unchanged on FG10 RKD. pyebsdindex is self-consistent in both engines;
the legacy path is comparable only with saturation protect OFF (then: mean diff 0.02,
max 1). Any future homebrew that reuses pyebsdindex's sigma must apply the same
saturation rule to its distances (the preview does).

## Status
- 2026-10-13: decision taken; Phase 0 done on the laptop (pyopencl 2026.1.4 installed,
  KPP_GPU=0 import guard, requirements note). HPC side of Phase 0 not started.
- 2026-10-13: Phase 1 done on the laptop (v1.6.0.dev1): batch routed to pyebsdindex,
  shared NLPAR object with sigma reuse, progress hook, legacy engine behind
  KPP_NLPAR_ENGINE=legacy, kernel bug work-around. Verified on a synthetic scan only
  . Not yet run on real data, not yet run on the HPC cluster.
- 2026-10-13: Phase 2 done on the laptop (v1.6.0.dev2): sigma via pyebsdindex calcsigma on
  the shared object, homebrew sigma code deleted, preview aligned (pyebsdindex sigma +
  saturation rule, own borders), makeautomask called directly. Synthetic-scan checks on
  both engines: interior preview == batch within 1 grey level, 1-2 ms/click, sigma reused
  (1 ms), direct-click path computes sigma once and matches the post-Optimize-Lambda map.
  Still not run on real data; HPC cluster untouched.
- 2026-10-13: Phase 3 done on the laptop (v1.6.0.dev3): IQ-range exclusion via zero-served
  patterns + sigma 0, saturation_protect / dthresh controls, upstream report drafted.
  Synthetic-scan checks pass on both engines. Legacy CuPy batch kept until the Phase 4 run. Still not run on real data; HPC cluster untouched.
- 2026-10-24 (v1.7.0.dev2): the excluded patterns are no longer zeroed in the output. The NLPAR
  batch puts the original patterns back at those positions (_restore_excluded_output) and
  Apply && Save copies them through unprocessed, so they stay in the data set and can be indexed.
- 2026-10-13: Phase 4, laptop part (v1.6.0.dev4): FG10 RKD 24-row subset run on all three
  engines; OpenCL kernel alignment/tail bugs found and fixed via patches/clnlpar_fixed.cl;
  OpenCL == numba (interior max 1); legacy == pyebsdindex with saturation protect off;
  timings in the maintainer's local notes. Remaining: HPC side (pyopencl in the venv, OpenCL ICD check), tag 1.6.0.

- 2026-10-13: full 138-row FG10 run in the GUI on the laptop worked. Legacy CuPy batch and
  helpers deleted. Repository generalised: machine/account-specific files moved to a
  git-ignored `local/` folder, launchers parameterised (`local/env.sh`).
