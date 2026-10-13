# OSC handoff — state of this folder as of 2026-09-25

Read this first if you are a new Claude session (or Max) picking up the
Kikuchi Pattern Processor on the Ohio Supercomputer Center. It records what
was done, what was verified, and what is deliberately NOT done yet.
`OSC_SETUP.md` (from the original repo) is still the general Linux/OSC guide;
this file is the delta on top of it.

## v1.6.0.dev5 (2026-10-13): code-review fixes before the first push

A review of the day's commits (origin/main..HEAD) found ten issues; all fixed:
- IQ exclusion on the OpenCL engine made every tile with an excluded pattern compute
  four times: the all-zero served pattern gave an all-zero output, which pyebsdindex's
  calcnlpar_cl takes for its Apple-silicon GPU failure and re-queues up to 3 times.
  Excluded patterns are now served as constant 1 (same exp(-huge) = 0 weight), outputs
  still zeroed afterwards. 8-row FG10 subset with 3 patterns excluded (3 tiles affected): batch time
  17.0 s vs 17.0 s without exclusion (would have been ~2x with the retries).
- Optimize Lambda ignored the Distance threshold on the OpenCL engine (nlpar_cl.opt_lambda
  swallows the keyword); the value is now set on the object first, both engines.
- A dthresh change no longer throws away sigma and the shared object (only saturation
  protect does); it only re-arms the lambda gate.
- Engine label / preview saturation factor follow the object actually in use
  (`_nlpar_engine_of`), not the module-level choice, for the forced-numba fallback.
- Busy guard (`_nlpar_busy`): while a pyebsdindex call runs, Optimize Lambda, the batch,
  file load, aperture and parameter changes are refused with a message instead of
  re-entering through the progress hook's processEvents; a preview click during a batch
  shows the raw pattern if sigma is not there yet.
- Sigma computed by calcnlpar itself is kept on the object (was discarded when no
  exclusion was active); sigma for an exclusion batch is ensured before the progress
  hook is installed.
- Preview mirrors both kernels for a neighbour with no comparable pixels (n = 0): weight 1.
- MaskedUPFile no longer copies a slab tile twice (slab reads are masked per row already).
- Derived kernel copies go to a per-user temp directory.
- Runtime transform of an unknown pyebsdindex clnlpar.cl (`_transform_clnlpar`) derives
  the full fix when every anchor matches; verified byte-identical output to the shipped
  copy. Upstream main is byte-identical to 0.3.9.1 today.
All smoke tests re-run on both engines (synthetic, streamed, uint16, exclusion, 8-row real).

## v1.6.0.dev4 (2026-10-13): first real data (Phase 4, laptop part). Two more pyebsdindex OpenCL bugs, fixed in a shipped kernel copy

Test set (Max's choice): first 24 rows of `FG10 3 Disc RKD.up1` (207 cols x 138 rows,
479x479 uint8, 6.55 GB; subset 1.14 GB), aperture Auto-detect (N_p 218736/229441), SR 6,
lambda fixed at 1.39 (Max's historical value; Optimize Lambda gives 1.54-1.55 here).

**pyebsdindex's OpenCL kernels crash on 479x479 patterns** (`clFinish failed:
INVALID_COMMAND_QUEUE`, within ~1 s, even on an 8x6 scan; TdrDelay is 10 s so not a
watchdog). Cause: both kernels read patterns through `*(__global float16*)` casts and
`calcnlpar` accumulates output through the same cast. For a pixel count that is not a
multiple of 16 this is (a) misaligned (faults when npix % 4 != 0: 49x49, 50x51, 479x479,
479x481) and (b) the last vector overruns into the next pattern / past the buffer (52x51
faults too; 50x50 survives by allocation luck) with a read-modify-write race on the
neighbour's first pixels. Standard EBSD sizes (60x60, 120x120, 480x480) are multiples of
16, so upstream never saw it. Fix: `patches/clnlpar_fixed.cl`, a complete copy of the
0.3.9.x kernel with vload16/vstore16, element-wise handling of the partial tail vector, a
register accumulator with one store per vector, and the sigma-index fix. The viewer uses
it when the installed `clnlpar.cl` has a known sha1 (`_KNOWN_SHIPPED_CLNLPAR`); for an
unknown upstream version it falls back to the one-line sigma fix and routes datasets with
npix % 16 != 0 to the numba engine (`_gpu_kernel_safe_for`). Verified: builds, runs on
every size that failed, and the interior output equals numba's within 1 grey level on
49x49, 50x50, 61x61, 95x97 and the real 479x479 subset (mean 0.03, max 1). Added to
`UPSTREAM_pyebsdindex_opencl_sigma_bug.md`.

**Results on the 24-row subset (laptop, RTX 5070 8 GB, 24 threads):**

| step | pyebsdindex OpenCL | pyebsdindex numba | legacy CuPy |
|---|---|---|---|
| Optimize Lambda (incl. sigma) | 2.9 s | 3.9 s | n/a |
| NLPAR batch, SR 6 | 67.5 s | 42.6 s | 110.8 s |
| full 138-row scan, extrapolated | ~6.5 min | ~4 min | ~10.5 min |
| preview, 479x479, SR 6 | 0.24-0.40 s/click | 0.6 s | n/a |

As estimated from source: with 479x479 patterns the OpenCL tiles are ~270 work-items and
numba on 24 cores beats it. `KPP_GPU=0` picks numba.

**Agreement:** pyebsdindex OpenCL vs numba, interior: mean 0.03, max 1 grey level (border
mean 1.8: different border rules). Legacy CuPy vs pyebsdindex with saturation protect
OFF: mean 0.02, max 1 (same mathematics, confirmed on real data). Preview vs batch at
6 interior pixels: within 1.01 (OpenCL) / 2.3 (numba; tile-max vs data-max saturation
threshold, see below).

**Why the legacy batch is no longer comparable with saturation protect ON.** The NLPAR
weight is a z-score, (d - 2N sigma^2) / (2 sigma^2 sqrt(2N)); with N ~ 2.2e5 the sqrt(2N)
= 661 makes it hypersensitive to sigma: a 5 % bias in sigma^2 shifts the z-score by ~16.
Since Phase 2 the legacy kernel takes pyebsdindex's sigma (saturated pixel pairs
excluded) but computes d with them included; this RKD scan has 0.6 % saturated pixels
and 36 % of pixel positions saturated in at least one neighbour, so every neighbour
weight collapsed and the legacy output equalled the input (|out - in| 0.5). Sigma and
d must be estimated on the same pixel set; pyebsdindex does that in both engines.
Legacy now prints a warning when saturation protect is on; it will be deleted.

**Saturation protect matters on this data**: OFF vs ON differ by mean 1.2, p99 7, max 74
grey levels in the interior. ON is pyebsdindex's default and what the panel defaults to.

**Preview** vectorised over the neighbourhood (was a per-neighbour Python loop with
masked copies, 0.9 s/click on 479x479): now 0.24 s (saturation off) / 0.40 s (on), of
which 0.1 s is reading the 169 patterns.

Not done: the full 138-row run in the GUI (Max does that next), OSC side.

## v1.6.0.dev3 (2026-10-13): features pyebsdindex lacks, built around it (Phase 3)

- **IQ-range exclusion in the batch, without touching pyebsdindex.** pyebsdindex has no
  per-pattern mask. Excluded patterns are served to it as all-zero patterns (RamUPFile /
  new MaskedUPFile `exclude` flag, set only for the duration of the batch) and their
  sigma is set to 0 in the sigma array handed to calcnlpar (true sigma restored after).
  For any real centre the normalised distance to a zero pattern is then enormous, so its
  weight is exp(-huge) = 0: excluded patterns drop out as neighbours exactly as in the
  legacy CuPy batch. Their own output is zeroed explicitly (`_zero_excluded_output`, RAM
  or file). Verified on the synthetic scan, both engines: excluded outputs are 0, 17 of
  19 direct neighbours of excluded patterns changed vs the unmasked batch (the other 2 had
  ~0 weight already), and the preview (which skips excluded neighbours itself) matches
  the exclusion-aware batch within 1 grey level; RAM and file results identical.
- **pyebsdindex parameters in the panel**: "Saturation protect" (default on) and
  "Distance threshold" (dthresh, default 0). Both go to opt_lambda, calcsigma, calcnlpar
  and the preview; changing either invalidates sigma, the shared object and the lambda
  gate. Verified: with saturation protect off and dthresh 0.5 the preview still matches
  the batch within 1 grey level on both engines.
- Scan borders: kept pyebsdindex's behaviour (numba shifts the window inward, OpenCL
  mirrors). Documented, not changed.
- Batch button now says which engine pyebsdindex resolved to.
- `UPSTREAM_pyebsdindex_opencl_sigma_bug.md`: draft issue + PR text for the kernel bug,
  not posted (Max decides).
- Legacy CuPy batch kept (KPP_NLPAR_ENGINE=legacy) by Max's decision until the Phase 4 run.

## v1.6.0.dev2 (2026-10-13): sigma and aperture through pyebsdindex, preview aligned (Phase 2)

Rule from Max (2026-10-13): use pyebsdindex modules wherever they exist; homebrew only
for features pyebsdindex lacks (Phase 3) or clearly worthwhile speedups (later). One
explicit exception: the single-pixel NLPAR/NPAR preview stays our NumPy code because it
must be near-instant on click and Max likes it.

- `_ensure_sigma_map` now takes sigma from the shared pyebsdindex object (`calcsigma`,
  nn=1, saturation_protect): after Optimize Lambda it is already there (1 ms); a click
  before Optimize Lambda pays the one-time calcsigma pass. `sigma_sq_map` = sigma**2,
  rows never recorded (truncated map) are 0. The CuPy and threaded-CPU sigma code
  (`_compute_sigma_map_gpu/_cpu`, ~140 lines) is deleted.
- `_compute_nlpar_single` (preview) aligned with the batch kernel: pyebsdindex sigma, and
  the saturation rule (pixel pairs with either value >= 0.9961*max (OpenCL) / 0.999*max
  (numba) of the data max are dropped, pixel count is per pair). Borders stay ours
  (zero-pad). Verified on the synthetic scan: interior preview == batch within 1 grey
  level on both engines, 1-2 ms per click; corner pixels differ (17-29 levels) because
  the border rules differ, as documented.
- Circular aperture calls `nlpar.NLPAR.makeautomask` instead of a local copy.
- `osc_gpu_selftest.py` now exits with a message: it compared the removed CuPy code.
  Rewrite is Phase 4.

## v1.6.0.dev1 (2026-10-13): NLPAR batch through pyebsdindex (Phase 1 of PLAN_NLPAR_PYEBSDINDEX.md)

Decision (Max, 2026-10-13): the full-dataset NLPAR runs through pyebsdindex's own
`nlpar` module from now on, because it is the accepted, publicly used code. pyebsdindex
picks the engine at import: its OpenCL kernels when `pyopencl` sees a GPU (laptop: RTX
5070 via the NVIDIA driver; OSC: needs `pip install pyopencl` in `~/EBSD/venvs/kikuchi`
and the NVIDIA OpenCL ICD on a GPU node, **not yet checked**), numba on the CPU
otherwise. `KPP_GPU=0` now also forces the CPU module (OpenCL would otherwise still
find the device). Startup prints `[NLPAR] engine: ...`.

What changed in `gpu_pattern_viewer.py`:
- `apply_nlpar_batch` -> `_apply_nlpar_pyebsdindex` (the old `_apply_nlpar_cpu` name is
  kept as an alias for the selftest scripts). The CuPy shift-and-compare batch is only
  reachable with `KPP_NLPAR_ENGINE=legacy`, for the Phase 4 comparison; it will be deleted.
- One pyebsdindex `NLPAR` object per (dataset, aperture, rows) (`_get_nlobj`): Optimize
  Lambda computes sigma (calcsigma, nn=1) and the batch reuses it, so no second sigma pass.
  Cleared on file load and aperture change together with the lambda gate.
- Progress: pyebsdindex exposes none, so `_hook_nlpar_progress` wraps the input object's
  `read_data` and the output object's `pat_writer` (one call per written tile row) for a
  percent / ETA in the status bar. Works for the RAM twins and real files.
- Temp output name is now `_nlpar_result.up1/.up2` (was `_cpu_nlpar_result`).
- **pyebsdindex OpenCL kernel bug, worked around**: `clnlpar.cl`, kernel `calcnlpar`,
  weight loop reads `sigma[indx_xy]` (centre) instead of `sigma[indx_ij]` (neighbour), so
  the GPU weight uses 2*sigma_i^2 instead of sigma_i^2 + sigma_j^2. Present in 0.3.9.1 and
  upstream main. On a synthetic scan with two noise levels the shipped GPU output differs
  from the CPU output by up to 119 grey levels along the noise boundary; with the one-line
  fix, GPU == CPU exactly in the scan interior. `_patch_pyebsdindex_nlpar_kernel()` writes
  a fixed copy of the kernel to the temp dir at startup and redirects pyebsdindex's
  `get_context` to it (no-op once upstream fixes the line; `KPP_NLPAR_KERNEL_PATCH=0`
  disables). Not yet reported upstream (Max's call).
- Remaining GPU-vs-CPU difference inside pyebsdindex is at the scan border only: the
  numba kernel shifts the window inward at edges, the OpenCL kernel mirrors indices.
  Also the saturation threshold differs (CPU 0.999*max, GPU 0.9961*max).

Verified headlessly (synthetic 24x16 scan, 48x48 uint8, SR 3): gate refuses the batch
before Optimize Lambda; lambda via OpenCL and numba agree to 3 decimals; sigma reused by
the batch; RAM result == file result byte-for-byte on both engines; progress hook counted
every written pattern; circular aperture reaches pyebsdindex as `nlobj.mask`
(N_p 1741/2304) and re-arms the gate. Not yet run on a real scan (Phase 4; dataset
undecided).

## v1.5.0 (2026-10-10): UP2 / 16-bit support (uncommitted, not yet run on real data)

Target dataset: `/fs/scratch/PAS2328/vegamichalak1/Scratch Downloads/FG10 TKD/FG TKD_FG10_Area 1_OIM Map 2.up2`
(note: project **PAS2328** scratch, not PAS2325): v4 header (58 B), 363x291 = 105633 patterns,
240x240 px uint16, data max ~40k (so the effective full scale is 65535), 12.2 GB. First real
run 2026-10-10 on a 96-core / 475 GB CPU desktop (no GPU): load, Optimize Lambda, CPU NLPAR
(SR 6) in RAM, Apply && Save with normalize + static background subtract -> complete, valid
`..._NLPAR6_Norm_BackSub.up2`. Only visible symptom: a harmless Qt `XCB error: 3 (BadWindow)
... TranslateCoords` line when the save dialog closed, plus a status bar frozen for the whole
127-row chunk -> v1.5.1 adds in-chunk progress/ETA, flushes the dialog before the loop, and
sizes CPU save chunks at 42 B/px instead of 18. The app
already opened `.up2` files, but the processing chain assumed uint8: B/C centred on
127.5, outputs clipped to 255 then written as uint16 (dark files), raw preview on a
0-255 scale, and the NLPAR temp files / save dialog forced `.up1`, which pyebsdindex
reads back as uint8. Changes in `gpu_pattern_viewer.py`:

- `self.src_ext`, `self.dtype_max` (255/65535), `self.scale_max` (effective full scale:
  255 for uint8; smallest 2^k-1 >= dataset max for uint16, so a 12-bit camera -> 4095),
  `self.data_max` (printed at load, shown in the file label). Set in `load_path`.
- Raw/absolute preview `vmax = scale_max`; absolute-mode B/C midpoint `scale_max/2`,
  clip to `scale_max`; normalized/global-stretch export scales to `dtype_max`
  (`process_pattern`, `_process_chunk_batch`, `_cpu_pipeline_subbatch`).
- `_gpu_nlpar_result` / `_cpu_nlpar_result` temp files and the `_NLPAR*` glob use the
  source extension; `Apply && Save` filters on it, forces it if the user types another
  one, and refuses to overwrite the loaded source file.
- `Load Existing NLPAR` checks grid, pattern size and dtype against the loaded file.
- uint8 behaviour is unchanged (scale_max stays 255). Headless smoke test on synthetic
  6x5 scans (uint16 12-bit-like and uint8): absolute-mode save is byte-identical to the
  input, normalized save fills 0-65535 / 0-255, CPU NLPAR on .up2 keeps the 16-bit scale.

## Where things are

| Path | What |
|---|---|
| `~/EBSD/kikuchi-pattern-processor/` | this repo, cloned from github.com/helpimlost97/kikuchi-pattern-processor |
| `~/EBSD/venvs/kikuchi/` | Python 3.12 venv with everything in `requirements.txt` + `joblib` (CPU only, no CuPy) |
| `run_gui.sh` | launcher: `module load python/3.12`, activate venv, `QT_QPA_PLATFORM=xcb`, run the app |
| `osc_selftest.py` | headless test/benchmark driver (offscreen Qt), see below |
| `/fs/scratch/PAS2325/vegamichalak1/kikuchi_test/` | test data + outputs (deletable): `subset_20rows.up1` (2 GB slice of the Ni RKD scan), `gpv_orig.py` (pre-OSC code), `*_selftest_*.up1`, `selftest_login.log`, `selftest_72.sbatch` (unused) |
| `~/EBSD/EMdata/XtalFolder/Matching EDS Location_Processed.up1` | full scan, 32.4 GB, v3 header, 430 cols x 328 rows, 479x479 px. Load from home; Max decided against staging data on scratch (2026-09-27) |
| `~/EBSD/EMdata/StainlessWork/` | 304L stainless datasets (base metal 5.2 GB + a 9-tile weld stitch); not worked on yet, testing stays on the Ni RKD scan |
| `Kikuchi Pattern Processor.desktop` | XFCE launcher (double-click in Thunar inside the Cardinal Desktop); wraps `run_gui.sh` in a terminal |

Related, same machine: EMSphInx and EMsoftOO are built under `~/EBSD/`
(`~/EBSD/env_ebsd.sh`); see `~/EBSD/README`-style notes in the build scripts.

## Changes made to `gpu_pattern_viewer.py` (UNCOMMITTED — `git diff` shows them)

Goal: same GUI, same workflow, same results as the laptop, but usable on a
CPU-only SLURM node. The CuPy/GPU code paths were left untouched; they still
activate if CuPy is installed on a GPU node.

1. **Thread count from the SLURM allocation.** `n_workers()` = size of
   `os.sched_getaffinity(0)` (override `KPP_THREADS=<n>`). Set before the
   pyebsdindex import so `NUMBA_NUM_THREADS` matches. Window title shows it.
2. **RAM budgets respect the cgroup.** `available_ram()` = min(psutil
   available, cgroup-v2 `memory.max` − `memory.current`). psutil alone
   reports the whole 500 GB node; the OOM killer enforces the job limit.
   Chunk budget cap raised from 3 GB to `1 GB × max(4, threads/2)`; NLPAR
   tile budget cap 4 → 16 GB.
3. **σ² map CPU path rewritten**: vectorized per scan row (one array op per
   neighbour offset over all columns) and rows spread across a thread pool.
   Same math (float32 diffs are exact for uint8 data, float64 accumulation).
4. **Apply && Save CPU path threaded**: the original CPU chain (static bg,
   dynamic bg, stretch/normalize, CLAHE, final scale) moved verbatim into
   `_cpu_pipeline_subbatch()`, run over ~3 sub-batches per thread by
   `_cpu_pipeline_threaded()`. No Qt widget access inside worker threads.
5. Minor: truncated-file check used a 16-byte header for v3 files (it is 42);
   now uses the reader's `filePos`. `ncols` cast to Python int (pyebsdindex
   returns numpy uint64, which overflowed on `ncols - 1`).
6. **Load (IQ map / mean pattern / pixel std) threaded** (2026-09-25 pm): each
   chunk is read once, then split into ~nthr pattern slices reduced by a
   thread pool; merge is exact (float64 sums of integer terms). Verified
   bit-identical `iq_map`, `mean_pattern`, `pixel_std` vs the pre-change code
   on the 20-row subset; 14.4 s -> 4.2 s on the login node with 16 threads.
   Before this the load was the one remaining single-core stage (it was the
   CuPy path on the laptop).
7. **Scratch staging removed** (2026-09-27): Apply && Save briefly wrote to
   `/fs/scratch` and background-copied to the destination. Measured home NFS
   write at 641 MB/s (a 32 GB save is under a minute against minutes of
   processing), so Max had it removed. Saves go straight to the chosen path.

8. **Tier 1: dataset held in RAM** (2026-09-27): `load_path` reads the whole
   file once with `read_patterns_parallel` (module-level; up to 16 threads each
   streaming one contiguous byte range straight into the destination array,
   no per-row copies) when the dataset is <= 60% of `available_ram()`
   (`KPP_IN_MEMORY=1/0` forces it). `self.patterns` is the (npat, ph, pw)
   native-dtype array. Every read site now goes through `_read_rows(fobj, ...)`
   / `_read_pattern`, which serve a VIEW from RAM for the original file and fall
   back to the parallel file reader (full rows) or pyebsdindex (column windows,
   NLPAR-file single patterns). pyebsdindex's own NLPAR batch / opt_lambda still
   read the file themselves. Otherwise the streaming behaviour is unchanged, just
   with the faster reader. The file label shows "(in RAM)" or "(streamed from disk)".
9. **cgroup headroom fix** (same day): `available_ram()` used memory.max minus
   memory.current, but `current` counts page cache, so right after reading 32 GB
   it reported ~5 GB and every threaded pass fell to 1 thread (load stats took
   280 s). `_cgroup_headroom()` now subtracts `memory.stat: file`; the in-RAM
   read also posix_fadvise(DONTNEED)s the file so the cache copy is released.

10. **Workflow gates** (2026-09-27, Max's spec, so students can use the app
    unassisted): `apply_and_save` refuses (info dialog, nothing written) while
    "Enable Preview" NLPAR is checked and no full-dataset NLPAR result exists
    (`self.f_nlpar is None`). `apply_nlpar_batch` refuses unless
    `self._lambda_optimized`, which `optimize_lambda` sets on success and file
    load / aperture change clear (Max asked for the batch gate; the "regardless
    of the preview checkbox" part was my suggestion, accepted). Manual edits of
    the lambda box do not re-arm the gate. The batch still unchecks Preview when
    it finishes (pre-existing behaviour). Chain: Optimize Lambda -> Apply NLPAR to
    Full Dataset -> Apply && Save.

11. **Second-dataset load fixes** (2026-09-27, code review only, NOT run): in
    `load_path` all per-dataset state (old NLPAR file, sigma map, global range,
    lambda gate, auto-detect aperture cache, current row/col, iq_map/iq_mask) is
    now cleared BEFORE the IQ sliders are reset, because `setValue` emits
    valueChanged -> `_update_iq_range` -> `update_display`, which previously ran
    against the old scan's NLPAR file / sigma map / click position with the new
    scan's mean pattern (shape mismatch or IndexError; no sys.excepthook, so PyQt
    aborts the app). `update_display` also bails on a row/col outside the current
    scan. The aperture cache key was (mode, thresh, ph, pw) and so survived a new
    file of the same pattern size (the 304L stitch tiles) with the old pixel_std.
    Also: `_remove_temp_files()` now runs on load as well as on exit, so the
    previous dataset's `_cpu_nlpar_result.up1` is deleted when a new file is opened.
    Not changed: "Enable Preview" NLPAR / NPAR / global-stretch checkboxes keep
    their state across loads, so a new file with preview NLPAR on immediately
    computes its full sigma map as part of the first display.

12. **Apply && Save: no more blank pre-fill; maps stopped early** (2026-09-28,
    found on the DDC `MB6-103` scan, 2131x1671 header, 2,338,772 patterns in the
    file). Two things:
    (a) `pyebsdindex.write_data()` on a path that does not exist yet calls
    `write_header(writeBlank=True)`, which writes one blank pattern per
    `tofile()` for every nCols*nRows pattern: 3.56 M 13 KB writes over NFS,
    measured 28 MB/s on home = ~28 min of zeros before the first real chunk
    (the Ni scan paid ~19 min the same way). Max killed that save at 42 GB;
    the file was all zeros. Now `apply_and_save` (and the GPU NLPAR writer)
    copy the source header verbatim and `truncate()` to the final size (sparse,
    6 ms), then write chunks with `writeHead=False`.
    (b) The save looped over all header rows, fabricating zero patterns for the
    574 missing rows, running them through the pipeline and writing them (16 GB
    here), and it dropped the 1065 patterns of the trailing partial row. It now
    writes exactly `n_actual_pats` patterns (full rows + partial row, header
    keeps the full grid, i.e. the same shape OIM wrote). The partial row is
    taken from the ORIGINAL file even when the NLPAR result is the source,
    because pyebsdindex only processes whole rows (its result has zeros there).
    New helper `_read_patterns(fobj, p0, n)` reads a pattern range across row
    boundaries (RAM view or parallel reader).
    Verified headless on the 20-row subset (scratchpad test, 8 threads): a
    12-row + 200-pattern truncated copy saves 5360 patterns, header byte-equal
    to the source, and the data is byte-equal to the first 5360 patterns of the
    full-file save. Full 20-row save 34.7 s vs ~60-75 s before (the pre-fill).

13. **Optimize Lambda / NLPAR batch read from RAM** (2026-09-28). Max reported
    Optimize Lambda "still slow as shit" on the DDC scan (31 GB, stopped early).
    Nothing truncation-specific: pyebsdindex reads the scan ITSELF through
    `getinfileobj().read_data()` -- tile by tile, one np.fromfile per row, a
    single NFS stream at 130-220 MB/s -- so every Optimize Lambda and every
    NLPAR batch re-read the whole file (3-4 min for 31 GB) although the same
    bytes were already in `self.patterns`. New `RamUPFile(ebsd_pattern.UPFile)`
    keeps the real header/copy_file and serves `read_data` (contiguous and
    [[c,r],[nc,nr]] slab forms, `returnArrayOnly=True`) from the RAM array with
    the same dtype casts as `pat_reader`. `_serve_nlpar_from_ram(nlobj)` binds
    it as `nlobj.getinfileobj` (mirroring its nrows/ncols sync, so the capped
    row count of a map stopped early still applies). Console line shows
    `source=RAM` or `source=file` (streamed datasets keep reading the file).
    Verified on the 20-row subset: lambda and the NLPAR output md5 identical
    for the file path and the RAM path.

14. **NLPAR batch result kept in RAM** (2026-09-28, Max: "apply nlpar should
    just write a second dataset to ram, not to the disk. Then Apply and save
    should be the only thing writing things to the disk"). `RamOutUPFile`
    (subclass of `RamUPFile`) overrides `pat_writer` to store into
    `self.nlpar_patterns` (np.empty_like(self.patterns); the partial last row
    is copied from the original since pyebsdindex only processes whole rows),
    `write_header`/`write_datamaxmin` are no-ops. `_nlpar_result_to_ram(nlobj)`
    allocates it (60 %-of-available rule, `KPP_NLPAR_IN_MEMORY=0` forces the
    file), binds `nlobj.getoutfileobj` to it and makes `nlobj.setoutfile` a
    no-op so no `_cpu_nlpar_result.up1` is created. `self.f_nlpar` becomes a
    `RamUPFile` over that array; `_read_rows`/`_read_patterns` now serve any
    object with a `_ram` attribute, `_read_pattern` goes through its
    `read_data` (slab form handles a partial row). Cleared on load, on
    "Load Existing NLPAR UP1", and before a new batch. Console:
    `source=RAM -> RAM`. Verified on the 20-row subset: RAM result byte-equal to
    the temp-file result, Apply && Save outputs md5-identical, no temp file,
    display read equal to the array. README 4.3 table has a new column for
    the memory this needs (two dataset-sized arrays).

15. **GPU mode usable on OSC; GPU NLPAR result in RAM** (2026-09-28, v1.4.0;
    three sonnet/opus agents did the work, Max's ask was "the small things" for
    GPU desktops; Claude Fable reviewed). CuPy 14.2.0 (`cupy-cuda12x`, log
    `~/EBSD/venv_cupy_install.log`) is in the venv; it brings no CUDA runtime
    wheels, so the `cuda/12.9.1` Lmod module must be loaded (NVRTC for kernel
    compilation, libs on LD_LIBRARY_PATH). `run_gui.sh` loads it if present,
    prints `nvidia-smi -L` or "no GPU in this job: CPU mode", and with
    `KPP_GPU=0` also sets `CUDA_VISIBLE_DEVICES=""`. `setup_all.sh` installs
    CuPy only with `WITH_CUPY=1`. In the app: the HAS_GPU block now requires
    `cp.cuda.runtime.getDeviceCount() > 0` (importing CuPy succeeds on nodes
    without a driver; only device calls raise `cudaErrorNoDevice`), honours
    `KPP_GPU=0`, and prints `[GPU] <name>, <n> GB VRAM` or
    `[GPU] CPU mode (<reason>)`; `_gpu_why` holds the reason. NO CuPy-specific
    plumbing was added: `_alloc_nlpar_ram(tag)` is the allocation half of #14's
    `_nlpar_result_to_ram`, split out so `_apply_nlpar_gpu` can use the same
    host array; the GPU batch writes each chunk with
    `ebsd_pattern.pat_flt2int(result_np, typeout=<file dtype>, method='clip',
    scalevalue=0.98, maxScale=None)`, which is exactly what `write_data`'s 1-D
    branch did into the temp file (checked byte-for-byte, incl. negatives and
    >255), then `self.f_nlpar = RamUPFile(...)` as in the CPU path; falls back
    to `_gpu_nlpar_result.up1` when a second copy does not fit. New
    `_host_padded_chunk_rows(bpp, sr)` = `_host_chunk_rows` charged on the
    padded (rows+2sr)x(ncols+2sr) host block; GPU sigma map uses 8 B/px on it,
    GPU NLPAR min(padded 8 B/px, 12 B/px on the chunk for the result/flt2int
    copies), GPU Apply && Save adds `_host_chunk_rows(33)` (float64 argument +
    result + `*255` temporary + clip output). Two reviewer fixes: (1) the GPU
    NLPAR file path looped to the header's nrows and crashed on a map stopped
    early once a chunk started past the recorded rows (`read_r1 < read_r0`
    -> negative slice); it now loops over `n_actual_rows` like the CPU path,
    with `mask_r1` capped there too. (2) `_sync_bg_controls` read `static_on`
    before the loop that may uncheck static, so a script setting static then
    dynamic ended with both off (the GUI could not reach it; osc_selftest.py
    had been running with neither background on while labelled "bg sub +
    dyn"); it re-reads the state now. Verification: CPU tests
    (`test_truncated_save.py`, `test_nlpar_ram_out.py` in the session
    scratchpad) PASS; the GPU NLPAR code was exercised on the login node with
    numpy standing in for CuPy (`test_gpu_shim.py`): RAM result byte-identical
    to file result, truncated-map loop covers n_actual_rows, partial row equals
    the original. NOT yet run on a real GPU: `osc_gpu_selftest.py` +
    `gpu_selftest.sbatch` (gpu partition, 1 H100, 24 cores, 1 h) are ready and
    validated with `sbatch --test-only`; ask Max before submitting. Parity work
    (border windows, saturation, sigma guards, IQ mask) is listed in README
    4.11 and in Claude's memory `project_gpu_nlpar_parity`.

Measured on the login node (8 threads), full Ni scan 32.4 GB from HOME:
- read into RAM: 34.2 s cold (945 MB/s), 14.9 s partly cached
- load statistics (IQ / mean / pixel std) after the read: 46 s at 8 threads
  (was ~5 min end-to-end before: single-stream NFS read + serialized stats)
- one whole-dataset pass from RAM (recompute background, single-threaded): 25 s
- single pattern read: 0.1 ms; peak RSS 34.2 GB
Subset selftest (in-RAM and forced streaming): Apply && Save output md5
19577903387459cf69b0d00cac10a156, identical to the pre-change reference;
sigma-map median 149.2 unchanged.

GUI note: the OnDemand desktop must be requested with enough memory for the
file plus ~2/3 headroom (Cardinal gives ~4.9 GB per core): the Ni scan needs
>= 12 cores, 16+ is comfortable and makes the load stats faster too.
Otherwise the app falls back to streaming and says so in the console.

## What was verified (login node, 8 threads, 20-row subset)

`osc_selftest.py subset_20rows.up1 --orig gpv_orig.py` ran the sequence
load → σ² map → NLPAR batch → Apply&&Save (bg subtract + dynamic bg +
per-pattern normalize + CLAHE) through the new code, then load → σ² map →
Apply&&Save through the ORIGINAL code, and compared:

- σ² map: max difference **0**
- Apply&&Save output `.up1`: **byte-identical**
- times, new vs original, same 8 threads: σ² map 42 s vs 2856 s (68×);
  Apply&&Save 79 s vs 183 s (2.3×; CLAHE scales worst); NLPAR batch 69 s
  (pyebsdindex/numba, unchanged code, already multi-core)
- GUI launches under X11 (`xcb`) on the Cardinal OS image (tested with Xvfb).

Not yet measured: scaling at 72–96 threads. `selftest_72.sbatch` on scratch
would do it (`sbatch` it from that directory); Max asked to hold off.

## How to run the GUI

OnDemand → Interactive Apps → **Cardinal Desktop** (choose cores, e.g. 48–96,
and hours; it is a SLURM job). In a terminal inside that desktop:

```bash
bash ~/EBSD/kikuchi-pattern-processor/run_gui.sh
```

Console prints `[CPU] N worker threads, X GB RAM available` on startup.
Inputs live in home (`~/EBSD/EMdata/`); the app writes its NLPAR temp files
next to the input. Scratch is only used for the test subset.

## Open items / decisions Max has NOT made yet

- Commits: v1.0.0 (OSC port), v1.1.0 (save fixes), v1.2.0 (NLPAR reads from
  RAM), v1.3.0 (NLPAR result in RAM) are committed and tagged locally; pushing
  needs Max's GitHub credentials (`git push && git push --tags`).
- Whether to run the 72-96-core timing job.
- Still open: NPAR not applied on Apply && Save; preview checkboxes survive a
  new load; headless CLI; scratch re-stream tier 2 (parked, "no scratch").
- GPU: not installed. If wanted later: a Cardinal GPU desktop +
  `pip install cupy-cuda12x` into the venv; code needs no change.
- Headless CLI for Apply&&Save (mentioned in OSC_SETUP.md) — not written.

## Working-style notes for the next session (from this one)

- Max runs things from his phone a lot; keep replies short, report results
  plainly, don't expand scope without asking. He said "slow your roll" after
  the comparison — stop at natural checkpoints.
- He does want builds/setup RUN, not handed to him as scripts; but SLURM
  jobs on his allocation: ask first.
- Never touch `/project/` or the PhaseField folder from this work.
