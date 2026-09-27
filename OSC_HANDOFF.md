# OSC handoff — state of this folder as of 2026-09-25

Read this first if you are a new Claude session (or Max) picking up the
Kikuchi Pattern Processor on the Ohio Supercomputer Center. It records what
was done, what was verified, and what is deliberately NOT done yet.
`OSC_SETUP.md` (from the original repo) is still the general Linux/OSC guide;
this file is the delta on top of it.

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

- Whether to commit the changes (nothing committed; author attribution TBD).
- Whether to run the 72-core timing job.
- Whether to update `OSC_SETUP.md`/`README.md` to reflect the above.
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
