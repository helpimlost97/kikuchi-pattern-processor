"""Headless GPU-vs-CPU self-test for gpu_pattern_viewer.py (needs a real GPU + CuPy;
NOT a fallback comparison of CPU code against itself).

    python osc_gpu_selftest.py <scan.up1> [--sr 3] [--lam 0.9]

This is a companion to osc_selftest.py (same conventions: QT_QPA_PLATFORM=offscreen,
load_module, patching QFileDialog.getSaveFileName / QMessageBox.*, w.load_path,
w.aperture_mode.setCurrentIndex(0), calling the CPU/GPU worker methods directly).
Where osc_selftest.py proves the CPU paths are internally consistent across code
versions, this script proves the GPU (CuPy) implementation and the CPU
(pyebsdindex/numba, or the plain-numpy paths) implementation agree with each other
on the SAME machine, for the SAME input.

Requires HAS_GPU True (a real CUDA device visible to CuPy). On a CPU-only node
(e.g. the OSC login node) gpu_pattern_viewer.py itself detects that HAS_GPU is
False and this script exits immediately with exit code 2 and a clear message --
it does not silently degrade to a CPU-only smoke test. Submit gpu_selftest.sbatch
on a GPU node to actually run it.

Sequence, all timed, on ONE MainWindow / ONE loaded dataset (aperture = "None",
matching osc_selftest.py's convention):

  1. load (in RAM)
  2. sigma-squared map:
       GPU:  _ensure_sigma_map() with mod.HAS_GPU forced True -> dispatches to
             _compute_sigma_map_gpu() internally and caches the result in
             w.sigma_sq_map.
       CPU:  _compute_sigma_map_cpu(aperture_2d, N_p) called DIRECTLY (not via
             _ensure_sigma_map() again -- that would see the cache key from the
             GPU call still matches and just return the cached GPU array without
             recomputing anything). aperture_2d/N_p are resolved the same way
             _ensure_sigma_map() resolves them: _resolve_aperture_mask(ph, pw).
     -> compare: max abs diff, max rel diff, median.
  3. NLPAR batch:
       GPU:  _apply_nlpar_gpu(sr, lam) -- the app's own CuPy shift-and-compare
             implementation. Always writes _gpu_nlpar_result.up1 next to the input
             (tracked in w._temp_files, so w.close() removes it).
       CPU:  _apply_nlpar_cpu(sr, lam) -- pyebsdindex/numba. Result lands in
             w.nlpar_patterns (RAM) when it fits under 60% of available RAM,
             else in _cpu_nlpar_result.up1 (also tracked/cleaned via
             w._temp_files).
     Both are called directly, bypassing apply_nlpar_batch()'s GUI dispatch and
     its "_lambda_optimized" gate (there is no lambda optimization step here;
     --lam is used as given, matching osc_selftest.py's w._apply_nlpar_cpu(3, 0.9)
     convention).
     -> compare the uint8 results (see _nlpar_result_array / compare_patterns
     below): max abs diff, mean abs diff, fraction of differing pixels, each
     computed twice -- over everything, and restricted to the "interior" (the
     outer `sr` scan rows/columns excluded, i.e. away from the edge of the map)
     -- because CPU pyebsdindex NLPAR and the GPU CuPy NLPAR are known to differ
     at scan borders (edge patterns have fewer/no real neighbours; the two
     implementations pad/mask that differently) and on saturated pixels
     (saturation_protect handling differs), so border differences are expected
     and must not be confused with a real GPU/CPU mismatch in the interior.
  4. Apply && Save, from the ORIGINAL patterns (w.f_nlpar = None first), with
     static background subtract, dynamic background, per-pattern normalize on,
     global stretch off, CLAHE OFF -- run once with mod.HAS_GPU forced True and
     once forced False. Both apply_and_save() and the _process_chunk_batch() it
     calls branch on the bare module-level name `HAS_GPU` read at call time
     (`if HAS_GPU:`), not a value cached at import or __init__ time, so flipping
     mod.HAS_GPU between the two calls genuinely switches which code path runs
     (checked by reading the source directly; see the caveat below about the
     window title, which DOES cache it, but only for display).
     NOTE (checkbox interlock, observed empirically, not a bug in this script):
     gpu_pattern_viewer.py's _sync_bg_controls() treats static and dynamic
     background as mutually exclusive and, as currently written, checking
     dyn_bg_chk right after static_bg_chk ends up unchecking BOTH (a stale
     local variable inside _sync_bg_controls). This script sets the checkboxes
     in the order requested (static, then dynamic, then normalize, then global
     off, then CLAHE off -- the same order osc_selftest.py uses) and then PRINTS
     the actual resulting checkbox states it read back, rather than assuming
     they match what was requested. Whatever the actual resulting state is, it
     is identical for the GPU run and the CPU run, so the GPU-vs-CPU comparison
     below is unaffected either way.
     -> compare the two output files' pattern bytes, same stats as step 3.

Never crashes: every stage is wrapped so a failure is reported (counted as a
difference) instead of raising and killing the run. PASS requires every INTERIOR
max-abs-diff (NLPAR and Apply&&Save) to be <= 2 gray levels (0-255 uint8 scale);
this threshold is restated on the RESULT line. The sigma-squared map comparison
is informational only (different units, not part of the PASS/FAIL gate).

Output files this script creates (<scan>_gpuselftest_gpu.up1 / _cpu.up1, plus
whatever _gpu_nlpar_result.up1 / _cpu_nlpar_result.up1 the app itself writes) are
removed before exit, whether the run passes or fails.
"""
import sys as _sys
_sys.exit("osc_gpu_selftest.py predates v1.6: it compared the removed CuPy NLPAR/sigma code against the CPU "
          "code. It is rewritten in Phase 4 of PLAN_NLPAR_PYEBSDINDEX.md to compare pyebsdindex OpenCL vs numba.")

import argparse
import os
import sys
import time
import importlib.util

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def diff_stats(a, b):
    """max abs diff, mean abs diff, fraction of differing entries -- for two
    same-shape arrays already cast to a common dtype wide enough to subtract."""
    if a.size == 0:
        return dict(n=0, max_abs=0.0, mean_abs=0.0, frac_differing=0.0)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(n=int(d.size), max_abs=float(d.max()), mean_abs=float(d.mean()),
                frac_differing=float(np.count_nonzero(d) / d.size))


def compare_patterns(a, b, ncols, sr, what):
    """Compare two (n_pat, ph, pw) pattern arrays in scan order: overall stats,
    plus stats restricted to the interior of the scan grid (outer `sr` rows and
    columns of full scan rows excluded). Returns (stats_all, stats_interior or
    None if the scan is too small for an interior at this sr)."""
    n = min(a.shape[0], b.shape[0])
    if n != a.shape[0] or n != b.shape[0]:
        print(f"  [{what}] WARNING: pattern counts differ (GPU {a.shape[0]}, CPU {b.shape[0]}); "
              f"comparing the common {n}")
    a, b = a[:n], b[:n]
    stats_all = diff_stats(a, b)
    n_full_rows = n // ncols
    stats_interior = None
    if n_full_rows > 2 * sr and ncols > 2 * sr:
        ga = a[:n_full_rows * ncols].reshape(n_full_rows, ncols, *a.shape[1:])
        gb = b[:n_full_rows * ncols].reshape(n_full_rows, ncols, *b.shape[1:])
        ga = ga[sr:n_full_rows - sr, sr:ncols - sr]
        gb = gb[sr:n_full_rows - sr, sr:ncols - sr]
        stats_interior = diff_stats(ga.reshape(-1, *a.shape[1:]), gb.reshape(-1, *b.shape[1:]))
    else:
        print(f"  [{what}] scan too small for an sr={sr} interior ({n_full_rows} full rows x "
              f"{ncols} cols) -- interior stats unavailable")
    return stats_all, stats_interior


def fmt_stats(s):
    if s is None:
        return "n/a"
    return (f"max_abs={s['max_abs']:.3g} mean_abs={s['mean_abs']:.3g} "
            f"frac_differing={s['frac_differing']:.3g} (n={s['n']})")


def nlpar_result_array(w, np_):
    """Copy of the current NLPAR batch result as an (n_pat, ph, pw) array in the
    original file's native dtype. Prefers the in-RAM result (w.nlpar_patterns,
    set by the CPU path when it fits under 60% of available RAM); otherwise reads
    the output file at w.f_nlpar.filepath starting after the ORIGINAL file's
    header length (w.f_orig.filePos) -- both the GPU and the CPU-to-disk paths
    copy that header verbatim onto their output, so the offset is the same one
    either way. Copies (not views) so a later call can safely overwrite state."""
    if w.nlpar_patterns is not None:
        return np_.array(w.nlpar_patterns, copy=True)
    if w.f_nlpar is None:
        return None
    ph, pw = int(w.f_orig.patternH), int(w.f_orig.patternW)
    n = int(getattr(w, 'n_actual_pats', w.nrows * w.ncols))
    dtype = w.f_orig.filedatatype
    offset = int(w.f_orig.filePos)
    arr = np_.fromfile(w.f_nlpar.filepath, dtype=dtype, count=n * ph * pw, offset=offset)
    if arr.size != n * ph * pw:
        raise RuntimeError(f"short read of {w.f_nlpar.filepath}: got {arr.size} elements, "
                            f"expected {n * ph * pw}")
    return arr.reshape(n, ph, pw)


def file_pattern_array(path, header_len, n, ph, pw, dtype, np_):
    arr = np_.fromfile(path, dtype=dtype, count=n * ph * pw, offset=header_len)
    if arr.size != n * ph * pw:
        raise RuntimeError(f"short read of {path}: got {arr.size} elements, expected {n * ph * pw}")
    return arr.reshape(n, ph, pw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scan')
    ap.add_argument('--sr', type=int, default=3, help="NLPAR search radius (default 3)")
    ap.add_argument('--lam', type=float, default=0.9, help="NLPAR lambda (default 0.9)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(here, 'gpu_pattern_viewer.py')
    print(f"== loading {mod_path}")
    mod = load_module(mod_path, 'gpv_gpu_selftest')

    if not mod.HAS_GPU:
        why = getattr(mod, '_gpu_why', 'no CUDA device visible to CuPy')
        print(f"== NO GPU: HAS_GPU is False ({why}).")
        print("== This script only exercises the GPU-vs-CPU comparison; it does not run on a "
              "CPU-only node. Submit gpu_selftest.sbatch on a GPU node instead.")
        sys.exit(2)

    default_has_gpu = mod.HAS_GPU  # restore on exit, out of an abundance of caution
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    # Silence dialogs. getSaveFileName's return path is reassigned per-call below.
    mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    mod.QMessageBox.critical = staticmethod(lambda *a, **k: None)
    mod.QMessageBox.warning = staticmethod(lambda *a, **k: None)

    scan_abspath = os.path.abspath(args.scan)
    base = os.path.splitext(scan_abspath)[0]
    scan_dir = os.path.dirname(scan_abspath)
    gpu_save_path = base + '_gpuselftest_gpu.up1'
    cpu_save_path = base + '_gpuselftest_cpu.up1'
    # Known output paths, computed unconditionally (not just on success) so a call that
    # raises partway through (file already created/truncated on disk before it fails)
    # still gets cleaned up. The two NLPAR temp files are normally cleaned by w.close()
    # via self._temp_files, but only the file paths that a call APPENDED there before
    # raising are covered -- a mid-loop failure leaves the file on disk but unregistered
    # (_apply_nlpar_gpu/_apply_nlpar_cpu only append to _temp_files after finishing), so
    # they are listed here too as a safety net; removing an already-removed file is a
    # silent no-op below.
    known_output_paths = [
        gpu_save_path, cpu_save_path,
        os.path.join(scan_dir, "_gpu_nlpar_result.up1"),
        os.path.join(scan_dir, "_cpu_nlpar_result.up1"),
    ]

    timings = {}
    overall_ok = True
    reasons = []

    w = mod.MainWindow()
    try:
        t0 = time.time()
        w.load_path(args.scan)
        timings['load'] = time.time() - t0
        print(f"  load: {w.ncols}x{w.nrows} patterns "
              f"{int(w.f_orig.patternW)}x{int(w.f_orig.patternH)}, "
              f"{getattr(w, 'n_actual_pats', w.nrows * w.ncols)} actual  "
              f"{timings['load']:.1f}s")
        w.aperture_mode.setCurrentIndex(0)  # aperture: None, matches osc_selftest.py

        ncols = int(w.ncols)
        ph, pw = int(w.f_orig.patternH), int(w.f_orig.patternW)

        # ---------------- σ² map: GPU vs CPU ----------------
        print("== sigma map")
        mod.HAS_GPU = True
        t0 = time.time()
        try:
            w._ensure_sigma_map()
            sig_gpu = w.sigma_sq_map.copy()
            timings['sigma_gpu'] = time.time() - t0
            print(f"  GPU: median {float(np.median(sig_gpu)):.4g}  {timings['sigma_gpu']:.1f}s")
        except Exception as e:
            import traceback; traceback.print_exc()
            sig_gpu = None
            timings['sigma_gpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"sigma GPU raised: {e}")

        aperture_2d, aperture_count, ap_label = w._resolve_aperture_mask(ph, pw)
        N_p = float(aperture_count)
        t0 = time.time()
        try:
            sig_cpu = w._compute_sigma_map_cpu(aperture_2d, N_p)
            timings['sigma_cpu'] = time.time() - t0
            print(f"  CPU: median {float(np.median(sig_cpu)):.4g}  {timings['sigma_cpu']:.1f}s "
                  f"(aperture={ap_label}, N_p={int(N_p)})")
        except Exception as e:
            import traceback; traceback.print_exc()
            sig_cpu = None
            timings['sigma_cpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"sigma CPU raised: {e}")

        if sig_gpu is not None and sig_cpu is not None:
            d = np.abs(sig_gpu.astype(np.float64) - sig_cpu.astype(np.float64))
            rel = d.max() / max(np.abs(sig_cpu).max(), 1e-30)
            print(f"  == sigma map: max abs diff {d.max():.3g}, max rel diff {rel:.3g}, "
                  f"median(GPU) {float(np.median(sig_gpu)):.4g}, median(CPU) {float(np.median(sig_cpu)):.4g}")
            print("  (sigma map is informational only -- not part of the PASS/FAIL gate; "
                  "different units than the uint8 pattern comparisons below)")
        else:
            print("  == sigma map: comparison skipped (one or both sides failed)")

        # ---------------- NLPAR batch: GPU vs CPU ----------------
        print(f"== NLPAR batch (SR={args.sr}, lam={args.lam})")
        w.f_nlpar = None; w.nlpar_patterns = None
        mod.HAS_GPU = True
        t0 = time.time()
        nlpar_gpu = None
        try:
            w._apply_nlpar_gpu(args.sr, args.lam)
            nlpar_gpu = nlpar_result_array(w, np)
            timings['nlpar_gpu'] = time.time() - t0
            print(f"  GPU: {timings['nlpar_gpu']:.1f}s -> "
                  f"{w.f_nlpar.filepath if w.f_nlpar is not None else 'NO OUTPUT'}")
        except Exception as e:
            import traceback; traceback.print_exc()
            timings['nlpar_gpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"NLPAR GPU raised: {e}")

        w.f_nlpar = None; w.nlpar_patterns = None
        t0 = time.time()
        nlpar_cpu = None
        try:
            w._apply_nlpar_cpu(args.sr, args.lam)
            nlpar_cpu = nlpar_result_array(w, np)
            timings['nlpar_cpu'] = time.time() - t0
            dest = 'RAM' if w.nlpar_patterns is not None else (
                w.f_nlpar.filepath if w.f_nlpar is not None else 'NO OUTPUT')
            print(f"  CPU: {timings['nlpar_cpu']:.1f}s -> {dest}")
        except Exception as e:
            import traceback; traceback.print_exc()
            timings['nlpar_cpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"NLPAR CPU raised: {e}")

        nlpar_interior = None
        if nlpar_gpu is not None and nlpar_cpu is not None:
            stats_all, nlpar_interior = compare_patterns(nlpar_gpu, nlpar_cpu, ncols, args.sr, "NLPAR")
            print(f"  == NLPAR all:      {fmt_stats(stats_all)}")
            print(f"  == NLPAR interior: {fmt_stats(nlpar_interior)}  (outer sr={args.sr} rows/cols excluded)")
        else:
            print("  == NLPAR: comparison skipped (one or both sides failed)")
            overall_ok = False; reasons.append("NLPAR comparison unavailable")
        del nlpar_gpu, nlpar_cpu

        # ---------------- Apply && Save: GPU vs CPU ----------------
        print("== Apply && Save (static bg subtract + dynamic bg + normalize, CLAHE OFF)")
        w.f_nlpar = None; w.nlpar_patterns = None
        w.static_bg_chk.setChecked(True); w.bg_mode.setCurrentText("Subtract")
        w.dyn_bg_chk.setChecked(True)
        w.normalize_chk.setChecked(True); w.global_chk.setChecked(False)
        w.clahe_chk.setChecked(False)
        print(f"  actual checkbox state used: static={w.static_bg_chk.isChecked()} "
              f"dyn={w.dyn_bg_chk.isChecked()} normalize={w.normalize_chk.isChecked()} "
              f"global={w.global_chk.isChecked()} clahe={w.clahe_chk.isChecked()}  "
              "(gpu_pattern_viewer.py's static/dynamic interlock may not match what was "
              "requested -- see the module docstring; both runs below see the same state)")

        def mock_save(path):
            mod.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ''))

        n_out = int(getattr(w, 'n_actual_pats', w.nrows * w.ncols))
        dtype = w.f_orig.filedatatype
        hdr_len = int(w.f_orig.filePos)

        save_gpu = None
        mod.HAS_GPU = True
        w.f_nlpar = None
        mock_save(gpu_save_path)
        if os.path.exists(gpu_save_path): os.remove(gpu_save_path)
        t0 = time.time()
        try:
            w.apply_and_save()
            timings['save_gpu'] = time.time() - t0
            print(f"  GPU: {timings['save_gpu']:.1f}s -> {gpu_save_path} "
                  f"({os.path.getsize(gpu_save_path)/1e9:.2f} GB)")
            save_gpu = file_pattern_array(gpu_save_path, hdr_len, n_out, ph, pw, dtype, np)
        except Exception as e:
            import traceback; traceback.print_exc()
            timings['save_gpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"Apply&&Save GPU raised: {e}")

        mod.HAS_GPU = False
        w.f_nlpar = None
        mock_save(cpu_save_path)
        if os.path.exists(cpu_save_path): os.remove(cpu_save_path)
        t0 = time.time()
        save_cpu = None
        try:
            w.apply_and_save()
            timings['save_cpu'] = time.time() - t0
            print(f"  CPU: {timings['save_cpu']:.1f}s -> {cpu_save_path} "
                  f"({os.path.getsize(cpu_save_path)/1e9:.2f} GB)")
            save_cpu = file_pattern_array(cpu_save_path, hdr_len, n_out, ph, pw, dtype, np)
        except Exception as e:
            import traceback; traceback.print_exc()
            timings['save_cpu'] = time.time() - t0
            overall_ok = False; reasons.append(f"Apply&&Save CPU raised: {e}")

        save_interior = None
        if save_gpu is not None and save_cpu is not None:
            stats_all, save_interior = compare_patterns(save_gpu, save_cpu, ncols, args.sr, "Apply&&Save")
            print(f"  == Apply&&Save all:      {fmt_stats(stats_all)}")
            print(f"  == Apply&&Save interior: {fmt_stats(save_interior)}  (outer sr={args.sr} rows/cols excluded)")
        else:
            print("  == Apply&&Save: comparison skipped (one or both sides failed)")
            overall_ok = False; reasons.append("Apply&&Save comparison unavailable")
        del save_gpu, save_cpu

        # ---------------- timing table ----------------
        print("\n== timings (s)")
        print(f"  {'stage':<12} {'GPU':>10} {'CPU':>10}")
        print(f"  {'load':<12} {timings.get('load', float('nan')):>10.1f} {'':>10}")
        print(f"  {'sigma map':<12} {timings.get('sigma_gpu', float('nan')):>10.1f} "
              f"{timings.get('sigma_cpu', float('nan')):>10.1f}")
        print(f"  {'NLPAR':<12} {timings.get('nlpar_gpu', float('nan')):>10.1f} "
              f"{timings.get('nlpar_cpu', float('nan')):>10.1f}")
        print(f"  {'Apply&&Save':<12} {timings.get('save_gpu', float('nan')):>10.1f} "
              f"{timings.get('save_cpu', float('nan')):>10.1f}")

        # ---------------- verdict ----------------
        THRESH = 2.0  # gray levels, 0-255 uint8 scale
        pass_ok = overall_ok
        for label, interior in (("NLPAR", nlpar_interior), ("Apply&&Save", save_interior)):
            if interior is None:
                pass_ok = False
                reasons.append(f"{label}: no interior stats available")
            elif interior['max_abs'] > THRESH:
                pass_ok = False
                reasons.append(f"{label}: interior max_abs={interior['max_abs']:.3g} > {THRESH}")

        print(f"\n== PASS threshold: interior max-abs-diff <= {THRESH} gray levels (0-255 uint8 scale), "
              "for both NLPAR and Apply&&Save")
        if pass_ok:
            print("== RESULT: PASS")
        else:
            print("== RESULT: DIFFERENCES")
            for r in reasons:
                print(f"     - {r}")
    except Exception as e:
        # Belt-and-suspenders: every individual stage above already catches its own
        # exceptions, but this outer guard covers anything unguarded (e.g. load_path
        # itself raising on a bad/missing file) so the script always reaches the
        # cleanup below and prints a RESULT line instead of an unhandled traceback.
        import traceback; traceback.print_exc()
        print(f"\n== UNHANDLED EXCEPTION: {e}")
        print("== RESULT: DIFFERENCES")
    finally:
        mod.HAS_GPU = default_has_gpu
        try:
            w.close()  # triggers closeEvent -> _remove_temp_files() for the app's own NLPAR temp files
        except Exception as e:
            print(f"[CLEANUP] w.close() raised: {e}")
        # Safety net beyond w.close(): also sweep the known output paths directly, in case
        # a call raised partway through (file created on disk but never registered in
        # w._temp_files -- see the comment where known_output_paths is built).
        for f in known_output_paths:
            try:
                if f and os.path.isfile(f):
                    os.remove(f)
                    print(f"[CLEANUP] removed {f}")
            except Exception as e:
                print(f"[CLEANUP] could not remove {f}: {e}")

    print("== selftest done")


if __name__ == '__main__':
    main()
