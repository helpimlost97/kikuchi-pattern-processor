"""Headless self-test / benchmark for gpu_pattern_viewer.py (no display needed).

    python selftest.py <scan.up1> [--orig gpv_orig.py] [--no-nlpar]

Runs, with timings:  load -> sigma-squared map (CPU) -> NLPAR batch (pyebsdindex,
OpenCL or numba) -> Apply && Save (static bg subtract + dynamic bg + per-pattern normalize +
CLAHE) on the ORIGINAL patterns, writing <scan>_selftest_out.up1 next to the input.

With --orig <file>, the same load -> sigma -> Apply&&Save sequence is also run
through that copy of the program (e.g. an older version from git) and the sigma
maps and output files are compared, to prove the threaded CPU paths give the same
answer. Thread count comes from the Slurm cpuset (or os.cpu_count), or KPP_THREADS=<n>.
"""
import os, sys, time, importlib.util, hashlib
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def run_sequence(mod, scan, out_path, do_nlpar):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    w = mod.MainWindow()
    t = {}
    t0 = time.time(); w.load_path(scan); t['load'] = time.time() - t0
    print(f"  load: {w.ncols}x{w.nrows} patterns {int(w.f_orig.patternW)}x{int(w.f_orig.patternH)}  {t['load']:.1f}s")
    w.aperture_mode.setCurrentIndex(0)                       # aperture: None
    t0 = time.time(); w._ensure_sigma_map(); t['sigma'] = time.time() - t0
    sig = w.sigma_sq_map.copy()
    print(f"  sigma map: median {float(__import__('numpy').median(sig)):.4g}  {t['sigma']:.1f}s")
    if do_nlpar:
        t0 = time.time(); w._apply_nlpar_cpu(3, 0.9); t['nlpar'] = time.time() - t0
        print(f"  NLPAR batch (SR=3, lam=0.9): {t['nlpar']:.1f}s -> {w.f_nlpar.filepath if w.f_nlpar else 'NO OUTPUT'}")
    # Apply && Save from the ORIGINAL patterns (same source in both versions)
    w.f_nlpar = None
    w.static_bg_chk.setChecked(True); w.bg_mode.setCurrentText("Subtract")
    w.dyn_bg_chk.setChecked(True); w.normalize_chk.setChecked(True); w.global_chk.setChecked(False)
    w.clahe_chk.setChecked(True)
    mod.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_path, ''))
    mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    if os.path.exists(out_path): os.remove(out_path)
    t0 = time.time(); w.apply_and_save(); t['save'] = time.time() - t0
    print(f"  Apply&&Save (bg sub + dyn + normalize + CLAHE): {t['save']:.1f}s -> {out_path} "
          f"({os.path.getsize(out_path)/1e9:.2f} GB)")
    w.close()
    return sig, t

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''): h.update(b)
    return h.hexdigest()

if __name__ == '__main__':
    args = sys.argv[1:]
    scan = args[0]
    orig = args[args.index('--orig') + 1] if '--orig' in args else None
    do_nlpar = '--no-nlpar' not in args
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.splitext(scan)[0]
    print(f"== NEW code ({here}/gpu_pattern_viewer.py), KPP_THREADS={os.environ.get('KPP_THREADS','<affinity>')}")
    new = load_module(os.path.join(here, 'gpu_pattern_viewer.py'), 'gpv_new')
    sig_new, t_new = run_sequence(new, scan, base + '_selftest_out.up1', do_nlpar)
    if orig:
        import numpy as np
        print(f"== ORIGINAL code ({orig})")
        old = load_module(orig, 'gpv_orig')
        sig_old, t_old = run_sequence(old, scan, base + '_selftest_orig.up1', False)
        d = np.abs(sig_new.astype(np.float64) - sig_old.astype(np.float64))
        rel = d.max() / max(np.abs(sig_old).max(), 1e-30)
        print(f"== sigma map: max abs diff {d.max():.3g}, max rel diff {rel:.3g}")
        same = md5(base + '_selftest_out.up1') == md5(base + '_selftest_orig.up1')
        print(f"== Apply&&Save output files byte-identical: {same}")
        print(f"== speedup: sigma {t_old['sigma']/max(t_new['sigma'],1e-9):.1f}x, save {t_old['save']/max(t_new['save'],1e-9):.1f}x")
    print("== selftest done")
