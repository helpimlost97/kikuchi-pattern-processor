__version__ = "1.2.0"   # shown in the window title; bump with each release (git tag v<version>)
import sys
import os
import glob

# ---------------------------------------------------------------------------
# OSC / Linux CPU-node support [MVM+Claude 2026-09-25]
# On a SLURM node the process only owns the cores/RAM it asked for, not the
# whole machine (Cardinal: 104 cores, ~500 GB). os.sched_getaffinity sees the
# cpuset SLURM pinned us to; the cgroup memory limit is what the OOM killer
# enforces, while psutil reports node-wide numbers. Everything CPU-parallel
# below sizes itself from these two helpers.
# ---------------------------------------------------------------------------
def n_workers():
    """Worker threads for the CPU paths (override with KPP_THREADS=<n>)."""
    env = os.environ.get('KPP_THREADS')
    if env:
        try: return max(1, int(env))
        except ValueError: pass
    try: return max(1, len(os.sched_getaffinity(0)))
    except Exception: return max(1, os.cpu_count() or 1)

# numba reads NUMBA_NUM_THREADS at import time; pyebsdindex (NLPAR) imports numba.
os.environ.setdefault('NUMBA_NUM_THREADS', str(n_workers()))

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFormLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QFileDialog, QMessageBox,
    QStatusBar, QSplitter, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyebsdindex import ebsd_pattern, nlpar
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
import time
from concurrent.futures import ThreadPoolExecutor

# Console prints use σ/λ/—; when stdout is a pipe/file on Windows it defaults to
# cp1252 and a bare print would raise UnicodeEncodeError mid-run. Degrade to '?'.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

# GPU acceleration via CuPy (falls back to CPU if unavailable)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    HAS_GPU = True
    _gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
    if isinstance(_gpu_name, bytes): _gpu_name = _gpu_name.decode()
    print(f"[GPU] CuPy detected: {_gpu_name}")
except Exception:
    HAS_GPU = False
    print("[GPU] CuPy not available — running CPU-only mode")

# psutil lets chunk sizes track ACTUAL free host RAM instead of a hardcoded row count.
# Optional: without it we fall back to a conservative fixed budget.
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

def _cgroup_headroom():
    """Bytes this process may still allocate under its cgroup (v2) memory limits, or None
    if unlimited. Walks from our own cgroup up to the root and takes the tightest level.

    `memory.current` counts the page cache too, and a 32 GB pattern file just read through
    the cache shows up as 32 GB "used" although the kernel reclaims cache on demand (it is
    not what the OOM killer counts against us). Charging it made available_ram() collapse
    to ~5 GB right after loading the Ni scan on the login node (64 GB user cap), which
    starved every threaded pass down to 1 thread. So reclaimable file cache
    (`memory.stat: file`) is subtracted from `current` at each level."""
    try:
        with open('/proc/self/cgroup') as f:
            rel = [l.split(':', 2)[2].strip() for l in f if l.startswith('0::')][0]
    except Exception:
        return None
    parts = [p for p in rel.split('/') if p]
    best = None
    for i in range(len(parts), -1, -1):
        d = '/sys/fs/cgroup/' + '/'.join(parts[:i])
        try:
            with open(d + '/memory.max') as f: lim = f.read().strip()
            if lim == 'max': continue
            lim = int(lim)
            with open(d + '/memory.current') as f: cur = int(f.read().strip())
            cache = 0
            with open(d + '/memory.stat') as f:
                for line in f:
                    if line.startswith('file '): cache = int(line.split()[1]); break
        except Exception:
            continue
        head = max(lim - max(cur - cache, 0), 0)
        best = head if best is None else min(best, head)
    return best

def available_ram():
    """Bytes we can actually allocate: min(what psutil sees, what the SLURM cgroup allows)."""
    avail = int(psutil.virtual_memory().available) if HAS_PSUTIL else int(8e9)
    head = _cgroup_headroom()
    if head is not None:
        avail = min(avail, head)
    return avail

print(f"[CPU] {n_workers()} worker threads, {available_ram()/1e9:.1f} GB RAM available to this process")



def read_patterns_parallel(path, offset, n_pat, ph, pw, dtype, out=None, nthr=None,
                           progress=None, min_bytes=32 << 20, drop_cache=False):
    """Read n_pat consecutive patterns starting at byte `offset` of `path` into an
    (n_pat, ph, pw) array, several threads each streaming one contiguous byte range.

    Why not pyebsdindex's slab reader: it zero-fills the block, then reads and copies
    it row by row through np.fromfile in ONE stream. Measured on Cardinal (Ni RKD scan
    in home, 2026-09-27): a single stream tops out at 130-220 MB/s on home NFS while
    8 streams reach ~830 MB/s; the row-by-row copy alone costs ~20 s per 32 GB even
    from the page cache. This reads straight into the destination buffer, no copies.
    `progress(done_bytes, total_bytes)` is called from the CALLING thread every 0.2 s
    (safe to touch Qt from it). drop_cache=True tells the kernel it may discard the
    page cache of the range just read (posix_fadvise DONTNEED): used when the bytes now
    live in our own array, so the cache copy is dead weight in the cgroup accounting."""
    itemsize = np.dtype(dtype).itemsize
    total = int(n_pat) * int(ph) * int(pw) * itemsize
    if out is None:
        out = np.empty((int(n_pat), int(ph), int(pw)), dtype=dtype)
    buf = memoryview(out).cast('B')                      # flat byte view of the output
    nthr = max(1, min(int(nthr or n_workers()), 16, max(1, total // int(min_bytes))))
    bounds = np.linspace(0, total, nthr + 1, dtype=np.int64)
    done = [0] * nthr

    def work(i):
        a, b = int(bounds[i]), int(bounds[i + 1])
        with open(path, 'rb', buffering=0) as f:
            f.seek(int(offset) + a); pos = a
            while pos < b:
                k = f.readinto(buf[pos:b])               # a single read may come back short
                if not k:
                    raise IOError(f"{path}: short read at byte {int(offset) + pos} (file truncated?)")
                pos += k; done[i] = pos - a
            if drop_cache:
                try: os.posix_fadvise(f.fileno(), int(offset) + a, b - a, os.POSIX_FADV_DONTNEED)
                except Exception: pass

    with ThreadPoolExecutor(max_workers=nthr) as pool:
        futs = [pool.submit(work, i) for i in range(nthr)]
        if progress is not None:
            while not all(f.done() for f in futs):
                progress(sum(done), total); time.sleep(0.2)
        for f in futs:
            f.result()
    if progress is not None:
        progress(total, total)
    return out


class ScanMapCanvas(FigureCanvas):
    pattern_clicked = pyqtSignal(int, int)
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('IQ Map - Ctrl+Scroll to zoom'); self.ax.axis('off')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.marker = None; self.nrows = self.ncols = 0
        self._orig_xlim = self._orig_ylim = None
        self._mask = None; self._img_handle = None; self._mask_handle = None
        self.mpl_connect('button_press_event', self._on_click)
        self.mpl_connect('scroll_event', self._on_scroll)
    def _on_click(self, event):
        if event.inaxes != self.ax: return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= col < self.ncols and 0 <= row < self.nrows:
            if self._mask is not None and not self._mask[row, col]: return
            self._place_marker(row, col); self.pattern_clicked.emit(row, col)
    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        if not (QApplication.keyboardModifiers() & Qt.ControlModifier): return
        factor = 0.7 if event.button == 'up' else 1.4
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        lx = (xl[1] - xl[0]) * factor; ly = (yl[1] - yl[0]) * factor
        rx = (cx - xl[0]) / (xl[1] - xl[0]); ry = (cy - yl[0]) / (yl[1] - yl[0])
        self.ax.set_xlim(cx - rx*lx, cx + (1-rx)*lx)
        self.ax.set_ylim(cy - ry*ly, cy + (1-ry)*ly); self.draw_idle()
    def _place_marker(self, row, col):
        if self.marker is not None: self.marker.remove()
        self.marker, = self.ax.plot(col, row, 'r+', markersize=15, markeredgewidth=2); self.draw_idle()
    def set_map(self, data, nrows, ncols):
        self.nrows, self.ncols = nrows, ncols
        self.ax.clear()
        self._img_handle = self.ax.imshow(data, cmap='gray')
        overlay = np.zeros((nrows, ncols, 4))
        self._mask_handle = self.ax.imshow(overlay, extent=self._img_handle.get_extent(),
                                            aspect=self._img_handle.axes.get_aspect())
        self._mask = np.ones((nrows, ncols), dtype=bool)
        self.ax.set_title('IQ Map - Ctrl+Scroll to zoom'); self.ax.axis('off')
        self.marker = None
        self._orig_xlim = self.ax.get_xlim(); self._orig_ylim = self.ax.get_ylim(); self.draw_idle()
    def update_mask(self, mask):
        self._mask = mask
        overlay = np.zeros((self.nrows, self.ncols, 4))
        overlay[~mask, 0] = 1.0; overlay[~mask, 3] = 0.5
        self._mask_handle.set_data(overlay); self.draw_idle()
    def reset_zoom(self):
        if self._orig_xlim is not None:
            self.ax.set_xlim(self._orig_xlim); self.ax.set_ylim(self._orig_ylim); self.draw_idle()


class PatternCanvas(FigureCanvas):
    def __init__(self, title="", parent=None):
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title); self.ax.axis('off')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    def show_pattern(self, pat, title="", fixed_range=False, vmin=None, vmax=None):
        """Display a pattern. Priority:
          1. explicit vmin/vmax → use those (e.g. raw uint8: vmin=0, vmax=255)
          2. fixed_range=True   → [0,1]   (for already-normalized data)
          3. neither            → matplotlib auto-stretch (per-pattern min→max)
        Auto-stretch hides absolute intensity differences across patterns; pass
        explicit vmin/vmax whenever you want to see true relative brightness.
        """
        self.ax.clear()
        if pat is not None:
            if vmin is not None or vmax is not None:
                self.ax.imshow(pat, cmap='gray', vmin=vmin, vmax=vmax)
            elif fixed_range:
                self.ax.imshow(pat, cmap='gray', vmin=0, vmax=1)
            else:
                self.ax.imshow(pat, cmap='gray')
        self.ax.set_title(title); self.ax.axis('off'); self.draw_idle()


class RamUPFile(ebsd_pattern.UPFile):
    """pyebsdindex pattern-file object whose reads are served from the in-RAM copy of
    the dataset (2026-09-28). NLPAR.opt_lambda / calcsigma / calcnlpar read the scan
    THEMSELVES through getinfileobj().read_data(): tile by tile, row by row, one
    np.fromfile per row, single stream from NFS. For a 31 GB scan that is 3-4 min of
    reading per Optimize Lambda and again per NLPAR batch, while the very same bytes
    already sit in self.patterns. Header, copy_file (the NLPAR output is cloned from
    the input) and everything else stay the real UPFile's; only read_data is served
    from RAM. Values are identical: the file's native dtype, cast exactly as
    pat_reader would (convertToFloat -> float32)."""
    def __init__(self, path, ram):
        super().__init__(path)
        self.read_header()
        self._ram = ram                      # (n_patterns, ph, pw) view, native dtype

    def _xyloc(self, idx):
        yx = np.unravel_index(idx.astype(np.uint64), (int(self.nRows), int(self.nCols)))
        xyloc = np.array([yx[1], yx[0]]).T.copy().astype(np.float32)
        xyloc[:, 0] -= self.nCols * 0.5; xyloc[:, 1] -= self.nRows * 0.5
        xyloc[:, 0] *= self.xStep;        xyloc[:, 1] *= self.yStep
        return xyloc

    def read_data(self, path=None, convertToFloat=False, patStartCount=[0, -1], returnArrayOnly=False):
        if not returnArrayOnly:              # packaged EBSDPatterns object: not used by NLPAR
            return super().read_data(path=path, convertToFloat=convertToFloat,
                                     patStartCount=patStartCount, returnArrayOnly=False)
        psc = np.asarray(patStartCount, dtype=np.int64)
        typeout = np.float32 if convertToFloat else self.filedatatype
        ph, pw = int(self.patternH), int(self.patternW)
        ncols, nrows = int(self.nCols), int(self.nRows)
        if psc.ndim == 1:                    # contiguous run of patterns
            p0 = int(psc[0]); n = int(psc[-1])
            if n == -1: n = int(self.nPatterns) - p0
            n = max(1, min(n, int(self.nPatterns) - p0, self._ram.shape[0] - p0))
            pats = self._ram[p0:p0 + n].astype(typeout)
            idx = np.arange(p0, p0 + n, dtype=np.int64)
        else:                                # [[col0, row0], [ncols, nrows]] slab
            c0, nc = int(psc[0, 0]), int(psc[1, 0]); r0, nr = int(psc[0, 1]), int(psc[1, 1])
            if nc < 0: nc = ncols - c0
            if nr < 0: nr = nrows - r0
            nc = min(nc, ncols - c0); nr = min(nr, nrows - r0)
            grid = self._ram[:nrows * ncols].reshape(nrows, ncols, ph, pw)   # view
            pats = grid[r0:r0 + nr, c0:c0 + nc].reshape(nr * nc, ph, pw).astype(typeout)
            idx = (np.arange(r0, r0 + nr, dtype=np.int64)[:, None] * ncols
                   + np.arange(c0, c0 + nc, dtype=np.int64)[None, :]).ravel()
        return pats, self._xyloc(idx)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Kikuchi Pattern Processor v{__version__} " + ("(GPU Accelerated)" if HAS_GPU else f"(CPU Mode, {n_workers()} threads)")); self.setGeometry(100, 100, 1400, 900)
        self.orig_path = None; self.f_orig = self.f_nlpar = None
        # Tier 1 (2026-09-27): the raw dataset held in RAM as one (npat, ph, pw) array in
        # the file's native dtype, when it fits. Every whole-dataset pass then slices this
        # instead of re-streaming the file (each pass was a full 32 GB read from home NFS).
        self.patterns = None
        # Workflow gates (2026-09-27, for students using the app unassisted):
        # Optimize lambda -> Apply NLPAR to full dataset -> Apply && Save. Set when
        # optimize_lambda succeeds; cleared on file load and aperture change (lambda is
        # fitted for one pixel mask). Editing the lambda box by hand does NOT clear it.
        self._lambda_optimized = False
        self.nrows = self.ncols = self.current_row = self.current_col = 0
        self.mean_pattern = None; self.iq_map = None; self.iq_mask = None
        self.pixel_std = None  # per-pixel std across dataset, for variance-based aperture detection
        # Per-pixel noise variance σ² map for canonical NLPAR (Brewick et al. 2019, Eq. 9).
        # Computed lazily; invalidated on file load or aperture change.
        # _sigma_aperture_key stores a tuple identifying which aperture the cached map was built for.
        self.sigma_sq_map = None
        self._sigma_aperture_key = None
        # Resolved aperture cache (see _current_aperture); keyed by mode/threshold/shape.
        self._aperture_cache = None
        self._aperture_cache_key = None
        self._temp_files = []
        # Global-stretch bounds (dataset-wide percentile, active pixels only).
        # Computed lazily, cached; invalidated on file load. Same (lo,hi) applied
        # to every pattern → preserves relative intensity ordering (IQ-safe).
        self.global_lo = None
        self.global_hi = None
        self._global_range_key = None
        self._build_ui()

    def _remove_temp_files(self):
        """Delete the NLPAR batch result files this session created (next to their
        inputs). Called on app exit and when a new dataset is loaded (Max, 2026-09-27),
        so a 5-32 GB _cpu_nlpar_result.up1 doesn't sit around until the app closes."""
        for f in self._temp_files:
            try:
                if os.path.isfile(f): os.remove(f); print(f"[TEMP] removed {f}")
            except Exception as e:
                print(f"[TEMP] could not remove {f}: {e}")
        self._temp_files = []

    def closeEvent(self, event):
        self._remove_temp_files()
        event.accept()

    def _get_iq_mask(self):
        if self.iq_map is None: return None
        iq_lo, iq_hi = self.iq_map.min(), self.iq_map.max()
        vmin = iq_lo + (iq_hi - iq_lo) * self.iq_min_slider.value() / 1000.0
        vmax = iq_lo + (iq_hi - iq_lo) * self.iq_max_slider.value() / 1000.0
        return (self.iq_map >= vmin) & (self.iq_map <= vmax)

    @staticmethod
    def _make_aperture_mask(h, w):
        """Aperture mask matching pyebsdindex's NLPAR.makeautomask exactly.
        Inscribed circle of radius 0.98*min(h,w)/2, centered via np.roll trick.
        Source: pyebsdindex/nlpar_cpu.py (David Rowenhorst, NRL - public domain).
        Disable for non-circular detector geometries (e.g. RKD)."""
        r = (min(h, w) * 0.98 * 0.5)
        x = np.arange(w, dtype=np.float32)
        x = np.minimum(x, (w - x))
        x = x.reshape(1, w)
        y = np.arange(h, dtype=np.float32)
        y = np.minimum(y, (h - y))
        y = y.reshape(h, 1)
        mask = np.sqrt(y ** 2 + x ** 2)
        mask = (mask < r).astype(np.uint8)
        mask = np.roll(mask, (int(h / 2), int(w / 2)), axis=(0, 1))
        return mask

    def _make_variance_mask(self, threshold_frac=0.05):
        """Auto-detect active detector pixels from per-pixel std across the dataset.
        Dead pixels (chip gaps, masked regions) have ~zero std; active pixels accumulate
        signal that varies pattern-to-pattern. Threshold = threshold_frac * 95th-percentile
        of std (95th percentile rather than max() avoids hot pixels setting the scale).
        Returns uint8 mask (1=active, 0=dead) or None if pixel_std hasn't been computed."""
        if self.pixel_std is None:
            return None
        p95 = float(np.percentile(self.pixel_std, 95))
        thresh = threshold_frac * p95
        return (self.pixel_std > thresh).astype(np.uint8)

    def _resolve_aperture_mask(self, ph, pw):
        """Returns (mask_2d_float32 or None, aperture_count_float, label_str) based on UI mode."""
        mode = self.aperture_mode.currentText()
        if mode == "Circular (EBSD)":
            m = self._make_aperture_mask(ph, pw).astype(np.float32)
            return m, float(m.sum()), "CIRCULAR"
        if mode == "Auto-detect":
            m_u8 = self._make_variance_mask(self.aperture_thresh_spin.value())
            if m_u8 is None:
                return None, float(ph * pw), "AUTO-DETECT (UNAVAILABLE - falling back to NONE)"
            m = m_u8.astype(np.float32)
            return m, float(m.sum()), "AUTO-DETECT"
        return None, float(ph * pw), "NONE"

    def _on_aperture_changed(self):
        """Aperture is a global gate: it changes N_p (NLPAR), the σ² estimate, the
        global-stretch percentile population, and which pixels survive to the output.
        Every cache that depends on it must be dropped and the preview refreshed."""
        self._invalidate_sigma_map()
        self._invalidate_global_range()
        self._lambda_optimized = False  # lambda was fitted for the previous aperture
        if self.f_orig is not None:
            self.update_display()

    def _current_aperture(self):
        """Resolved aperture as a (ph, pw) float32 array of 1.0/0.0, or None when the
        mode is 'None' (or auto-detect has no pixel_std yet). Cached per (mode, threshold,
        shape) so the per-chunk save loop isn't rebuilding it on every iteration."""
        if self.f_orig is None: return None
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        key = (self.aperture_mode.currentText(), self.aperture_thresh_spin.value(), ph, pw)
        if getattr(self, '_aperture_cache_key', None) != key:
            m, _, _ = self._resolve_aperture_mask(ph, pw)
            self._aperture_cache = m
            self._aperture_cache_key = key
        return self._aperture_cache

    def _host_chunk_rows(self, bytes_per_pixel_live, budget=None, n_rows_total=None):
        """Rows per chunk that keep host RAM for one chunk inside `budget` bytes.

        `bytes_per_pixel_live` is the SUM of bytes/pixel over every copy of the chunk that
        is alive at the same time (e.g. uint8 read + float32 cast + a float32 temporary = 9).
        Hardcoded row counts blew up on large scans: 430 cols x 479x479 px is ~99 MB/row as
        uint8 but ~790 MB/row as float64, so 50 rows of float64 is ~40 GB. Default budget is
        the smaller of 3 GB and a quarter of the currently AVAILABLE RAM, so a busy machine
        chunks smaller instead of swapping (or dying with _ArrayMemoryError)."""
        if budget is None:
            # Was min(3 GB, 25% of available). On a 500 GB OSC node that meant 1-row chunks
            # (a 430-col 479x479 row is ~1.8 GB at 18 B/px), starving the threaded paths.
            # Scale the cap with the thread count, but never exceed a quarter of what the
            # SLURM cgroup actually lets us allocate.
            budget = min(1.0e9 * max(4, n_workers() // 2), 0.25 * available_ram())
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        bytes_per_row = self.ncols * ph * pw * float(bytes_per_pixel_live)
        n_total = n_rows_total if n_rows_total is not None else self.nrows
        return max(1, min(int(n_total), int(budget // max(bytes_per_row, 1))))

    def _invalidate_global_range(self):
        """Discard cached global-stretch bounds. Call on file load or when the
        background configuration changes (subtract/divide/off, or recomputed mean)."""
        self.global_lo = None
        self.global_hi = None
        self._global_range_key = None

    def _bg_config_key(self):
        """Identifier for the STATIC background config; global-range cache invalidates
        when this changes so (lo,hi) is always computed in the same domain it's applied in.
        Global stretch is intentionally coupled to STATIC background only — dynamic
        background is excluded so dragging its sigma never triggers a dataset rescan."""
        static_on = self.static_bg_chk.isChecked() and self.mean_pattern is not None
        mode = self.bg_mode.currentText() if static_on else None
        # Mean pattern identity matters (recompute-from-IQ changes it); use a cheap hash.
        mean_id = float(np.sum(self.mean_pattern)) if static_on else None
        return (static_on, mode, mean_id)

    def _apply_static_background(self, chunk, mean_dtype=None):
        """Apply ONLY the current STATIC background to a chunk in-memory (no file).
        Used for the global-stretch range so its domain matches static subtraction.
        Dynamic background is deliberately NOT applied here — global stretch works
        with static background only.

        mean_dtype: cast mean_pattern (float64) down before the arithmetic. The range scan
        passes float32 so a whole chunk isn't silently promoted to float64 (2x RAM) just
        because the mean pattern is float64."""
        out = chunk
        if self.static_bg_chk.isChecked() and self.mean_pattern is not None:
            mp = self.mean_pattern if mean_dtype is None else self.mean_pattern.astype(mean_dtype, copy=False)
            if self.bg_mode.currentText() == "Subtract":
                out = out - mp
            else:
                out = out / (mp + 1e-10)
        return out

    def _ensure_global_range(self):
        """Compute dataset-wide (0.5, 99.5) percentile intensity bounds for the global
        stretch, over ACTIVE pixels only (dead/vignetted corners excluded), AFTER applying
        the current background settings so the range lives in the same domain it's applied
        in. Cached and keyed by the background config; recomputed when that changes.

        Two-pass streaming (background-applied values are signed floats, so a fixed uint8
        histogram won't work):
          pass 1 — global min/max of background-applied active pixels
          pass 2 — 1024-bin histogram over [min,max] → 0.5/99.5 percentiles
        Same (lo,hi) is then applied identically to every pattern → relative pattern-to-
        pattern band contrast (and thus IQ ordering) is preserved.
        """
        key = self._bg_config_key()
        if self.global_lo is not None and self._global_range_key == key:
            return
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        # Active-pixel mask over the pattern frame (exclude dead corners).
        if self.pixel_std is not None:
            p95 = float(np.percentile(self.pixel_std, 95))
            active = self.pixel_std > (0.05 * p95)
            if not active.any():
                active = np.ones((ph, pw), dtype=bool)
        else:
            active = np.ones((ph, pw), dtype=bool)
        active_flat = active.ravel()
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        # Size chunk to the live host-RAM budget. float64 here was gratuitous: percentiles
        # of 8-bit data don't need it, and it doubled every copy. Live per pixel:
        # native read (1–2) + float32 cast (4) + bg-applied temporary (4) + the
        # active-pixel gather (≤4) ≈ 13.
        CHUNK_ROWS = self._host_chunk_rows(13, n_rows_total=max(n_avail_rows, 1))
        print(f"[GLOBAL STRETCH] CHUNK_ROWS: {CHUNK_ROWS} of {n_avail_rows} rows")

        # ---- Pass 1: global min/max of background-applied active pixels ----
        self.statusBar().showMessage("Global range pass 1/2 (min/max)..."); QApplication.processEvents()
        gmin, gmax = np.inf, -np.inf
        for row_start in range(0, n_avail_rows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, n_avail_rows - row_start)
            chunk = self._read_rows(self.f_orig, row_start, rows_to_read)
            chunk = self._apply_static_background(chunk.astype(np.float32), mean_dtype=np.float32)
            vals = chunk.reshape(chunk.shape[0], -1)[:, active_flat]
            if vals.size:
                gmin = min(gmin, float(vals.min())); gmax = max(gmax, float(vals.max()))
            del chunk, vals
            self.statusBar().showMessage(f"Global range p1: row {row_start + rows_to_read}/{n_avail_rows}")
            QApplication.processEvents()
        if not np.isfinite(gmin) or gmax <= gmin:
            self.global_lo, self.global_hi = 0.0, 255.0
            self._global_range_key = key
            return

        # ---- Pass 2: histogram over [gmin,gmax] → percentiles ----
        self.statusBar().showMessage("Global range pass 2/2 (percentiles)..."); QApplication.processEvents()
        NB = 1024
        hist = np.zeros(NB, dtype=np.int64)
        edges = np.linspace(gmin, gmax, NB + 1)
        for row_start in range(0, n_avail_rows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, n_avail_rows - row_start)
            chunk = self._read_rows(self.f_orig, row_start, rows_to_read)
            chunk = self._apply_static_background(chunk.astype(np.float32), mean_dtype=np.float32)
            vals = chunk.reshape(chunk.shape[0], -1)[:, active_flat].ravel()
            hist += np.histogram(vals, bins=edges)[0]
            del chunk, vals
            self.statusBar().showMessage(f"Global range p2: row {row_start + rows_to_read}/{n_avail_rows}")
            QApplication.processEvents()
        total = hist.sum()
        if total == 0:
            self.global_lo, self.global_hi = gmin, gmax
            self._global_range_key = key
            return
        cdf = np.cumsum(hist) / total
        centers = 0.5 * (edges[:-1] + edges[1:])
        lo = float(centers[np.searchsorted(cdf, 0.005)])
        hi = float(centers[min(np.searchsorted(cdf, 0.995), NB - 1)])
        if hi <= lo:
            hi = lo + (gmax - gmin) / NB
        self.global_lo, self.global_hi = lo, hi
        self._global_range_key = key
        self.statusBar().showMessage(f"Global stretch range: [{lo:.2f}, {hi:.2f}] (0.5–99.5 pctile, bg-applied, active px)")
        print(f"[GLOBAL STRETCH] in_range = [{lo:.3f}, {hi:.3f}] over {int(active.sum())}/{ph*pw} active px, bg_key={key}")


    def show_aperture_mask(self):
        """Preview the currently-selected aperture mask: shows the currently-selected
        raw pattern with dead zones blacked out, normalized over the active region.
        Click around the IQ map and re-click this button to spot-check the mask
        against different regions."""
        if self.f_orig is None:
            QMessageBox.warning(self, "Error", "Load a UP1 file first")
            return
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        m, count, label = self._resolve_aperture_mask(ph, pw)
        n_total = ph * pw
        if m is None:
            # "None" mode or auto-detect with no std: show unmasked pattern
            active_mask = np.ones((ph, pw), dtype=np.float32)
            count = float(n_total)
        else:
            active_mask = m
        # Use the currently-displayed raw pattern (not the mean)
        pat = self._read_pattern(self.f_orig).astype(np.float64)
        # Normalize over active pixels only, then zero dead zones
        active_vals = pat[active_mask > 0]
        if len(active_vals) > 0:
            vmin, vmax = float(active_vals.min()), float(active_vals.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmax > vmin:
            disp = (pat - vmin) / (vmax - vmin)
        else:
            disp = pat.copy()
        disp = np.clip(disp, 0, 1) * active_mask  # blacken dead zones
        self.proc_canvas.show_pattern(disp.astype(np.float64),
            f"Mask [{label}] @ [{self.current_col},{self.current_row}]: {int(count)}/{n_total} active ({100*count/n_total:.1f}%)",
            fixed_range=True)

    # ---- On-demand neighborhood reading ----
    def _read_neighborhood(self, fobj, row, col, radius):
        """Read a (2r+1) x (2r+1) neighborhood of patterns around (row, col).
        Returns (patterns_2d_array, mask_2d, center_index_in_array)."""
        r = radius
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        r0, r1 = max(0, row - r), min(n_avail_rows, row + r + 1)
        c0, c1 = max(0, col - r), min(self.ncols, col + r + 1)
        n_rows_read = r1 - r0; n_cols_read = c1 - c0
        if n_rows_read <= 0 or n_cols_read <= 0:
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            dummy = np.zeros((1, 1, ph, pw))
            return dummy, np.ones((1, 1), dtype=bool), 0, 0
        pats = self._read_rows(fobj, r0, n_rows_read, c0, n_cols_read, as_float=True)
        ph, pw = int(fobj.patternH), int(fobj.patternW)
        pats = pats.reshape(n_rows_read, n_cols_read, ph, pw)
        # Build mask for this neighborhood
        mask = self.iq_mask[r0:r1, c0:c1] if self.iq_mask is not None else np.ones((n_rows_read, n_cols_read), dtype=bool)
        # Center pattern index within the read block
        center_r, center_c = row - r0, col - c0
        return pats, mask, center_r, center_c

    def _compute_npar_single(self, row, col):
        """Compute NPAR for a single pixel on-demand."""
        r = self.npar_radius_spin.value()
        pats, mask, cr, cc = self._read_neighborhood(self.f_orig, row, col, r)
        valid = mask.astype(np.float64)
        weight_sum = valid.sum()
        if weight_sum < 1: return pats[cr, cc]
        result = np.sum(pats * valid[:, :, np.newaxis, np.newaxis], axis=(0, 1)) / weight_sum
        return result

    def _compute_nlpar_single(self, row, col):
        """Compute NLPAR for a single pixel on-demand using the canonical
        Brewick/Wright/Rowenhorst 2019 formulation (Ultramicroscopy 200, 50-61).

        Equations:
          (3) Raw squared Euclidean distance:
              d(p_i, p_j) = Σ_k (p_i[k] - p_j[k])²    (over masked pixels k)
          (7) Normalized dissimilarity (a Z-score under the null hypothesis u_i = u_j):
              d̃ = [d(p_i, p_j) - N_p(σ²_i + σ²_j)] / [sqrt(2 N_p) · (σ²_i + σ²_j)]
          (8) Weight with negative-d̃ clipping:
              w(i, j) = exp(-max(d̃, 0) / λ²)

        With this normalization, λ is dimensionless and canonical values are 0.5–2.0
        (opt_lambda typically returns ~0.6–1.2). λ → ∞ recovers NPAR; λ → 0 recovers
        the identity (no smoothing).
        """
        sr = self.sr_spin.value()
        lam = self.lam_spin.value()
        # Ensure sigma² map is available for the current aperture.
        self._ensure_sigma_map()
        pats, mask, cr, cc = self._read_neighborhood(self.f_orig, row, col, sr)
        center_pat = pats[cr, cc].astype(np.float64)
        ph, pw = center_pat.shape
        nr, nc = pats.shape[0], pats.shape[1]

        # Aperture mask + active-pixel count N_p (canonical paper notation).
        aperture_2d, aperture_count, _ = self._resolve_aperture_mask(ph, pw)
        N_p = float(aperture_count)
        ap_flat = aperture_2d.ravel().astype(np.float64) if aperture_2d is not None else None

        # σ² of the center pixel and the corresponding neighborhood slice in sigma_sq_map.
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        r0 = max(0, row - sr); r1 = min(n_avail_rows, row + sr + 1)
        c0 = max(0, col - sr); c1 = min(self.ncols, col + sr + 1)
        sigma_sq_nb = self.sigma_sq_map[r0:r1, c0:c1].astype(np.float64)
        sigma_sq_i = float(self.sigma_sq_map[row, col])

        weighted_sum = np.zeros((ph, pw), dtype=np.float64)
        weight_total = 0.0
        lam_sq = lam * lam
        sqrt_2Np = np.sqrt(2.0 * N_p)
        center_flat = center_pat.ravel()

        for ri in range(nr):
            for ci in range(nc):
                if not mask[ri, ci]:
                    continue
                nb_pat = pats[ri, ci].astype(np.float64)
                diff = nb_pat.ravel() - center_flat
                if ap_flat is not None:
                    d_raw = float(np.sum(diff * diff * ap_flat))
                else:
                    d_raw = float(np.dot(diff, diff))
                sigma_sq_j = float(sigma_sq_nb[ri, ci])
                sigma_combined = sigma_sq_i + sigma_sq_j
                if sigma_combined < 1e-10:
                    # Degenerate σ² → weight only the center pattern.
                    w = 1.0 if (ri == cr and ci == cc) else 0.0
                else:
                    d_norm = (d_raw - N_p * sigma_combined) / (sqrt_2Np * sigma_combined)
                    d_clipped = max(d_norm, 0.0)
                    w = np.exp(-d_clipped / lam_sq)
                weighted_sum += w * nb_pat
                weight_total += w

        if weight_total > 0:
            return weighted_sum / weight_total
        return center_pat

    # ------------------------------------------------------------------
    # σ² map: per-pixel noise variance estimate (Brewick et al. 2019, Eq. 9).
    # ------------------------------------------------------------------
    def _aperture_key(self):
        """Identifier tuple for the current aperture configuration; used for sigma cache invalidation."""
        mode = self.aperture_mode.currentText()
        if mode == "Auto-detect":
            return (mode, float(self.aperture_thresh_spin.value()))
        return (mode, None)

    def _invalidate_sigma_map(self):
        """Discard cached σ² map. Call when file changes or aperture changes."""
        self.sigma_sq_map = None
        self._sigma_aperture_key = None

    def _ensure_sigma_map(self):
        """Compute σ² map for the current aperture if not already cached.
        σ²_i is estimated from the minimum d² to the 8 nearest neighbors (Eq. 9):
            σ̂²_i = min_{j ∈ SR=1}( d(p_i, p_j) / (2 N_p) )
        where d is the squared Euclidean distance summed over the aperture pixels.
        """
        key = self._aperture_key()
        if self.sigma_sq_map is not None and self._sigma_aperture_key == key:
            return
        self.statusBar().showMessage("Computing σ² map for canonical NLPAR (one-time per aperture)...")
        QApplication.processEvents()
        t0 = time.time()
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        aperture_2d, aperture_count, ap_label = self._resolve_aperture_mask(ph, pw)
        N_p = float(aperture_count)
        if HAS_GPU:
            self.sigma_sq_map = self._compute_sigma_map_gpu(aperture_2d, N_p)
        else:
            self.sigma_sq_map = self._compute_sigma_map_cpu(aperture_2d, N_p)
        self._sigma_aperture_key = key
        # Diagnostic summary so we can sanity-check σ values vs. raw pattern intensity scale.
        valid = self.sigma_sq_map[self.sigma_sq_map > 0]
        if len(valid) > 0:
            med = float(np.median(valid)); lo = float(np.percentile(valid, 5)); hi = float(np.percentile(valid, 95))
            print(f"[σ² MAP] aperture={ap_label} N_p={int(N_p)} "
                  f"σ²: median={med:.3g}, 5–95%=[{lo:.3g}, {hi:.3g}], "
                  f"σ (intensity units): median={np.sqrt(med):.3g}, time={time.time()-t0:.1f}s")
        self.statusBar().showMessage(f"σ² map ready ({time.time()-t0:.1f}s)")

    def _compute_sigma_map_gpu(self, aperture_2d, N_p):
        """GPU σ² estimation. SR=1 (8 nearest neighbors), one shift-and-compare pass."""
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        sr = 1
        g_aperture = cp.asarray(aperture_2d) if aperture_2d is not None else None
        sigma_sq_map = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        n_avail = getattr(self, 'n_actual_rows', self.nrows)

        # Memory budget: same shape as main NLPAR pass but with sr=1, so fits easily.
        pat_bytes = ph * pw * 4
        mem_per_row = 5 * (self.ncols + 2 * sr) * pat_bytes
        vram_budget = int(cp.cuda.Device(0).mem_info[1] * 0.5)
        CHUNK_ROWS = max(2, vram_budget // mem_per_row)
        CHUNK_ROWS = min(CHUNK_ROWS, self.nrows)
        # Host RAM caps this too: the float32 read and the float32 `padded` block are both
        # live at once (~4+4 B/px, plus border rows), and an 8.5 GB GPU on a 33 GB host can
        # easily pick a chunk that fits VRAM but not RAM.
        CHUNK_ROWS = min(CHUNK_ROWS, self._host_chunk_rows(12))

        for chunk_start in range(0, n_avail, CHUNK_ROWS):
            chunk_rows = min(CHUNK_ROWS, n_avail - chunk_start)
            read_r0 = max(0, chunk_start - sr)
            read_r1 = min(n_avail, chunk_start + chunk_rows + sr)
            data_np = self._read_rows(self.f_orig, read_r0, read_r1 - read_r0, as_float=True)
            # copy=False: read_data already returned float32, don't duplicate the block.
            data_np = data_np.reshape(read_r1 - read_r0, self.ncols, ph, pw).astype(np.float32, copy=False)

            full_h = chunk_rows + 2 * sr
            full_w = self.ncols + 2 * sr
            padded = np.zeros((full_h, full_w, ph, pw), dtype=np.float32)
            valid = np.zeros((full_h, full_w), dtype=bool)
            dst_r0 = sr - (chunk_start - read_r0)
            dst_r1 = dst_r0 + (read_r1 - read_r0)
            padded[dst_r0:dst_r1, sr:sr+self.ncols, :, :] = data_np
            valid[dst_r0:dst_r1, sr:sr+self.ncols] = True
            del data_np

            g_padded = cp.asarray(padded); g_valid = cp.asarray(valid)
            del padded, valid
            center = g_padded[sr:sr+chunk_rows, sr:sr+self.ncols, :, :]
            min_d_raw = cp.full((chunk_rows, self.ncols), cp.inf, dtype=cp.float32)

            for dr in range(-sr, sr + 1):
                for dc in range(-sr, sr + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = g_padded[sr+dr:sr+dr+chunk_rows, sr+dc:sr+dc+self.ncols, :, :]
                    nb_valid = g_valid[sr+dr:sr+dr+chunk_rows, sr+dc:sr+dc+self.ncols]
                    diff = center - nb
                    if g_aperture is not None:
                        d_raw = (diff * diff * g_aperture).reshape(chunk_rows, self.ncols, -1).sum(axis=2)
                    else:
                        d_raw = (diff * diff).reshape(chunk_rows, self.ncols, -1).sum(axis=2)
                    d_raw = cp.where(nb_valid, d_raw, cp.inf)
                    min_d_raw = cp.minimum(min_d_raw, d_raw)
                    del diff, d_raw, nb, nb_valid

            # σ²_i = min_d_raw / (2 N_p)  (Eq. 9)
            sigma_sq_chunk = min_d_raw / (2.0 * N_p)
            sigma_sq_chunk = cp.where(cp.isfinite(sigma_sq_chunk), sigma_sq_chunk, 0.0)
            sigma_sq_map[chunk_start:chunk_start+chunk_rows, :] = cp.asnumpy(sigma_sq_chunk)
            del g_padded, g_valid, min_d_raw, sigma_sq_chunk
            cp.get_default_memory_pool().free_all_blocks()

            self.statusBar().showMessage(f"σ² map: row {chunk_start+chunk_rows}/{n_avail}")
            QApplication.processEvents()

        # Patch isolated/invalid σ²=0 pixels with the dataset median to avoid div-by-zero downstream.
        valid_sigmas = sigma_sq_map[sigma_sq_map > 0]
        if len(valid_sigmas) > 0:
            med = float(np.median(valid_sigmas))
            sigma_sq_map[sigma_sq_map <= 0] = med
        return sigma_sq_map

    def _compute_sigma_map_cpu(self, aperture_2d, N_p):
        """CPU σ² estimation, vectorized per scan row and threaded across rows.

        Same quantity as before (min over the 8 neighbours of the aperture-weighted SSD,
        divided by 2·N_p): patterns are uint8/uint16 so the float32 differences and their
        squares are exact, and the sums accumulate in float64 as the old loop did. Each
        (row, neighbour offset) is one array op over all columns instead of a Python loop
        over patterns, and the rows of a chunk are spread over a thread pool (numpy
        releases the GIL in these ops)."""
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        npix = ph * pw
        ncols = int(self.ncols)  # pyebsdindex gives numpy uint64; keep index math in Python ints
        sigma_sq_map = np.zeros((int(self.nrows), ncols), dtype=np.float32)
        ap_flat = aperture_2d.ravel().astype(np.float32) if aperture_2d is not None else None
        n_avail = getattr(self, 'n_actual_rows', self.nrows)
        # Live host memory: float32 chunk (4 B/px) + per-thread temporaries of ~2 rows of
        # float32 (the diff and its square). Cap threads so the temporaries fit too.
        row_bytes = ncols * npix * 4
        nthr = int(max(1, min(n_workers(), (0.25 * available_ram()) // (2 * row_bytes))))
        CHUNK_ROWS = self._host_chunk_rows(4, n_rows_total=max(n_avail, 1))
        print(f"[SIGMA MAP CPU] CHUNK_ROWS: {CHUNK_ROWS} of {n_avail} rows, {nthr} threads")
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

        def row_min_ssd(data, lr):
            A = data[lr]                                    # (ncols, npix) float32
            best = np.full(ncols, np.inf, dtype=np.float64)
            for dr, dc in offsets:
                lr2 = lr + dr
                if lr2 < 0 or lr2 >= data.shape[0]:
                    continue
                c0, c1 = max(0, -dc), ncols - max(0, dc)    # centre columns whose neighbour exists
                if c1 <= c0:
                    continue
                diff = data[lr2, c0 + dc:c1 + dc] - A[c0:c1]
                np.multiply(diff, diff, out=diff)
                if ap_flat is not None:
                    d = np.einsum('ij,j->i', diff, ap_flat, dtype=np.float64)
                else:
                    d = diff.sum(axis=1, dtype=np.float64)
                np.minimum(best[c0:c1], d, out=best[c0:c1])
            return best

        with ThreadPoolExecutor(max_workers=nthr) as pool:
            for chunk_start in range(0, n_avail, CHUNK_ROWS):
                chunk_rows = min(CHUNK_ROWS, n_avail - chunk_start)
                read_r0 = max(0, chunk_start - 1)
                read_r1 = min(n_avail, chunk_start + chunk_rows + 1)
                data = self._read_rows(self.f_orig, read_r0, read_r1 - read_r0, as_float=True)
                data = data.reshape(read_r1 - read_r0, ncols, npix)
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                local_rows = [r + (chunk_start - read_r0) for r in range(chunk_rows)]
                for r, best in enumerate(pool.map(lambda lr: row_min_ssd(data, lr), local_rows)):
                    sigma_sq_map[chunk_start + r] = np.where(np.isfinite(best), best / (2.0 * N_p), 0.0)
                del data
                self.statusBar().showMessage(f"σ² map (CPU): row {chunk_start+chunk_rows}/{n_avail}")
                QApplication.processEvents()
        valid_sigmas = sigma_sq_map[sigma_sq_map > 0]
        if len(valid_sigmas) > 0:
            med = float(np.median(valid_sigmas))
            sigma_sq_map[sigma_sq_map <= 0] = med
        return sigma_sq_map

    # ---- Base pattern retrieval (on-demand) ----
    def _serve_nlpar_from_ram(self, nlobj):
        """Make a pyebsdindex NLPAR object read the scan from self.patterns instead of
        re-reading the file (no-op when the dataset is streamed). Mirrors what
        NLPAR.getinfileobj() does with nrows/ncols so a capped row count (map stopped
        early) still applies."""
        if self.patterns is None:
            return False
        ram = RamUPFile(self.orig_path, self.patterns)
        def infile():
            if nlobj.nrows is None: nlobj.nrows = ram.nRows
            else: ram.nRows = nlobj.nrows
            if nlobj.ncols is None: nlobj.ncols = ram.nCols
            else: ram.nCols = nlobj.ncols
            return ram
        nlobj.getinfileobj = infile
        return True

    def _read_patterns(self, fobj, p0, n):
        """Patterns [p0, p0+n) of `fobj` in file order, regardless of row boundaries
        (used for the partial last row of a map that was stopped early). RAM view when
        `fobj` is the in-memory original, else the parallel reader."""
        ph, pw = int(fobj.patternH), int(fobj.patternW)
        p0, n = int(p0), int(n)
        if fobj is self.f_orig and self.patterns is not None:
            return self.patterns[p0:p0 + n]
        itemsize = np.dtype(fobj.filedatatype).itemsize
        return read_patterns_parallel(fobj.filepath, int(fobj.filePos) + p0 * ph * pw * itemsize,
                                      n, ph, pw, fobj.filedatatype)

    def _read_rows(self, fobj, row_start, n_rows, col_start=0, n_cols=None, as_float=False):
        """Patterns of scan rows [row_start, row_start+n_rows) of `fobj`, optionally only
        columns [col_start, col_start+n_cols), as an (n, ph, pw) array in scan order.
        Served from RAM when `fobj` is the original file and it was loaded in memory —
        as a VIEW, so callers must not write into it — otherwise read from the file with
        the parallel reader (full rows) or pyebsdindex (column windows).
        as_float=True returns a fresh float32 copy, like read_data(convertToFloat=True)."""
        ncols = int(fobj.nCols); ph, pw = int(fobj.patternH), int(fobj.patternW)
        row_start, n_rows = int(row_start), int(n_rows)
        if n_cols is None: n_cols = ncols - col_start
        full = (col_start == 0 and n_cols == ncols)
        if fobj is self.f_orig and self.patterns is not None:
            block = self.patterns[row_start * ncols:(row_start + n_rows) * ncols]
            if not full:
                block = block.reshape(n_rows, ncols, ph, pw)[:, col_start:col_start + n_cols].reshape(-1, ph, pw)
        elif full:
            itemsize = np.dtype(fobj.filedatatype).itemsize
            block = read_patterns_parallel(fobj.filepath, int(fobj.filePos) + row_start * ncols * ph * pw * itemsize,
                                           n_rows * ncols, ph, pw, fobj.filedatatype)
        else:
            block, _ = fobj.read_data(returnArrayOnly=True, convertToFloat=False,
                patStartCount=[[col_start, row_start], [n_cols, n_rows]])
        return block.astype(np.float32) if as_float else block

    def _read_pattern(self, fobj, row=None, col=None):
        if row is None: row = self.current_row
        if col is None: col = self.current_col
        # Return zeros for patterns beyond actual data on disk
        pat_idx = row * self.ncols + col
        n_avail = getattr(self, 'n_actual_pats', self.nrows * self.ncols)
        if fobj is self.f_orig and pat_idx >= n_avail:
            return np.zeros((int(self.f_orig.patternH), int(self.f_orig.patternW)))
        if fobj is self.f_orig and self.patterns is not None:
            return self.patterns[int(pat_idx)].astype(np.float32)
        try:
            pat, _ = fobj.read_data(returnArrayOnly=True, convertToFloat=True,
                patStartCount=[[col, row], [1, 1]])
            return pat[0]
        except:
            return np.zeros((int(self.f_orig.patternH), int(self.f_orig.patternW)))

    def _get_base_pattern(self):
        """Get the base pattern for display, applying on-demand NPAR/NLPAR."""
        # Priority: loaded NLPAR file > on-demand NLPAR > on-demand NPAR > raw
        if self.f_nlpar is not None:
            return self._read_pattern(self.f_nlpar)
        if self.nlpar_chk.isChecked():
            try:
                result = self._compute_nlpar_single(self.current_row, self.current_col)
                raw = self._read_pattern(self.f_orig)
                diff = np.abs(result - raw).max()
                print(f"[NLPAR DEBUG] max diff from raw: {diff:.6f}, result range: [{result.min():.2f}, {result.max():.2f}], raw range: [{raw.min():.2f}, {raw.max():.2f}]")
                return result
            except Exception as e:
                print(f"[NLPAR DEBUG] ERROR: {e}")
                import traceback; traceback.print_exc()
                return self._read_pattern(self.f_orig)
        if self.npar_chk.isChecked():
            return self._compute_npar_single(self.current_row, self.current_col)
        return self._read_pattern(self.f_orig)

    def process_pattern(self, pat):
        """Single-pattern preview pipeline. Output range depends on normalize toggle:
          normalize on  → returns values in [0, 1]
          normalize off → returns values in absolute uint8 scale [0, 255]
        The two output ranges are handled in update_display when picking display vmin/vmax,
        and in apply_and_save / _process_chunk_batch when scaling to uint8 for the UP1 file.
        """
        if pat is None: return None
        p = pat.astype(np.float64)
        # ---- Aperture: statistics gate only [MVM 2026-09-08] ----
        # The aperture is NEVER multiplied into pixel values. It only supplies the
        # active-pixel boolean so per-pattern min/max, global stretch bounds and the
        # CLAHE source range are computed over real signal. Without that, pmin would
        # be pinned by the dead field and every pattern would be stretched against a
        # denominator that is too large. Masked pixels flow through the pipeline
        # untouched so they survive to the exported file.
        ap = self._current_aperture()
        ap_bool = None
        if ap is not None:
            ap_bool = ap > 0
        if self.static_bg_chk.isChecked() and self.mean_pattern is not None:
            if self.bg_mode.currentText() == "Subtract": p = p - self.mean_pattern
            else: p = p / (self.mean_pattern + 1e-10)
        if self.dyn_bg_chk.isChecked():
            p = p - gaussian_filter(p, sigma=self.dyn_sigma.value() / 10.0)
        normalize = self.normalize_chk.isChecked()
        global_stretch = self.global_chk.isChecked()
        bval, cval = self.bright.value() / 100.0, self.cont.value() / 100.0
        if global_stretch:
            # Dataset-wide stretch: SAME (lo,hi) for every pattern → relative
            # intensity preserved (IQ-safe). Output in [0,1] like per-pattern path.
            self._ensure_global_range()
            lo, hi = self.global_lo, self.global_hi
            if hi > lo: p = (p - lo) / (hi - lo)
            p = (p - 0.5) * (1.0 + cval) + 0.5 + bval
            if self.clahe_chk.isChecked():
                p = np.clip(p, 0, 1)
                p = equalize_adapthist(p, clip_limit=self.clahe_clip.value() / 1000.0)
            p = np.clip(p, 0, 1)
            return p
        if normalize:
            # Per-pattern min-max stretch to [0,1], then B/C in [0,1] units (mid=0.5).
            # min/max over ACTIVE pixels only — dead pixels must not set the range.
            src = p[ap_bool] if ap_bool is not None else p
            pmin, pmax = src.min(), src.max()
            if pmax > pmin: p = (p - pmin) / (pmax - pmin)
            p = (p - 0.5) * (1.0 + cval) + 0.5 + bval
            if self.clahe_chk.isChecked():
                p = np.clip(p, 0, 1)
                p = equalize_adapthist(p, clip_limit=self.clahe_clip.value() / 1000.0)
            p = np.clip(p, 0, 1)
            return p
        else:
            # Preserve absolute intensity. B/C centered on the uint8 midpoint (127.5).
            # bright slider [-100..100]/100 → bval is in [-1,1], scaled to ±255 here.
            p = (p - 127.5) * (1.0 + cval) + 127.5 + bval * 255.0
            if self.clahe_chk.isChecked():
                # CLAHE requires [0,1]; temporarily scale per-pattern, apply, then
                # map back to the same [pmin, pmax] absolute span we started with.
                # This preserves the absolute intensity scale while still getting
                # CLAHE's local-histogram redistribution.
                src = p[ap_bool] if ap_bool is not None else p
                pmin, pmax = src.min(), src.max()
                if pmax > pmin:
                    p_scaled = np.clip((p - pmin) / (pmax - pmin), 0, 1)
                    p_scaled = equalize_adapthist(p_scaled, clip_limit=self.clahe_clip.value() / 1000.0)
                    p = p_scaled * (pmax - pmin) + pmin
            p = np.clip(p, 0, 255)
            return p

    # ---- Batch processing (GPU-accelerated when available) ----
    def _process_chunk_batch(self, chunk, mask_rows):
        """Process a chunk of patterns in batch. chunk: (N, ph, pw) float64.
        Returns processed chunk in float64, clipped to [0, 255] ready for uint8 write.
        Mirrors process_pattern: when normalize is on, internal range is [0,1] and
        we *255 at the end; when off, internal range stays in absolute uint8 scale
        and we just clip."""
        N = chunk.shape[0]
        use_static = self.static_bg_chk.isChecked() and self.mean_pattern is not None
        use_dyn = self.dyn_bg_chk.isChecked()
        use_clahe = self.clahe_chk.isChecked()
        normalize = self.normalize_chk.isChecked()
        global_stretch = self.global_chk.isChecked()
        if global_stretch:
            self._ensure_global_range()
            g_lo, g_hi = self.global_lo, self.global_hi
        # [0,1]-domain flag: both per-pattern normalize and global stretch land in [0,1].
        unit_domain = normalize or global_stretch
        sigma = self.dyn_sigma.value() / 10.0
        bval = self.bright.value() / 100.0
        cval = self.cont.value() / 100.0
        clip_limit = self.clahe_clip.value() / 1000.0
        # Aperture gate, resolved once for the whole chunk. ap_bool is the CPU-side
        # active-pixel boolean used by the CLAHE branch below.
        ap = self._current_aperture()
        ap_bool = (ap > 0) if ap is not None else None

        if HAS_GPU:
            d = cp.asarray(chunk, dtype=cp.float32)
            g_ap = cp.asarray(ap, dtype=cp.float32) if ap is not None else None

            # Static background (broadcast across batch)
            if use_static:
                gpu_mean = cp.asarray(self.mean_pattern, dtype=cp.float32)
                if self.bg_mode.currentText() == "Subtract":
                    d -= gpu_mean
                else:
                    d /= (gpu_mean + 1e-10)
                del gpu_mean

            # Dynamic background: batch gaussian filter sigma=(0, s, s) skips batch dim
            if use_dyn:
                d -= cp_ndimage.gaussian_filter(d, sigma=(0, sigma, sigma))

            if global_stretch:
                # Dataset-wide stretch: same (lo,hi) for all → IQ-safe. Lands in [0,1].
                if g_hi > g_lo:
                    d = (d - cp.float32(g_lo)) / cp.float32(g_hi - g_lo)
                d = (d - 0.5) * (1.0 + cval) + 0.5 + bval
            elif normalize:
                # Per-pattern normalize: compute min/max along pixel dims over ACTIVE
                # pixels only. Dead pixels are pushed to ±inf so they can never win the
                # reduction; otherwise pmin pins to 0 and every pattern is compressed.
                flat = d.reshape(N, -1)
                if g_ap is not None:
                    act = (g_ap.reshape(1, -1) > 0)
                    pmin = cp.where(act, flat, cp.float32(cp.inf)).min(axis=1).reshape(N, 1, 1)
                    pmax = cp.where(act, flat, cp.float32(-cp.inf)).max(axis=1).reshape(N, 1, 1)
                    del act
                else:
                    pmin = flat.min(axis=1).reshape(N, 1, 1)
                    pmax = flat.max(axis=1).reshape(N, 1, 1)
                spread = pmax - pmin
                valid = spread > 0
                d = cp.where(valid, (d - pmin) / cp.where(valid, spread, cp.ones_like(spread)), d)
                del flat, pmin, pmax, spread, valid
                # B/C in [0,1] units
                d = (d - 0.5) * (1.0 + cval) + 0.5 + bval
            else:
                # Preserve absolute intensity. B/C centered on uint8 midpoint.
                d = (d - cp.float32(127.5)) * (1.0 + cval) + cp.float32(127.5) + cp.float32(bval * 255.0)

            # Transfer back to CPU
            result = cp.asnumpy(d).astype(np.float64)
            del d
            if g_ap is not None: del g_ap
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU: run the whole chain (background, stretch, CLAHE, final scale) on
            # sub-batches across a thread pool; see _cpu_pipeline_threaded.
            result = self._cpu_pipeline_threaded(chunk, mask_rows,
                use_static=use_static, bg_subtract=(self.bg_mode.currentText() == "Subtract"),
                use_dyn=use_dyn, use_clahe=use_clahe, normalize=normalize,
                global_stretch=global_stretch, unit_domain=unit_domain, sigma=sigma,
                bval=bval, cval=cval, clip_limit=clip_limit, ap_bool=ap_bool,
                g_lo=(g_lo if global_stretch else None), g_hi=(g_hi if global_stretch else None))
            result[~mask_rows] = 0.0
            return result

        # CLAHE stays on CPU (per-pattern, no good batch GPU impl yet).
        # When normalize is on, input is in [0,1] and we apply CLAHE directly.
        # When off, we temporarily scale per-pattern to [0,1] for CLAHE then map
        # back to the original [pmin, pmax] span to preserve absolute intensity.
        if use_clahe:
            for i in range(N):
                if not mask_rows[i]:
                    continue
                if unit_domain:
                    p = np.clip(result[i], 0, 1)
                    result[i] = equalize_adapthist(p, clip_limit=clip_limit)
                else:
                    pi = result[i]
                    src_i = pi[ap_bool] if ap_bool is not None else pi
                    pmin_i, pmax_i = src_i.min(), src_i.max()
                    if pmax_i > pmin_i:
                        p_scaled = np.clip((pi - pmin_i) / (pmax_i - pmin_i), 0, 1)
                        p_scaled = equalize_adapthist(p_scaled, clip_limit=clip_limit)
                        result[i] = p_scaled * (pmax_i - pmin_i) + pmin_i

        # Final scale to [0, 255]. When in unit domain (per-pattern normalize OR
        # global stretch), data is in [0,1] so *255 first; when off, data is already
        # in absolute uint8 scale so just clip.
        if unit_domain:
            result = np.clip(result * 255.0, 0, 255)
        else:
            result = np.clip(result, 0, 255)
        # [MVM 2026-09-08] The aperture is deliberately NOT re-asserted here. It is a
        # statistics gate only (min/max, global stretch bounds, CLAHE source range,
        # N_p in NLPAR); masked pixels keep their own processed values and reach disk
        # intact, so exported patterns are complete for OIM/EMSphInx/OpenXY.
        # IQ-range exclusion below is a different thing and still zeroes whole patterns.
        result[~mask_rows] = 0.0
        return result

    def _cpu_pipeline_subbatch(self, sub, mask_sub, use_static, bg_subtract, use_dyn, use_clahe,
                               normalize, global_stretch, unit_domain, sigma, bval, cval,
                               clip_limit, ap_bool, g_lo, g_hi):
        """CPU processing chain for one sub-batch (n, ph, pw) float64 -> float64 in [0,255].
        This is the original CPU fallback of _process_chunk_batch plus its CLAHE loop and
        final scaling, unchanged in the math, split out so several copies can run at once.
        No Qt widget access in here: everything UI-derived arrives as an argument."""
        result = sub.copy()
        n = result.shape[0]
        if use_static:
            if bg_subtract:
                result -= self.mean_pattern
            else:
                result /= (self.mean_pattern + 1e-10)
        if use_dyn:
            # sigma=(0, s, s): first axis is the batch
            result -= gaussian_filter(result, sigma=(0, sigma, sigma))
        if global_stretch:
            if g_hi > g_lo:
                result = (result - g_lo) / (g_hi - g_lo)
            result = (result - 0.5) * (1.0 + cval) + 0.5 + bval
        elif normalize:
            flat = result.reshape(n, -1)
            if ap_bool is not None:
                act = ap_bool.reshape(1, -1)
                pmin = np.where(act, flat, np.inf).min(axis=1).reshape(n, 1, 1)
                pmax = np.where(act, flat, -np.inf).max(axis=1).reshape(n, 1, 1)
            else:
                pmin = flat.min(axis=1).reshape(n, 1, 1)
                pmax = flat.max(axis=1).reshape(n, 1, 1)
            spread = pmax - pmin
            valid = spread > 0
            result = np.where(valid, (result - pmin) / np.where(valid, spread, np.ones_like(spread)), result)
            result = (result - 0.5) * (1.0 + cval) + 0.5 + bval
        else:
            result = (result - 127.5) * (1.0 + cval) + 127.5 + bval * 255.0
        if use_clahe:
            for i in range(n):
                if not mask_sub[i]:
                    continue
                if unit_domain:
                    result[i] = equalize_adapthist(np.clip(result[i], 0, 1), clip_limit=clip_limit)
                else:
                    pi = result[i]
                    src_i = pi[ap_bool] if ap_bool is not None else pi
                    pmin_i, pmax_i = src_i.min(), src_i.max()
                    if pmax_i > pmin_i:
                        p_scaled = np.clip((pi - pmin_i) / (pmax_i - pmin_i), 0, 1)
                        p_scaled = equalize_adapthist(p_scaled, clip_limit=clip_limit)
                        result[i] = p_scaled * (pmax_i - pmin_i) + pmin_i
        if unit_domain:
            result = np.clip(result * 255.0, 0, 255)
        else:
            result = np.clip(result, 0, 255)
        return result

    def _cpu_pipeline_threaded(self, chunk, mask_rows, **kw):
        """Split a chunk into ~3 sub-batches per worker thread and run _cpu_pipeline_subbatch
        on each in a thread pool. scipy.ndimage, the big numpy reductions and most of
        skimage's CLAHE release the GIL, so this scales across the cores SLURM gave us
        (n_workers). Output is assembled into one float64 array of the chunk's shape."""
        N = chunk.shape[0]
        nthr = max(1, min(n_workers(), N))
        nsub = max(1, min(N, nthr * 3))
        bounds = np.linspace(0, N, nsub + 1).astype(int)
        out = np.empty_like(chunk)
        def work(k):
            i0, i1 = int(bounds[k]), int(bounds[k + 1])
            if i1 > i0:
                out[i0:i1] = self._cpu_pipeline_subbatch(chunk[i0:i1], mask_rows[i0:i1], **kw)
        with ThreadPoolExecutor(max_workers=nthr) as pool:
            list(pool.map(work, range(nsub)))
        return out

    # ---- Display ----
    def on_pattern_clicked(self, row, col):
        self.current_row, self.current_col = row, col
        self.scan_canvas._place_marker(row, col); self.update_display()

    def update_display(self):
        if self.f_orig is None: return
        orig = self._read_pattern(self.f_orig)
        # Raw panel: true absolute uint8 scale (vmin=0, vmax=255). Patterns with
        # weaker raw signal will visibly look dimmer, instead of being silently
        # stretched to fill the colormap by matplotlib's auto-normalize.
        self.raw_canvas.show_pattern(orig, f"Original [{self.current_col}, {self.current_row}]",
                                     vmin=0, vmax=255)
        if self.current_row >= self.nrows or self.current_col >= self.ncols: return  # stale click from a larger scan
        if self.iq_mask is not None and not self.iq_mask[self.current_row, self.current_col]:
            self.proc_canvas.show_pattern(None, "Excluded by IQ range"); return
        t0 = time.time()
        base_pat = self._get_base_pattern()
        processed = self.process_pattern(base_pat)
        elapsed = time.time() - t0
        # Build source label
        src = ""
        if self.f_nlpar is not None: src = "NLPAR+"
        elif self.nlpar_chk.isChecked(): src = "NLPAR+"
        elif self.npar_chk.isChecked(): src = "NPAR+"
        # Proc display range follows the normalize toggle:
        #   normalize on  → output is in [0,1]  → fixed_range=True
        #   normalize off → output is in absolute uint8 scale → vmin=0, vmax=255
        # Using vmin=0,vmax=255 in the unnormalized case matches what apply_and_save
        # actually writes to the UP1, so the preview is honest about the export.
        if self.normalize_chk.isChecked() or self.global_chk.isChecked():
            self.proc_canvas.show_pattern(processed,
                f"{src}Processed [{self.current_col}, {self.current_row}]", fixed_range=True)
        else:
            self.proc_canvas.show_pattern(processed,
                f"{src}Processed [{self.current_col}, {self.current_row}]",
                vmin=0, vmax=255)
        self.statusBar().showMessage(f"Pattern processed in {elapsed*1000:.0f} ms")

    # ---- File loading ----
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open UP1 File",
            "", "UP1 Files (*.up1 *.up2)")
        if not path: return
        self.load_path(path)

    def load_path(self, path):
        """Load a UP1/UP2 file by path (no dialog). Split out of load_file so the
        loader can be driven headlessly for testing."""
        self.statusBar().showMessage("Loading patterns..."); QApplication.processEvents()
        # Drop EVERYTHING that belongs to the previous dataset before touching any widget.
        # The slider resets below emit valueChanged -> _update_iq_range -> update_display,
        # so anything still pointing at the old scan at that moment (its NLPAR file, its
        # sigma map, the last clicked row/col, the auto-detect aperture cache) is used
        # against the new one: wrong shapes, IndexError, and with no excepthook installed
        # PyQt aborts the whole app on an exception inside a slot. [2026-09-27 review]
        self.f_nlpar = None; self.nlpar_status.setText("No NLPAR file loaded")
        self._remove_temp_files()          # previous dataset's NLPAR batch result
        self.current_row = self.current_col = 0
        self.iq_map = None; self.iq_mask = None
        self._invalidate_sigma_map()      # sigma map belongs to the old scan
        self._invalidate_global_range()   # so do the global-stretch bounds
        self._lambda_optimized = False    # lambda gate re-arms for the new file
        self._aperture_cache = None; self._aperture_cache_key = None  # auto-detect mask is per dataset
        self.orig_path = path; self.f_orig = ebsd_pattern.get_pattern_file_obj(path)
        self.nrows, self.ncols = self.f_orig.nRows, self.f_orig.nCols
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        # Check for truncated file but KEEP original header dimensions for OIM compat
        fsize = int(os.path.getsize(path))
        bpp = 2 if path.lower().endswith('.up2') else 1
        header_bytes = int(getattr(self.f_orig, 'filePos', 42) or 0)  # real data offset (42 for v3, more for v4)
        self.n_actual_pats = int((fsize - header_bytes) // (ph * pw * bpp))
        self.n_actual_rows = self.n_actual_pats // self.ncols
        n_expected = self.nrows * self.ncols
        if self.n_actual_pats < n_expected:
            self.statusBar().showMessage(
                f"File truncated: {n_expected - self.n_actual_pats} patterns missing at end")
            QApplication.processEvents()
        else:
            self.n_actual_rows = self.nrows
            self.n_actual_pats = n_expected
        # ---- Tier 1: hold the whole dataset in RAM when it fits (2026-09-27) ----
        # Native dtype (uint8 = the file size). Threshold 60% of what this process may
        # still allocate, leaving the rest for the per-pass temporaries the chunk sizing
        # below draws from the same available_ram(). KPP_IN_MEMORY=1/0 forces it.
        self.patterns = None                       # release a previous dataset first
        need = self.n_actual_pats * ph * pw * bpp
        avail = available_ram()
        force = os.environ.get('KPP_IN_MEMORY')
        in_mem = (need <= 0.6 * avail) if force is None else force.strip().lower() not in ('0', 'no', 'false')
        why = f"{need/1e9:.1f} GB dataset, {avail/1e9:.0f} GB RAM available to this process"
        if in_mem:
            print(f"[LOAD] holding the dataset in RAM ({why})")
            self.patterns = np.empty((self.n_actual_pats, ph, pw), dtype=self.f_orig.filedatatype)
            t_read = time.time()
            def prog(done, total):
                rate = done / max(time.time() - t_read, 1e-9) / 1e6
                self.statusBar().showMessage(
                    f"Reading {os.path.basename(path)} into RAM: {done/1e9:.1f}/{total/1e9:.1f} GB ({rate:.0f} MB/s)")
                QApplication.processEvents()
            read_patterns_parallel(path, header_bytes, self.n_actual_pats, ph, pw,
                                   self.f_orig.filedatatype, out=self.patterns, progress=prog, drop_cache=True)
            t_read = time.time() - t_read
            print(f"[LOAD] read {need/1e9:.1f} GB in {t_read:.1f}s ({need/1e6/max(t_read,1e-9):.0f} MB/s)")
            self.mem_mode = "in RAM"
        else:
            print(f"[LOAD] streaming from disk on every pass ({why}; needs <= 60% of available, or KPP_IN_MEMORY=1)")
            self.mem_mode = "streamed from disk"
        # Chunked statistics for IQ map and mean pattern (only rows that exist).
        # Native dtype (uint8/uint16), cast one chunk at a time: a 20-row float32 read
        # of a 430x479x479 scan is ~8 GB before we touch it.
        # Live per pixel: raw chunk (1–2) + float32 cast (4) + std/square temporaries (~8) ≈ 13.
        CHUNK_ROWS = self._host_chunk_rows(13, n_rows_total=max(self.n_actual_rows, 1))
        print(f"[LOAD] {ph}x{pw} patterns, CHUNK_ROWS: {CHUNK_ROWS} of {self.n_actual_rows} rows")
        iq_values = []
        mean_accum = np.zeros((ph, pw), dtype=np.float64)
        sumsq_accum = np.zeros((ph, pw), dtype=np.float64)  # for per-pixel std (variance-based aperture detection)
        total_pats = 0
        # Threaded statistics (OSC): each chunk is read once in this thread, then split
        # into ~nthr slices of patterns that the pool reduces independently. The merge
        # is exact: the per-pattern std is independent of the split, and both running
        # sums are float64 over integer-valued terms, so the result is bit-identical to
        # the old single-threaded loop. Per-slice temporaries are float32 (cast + square),
        # so cap the threads such that they fit in a quarter of the available RAM.
        ncols = int(self.ncols)
        row_bytes_f32 = ncols * ph * pw * 4
        nthr = int(max(1, min(n_workers(), (0.25 * available_ram()) // (2 * row_bytes_f32))))
        print(f"[LOAD] {nthr} threads")

        def slice_stats(sl):
            cf = sl.astype(np.float32)
            iq = cf.std(axis=(1, 2))
            s1 = sl.sum(axis=0, dtype=np.float64)
            s2 = np.square(cf).sum(axis=0, dtype=np.float64)
            return iq, s1, s2

        with ThreadPoolExecutor(max_workers=nthr) as pool:
            for row_start in range(0, self.n_actual_rows, CHUNK_ROWS):
                rows_to_read = min(CHUNK_ROWS, self.n_actual_rows - row_start)
                chunk = self._read_rows(self.f_orig, row_start, rows_to_read)
                n = chunk.shape[0]
                bounds = np.linspace(0, n, min(nthr, n) + 1, dtype=int)
                slices = [chunk[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
                for iq, s1, s2 in pool.map(slice_stats, slices):
                    iq_values.append(iq)
                    mean_accum += s1
                    sumsq_accum += s2
                total_pats += n
                self.statusBar().showMessage(f"Loading: row {row_start + rows_to_read}/{self.nrows}")
                QApplication.processEvents()
                del chunk, slices
        # Pad IQ map with zeros for missing rows
        iq_flat = np.concatenate(iq_values)
        if len(iq_flat) < self.nrows * self.ncols:
            iq_flat = np.concatenate([iq_flat, np.zeros(self.nrows * self.ncols - len(iq_flat))])
        self.iq_map = iq_flat.reshape(self.nrows, self.ncols)
        self.mean_pattern = mean_accum / max(total_pats, 1)
        # Per-pixel std: sqrt(E[X^2] - E[X]^2). max(.,0) guards against tiny negative values from FP precision.
        mean_sq = sumsq_accum / max(total_pats, 1)
        self.pixel_std = np.sqrt(np.maximum(mean_sq - self.mean_pattern ** 2, 0.0))
        self.file_label.setText(f"{os.path.basename(path)}\n{self.ncols} x {self.nrows} = {self.ncols*self.nrows} patterns ({self.mem_mode})")
        self.bg_status.setText("Background: all patterns")
        self.scan_canvas.set_map(self.iq_map, self.nrows, self.ncols)
        self.iq_mask = np.ones((self.nrows, self.ncols), dtype=bool)
        self.iq_min_slider.setValue(0); self.iq_max_slider.setValue(1000); self._update_iq_range()
        self.on_pattern_clicked(0, 0); self.statusBar().showMessage("Ready")

    def load_nlpar_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NLPAR UP1 File",
            "", "UP1 Files (*.up1 *.up2)")
        if not path: return
        self.f_nlpar = ebsd_pattern.get_pattern_file_obj(path)
        self.nlpar_chk.setChecked(False)  # disable on-demand since we have a file
        self.nlpar_status.setText(f"Loaded: {os.path.basename(path)}"); self.update_display()

    def recompute_background(self):
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        mask = self._get_iq_mask()
        n_valid = int(mask.sum())
        if n_valid == 0: QMessageBox.warning(self, "Error", "No patterns in IQ range"); return
        self.statusBar().showMessage("Recomputing background from IQ range..."); QApplication.processEvents()
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        mean_accum = np.zeros((ph, pw), dtype=np.float64)
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        # Native-dtype read + a masked copy of at most the same size; the sum accumulates
        # straight into float64 (exact for integer input). 50 rows of float32 here was
        # ~20 GB on a 430-col 479x479 scan — that's the _ArrayMemoryError we hit.
        CHUNK_ROWS = self._host_chunk_rows(3, n_rows_total=max(n_avail_rows, 1))
        print(f"[BG] CHUNK_ROWS: {CHUNK_ROWS} of {n_avail_rows} rows")
        for row_start in range(0, n_avail_rows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, n_avail_rows - row_start)
            chunk = self._read_rows(self.f_orig, row_start, rows_to_read)
            mask_flat = mask[row_start:row_start+rows_to_read, :].flatten()
            if mask_flat.any(): mean_accum += chunk[mask_flat].sum(axis=0, dtype=np.float64)
            del chunk
            self.statusBar().showMessage(f"Recomputing BG: row {row_start + rows_to_read}/{self.nrows}")
            QApplication.processEvents()
        self.mean_pattern = mean_accum / n_valid
        n_total = self.nrows * self.ncols
        self.bg_status.setText(f"Background: {n_valid}/{n_total} patterns")
        self.statusBar().showMessage(f"Background recomputed from {n_valid} patterns"); self.update_display()

    def _nlpar_chunk_bytes(self):
        """Byte budget to hand pyebsdindex as `chunksize`.

        pyebsdindex's _calcchunks assumes 8 bytes/pixel (a float32 input array plus a
        float32 output array) when it turns this number into a tile size, so a 4 GB
        budget means each in-memory float32 tile is about 2 GB. Its own default is
        min(32 GB, total_RAM/4), which on a 33.7 GB machine asks for an 8.4 GB tile and
        dies with _ArrayMemoryError on a 479x479 scan. Track AVAILABLE RAM instead so a
        busy machine simply tiles smaller."""
        # 16 GB cap (was 4): bigger numba tiles amortize better across many cores.
        return int(min(16.0e9, 0.25 * available_ram()))

    def optimize_lambda(self):
        """Estimate the NLPAR lambda straight off self.orig_path — no temp copy at all.

        Two deliberate departures from the old implementation:

        * The IQ-range exclusion is NOT applied during lambda estimation. lambda is a
          single global number (the median of three Nelder-Mead fits over the whole d2
          population), so the patterns the IQ sliders exclude have negligible influence
          on it. The old code materialised a fully masked duplicate of the scan just to
          drop them, which meant a second copy of the dataset on disk (31 GB for a real
          RKD scan) and float32 read chunks far larger than host RAM.
        * The aperture is passed through as pyebsdindex's own pixel mask
          (`nlobj.mask`). That is the canonical way to keep dead pixels out of d2:
          calcsigma_cpu does `indices = self.mask.flatten().nonzero()[0]` and only those
          pixels ever enter sigma or d2. Zeroing dead pixels in a temp file was both
          wasteful and wrong — `automask=True` then laid pyebsdindex's circular mask on
          top of it, so for RKD data lambda was fitted over
          (circle INTERSECT detected-active) instead of the aperture the rest of the
          viewer uses. automask is now False; for mode "Circular (EBSD)" the aperture we
          hand over IS makeautomask, so that case is unchanged.
        """
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        try:
            sr = self.sr_spin.value()
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            nlobj = nlpar.NLPAR(self.orig_path, searchradius=sr)
            # Truncated file: cap the scan rows pyebsdindex will read. getinfileobj()
            # forces fID.nRows = self.nrows whenever self.nrows is not None, so setting
            # the attribute here is enough (the constructor's nrows= kwarg does the same).
            n_avail = getattr(self, 'n_actual_rows', self.nrows)
            if n_avail < self.nrows:
                nlobj.nrows = int(n_avail)
            from_ram = self._serve_nlpar_from_ram(nlobj)
            ap = self._current_aperture()
            if ap is not None:
                nlobj.mask = (ap > 0).astype(np.uint8)
                N_p = int(nlobj.mask.sum())
                ap_label = self.aperture_mode.currentText()
            else:
                N_p = ph * pw
                ap_label = "None"
            budget = self._nlpar_chunk_bytes()
            print(f"[OPT LAMBDA] file={os.path.basename(self.orig_path)} SR={sr} "
                  f"aperture={ap_label} N_p={N_p}/{ph*pw} ({100.0*N_p/(ph*pw):.1f}%) "
                  f"rows={int(n_avail)}/{self.nrows} chunk_budget={budget/1e9:.2f} GB "
                  f"source={'RAM' if from_ram else 'file'}")
            self.statusBar().showMessage(
                f"Optimizing lambda (aperture={ap_label}, N_p={N_p})..."); QApplication.processEvents()
            nlobj.opt_lambda(chunksize=budget, automask=False, autoupdate=True, backsub=False)
            self.lam_spin.setValue(float(nlobj.lam))
            self._lambda_optimized = True
            print(f"[OPT LAMBDA] lambda = {float(nlobj.lam):.4f}  (N_p = {N_p})")
            self.statusBar().showMessage(f"Optimal lambda: {float(nlobj.lam):.3f} (N_p={N_p})")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage("Optimize lambda failed")
            QMessageBox.critical(self, "Optimize lambda failed", str(e))

    def apply_nlpar_batch(self):
        """Run NLPAR on the full dataset. Uses GPU shift-and-compare when available, else pyebsdindex CPU."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        if not self._lambda_optimized:
            QMessageBox.information(self, "Optimize lambda first",
                "Need to optimize lambda before applying NLPAR to the full dataset.\n\n"
                "Click 'Optimize Lambda', wait for it to finish, then click "
                "'Apply NLPAR to Full Dataset' again.")
            return
        sr, lam = self.sr_spin.value(), self.lam_spin.value()

        if HAS_GPU:
            # The CPU path reports its own failures; only the GPU path needs a guard
            # here so an OOM/driver error becomes a dialog instead of killing the app.
            try:
                self._apply_nlpar_gpu(sr, lam)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.statusBar().showMessage("NLPAR (GPU) failed")
                QMessageBox.critical(self, "NLPAR (GPU) failed", str(e))
        else:
            self._apply_nlpar_cpu(sr, lam)

    def _apply_nlpar_gpu(self, sr, lam):
        """GPU-accelerated NLPAR using the canonical Brewick/Wright/Rowenhorst 2019
        formulation. See _compute_nlpar_single docstring for the equations."""
        # Ensure σ² map is available for the current aperture.
        self._ensure_sigma_map()
        self.statusBar().showMessage(f"Running NLPAR GPU (SR={sr}, lam={lam:.2f})..."); QApplication.processEvents()
        t0 = time.time()
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        lam_sq = np.float32(lam * lam)
        # Aperture mask (pyebsdindex-compatible) - applied to SSD if mode != "None".
        # Modes: None / Circular (EBSD) / Auto-detect (variance-based, for RKD or other geometries).
        aperture_2d, aperture_count, ap_label = self._resolve_aperture_mask(ph, pw)
        N_p = float(aperture_count)
        sqrt_2Np = np.float32(np.sqrt(2.0 * N_p))
        N_p_f32 = np.float32(N_p)
        if aperture_2d is not None:
            g_aperture = cp.asarray(aperture_2d)
            print(f"[GPU NLPAR] Canonical Brewick et al. 2019 formula. Aperture {ap_label}: "
                  f"N_p = {int(N_p)}/{ph*pw} pixels active ({100*N_p/(ph*pw):.1f}%)")
        else:
            g_aperture = None
            print(f"[GPU NLPAR] Canonical Brewick et al. 2019 formula. Aperture {ap_label}: "
                  f"using all N_p = {int(N_p)} pixels")
        # Pre-load σ² map onto GPU (small: nrows*ncols*4 bytes)
        g_sigma_full = cp.asarray(self.sigma_sq_map.astype(np.float32))
        mask = self._get_iq_mask()  # (nrows, ncols) bool
        n_avail = getattr(self, 'n_actual_rows', self.nrows)

        # Output file setup
        tmp_dir = os.path.dirname(self.orig_path)
        out_path = os.path.join(tmp_dir, "_gpu_nlpar_result.up1")
        # Create the file ourselves: source header verbatim + sparse truncate to the
        # number of patterns the source actually has. Letting pyebsdindex create it
        # pre-fills every pattern with zeros one tofile() at a time (see apply_and_save).
        hdr_len = int(self.f_orig.filePos)
        with open(self.orig_path, 'rb') as f_src:
            hdr = f_src.read(hdr_len)
        with open(out_path, 'wb') as f_dst:
            f_dst.write(hdr)
            f_dst.truncate(hdr_len + int(getattr(self, 'n_actual_pats', self.nrows * self.ncols)) * ph * pw
                           * (2 if self.orig_path.lower().endswith('.up2') else 1))
        out_obj = ebsd_pattern.get_pattern_file_obj(out_path)

        # Adaptive chunk size based on VRAM and pattern dimensions
        pat_bytes = ph * pw * 4  # float32
        # Memory per chunk_row: ~5x (padded + weighted_sum + temporaries) × (ncols + 2*sr) × pat_bytes
        mem_per_row = 5 * (self.ncols + 2 * sr) * pat_bytes
        vram_budget = int(cp.cuda.Device(0).mem_info[1] * 0.6)  # use 60% of total VRAM
        CHUNK_ROWS = max(2, vram_budget // mem_per_row)
        CHUNK_ROWS = min(CHUNK_ROWS, self.nrows)
        # Host-side cap: float32 read + float32 `padded` + the float32 result pulled back
        # from the GPU are all live at once (~12 B/px, plus border rows). Fitting VRAM is
        # not the same as fitting RAM.
        CHUNK_ROWS = min(CHUNK_ROWS, self._host_chunk_rows(12))
        print(f"[GPU NLPAR] Pattern: {ph}x{pw}, VRAM budget: {vram_budget/1e9:.1f} GB, CHUNK_ROWS: {CHUNK_ROWS}")

        for chunk_start in range(0, self.nrows, CHUNK_ROWS):
            chunk_rows = min(CHUNK_ROWS, self.nrows - chunk_start)

            # Read range with border (clamped to available data)
            read_r0 = max(0, chunk_start - sr)
            read_r1 = min(n_avail, chunk_start + chunk_rows + sr)
            # Also cap to nrows for mask indexing
            mask_r1 = min(self.nrows, chunk_start + chunk_rows + sr)

            if read_r1 > read_r0:
                data_np = self._read_rows(self.f_orig, read_r0, read_r1 - read_r0, as_float=True)
                # copy=False: read_data already returned float32, don't duplicate the block.
                data_np = data_np.reshape(read_r1 - read_r0, self.ncols, ph, pw).astype(np.float32, copy=False)
            else:
                data_np = np.zeros((0, self.ncols, ph, pw), dtype=np.float32)

            # Build full padded block: (chunk_rows + 2*sr, ncols + 2*sr, ph, pw)
            full_h = chunk_rows + 2 * sr
            full_w = self.ncols + 2 * sr
            padded = np.zeros((full_h, full_w, ph, pw), dtype=np.float32)
            pad_mask = np.zeros((full_h, full_w), dtype=bool)
            # Padded σ² for the chunk: same shape as pad_mask. Pad with 0 (handled below).
            pad_sigma = np.zeros((full_h, full_w), dtype=np.float32)

            # Where does real data sit inside padded array?
            dst_r0 = sr - (chunk_start - read_r0)  # offset in padded for first read row
            dst_r1 = dst_r0 + (read_r1 - read_r0)
            padded[dst_r0:dst_r1, sr:sr+self.ncols, :, :] = data_np

            # Mask: get corresponding rows, pad with False
            mask_slice_r0 = read_r0
            mask_slice_r1 = min(mask_r1, self.nrows)
            real_mask = mask[mask_slice_r0:mask_slice_r1, :]
            pad_mask[dst_r0:dst_r0+real_mask.shape[0], sr:sr+self.ncols] = real_mask
            # σ² for the same rows
            real_sigma = self.sigma_sq_map[mask_slice_r0:mask_slice_r1, :]
            pad_sigma[dst_r0:dst_r0+real_sigma.shape[0], sr:sr+self.ncols] = real_sigma.astype(np.float32)

            del data_np

            # Upload to GPU
            g_padded = cp.asarray(padded)
            g_mask = cp.asarray(pad_mask)
            g_sigma = cp.asarray(pad_sigma)
            del padded, pad_mask, pad_sigma

            # Center patterns: rows [sr : sr+chunk_rows], cols [sr : sr+ncols]
            center = g_padded[sr:sr+chunk_rows, sr:sr+self.ncols, :, :]  # view
            center_sigma = g_sigma[sr:sr+chunk_rows, sr:sr+self.ncols]
            weighted_sum = cp.zeros((chunk_rows, self.ncols, ph, pw), dtype=cp.float32)
            weight_total = cp.zeros((chunk_rows, self.ncols), dtype=cp.float32)

            # Shift-and-compare over neighborhood. Canonical NLPAR formula:
            #   d_raw   = Σ_k (p_i[k] - p_j[k])²  over aperture pixels
            #   d̃     = (d_raw − N_p (σ²_i + σ²_j)) / (sqrt(2 N_p) (σ²_i + σ²_j))
            #   w(i,j) = exp(-max(d̃, 0) / λ²)
            for dr in range(-sr, sr + 1):
                for dc in range(-sr, sr + 1):
                    nb = g_padded[sr+dr:sr+dr+chunk_rows, sr+dc:sr+dc+self.ncols, :, :]
                    nb_mask = g_mask[sr+dr:sr+dr+chunk_rows, sr+dc:sr+dc+self.ncols]
                    nb_sigma = g_sigma[sr+dr:sr+dr+chunk_rows, sr+dc:sr+dc+self.ncols]

                    diff = center - nb
                    if g_aperture is not None:
                        # Sum of squared differences over aperture pixels (NOT mean — Eq. 3 is a sum).
                        d_raw = (diff * diff * g_aperture).reshape(chunk_rows, self.ncols, -1).sum(axis=2)
                    else:
                        d_raw = (diff * diff).reshape(chunk_rows, self.ncols, -1).sum(axis=2)
                    sigma_combined = center_sigma + nb_sigma  # (chunk_rows, ncols)
                    # Guard against σ²_combined ≈ 0 (should be rare after median-fill in σ map)
                    safe_sc = cp.where(sigma_combined > cp.float32(1e-10),
                                       sigma_combined,
                                       cp.float32(1.0))
                    d_norm = (d_raw - N_p_f32 * sigma_combined) / (sqrt_2Np * safe_sc)
                    d_clipped = cp.maximum(d_norm, cp.float32(0.0))
                    w = cp.exp(-d_clipped / lam_sq) * nb_mask.astype(cp.float32)
                    # Zero out the (almost certainly impossible) degenerate σ²=0 case
                    # except for the self-pair, where center contributes with weight 1.
                    if dr != 0 or dc != 0:
                        w = cp.where(sigma_combined > cp.float32(1e-10), w, cp.float32(0.0))

                    weighted_sum += w[:, :, None, None] * nb
                    weight_total += w
                    del diff, d_raw, sigma_combined, safe_sc, d_norm, d_clipped, w, nb, nb_mask, nb_sigma

            # Normalize
            safe_wt = cp.where(weight_total > 0, weight_total, cp.ones_like(weight_total))
            result = weighted_sum / safe_wt[:, :, None, None]
            # Where weight is 0, keep center pattern
            zero_wt = (weight_total == 0)
            result[zero_wt] = center[zero_wt]

            result_np = cp.asnumpy(result).reshape(chunk_rows * self.ncols, ph, pw).astype(np.float32)
            del g_padded, g_mask, g_sigma, weighted_sum, weight_total, result, center, center_sigma
            cp.get_default_memory_pool().free_all_blocks()

            # Zero out masked patterns in output
            mask_chunk = mask[chunk_start:chunk_start+chunk_rows, :].flatten()
            result_np[~mask_chunk] = 0.0

            # Write to output
            out_obj.write_data(newpatterns=result_np, patStartCount=[chunk_start * self.ncols, -1], writeHead=False)
            del result_np

            elapsed_so_far = time.time() - t0
            rows_done = chunk_start + chunk_rows
            eta = (elapsed_so_far / rows_done) * (self.nrows - rows_done) if rows_done > 0 else 0
            self.statusBar().showMessage(
                f"NLPAR GPU: row {rows_done}/{self.nrows} ({elapsed_so_far:.0f}s, ETA {eta:.0f}s)")
            QApplication.processEvents()

        elapsed = time.time() - t0
        # Free the full-dataset σ² we kept on GPU.
        del g_sigma_full
        cp.get_default_memory_pool().free_all_blocks()
        self._temp_files.append(out_path)
        self.f_nlpar = ebsd_pattern.get_pattern_file_obj(out_path)
        self.nlpar_chk.setChecked(False)
        self.nlpar_status.setText(f"GPU NLPAR: SR={sr}, lam={lam:.2f} ({elapsed:.1f}s)")
        self.statusBar().showMessage(f"NLPAR GPU complete ({elapsed:.1f}s)")
        self.update_display()

    def _apply_nlpar_cpu(self, sr, lam):
        """CPU fallback: run pyebsdindex NLPAR straight off self.orig_path — no temp copy.

        Matches the GPU path's contract on two points that the old implementation got
        wrong because it fed NLPAR a pre-masked duplicate of the scan:

        * Dead pixels KEEP their values in the output. They are excluded from the d2
          statistics via `nlobj.mask` (pyebsdindex turns that into the `indices` array
          that gates the d2 sums) but nlpar_nb still averages every pixel of the
          neighbourhood, so a dead pixel comes out as the weighted average of its
          neighbours' dead pixels — i.e. its original value — instead of the hard 0 the
          masked temp file used to bake in.
        * automask=False. The aperture we pass IS the viewer's aperture; letting
          pyebsdindex add its circular mask on top would silently intersect the two.

        Output goes to `_cpu_nlpar_result.up1` next to the input via calcnlpar's
        `fileout=` (setoutfile -> copy_file(empty_data=True) clones the header and
        reserves the file, then each tile is written in place). Without `fileout`
        pyebsdindex names the file itself as `<stem>_NLPAR_l{lam}sr{sr}.up1`, so the
        glob below is kept as a fallback."""
        try:
            self.statusBar().showMessage(f"Running NLPAR batch CPU (SR={sr}, lam={lam:.2f})..."); QApplication.processEvents()
            t0 = time.time()
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            nlobj = nlpar.NLPAR(self.orig_path, lam=lam, searchradius=sr)
            n_avail = getattr(self, 'n_actual_rows', self.nrows)
            if n_avail < self.nrows:
                nlobj.nrows = int(n_avail)
            from_ram = self._serve_nlpar_from_ram(nlobj)
            ap = self._current_aperture()
            if ap is not None:
                nlobj.mask = (ap > 0).astype(np.uint8)
                N_p = int(nlobj.mask.sum())
                ap_label = self.aperture_mode.currentText()
            else:
                N_p = ph * pw
                ap_label = "None"
            out_path = os.path.join(os.path.dirname(self.orig_path), "_cpu_nlpar_result.up1")
            budget = self._nlpar_chunk_bytes()
            print(f"[CPU NLPAR] SR={sr} lam={lam:.3f} aperture={ap_label} "
                  f"N_p={N_p}/{ph*pw} ({100.0*N_p/(ph*pw):.1f}%) "
                  f"rows={int(n_avail)}/{self.nrows} chunk_budget={budget/1e9:.2f} GB "
                  f"source={'RAM' if from_ram else 'file'} -> {out_path}")
            result_path = nlobj.calcnlpar(chunksize=budget, saturation_protect=True,
                                          automask=False, backsub=False, fileout=out_path)
            elapsed = time.time() - t0
            if not (result_path and os.path.isfile(result_path)):
                base = os.path.splitext(self.orig_path)[0]
                candidates = sorted(glob.glob(f"{base}_NLPAR*.up1"), key=os.path.getmtime, reverse=True)
                result_path = candidates[0] if candidates else None
            if result_path and os.path.isfile(result_path):
                if result_path not in self._temp_files:
                    self._temp_files.append(result_path)
                self.f_nlpar = ebsd_pattern.get_pattern_file_obj(result_path)
                self.nlpar_chk.setChecked(False)
                self.nlpar_status.setText(f"CPU NLPAR: SR={sr}, lam={lam:.2f} ({elapsed:.1f}s)")
                self.statusBar().showMessage(f"NLPAR batch complete ({elapsed:.1f}s, N_p={N_p})")
            else:
                QMessageBox.warning(self, "Warning", "NLPAR ran but output not found.")
            self.update_display()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage("NLPAR (CPU) failed")
            QMessageBox.critical(self, "NLPAR (CPU) failed", str(e))

    def apply_and_save(self):
        """Save all processed patterns to a new UP1 file (chunked)."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        if self.nlpar_chk.isChecked() and self.f_nlpar is None:
            QMessageBox.information(self, "NLPAR not applied to full dataset",
                "Need to apply NLPAR to the full dataset before saving.\n\n"
                "Click 'Apply NLPAR to Full Dataset', wait for it to finish, then "
                "click 'Apply && Save' again.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Patterns",
            os.path.dirname(self.orig_path), "UP1 Files (*.up1)")
        if not path: return
        path = os.path.abspath(path)
        self.statusBar().showMessage(f"Processing all patterns ({'GPU' if HAS_GPU else 'CPU'})..."); QApplication.processEvents()
        t0 = time.time()
        mask = self._get_iq_mask()
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        # Determine source file object
        src_fobj = self.f_nlpar if self.f_nlpar is not None else self.f_orig
        src_label = "NLPAR" if self.f_nlpar else "Original"
        use_npar = self.npar_chk.isChecked() and self.f_nlpar is None
        use_nlpar_od = self.nlpar_chk.isChecked() and self.f_nlpar is None
        if use_npar: src_label = "NPAR"
        if use_nlpar_od: src_label = "NLPAR(on-demand)"
        if HAS_GPU:
            pat_bytes = ph * pw * 4  # float32 on GPU
            mem_per_row = 3 * self.ncols * pat_bytes  # chunk + gaussian intermediates
            vram_budget = int(cp.cuda.Device(0).mem_info[1] * 0.5)
            CHUNK_ROWS = max(2, vram_budget // mem_per_row)
            CHUNK_ROWS = min(CHUNK_ROWS, self.nrows)
        else:
            CHUNK_ROWS = self.nrows
        # VRAM is only half the story: the chunk also lives in host RAM as the native read
        # (1–2 B/px) + the float64 cast the batch function needs (8) + the float64 result (8)
        # + write-side uint8/uint16 conversion (~1) ≈ 18. Cap by whichever is tighter.
        CHUNK_ROWS = min(CHUNK_ROWS, self._host_chunk_rows(18))
        # ---- Output size: exactly the patterns the source has (2026-09-28) ----
        # A map that was stopped early has fewer patterns in the .up1 than nCols*nRows
        # in its header (OIM writes it that way). The header keeps the full grid so OIM
        # still lines the file up with its scan, but we write only the patterns that
        # exist -- full rows plus the trailing partial row -- instead of fabricating,
        # processing and writing zero patterns for every missing row (a third of the
        # work and 16 GB on a 2131x1671 scan stopped at row 1097).
        n_out = int(getattr(self, 'n_actual_pats', self.nrows * self.ncols))
        ncols = int(self.ncols)
        n_rows_out = (n_out + ncols - 1) // ncols            # includes a partial last row
        n_full_rows = int(getattr(self, 'n_actual_rows', self.nrows))
        print(f"[SAVE] CHUNK_ROWS: {CHUNK_ROWS} of {n_rows_out} rows ({n_out} patterns"
              f"{'' if n_out == self.nrows * ncols else f', header grid {ncols}x{self.nrows}'})")
        # ---- Create the file ourselves (2026-09-28) ----
        # pyebsdindex.write_data() on a path that does not exist yet pre-fills the whole
        # file with blank patterns, one 13 KB tofile() per pattern: 3.56 M writes over NFS
        # at ~28 MB/s, i.e. ~28 min of zeros before the first real chunk lands (Ni: ~19 min).
        # Copy the source header verbatim (same grid, version, steps, v4 extra bytes) and
        # reserve the final size with a sparse truncate (milliseconds); the chunk writes
        # below then fill it in place.
        ph_bytes = ph * pw * (2 if self.orig_path.lower().endswith('.up2') else 1)
        hdr_len = int(self.f_orig.filePos)
        with open(self.orig_path, 'rb') as f_src:
            hdr = f_src.read(hdr_len)
        with open(path, 'wb') as f_dst:
            f_dst.write(hdr)
            f_dst.truncate(hdr_len + n_out * ph_bytes)
        out_obj = ebsd_pattern.get_pattern_file_obj(path)   # reads the header we just wrote
        mask_flat = mask.reshape(-1)
        n_saved = 0
        for row_start in range(0, n_rows_out, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, n_rows_out - row_start)
            p0 = row_start * ncols
            p1 = min((row_start + rows_to_read) * ncols, n_out)
            if p1 <= n_full_rows * ncols:
                chunk = self._read_rows(src_fobj, row_start, rows_to_read)
            else:
                # Chunk ends in the partial row. Full rows come from the processing source;
                # the partial row's patterns come from the original file (the NLPAR batch
                # result has zeros there because pyebsdindex only processed whole rows).
                parts = []
                if p0 < n_full_rows * ncols:
                    parts.append(self._read_rows(src_fobj, row_start, n_full_rows - row_start))
                q0 = max(p0, n_full_rows * ncols)
                parts.append(self._read_patterns(self.f_orig, q0, p1 - q0))
                chunk = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
                del parts
            mask_rows = mask_flat[p0:p1]
            # Batch process entire chunk (GPU-accelerated if available); it expects float64.
            chunk = self._process_chunk_batch(chunk.astype(np.float64), mask_rows)
            n_saved += int(mask_rows.sum())
            out_obj.write_data(newpatterns=chunk, patStartCount=[p0, -1], writeHead=False)
            del chunk
            self.statusBar().showMessage(f"Saving: row {row_start + rows_to_read}/{n_rows_out}")
            QApplication.processEvents()
        elapsed = time.time() - t0
        n_total = n_out
        self.statusBar().showMessage(f"Saved to {os.path.basename(path)} ({elapsed:.1f}s)")
        QMessageBox.information(self, "Saved",
            f"Processed patterns saved to:\n{path}\n\nSource: {src_label}\nMode: {'GPU' if HAS_GPU else 'CPU'}\nPatterns: {n_saved}/{n_total}\nTime: {elapsed:.1f}s")

    def _update_iq_range(self):
        if self.iq_map is None: return
        self.iq_mask = self._get_iq_mask()
        iq_lo, iq_hi = self.iq_map.min(), self.iq_map.max()
        vmin = iq_lo + (iq_hi - iq_lo) * self.iq_min_slider.value() / 1000.0
        vmax = iq_lo + (iq_hi - iq_lo) * self.iq_max_slider.value() / 1000.0
        self.iq_min_lbl.setText(f"Min IQ: {vmin:.1f}"); self.iq_max_lbl.setText(f"Max IQ: {vmax:.1f}")
        self.scan_canvas.update_mask(self.iq_mask); self.update_display()

    def _on_normalize_toggled(self, checked):
        """Per-pattern and global stretch are mutually exclusive."""
        if checked and self.global_chk.isChecked():
            self.global_chk.blockSignals(True); self.global_chk.setChecked(False); self.global_chk.blockSignals(False)
        self._sync_bg_controls()

    def _on_global_toggled(self, checked):
        """Per-pattern and global stretch are mutually exclusive."""
        if checked and self.normalize_chk.isChecked():
            self.normalize_chk.blockSignals(True); self.normalize_chk.setChecked(False); self.normalize_chk.blockSignals(False)
        self._sync_bg_controls()

    def _sync_bg_controls(self):
        """Enforce the background/stretch interlocks by enabling/disabling controls:
          - Static and dynamic background are mutually exclusive.
          - Global stretch is coupled to the static domain, so it is unavailable
            while dynamic background is active.
        Controls that become unavailable are also UNCHECKED (not just greyed) so a
        checked-but-disabled box can't keep affecting processing. Uses blockSignals
        around state changes to avoid recursive/duplicate update_display calls."""
        dyn_on = self.dyn_bg_chk.isChecked()
        static_on = self.static_bg_chk.isChecked()
        # Dynamic active → static and global stretch are disabled (and unchecked).
        for w, should_disable in ((self.static_bg_chk, dyn_on), (self.global_chk, dyn_on)):
            if should_disable and w.isChecked():
                w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
            w.setEnabled(not should_disable)
        # Static (or global, which implies static domain) active → dynamic disabled.
        block_dyn = static_on or self.global_chk.isChecked()
        if block_dyn and self.dyn_bg_chk.isChecked():
            self.dyn_bg_chk.blockSignals(True); self.dyn_bg_chk.setChecked(False); self.dyn_bg_chk.blockSignals(False)
        self.dyn_bg_chk.setEnabled(not block_dyn)
        # Mean/range domain may have changed → invalidate cached global range.
        self._invalidate_global_range()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_lay = QHBoxLayout(central)
        ctrl = QWidget(); ctrl.setFixedWidth(310); cl = QVBoxLayout(ctrl)
        # Wrap ctrl in a scroll area so the window's minimum height isn't tied to
        # the sum of every control's minimum size (otherwise this exceeds 1440p displays).
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(330)  # 310 content + ~20 for vertical scrollbar
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ctrl_scroll.setFrameShape(0)  # no frame border (QFrame.NoFrame == 0)

        # File
        g = QGroupBox("File"); gl = QVBoxLayout(g)
        self.file_label = QLabel("No file loaded"); self.file_label.setWordWrap(True)
        btn = QPushButton("Load UP1"); btn.clicked.connect(self.load_file)
        gl.addWidget(self.file_label); gl.addWidget(btn); cl.addWidget(g)

        # Aperture mask — GLOBAL gate, applied before every other operation.
        # None / Circular (EBSD) / Auto-detect (variance-based, e.g. RKD quad-chip).
        # When active, dead pixels are zeroed on read and re-zeroed after the last
        # operation, and every statistic that would otherwise be biased by that field
        # of zeros (per-pattern min/max, global percentiles, NLPAR N_p and σ²) is
        # computed over active pixels only.
        g = QGroupBox("Aperture Mask (applied before all processing)"); gl = QFormLayout(g)
        self.aperture_mode = QComboBox()
        self.aperture_mode.addItems(["None", "Circular (EBSD)", "Auto-detect"])
        self.aperture_mode.setCurrentIndex(0)  # default None
        self.aperture_mode.setMinimumContentsLength(15)  # ensure "Circular (EBSD)" fits without ellipsis
        self.aperture_mode.currentIndexChanged.connect(lambda *_: self._on_aperture_changed())
        gl.addRow("Mode:", self.aperture_mode)
        self.aperture_thresh_spin = QDoubleSpinBox()
        self.aperture_thresh_spin.setRange(0.001, 1.0)
        self.aperture_thresh_spin.setValue(0.05)
        self.aperture_thresh_spin.setSingleStep(0.01)
        self.aperture_thresh_spin.setDecimals(3)
        self.aperture_thresh_spin.setToolTip("Auto-detect threshold as fraction of 95th-percentile pixel std")
        self.aperture_thresh_spin.valueChanged.connect(lambda *_: self._on_aperture_changed())
        gl.addRow("Threshold:", self.aperture_thresh_spin)
        b_show_mask = QPushButton("Show detected mask")
        b_show_mask.clicked.connect(self.show_aperture_mask)
        gl.addRow(b_show_mask)
        cl.addWidget(g)

        # IQ Range
        g = QGroupBox("IQ Range (excludes patterns outside range)"); gl = QVBoxLayout(g)
        self.iq_min_lbl = QLabel("Min IQ: 0"); self.iq_max_lbl = QLabel("Max IQ: 100")
        self.iq_min_slider = QSlider(Qt.Horizontal); self.iq_min_slider.setRange(0, 1000); self.iq_min_slider.setValue(0)
        self.iq_max_slider = QSlider(Qt.Horizontal); self.iq_max_slider.setRange(0, 1000); self.iq_max_slider.setValue(1000)
        self.iq_min_slider.valueChanged.connect(self._update_iq_range)
        self.iq_max_slider.valueChanged.connect(self._update_iq_range)
        gl.addWidget(self.iq_min_lbl); gl.addWidget(self.iq_min_slider)
        gl.addWidget(self.iq_max_lbl); gl.addWidget(self.iq_max_slider); cl.addWidget(g)

        # NPAR - on-demand via checkbox
        g = QGroupBox("NPAR"); gl = QFormLayout(g)
        self.npar_chk = QCheckBox("Enable"); self.npar_chk.stateChanged.connect(self.update_display)
        self.npar_radius_spin = QSpinBox(); self.npar_radius_spin.setRange(1, 10); self.npar_radius_spin.setValue(1)
        self.npar_radius_spin.valueChanged.connect(self.update_display)
        gl.addRow(self.npar_chk); gl.addRow("Radius:", self.npar_radius_spin); cl.addWidget(g)

        # NLPAR - on-demand preview + batch operations
        g = QGroupBox("NLPAR"); gl = QFormLayout(g)
        self.nlpar_chk = QCheckBox("Enable Preview"); self.nlpar_chk.stateChanged.connect(self.update_display)
        self.sr_spin = QSpinBox(); self.sr_spin.setRange(1, 10); self.sr_spin.setValue(3)
        self.sr_spin.valueChanged.connect(self.update_display)
        self.lam_spin = QDoubleSpinBox(); self.lam_spin.setRange(0.01, 100.0)
        # Canonical NLPAR lambda is dimensionless (post σ² normalization).
        # Optimal values per Brewick et al. 2019 are typically 0.6–1.2.
        self.lam_spin.setValue(1.0); self.lam_spin.setSingleStep(0.05); self.lam_spin.setDecimals(2)
        self.lam_spin.valueChanged.connect(self.update_display)
        gl.addRow(self.nlpar_chk)
        gl.addRow("Search Radius:", self.sr_spin); gl.addRow("Lambda:", self.lam_spin)
        b1 = QPushButton("Optimize Lambda"); b1.clicked.connect(self.optimize_lambda)
        b2 = QPushButton("Apply NLPAR to Full Dataset (GPU)" if HAS_GPU else "Apply NLPAR to Full Dataset (CPU)"); b2.clicked.connect(self.apply_nlpar_batch)
        b3 = QPushButton("Load Existing NLPAR UP1"); b3.clicked.connect(self.load_nlpar_file)
        gl.addRow(b1); gl.addRow(b2); gl.addRow(b3)
        self.nlpar_status = QLabel("No NLPAR file loaded"); gl.addRow(self.nlpar_status); cl.addWidget(g)

        # Static BG
        g = QGroupBox("Static Background"); gl = QVBoxLayout(g)
        self.static_bg_chk = QCheckBox("Enable"); self.static_bg_chk.stateChanged.connect(self.update_display)
        self.static_bg_chk.toggled.connect(lambda *_: self._sync_bg_controls())
        self.bg_mode = QComboBox(); self.bg_mode.addItems(["Subtract", "Divide"])
        self.bg_mode.currentIndexChanged.connect(self.update_display)
        b_recomp = QPushButton("Recompute Background (from IQ range)")
        b_recomp.clicked.connect(self.recompute_background)
        self.bg_status = QLabel("Background: all patterns")
        gl.addWidget(self.static_bg_chk); gl.addWidget(QLabel("Mode:")); gl.addWidget(self.bg_mode)
        gl.addWidget(b_recomp); gl.addWidget(self.bg_status); cl.addWidget(g)

        # Dynamic BG
        g = QGroupBox("Dynamic Background"); gl = QVBoxLayout(g)
        self.dyn_bg_chk = QCheckBox("Enable"); self.dyn_bg_chk.stateChanged.connect(self.update_display)
        self.dyn_bg_chk.toggled.connect(lambda *_: self._sync_bg_controls())
        self.dyn_sigma_lbl = QLabel("Sigma: 5.0")
        self.dyn_sigma = QSlider(Qt.Horizontal); self.dyn_sigma.setRange(1, 200); self.dyn_sigma.setValue(50)
        self.dyn_sigma.valueChanged.connect(self.update_display)
        self.dyn_sigma.valueChanged.connect(lambda v: self.dyn_sigma_lbl.setText(f"Sigma: {v/10:.1f}"))
        gl.addWidget(self.dyn_bg_chk); gl.addWidget(self.dyn_sigma_lbl); gl.addWidget(self.dyn_sigma); cl.addWidget(g)

        # Normalization — per-pattern min-max stretch to [0,1].
        # When ON  (default): every pattern is independently stretched to fill the
        #   display range. Equalizes the visual appearance across the scan but
        #   hides absolute intensity differences (e.g. detector gain drift).
        # When OFF: absolute uint8 intensity is preserved end-to-end (preview AND
        #   exported UP1). Dim patterns look dim. Use this if you want the saved
        #   patterns to reflect the actual signal level recorded by the detector.
        g = QGroupBox("Normalization"); gl = QVBoxLayout(g)
        self.normalize_chk = QCheckBox("Per-pattern stretch to [0,1]")
        self.normalize_chk.setChecked(True)
        self.normalize_chk.setToolTip(
            "ON: each pattern independently stretched to fill the display range.\n"
            "OFF: absolute uint8 intensity preserved in preview AND exported UP1.\n"
            "Turn OFF to see real intensity differences across the scan (e.g. detector gain steps)."
        )
        self.normalize_chk.stateChanged.connect(self.update_display)
        gl.addWidget(self.normalize_chk)
        # Global stretch — dataset-wide percentile stretch (mutually exclusive with per-pattern).
        # Same (lo,hi) applied to every pattern, from the 0.5/99.5 percentiles over
        # ACTIVE pixels (dead detector corners excluded). Unlike per-pattern stretch,
        # this preserves RELATIVE intensity across patterns → safe for downstream IQ
        # (OIM/EMsoft) since it does not reorder pattern-to-pattern band contrast.
        # Carries into both the preview and the exported UP1.
        self.global_chk = QCheckBox("Global stretch (dataset 0.5–99.5 pctile)")
        self.global_chk.setChecked(False)
        self.global_chk.setToolTip(
            "Dataset-wide stretch: one (lo,hi) for ALL patterns, from the 0.5/99.5\n"
            "percentiles over active pixels (dead corners excluded).\n"
            "Preserves relative intensity across patterns (IQ-safe). Mutually\n"
            "exclusive with per-pattern stretch. Applies to preview AND saved UP1."
        )
        self.normalize_chk.toggled.connect(self._on_normalize_toggled)
        self.global_chk.toggled.connect(self._on_global_toggled)
        self.global_chk.stateChanged.connect(self.update_display)
        gl.addWidget(self.global_chk); cl.addWidget(g)

        # Brightness / Contrast
        g = QGroupBox("Brightness / Contrast"); gl = QVBoxLayout(g)
        self.bright_lbl = QLabel("Brightness: 0")
        self.bright = QSlider(Qt.Horizontal); self.bright.setRange(-100, 100); self.bright.setValue(0)
        self.bright.valueChanged.connect(self.update_display)
        self.bright.valueChanged.connect(lambda v: self.bright_lbl.setText(f"Brightness: {v}"))
        self.cont_lbl = QLabel("Contrast: 0")
        self.cont = QSlider(Qt.Horizontal); self.cont.setRange(-100, 100); self.cont.setValue(0)
        self.cont.valueChanged.connect(self.update_display)
        self.cont.valueChanged.connect(lambda v: self.cont_lbl.setText(f"Contrast: {v}"))
        gl.addWidget(self.bright_lbl); gl.addWidget(self.bright)
        gl.addWidget(self.cont_lbl); gl.addWidget(self.cont); cl.addWidget(g)

        # CLAHE
        g = QGroupBox("CLAHE"); gl = QVBoxLayout(g)
        self.clahe_chk = QCheckBox("Enable"); self.clahe_chk.stateChanged.connect(self.update_display)
        self.clahe_clip_lbl = QLabel("Clip Limit: 0.010")
        self.clahe_clip = QSlider(Qt.Horizontal); self.clahe_clip.setRange(1, 50); self.clahe_clip.setValue(10)
        self.clahe_clip.valueChanged.connect(self.update_display)
        self.clahe_clip.valueChanged.connect(lambda v: self.clahe_clip_lbl.setText(f"Clip Limit: {v/1000:.3f}"))
        gl.addWidget(self.clahe_chk); gl.addWidget(self.clahe_clip_lbl); gl.addWidget(self.clahe_clip); cl.addWidget(g)

        cl.addStretch()
        # Apply & Save button
        self.apply_save_btn = QPushButton("Apply && Save Processed Patterns")
        self.apply_save_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.apply_save_btn.clicked.connect(self.apply_and_save)
        cl.addWidget(self.apply_save_btn)

        # ---- Right: splitter ----
        splitter = QSplitter(Qt.Vertical)
        top_widget = QWidget(); top_lay = QVBoxLayout(top_widget)
        top_lay.setContentsMargins(0, 0, 0, 0); top_lay.setSpacing(2)
        zoom_row = QWidget(); zl = QHBoxLayout(zoom_row); zl.setContentsMargins(0, 0, 0, 0)
        btn_zrst = QPushButton("Reset Zoom"); zl.addWidget(btn_zrst); zl.addStretch()
        top_lay.addWidget(zoom_row)
        self.scan_canvas = ScanMapCanvas()
        self.scan_canvas.pattern_clicked.connect(self.on_pattern_clicked)
        btn_zrst.clicked.connect(self.scan_canvas.reset_zoom)
        top_lay.addWidget(self.scan_canvas, stretch=1)
        bot_widget = QWidget(); bot_lay = QHBoxLayout(bot_widget); bot_lay.setContentsMargins(0, 0, 0, 0)
        self.raw_canvas = PatternCanvas("Raw"); self.proc_canvas = PatternCanvas("Processed")
        bot_lay.addWidget(self.raw_canvas); bot_lay.addWidget(self.proc_canvas)
        splitter.addWidget(top_widget); splitter.addWidget(bot_widget)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setSizes([450, 450])
        main_lay.addWidget(ctrl_scroll); main_lay.addWidget(splitter, stretch=1)


if __name__ == '__main__':
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec_())
