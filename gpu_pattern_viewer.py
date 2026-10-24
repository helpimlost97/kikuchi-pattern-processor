__version__ = "1.7.0.dev2"   # shown in the window title; bump with each release (git tag v<version>)
import sys
import os
import gc
import glob

# ---------------------------------------------------------------------------
# Linux / HPC CPU-node support (2026-09-25)
# On a SLURM node the process only owns the cores/RAM it asked for, not the
# whole machine (e.g. a 104-core, 500 GB node). os.sched_getaffinity sees the
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
    QStatusBar, QSplitter, QSizePolicy, QScrollArea, QLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyebsdindex import ebsd_pattern
# pyebsdindex picks its NLPAR class at import: the OpenCL (pyopencl) version when a GPU is
# visible, else the numba CPU version. KPP_GPU=0 must force CPU here too: it only hides
# CUDA from CuPy (below), OpenCL would still find the device. PLAN_NLPAR_PYEBSDINDEX.md, Phase 0.
if os.environ.get('KPP_GPU', '').strip() == '0':
    from pyebsdindex import nlpar_cpu as nlpar
else:
    from pyebsdindex import nlpar
# Which engine pyebsdindex resolved to. The NLPAR batch and Optimize Lambda both run
# through it (Phase 1).
NLPAR_ON_GPU = nlpar.NLPAR.__module__.startswith('pyebsdindex.opencl')
NLPAR_ENGINE = "pyebsdindex OpenCL" if NLPAR_ON_GPU else "pyebsdindex CPU (numba)"

def _patch_pyebsdindex_nlpar_kernel():
    """Work around a bug in pyebsdindex's OpenCL NLPAR kernel (clnlpar.cl, kernel
    calcnlpar, 'calculate the weights' loop; present in 0.3.9.x and upstream main on
    2026-10-13): it reads `sigma[indx_xy]` (the centre pattern's sigma) for every
    neighbour where `sigma[indx_ij]` (the neighbour's sigma) is meant, so the weight's
    (sigma_i^2 + sigma_j^2) becomes 2 sigma_i^2. The CPU (numba) kernel is right. On a
    two-noise-level synthetic scan the shipped GPU result differs from the CPU result by
    up to 119 intensity levels along the noise boundary; with this one-line fix the two
    agree exactly in the scan interior (borders still differ: the CPU shifts the window
    inward, the GPU mirrors indices).

    openclparam.get_context builds `path.join(<package dir>, kfile)`, so handing it an
    absolute kfile path redirects the build to a patched copy. The copy is regenerated
    from the installed file at every start; once upstream fixes the line the replace is
    a no-op and nothing is redirected. Reported upstream: see PLAN_NLPAR_PYEBSDINDEX.md."""
    if not NLPAR_ON_GPU:
        return None, None
    try:
        import hashlib, tempfile
        from pyebsdindex.opencl import openclparam
        src = os.path.join(os.path.dirname(openclparam.__file__), 'clnlpar.cl')
        text = open(src, encoding='utf-8').read()
        sha = hashlib.sha1(text.encode('utf-8')).hexdigest()
        full = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patches', 'clnlpar_fixed.cl')
        # Per-user cache dir: the system temp dir is shared on a cluster node and a
        # directory created there by another user is not writable (code review 2026-10-13).
        import getpass
        try: user = getpass.getuser()
        except Exception: user = 'user'
        kdir = os.path.join(tempfile.gettempdir(), f'kikuchi_pattern_processor_{user}')
        if sha in _KNOWN_SHIPPED_CLNLPAR and os.path.isfile(full):
            dst, level = full, 'full'            # known 0.3.9.x file: use our complete fixed copy
        else:
            # Unknown upstream version: derive the full fix from the installed text with the
            # same line-level transform that produced patches/clnlpar_fixed.cl. Every anchor
            # must hit exactly once, otherwise only the sigma line is fixed.
            fixed, level = _transform_clnlpar(text)
            if fixed is None:
                return None, None                # fixed upstream (or changed beyond recognition): leave it
            tag = hashlib.sha1(fixed.encode()).hexdigest()[:12]
            os.makedirs(kdir, exist_ok=True)
            dst = os.path.join(kdir, f'clnlpar_fixed_{level}_{tag}.cl')
            if not os.path.exists(dst):
                with open(dst, 'w', encoding='utf-8') as f: f.write(fixed)
            print(f"[NLPAR] pyebsdindex clnlpar.cl sha1 {sha[:12]} is not the version the shipped patch was made "
                  f"from; derived patch level: {level}"
                  + ("" if level == 'full' else "; pattern sizes with (height*width) % 16 != 0 will run NLPAR on the CPU instead"))
        orig = openclparam.OpenClParam.get_context
        def get_context(self, gpu_id=None, kfile='clkernels.cl'):
            if kfile == 'clnlpar.cl':
                kfile = dst
            return orig(self, gpu_id=gpu_id, kfile=kfile)
        openclparam.OpenClParam.get_context = get_context
        return dst, level
    except Exception as _e:
        print(f"[NLPAR] could not patch pyebsdindex's OpenCL kernel ({_e}); using it as shipped")
        return None, None

def _transform_clnlpar(text):
    """Apply the kernel fixes to the text of an arbitrary pyebsdindex clnlpar.cl.
    Returns (fixed_text, 'full') when every anchor of the alignment/tail fix matched exactly
    once, (fixed_text, 'sigma-only') when only the sigma line could be fixed, (None, None)
    when not even that line exists. Same transform that produced patches/clnlpar_fixed.cl."""
    import re
    bad, good = "sigma_ij = sigma[indx_xy];", "sigma_ij = sigma[indx_ij];"
    if text.count(bad) != 1:
        return None, None
    s = text.replace(bad, good)
    helpers = (
        "\n/* Patched at runtime by the Kikuchi Pattern Processor: vload16/vstore16 instead of\n"
        "   float16 pointer casts, element-wise partial last vector, register accumulation.\n"
        "   See patches/clnlpar_fixed.cl in that repo for the annotated version. */\n"
        "inline float16 loadpat16(const __global float *p, const long nvalid){\n"
        "  if (nvalid >= 16) { return vload16(0, p); }\n"
        "  float tmp[16];\n"
        "  for (int q = 0; q < 16; ++q) { tmp[q] = (q < nvalid) ? p[q] : 0.0f; }\n"
        "  return vload16(0, tmp);\n"
        "}\n"
        "inline void storepat16(__global float *p, const float16 v, const long nvalid){\n"
        "  if (nvalid >= 16) { vstore16(v, 0, p); return; }\n"
        "  float tmp[16];\n"
        "  vstore16(v, 0, tmp);\n"
        "  for (int q = 0; q < nvalid; ++q) { p[q] = tmp[q]; }\n"
        "}\n\n__kernel void nlloadpat8bit(")
    ok = (s.count('__kernel void nlloadpat8bit(') == 1
          and s.count('*(__global float16*) (data + indx0+16*z)') == 3
          and s.count('*(__global float16*) (data + indx_ij+16*z)') == 3
          and s.count('*(__global float16*) (dataout + indx0+16*z) += d1;') == 1)
    if not ok:
        return s, 'sigma-only'
    t = s.replace('__kernel void nlloadpat8bit(', helpers, 1)
    t = t.replace('*(__global float16*) (data + indx0+16*z)', 'loadpat16(data + indx0+16*z, npatpoint - 16*z)')
    t = t.replace('*(__global float16*) (data + indx_ij+16*z)', 'loadpat16(data + indx_ij+16*z, npatpoint - 16*z)')
    t = t.replace('*(__global float16*) (dataout + indx0+16*z) += d1;', 'acc += d1;')
    m = re.search(r'\n(\s*)mask0 = mask\[z\];\s*\n\s*mask0 =  \(mask0 > \(float16\) -1\.e-3\)', t)
    if not m or t.count('acc += d1;') != 1:
        return s, 'sigma-only'
    t = t[:m.start()] + '\n' + m.group(1) + 'float16 acc = (float16) (0.0f);' + t[m.start():]
    m2 = re.compile(r'count \+= 1;[ \t]*\n([ \t]*)\}[ \t]*\n([ \t]*)\}[ \t]*\n([ \t]*)\}').search(t, t.index('acc += d1;'))
    if not m2:
        return s, 'sigma-only'
    pos = m2.start(3)
    t = t[:pos] + m2.group(2) + 'storepat16(dataout + indx0+16*z, acc, npatpoint - 16*z);\n' + t[pos:]
    if '*(__global float16*)' in t:
        return s, 'sigma-only'
    return t, 'full'

# sha1 of pyebsdindex/opencl/clnlpar.cl as shipped in releases our full patch was made from.
_KNOWN_SHIPPED_CLNLPAR = {
    '14c0d609073680260f57d0d48aa71fcb29e18f3f',   # 0.3.9.1 (pip, 2026-10)
}
if os.environ.get('KPP_NLPAR_KERNEL_PATCH', '').strip() == '0':
    NLPAR_KERNEL_PATCH, NLPAR_KERNEL_PATCH_LEVEL = None, None
else:
    NLPAR_KERNEL_PATCH, NLPAR_KERNEL_PATCH_LEVEL = _patch_pyebsdindex_nlpar_kernel()

def _gpu_kernel_safe_for(ph, pw):
    """The shipped OpenCL kernels fault (INVALID_COMMAND_QUEUE) or corrupt data when the
    pixel count is not a multiple of 16 (float16 pointer casts + an unguarded tail vector);
    our full patch fixes that. Without it, such datasets must use the CPU engine."""
    return (not NLPAR_ON_GPU) or NLPAR_KERNEL_PATCH_LEVEL == 'full' or ((int(ph) * int(pw)) % 16 == 0)
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Console prints use σ/λ/—; when stdout is a pipe/file on Windows it defaults to
# cp1252 and a bare print would raise UnicodeEncodeError mid-run. Degrade to '?'.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

# GPU acceleration via CuPy (falls back to CPU if unavailable)
# CuPy importing is not enough: on a login node / CPU-only compute node the wheel imports
# fine but there is no driver or device, and the first kernel launch would fail deep
# inside a pass. So HAS_GPU also requires a visible device, and KPP_GPU=0 forces CPU.
_gpu_why = None
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    try:
        HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        HAS_GPU = False
    if not HAS_GPU: _gpu_why = "CuPy imported but no CUDA device/driver visible"
except Exception:
    HAS_GPU = False
    _gpu_why = "CuPy not available"
if HAS_GPU and os.environ.get('KPP_GPU', '').strip() == '0':
    HAS_GPU = False
    _gpu_why = "KPP_GPU=0"
if HAS_GPU:
    try:
        _gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        _gpu_name = _gpu_props['name']
        if isinstance(_gpu_name, bytes): _gpu_name = _gpu_name.decode()
        print(f"[GPU] {_gpu_name}, {_gpu_props['totalGlobalMem']/1e9:.1f} GB VRAM")
    except Exception as _e:          # device counted but not queryable: don't trust it
        HAS_GPU = False
        _gpu_why = f"CUDA device not usable ({_e})"
if not HAS_GPU:
    print(f"[GPU] CPU mode ({_gpu_why})")
print(f"[NLPAR] engine: {NLPAR_ENGINE}"
      + (f"; OpenCL kernel patch: {NLPAR_KERNEL_PATCH_LEVEL} ({os.path.basename(NLPAR_KERNEL_PATCH)})" if NLPAR_KERNEL_PATCH
         else ("; kernel used as shipped" if NLPAR_ON_GPU else "")))

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
    it row by row through np.fromfile in ONE stream. Measured on a 104-core HPC node (Ni RKD scan
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
        self.exclude = None                  # optional (nPatterns,) bool: serve these as constant patterns

    def _apply_exclude(self, pats, idx):
        """Phase 3 IQ-range exclusion without touching pyebsdindex: excluded patterns are
        served as a constant pattern of value 1. Together with sigma = 0 at those positions
        (set by the batch) their normalised distance to any real pattern is enormous, so
        their weight is exp(-huge) = 0: they are dropped as neighbours. Their own output is
        that constant pattern, which the batch replaces by the original, unprocessed pattern
        (_restore_excluded_output) so that they stay in the data set. Value 1 rather than 0:
        pyebsdindex's calcnlpar_cl treats an all-zero output pattern as the Apple-silicon
        GPU failure and recomputes the tile up to three times (code review 2026-10-13).
        See PLAN_NLPAR_PYEBSDINDEX.md, Phase 3."""
        if self.exclude is not None:
            sel = self.exclude[idx]
            if sel.any():
                pats[sel] = 1
        return pats

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
            n_ram = self._ram.shape[0]
            full_rows = n_ram // ncols          # a map stopped early has fewer rows than the header
            if r0 + nr <= full_rows:
                grid = self._ram[:full_rows * ncols].reshape(full_rows, ncols, ph, pw)   # view
                pats = grid[r0:r0 + nr, c0:c0 + nc].reshape(nr * nc, ph, pw).astype(typeout)
            else:                            # touches the partial row / missing rows: zeros there
                pats = np.zeros((nr * nc, ph, pw), dtype=typeout)
                for i in range(nr):
                    p0 = (r0 + i) * ncols + c0; p1 = min(p0 + nc, n_ram)
                    if p1 > p0: pats[i * nc:i * nc + (p1 - p0)] = self._ram[p0:p1]
            idx = (np.arange(r0, r0 + nr, dtype=np.int64)[:, None] * ncols
                   + np.arange(c0, c0 + nc, dtype=np.int64)[None, :]).ravel()
        return self._apply_exclude(pats, idx), self._xyloc(idx)


class MaskedUPFile(ebsd_pattern.UPFile):
    """Streamed-mode twin of RamUPFile's exclusion: a real pyebsdindex UPFile whose
    returnArrayOnly reads serve the patterns flagged in `exclude` as constant patterns
    (same contract as RamUPFile._apply_exclude). Everything else is pyebsdindex's own file code."""
    def __init__(self, path):
        super().__init__(path)
        self.read_header()
        self.exclude = None

    def read_data(self, path=None, convertToFloat=False, patStartCount=[0, -1], returnArrayOnly=False):
        out = super().read_data(path=path, convertToFloat=convertToFloat,
                                patStartCount=patStartCount, returnArrayOnly=returnArrayOnly)
        if not returnArrayOnly or self.exclude is None:
            return out
        psc = np.asarray(patStartCount, dtype=np.int64)
        if psc.ndim != 1:
            # Slab read: UPFile.read_data assembles it from one self.read_data call per
            # row, each of which already passed through this branch. Nothing left to do.
            return out
        pats, xyloc = out
        p0 = int(psc[0]); idx = np.arange(p0, p0 + pats.shape[0], dtype=np.int64)
        sel = self.exclude[idx]
        if sel.any():
            pats[sel] = 1                        # fresh array from pat_reader: in place is safe
        return pats, xyloc


class RamOutUPFile(RamUPFile):
    """Output-side twin: pyebsdindex's NLPAR writes its result through
    getoutfileobj().write_data() -> pat_writer(); here the patterns land in the RAM
    array instead of a file. The int conversion (pat_flt2int, 'clip') still happens
    in write_data, so the values are exactly what the file would have held. The
    file on disk (self.filepath = the input) is never touched."""
    def pat_writer(self, pat2write, patStart, nPatToWrite, typewrite):
        p0, n = int(patStart), int(nPatToWrite)
        n = min(n, self._ram.shape[0] - p0)
        if n > 0: self._ram[p0:p0 + n] = pat2write[:n].astype(self._ram.dtype, copy=False)
    def write_header(self, *a, **k): return 0
    def write_datamaxmin(self): pass


class BatchCancelled(Exception):
    """Raised by MainWindow.save_processed / batch_process when the caller's `cancel()`
    returned True. The partial output file of the interrupted save has been deleted."""


class _ControlPanel(QWidget):
    """The control column: never asks for more width than the splitter gives it."""
    def minimumSizeHint(self):
        hint = super().minimumSizeHint(); hint.setWidth(200); return hint


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Kikuchi Pattern Processor v{__version__} " + ("(GPU Accelerated)" if HAS_GPU else f"(CPU Mode, {n_workers()} threads)")); self.setGeometry(100, 100, 1400, 900)
        self.orig_path = None; self.f_orig = self.f_nlpar = None
        self.nlpar_patterns = None        # in-RAM NLPAR batch result (v1.3.0)
        # Bit depth (v1.5.0, UP2 support). pyebsdindex takes the pattern dtype from the
        # file extension (.up1 = uint8, .up2 = uint16), so every output file must keep the
        # source extension or it is read back with the wrong dtype.
        #   src_ext   : '.up1' or '.up2' of the loaded file
        #   dtype_max : largest value the file dtype can hold (255 / 65535); the
        #               normalized ([0,1]-domain) export is scaled to this
        #   scale_max : the intensity full scale the DATA actually uses. uint8 files are
        #               always 255 (unchanged behaviour). A uint16 container often holds
        #               12/14-bit camera data, so it is the smallest 2^k-1 >= the dataset
        #               max (4095, 16383, 65535, ...). Raw/absolute preview range and the
        #               brightness/contrast midpoint of the unnormalized path use this.
        self.src_ext = '.up1'
        self.dtype_max = 255
        self.scale_max = 255.0
        self.data_max = None
        # Tier 1 (2026-09-27): the raw dataset held in RAM as one (npat, ph, pw) array in
        # the file's native dtype, when it fits. Every whole-dataset pass then slices this
        # instead of re-streaming the file (each pass was a full 32 GB read from home NFS).
        self.patterns = None
        # Workflow gates (2026-09-27, for students using the app unassisted):
        # Optimize lambda -> Apply NLPAR to full dataset -> Apply && Save. Set when
        # optimize_lambda succeeds; cleared on file load and aperture change (lambda is
        # fitted for one pixel mask). Editing the lambda box by hand does NOT clear it.
        self._lambda_optimized = False
        # pyebsdindex NLPAR object shared by Optimize Lambda and the batch (sigma reuse);
        # keyed on (path, aperture, rows). See _get_nlobj.
        self._nlobj = None; self._nlobj_key = None; self._nlobj_from_ram = False
        self._nl_infile = None               # the object's input file (RamUPFile / MaskedUPFile)
        # True while a pyebsdindex call is in flight. The progress hook pumps the Qt event
        # loop, so button clicks and control changes arrive mid-call; the entry points
        # refuse while busy instead of re-entering on the same object (code review 2026-10-13).
        self._nlpar_busy = False
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
        # Montage batch (v1.7.0): the mean pattern over all tiles from
        # compute_shared_background (with the tuple of paths it was computed from), and the
        # per-job result dicts of the last batch_process call.
        self.shared_background = None
        self.shared_background_key = None
        self.batch_results = []
        self._batch_running = False
        self._build_ui()

    def _remove_temp_files(self):
        """Delete the NLPAR batch result files this session created (next to their
        inputs). Called on app exit and when a new dataset is loaded (2026-09-27),
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
        """Circular aperture = pyebsdindex's own NLPAR.makeautomask (inscribed circle of
        radius 0.98*min(h,w)/2). Was a verbatim copy until Phase 2 (PLAN_NLPAR_PYEBSDINDEX.md);
        calling theirs keeps the two in step. Disable for non-circular geometries (RKD)."""
        return np.asarray(nlpar.NLPAR.makeautomask(int(h), int(w)), dtype=np.uint8)

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
        if self._nlpar_busy:
            self.statusBar().showMessage("NLPAR is running: aperture change ignored until it finishes")
            return
        self._invalidate_sigma_map()
        self._invalidate_global_range()
        self._lambda_optimized = False  # lambda was fitted for the previous aperture
        self._nlobj = None; self._nlobj_key = None   # its sigma was computed under the old mask
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

    @staticmethod
    def _host_chunk_budget():
        """Default host-RAM byte budget for one chunk of a whole-dataset pass.
        Was min(3 GB, 25% of available). On a 500 GB HPC node that meant 1-row chunks
        (a 430-col 479x479 row is ~1.8 GB at 18 B/px), starving the threaded paths.
        Scale the cap with the thread count, but never exceed a quarter of what the
        SLURM cgroup actually lets us allocate. Split out of _host_chunk_rows (v1.7.0) so
        compute_shared_background can size its chunks for files that are not loaded."""
        return min(1.0e9 * max(4, n_workers() // 2), 0.25 * available_ram())

    def _host_chunk_rows(self, bytes_per_pixel_live, budget=None, n_rows_total=None):
        """Rows per chunk that keep host RAM for one chunk inside `budget` bytes.

        `bytes_per_pixel_live` is the SUM of bytes/pixel over every copy of the chunk that
        is alive at the same time (e.g. uint8 read + float32 cast + a float32 temporary = 9).
        Hardcoded row counts blew up on large scans: 430 cols x 479x479 px is ~99 MB/row as
        uint8 but ~790 MB/row as float64, so 50 rows of float64 is ~40 GB. Default budget is
        the smaller of 3 GB and a quarter of the currently AVAILABLE RAM, so a busy machine
        chunks smaller instead of swapping (or dying with _ArrayMemoryError)."""
        if budget is None:
            budget = self._host_chunk_budget()
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

    def _static_background_term(self, preserve_mean, ap_bool=None):
        """The static background the pipeline subtracts or divides by.

        In the absolute-intensity mode (per-pattern stretch and global stretch both off) the
        raw mean pattern must not be used as is: subtracting it centres every pattern on zero
        and dividing by it lands every pattern near 1, so the final clip to [0, scale_max]
        leaves a black pattern with a few bright specks, in the preview and in the saved file.
        Following pyebsdindex's backsub (which does `backfit -= mean(backfit)` before
        `data -= backfit`), the background is made mean-free for "Subtract" and mean-one for
        "Divide": the shape of the background (bright centre, gradient) is removed while every
        pattern keeps its own mean brightness, so relative intensity across the scan survives.
        The mean level is taken over the active (aperture) pixels when an aperture is set.
        In the stretched modes the per-pattern / global stretch removes any offset, so the
        raw mean pattern is used there as before (2026-10-19)."""
        mp = self.mean_pattern
        if not preserve_mean:
            return mp
        level = float(mp[ap_bool].mean()) if ap_bool is not None else float(mp.mean())
        if self.bg_mode.currentText() == "Subtract":
            return mp - level
        return mp / max(level, 1e-10)

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
            self.global_lo, self.global_hi = 0.0, float(self.scale_max)
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
            QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first")
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

        Kept as our own NumPy code by design decision (2026-10-13; PLAN_NLPAR_PYEBSDINDEX.md):
        it is the fast interactive preview. Two things are aligned with pyebsdindex's batch
        kernel so interior pixels come out the same (Phase 2):
          * σ² comes from pyebsdindex's calcsigma (see _ensure_sigma_map);
          * saturation_protect: a pixel pair where either value is at/above the saturation
            threshold is dropped from d, and N_p becomes the per-pair count n of surviving
            aperture pixels. pyebsdindex takes the threshold from the max of the tile it is
            processing (numba: 0.999*max, OpenCL: 0.9961*max); here the dataset max stands
            in for the tile max, which is the same number for any scan with a saturated
            pixel somewhere.
        Scan borders stay ours: missing neighbours are simply absent (zero-pad), where
        pyebsdindex's numba kernel shifts the window inward and its OpenCL kernel mirrors.
        """
        sr = self.sr_spin.value()
        lam = self.lam_spin.value()
        # Ensure sigma² map is available for the current aperture.
        self._ensure_sigma_map()
        pats, mask, cr, cc = self._read_neighborhood(self.f_orig, row, col, sr)
        center_pat = pats[cr, cc].astype(np.float64)
        if self.sigma_sq_map is None:
            return center_pat                    # sigma unavailable while a batch is running
        on_gpu = self._nlpar_engine_of(self._nlobj)[0] if self._nlobj is not None else NLPAR_ON_GPU
        ph, pw = center_pat.shape
        nr, nc = pats.shape[0], pats.shape[1]

        # Aperture mask (canonical paper notation N_p = number of active pixels).
        aperture_2d, aperture_count, _ = self._resolve_aperture_mask(ph, pw)
        ap_flat = (aperture_2d.ravel() > 0) if aperture_2d is not None else np.ones(ph * pw, dtype=bool)
        # pyebsdindex saturation rule, engine-specific factor on the tile max (here: data max);
        # off -> threshold above any value (pyebsdindex uses max+1). dthresh as in the kernels.
        params = self._nlpar_params()
        data_max = float(self.data_max) if getattr(self, 'data_max', None) else float(np.max(pats))
        sat_thresh = data_max * (0.9961 if on_gpu else 0.999) if params['saturation_protect'] else np.inf
        dthresh = params['dthresh']

        # σ² of the center pixel and the corresponding neighborhood slice in sigma_sq_map.
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        r0 = max(0, row - sr); r1 = min(n_avail_rows, row + sr + 1)
        c0 = max(0, col - sr); c1 = min(self.ncols, col + sr + 1)
        sigma_sq_nb = self.sigma_sq_map[r0:r1, c0:c1].astype(np.float64)
        sigma_sq_i = float(self.sigma_sq_map[row, col])

        lam_sq = lam * lam
        # Vectorised over the neighbourhood (a 479x479 pattern with SR 6 is 169 x 229k
        # values; a per-neighbour Python loop with masked copies took ~0.9 s per click,
        # this takes a fraction of that). float32 is exact for uint8/uint16 differences;
        # the sums run in float64.
        npix = ph * pw
        P = pats.reshape(nr * nc, npix).astype(np.float32, copy=False)            # (N, npix)
        c32 = center_pat.astype(np.float32).ravel()
        center_ok = ap_flat & (c32 < sat_thresh)
        keep = np.asarray(mask, dtype=bool).ravel()                               # IQ-range mask
        d_raw = np.zeros(nr * nc, dtype=np.float64); n_pair = np.zeros(nr * nc, dtype=np.float64)
        step = max(1, int(1.5e8 // max(npix * 4, 1)))                             # ~150 MB per block
        c_ok32 = center_ok.astype(np.float32)
        for a in range(0, nr * nc, step):
            blk = P[a:a + step]                                                   # view, no gather copy
            diff = blk - c32                                                      # (k, npix) float32
            diff *= diff
            if np.isfinite(sat_thresh):
                valid = (blk < sat_thresh).astype(np.float32); valid *= c_ok32    # per-pair pixel set
                diff *= valid
                n_pair[a:a + step] = valid.sum(axis=1, dtype=np.float64)
            else:
                diff *= c_ok32
                n_pair[a:a + step] = float(center_ok.sum())
            d_raw[a:a + step] = diff.sum(axis=1, dtype=np.float64)
        sigma_comb = sigma_sq_i + sigma_sq_nb.ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            d_norm = (d_raw - n_pair * sigma_comb) / (sigma_comb * np.sqrt(2.0 * n_pair))
        w = np.exp(-np.maximum(d_norm - dthresh, 0.0) / lam_sq)
        # Degenerate pairs, mirroring both kernels: a vanishing normaliser gives
        # dd = 1e6*n -> weight 0; a pair with NO comparable pixels (n = 0, e.g. a neighbour
        # entirely at/above the saturation threshold) gives dd = 0 -> weight 1.
        w[(sigma_comb < 1e-10) | ~np.isfinite(w)] = 0.0
        w[n_pair < 1.0] = 1.0
        w[~keep] = 0.0                                                            # excluded neighbours
        w[cr * nc + cc] = 1.0                                                     # self-pair, as in both kernels
        weight_total = float(w.sum())
        if weight_total > 0:
            # float32 matvec over <= (2SR+1)^2 terms: error ~1e-6 relative, far below 1 grey level
            return (w.astype(np.float32) @ P).astype(np.float64).reshape(ph, pw) / weight_total
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
        """σ² map for the current aperture, from pyebsdindex (Phase 2).

        pyebsdindex's calcsigma (OpenCL or numba, whichever engine was resolved) is the
        same estimate as before, Brewick et al. 2019 Eq. 9 over the 8 nearest neighbours,
            σ̂_i = sqrt( min_j d(p_i, p_j) / (2 n_ij) ),
        except that n_ij is the per-pair count of aperture pixels that are below the
        saturation threshold in both patterns (saturation_protect), where we used the
        fixed aperture count. The map lives on the shared NLPAR object (_get_nlobj):
        Optimize Lambda computes it through the very same call, so after Optimize Lambda
        this is free and the first preview click is instant; otherwise the one-time pass
        happens here. self.sigma_sq_map = σ² as (nrows, ncols) float32; rows a truncated
        map never recorded are 0 (the preview gives such neighbours zero weight)."""
        key = self._aperture_key()
        if self.sigma_sq_map is not None and self._sigma_aperture_key == key:
            return
        t0 = time.time()
        nlobj, from_ram = self._get_nlobj(self.sr_spin.value())
        n_avail = int(getattr(self, 'n_actual_rows', self.nrows))
        sig = None if nlobj.sigma is None else np.asarray(nlobj.sigma)
        if sig is None or sig.ndim != 2 or sig.shape != (n_avail, int(self.ncols)):
            if self._nlpar_busy:
                # A pyebsdindex call is in flight on this object (preview click during a
                # batch). Do not start a second one; the caller falls back gracefully.
                return
            N_p, ap_label = self._nlobj_mask_info(nlobj)
            engine = self._nlpar_engine_of(nlobj)[1]
            print(f"[σ² MAP] engine={engine} aperture={ap_label} N_p={N_p} "
                  f"rows={n_avail}/{self.nrows} source={'RAM' if from_ram else 'file'}")
            self.statusBar().showMessage(f"Computing σ map with {engine} (one-time per aperture)...")
            QApplication.processEvents()
            self._unhook_nlpar_progress(nlobj)
            self._hook_nlpar_progress(nlobj, "σ map (pyebsdindex)", n_avail * int(self.ncols))
            # Both engines return (sigma, dist, count). calcsigma_cl also stores sigma on the
            # object, calcsigma_cpu does not (only opt_lambda_cpu does), so store it here so
            # the batch reuses it either way.
            self._nlpar_busy = True
            try:
                sig = np.asarray(nlobj.calcsigma(nn=1, automask=False, chunksize=self._nlpar_chunk_bytes(),
                                                 saturation_protect=self._nlpar_params()['saturation_protect'])[0])
            finally:
                self._nlpar_busy = False
            nlobj.sigma = sig
        sigma_sq = np.zeros((int(self.nrows), int(self.ncols)), dtype=np.float32)
        sigma_sq[:sig.shape[0], :sig.shape[1]] = sig.astype(np.float32) ** 2
        self.sigma_sq_map = sigma_sq
        self._sigma_aperture_key = key
        N_p, ap_label = self._nlobj_mask_info(nlobj)
        # Diagnostic summary so we can sanity-check σ values vs. raw pattern intensity scale.
        valid = self.sigma_sq_map[self.sigma_sq_map > 0]
        if len(valid) > 0:
            med = float(np.median(valid)); lo = float(np.percentile(valid, 5)); hi = float(np.percentile(valid, 95))
            print(f"[σ² MAP] {NLPAR_ENGINE}, aperture={ap_label} N_p={int(N_p)} "
                  f"σ²: median={med:.3g}, 5–95%=[{lo:.3g}, {hi:.3g}], "
                  f"σ (intensity units): median={np.sqrt(med):.3g}, time={time.time()-t0:.1f}s")
        self.statusBar().showMessage(f"σ² map ready ({time.time()-t0:.1f}s)")

    # ---- Base pattern retrieval (on-demand) ----
    # ------------------------------------------------------------------
    # pyebsdindex NLPAR object shared by Optimize Lambda and the batch
    # (PLAN_NLPAR_PYEBSDINDEX.md, Phase 1). opt_lambda computes the sigma map
    # (calcsigma, nn=1) and leaves it on the object; calcnlpar then reuses it instead
    # of a second full pass. sigma depends on the dataset, the row cap and the pixel
    # mask, not on SR or lambda, so the object is keyed on (path, aperture, rows) and
    # SR / lambda are set per call.
    # ------------------------------------------------------------------
    def _get_nlobj(self, sr):
        """Return (nlobj, from_ram) for the loaded dataset, creating it on a key change.
        The aperture goes in as pyebsdindex's own pixel mask (nlobj.mask); mode 'None'
        leaves it None and automask=False at the call sites makes that an all-ones mask."""
        n_avail = int(getattr(self, 'n_actual_rows', self.nrows))
        key = (self.orig_path, self._aperture_key(), n_avail)
        if self._nlobj is None or self._nlobj_key != key:
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            cls = nlpar.NLPAR
            if not _gpu_kernel_safe_for(ph, pw):
                from pyebsdindex import nlpar_cpu
                cls = nlpar_cpu.NLPAR
                print(f"[NLPAR] {ph}x{pw} patterns ({ph*pw} px, not a multiple of 16) and the OpenCL kernel is "
                      f"not fully patched: using pyebsdindex's CPU engine for this dataset")
            nlobj = cls(self.orig_path, searchradius=int(sr))
            # Truncated file: cap the scan rows pyebsdindex will read. getinfileobj()
            # forces fID.nRows = self.nrows whenever self.nrows is not None.
            if n_avail < self.nrows:
                nlobj.nrows = n_avail
            from_ram = self._serve_nlpar_input(nlobj)
            ap = self._current_aperture()
            nlobj.mask = (ap > 0).astype(np.uint8) if ap is not None else None
            self._nlobj, self._nlobj_key, self._nlobj_from_ram = nlobj, key, from_ram
        nlobj = self._nlobj
        nlobj.searchradius = int(sr)
        return nlobj, self._nlobj_from_ram

    # pyebsdindex parameters exposed in the NLPAR panel (Phase 3). Both feed sigma / lambda
    # / batch / preview alike; changing either invalidates sigma, the shared object and
    # the lambda gate (sigma depends on saturation_protect, the lambda fit on dthresh).
    def _nlpar_params(self):
        """dict(saturation_protect=bool, dthresh=float) from the panel."""
        return dict(saturation_protect=bool(self.sat_chk.isChecked()),
                    dthresh=float(self.dthresh_spin.value()))

    def _on_nlpar_param_changed(self, which):
        """which='saturation': sigma depends on it -> drop sigma, the shared object and the
        lambda gate. which='dthresh': only the lambda fit depends on it -> re-arm the gate,
        keep sigma (a full pass on a 31 GB scan; code review 2026-10-13)."""
        if self._nlpar_busy:
            return
        if which == 'saturation':
            self._invalidate_sigma_map()
            self._nlobj = None; self._nlobj_key = None
        self._lambda_optimized = False
        if self.f_orig is not None:
            self.update_display()

    def _nlpar_engine_of(self, nlobj):
        """(on_gpu, label) for the object actually in use: the module-level choice can be
        overridden per dataset by _get_nlobj (kernel not safe for this pattern size)."""
        on_gpu = type(nlobj).__module__.startswith('pyebsdindex.opencl')
        return on_gpu, ("pyebsdindex OpenCL" if on_gpu else "pyebsdindex CPU (numba)")

    def _nlpar_exclusion(self):
        """(nPatterns,) bool of patterns the IQ range excludes, or None if none.
        Only whole recorded rows count (as everywhere else in the batch)."""
        mask = self._get_iq_mask()
        if mask is None or mask.all():
            return None
        n_avail = int(getattr(self, 'n_actual_rows', self.nrows)); ncols = int(self.ncols)
        n_pat = int(getattr(self, 'n_actual_pats', n_avail * ncols))
        excl = np.zeros(n_pat, dtype=bool)
        m = ~np.asarray(mask, dtype=bool)[:n_avail].ravel()
        excl[:min(m.size, n_pat)] = m[:n_pat]
        return excl if excl.any() else None

    def _nlobj_mask_info(self, nlobj):
        """(N_p, label) for console / status messages."""
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        if nlobj.mask is not None:
            return int(np.asarray(nlobj.mask).sum()), self.aperture_mode.currentText()
        return ph * pw, "None"

    @staticmethod
    def _reset_nlobj_output(nlobj):
        """Drop any per-batch output override (_nlpar_result_to_ram replaces setoutfile /
        getoutfileobj on the instance) so pyebsdindex's own file output applies again."""
        for name in ('setoutfile', 'getoutfileobj'):
            nlobj.__dict__.pop(name, None)

    @staticmethod
    def _unhook_nlpar_progress(nlobj):
        prev = nlobj.__dict__.pop('_kpp_prev_io', None)
        if prev is not None:
            nlobj.getinfileobj, nlobj.getoutfileobj = prev

    def _hook_nlpar_progress(self, nlobj, stage, total_pats):
        """Drive the status bar from pyebsdindex's own I/O, which is the only progress
        signal it exposes: every tile it reads goes through getinfileobj().read_data and
        every tile it writes through getoutfileobj().pat_writer (the slab branch of
        write_data recurses into one pat_writer call per row). Wraps whichever objects
        are current, the real UPFile or the RAM-serving twins. Call _unhook first on an
        object that was hooked before; returns a closure that prints the final state."""
        t0 = time.time()
        state = {'read': 0, 'written': 0, 'last': 0.0}
        total = max(int(total_pats), 1)
        def show(force=False):
            now = time.time()
            if not force and now - state['last'] < 0.5: return
            state['last'] = now
            el = now - t0
            if state['written'] > 0:
                frac = min(state['written'] / total, 1.0)
                eta = (el / frac - el) if frac > 0 else 0.0
                msg = f"{stage}: {100*frac:.0f}% written ({el:.0f}s, ETA {eta:.0f}s)"
            else:
                frac = min(state['read'] / total, 1.0)     # tiles overlap, so clamp
                msg = f"{stage}: reading {100*frac:.0f}% ({el:.0f}s)"
            self.statusBar().showMessage(msg); QApplication.processEvents()
        self._nlpar_prog = {'state': state, 'show': show}   # the wrappers below read this
        orig_in, orig_out = nlobj.getinfileobj, nlobj.getoutfileobj
        win = self
        def infile():
            f = orig_in()
            if f is not None and not getattr(f, '_kpp_hooked_in', False):
                rd = f.read_data
                def read_data(*a, **k):
                    r = rd(*a, **k)
                    try:
                        n = r[0].shape[0] if isinstance(r, tuple) else getattr(r, 'nPatterns', 0)
                        win._nlpar_prog['state']['read'] += int(n); win._nlpar_prog['show']()
                    except Exception: pass
                    return r
                f.read_data = read_data; f._kpp_hooked_in = True
            return f
        def outfile():
            f = orig_out()
            if f is not None and not getattr(f, '_kpp_hooked_out', False):
                pw_ = f.pat_writer
                def pat_writer(pat2write, patStart, nPatToWrite, *a, **k):
                    r = pw_(pat2write, patStart, nPatToWrite, *a, **k)
                    try:
                        win._nlpar_prog['state']['written'] += int(nPatToWrite); win._nlpar_prog['show']()
                    except Exception: pass
                    return r
                f.pat_writer = pat_writer; f._kpp_hooked_out = True
            return f
        nlobj._kpp_prev_io = (orig_in, orig_out)
        nlobj.getinfileobj, nlobj.getoutfileobj = infile, outfile
        return lambda: show(force=True)

    def _serve_nlpar_input(self, nlobj):
        """Give a pyebsdindex NLPAR object its input file object: RamUPFile serving
        self.patterns when the dataset is in RAM (no re-read of the file), else a
        MaskedUPFile on the original path. Both carry the Phase 3 `exclude` flag the
        batch uses for the IQ-range exclusion; the object is kept as self._nl_infile so
        the batch can set it. Mirrors what NLPAR.getinfileobj() does with nrows/ncols so a
        capped row count (map stopped early) still applies. Returns from_ram."""
        from_ram = self.patterns is not None
        fobj = RamUPFile(self.orig_path, self.patterns) if from_ram else MaskedUPFile(self.orig_path)
        def infile():
            if nlobj.nrows is None: nlobj.nrows = fobj.nRows
            else: fobj.nRows = nlobj.nrows
            if nlobj.ncols is None: nlobj.ncols = fobj.nCols
            else: fobj.nCols = nlobj.ncols
            return fobj
        nlobj.getinfileobj = infile
        self._nl_infile = fobj
        return from_ram

    _serve_nlpar_from_ram = _serve_nlpar_input          # old name

    def _alloc_nlpar_ram(self, tag="[CPU NLPAR]"):
        """Allocate self.nlpar_patterns, the in-RAM NLPAR batch result (same shape and
        native dtype as self.patterns), for either batch path to write into. Needs the
        original in RAM and room for a second copy (same 60%-of-available rule as the
        load); otherwise returns False and the caller writes its temp file.
        KPP_NLPAR_IN_MEMORY=0 forces the file. Only whole rows [0, n_actual_rows) are
        produced by NLPAR; the partial last row of a map stopped early is copied here
        from the original. `tag` only prefixes the console message."""
        if self.patterns is None:
            return False
        if os.environ.get('KPP_NLPAR_IN_MEMORY', '').strip().lower() in ('0', 'no', 'false'):
            return False
        self.nlpar_patterns = None           # free a previous result first
        need = self.patterns.nbytes; avail = available_ram()
        if need > 0.6 * avail:
            print(f"{tag} result goes to disk: a second {need/1e9:.1f} GB copy does not fit "
                  f"({avail/1e9:.0f} GB available to this process)")
            return False
        self.nlpar_patterns = np.empty_like(self.patterns)
        n_full = int(getattr(self, 'n_actual_rows', self.nrows)) * int(self.ncols)
        if n_full < self.nlpar_patterns.shape[0]:      # partial last row: NLPAR does whole rows only
            self.nlpar_patterns[n_full:] = self.patterns[n_full:]
        return True

    def _nlpar_result_to_ram(self, nlobj):
        """Point a pyebsdindex NLPAR object's OUTPUT at a second in-RAM array
        (self.nlpar_patterns) instead of _cpu_nlpar_result.up1 (2026-09-28: only
        Apply && Save should write to disk). Allocation rules: _alloc_nlpar_ram. Returns
        False (caller writes the temp file as before) when it does not fit."""
        if not self._alloc_nlpar_ram():
            return False
        out = RamOutUPFile(self.orig_path, self.nlpar_patterns)
        def outfile():
            if nlobj.nrows is not None: out.nRows = nlobj.nrows
            if nlobj.ncols is not None: out.nCols = nlobj.ncols
            return out
        nlobj.setoutfile = lambda patternfile, filepath=None: None   # no temp file on disk
        nlobj.getoutfileobj = outfile
        return True

    def _read_patterns(self, fobj, p0, n):
        """Patterns [p0, p0+n) of `fobj` in file order, regardless of row boundaries
        (used for the partial last row of a map that was stopped early). RAM view when
        `fobj` is the in-memory original, else the parallel reader."""
        ph, pw = int(fobj.patternH), int(fobj.patternW)
        p0, n = int(p0), int(n)
        ram = self.patterns if fobj is self.f_orig else getattr(fobj, '_ram', None)
        if ram is not None:
            return ram[p0:p0 + n]
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
        ram = self.patterns if fobj is self.f_orig else getattr(fobj, '_ram', None)
        if ram is not None:                  # in-RAM original, or the in-RAM NLPAR result
            block = ram[row_start * ncols:(row_start + n_rows) * ncols]
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
          normalize off → returns values in absolute intensity [0, scale_max]
                          (255 for uint8; the effective full scale for uint16, see __init__)
        The two output ranges are handled in update_display when picking display vmin/vmax,
        and in apply_and_save / _process_chunk_batch when scaling to the file dtype.
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
        normalize = self.normalize_chk.isChecked()
        global_stretch = self.global_chk.isChecked()
        if self.static_bg_chk.isChecked() and self.mean_pattern is not None:
            bg = self._static_background_term(not (normalize or global_stretch), ap_bool)
            if self.bg_mode.currentText() == "Subtract": p = p - bg
            else: p = p / (bg + 1e-10)
        if self.dyn_bg_chk.isChecked():
            p = p - gaussian_filter(p, sigma=self.dyn_sigma.value() / 10.0)
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
            # Preserve absolute intensity. B/C centered on the intensity midpoint
            # (127.5 for uint8, scale_max/2 in general). bright slider [-100..100]/100 →
            # bval is in [-1,1], scaled to ±scale_max here.
            mid = 0.5 * self.scale_max
            p = (p - mid) * (1.0 + cval) + mid + bval * self.scale_max
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
            p = np.clip(p, 0, self.scale_max)
            return p

    # ---- Batch processing (GPU-accelerated when available) ----
    def _process_chunk_batch(self, chunk, mask_rows, progress=None):
        """Process a chunk of patterns in batch. chunk: (N, ph, pw) float64.
        `progress(done, N)` (optional) is called from this thread while the chunk is
        processed, so the caller can keep the status bar alive inside a long chunk.
        Returns processed chunk in float64, clipped to the file's integer range, ready for
        the uint8/uint16 write. Mirrors process_pattern: when normalize (or global stretch)
        is on, internal range is [0,1] and we scale to the full dtype range (255 / 65535)
        at the end; when off, internal range stays in absolute intensity and we clip to
        scale_max. Patterns with mask_rows False (outside the IQ range) are returned as
        they came in, unprocessed: they take no part in the statistics (background mean,
        global stretch range, NLPAR sigma and neighbours), but they stay in the data set
        so that they can still be indexed."""
        N = chunk.shape[0]
        excluded = ~np.asarray(mask_rows, dtype=bool)
        raw_excluded = chunk[excluded] if excluded.any() else None   # copy, before any pipeline touches the chunk
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
                gpu_mean = cp.asarray(self._static_background_term(not unit_domain, ap_bool), dtype=cp.float32)
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
                # Preserve absolute intensity. B/C centered on the intensity midpoint.
                mid = cp.float32(0.5 * self.scale_max)
                d = (d - mid) * (1.0 + cval) + mid + cp.float32(bval * self.scale_max)

            # Transfer back to CPU
            result = cp.asnumpy(d).astype(np.float64)
            del d
            if g_ap is not None: del g_ap
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU: run the whole chain (background, stretch, CLAHE, final scale) on
            # sub-batches across a thread pool; see _cpu_pipeline_threaded.
            result = self._cpu_pipeline_threaded(chunk, mask_rows, progress=progress,
                use_static=use_static, bg_subtract=(self.bg_mode.currentText() == "Subtract"),
                static_bg=(self._static_background_term(not unit_domain, ap_bool) if use_static else None),
                use_dyn=use_dyn, use_clahe=use_clahe, normalize=normalize,
                global_stretch=global_stretch, unit_domain=unit_domain, sigma=sigma,
                bval=bval, cval=cval, clip_limit=clip_limit, ap_bool=ap_bool,
                g_lo=(g_lo if global_stretch else None), g_hi=(g_hi if global_stretch else None),
                scale_max=float(self.scale_max), dtype_max=float(self.dtype_max))
            if raw_excluded is not None:
                result[excluded] = raw_excluded
            return result

        # CLAHE stays on CPU (per-pattern, no good batch GPU impl yet).
        # When normalize is on, input is in [0,1] and we apply CLAHE directly.
        # When off, we temporarily scale per-pattern to [0,1] for CLAHE then map
        # back to the original [pmin, pmax] span to preserve absolute intensity.
        if use_clahe:
            last = time.time()
            for i in range(N):
                if progress is not None and time.time() - last > 0.5:
                    progress(i, N); last = time.time()
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

        # Final scale to the file's integer range. When in unit domain (per-pattern
        # normalize OR global stretch), data is in [0,1] so scale to the full dtype range
        # (255 for .up1, 65535 for .up2) first; when off, data is already in absolute
        # intensity so just clip to the effective full scale.
        if unit_domain:
            result = np.clip(result * float(self.dtype_max), 0, self.dtype_max)
        else:
            result = np.clip(result, 0, self.scale_max)
        # [MVM 2026-09-08] The aperture is deliberately NOT re-asserted here. It is a
        # statistics gate only (min/max, global stretch bounds, CLAHE source range,
        # N_p in NLPAR); masked pixels keep their own processed values and reach disk
        # intact, so exported patterns are complete for OIM/EMSphInx/OpenXY.
        # IQ-range exclusion is a different thing: those whole patterns go out unprocessed.
        if raw_excluded is not None:
            result[excluded] = raw_excluded
        return result

    def _cpu_pipeline_subbatch(self, sub, mask_sub, use_static, bg_subtract, use_dyn, use_clahe,
                               normalize, global_stretch, unit_domain, sigma, bval, cval,
                               clip_limit, ap_bool, g_lo, g_hi, scale_max=255.0, dtype_max=255.0,
                               static_bg=None):
        """CPU processing chain for one sub-batch (n, ph, pw) float64 -> float64 in
        [0, dtype_max] (unit domain) or [0, scale_max] (absolute intensity).
        This is the original CPU fallback of _process_chunk_batch plus its CLAHE loop and
        final scaling, unchanged in the math, split out so several copies can run at once.
        No Qt widget access in here: everything UI-derived arrives as an argument."""
        result = sub.copy()
        n = result.shape[0]
        if use_static:
            bg = self.mean_pattern if static_bg is None else static_bg   # see _static_background_term
            if bg_subtract:
                result -= bg
            else:
                result /= (bg + 1e-10)
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
            mid = 0.5 * scale_max
            result = (result - mid) * (1.0 + cval) + mid + bval * scale_max
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
            result = np.clip(result * dtype_max, 0, dtype_max)
        else:
            result = np.clip(result, 0, scale_max)
        return result

    def _cpu_pipeline_threaded(self, chunk, mask_rows, progress=None, **kw):
        """Split a chunk into ~3 sub-batches per worker thread and run _cpu_pipeline_subbatch
        on each in a thread pool. scipy.ndimage, the big numpy reductions and most of
        skimage's CLAHE release the GIL, so this scales across the cores SLURM gave us
        (n_workers). Output is assembled into one float64 array of the chunk's shape.
        `progress(done_patterns, N)` is called from the CALLING thread as sub-batches
        finish (safe to touch Qt from it): a 127-row chunk of the FG10 TKD scan took
        minutes with no status-bar change, which looked like a hang (2026-10-10)."""
        N = chunk.shape[0]
        nthr = max(1, min(n_workers(), N))
        nsub = max(1, min(N, nthr * 3))
        bounds = np.linspace(0, N, nsub + 1).astype(int)
        out = np.empty_like(chunk)
        def work(k):
            i0, i1 = int(bounds[k]), int(bounds[k + 1])
            if i1 > i0:
                out[i0:i1] = self._cpu_pipeline_subbatch(chunk[i0:i1], mask_rows[i0:i1], **kw)
            return i1 - i0
        with ThreadPoolExecutor(max_workers=nthr) as pool:
            futs = [pool.submit(work, k) for k in range(nsub)]
            if progress is None:
                for f in futs: f.result()
            else:
                done = 0; last = 0.0
                for f in as_completed(futs):
                    done += f.result()
                    if time.time() - last > 0.5:
                        progress(done, N); last = time.time()
                progress(N, N)
        return out

    # ---- Display ----
    def on_pattern_clicked(self, row, col):
        self.current_row, self.current_col = row, col
        self.scan_canvas._place_marker(row, col); self.update_display()

    def update_display(self):
        if self.f_orig is None: return
        orig = self._read_pattern(self.f_orig)
        # Raw panel: true absolute scale (vmin=0, vmax=scale_max: 255 for uint8, the
        # effective full scale for uint16). Patterns with weaker raw signal will visibly
        # look dimmer, instead of being silently stretched to fill the colormap by
        # matplotlib's auto-normalize.
        self.raw_canvas.show_pattern(orig, f"Original [{self.current_col}, {self.current_row}]",
                                     vmin=0, vmax=self.scale_max)
        if self.current_row >= self.nrows or self.current_col >= self.ncols: return  # stale click from a larger scan
        if self.iq_mask is not None and not self.iq_mask[self.current_row, self.current_col]:
            # Outside the IQ range: the batch writes this pattern unprocessed, so show it as it is
            self.proc_canvas.show_pattern(orig, f"Outside IQ range: kept unprocessed [{self.current_col}, {self.current_row}]",
                                          vmin=0, vmax=self.scale_max)
            return
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
        #   normalize off → output is in absolute intensity → vmin=0, vmax=scale_max
        # Using vmin=0,vmax=scale_max in the unnormalized case matches what apply_and_save
        # actually writes to the file, so the preview is honest about the export.
        if self.normalize_chk.isChecked() or self.global_chk.isChecked():
            self.proc_canvas.show_pattern(processed,
                f"{src}Processed [{self.current_col}, {self.current_row}]", fixed_range=True)
        else:
            self.proc_canvas.show_pattern(processed,
                f"{src}Processed [{self.current_col}, {self.current_row}]",
                vmin=0, vmax=self.scale_max)
        self.statusBar().showMessage(f"Pattern processed in {elapsed*1000:.0f} ms")

    # ---- File loading ----
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open UP1/UP2 File",
            "", "UP1/UP2 Files (*.up1 *.up2)")
        if not path: return
        self.load_path(path)

    def load_path(self, path):
        """Load a UP1/UP2 file by path (no dialog). Split out of load_file so the
        loader can be driven headlessly for testing."""
        if self._nlpar_busy:
            QMessageBox.information(self, "NLPAR is running", "Wait for the NLPAR step to finish before loading another file.")
            return
        self.statusBar().showMessage("Loading patterns..."); QApplication.processEvents()
        # Drop EVERYTHING that belongs to the previous dataset before touching any widget.
        # The slider resets below emit valueChanged -> _update_iq_range -> update_display,
        # so anything still pointing at the old scan at that moment (its NLPAR file, its
        # sigma map, the last clicked row/col, the auto-detect aperture cache) is used
        # against the new one: wrong shapes, IndexError, and with no excepthook installed
        # PyQt aborts the whole app on an exception inside a slot. [2026-09-27 review]
        self.f_nlpar = None; self.nlpar_patterns = None; self.nlpar_status.setText("No NLPAR file loaded")
        self._remove_temp_files()          # previous dataset's NLPAR batch result
        self.current_row = self.current_col = 0
        self.iq_map = None; self.iq_mask = None
        self._invalidate_sigma_map()      # sigma map belongs to the old scan
        self._invalidate_global_range()   # so do the global-stretch bounds
        self._lambda_optimized = False    # lambda gate re-arms for the new file
        self._nlobj = None; self._nlobj_key = None   # pyebsdindex object (and its sigma) is per dataset
        self._aperture_cache = None; self._aperture_cache_key = None  # auto-detect mask is per dataset
        self.orig_path = path; self.f_orig = ebsd_pattern.get_pattern_file_obj(path)
        self.nrows, self.ncols = self.f_orig.nRows, self.f_orig.nCols
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        # Check for truncated file but KEEP original header dimensions for OIM compat
        fsize = int(os.path.getsize(path))
        bpp = int(np.dtype(self.f_orig.filedatatype).itemsize)   # 1 (.up1) or 2 (.up2)
        self.src_ext = os.path.splitext(path)[1].lower()
        self.dtype_max = int(np.iinfo(self.f_orig.filedatatype).max)
        self.scale_max = float(self.dtype_max); self.data_max = None
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
        # Threaded statistics (HPC CPU nodes): each chunk is read once in this thread, then split
        # into ~nthr slices of patterns that the pool reduces independently. The merge
        # is exact: the per-pattern std is independent of the split, and both running
        # sums are float64 over integer-valued terms, so the result is bit-identical to
        # the old single-threaded loop. Per-slice temporaries are float32 (cast + square),
        # so cap the threads such that they fit in a quarter of the available RAM.
        ncols = int(self.ncols)
        row_bytes_f32 = ncols * ph * pw * 4
        nthr = int(max(1, min(n_workers(), (0.25 * available_ram()) // (2 * row_bytes_f32))))
        print(f"[LOAD] {nthr} threads")

        data_max = 0

        def slice_stats(sl):
            cf = sl.astype(np.float32)
            iq = cf.std(axis=(1, 2))
            s1 = sl.sum(axis=0, dtype=np.float64)
            s2 = np.square(cf).sum(axis=0, dtype=np.float64)
            mx = int(sl.max()) if sl.size else 0      # dataset max -> effective bit depth
            return iq, s1, s2, mx

        with ThreadPoolExecutor(max_workers=nthr) as pool:
            for row_start in range(0, self.n_actual_rows, CHUNK_ROWS):
                rows_to_read = min(CHUNK_ROWS, self.n_actual_rows - row_start)
                chunk = self._read_rows(self.f_orig, row_start, rows_to_read)
                n = chunk.shape[0]
                bounds = np.linspace(0, n, min(nthr, n) + 1, dtype=int)
                slices = [chunk[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
                for iq, s1, s2, mx in pool.map(slice_stats, slices):
                    iq_values.append(iq)
                    mean_accum += s1
                    sumsq_accum += s2
                    data_max = max(data_max, mx)
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
        # Intensity full scale (see __init__): uint8 stays 255; a uint16 file gets the
        # smallest 2^k-1 that holds its data (a 12-bit camera in a .up2 -> 4095), so the
        # raw/absolute preview and the B/C midpoint track the signal, not the container.
        self.data_max = int(data_max)
        if self.dtype_max > 255:
            bits = max(8, int(np.ceil(np.log2(self.data_max + 1)))) if self.data_max > 0 else 8
            self.scale_max = float(min((1 << bits) - 1, self.dtype_max))
        else:
            self.scale_max = 255.0
        print(f"[LOAD] {np.dtype(self.f_orig.filedatatype).name} patterns ({self.src_ext}): "
              f"data max = {self.data_max}, intensity full scale = {int(self.scale_max)} "
              f"(dtype max {self.dtype_max})")
        self.file_label.setText(f"{os.path.basename(path)}\n{self.ncols} x {self.nrows} = {self.ncols*self.nrows} patterns ({self.mem_mode})"
                                f"\n{8 * bpp}-bit ({self.src_ext}), data max {self.data_max}, scale 0-{int(self.scale_max)}")
        self.bg_status.setText("Background: all patterns")
        self.scan_canvas.set_map(self.iq_map, self.nrows, self.ncols)
        self.iq_mask = np.ones((self.nrows, self.ncols), dtype=bool)
        self.iq_min_slider.setValue(0); self.iq_max_slider.setValue(1000); self._update_iq_range()
        self.on_pattern_clicked(0, 0); self.statusBar().showMessage("Ready")

    def unload(self):
        """Drop the loaded data set and everything derived from it, and clear the plots (2026-10-19).
        The workbench calls this before loading a file into its indexing tab, so that only one data
        set is in memory at a time. Returns False (and does nothing) while a pyebsdindex call runs.
        Harmless when nothing is loaded."""
        if self._nlpar_busy:
            return False
        self.f_nlpar = None; self.nlpar_patterns = None; self.nlpar_status.setText("No NLPAR file loaded")
        self._remove_temp_files()
        self.orig_path = None; self.f_orig = None; self.patterns = None
        self.nrows = self.ncols = self.current_row = self.current_col = 0
        self.iq_map = None; self.iq_mask = None; self.mean_pattern = None; self.pixel_std = None; self.data_max = None
        self._invalidate_sigma_map(); self._invalidate_global_range(); self._lambda_optimized = False
        self._nlobj = None; self._nlobj_key = None; self._nl_infile = None
        self._aperture_cache = None; self._aperture_cache_key = None
        self.file_label.setText("No file loaded"); self.bg_status.setText("Background: all patterns")
        for canvas in (self.scan_canvas, self.raw_canvas, self.proc_canvas):
            canvas.ax.clear(); canvas.ax.axis('off'); canvas.draw_idle()
        self.scan_canvas.marker = None; self.scan_canvas._mask = None; self.scan_canvas._img_handle = None
        self.scan_canvas._mask_handle = None; self.scan_canvas.nrows = self.scan_canvas.ncols = 0
        gc.collect()
        self.statusBar().showMessage("Data set unloaded")
        return True

    def load_nlpar_file(self):
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first"); return
        path, _ = QFileDialog.getOpenFileName(self, "Open NLPAR UP1/UP2 File",
            os.path.dirname(self.orig_path), "UP1/UP2 Files (*.up1 *.up2)")
        if not path: return
        f_new = ebsd_pattern.get_pattern_file_obj(path)
        # The NLPAR file stands in for the original in every pass, so it must have the
        # same grid, pattern size and bit depth (a .up1 next to a .up2 source is uint8).
        same = (int(f_new.patternH) == int(self.f_orig.patternH) and int(f_new.patternW) == int(self.f_orig.patternW)
                and int(f_new.nCols) == int(self.ncols) and int(f_new.nRows) == int(self.nrows)
                and np.dtype(f_new.filedatatype) == np.dtype(self.f_orig.filedatatype))
        if not same:
            QMessageBox.warning(self, "NLPAR file does not match",
                f"{os.path.basename(path)} is {int(f_new.nCols)}x{int(f_new.nRows)} scan points of "
                f"{int(f_new.patternW)}x{int(f_new.patternH)} px, {np.dtype(f_new.filedatatype).name};\n"
                f"the loaded dataset is {self.ncols}x{self.nrows} of "
                f"{int(self.f_orig.patternW)}x{int(self.f_orig.patternH)} px, {np.dtype(self.f_orig.filedatatype).name}.")
            return
        self.nlpar_patterns = None       # a file replaces any in-RAM result
        self.f_nlpar = f_new
        self.nlpar_chk.setChecked(False)  # disable on-demand since we have a file
        self.nlpar_status.setText(f"Loaded: {os.path.basename(path)}"); self.update_display()

    def recompute_background(self):
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first"); return
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
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first"); return
        if self._nlpar_busy:
            QMessageBox.information(self, "NLPAR is running", "Wait for the running NLPAR step to finish."); return
        try:
            t0 = time.time()
            sr = self.sr_spin.value()
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            nlobj, from_ram = self._get_nlobj(sr)
            self._unhook_nlpar_progress(nlobj)
            n_avail = int(getattr(self, 'n_actual_rows', self.nrows))
            N_p, ap_label = self._nlobj_mask_info(nlobj)
            budget = self._nlpar_chunk_bytes()
            engine = self._nlpar_engine_of(nlobj)[1]
            print(f"[OPT LAMBDA] engine={engine} file={os.path.basename(self.orig_path)} SR={sr} "
                  f"aperture={ap_label} N_p={N_p}/{ph*pw} ({100.0*N_p/(ph*pw):.1f}%) "
                  f"rows={n_avail}/{self.nrows} chunk_budget={budget/1e9:.2f} GB "
                  f"source={'RAM' if from_ram else 'file'}")
            self.statusBar().showMessage(
                f"Optimizing lambda (aperture={ap_label}, N_p={N_p})..."); QApplication.processEvents()
            self._hook_nlpar_progress(nlobj, "Optimize lambda (sigma pass)", n_avail * int(self.ncols))
            params = self._nlpar_params()
            # nlpar_cl.NLPAR.opt_lambda swallows a dthresh keyword instead of forwarding it
            # (code review 2026-10-13); both engines read self.dthresh, so set it on the object.
            nlobj.dthresh = float(params['dthresh'])
            self._nlpar_busy = True
            try:
                nlobj.opt_lambda(chunksize=budget, automask=False, autoupdate=True, backsub=False, **params)
            finally:
                self._nlpar_busy = False
            self.lam_spin.setValue(float(nlobj.lam))
            self._lambda_optimized = True
            elapsed = time.time() - t0
            print(f"[OPT LAMBDA] lambda = {float(nlobj.lam):.4f}  (N_p = {N_p}, {elapsed:.1f}s; "
                  f"sigma kept on the NLPAR object for the batch)")
            self.statusBar().showMessage(f"Optimal lambda: {float(nlobj.lam):.3f} (N_p={N_p}, {elapsed:.0f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage("Optimize lambda failed")
            QMessageBox.critical(self, "Optimize lambda failed", str(e))

    def apply_nlpar_batch(self):
        """Run NLPAR on the full dataset through pyebsdindex (OpenCL or numba, chosen at
        import). See PLAN_NLPAR_PYEBSDINDEX.md."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first"); return
        if not self._lambda_optimized:
            QMessageBox.information(self, "Optimize lambda first",
                "Need to optimize lambda before applying NLPAR to the full dataset.\n\n"
                "Click 'Optimize Lambda', wait for it to finish, then click "
                "'Apply NLPAR to Full Dataset' again.")
            return
        sr, lam = self.sr_spin.value(), self.lam_spin.value()
        if self._nlpar_busy:
            QMessageBox.information(self, "NLPAR is running", "Wait for the running NLPAR step to finish."); return

        self._apply_nlpar_pyebsdindex(sr, lam)

    def _apply_nlpar_pyebsdindex(self, sr, lam, raise_errors=False):
        """Full-dataset NLPAR through pyebsdindex, straight off self.orig_path (or the
        in-RAM copy) — no temp copy of the input. pyebsdindex resolved the engine at
        import (NLPAR_ENGINE): its OpenCL kernels when pyopencl sees a GPU, numba on the
        CPU otherwise. The NLPAR object comes from _get_nlobj, so the sigma map computed
        by Optimize Lambda is reused here instead of a second full pass; lambda and SR
        are taken from the spin boxes (a hand-edited lambda wins over nlobj.lam).

        Two contract points that the pre-1.0 implementation got wrong because it fed
        NLPAR a pre-masked duplicate of the scan:

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
        glob below is kept as a fallback.

        raise_errors=True (batch_process, v1.7.0): a failure is raised to the caller
        instead of being shown in a modal QMessageBox, which would block a headless batch."""
        label, engine = ("GPU" if NLPAR_ON_GPU else "CPU"), NLPAR_ENGINE
        try:
            self.statusBar().showMessage(f"Running NLPAR batch, {NLPAR_ENGINE} (SR={sr}, lam={lam:.2f})..."); QApplication.processEvents()
            t0 = time.time()
            ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
            nlobj, from_ram = self._get_nlobj(sr)
            on_gpu, engine = self._nlpar_engine_of(nlobj); label = "GPU" if on_gpu else "CPU"
            # Order matters: unhook restores the I/O methods as they were before the last
            # hook, reset drops a previous batch's RAM output override, then this batch's
            # output is decided and the progress hook wraps the final methods.
            self._unhook_nlpar_progress(nlobj)
            self._reset_nlobj_output(nlobj)
            to_ram = self._nlpar_result_to_ram(nlobj)
            n_avail = int(getattr(self, 'n_actual_rows', self.nrows))
            N_p, ap_label = self._nlobj_mask_info(nlobj)
            out_path = os.path.join(os.path.dirname(self.orig_path), "_nlpar_result" + self.src_ext)
            budget = self._nlpar_chunk_bytes()      # CPU tiles; the OpenCL path sizes its own
            # IQ-range exclusion (Phase 3): excluded patterns are served to pyebsdindex as
            # constant patterns with sigma 0, which gives them zero weight as neighbours (see
            # RamUPFile._apply_exclude); their own output is replaced by the original pattern
            # below, so they stay in the result unprocessed. The true sigma map is put back
            # afterwards so preview / lambda keep using it.
            excl = self._nlpar_exclusion()
            if excl is not None and (nlobj.sigma is None or np.asarray(nlobj.sigma).ndim != 2):
                # Sigma must come from the real patterns, not from the constant-served ones
                # (a flat pattern next to real ones would get a huge sigma and, through the
                # normalisation, full weight). Normally Optimize Lambda already did this.
                # Before the progress hook below, so its own hook/state are not disturbed.
                self._ensure_sigma_map()
            sigma_reused = nlobj.sigma is not None and np.asarray(nlobj.sigma).size > 1
            print(f"[NLPAR {label}] engine={engine} SR={sr} lam={lam:.3f} aperture={ap_label} "
                  f"N_p={N_p}/{ph*pw} ({100.0*N_p/(ph*pw):.1f}%) "
                  f"rows={n_avail}/{self.nrows} chunk_budget={budget/1e9:.2f} GB "
                  f"sigma={'reused from Optimize Lambda' if sigma_reused else 'computed now'} "
                  f"source={'RAM' if from_ram else 'file'} -> {'RAM' if to_ram else out_path}")
            kw = dict(chunksize=budget, lam=float(lam), searchradius=int(sr),
                      automask=False, backsub=False, **self._nlpar_params())
            true_sigma = nlobj.sigma
            if excl is not None:
                print(f"[NLPAR {label}] IQ range excludes {int(excl.sum())} of {excl.size} patterns "
                      f"(served as constant patterns, sigma 0; kept unprocessed in the result)")
                self._nl_infile.exclude = excl
                if true_sigma is not None and np.asarray(true_sigma).ndim == 2:
                    s = np.array(true_sigma, dtype=np.float32, copy=True)
                    keep = ~excl[:s.size].reshape(-1)
                    s.reshape(-1)[:keep.size] *= keep.astype(np.float32)
                    nlobj.sigma = s
            finish = self._hook_nlpar_progress(nlobj, f"NLPAR batch ({label})", n_avail * int(self.ncols))
            self._nlpar_busy = True
            try:
                if to_ram:
                    nlobj.calcnlpar(**kw)
                else:
                    result_path = nlobj.calcnlpar(fileout=out_path, **kw)
            finally:
                self._nlpar_busy = False
                self._nl_infile.exclude = None
                if excl is not None:
                    nlobj.sigma = true_sigma      # only then was the array replaced; otherwise keep
                                                  # whatever calcnlpar computed for later reuse
            finish()
            if excl is not None and (to_ram or (result_path and os.path.isfile(result_path))):
                self._restore_excluded_output(excl, self.nlpar_patterns if to_ram else result_path)
            if to_ram:
                elapsed = time.time() - t0
                self.f_nlpar = RamUPFile(self.orig_path, self.nlpar_patterns)
                self.nlpar_chk.setChecked(False)
                self.nlpar_status.setText(f"NLPAR in RAM ({label}): SR={sr}, lam={lam:.2f} ({elapsed:.1f}s)")
                self.statusBar().showMessage(f"NLPAR batch complete in RAM ({elapsed:.1f}s, {engine}, N_p={N_p})")
                print(f"[NLPAR {label}] done in {elapsed:.1f}s -> RAM")
                self.update_display()
                return
            elapsed = time.time() - t0
            if not (result_path and os.path.isfile(result_path)):
                base = os.path.splitext(self.orig_path)[0]
                candidates = sorted(glob.glob(f"{base}_NLPAR*{self.src_ext}"), key=os.path.getmtime, reverse=True)
                result_path = candidates[0] if candidates else None
            if result_path and os.path.isfile(result_path):
                if result_path not in self._temp_files:
                    self._temp_files.append(result_path)
                self.f_nlpar = ebsd_pattern.get_pattern_file_obj(result_path)
                self.nlpar_chk.setChecked(False)
                self.nlpar_status.setText(f"NLPAR ({label}): SR={sr}, lam={lam:.2f} ({elapsed:.1f}s)")
                self.statusBar().showMessage(f"NLPAR batch complete ({elapsed:.1f}s, {engine}, N_p={N_p})")
                print(f"[NLPAR {label}] done in {elapsed:.1f}s -> {result_path}")
            else:
                if raise_errors:
                    raise RuntimeError("NLPAR ran but output not found.")
                QMessageBox.warning(self, "Warning", "NLPAR ran but output not found.")
            self.update_display()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f"NLPAR ({NLPAR_ENGINE}) failed")
            if raise_errors:
                raise
            QMessageBox.critical(self, f"NLPAR ({NLPAR_ENGINE}) failed", str(e))

    # Old name, still used by osc_selftest.py / osc_gpu_selftest.py (rewritten in Phase 4).
    _apply_nlpar_cpu = _apply_nlpar_pyebsdindex

    def _restore_excluded_output(self, excl, target):
        """Put the ORIGINAL patterns back at the excluded positions of the NLPAR batch
        result, so that patterns outside the IQ range stay in the data set unprocessed
        (they were served to the kernel as constant patterns, whose output is meaningless).
        `target` is the in-RAM result array or the path of the result file."""
        idx = np.flatnonzero(excl)
        if idx.size == 0:
            return
        fobj = None if isinstance(target, np.ndarray) else ebsd_pattern.get_pattern_file_obj(target)
        # group consecutive indices into runs -> one read / write per run
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        for run in np.split(idx, breaks):
            p0, n = int(run[0]), int(run.size)
            orig = self._read_patterns(self.f_orig, p0, n)
            if fobj is None:
                target[p0:p0 + n] = orig
            else:
                fobj.write_data(newpatterns=np.ascontiguousarray(orig, dtype=fobj.filedatatype),
                                patStartCount=[p0, n], writeHead=False)

    def apply_and_save(self):
        """Save all processed patterns to a new UP1/UP2 file (chunked). The output keeps
        the source's extension and bit depth: pyebsdindex (and OIM) take the dtype from
        the extension, so a .up2 source is always written as a .up2. The GUI side (checks,
        file dialog, final message box) lives here; the work is save_processed (v1.7.0)."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1/UP2 file first"); return
        if self.nlpar_chk.isChecked() and self.f_nlpar is None:
            QMessageBox.information(self, "NLPAR not applied to full dataset",
                "Need to apply NLPAR to the full dataset before saving.\n\n"
                "Click 'Apply NLPAR to Full Dataset', wait for it to finish, then "
                "click 'Apply && Save' again.")
            return
        ext = self.src_ext
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Patterns",
            os.path.dirname(self.orig_path), f"{ext[1:].upper()} Files (*{ext})")
        if not path: return
        path = os.path.abspath(path)
        if os.path.splitext(path)[1].lower() != ext:
            # A different extension would change the bit depth the file is read with.
            path = os.path.splitext(path)[0] + ext
            print(f"[SAVE] output extension forced to {ext} (same bit depth as the source)")
        if os.path.exists(path) and os.path.samefile(path, self.orig_path):
            QMessageBox.warning(self, "Error", "Cannot overwrite the loaded source file. Choose another name.")
            return
        # Let the file dialog finish tearing down before the main thread goes busy.
        # Otherwise its deferred X11 events are handled mid-save and the xcb backend logs
        # "XCB error: 3 (BadWindow) ... TranslateCoords" for the already-destroyed dialog
        # window. Harmless, but it reads like a failure (2026-10-10).
        for _ in range(3): QApplication.processEvents()
        r = self.save_processed(path)
        QMessageBox.information(self, "Saved",
            f"Processed patterns saved to:\n{r['path']}\n\nSource: {r['source_label']}\nMode: {'GPU' if HAS_GPU else 'CPU'}\nPatterns: {r['n_saved']}/{r['n_total']}\nTime: {r['seconds']:.1f}s")

    def save_processed(self, path, progress=None, cancel=None):
        """Process the whole loaded data set with the CURRENT widget states and write it to
        `path`, without any dialog (v1.7.0; the body of the former apply_and_save). Used by
        apply_and_save and batch_process.

        The source header is copied byte for byte (a v4 header's map GUID and extra bytes
        survive), the file is pre-sized with a sparse truncate and filled chunk by chunk.
        The output extension is forced to the source's (bit depth). Patterns outside the IQ
        range are written as they are in the source, unprocessed (they are left out of the
        statistics, not of the file, so they can still be indexed). If an NLPAR result is
        loaded (self.f_nlpar) it is the processing source.

        progress(done_patterns, total_patterns): called after each chunk is written.
        cancel(): checked before each chunk; when it returns True the partial output file
        is deleted and BatchCancelled is raised.
        Returns dict(path, n_saved, n_total, seconds, source_label): n_saved is the number of
        patterns processed, n_total the number written (n_total - n_saved went out unprocessed).
        Raises RuntimeError when nothing is loaded or NLPAR is checked but not applied,
        ValueError when `path` is the loaded source file."""
        if self.f_orig is None:
            raise RuntimeError("no UP1/UP2 file loaded")
        if self.nlpar_chk.isChecked() and self.f_nlpar is None:
            raise RuntimeError("NLPAR is enabled but has not been applied to the full data set")
        ext = self.src_ext
        path = os.path.abspath(path)
        if os.path.splitext(path)[1].lower() != ext:
            # A different extension would change the bit depth the file is read with.
            path = os.path.splitext(path)[0] + ext
            print(f"[SAVE] output extension forced to {ext} (same bit depth as the source)")
        if os.path.abspath(path) == os.path.abspath(self.orig_path) or \
                (os.path.exists(path) and os.path.samefile(path, self.orig_path)):
            raise ValueError(f"{path}: cannot overwrite the loaded source file")
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
            # Host side of the GPU branch of _process_chunk_batch, at its peak (the final
            # np.clip(result * 255.0, 0, 255)): native read (1-2 B/px, 0 for a RAM view)
            # + the float64 argument still referenced by the caller (8) + float64 `result`
            # from cp.asnumpy(d).astype(np.float64) (8) + `result * 255.0` temporary (8)
            # + np.clip output (8) ≈ 33 B/px. Earlier moments are lower: the float32 host
            # staging of cp.asarray(chunk, float32) if CuPy casts on the host (4), and
            # cp.asnumpy's float32 copy (4) while `result` is cast. No padding here.
            CHUNK_ROWS = min(CHUNK_ROWS, self._host_chunk_rows(33))
        else:
            CHUNK_ROWS = self.nrows
        # VRAM is only half the story: the chunk also lives in host RAM as the native read
        # (1–2 B/px) + the float64 cast the batch function needs (8) + the float64 result (8)
        # + write-side uint8/uint16 conversion (~1) ≈ 18. Cap by whichever is tighter.
        # The CPU path holds more at its peak: on top of those, _cpu_pipeline_threaded's
        # float64 `out` (8), each sub-batch's `result = sub.copy()` (8 over the chunk) and
        # the np.where / gaussian_filter temporaries (~8-16) ≈ 42 B/px (2026-10-10).
        CHUNK_ROWS = min(CHUNK_ROWS, self._host_chunk_rows(18 if HAS_GPU else 42))
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
        ph_bytes = ph * pw * int(np.dtype(self.f_orig.filedatatype).itemsize)
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
            if cancel is not None and cancel():
                # Between chunks: drop the partial output (header + the chunks so far, the
                # rest a sparse hole) so nothing downstream mistakes it for a result.
                del out_obj
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"[SAVE] could not remove the partial output {path}: {e}")
                print(f"[SAVE] cancelled at row {row_start}/{n_rows_out}; partial output removed")
                self.statusBar().showMessage(f"Save cancelled at row {row_start}/{n_rows_out}; partial output removed")
                raise BatchCancelled(f"save of {path} cancelled at row {row_start}/{n_rows_out}")
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
            # In-chunk progress: patterns done in this chunk -> rows done overall + ETA.
            def chunk_progress(done, n_in_chunk, _r0=row_start, _p0=p0):
                frac_rows = _r0 + rows_to_read * (done / max(n_in_chunk, 1))
                el = time.time() - t0
                eta = el / max(frac_rows, 1e-9) * (n_rows_out - frac_rows) if frac_rows > 0 else 0.0
                self.statusBar().showMessage(
                    f"Saving: processing row {frac_rows:.0f}/{n_rows_out} "
                    f"({100.0 * frac_rows / n_rows_out:.0f}%, {el:.0f}s elapsed, ETA {eta:.0f}s)")
                QApplication.processEvents()
            chunk = self._process_chunk_batch(chunk.astype(np.float64), mask_rows, progress=chunk_progress)
            n_saved += int(mask_rows.sum())
            self.statusBar().showMessage(f"Saving: writing rows {row_start + 1}-{row_start + rows_to_read} of {n_rows_out} to disk...")
            QApplication.processEvents()
            out_obj.write_data(newpatterns=chunk, patStartCount=[p0, -1], writeHead=False)
            del chunk
            print(f"[SAVE] rows {row_start + rows_to_read}/{n_rows_out} done ({time.time() - t0:.0f}s)")
            self.statusBar().showMessage(f"Saving: row {row_start + rows_to_read}/{n_rows_out}")
            QApplication.processEvents()
            if progress is not None:
                progress(p1, n_out)
        elapsed = time.time() - t0
        n_total = n_out
        self.statusBar().showMessage(f"Saved to {os.path.basename(path)} ({elapsed:.1f}s)")
        return dict(path=path, n_saved=n_saved, n_total=n_total, seconds=elapsed, source_label=src_label)

    # ---- Montage batch (v1.7.0) ----
    # A montage is one scan split into many pattern files ("tiles"). The three methods
    # below process all tiles headlessly with the settings the user chose on a preview
    # tile: compute_shared_background (one static background for every tile, so the
    # background subtraction does not change from tile to tile), save_processed (Apply &&
    # Save without the dialog) and batch_process (load -> [NLPAR] -> save -> unload per
    # tile). Everything runs in the GUI thread and pumps the event loop, like the other
    # long passes; the caller provides `cancel` (e.g. from a button).
    def compute_shared_background(self, paths, progress=None):
        """Mean pattern (float64, (ph, pw)) over ALL patterns of ALL files in `paths`
        (.up1/.up2), streamed in chunks sized like recompute_background (never a whole
        file in RAM). The GUI state (loaded data set, plots) is not touched.

        Every file must have the pattern size and dtype of the first, else ValueError
        naming the file. The patterns a file actually holds are used: a map stopped early
        has fewer than nRows*nCols, (size - header) // (ph*pw*bytes), as in load_path.
        Unlike load_path's own mean (whole rows only), a trailing partial row is included.
        progress(done_patterns, total_patterns, path) is called per chunk.
        The result is also stored as self.shared_background (self.shared_background_key =
        tuple(paths)) for batch_process."""
        paths = list(paths)
        if not paths:
            raise ValueError("compute_shared_background: no files given")
        files = []                       # (path, data offset, n_patterns, ncols)
        ref = None
        for p in paths:
            p = str(p)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{p}: file not found")
            f = ebsd_pattern.get_pattern_file_obj(p)
            ph, pw = int(f.patternH), int(f.patternW)
            dt = np.dtype(f.filedatatype)
            hdr = int(getattr(f, 'filePos', 42) or 0)
            ncols, nrows = int(f.nCols), int(f.nRows)
            del f
            if ref is None:
                ref = (ph, pw, dt, p)
            elif (ph, pw) != ref[:2]:
                raise ValueError(f"{p}: patterns are {pw}x{ph} px, but {os.path.basename(ref[3])} "
                                 f"has {ref[1]}x{ref[0]} px; all files must have the same pattern size")
            elif dt != ref[2]:
                raise ValueError(f"{p}: patterns are {dt.name}, but {os.path.basename(ref[3])} "
                                 f"has {ref[2].name}; all files must have the same bit depth")
            n_pat = int((int(os.path.getsize(p)) - hdr) // (ph * pw * dt.itemsize))
            n_pat = max(0, min(n_pat, nrows * ncols))
            files.append((p, hdr, n_pat, max(ncols, 1)))
        ph, pw, dt = ref[0], ref[1], ref[2]
        total = sum(f[2] for f in files)
        if total == 0:
            raise ValueError("compute_shared_background: the files hold no patterns")
        # Same sizing as recompute_background: native read + slack, 3 B/px live (the float64
        # sum is a buffered reduction, it never materialises a float64 copy of the chunk).
        budget = self._host_chunk_budget()
        pat_bytes = ph * pw * dt.itemsize
        t0 = time.time()
        print(f"[SHARED BG] {len(files)} files, {total} patterns of {ph}x{pw} px ({dt.name})")
        self.statusBar().showMessage(f"Shared background: {len(files)} files, {total} patterns..."); QApplication.processEvents()
        acc = np.zeros((ph, pw), dtype=np.float64)
        done = 0
        for k, (p, hdr, n_pat, ncols) in enumerate(files):
            rows = max(1, int(budget // max(ncols * ph * pw * 3.0, 1)))
            step = rows * ncols                       # whole scan rows per chunk
            for q0 in range(0, n_pat, step):
                n = min(step, n_pat - q0)
                chunk = read_patterns_parallel(p, hdr + q0 * pat_bytes, n, ph, pw, dt)
                acc += chunk.sum(axis=0, dtype=np.float64)
                del chunk
                done += n
                self.statusBar().showMessage(
                    f"Shared background: {os.path.basename(p)} ({k + 1}/{len(files)}), "
                    f"{done}/{total} patterns")
                QApplication.processEvents()
                if progress is not None:
                    progress(done, total, p)
        mean = acc / total
        self.shared_background = mean
        self.shared_background_key = tuple(paths)
        el = time.time() - t0
        print(f"[SHARED BG] mean over {total} patterns of {len(files)} files in {el:.1f}s")
        self.statusBar().showMessage(f"Shared background: {total} patterns of {len(files)} files ({el:.1f}s)")
        return mean

    def batch_process(self, jobs, shared_background=None, progress=None, cancel=None):
        """Process several pattern files with the current widget settings, one after the
        other: `jobs` is a list of (source_path, output_path). Per job:

          load_path(source)  -> the tile's own IQ map; the IQ-range sliders (fractions of
                                each tile's IQ range) are re-applied after the load, which
                                resets them to the full range
          shared_background  -> if given, replaces the tile's mean pattern (static bg)
          NLPAR              -> if 'NLPAR: Enable Preview' was checked when the batch
                                started: full-data-set NLPAR with the current SR / lambda
                                (lambda optimized on the preview tile is reused)
          save_processed(output)   (output folder created if missing)
          unload() + gc.collect()  -> nothing of the tile stays in RAM

        A failing job (other than BatchCancelled) is printed, recorded with ok=False and
        the batch goes on (a partial output of a failed save is deleted). cancel() is
        checked between stages and between save chunks; on cancel the tile is unloaded
        and BatchCancelled re-raised, with the finished jobs in self.batch_results.
        progress(job_index, n_jobs, source_path, stage), stage in load/nlpar/save/done.
        Returns the list of result dicts (source, output, ok, error, n_saved, n_total,
        seconds, n_excluded), also kept as self.batch_results; n_excluded patterns were
        outside the tile's IQ range and are in the output unprocessed."""
        if self._nlpar_busy:
            raise RuntimeError("NLPAR is running: wait for it to finish before starting a batch")
        if self._batch_running:
            raise RuntimeError("a batch is already running")
        jobs = [(str(s), str(o)) for s, o in jobs]
        n_jobs = len(jobs)
        self.batch_results = []
        if shared_background is not None:
            shared_background = np.array(shared_background, dtype=np.float64, copy=True)
        # Settings captured once: load_path resets the IQ sliders, _apply_nlpar_pyebsdindex
        # unchecks the NLPAR box after a successful run.
        use_nlpar = self.nlpar_chk.isChecked()
        iq_lo, iq_hi = self.iq_min_slider.value(), self.iq_max_slider.value()
        sr, lam = int(self.sr_spin.value()), float(self.lam_spin.value())

        def report(i, src, stage):
            if progress is not None:
                progress(i, n_jobs, src, stage)

        def check_cancel(i, src, where):
            if cancel is not None and cancel():
                raise BatchCancelled(f"batch cancelled at job {i + 1}/{n_jobs} ({os.path.basename(src)}, {where})")

        bg_desc = ("off" if not self.static_bg_chk.isChecked() else
                   f"{self.bg_mode.currentText()}, {'shared over all tiles' if shared_background is not None else 'per tile'}")
        print(f"[BATCH] {n_jobs} jobs; static bg: {bg_desc}; dynamic bg: {self.dyn_bg_chk.isChecked()}; "
              f"per-pattern stretch: {self.normalize_chk.isChecked()}; global stretch: {self.global_chk.isChecked()}; "
              f"CLAHE: {self.clahe_chk.isChecked()}; IQ range: {iq_lo / 10:.1f}-{iq_hi / 10:.1f}% of each tile's range; "
              f"NLPAR: {use_nlpar}")
        if use_nlpar:
            print(f"[BATCH] NLPAR SR={sr}, lambda={lam:.3f}: the preview tile's lambda is reused for every tile"
                  f"{'' if self._lambda_optimized else ' (it was not optimized on the loaded tile)'}")
        t_batch = time.time()
        self._batch_running = True
        if use_nlpar:
            # Off during the loads, so each load does not run the single-pattern NLPAR
            # preview; the full-data-set NLPAR below is explicit. Restored at the end.
            self.nlpar_chk.blockSignals(True); self.nlpar_chk.setChecked(False); self.nlpar_chk.blockSignals(False)
        try:
            for i, (src, out) in enumerate(jobs):
                res = dict(source=src, output=out, ok=False, error=None, n_saved=0, n_total=0,
                           seconds=0.0, n_excluded=0)
                t0 = time.time()
                stage = None; out_abs = None
                try:
                    check_cancel(i, src, "before load")
                    stage = "load"; report(i, src, "load")
                    src_abs = os.path.abspath(src)
                    if not os.path.isfile(src_abs):
                        raise FileNotFoundError(f"{src}: source file not found")
                    ext = os.path.splitext(src_abs)[1].lower()
                    out_abs = os.path.abspath(out)
                    if os.path.splitext(out_abs)[1].lower() != ext:
                        out_abs = os.path.splitext(out_abs)[0] + ext   # as save_processed does
                    if out_abs == src_abs or (os.path.exists(out_abs) and os.path.samefile(out_abs, src_abs)):
                        raise ValueError(f"{out}: output path is the source file")
                    self.statusBar().showMessage(f"Batch {i + 1}/{n_jobs}: loading {os.path.basename(src)}")
                    QApplication.processEvents()
                    self.load_path(src_abs)
                    if self.f_orig is None or self.orig_path != src_abs:
                        raise RuntimeError(f"{src}: load failed")
                    for sl, v in ((self.iq_min_slider, iq_lo), (self.iq_max_slider, iq_hi)):
                        sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
                    self._update_iq_range()
                    if shared_background is not None:
                        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
                        if shared_background.shape != (ph, pw):
                            raise ValueError(f"{src}: patterns are {pw}x{ph} px, the shared background is "
                                             f"{shared_background.shape[1]}x{shared_background.shape[0]}")
                        self.mean_pattern = shared_background.copy()
                        self.bg_status.setText("Background: shared, all tiles")
                        # Global-stretch bounds are computed in the bg-applied domain and keyed
                        # on the mean pattern (_bg_config_key); nothing else caches it.
                        self._invalidate_global_range()
                    check_cancel(i, src, "after load")
                    if use_nlpar:
                        stage = "nlpar"; report(i, src, "nlpar")
                        # pyebsdindex's CPU NLPAR crashes the whole process (segfault, not an
                        # exception) when the scan has fewer than 2*SR+1 rows: seen with
                        # pyebsdindex 0.3.10.1 on 3-row montage tiles at SR >= 2, 4-row at SR=2
                        # (and in the GUI's own NLPAR too). Fail this job instead of the batch.
                        n_rows_nl = int(getattr(self, 'n_actual_rows', self.nrows))
                        if min(n_rows_nl, int(self.ncols)) < 2 * sr + 1:
                            raise ValueError(f"{src}: NLPAR with search radius {sr} needs at least "
                                             f"{2 * sr + 1} scan rows and columns; this file has "
                                             f"{int(self.ncols)} x {n_rows_nl}. Use SR <= "
                                             f"{max((min(n_rows_nl, int(self.ncols)) - 1) // 2, 0)}.")
                        # Sigma first, as in the GUI, where Optimize Lambda (or the preview)
                        # has always computed it before the batch: calcnlpar would otherwise
                        # estimate sigma inline, tile by tile, and give a different result.
                        self._ensure_sigma_map()
                        self._apply_nlpar_pyebsdindex(sr, lam, raise_errors=True)
                        if self.f_nlpar is None:
                            raise RuntimeError(f"{src}: NLPAR produced no result")
                        check_cancel(i, src, "after NLPAR")
                    stage = "save"; report(i, src, "save")
                    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
                    self.statusBar().showMessage(f"Batch {i + 1}/{n_jobs}: saving {os.path.basename(out_abs)}")
                    QApplication.processEvents()
                    r = self.save_processed(out_abs, cancel=cancel)
                    res.update(output=r['path'], ok=True, n_saved=int(r['n_saved']), n_total=int(r['n_total']),
                               n_excluded=int(r['n_total']) - int(r['n_saved']))
                    print(f"[BATCH] {i + 1}/{n_jobs} {os.path.basename(src)} -> {r['path']}: "
                          f"{r['n_saved']}/{r['n_total']} patterns processed ({res['n_excluded']} outside the IQ range kept unprocessed), "
                          f"source {r['source_label']}, {time.time() - t0:.1f}s")
                except BatchCancelled:
                    raise
                except Exception as e:
                    import traceback; traceback.print_exc()
                    res['error'] = f"{type(e).__name__}: {e}"
                    print(f"[BATCH] {i + 1}/{n_jobs} {src} FAILED ({stage or 'setup'}): {res['error']}")
                    if stage == "save" and out_abs and os.path.isfile(out_abs):
                        try:
                            os.remove(out_abs); print(f"[BATCH] removed the partial output {out_abs}")
                        except OSError:
                            pass
                finally:
                    if not self.unload():
                        print("[BATCH] warning: unload refused (an NLPAR call is in flight)")
                    gc.collect()
                res['seconds'] = time.time() - t0
                self.batch_results.append(res)
                report(i, src, "done")
        except BatchCancelled as e:
            print(f"[BATCH] {e}; {len(self.batch_results)} of {n_jobs} jobs finished")
            self.statusBar().showMessage(f"Batch cancelled: {len(self.batch_results)} of {n_jobs} jobs finished")
            raise
        finally:
            self._batch_running = False
            if use_nlpar:
                self.nlpar_chk.blockSignals(True); self.nlpar_chk.setChecked(True); self.nlpar_chk.blockSignals(False)
        n_ok = sum(1 for r in self.batch_results if r['ok'])
        el = time.time() - t_batch
        print(f"[BATCH] done: {n_ok}/{n_jobs} ok in {el:.1f}s")
        self.statusBar().showMessage(f"Batch done: {n_ok}/{n_jobs} files saved ({el:.0f}s)")
        return self.batch_results

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
        # Dynamic active → static and global stretch are disabled (and unchecked).
        for w, should_disable in ((self.static_bg_chk, dyn_on), (self.global_chk, dyn_on)):
            if should_disable and w.isChecked():
                w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
            w.setEnabled(not should_disable)
        # Static (or global, which implies static domain) active → dynamic disabled.
        # Re-read: the loop above may just have unchecked static because dynamic won
        # (a stale `static_on` used to end with BOTH boxes off when a script set
        # static and then dynamic; the GUI could not reach that. 2026-09-28)
        block_dyn = self.static_bg_chk.isChecked() or self.global_chk.isChecked()
        if block_dyn and self.dyn_bg_chk.isChecked():
            self.dyn_bg_chk.blockSignals(True); self.dyn_bg_chk.setChecked(False); self.dyn_bg_chk.blockSignals(False)
        self.dyn_bg_chk.setEnabled(not block_dyn)
        # Mean/range domain may have changed → invalidate cached global range.
        self._invalidate_global_range()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_lay = QHBoxLayout(central)
        # The control column sits in a horizontal splitter next to the plots (2026-10-19): drag the
        # divider to widen or narrow it. The column takes the width the splitter gives it (its
        # layout does not impose the width of its longest row), and the fields shrink with it.
        ctrl = _ControlPanel(); cl = QVBoxLayout(ctrl)
        cl.setSizeConstraint(QLayout.SetNoConstraint)
        # Wrap ctrl in a scroll area so the window's minimum height isn't tied to
        # the sum of every control's minimum size (otherwise this exceeds 1440p displays).
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(300)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ctrl_scroll.setFrameShape(0)  # no frame border (QFrame.NoFrame == 0)

        # File
        g = QGroupBox("File"); gl = QVBoxLayout(g)
        self.file_label = QLabel("No file loaded"); self.file_label.setWordWrap(True)
        btn = QPushButton("Load UP1 / UP2"); btn.clicked.connect(self.load_file)
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
        g = QGroupBox("IQ Range (patterns outside it are left unprocessed)"); gl = QVBoxLayout(g)
        g.setToolTip("Patterns outside the IQ range take no part in the processing (background mean, global stretch,\n"
                     "NLPAR sigma and neighbours) and are written to the output as they are, unprocessed,\n"
                     "so that they stay in the data set and can still be indexed.")
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
        # pyebsdindex parameters (Phase 3): saturation_protect drops near-saturated pixel pairs
        # from the distance (default on, pyebsdindex default); dthresh is subtracted from the
        # normalised dissimilarity before the exponent (default 0 = off, pyebsdindex default).
        self.sat_chk = QCheckBox("Saturation protect (pyebsdindex)"); self.sat_chk.setChecked(True)
        self.sat_chk.setToolTip("Drop pixel pairs where either value is at/above ~max intensity from the "
                                "NLPAR distance and the sigma estimate (pyebsdindex saturation_protect).")
        self.sat_chk.toggled.connect(lambda *_: self._on_nlpar_param_changed('saturation'))
        self.dthresh_spin = QDoubleSpinBox(); self.dthresh_spin.setRange(0.0, 100.0)
        self.dthresh_spin.setValue(0.0); self.dthresh_spin.setSingleStep(0.1); self.dthresh_spin.setDecimals(2)
        self.dthresh_spin.setToolTip("pyebsdindex dthresh: subtracted from the normalised dissimilarity before "
                                     "the exponent, so pairs below it count as identical. 0 = off.")
        self.dthresh_spin.valueChanged.connect(lambda *_: self._on_nlpar_param_changed('dthresh'))
        gl.addRow(self.sat_chk); gl.addRow("Distance threshold:", self.dthresh_spin)
        b1 = QPushButton("Optimize Lambda"); b1.clicked.connect(self.optimize_lambda)
        b2 = QPushButton(f"Apply NLPAR to Full Dataset ({'GPU' if NLPAR_ON_GPU else 'CPU'}, pyebsdindex)"); b2.clicked.connect(self.apply_nlpar_batch)
        b3 = QPushButton("Load Existing NLPAR UP1/UP2"); b3.clicked.connect(self.load_nlpar_file)
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
        # When OFF: absolute intensity (native bit depth) is preserved end-to-end (preview
        #   AND exported UP1/UP2). Dim patterns look dim. Use this if you want the saved
        #   patterns to reflect the actual signal level recorded by the detector.
        g = QGroupBox("Normalization"); gl = QVBoxLayout(g)
        self.normalize_chk = QCheckBox("Per-pattern stretch to [0,1]")
        self.normalize_chk.setChecked(True)
        self.normalize_chk.setToolTip(
            "ON: each pattern independently stretched to fill the display range.\n"
            "OFF: absolute intensity (native 8/16-bit scale) preserved in preview AND exported file.\n"
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
        panel_split = QSplitter(Qt.Horizontal)
        panel_split.addWidget(ctrl_scroll); panel_split.addWidget(splitter)
        panel_split.setCollapsible(0, False); panel_split.setCollapsible(1, False)
        panel_split.setStretchFactor(0, 0); panel_split.setStretchFactor(1, 1)
        panel_split.setHandleWidth(8); panel_split.setSizes([330, 1070])
        self.panel_splitter = panel_split
        main_lay.addWidget(panel_split)
        # Fields take the width their column has instead of the width of their contents, so a
        # narrow column elides a long combo text rather than pushing the column wider.
        for w in ctrl.findChildren((QComboBox, QSpinBox, QDoubleSpinBox)):
            pol = w.sizePolicy(); pol.setHorizontalPolicy(QSizePolicy.Ignored); w.setSizePolicy(pol)
            w.setMinimumWidth(60)
        for w in ctrl.findChildren(QLabel):
            w.setWordWrap(True)


if __name__ == '__main__':
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec_())
