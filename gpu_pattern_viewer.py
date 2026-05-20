import sys
import os
import glob
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
    def show_pattern(self, pat, title="", fixed_range=False):
        self.ax.clear()
        if pat is not None:
            if fixed_range: self.ax.imshow(pat, cmap='gray', vmin=0, vmax=1)
            else: self.ax.imshow(pat, cmap='gray')
        self.ax.set_title(title); self.ax.axis('off'); self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EBSD Pattern Processor (GPU Accelerated)" if HAS_GPU else "EBSD Pattern Processor (CPU Mode)"); self.setGeometry(100, 100, 1400, 900)
        self.orig_path = None; self.f_orig = self.f_nlpar = None
        self.nrows = self.ncols = self.current_row = self.current_col = 0
        self.mean_pattern = None; self.iq_map = None; self.iq_mask = None
        self.pixel_std = None  # per-pixel std across dataset, for variance-based aperture detection
        # Per-pixel noise variance σ² map for canonical NLPAR (Brewick et al. 2019, Eq. 9).
        # Computed lazily; invalidated on file load or aperture change.
        # _sigma_aperture_key stores a tuple identifying which aperture the cached map was built for.
        self.sigma_sq_map = None
        self._sigma_aperture_key = None
        self._temp_files = []
        self._build_ui()

    def closeEvent(self, event):
        for f in self._temp_files:
            try:
                if os.path.isfile(f): os.remove(f)
            except: pass
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
        pats, _ = fobj.read_data(returnArrayOnly=True, convertToFloat=True,
            patStartCount=[[c0, r0], [n_cols_read, n_rows_read]])
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

        for chunk_start in range(0, n_avail, CHUNK_ROWS):
            chunk_rows = min(CHUNK_ROWS, n_avail - chunk_start)
            read_r0 = max(0, chunk_start - sr)
            read_r1 = min(n_avail, chunk_start + chunk_rows + sr)
            data_np, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                patStartCount=[[0, read_r0], [self.ncols, read_r1 - read_r0]])
            data_np = data_np.reshape(read_r1 - read_r0, self.ncols, ph, pw).astype(np.float32)

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
        """CPU fallback σ² estimation. Slower but matches GPU output."""
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        sigma_sq_map = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        ap_flat = aperture_2d.ravel().astype(np.float64) if aperture_2d is not None else None
        n_avail = getattr(self, 'n_actual_rows', self.nrows)
        # Stream row-by-row with 1-row borders.
        CHUNK_ROWS = 8
        for chunk_start in range(0, n_avail, CHUNK_ROWS):
            chunk_rows = min(CHUNK_ROWS, n_avail - chunk_start)
            read_r0 = max(0, chunk_start - 1)
            read_r1 = min(n_avail, chunk_start + chunk_rows + 1)
            data, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                patStartCount=[[0, read_r0], [self.ncols, read_r1 - read_r0]])
            data = data.reshape(read_r1 - read_r0, self.ncols, ph, pw).astype(np.float64)
            for r in range(chunk_rows):
                gr = chunk_start + r       # global row index
                lr = r + (chunk_start - read_r0)  # local row index in the chunk
                for c in range(self.ncols):
                    center = data[lr, c].ravel()
                    min_d_raw = np.inf
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            lr2, c2 = lr + dr, c + dc
                            if lr2 < 0 or lr2 >= data.shape[0] or c2 < 0 or c2 >= self.ncols:
                                continue
                            diff = data[lr2, c2].ravel() - center
                            if ap_flat is not None:
                                d_raw = float(np.sum(diff * diff * ap_flat))
                            else:
                                d_raw = float(np.dot(diff, diff))
                            if d_raw < min_d_raw:
                                min_d_raw = d_raw
                    sigma_sq_map[gr, c] = (min_d_raw / (2.0 * N_p)) if np.isfinite(min_d_raw) else 0.0
            self.statusBar().showMessage(f"σ² map (CPU): row {chunk_start+chunk_rows}/{n_avail}")
            QApplication.processEvents()
        valid_sigmas = sigma_sq_map[sigma_sq_map > 0]
        if len(valid_sigmas) > 0:
            med = float(np.median(valid_sigmas))
            sigma_sq_map[sigma_sq_map <= 0] = med
        return sigma_sq_map

    # ---- Base pattern retrieval (on-demand) ----
    def _read_pattern(self, fobj, row=None, col=None):
        if row is None: row = self.current_row
        if col is None: col = self.current_col
        # Return zeros for patterns beyond actual data on disk
        pat_idx = row * self.ncols + col
        n_avail = getattr(self, 'n_actual_pats', self.nrows * self.ncols)
        if fobj is self.f_orig and pat_idx >= n_avail:
            return np.zeros((int(self.f_orig.patternH), int(self.f_orig.patternW)))
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
        if pat is None: return None
        p = pat.astype(np.float64)
        if self.static_bg_chk.isChecked() and self.mean_pattern is not None:
            if self.bg_mode.currentText() == "Subtract": p = p - self.mean_pattern
            else: p = p / (self.mean_pattern + 1e-10)
        if self.dyn_bg_chk.isChecked():
            p = p - gaussian_filter(p, sigma=self.dyn_sigma.value() / 10.0)
        pmin, pmax = p.min(), p.max()
        if pmax > pmin: p = (p - pmin) / (pmax - pmin)
        bval, cval = self.bright.value() / 100.0, self.cont.value() / 100.0
        p = (p - 0.5) * (1.0 + cval) + 0.5 + bval
        if self.clahe_chk.isChecked():
            p = np.clip(p, 0, 1)
            p = equalize_adapthist(p, clip_limit=self.clahe_clip.value() / 1000.0)
        return np.clip(p, 0, 1)

    # ---- Batch processing (GPU-accelerated when available) ----
    def _process_chunk_batch(self, chunk, mask_rows):
        """Process a chunk of patterns in batch. chunk: (N, ph, pw) float64.
        Returns processed chunk scaled to [0, 255] as float64."""
        N = chunk.shape[0]
        use_static = self.static_bg_chk.isChecked() and self.mean_pattern is not None
        use_dyn = self.dyn_bg_chk.isChecked()
        use_clahe = self.clahe_chk.isChecked()
        sigma = self.dyn_sigma.value() / 10.0
        bval = self.bright.value() / 100.0
        cval = self.cont.value() / 100.0
        clip_limit = self.clahe_clip.value() / 1000.0

        if HAS_GPU:
            d = cp.asarray(chunk, dtype=cp.float32)

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

            # Per-pattern normalize: compute min/max along pixel dims
            flat = d.reshape(N, -1)
            pmin = flat.min(axis=1).reshape(N, 1, 1)
            pmax = flat.max(axis=1).reshape(N, 1, 1)
            spread = pmax - pmin
            valid = spread > 0
            d = cp.where(valid, (d - pmin) / cp.where(valid, spread, cp.ones_like(spread)), d)
            del flat, pmin, pmax, spread, valid

            # Brightness / Contrast
            d = (d - 0.5) * (1.0 + cval) + 0.5 + bval

            # Transfer back to CPU
            result = cp.asnumpy(d).astype(np.float64)
            del d
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU fallback: vectorized numpy (still batched, no per-pattern loop)
            result = chunk.copy()

            if use_static:
                if self.bg_mode.currentText() == "Subtract":
                    result -= self.mean_pattern
                else:
                    result /= (self.mean_pattern + 1e-10)

            if use_dyn:
                # scipy gaussian_filter with sigma=(0, s, s) treats first axis as batch
                from scipy.ndimage import gaussian_filter as sp_gauss
                result -= sp_gauss(result, sigma=(0, sigma, sigma))

            flat = result.reshape(N, -1)
            pmin = flat.min(axis=1).reshape(N, 1, 1)
            pmax = flat.max(axis=1).reshape(N, 1, 1)
            spread = pmax - pmin
            valid = spread > 0
            result = np.where(valid, (result - pmin) / np.where(valid, spread, np.ones_like(spread)), result)

            result = (result - 0.5) * (1.0 + cval) + 0.5 + bval

        # CLAHE stays on CPU (per-pattern, no good batch GPU impl yet)
        if use_clahe:
            for i in range(N):
                if not mask_rows[i]:
                    continue
                p = np.clip(result[i], 0, 1)
                result[i] = equalize_adapthist(p, clip_limit=clip_limit)

        # Final scale to [0, 255] and zero masked patterns
        result = np.clip(result * 255.0, 0, 255)
        result[~mask_rows] = 0.0
        return result

    # ---- Display ----
    def on_pattern_clicked(self, row, col):
        self.current_row, self.current_col = row, col
        self.scan_canvas._place_marker(row, col); self.update_display()

    def update_display(self):
        if self.f_orig is None: return
        orig = self._read_pattern(self.f_orig)
        self.raw_canvas.show_pattern(orig, f"Original [{self.current_col}, {self.current_row}]")
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
        self.proc_canvas.show_pattern(processed,
            f"{src}Processed [{self.current_col}, {self.current_row}]", fixed_range=True)
        self.statusBar().showMessage(f"Pattern processed in {elapsed*1000:.0f} ms")

    # ---- File loading ----
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open UP1 File",
            r"D:\EBSD TKD Pattern Processing", "UP1 Files (*.up1)")
        if not path: return
        self.statusBar().showMessage("Loading patterns..."); QApplication.processEvents()
        self.orig_path = path; self.f_orig = ebsd_pattern.get_pattern_file_obj(path)
        self.nrows, self.ncols = self.f_orig.nRows, self.f_orig.nCols
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        # Check for truncated file but KEEP original header dimensions for OIM compat
        fsize = int(os.path.getsize(path))
        bpp = 2 if path.lower().endswith('.up2') else 1
        header_bytes = 16 if getattr(self.f_orig, 'version', 1) >= 3 else 0
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
        # Chunked loading for IQ map and mean pattern (only read rows that exist)
        CHUNK_ROWS = 20
        iq_values = []
        mean_accum = np.zeros((ph, pw), dtype=np.float64)
        sumsq_accum = np.zeros((ph, pw), dtype=np.float64)  # for per-pixel std (variance-based aperture detection)
        total_pats = 0
        for row_start in range(0, self.n_actual_rows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, self.n_actual_rows - row_start)
            chunk, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                patStartCount=[[0, row_start], [self.ncols, rows_to_read]])
            iq_values.append(np.std(chunk, axis=(1, 2)))
            mean_accum += np.sum(chunk, axis=0)
            sumsq_accum += np.sum(chunk * chunk, axis=0)
            total_pats += chunk.shape[0]
            self.statusBar().showMessage(f"Loading: row {row_start + rows_to_read}/{self.nrows}")
            QApplication.processEvents()
            del chunk
        # Pad IQ map with zeros for missing rows
        iq_flat = np.concatenate(iq_values)
        if len(iq_flat) < self.nrows * self.ncols:
            iq_flat = np.concatenate([iq_flat, np.zeros(self.nrows * self.ncols - len(iq_flat))])
        self.iq_map = iq_flat.reshape(self.nrows, self.ncols)
        self.mean_pattern = mean_accum / max(total_pats, 1)
        # Per-pixel std: sqrt(E[X^2] - E[X]^2). max(.,0) guards against tiny negative values from FP precision.
        mean_sq = sumsq_accum / max(total_pats, 1)
        self.pixel_std = np.sqrt(np.maximum(mean_sq - self.mean_pattern ** 2, 0.0))
        self.file_label.setText(f"{os.path.basename(path)}\n{self.ncols} x {self.nrows} = {self.ncols*self.nrows} patterns")
        self.bg_status.setText("Background: all patterns")
        self.scan_canvas.set_map(self.iq_map, self.nrows, self.ncols)
        self.iq_mask = np.ones((self.nrows, self.ncols), dtype=bool)
        self.iq_min_slider.setValue(0); self.iq_max_slider.setValue(1000); self._update_iq_range()
        self.f_nlpar = None; self.nlpar_status.setText("No NLPAR file loaded")
        self._invalidate_sigma_map()  # New file → σ² map must be recomputed
        self.on_pattern_clicked(0, 0); self.statusBar().showMessage("Ready")

    def load_nlpar_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NLPAR UP1 File",
            r"D:\EBSD TKD Pattern Processing", "UP1 Files (*.up1)")
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
        CHUNK_ROWS = 50
        for row_start in range(0, n_avail_rows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, n_avail_rows - row_start)
            chunk, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                patStartCount=[[0, row_start], [self.ncols, rows_to_read]])
            mask_flat = mask[row_start:row_start+rows_to_read, :].flatten()
            if mask_flat.any(): mean_accum += np.sum(chunk[mask_flat], axis=0)
            del chunk
            self.statusBar().showMessage(f"Recomputing BG: row {row_start + rows_to_read}/{self.nrows}")
            QApplication.processEvents()
        self.mean_pattern = mean_accum / n_valid
        n_total = self.nrows * self.ncols
        self.bg_status.setText(f"Background: {n_valid}/{n_total} patterns")
        self.statusBar().showMessage(f"Background recomputed from {n_valid} patterns"); self.update_display()

    def _write_masked_up1(self):
        mask = self._get_iq_mask()
        if mask.all(): return self.orig_path
        self.statusBar().showMessage("Writing masked temp UP1..."); QApplication.processEvents()
        tmp_dir = os.path.dirname(self.orig_path)
        tmp_path = os.path.join(tmp_dir, "_temp_masked.up1")
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        CHUNK_ROWS = 50
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        ph, pw = int(self.f_orig.patternH), int(self.f_orig.patternW)
        first_chunk = True
        for row_start in range(0, self.nrows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, self.nrows - row_start)
            if row_start >= n_avail_rows:
                chunk = np.zeros((rows_to_read * self.ncols, ph, pw), dtype=np.float64)
            elif row_start + rows_to_read > n_avail_rows:
                real_rows = n_avail_rows - row_start
                real_chunk, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                    patStartCount=[[0, row_start], [self.ncols, real_rows]])
                pad = np.zeros(((rows_to_read - real_rows) * self.ncols, ph, pw), dtype=np.float64)
                chunk = np.concatenate([real_chunk, pad], axis=0)
                del real_chunk, pad
            else:
                chunk, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                    patStartCount=[[0, row_start], [self.ncols, rows_to_read]])
            mask_flat = mask[row_start:row_start+rows_to_read, :].flatten()
            chunk[~mask_flat] = 0.0
            if first_chunk:
                tmp_obj = ebsd_pattern.get_pattern_file_obj(tmp_path)
                tmp_obj.patternW = self.f_orig.patternW; tmp_obj.patternH = self.f_orig.patternH
                tmp_obj.nCols = self.ncols; tmp_obj.nRows = self.nrows
                tmp_obj.nPatterns = self.ncols * self.nrows
                tmp_obj.xStep = self.f_orig.xStep; tmp_obj.yStep = self.f_orig.yStep
                tmp_obj.hexflag = self.f_orig.hexflag
                tmp_obj.extraPatterns = self.f_orig.extraPatterns; tmp_obj.version = self.f_orig.version
                tmp_obj.filePos = self.f_orig.filePos
                tmp_obj.write_data(newpatterns=chunk, patStartCount=[0, -1], writeHead=True)
                # Copy extra header bytes (v4+ has 16 bytes between standard 42-byte header and filePos)
                if self.f_orig.filePos > 42:
                    with open(self.orig_path, 'rb') as f_src:
                        f_src.seek(42)
                        extra_hdr = f_src.read(self.f_orig.filePos - 42)
                    with open(tmp_path, 'r+b') as f_dst:
                        f_dst.seek(42)
                        f_dst.write(extra_hdr)
                first_chunk = False
            else:
                tmp_obj.write_data(newpatterns=chunk, patStartCount=[row_start * self.ncols, -1], writeHead=False)
            del chunk
            self.statusBar().showMessage(f"Writing masked UP1: row {row_start + rows_to_read}/{self.nrows}")
            QApplication.processEvents()
        self._temp_files.append(tmp_path)
        return tmp_path

    def optimize_lambda(self):
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        tmp_path = self._write_masked_up1()
        self.statusBar().showMessage("Optimizing lambda on masked data..."); QApplication.processEvents()
        nlobj = nlpar.NLPAR(tmp_path, searchradius=self.sr_spin.value())
        nlobj.opt_lambda(chunksize=0, automask=True, autoupdate=True, backsub=False)
        self.lam_spin.setValue(nlobj.lam)
        self.statusBar().showMessage(f"Optimal lambda: {nlobj.lam:.2f}")

    def apply_nlpar_batch(self):
        """Run NLPAR on the full dataset. Uses GPU shift-and-compare when available, else pyebsdindex CPU."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        sr, lam = self.sr_spin.value(), self.lam_spin.value()

        if HAS_GPU:
            self._apply_nlpar_gpu(sr, lam)
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
        out_obj = ebsd_pattern.get_pattern_file_obj(out_path)
        out_obj.patternW = pw; out_obj.patternH = ph
        out_obj.nCols = self.ncols; out_obj.nRows = self.nrows
        out_obj.nPatterns = self.ncols * self.nrows
        out_obj.xStep = self.f_orig.xStep; out_obj.yStep = self.f_orig.yStep
        out_obj.hexflag = self.f_orig.hexflag
        out_obj.extraPatterns = self.f_orig.extraPatterns
        out_obj.version = self.f_orig.version
        out_obj.filePos = self.f_orig.filePos

        # Adaptive chunk size based on VRAM and pattern dimensions
        pat_bytes = ph * pw * 4  # float32
        # Memory per chunk_row: ~5x (padded + weighted_sum + temporaries) × (ncols + 2*sr) × pat_bytes
        mem_per_row = 5 * (self.ncols + 2 * sr) * pat_bytes
        vram_budget = int(cp.cuda.Device(0).mem_info[1] * 0.6)  # use 60% of total VRAM
        CHUNK_ROWS = max(2, vram_budget // mem_per_row)
        CHUNK_ROWS = min(CHUNK_ROWS, self.nrows)
        print(f"[GPU NLPAR] Pattern: {ph}x{pw}, VRAM budget: {vram_budget/1e9:.1f} GB, CHUNK_ROWS: {CHUNK_ROWS}")
        first_write = True

        for chunk_start in range(0, self.nrows, CHUNK_ROWS):
            chunk_rows = min(CHUNK_ROWS, self.nrows - chunk_start)

            # Read range with border (clamped to available data)
            read_r0 = max(0, chunk_start - sr)
            read_r1 = min(n_avail, chunk_start + chunk_rows + sr)
            # Also cap to nrows for mask indexing
            mask_r1 = min(self.nrows, chunk_start + chunk_rows + sr)

            if read_r1 > read_r0:
                data_np, _ = self.f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                    patStartCount=[[0, read_r0], [self.ncols, read_r1 - read_r0]])
                data_np = data_np.reshape(read_r1 - read_r0, self.ncols, ph, pw).astype(np.float32)
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
            if first_write:
                out_obj.write_data(newpatterns=result_np, patStartCount=[0, -1], writeHead=True)
                if self.f_orig.filePos > 42:
                    with open(self.orig_path, 'rb') as f_src:
                        f_src.seek(42)
                        extra_hdr = f_src.read(self.f_orig.filePos - 42)
                    with open(out_path, 'r+b') as f_dst:
                        f_dst.seek(42)
                        f_dst.write(extra_hdr)
                first_write = False
            else:
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
        """CPU fallback: run NLPAR via pyebsdindex."""
        tmp_path = self._write_masked_up1()
        self.statusBar().showMessage(f"Running NLPAR batch CPU (SR={sr}, lam={lam:.2f})..."); QApplication.processEvents()
        t0 = time.time()
        nlobj = nlpar.NLPAR(tmp_path, lam=lam, searchradius=sr)
        nlobj.calcnlpar(chunksize=0, saturation_protect=True, automask=True, backsub=False)
        elapsed = time.time() - t0
        base = os.path.splitext(tmp_path)[0]
        candidates = sorted(glob.glob(f"{base}_NLPAR*.up1"), key=os.path.getmtime, reverse=True)
        if candidates:
            self._temp_files.append(candidates[0])
            self.f_nlpar = ebsd_pattern.get_pattern_file_obj(candidates[0])
            self.nlpar_chk.setChecked(False)
            self.nlpar_status.setText(f"Batch: SR={sr}, lam={lam:.2f} ({elapsed:.1f}s)")
            self.statusBar().showMessage("NLPAR batch complete")
        else:
            QMessageBox.warning(self, "Warning", "NLPAR ran but output not found.")
        self.update_display()

    def apply_and_save(self):
        """Save all processed patterns to a new UP1 file (chunked)."""
        if self.f_orig is None: QMessageBox.warning(self, "Error", "Load a UP1 file first"); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Patterns",
            os.path.dirname(self.orig_path), "UP1 Files (*.up1)")
        if not path: return
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
            CHUNK_ROWS = 20
        n_avail_rows = getattr(self, 'n_actual_rows', self.nrows)
        first_chunk = True; out_obj = None; n_saved = 0
        for row_start in range(0, self.nrows, CHUNK_ROWS):
            rows_to_read = min(CHUNK_ROWS, self.nrows - row_start)
            # Read real data or generate zeros for missing rows
            if row_start >= n_avail_rows:
                chunk = np.zeros((rows_to_read * self.ncols, ph, pw), dtype=np.float64)
            elif row_start + rows_to_read > n_avail_rows:
                real_rows = n_avail_rows - row_start
                real_chunk, _ = src_fobj.read_data(returnArrayOnly=True, convertToFloat=True,
                    patStartCount=[[0, row_start], [self.ncols, real_rows]])
                pad_count = (rows_to_read - real_rows) * self.ncols
                pad = np.zeros((pad_count, ph, pw), dtype=np.float64)
                chunk = np.concatenate([real_chunk, pad], axis=0)
                del real_chunk, pad
            else:
                chunk, _ = src_fobj.read_data(returnArrayOnly=True, convertToFloat=True,
                    patStartCount=[[0, row_start], [self.ncols, rows_to_read]])
            mask_rows = mask[row_start:row_start+rows_to_read, :].flatten()
            # Batch process entire chunk (GPU-accelerated if available)
            chunk = self._process_chunk_batch(chunk.astype(np.float64), mask_rows)
            n_saved += int(mask_rows.sum())
            # Write chunk
            if first_chunk:
                out_obj = ebsd_pattern.get_pattern_file_obj(path)
                out_obj.patternW = pw; out_obj.patternH = ph
                out_obj.nCols = self.ncols; out_obj.nRows = self.nrows
                out_obj.nPatterns = self.ncols * self.nrows
                out_obj.xStep = self.f_orig.xStep; out_obj.yStep = self.f_orig.yStep
                out_obj.hexflag = self.f_orig.hexflag
                out_obj.extraPatterns = self.f_orig.extraPatterns; out_obj.version = self.f_orig.version
                out_obj.filePos = self.f_orig.filePos
                out_obj.write_data(newpatterns=chunk, patStartCount=[0, -1], writeHead=True)
                # Copy extra header bytes (v4+ has 16 bytes between standard 42-byte header and filePos)
                if self.f_orig.filePos > 42:
                    with open(self.orig_path, 'rb') as f_src:
                        f_src.seek(42)
                        extra_hdr = f_src.read(self.f_orig.filePos - 42)
                    with open(path, 'r+b') as f_dst:
                        f_dst.seek(42)
                        f_dst.write(extra_hdr)
                first_chunk = False
            else:
                out_obj.write_data(newpatterns=chunk, patStartCount=[row_start * self.ncols, -1], writeHead=False)
            del chunk
            self.statusBar().showMessage(f"Saving: row {row_start + rows_to_read}/{self.nrows}")
            QApplication.processEvents()
        elapsed = time.time() - t0
        n_total = self.nrows * self.ncols
        self.statusBar().showMessage(f"Saved to {os.path.basename(path)} ({elapsed:.1f}s)")
        QMessageBox.information(self, "Saved",
            f"Processed patterns saved to:\n{path}\n\n"
            f"Source: {src_label}\nMode: {'GPU' if HAS_GPU else 'CPU'}\nPatterns: {n_saved}/{n_total}\nTime: {elapsed:.1f}s")

    def _update_iq_range(self):
        if self.iq_map is None: return
        self.iq_mask = self._get_iq_mask()
        iq_lo, iq_hi = self.iq_map.min(), self.iq_map.max()
        vmin = iq_lo + (iq_hi - iq_lo) * self.iq_min_slider.value() / 1000.0
        vmax = iq_lo + (iq_hi - iq_lo) * self.iq_max_slider.value() / 1000.0
        self.iq_min_lbl.setText(f"Min IQ: {vmin:.1f}"); self.iq_max_lbl.setText(f"Max IQ: {vmax:.1f}")
        self.scan_canvas.update_mask(self.iq_mask); self.update_display()

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
        # Aperture mask: None / Circular (EBSD) / Auto-detect (variance-based, e.g. RKD)
        self.aperture_mode = QComboBox()
        self.aperture_mode.addItems(["None", "Circular (EBSD)", "Auto-detect"])
        self.aperture_mode.setCurrentIndex(0)  # default None
        self.aperture_mode.setMinimumContentsLength(15)  # ensure "Circular (EBSD)" fits without ellipsis
        # Aperture changes N_p → invalidate σ² map so it recomputes for the new aperture.
        self.aperture_mode.currentIndexChanged.connect(lambda *_: self._invalidate_sigma_map())
        gl.addRow("Aperture mask:", self.aperture_mode)
        self.aperture_thresh_spin = QDoubleSpinBox()
        self.aperture_thresh_spin.setRange(0.001, 1.0)
        self.aperture_thresh_spin.setValue(0.05)
        self.aperture_thresh_spin.setSingleStep(0.01)
        self.aperture_thresh_spin.setDecimals(3)
        self.aperture_thresh_spin.setToolTip("Auto-detect threshold as fraction of 95th-percentile pixel std")
        self.aperture_thresh_spin.valueChanged.connect(lambda *_: self._invalidate_sigma_map())
        gl.addRow("Threshold:", self.aperture_thresh_spin)
        b_show_mask = QPushButton("Show detected mask")
        b_show_mask.clicked.connect(self.show_aperture_mask)
        gl.addRow(b_show_mask)
        b1 = QPushButton("Optimize Lambda"); b1.clicked.connect(self.optimize_lambda)
        b2 = QPushButton("Apply NLPAR to Full Dataset (GPU)" if HAS_GPU else "Apply NLPAR to Full Dataset (CPU)"); b2.clicked.connect(self.apply_nlpar_batch)
        b3 = QPushButton("Load Existing NLPAR UP1"); b3.clicked.connect(self.load_nlpar_file)
        gl.addRow(b1); gl.addRow(b2); gl.addRow(b3)
        self.nlpar_status = QLabel("No NLPAR file loaded"); gl.addRow(self.nlpar_status); cl.addWidget(g)

        # Static BG
        g = QGroupBox("Static Background"); gl = QVBoxLayout(g)
        self.static_bg_chk = QCheckBox("Enable"); self.static_bg_chk.stateChanged.connect(self.update_display)
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
        self.dyn_sigma_lbl = QLabel("Sigma: 5.0")
        self.dyn_sigma = QSlider(Qt.Horizontal); self.dyn_sigma.setRange(1, 200); self.dyn_sigma.setValue(50)
        self.dyn_sigma.valueChanged.connect(self.update_display)
        self.dyn_sigma.valueChanged.connect(lambda v: self.dyn_sigma_lbl.setText(f"Sigma: {v/10:.1f}"))
        gl.addWidget(self.dyn_bg_chk); gl.addWidget(self.dyn_sigma_lbl); gl.addWidget(self.dyn_sigma); cl.addWidget(g)

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
