# views/slice_viewer.py
"""
BFieldSliceWidget — Professional B-field cross-section viewer.

Architecture
------------
1. Background worker computes a uniform 3-D |B| grid (xs, ys, zs, B_vol).
2. A RegularGridInterpolator wraps the grid for instant 2-D slice extraction
   at any axis (PCA, X, Y, Z) and any position.
3. matplotlib backend : imshow on persistent axes — slider changes call
   im.set_data() only, no axes rebuild → smooth, low-latency scrubbing.
   Background pre-render thread caches all frames for the current axis so
   playback is pure array lookup + draw_idle().
4. PyVista backend   : pv.ImageData + .slice() — updated in-place.

Slider resolution : 1 000 steps (continuous scrubbing).
Pre-render frames : 400 per axis.
Interpolation res : 200 × 200 pixels per slice.
"""

import numpy as np
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QProgressBar, QSizePolicy,
    QComboBox, QSpinBox, QFileDialog, QCheckBox,
)
from gui.gui_utils import THEME, WatermarkedCanvas, PlotToolbar

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

_STEPS      = 1000   # slider granularity
_INTERP_RES = 200    # 2-D heat-map grid resolution (pixels per axis)
_N_PRERENDER = 400   # frames cached per axis in background


# ─────────────────────────────────────────────────────────────────────────────
# 3-D grid interpolator — scipy when available, pure-NumPy fallback
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.interpolate import RegularGridInterpolator as _SciRGI

    def _make_interp(xs, ys, zs, values):
        return _SciRGI((xs, ys, zs), values,
                       method='linear', bounds_error=False, fill_value=0.0)
except ImportError:
    def _make_interp(xs, ys, zs, values):
        return _TrilinearInterp(xs, ys, zs, values)


class _TrilinearInterp:
    """Pure-NumPy trilinear interpolation on a uniform 3-D grid."""

    def __init__(self, xs, ys, zs, values):
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.zs = np.asarray(zs, dtype=np.float64)
        self.v  = np.asarray(values, dtype=np.float64)
        self.nx = len(xs)
        self.ny = len(ys)
        self.nz = len(zs)

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)

        def _fidx(coord, axis):
            lo, hi, n = axis[0], axis[-1], len(axis)
            d = (hi - lo) / max(n - 1, 1)
            return np.clip((coord - lo) / d, 0.0, n - 1 - 1e-9)

        fx = _fidx(pts[:, 0], self.xs)
        fy = _fidx(pts[:, 1], self.ys)
        fz = _fidx(pts[:, 2], self.zs)

        ix = np.clip(np.floor(fx).astype(np.int32), 0, self.nx - 2)
        iy = np.clip(np.floor(fy).astype(np.int32), 0, self.ny - 2)
        iz = np.clip(np.floor(fz).astype(np.int32), 0, self.nz - 2)
        dx, dy, dz = fx - ix, fy - iy, fz - iz

        v = self.v
        return (v[ix,   iy,   iz  ] * (1-dx) * (1-dy) * (1-dz) +
                v[ix+1, iy,   iz  ] * dx     * (1-dy) * (1-dz) +
                v[ix,   iy+1, iz  ] * (1-dx) * dy     * (1-dz) +
                v[ix+1, iy+1, iz  ] * dx     * dy     * (1-dz) +
                v[ix,   iy,   iz+1] * (1-dx) * (1-dy) * dz     +
                v[ix+1, iy,   iz+1] * dx     * (1-dy) * dz     +
                v[ix,   iy+1, iz+1] * (1-dx) * dy     * dz     +
                v[ix+1, iy+1, iz+1] * dx     * dy     * dz     )


# ─────────────────────────────────────────────────────────────────────────────
# Volume computation worker
# ─────────────────────────────────────────────────────────────────────────────

class _VolumeWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, engine, n_vox: int):
        super().__init__()
        self.engine = engine
        self.n_vox  = n_vox

    @pyqtSlot()
    def run(self):
        data = self.engine.compute_bfield_volume(
            n_vox=self.n_vox,
            progress_callback=self.progress.emit,
        )
        self.finished.emit(data)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-render worker  (reads self._viewer — all data is read-only, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class _PreRenderWorker(QObject):
    finished = pyqtSignal(str, object)   # (axis_name, list[np.ndarray])

    def __init__(self, viewer: 'BFieldSliceWidget', axis_name: str):
        super().__init__()
        self._viewer = viewer
        self._axis   = axis_name
        self._abort  = False

    def abort(self):
        self._abort = True

    @pyqtSlot()
    def run(self):
        lo, hi = self._viewer._axis_range(self._axis)
        n      = _N_PRERENDER
        frames = []
        for i in range(n):
            if self._abort:
                return
            pos   = lo + (i / max(n - 1, 1)) * (hi - lo)
            B_2d, _, _, _ = self._viewer._compute_slice_2d(self._axis, pos)
            frames.append(B_2d)
        self.finished.emit(self._axis, frames)


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class BFieldSliceWidget(QWidget):
    """
    Interactive B-field cross-section viewer.

    Workflow
    --------
    1. User clicks "Compute" → 3-D |B| volume evaluated once in background.
    2. Viewer appears with a 2-D heat map (imshow).  Slider scrubs the slice
       position; axis combo changes the cutting plane.
    3. Background pre-render caches all frames → subsequent playback is
       instant (pure im.set_data() + draw_idle).
    """

    def __init__(self, engine, n_vox: int = 48, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._n_vox  = n_vox
        self._data   = None
        self._interp = None
        self._thread = None

        # PCA geometry (filled after computation)
        self._pca_a  = None
        self._pca_e1 = None
        self._pca_e2 = None
        self._pca_R  = 0.0
        self._pca_lo = 0.0
        self._pca_hi = 0.0

        # Pre-render cache
        self._prerendered: dict = {}
        self._pr_thread  = None
        self._pr_worker  = None

        # matplotlib state
        self._mpl_fig    = None
        self._mpl_canvas = None
        self._mpl_im     = None
        self._mpl_ax     = None
        self._mpl_coil   = None
        self._mpl_axis_cb = None
        self._mpl_slider  = None
        self._mpl_pos_lbl = None
        self._mpl_chk_coil = None
        self._current_axis = "PCA Axis"

        self._build_loading_ui()

    # ── Loading state ─────────────────────────────────────────────────────────

    def _build_loading_ui(self):
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(20, 20, 20, 20)
        self._outer.setSpacing(12)

        desc = QLabel(
            "Computes a 3-D |B| volume once, then renders cross-sectional heat maps "
            "at any axis or position in real time."
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{THEME['text_dim']}; font-size:9pt;")

        gsz_row = QHBoxLayout()
        gsz_lbl = QLabel("Grid resolution (pts/axis):")
        gsz_lbl.setStyleSheet(f"color:{THEME['text_dim']};")
        self._nvox_spin = QSpinBox()
        self._nvox_spin.setRange(20, 120)
        self._nvox_spin.setSingleStep(8)
        self._nvox_spin.setValue(self._n_vox)
        self._nvox_spin.setFixedWidth(72)
        gsz_row.addStretch()
        gsz_row.addWidget(gsz_lbl)
        gsz_row.addWidget(self._nvox_spin)
        gsz_row.addStretch()

        self._load_btn = QPushButton("▶  Compute B-Field Volume")
        self._load_btn.setObjectName("GenerateButton")
        self._load_btn.setFixedWidth(280)
        self._load_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._load_btn.clicked.connect(self._start_computation)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._load_btn)
        btn_row.addStretch()

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.hide()

        self._prog_lbl = QLabel("")
        self._prog_lbl.setAlignment(Qt.AlignCenter)
        self._prog_lbl.setStyleSheet(f"color:{THEME['text_dim']};")
        self._prog_lbl.hide()

        self._outer.addStretch()
        self._outer.addWidget(desc)
        self._outer.addSpacing(8)
        self._outer.addLayout(gsz_row)
        self._outer.addLayout(btn_row)
        self._outer.addWidget(self._prog_bar)
        self._outer.addWidget(self._prog_lbl)
        self._outer.addStretch()

    # ── Volume computation ────────────────────────────────────────────────────

    def _start_computation(self):
        self._n_vox = self._nvox_spin.value()
        self._load_btn.setEnabled(False)
        self._load_btn.setText("Computing…")
        self._prog_bar.show()
        self._prog_lbl.show()
        N = self._n_vox
        self._prog_lbl.setText(
            f"Evaluating |B| at {N}³ = {N**3:,} grid points…"
        )

        self._thread = QThread(self)
        self._worker = _VolumeWorker(self._engine, self._n_vox)
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._prog_bar.setValue)
        self._worker.finished.connect(self._on_computation_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _on_computation_done(self, data):
        self._data   = data
        xs, ys, zs, B_vol = data

        self._interp = _make_interp(xs, ys, zs, B_vol)

        # PCA geometry
        a = self._engine.axis / np.linalg.norm(self._engine.axis)
        self._pca_a = a
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, helper); e1 /= np.linalg.norm(e1); e1 = -e1
        self._pca_e1 = e1
        self._pca_e2 = np.cross(a, e1)

        offsets    = self._engine.midpoints - self._engine.mean_point
        radial     = offsets - (offsets @ a)[:, None] * a
        self._pca_R = float(np.linalg.norm(radial, axis=1).max()) * 1.2
        pca_projs  = offsets @ a
        pad = max(float(np.abs(pca_projs).max()) * 0.15, self._pca_R * 0.1)
        self._pca_lo = float(pca_projs.min()) - pad
        self._pca_hi = float(pca_projs.max()) + pad

        # Colour limits
        B_flat   = B_vol.ravel()
        pos_vals = B_flat[B_flat > 0]
        self._vmin = float(pos_vals.min()) if pos_vals.size else 0.0
        self._vmax = float(B_flat.max())

        # Clear loading layout
        for w in (self._load_btn, self._prog_bar, self._prog_lbl, self._nvox_spin):
            w.setParent(None)
        while self._outer.count():
            self._outer.takeAt(0)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        if HAS_PYVISTA:
            self._build_pyvista_viewer()
        else:
            self._build_matplotlib_viewer()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _axis_range(self, axis_name: str):
        xs, ys, zs, _ = self._data
        if axis_name == "PCA Axis":
            return self._pca_lo, self._pca_hi
        elif axis_name == "X Axis":
            return float(xs[0]), float(xs[-1])
        elif axis_name == "Y Axis":
            return float(ys[0]), float(ys[-1])
        else:
            return float(zs[0]), float(zs[-1])

    def _slider_to_pos(self, val: int, axis_name: str) -> float:
        lo, hi = self._axis_range(axis_name)
        return lo + (val / (_STEPS - 1)) * (hi - lo)

    def _compute_slice_2d(self, axis_name: str, pos: float):
        """
        Extract a 2-D |B| slice.
        Returns: (B_2d [res×res float64], extent, xlabel, ylabel)
        B_2d[i,j] is the field at (u=u_i, v=v_j).
        imshow should use B_2d.T with origin='lower'.
        """
        xs, ys, zs, _ = self._data
        res = _INTERP_RES

        if axis_name == "PCA Axis":
            a, e1, e2 = self._pca_a, self._pca_e1, self._pca_e2
            R      = self._pca_R
            center = self._engine.mean_point + pos * a
            lin    = np.linspace(-R, R, res)
            U, V   = np.meshgrid(lin, lin, indexing='ij')
            X3d = center[0] + U*e1[0] + V*e2[0]
            Y3d = center[1] + U*e1[1] + V*e2[1]
            Z3d = center[2] + U*e1[2] + V*e2[2]
            pts    = np.column_stack([X3d.ravel(), Y3d.ravel(), Z3d.ravel()])
            B_raw  = self._interp(pts).reshape(res, res)
            mask   = (U**2 + V**2) > R**2
            B_2d   = np.where(mask, np.nan, B_raw)
            extent = [-R, R, -R, R]
            xl, yl = 'e\u2081 (m)', 'e\u2082 (m)'

        elif axis_name == "X Axis":
            lin_y = np.linspace(float(ys[0]), float(ys[-1]), res)
            lin_z = np.linspace(float(zs[0]), float(zs[-1]), res)
            U, V  = np.meshgrid(lin_y, lin_z, indexing='ij')
            pts   = np.column_stack([np.full(U.size, pos), U.ravel(), V.ravel()])
            B_2d  = self._interp(pts).reshape(res, res)
            extent = [float(ys[0]), float(ys[-1]), float(zs[0]), float(zs[-1])]
            xl, yl = 'Y (m)', 'Z (m)'

        elif axis_name == "Y Axis":
            lin_x = np.linspace(float(xs[0]), float(xs[-1]), res)
            lin_z = np.linspace(float(zs[0]), float(zs[-1]), res)
            U, V  = np.meshgrid(lin_x, lin_z, indexing='ij')
            pts   = np.column_stack([U.ravel(), np.full(U.size, pos), V.ravel()])
            B_2d  = self._interp(pts).reshape(res, res)
            extent = [float(xs[0]), float(xs[-1]), float(zs[0]), float(zs[-1])]
            xl, yl = 'X (m)', 'Z (m)'

        else:   # Z Axis
            lin_x = np.linspace(float(xs[0]), float(xs[-1]), res)
            lin_y = np.linspace(float(ys[0]), float(ys[-1]), res)
            U, V  = np.meshgrid(lin_x, lin_y, indexing='ij')
            pts   = np.column_stack([U.ravel(), V.ravel(), np.full(U.size, pos)])
            B_2d  = self._interp(pts).reshape(res, res)
            extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
            xl, yl = 'X (m)', 'Y (m)'

        return B_2d, extent, xl, yl

    def _coil_projection(self, axis_name: str):
        """Return (u_coords, v_coords) for coil projected onto the slice plane."""
        c  = self._engine.coords
        mp = self._engine.mean_point
        if axis_name == "PCA Axis":
            off = c - mp
            u = off @ self._pca_e1
            v = off @ self._pca_e2
        elif axis_name == "X Axis":
            u, v = c[:, 1], c[:, 2]
        elif axis_name == "Y Axis":
            u, v = c[:, 0], c[:, 2]
        else:
            u, v = c[:, 0], c[:, 1]
        return u, v

    # ─────────────────────────────────────────────────────────────────────────
    # Controls bar (shared between backends)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_controls(self, save_cb, reset_cb=None):
        """Build axis selector + slider row; inserts into self._outer."""
        ctrl = QWidget()
        ctrl.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
        )
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(12)

        def _lbl(txt):
            l = QLabel(txt)
            l.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8pt;")
            return l

        axis_cb = QComboBox()
        axis_cb.addItems(["PCA Axis", "X Axis", "Y Axis", "Z Axis"])
        axis_cb.setFixedWidth(115)

        pos_lbl = QLabel("pos = —")
        pos_lbl.setStyleSheet(
            f"color:{THEME['accent']}; font-size:9pt; font-weight:bold;"
        )

        cl.addWidget(_lbl("Slice axis:"))
        cl.addWidget(axis_cb)
        cl.addStretch()
        cl.addWidget(pos_lbl)
        cl.addStretch()

        if reset_cb:
            btn_reset = QPushButton("⟳ Reset View")
            btn_reset.clicked.connect(reset_cb)
            cl.addWidget(btn_reset)

        btn_save = QPushButton("💾 Save Image")
        btn_save.clicked.connect(save_cb)
        cl.addWidget(btn_save)

        sl_widget = QWidget()
        sl_lay = QHBoxLayout(sl_widget)
        sl_lay.setContentsMargins(10, 4, 10, 4)
        sl_lay.setSpacing(8)

        lo_lbl = QLabel("←")
        lo_lbl.setStyleSheet(f"color:{THEME['text_dim']};")
        hi_lbl = QLabel("→")
        hi_lbl.setStyleSheet(f"color:{THEME['text_dim']};")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, _STEPS - 1)
        slider.setValue(_STEPS // 2)

        sl_lay.addWidget(lo_lbl)
        sl_lay.addWidget(slider, stretch=1)
        sl_lay.addWidget(hi_lbl)

        self._outer.addWidget(ctrl)
        self._outer.addWidget(sl_widget)

        return axis_cb, slider, pos_lbl

    # ─────────────────────────────────────────────────────────────────────────
    # PyVista viewer
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pyvista_viewer(self):
        xs, ys, zs, B_vol = self._data

        grid = pv.ImageData()
        grid.origin     = (float(xs[0]), float(ys[0]), float(zs[0]))
        grid.spacing    = (float(xs[1]-xs[0]), float(ys[1]-ys[0]), float(zs[1]-zs[0]))
        grid.dimensions = (len(xs), len(ys), len(zs))
        grid.point_data['|B| (T)'] = B_vol.ravel(order='F').astype(np.float32)
        self._pv_grid = grid

        # Controls row + coil toggle
        self._pv_axis_cb, self._pv_slider, self._pv_pos_lbl = \
            self._build_controls(self._pv_save, self._pv_reset_view)

        # Coil toggle (inserted into controls)
        coil_ctrl = QWidget()
        coil_ctrl.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
        )
        cc = QHBoxLayout(coil_ctrl)
        cc.setContentsMargins(10, 4, 10, 4)
        chk = QCheckBox("Show coil")
        chk.setChecked(True)
        chk.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        chk.toggled.connect(self._pv_set_coil_visible)
        cc.addWidget(chk)
        cc.addStretch()
        self._outer.addWidget(coil_ctrl)

        self._plotter = QtInteractor(self, auto_update=False)
        self._plotter.set_background(THEME['bg'])
        self._outer.addWidget(self._plotter.interactor, stretch=1)

        # Coil tube
        coords = self._engine.coords
        n_pts  = len(coords)
        cells  = np.hstack([[2, i, i+1] for i in range(n_pts-1)]).astype(np.int_)
        coil   = pv.PolyData(coords.astype(np.float32))
        coil.lines = cells
        self._pv_coil_actor = self._plotter.add_mesh(
            coil, color=THEME['accent'], line_width=3,
            render_lines_as_tubes=True, name='coil',
        )

        self._plotter.camera_position = 'iso'
        self._plotter.show()

        self._pv_update_slice()

        self._pv_slider.valueChanged.connect(self._pv_update_slice)
        self._pv_axis_cb.currentIndexChanged.connect(self._pv_on_axis_changed)

    def _pv_on_axis_changed(self):
        self._pv_slider.setValue(_STEPS // 2)
        self._pv_update_slice()

    def _pv_update_slice(self):
        axis = self._pv_axis_cb.currentText()
        pos  = self._slider_to_pos(self._pv_slider.value(), axis)
        mp   = self._engine.mean_point

        if axis == "PCA Axis":
            normal = self._pca_a.tolist()
            origin = (mp + pos * self._pca_a).tolist()
        elif axis == "X Axis":
            normal = [1, 0, 0]
            origin = [pos, float(mp[1]), float(mp[2])]
        elif axis == "Y Axis":
            normal = [0, 1, 0]
            origin = [float(mp[0]), pos, float(mp[2])]
        else:
            normal = [0, 0, 1]
            origin = [float(mp[0]), float(mp[1]), pos]

        sliced = self._pv_grid.slice(normal=normal, origin=origin)
        self._plotter.add_mesh(
            sliced, scalars='|B| (T)',
            cmap='plasma', clim=[self._vmin, self._vmax],
            show_scalar_bar=True,
            scalar_bar_args={
                'title': '|B| (T)',
                'color': THEME['text'],
                'title_font_size': 10,
                'label_font_size': 9,
                'vertical': True,
            },
            name='slice',
        )
        self._pv_pos_lbl.setText(f"pos = {pos:+.4f} m")

    def _pv_set_coil_visible(self, visible: bool):
        try:
            actors = self._plotter.renderer.actors
            if 'coil' in actors:
                actors['coil'].SetVisibility(visible)
            self._plotter.render()
        except Exception:
            pass

    def _pv_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "bfield_slice.png",
            "PNG image (*.png);;TIFF image (*.tiff)",
        )
        if path:
            self._plotter.screenshot(path)

    def _pv_reset_view(self):
        self._plotter.reset_camera()
        self._plotter.render()

    # ─────────────────────────────────────────────────────────────────────────
    # matplotlib viewer  (2-D imshow — fast, professional heat-map style)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_matplotlib_viewer(self):
        import matplotlib.pyplot as plt

        # Controls
        self._mpl_axis_cb, self._mpl_slider, self._mpl_pos_lbl = \
            self._build_controls(self._mpl_save)

        # Coil toggle + status label row
        coil_row = QWidget()
        coil_row.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
        )
        cr = QHBoxLayout(coil_row)
        cr.setContentsMargins(10, 4, 10, 4)
        cr.setSpacing(16)

        self._mpl_chk_coil = QCheckBox("Show coil overlay")
        self._mpl_chk_coil.setChecked(True)
        self._mpl_chk_coil.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        self._mpl_chk_coil.toggled.connect(self._on_coil_toggle)

        self._pr_status_lbl = QLabel("")
        self._pr_status_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:7pt; font-style:italic;"
        )

        cr.addWidget(self._mpl_chk_coil)
        cr.addWidget(self._pr_status_lbl)
        cr.addStretch()
        self._outer.addWidget(coil_row)

        # Persistent canvas (rebuilt per axis change via _mpl_setup_axis)
        self._mpl_fig = plt.figure(figsize=(9, 7), facecolor=THEME['bg'])
        plt.close(self._mpl_fig)
        self._mpl_canvas = WatermarkedCanvas(self._mpl_fig)

        canvas_holder = QWidget()
        ch_lay = QVBoxLayout(canvas_holder)
        ch_lay.setContentsMargins(0, 0, 0, 0)
        ch_lay.setSpacing(0)
        ch_lay.addWidget(self._mpl_canvas, stretch=1)
        ch_lay.addWidget(PlotToolbar(self._mpl_canvas))
        self._outer.addWidget(canvas_holder, stretch=1)

        # Initial draw + start pre-render
        self._mpl_setup_axis("PCA Axis")

        # Connect signals
        self._mpl_slider.valueChanged.connect(lambda _: self._mpl_draw_slice())
        self._mpl_axis_cb.currentIndexChanged.connect(self._on_axis_changed_mpl)

        # Kick off pre-render for default axis
        self._start_prerender("PCA Axis")

    def _mpl_setup_axis(self, axis_name: str):
        """
        Full axes rebuild for a new slice orientation (called once per axis change).
        Sets up imshow, coil scatter, colorbar — all persistent until next axis change.
        """
        import matplotlib.pyplot as plt

        self._current_axis = axis_name
        fig = self._mpl_fig
        fig.clear()
        ax = fig.add_subplot(111)

        # Dark style
        fig.patch.set_facecolor(THEME['bg'])
        ax.set_facecolor(THEME['panel'])
        ax.tick_params(colors=THEME['text'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME['border'])

        # Initial mid-position slice
        mid_pos = self._slider_to_pos(_STEPS // 2, axis_name)
        B_2d, extent, xl, yl = self._compute_slice_2d(axis_name, mid_pos)

        # Colormap: NaN → background colour
        cmap = plt.cm.plasma.copy()
        cmap.set_bad(color=THEME['bg'])
        cmap.set_under(color=THEME['panel'])

        im = ax.imshow(
            B_2d.T, origin='lower', extent=extent,
            cmap=cmap,
            vmin=self._vmin, vmax=self._vmax,
            aspect='equal',
            interpolation='bicubic',
        )

        # Coil projection overlay
        cu, cv = self._coil_projection(axis_name)
        coil_art, = ax.plot(
            cu, cv, '.', color=THEME['accent'],
            ms=2.0, alpha=0.45, label='Coil', zorder=5,
        )
        show = self._mpl_chk_coil.isChecked() if self._mpl_chk_coil else True
        coil_art.set_visible(show)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, label='|B| (T)', fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color(THEME['text'])
        cbar.ax.tick_params(colors=THEME['text'], labelsize=8)

        # Labels and title
        ax.set_xlabel(xl, color=THEME['text'], fontsize=9)
        ax.set_ylabel(yl, color=THEME['text'], fontsize=9)
        ax.set_title(
            f'|B| Field Map  —  {axis_name}    pos = {mid_pos:+.4f} m',
            color=THEME['text'], fontsize=10,
        )
        ax.legend(loc='upper right', fontsize=7,
                  facecolor=THEME['panel'], edgecolor=THEME['border'],
                  labelcolor=THEME['text'])
        ax.grid(True, color=THEME['border'], alpha=0.3, linewidth=0.5, linestyle='--')

        self._mpl_im   = im
        self._mpl_ax   = ax
        self._mpl_coil = coil_art

        self._mpl_canvas.draw()
        self._mpl_pos_lbl.setText(f"pos = {mid_pos:+.4f} m")

    def _mpl_draw_slice(self):
        """
        Fast update on slider change.
        Uses pre-rendered cache when available; otherwise live interpolation.
        Only calls im.set_data() — no axes rebuild.
        """
        if self._mpl_im is None:
            return

        axis = self._mpl_axis_cb.currentText()

        # Axis changed without a full rebuild — trigger rebuild first
        if axis != self._current_axis:
            self._mpl_setup_axis(axis)
            return

        val = self._mpl_slider.value()
        pos = self._slider_to_pos(val, axis)
        self._mpl_pos_lbl.setText(f"pos = {pos:+.4f} m")

        # Try pre-rendered cache
        frames = self._prerendered.get(axis)
        if frames and len(frames) >= _N_PRERENDER:
            idx  = int(val * (_N_PRERENDER - 1) / (_STEPS - 1))
            B_2d = frames[min(idx, len(frames) - 1)]
        else:
            B_2d, _, _, _ = self._compute_slice_2d(axis, pos)

        self._mpl_im.set_data(B_2d.T)

        # Update title with current position
        self._mpl_ax.set_title(
            f'|B| Field Map  —  {axis}    pos = {pos:+.4f} m',
            color=THEME['text'], fontsize=10,
        )

        self._mpl_canvas.draw_idle()

    def _on_axis_changed_mpl(self):
        axis = self._mpl_axis_cb.currentText()
        self._mpl_slider.setValue(_STEPS // 2)
        self._mpl_setup_axis(axis)
        self._start_prerender(axis)

    def _on_coil_toggle(self, checked: bool):
        if self._mpl_coil is not None:
            self._mpl_coil.set_visible(checked)
            if self._mpl_canvas:
                self._mpl_canvas.draw_idle()

    def _mpl_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "bfield_slice.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._mpl_fig.savefig(
                path, dpi=150, bbox_inches='tight',
                facecolor=THEME['bg'],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Background pre-render
    # ─────────────────────────────────────────────────────────────────────────

    def _start_prerender(self, axis_name: str):
        """Start (or restart) background pre-rendering for axis_name."""
        # Abort any running pre-render
        if self._pr_worker is not None:
            self._pr_worker.abort()
        if self._pr_thread is not None and self._pr_thread.isRunning():
            self._pr_thread.quit()
            self._pr_thread.wait(300)

        # Skip if already cached
        frames = self._prerendered.get(axis_name)
        if frames and len(frames) >= _N_PRERENDER:
            self._update_pr_status(axis_name)
            return

        if hasattr(self, '_pr_status_lbl'):
            self._pr_status_lbl.setText(f"Pre-rendering {axis_name}…")

        self._pr_thread = QThread(self)
        self._pr_worker = _PreRenderWorker(self, axis_name)
        self._pr_worker.moveToThread(self._pr_thread)
        self._pr_worker.finished.connect(self._on_prerender_done)
        self._pr_worker.finished.connect(self._pr_thread.quit)
        self._pr_worker.finished.connect(self._pr_worker.deleteLater)
        self._pr_thread.finished.connect(self._pr_thread.deleteLater)
        self._pr_thread.started.connect(self._pr_worker.run)
        self._pr_thread.start()

    def _on_prerender_done(self, axis_name: str, frames: list):
        if frames:
            self._prerendered[axis_name] = frames
        self._update_pr_status(axis_name)

    def _update_pr_status(self, axis_name: str):
        if not hasattr(self, '_pr_status_lbl'):
            return
        cached = [k for k, v in self._prerendered.items() if len(v) >= _N_PRERENDER]
        if cached:
            self._pr_status_lbl.setText(
                f"Cached: {', '.join(cached)}"
            )
        else:
            self._pr_status_lbl.setText("")
