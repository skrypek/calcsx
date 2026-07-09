# primary/project_view.py
"""
ProjectView — one self-contained CalcSX scene.

Owns the browser, properties panel, 3D workspace, and every piece of
per-scene state (coil engines, circuits, MultiCoilEnvironment, Hall
probes, worker threads, inspection caches). MainWindow holds one
ProjectView per tab and routes ribbon signals to the currently active
instance.
"""

import os
import sys
import base64
import pickle
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QProgressDialog,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
)
from PyQt5.QtCore import Qt, QThread, QTimer

from CalcSX_app.physics.superposition import MultiCoilEnvironment
from CalcSX_app.gui.gui_utils import ProgressReporter, THEME
from CalcSX_app.views.workspace_3d import Workspace3DView

# Workers + dialogs are defined in main_utils; importing at module load is safe
# because main_utils does NOT top-level-import project_view (only MainWindow.__init__
# does, lazily). Panel classes are still late-imported inside __init__.
from CalcSX_app.primary.main_utils import (
    AnalysisWorker,
    FieldLinesWorker,
    CrossSectionWorker,
    GlobalFieldLinesWorker,
    LMatrixWorker,
    CoilGeneratorDialog,
    _get_coil_colors,
    _b_field_unit,
    _fmt_b,
)


class RelativeDistanceDialog(QDialog):
    """Modal dialog for the CONSTRUCT → Relative Distance tool.

    Shows a dropdown for each of two coils, a live readout of their current
    centroid-to-centroid distance, and a spinbox for the desired distance.
    The dialog does NOT move any coils — ProjectView handles that on accept
    using `.result()` to fetch the user's choices.
    """

    def __init__(self, parent, coil_ids, names_map, pinned_set, centroid_fn):
        super().__init__(parent)
        self.setWindowTitle("Relative Distance")
        self.setModal(True)
        self._coil_ids = list(coil_ids)
        self._pinned = set(pinned_set)
        self._centroid_fn = centroid_fn

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self._cmb_ref = QComboBox()
        self._cmb_tgt = QComboBox()
        for cid in self._coil_ids:
            label = names_map.get(cid, cid)
            if cid in self._pinned:
                label = f"⚲ {label}"
            self._cmb_ref.addItem(label, cid)
            self._cmb_tgt.addItem(label, cid)
        if len(self._coil_ids) >= 2:
            self._cmb_tgt.setCurrentIndex(1)

        form.addRow("Reference coil:", self._cmb_ref)
        form.addRow("Target coil:", self._cmb_tgt)

        self._current_lbl = QLabel("—")
        form.addRow("Current distance:", self._current_lbl)

        self._spin_dist = QDoubleSpinBox()
        self._spin_dist.setRange(0.0, 1000.0)
        self._spin_dist.setDecimals(4)
        self._spin_dist.setSingleStep(0.01)
        self._spin_dist.setSuffix(" m")
        form.addRow("New distance:", self._spin_dist)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._cmb_ref.currentIndexChanged.connect(self._refresh_current)
        self._cmb_tgt.currentIndexChanged.connect(self._refresh_current)
        self._refresh_current()

    def _refresh_current(self) -> None:
        ref_id = self._cmb_ref.currentData()
        tgt_id = self._cmb_tgt.currentData()
        if not ref_id or not tgt_id or ref_id == tgt_id:
            self._current_lbl.setText("—")
            return
        rc = self._centroid_fn(ref_id)
        tc = self._centroid_fn(tgt_id)
        if rc is None or tc is None:
            self._current_lbl.setText("—")
            return
        d = float(np.linalg.norm(np.asarray(tc) - np.asarray(rc)))
        self._current_lbl.setText(f"{d:.4f} m")
        # Seed the spinbox with the current distance on first open.
        if self._spin_dist.value() == 0.0:
            self._spin_dist.setValue(d)

    def result(self) -> tuple:
        """Return (ref_id, tgt_id, new_distance_m). Only valid after accept()."""
        return (
            self._cmb_ref.currentData(),
            self._cmb_tgt.currentData(),
            float(self._spin_dist.value()),
        )


class StrayArrayDialog(QDialog):
    """Configuration dialog for a linear stray-field probe array.

    Built-in presets include the two TEAM 22 scoring lines (a + b). Custom
    mode exposes start/end/n-points spinboxes for arbitrary linear arrays.
    """

    PRESETS = {
        'Custom':              None,
        'TEAM 22 line a (z-axis, 11 pts, 0–10 m)':  ((0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 11),
        'TEAM 22 line b (x-axis, 11 pts, 0–10 m)':  ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 11),
    }

    def __init__(self, parent, default_name: str):
        super().__init__(parent)
        self.setWindowTitle("Add Stray-Field Array")
        self.setModal(True)

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self._cmb_preset = QComboBox()
        for label in self.PRESETS:
            self._cmb_preset.addItem(label)
        form.addRow("Preset:", self._cmb_preset)

        self._name = QInputDialog()  # placeholder; we use QLineEdit below
        from PyQt5.QtWidgets import QLineEdit
        self._name = QLineEdit(default_name)
        form.addRow("Name:", self._name)

        def _spin(value: float) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-1000.0, 1000.0)
            s.setDecimals(3)
            s.setSuffix(" m")
            s.setValue(value)
            return s

        self._sx = _spin(0.0); self._sy = _spin(0.0); self._sz = _spin(0.0)
        self._ex = _spin(0.0); self._ey = _spin(0.0); self._ez = _spin(10.0)
        start_row = QHBoxLayout()
        start_row.setSpacing(2)
        for w in (self._sx, self._sy, self._sz):
            start_row.addWidget(w)
        start_w = QWidget(); start_w.setLayout(start_row)
        end_row = QHBoxLayout()
        end_row.setSpacing(2)
        for w in (self._ex, self._ey, self._ez):
            end_row.addWidget(w)
        end_w = QWidget(); end_w.setLayout(end_row)
        form.addRow("Start (x, y, z):", start_w)
        form.addRow("End (x, y, z):",   end_w)

        from PyQt5.QtWidgets import QSpinBox
        self._n_pts = QSpinBox(); self._n_pts.setRange(2, 1000); self._n_pts.setValue(11)
        form.addRow("Points:", self._n_pts)

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._cmb_preset.currentTextChanged.connect(self._apply_preset)

    def _apply_preset(self, label: str) -> None:
        spec = self.PRESETS.get(label)
        if spec is None:
            return
        (sx, sy, sz), (ex, ey, ez), n = spec
        self._sx.setValue(sx); self._sy.setValue(sy); self._sz.setValue(sz)
        self._ex.setValue(ex); self._ey.setValue(ey); self._ez.setValue(ez)
        self._n_pts.setValue(int(n))

    def result(self) -> tuple:
        """Return (name, positions_array_Nx3). Only valid after accept()."""
        start = np.array([self._sx.value(), self._sy.value(), self._sz.value()])
        end   = np.array([self._ex.value(), self._ey.value(), self._ez.value()])
        n     = int(self._n_pts.value())
        ts    = np.linspace(0.0, 1.0, max(n, 2))
        positions = start[None, :] + ts[:, None] * (end - start)[None, :]
        name = self._name.text().strip() or "Stray Array"
        return name, positions


class ProjectView(QWidget):
    """Single-project scene container. See module docstring."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        # Back-pointer to the hosting MainWindow — used to reach the shared
        # ribbon and the MainWindow-level theme/settings handlers.
        self._main = main_window
        # Display name (set by MainWindow when the tab is created).
        self.name: str = ""

        # ── Scene state ──────────────────────────────────────────────────────
        self._coords   = None
        self._multi_env = MultiCoilEnvironment()
        self._a_thread = None
        self._a_worker         = None
        self._i_thread         = None
        self._i_worker         = None
        self._inspect_reporter = None
        # Background L-matrix precompute (warms _multi_env._L_cache so the
        # first circuit-header click doesn't freeze the UI).
        self._lmx_thread = None
        self._lmx_worker = None

        # Multi-coil tracking
        self._coil_counter   = 0                # auto-increment for unique IDs
        self._coil_names:    dict = {}          # coil_id → display name
        self._coil_paths:    dict = {}          # coil_id → CSV file path (absolute)
        self._coil_coords:   dict = {}          # coil_id → np.ndarray
        self._active_coil_id:    str | None = None
        self._multi_edit_ids:    list = []      # IDs of coils in a bulk-edit selection (len>=2); empty otherwise
        # Circuit groups: each group_id → {kind, coil_ids, signs, color, name}
        self._circuit_groups:    dict = {}
        self._coil_group_map:    dict = {}          # coil_id → group_id (if grouped)
        self._circuit_counter:   int  = 0
        self._active_circuit_id: str | None = None
        self._analyzed_coil_id:  str | None = None   # coil that owns the in-progress analysis
        self._coil_engines:      dict       = {}      # coil_id → CoilAnalysis engine
        self._coil_inspect_cache:  dict     = {}      # coil_id → {field_lines, field_mags, cross_section}
        self._coil_params_map:   dict       = {}      # coil_id → {winds, current, thickness, width, axis_num}
        self._inspect_coil_id:   str | None = None    # coil being currently inspected
        self._pending_inspect:        str | None = None   # 'field_lines' | 'cross_section'
        self._analysis_auto_triggered: bool = False        # True when analysis kicked off by inspect
        self._global_fl_cache:        tuple | None = None  # (lines, B_mags) or None
        self._global_fl_dirty:        bool = True          # True when environment changed since last global FL
        self._pre_global_eye_state:   dict       = {}      # coil_id → bool (eye state before global mode)

        # Hall probes (multiple)
        self._probe_counter = 0
        self._probe_timer   = None
        self._probe_state: dict = {}   # probe_id → {mode, coil_ref, uvw, name}

        # Set of coil IDs whose position is "pinned" — transforms refuse to
        # move them, and the Relative Distance tool treats them as anchors.
        self._pinned_coils: set[str] = set()

        # System Energy meter (singleton instrument). True ⇔ the meter row
        # exists in the browser; the displayed value is recomputed on every
        # event that could change ½·Iᵀ·L·I.
        self._has_system_energy: bool = False

        # Stray-field probe arrays. array_id → {name, positions (N×3 ndarray)}
        self._stray_arrays: dict = {}
        self._stray_array_counter: int = 0

        # Tracks which coil's params the PropertiesPanel spinboxes are
        # actually displaying. Used by _on_coil_selected /
        # _on_coils_multi_selected to GATE the "save UI back to old_cid"
        # step — without this gate, any selection-changed signal that
        # fires while the UI is mid-sync (e.g. during load, generate, or
        # bobbin import) overwrites the wrong coil with stale UI values.
        self._props_showing_coil_id: str | None = None

        # ── Widgets ──────────────────────────────────────────────────────────
        # Late-import to avoid circular import with main_utils.
        from CalcSX_app.primary.main_utils import (
            BrowserPanel, PropertiesPanel, _hdivider,
        )

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_lay.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(190)
        left.setMaximumWidth(320)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        self.browser = BrowserPanel()
        self.props   = PropertiesPanel()
        self.props.hide()

        left_lay.addWidget(self.browser, stretch=3)
        left_lay.addWidget(_hdivider())
        left_lay.addWidget(self.props,   stretch=2)

        self.workspace = Workspace3DView()

        splitter.addWidget(left)
        splitter.addWidget(self.workspace)
        splitter.setSizes([250, 1030])

        # ── Internal wiring (browser ↔ props ↔ workspace) ────────────────────
        self.browser.layer_toggled.connect(self.workspace.set_layer_visible)
        self.browser.coil_visibility_toggled.connect(self.workspace.set_coil_visible)
        self.browser.coil_delete_requested.connect(self._on_coil_delete)
        self.browser.layer_delete_requested.connect(self._on_layer_delete)
        self.browser.coil_selected.connect(self._on_coil_selected)
        self.browser.coils_multi_selected.connect(self._on_coils_multi_selected)
        self.browser.coil_renamed.connect(self._on_coil_renamed)
        self.browser.circuit_selected.connect(self._on_circuit_selected)
        self.browser.coil_recolored.connect(self._on_coil_recolored)
        self.browser.probe_selected.connect(self._on_probe_selected)
        self.browser.probe_delete_requested.connect(self._on_probe_delete)
        self.browser.probe_recolored.connect(
            lambda pid, c: self.workspace.set_probe_color(pid, c)
        )
        self.browser.system_energy_delete_requested.connect(
            self._on_system_energy_delete
        )
        self.browser.stray_array_delete_requested.connect(
            self._on_stray_array_delete
        )
        self.props.circuit_current_changed.connect(self._on_circuit_current_changed)
        self.props.probe_position_changed.connect(self._on_probe_xyz_edit)
        self.props.probe_pca_changed.connect(self._on_probe_pca_edit)
        self.props.probe_mode_changed.connect(self._on_probe_mode_change)
        self.workspace.set_stale_callback(self._on_layers_stale)

        for w in (self.props.spin_winds, self.props.dspin_current,
                  self.props.dspin_thick, self.props.dspin_width,
                  self.props.spin_axis_pts):
            w.valueChanged.connect(self._on_coil_param_changed)
        # Stack growth direction is also a coil-stale parameter.
        self.props.cmb_stack.currentIndexChanged.connect(
            self._on_coil_param_changed
        )
        # Red current-direction arrows: shown only while the user is editing
        # the Current spinbox; refreshed on every value change so the arrows
        # flip live when the sign changes.
        self.props.current_edit_started.connect(self._on_current_edit_started)
        self.props.current_edit_finished.connect(self._on_current_edit_finished)
        self.props.dspin_current.valueChanged.connect(self._on_current_value_changed)

    # ── Dirtiness check (used by MainWindow for close-tab prompt) ────────────

    def is_dirty(self) -> bool:
        """True if this project has scene state that would be lost on close.

        Blank projects (no coils, probes, or bobbins) return False so
        MainWindow can close them silently.
        """
        if self._coil_coords:
            return True
        if self._probe_state:
            return True
        for (_bid, lname) in self.workspace._layers.keys():
            if lname == 'Bobbin':
                return True
        return False

    def shutdown_workers(self, wait_ms: int = 500) -> None:
        """Stop every background worker + timer this project owns.

        Called by MainWindow before tab close to avoid
        "QThread: Destroyed while thread is still running" and leaked
        QTimer ticks after the widget is gone.
        """
        for attr in ('_a_thread', '_i_thread', '_lmx_thread'):
            t = getattr(self, attr, None)
            if t is not None:
                try:
                    if t.isRunning():
                        t.quit()
                        t.wait(wait_ms)
                except RuntimeError:
                    # Underlying C++ QThread already deleted.
                    pass
                setattr(self, attr, None)
        if self._probe_timer is not None:
            try:
                self._probe_timer.stop()
            except RuntimeError:
                pass
            self._probe_timer = None

    # ── Signal handlers (migrated verbatim from MainWindow) ──────────────────

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select coil geometry", "",
            "Coil files (*.csv *.step *.stp *.iges *.igs);;"
            "CSV files (*.csv);;"
            "STEP files (*.step *.stp);;"
            "IGES files (*.iges *.igs)",
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.step', '.stp', '.iges', '.igs'):
                from CalcSX_app.physics.geometry import import_step_centerline
                coords = import_step_centerline(path)
            else:
                df = pd.read_csv(path)
                coords = (
                    df[['x', 'y', 'z']].values
                    if {'x', 'y', 'z'}.issubset(df.columns)
                    else df.iloc[:, :3].values
                )
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack((coords, coords[0]))
        except ImportError as exc:
            QMessageBox.warning(self, "Missing Dependency", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        # Generate unique coil id and display name
        self._coil_counter += 1
        coil_id   = f"coil_{self._coil_counter}"
        fname     = os.path.basename(path)
        base_name = os.path.splitext(fname)[0]
        _cc = _get_coil_colors()
        color = _cc[(self._coil_counter - 1) % len(_cc)]

        self._coords              = coords
        self._active_coil_id      = coil_id
        self._coil_coords[coil_id] = coords
        self._coil_paths[coil_id]  = os.path.abspath(path)
        self._coil_names[coil_id]  = base_name

        # Store per-coil parameters (snapshot from current spinbox values)
        self._coil_params_map[coil_id] = self.props.get_params()

        params = self._coil_params_map[coil_id]
        total_t = params['thickness'] * 1e-6 * params['winds']  # µm→m × winds
        tape_w  = params['width'] * 1e-3                        # mm→m
        growth  = params.get('stack_growth', 'symmetric')
        self.workspace.add_coil(coords, coil_id, color=color,
                                 total_thickness=total_t, tape_width=tape_w,
                                 winding_growth=growth)
        self.browser.add_coil_item(coil_id, base_name, color)

        # Register in superposition environment — marks existing coils stale
        self._multi_env.register_coil(
            coil_id, coords,
            winds=params['winds'], current=params['current'],
            thickness=params['thickness'], width=params['width'],
            winding_growth=growth,
        )
        self._propagate_staleness()

        self._main.ribbon.set_inspect_enabled(True)   # enabled — analysis runs automatically if needed
        self._main.ribbon.set_construct_enabled(True)
        self._main.ribbon._btn_translate.setChecked(False)
        self._main.ribbon._btn_rotate.setChecked(False)
        self.props.show()
        # Sync the Properties panel to the new coil so the next browser click
        # doesn't write the stale (previous-coil) UI values into it.
        self._load_coil_params(coil_id)
        self._sync_ribbon_transform_ui(coil_id)

    def _on_generate_coil(self) -> None:
        """Open the parametric coil generator dialog and load the result."""
        dlg = CoilGeneratorDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        coords = dlg.get_coords()
        if coords is None or len(coords) < 3:
            return
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack((coords, coords[0]))

        shape_name = dlg._combo.currentText()
        self._coil_counter += 1
        coil_id = f"coil_{self._coil_counter}"
        _cc = _get_coil_colors()
        color = _cc[(self._coil_counter - 1) % len(_cc)]

        self._coords = coords
        self._active_coil_id = coil_id
        self._coil_coords[coil_id] = coords
        self._coil_paths[coil_id] = f"<generated:{shape_name}>"
        self._coil_names[coil_id] = f"{shape_name}_{self._coil_counter}"
        self._coil_params_map[coil_id] = self.props.get_params()

        params = self._coil_params_map[coil_id]
        total_t = params['thickness'] * 1e-6 * params['winds']
        tape_w = params['width'] * 1e-3
        growth = params.get('stack_growth', 'symmetric')
        self.workspace.add_coil(coords, coil_id, color=color,
                                 total_thickness=total_t, tape_width=tape_w,
                                 winding_growth=growth)
        self.browser.add_coil_item(coil_id, self._coil_names[coil_id], color)

        self._multi_env.register_coil(
            coil_id, coords,
            winds=params['winds'], current=params['current'],
            thickness=params['thickness'], width=params['width'],
            winding_growth=growth,
        )
        self._propagate_staleness()

        self._main.ribbon.set_inspect_enabled(True)
        self._main.ribbon.set_construct_enabled(True)
        self.props.show()
        # Sync the Properties panel to the new coil so the next browser click
        # doesn't write the stale UI values into it.
        self._load_coil_params(coil_id)
        self._sync_ribbon_transform_ui(coil_id)

    def _on_import_bobbin(self) -> None:
        """Import a .bobsx bobbin file (exported from Fusion 360)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Bobbin", "",
            "CalcSX Bobbin (*.bobsx);;"
            "All files (*)",
        )
        if not path:
            return

        try:
            import json as _json
            with open(path, 'r') as f:
                data = _json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error",
                                 f"Could not read file:\n{exc}")
            return

        version = data.get('version', 1)
        unit = data.get('unit', 'cm')
        # Fusion uses cm internally; convert to metres for CalcSX
        scale = {'cm': 1e-2, 'mm': 1e-3, 'm': 1.0}.get(unit, 1e-2)

        channels = data.get('channels', [])
        if not channels:
            QMessageBox.warning(self, "Import Warning",
                                "No channels found in the file.")
            return

        # Optional bobbin display mesh — add as a toggleable coil entry
        bobbin_id = f"bobbin_{self._coil_counter + 1}"
        bm = data.get('bobbin_mesh')
        if bm and 'vertices' in bm and 'faces' in bm:
            try:
                import pyvista as pv
                verts = np.array(bm['vertices'], dtype=np.float32) * scale
                tri = np.array(bm['faces'], dtype=np.int64)
                pv_faces = np.column_stack([
                    np.full(len(tri), 3, dtype=np.int64), tri
                ]).ravel()
                mesh = pv.PolyData(verts, pv_faces)
                self.workspace.add_bobbin_mesh(bobbin_id, mesh)
                # Add to browser so user can toggle/delete it
                self.browser.add_coil_item(
                    bobbin_id, "Bobbin", '#888888')
            except Exception:
                pass

        # Create a coil for each channel
        _cc = _get_coil_colors()
        params = self.props.get_params()

        for ch in channels:
            pts = ch.get('points', [])
            if len(pts) < 3:
                continue

            coords = np.array(
                [[p['x'], p['y'], p['z']] for p in pts],
                dtype=np.float64) * scale
            normals = np.array(
                [[p['nx'], p['ny'], p['nz']] for p in pts],
                dtype=np.float64)

            # Normalise normals
            nlen = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(nlen, 1e-10)

            # Close the loop if needed
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack((coords, coords[0]))
                normals = np.vstack((normals, normals[-1:]))

            winds = ch.get('winds', params.get('winds', 200))

            self._coil_counter += 1
            coil_id = f"coil_{self._coil_counter}"
            color = _cc[(self._coil_counter - 1) % len(_cc)]
            name = ch.get('name', f"groove_{self._coil_counter}")

            self._coords = coords
            self._active_coil_id = coil_id
            self._coil_coords[coil_id] = coords
            self._coil_paths[coil_id] = os.path.abspath(path)
            self._coil_names[coil_id] = name

            p = dict(params)
            p['winds'] = winds
            p['tape_normals'] = normals
            # Per-channel parameter overrides (extension to .bobsx v1):
            # current (A), thickness (µm), width (mm), stack_growth ('up' |
            # 'symmetric'). Anything missing falls back to the Properties-panel
            # values active at import time. Lets a single .bobsx fully describe
            # a benchmark configuration (e.g. TEAM 22 SMES).
            if 'current' in ch:
                p['current'] = float(ch['current'])
            if 'thickness' in ch:
                p['thickness'] = float(ch['thickness'])
            if 'width' in ch:
                p['width'] = float(ch['width'])
            growth = ch.get('stack_growth', 'up')
            if growth not in ('symmetric', 'up'):
                growth = 'up'
            p['stack_growth'] = growth
            self._coil_params_map[coil_id] = p

            total_t = p['thickness'] * 1e-6 * winds
            tape_w = p['width'] * 1e-3
            self.workspace.add_coil(coords, coil_id, color=color,
                                     total_thickness=total_t,
                                     tape_width=tape_w,
                                     tape_normals=normals,
                                     winding_growth=growth)
            self.browser.add_coil_item(coil_id, name, color)

            self._multi_env.register_coil(
                coil_id, coords,
                winds=winds, current=p['current'],
                thickness=p['thickness'], width=p['width'],
                tape_normals=normals,
                winding_growth=growth,
            )

        self._propagate_staleness()
        self._main.ribbon.set_inspect_enabled(True)
        self._main.ribbon.set_construct_enabled(True)
        self.props.show()
        # After the channel loop, sync the Properties panel to the LAST
        # imported coil (which is also self._active_coil_id). Without this,
        # a subsequent browser click would write the stale UI values into
        # whichever coil was last imported.
        if self._active_coil_id:
            self._load_coil_params(self._active_coil_id)
            self._sync_ribbon_transform_ui(self._active_coil_id)
        if self.workspace._plotter:
            self.workspace._plotter.reset_camera()
            self.workspace._plotter.render()

    def _snapshot_camera(self):
        """Return an opaque snapshot of the workspace camera state, or None
        if unavailable. Used to protect camera view across destructive
        operations that may trigger an implicit VTK reset_camera."""
        plotter = getattr(self.workspace, '_plotter', None)
        if plotter is None:
            return None
        try:
            cam = plotter.renderer.GetActiveCamera()
            return (
                cam.GetPosition(),
                cam.GetFocalPoint(),
                cam.GetViewUp(),
                cam.GetParallelScale(),
                cam.GetViewAngle(),
            )
        except Exception:
            return None

    def _restore_camera(self, snapshot) -> None:
        """Restore a previously snapshotted camera state and render once."""
        plotter = getattr(self.workspace, '_plotter', None)
        if plotter is None or snapshot is None:
            return
        try:
            cam = plotter.renderer.GetActiveCamera()
            cam.SetPosition(*snapshot[0])
            cam.SetFocalPoint(*snapshot[1])
            cam.SetViewUp(*snapshot[2])
            cam.SetParallelScale(snapshot[3])
            cam.SetViewAngle(snapshot[4])
            plotter.render()
        except Exception:
            pass

    def _on_coil_delete(self, coil_id: str) -> None:
        # Preserve the camera view across the delete — otherwise an implicit
        # reset from removing actors / rebuilding the floor would snap the
        # view back to the default framing.
        cam_snap = self._snapshot_camera()
        # Bobbin display mesh — stored as a layer, not a coil
        if coil_id.startswith('bobbin_'):
            key = (coil_id, 'Bobbin')
            if key in self.workspace._layers:
                layer = self.workspace._layers[key]
                for actor in layer.actors:
                    try:
                        self.workspace._plotter.remove_actor(
                            actor, render=False)
                    except Exception:
                        pass
                del self.workspace._layers[key]
                self.workspace._plotter.render()
            self.browser.remove_coil_item(coil_id)
            self._restore_camera(cam_snap)
            return

        # Clear analysis layers and engine for this coil
        self.workspace.clear_analysis_layers(coil_id)
        self._coil_engines.pop(coil_id, None)
        self._coil_inspect_cache.pop(coil_id, None)
        self._multi_env.unregister_coil(coil_id)
        self._pinned_coils.discard(coil_id)
        if self._props_showing_coil_id == coil_id:
            self._props_showing_coil_id = None
        # Drop this coil from any circuit group; dissolve the group if it
        # falls below 2 members.
        gid = self._coil_group_map.pop(coil_id, None)
        if gid and gid in self._circuit_groups:
            g = self._circuit_groups[gid]
            g['coil_ids'] = [c for c in g['coil_ids'] if c != coil_id]
            g['signs'].pop(coil_id, None)
            if len(g['coil_ids']) < 2:
                self._dissolve_group(gid)
        self._propagate_staleness()
        if coil_id == self._analyzed_coil_id:
            self._analyzed_coil_id = None
        # Remove coil from workspace and browser (browser removes nested analysis too)
        self.workspace.remove_coil(coil_id)
        self.browser.remove_coil_item(coil_id)
        self._coil_coords.pop(coil_id, None)
        self._coil_paths.pop(coil_id, None)
        self._coil_names.pop(coil_id, None)
        self._coil_params_map.pop(coil_id, None)
        if self._active_coil_id == coil_id:
            new_id = self.workspace._active_coil_id
            self._active_coil_id = new_id
            self._coords = self._coil_coords.get(new_id) if new_id else None
        if not self._coil_coords:
            self._main.ribbon.set_construct_enabled(False)
            self._main.ribbon.set_inspect_enabled(False)
        # Restore the pre-delete camera so the view doesn't jump
        self._restore_camera(cam_snap)

    def _on_layer_delete(self, coil_id: str, layer_name: str) -> None:
        """Delete a user-created sub-layer (Cross Section or Field Lines)."""
        cam_snap = self._snapshot_camera()
        if layer_name == 'Cross Section':
            self.workspace.clear_cross_section_layer(coil_id)
            self.browser.remove_layer_from_coil(coil_id, 'Cross Section')
            ic = self._coil_inspect_cache.get(coil_id)
            if ic is not None:
                ic.pop('cross_section', None)
                if not ic:
                    self._coil_inspect_cache.pop(coil_id, None)
        elif layer_name == 'Field Lines':
            self.workspace.clear_field_lines_layer(coil_id)
            self.browser.remove_layer_from_coil(coil_id, 'Field Lines')
            ic = self._coil_inspect_cache.get(coil_id)
            if ic is not None:
                ic.pop('field_lines', None)
                ic.pop('field_mags', None)
                ic.pop('field_seeds', None)
                if not ic:
                    self._coil_inspect_cache.pop(coil_id, None)
            # Re-unify the remaining field-line scale bar
            self.workspace.rescale_all_field_line_layers()
        self._restore_camera(cam_snap)

    def _refresh_summary_for(self, coil_id: str) -> None:
        """Populate the Properties summary from the coil's engine, if any.
        If the coil is part of a circuit group, also compute and push the
        circuit-level inductance into the summary and show a group banner."""
        engine = self._coil_engines.get(coil_id)
        if engine is not None:
            self.props.update_summary(engine)
        gid = self._coil_group_map.get(coil_id)
        if gid:
            g = self._circuit_groups[gid]
            self.props.set_circuit_banner(g)
            # Hide per-coil Induct. row — the circuit L is the meaningful
            # value for a grouped coil.
            self.props.set_summary_row_visible('Induct.', False)
            self.props.set_summary_row_visible('Circuit L', True)
            # Only compute synchronously if the L-matrix cache is warm —
            # otherwise defer to the background worker to avoid freezing
            # the UI on coil selection.
            env = self._multi_env
            cache = getattr(env, '_L_cache', None)
            engine_ids = list(getattr(env, '_engines', {}).keys())
            cache_warm = (cache is not None
                          and cache.get('coil_ids') == engine_ids)
            L_c = None
            if cache_warm:
                try:
                    L_c = self._compute_circuit_inductance(gid)
                except Exception:
                    L_c = None
            else:
                self._schedule_l_matrix_precompute()
            if hasattr(self.props, '_sum_lbls') \
                    and 'Circuit L' in self.props._sum_lbls:
                if L_c is None:
                    self.props._sum_lbls['Circuit L'].setText("computing…")
                else:
                    if L_c >= 1.0:
                        txt = f"{L_c:.3f} H"
                    elif L_c >= 1e-3:
                        txt = f"{L_c*1e3:.3f} mH"
                    else:
                        txt = f"{L_c*1e6:.2f} µH"
                    self.props._sum_lbls['Circuit L'].setText(txt)
        else:
            self.props.set_circuit_banner(None)
            # Ungrouped coil — show per-coil self-inductance, hide Circuit L
            self.props.set_summary_row_visible('Induct.', True)
            self.props.set_summary_row_visible('Circuit L', False)
            if hasattr(self.props, '_sum_lbls') \
                    and 'Circuit L' in self.props._sum_lbls:
                self.props._sum_lbls['Circuit L'].setText("—")

    def _sync_ribbon_transform_ui(self, coil_id: str | None) -> None:
        """Push the coil's current (pin state, Tx/Ty/Tz, Rx/Ry/Rz) into the
        ribbon Pin toggle and VALUES spinboxes.

        Called on selection change and after any gizmo drag so the UI stays
        in lockstep with the workspace's stored xfm_params.
        """
        if not coil_id or coil_id.startswith('bobbin_'):
            self._main.ribbon.set_pin_state(False)
            self._main.ribbon.set_transform_values(0, 0, 0, 0, 0, 0)
            return
        pinned = coil_id in self._pinned_coils
        self._main.ribbon.set_pin_state(pinned)
        entry = self.workspace._coil_entries.get(coil_id, {})
        xfm = entry.get('xfm_params')
        if xfm is None:
            self._main.ribbon.set_transform_values(0, 0, 0, 0, 0, 0)
        else:
            tx, ty, tz, rx, ry, rz = xfm[:6]
            self._main.ribbon.set_transform_values(tx, ty, tz, rx, ry, rz)

    # ── Transient current-direction arrows ───────────────────────────────────

    def _on_current_edit_started(self) -> None:
        """User focused the Current spinbox — show red direction arrows."""
        cid = self._active_coil_id
        if not cid or cid.startswith('bobbin_'):
            return
        self._current_arrows_active = True
        self.workspace.show_current_arrows(
            cid, float(self.props.dspin_current.value()),
        )

    def _on_current_edit_finished(self) -> None:
        """User defocused the Current spinbox — hide the arrows."""
        self._current_arrows_active = False
        self.workspace.hide_current_arrows()

    def _on_current_value_changed(self, value: float) -> None:
        """Live-update arrow direction while the user is editing.

        Only acts when arrows are already visible (focused state); otherwise
        the regular _on_coil_param_changed handler takes care of the value
        update without spawning arrows.
        """
        if not getattr(self, '_current_arrows_active', False):
            return
        cid = self._active_coil_id
        if not cid or cid.startswith('bobbin_'):
            return
        self.workspace.show_current_arrows(cid, float(value))

    # ── Pin / Relative Distance / type-in transform handlers ─────────────────

    def _on_pin_toggled(self, checked: bool) -> None:
        """User toggled Pin on the active coil."""
        cid = self._active_coil_id
        if not cid or cid.startswith('bobbin_'):
            return
        if checked:
            self._pinned_coils.add(cid)
            # Cancel any active gizmo edit on this coil.
            if self._main.ribbon._btn_translate.isChecked():
                self._main.ribbon._btn_translate.setChecked(False)
            if self._main.ribbon._btn_rotate.isChecked():
                self._main.ribbon._btn_rotate.setChecked(False)
            self.workspace.hide_gizmo()
        else:
            self._pinned_coils.discard(cid)
        # Refresh Pin toggle state (already reflects the new state) and
        # re-enable/disable spinboxes.
        self._main.ribbon.set_pin_state(checked)

    def _on_transform_values_changed(
        self, tx: float, ty: float, tz: float,
        rx: float, ry: float, rz: float,
    ) -> None:
        """User typed a transform value into the VALUES spinboxes."""
        cid = self._active_coil_id
        if not cid or cid.startswith('bobbin_'):
            return
        if cid in self._pinned_coils:
            # Pin guard — revert spinboxes to the stored transform.
            self._sync_ribbon_transform_ui(cid)
            return
        self.workspace.set_active_coil(cid)
        self.workspace.apply_coil_transform(tx, ty, tz, rx, ry, rz)

    def _on_relative_distance(self) -> None:
        """Open the Relative Distance dialog.

        Moves the non-pinned coil along the line connecting centroids so the
        two selected coils sit at the requested distance. Errors out if both
        coils are pinned.
        """
        coil_ids = [cid for cid in self._coil_coords.keys()]
        if len(coil_ids) < 2:
            QMessageBox.information(
                self, "Relative Distance",
                "Need at least two coils to set a relative distance.",
            )
            return
        dlg = RelativeDistanceDialog(
            self, coil_ids, self._coil_names,
            self._pinned_coils, self._centroid_of,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        ref_id, tgt_id, new_dist = dlg.result()
        if ref_id in self._pinned_coils and tgt_id in self._pinned_coils:
            QMessageBox.warning(
                self, "Relative Distance",
                "Both coils are pinned — nothing to move.\n"
                "Unpin at least one of them first.",
            )
            return
        # The pinned coil (if any) is the anchor; the other moves.
        if tgt_id in self._pinned_coils and ref_id not in self._pinned_coils:
            ref_id, tgt_id = tgt_id, ref_id
        self._apply_relative_distance(ref_id, tgt_id, new_dist)

    def _centroid_of(self, coil_id: str):
        """World-space centroid of the given coil (post-transform)."""
        wc = self.workspace.get_transformed_coords(coil_id)
        if wc is None:
            wc = self._coil_coords.get(coil_id)
        if wc is None:
            return None
        return np.mean(np.asarray(wc, dtype=float), axis=0)

    def _apply_relative_distance(self, ref_id: str, tgt_id: str,
                                 new_dist: float) -> None:
        """Translate the target coil along (target_centroid − ref_centroid)
        until |target_centroid − ref_centroid| == new_dist."""
        ref_c = self._centroid_of(ref_id)
        tgt_c = self._centroid_of(tgt_id)
        if ref_c is None or tgt_c is None:
            return
        vec = tgt_c - ref_c
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            # Coils are coincident — pick the +x direction as a deterministic
            # default. User can rotate after.
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = vec / norm
        desired_tgt = ref_c + direction * float(new_dist)
        delta = desired_tgt - tgt_c
        # Compose with existing transform: add delta to current (tx, ty, tz).
        entry = self.workspace._coil_entries.get(tgt_id, {})
        xfm = entry.get('xfm_params')
        if xfm is None:
            tx_cur = ty_cur = tz_cur = 0.0
            rx_cur = ry_cur = rz_cur = 0.0
        else:
            tx_cur, ty_cur, tz_cur, rx_cur, ry_cur, rz_cur = xfm[:6]
        # Make tgt_id the gizmo's active coil so apply_coil_transform acts on it.
        self.workspace.set_active_coil(tgt_id)
        self.workspace.apply_coil_transform(
            tx_cur + float(delta[0]),
            ty_cur + float(delta[1]),
            tz_cur + float(delta[2]),
            rx_cur, ry_cur, rz_cur,
        )
        # Restore user's selection focus.
        if self._active_coil_id and self._active_coil_id != tgt_id:
            self.workspace.set_active_coil(self._active_coil_id)
        self._sync_ribbon_transform_ui(self._active_coil_id)

    def _on_coil_selected(self, coil_id: str) -> None:
        # Save current coil's spinbox values before switching — BUT only if
        # the Properties panel was actually showing old_cid's values.
        # Without that gate, any selection-changed signal that fires while
        # the UI is mid-sync (load, import, generate) overwrites the wrong
        # coil with stale values.
        old_cid = self._active_coil_id
        if (old_cid
                and old_cid in self._coil_params_map
                and not self._multi_edit_ids
                and self._props_showing_coil_id == old_cid):
            ui_params = self.props.get_params()
            # If the old coil is inside a circuit, its Current spinbox was
            # hidden and carries the stale pre-group value. Dropping it
            # here prevents that stale value from clobbering the branch
            # current the circuit just wrote into _coil_params_map.
            if old_cid in self._coil_group_map:
                ui_params.pop('current', None)
            self._coil_params_map[old_cid].update(ui_params)
        # Note: multi-edit state is owned by `_on_coils_multi_selected`, which
        # fires right after this from the tree's selectionChanged signal with
        # the authoritative list. Don't clear it here — that would stomp on
        # a just-built multi selection and cause the banner to flash off.
        self._active_coil_id = coil_id
        self._active_circuit_id = None
        self._coords = self._coil_coords.get(coil_id)
        # Switch gizmo target back to coil
        self.workspace.set_gizmo_target('coil')
        # Bobbin layers are visual-only — no physics params to show.
        if coil_id.startswith('bobbin_'):
            self.props.show_bobbin_view()
            self.workspace.set_active_coil(coil_id)
            if self._main.ribbon._btn_translate.isChecked():
                self.workspace.show_gizmo('T')
            elif self._main.ribbon._btn_rotate.isChecked():
                self.workspace.show_gizmo('R')
            return
        # Swap Properties back to coil view
        self.props.show_coil_controls()
        # Load the new coil's params into spinboxes
        self._load_coil_params(coil_id)
        # If this coil is part of a circuit, hide per-coil Current and show
        # an inheritance note with the actual BRANCH current (for parallel
        # circuits this is the circuit total divided across branches, not
        # the circuit total itself).
        gid = self._coil_group_map.get(coil_id)
        if gid and gid in self._circuit_groups:
            g = self._circuit_groups[gid]
            branch_I = float(self._coil_params_map.get(coil_id, {}).get(
                'current', g.get('current', 0.0)
            ))
            self.props.set_coil_current_editable(
                False,
                inherited_from=g.get('name', gid),
                inherited_value=branch_I,
            )
        else:
            self.props.set_coil_current_editable(True)
        self.workspace.set_active_coil(coil_id)
        # Show the selected coil's analysis summary if available
        self._refresh_summary_for(coil_id)
        # Push pin state + transform values into the CONSTRUCT ribbon UI.
        self._sync_ribbon_transform_ui(coil_id)
        # If gizmo is active, seamlessly move it to the new coil (unless the
        # coil is pinned — pinning blocks both Translate and Rotate).
        if coil_id not in self._pinned_coils:
            if self._main.ribbon._btn_translate.isChecked():
                self.workspace.show_gizmo('T')
            elif self._main.ribbon._btn_rotate.isChecked():
                self.workspace.show_gizmo('R')

    def _on_coils_multi_selected(self, coil_ids: list) -> None:
        """Authoritative multi-selection handler. Called by the browser on
        *every* selection change (including when the selection collapses to
        one or zero coils), so it decides whether to enter or leave
        multi-edit / circuit-ready mode.

        ≥ 2 selected → enter multi-edit: banner shown, bulk-edit broadcast
                        active, CIRCUITS ribbon enabled.
          1 selected → ignore here (single-coil routing is done by the
                        ``coil_selected`` → ``_on_coil_selected`` path that
                        fires on click).
          0 selected → leave multi-edit state; keep active coil as-is.
        """
        coil_ids = [cid for cid in coil_ids if cid in self._coil_params_map]
        if len(coil_ids) >= 2:
            # Save any in-flight edits on the currently-displayed coil first —
            # gated on the same "UI must actually be showing this coil"
            # invariant used in _on_coil_selected.
            if (self._active_coil_id
                    and self._active_coil_id in self._coil_params_map
                    and not self._multi_edit_ids
                    and self._props_showing_coil_id == self._active_coil_id):
                ui_params = self.props.get_params()
                if self._active_coil_id in self._coil_group_map:
                    ui_params.pop('current', None)
                self._coil_params_map[self._active_coil_id].update(ui_params)
            self._multi_edit_ids = list(coil_ids)
            self._active_coil_id = coil_ids[0]
            self.props.show_coil_controls()
            self._load_coil_params(coil_ids[0])
            # Flag parameters that differ across the selection
            mixed: list = []
            first = self._coil_params_map[coil_ids[0]]
            for key, label in (('winds', 'Winds'), ('current', 'Current'),
                               ('thickness', 'Thick'), ('width', 'Width'),
                               ('axis_num', 'Axis')):
                ref = first.get(key)
                if any(self._coil_params_map[c].get(key) != ref
                       for c in coil_ids[1:]):
                    mixed.append(label)
            self.props.set_multi_edit_banner(len(coil_ids), mixed)
            any_grouped = any(cid in self._coil_group_map for cid in coil_ids)
            self._main.ribbon.set_circuit_enabled(group_ok=True,
                                             ungroup_ok=any_grouped)
            return
        # Single-item or empty selection — leave multi-edit mode. Don't touch
        # `_active_coil_id` or the Properties display; those are owned by
        # `_on_coil_selected` (fired on click) or by whatever is left active.
        if self._multi_edit_ids:
            self._multi_edit_ids = []
            self.props.set_multi_edit_banner(0)
        grouped = (self._active_coil_id in self._coil_group_map
                   if self._active_coil_id else False)
        self._main.ribbon.set_circuit_enabled(group_ok=False, ungroup_ok=grouped)

    # ── Circuit grouping ──────────────────────────────────────────────────

    def _coil_group_colors(self) -> list:
        """Palette for circuit-group accent colors. Picked to be distinct
        from the per-coil palette."""
        return ['#ff9f43', '#4bc0c8', '#ac92ec', '#ed5565',
                '#48cfad', '#ffcc5c', '#5d9cec', '#a0d468']

    def _next_group_color(self) -> str:
        palette = self._coil_group_colors()
        return palette[len(self._circuit_groups) % len(palette)]

    def _on_circuit_selected(self, group_id: str) -> None:
        """User clicked a circuit folder header in the browser — switch the
        Properties panel to the circuit-level view (current, L, members)."""
        g = self._circuit_groups.get(group_id)
        if not g:
            return
        # Clear multi-edit state (a header click isn't a coil multi-edit)
        self._multi_edit_ids = []
        self.props.set_multi_edit_banner(0)
        self._active_circuit_id = group_id
        member_names = []
        for cid in g.get('coil_ids', []):
            member_names.append(self._coil_names.get(cid, cid))
        # If the L-matrix cache is cold and a background precompute is
        # running (or can be started), don't block the UI on the full
        # double-sum here — show a placeholder and let _on_l_matrix_ready
        # fill it in when the worker finishes.
        env = self._multi_env
        cache = getattr(env, '_L_cache', None)
        engine_ids = list(getattr(env, '_engines', {}).keys())
        cache_warm = (cache is not None
                      and cache.get('coil_ids') == engine_ids)
        if cache_warm:
            try:
                L_c = self._compute_circuit_inductance(group_id)
            except Exception:
                L_c = None
        else:
            L_c = None
            self._schedule_l_matrix_precompute()
        self.props.show_circuit_view(group_id, g, member_names, L_c)
        self._main.ribbon.set_circuit_enabled(group_ok=False, ungroup_ok=True)

    def _on_circuit_current_changed(self, group_id: str, amps: float) -> None:
        """Circuit's Current spinbox was changed — propagate the branch-level
        currents to every member coil, applying the correct split based on
        circuit kind:

            series   → I_i = s_i · I_total    (same magnitude, signed direction)
            parallel → I_i = I_total · (L⁻¹ 1)_i / (1ᵀ · L⁻¹ · 1)
                        (for two identical coils this simplifies to I/2)

        The parallel split uses the full mutual-inductance matrix; for
        identical coils the off-diagonal M_ij cancels and each branch
        carries I_total/N regardless of coupling. For non-identical coils
        the matrix solve captures the impedance-weighted split correctly.
        """
        g = self._circuit_groups.get(group_id)
        if not g:
            return
        g['current'] = float(amps)
        coil_ids = list(g.get('coil_ids', []))
        kind = g.get('kind', 'series')
        signs = g.get('signs', {})

        # Compute per-coil branch currents
        #   parallel → I_total / N  (uniform split; exact for identical coils,
        #                            which is the practical case. Non-identical
        #                            parallel branches are mis-modeled by
        #                            this, but the user can re-analyze for
        #                            the exact steady-state if needed.)
        #   series   → I_total · s_i (with ± sign per coil)
        branch_I: dict = {}
        if kind == 'parallel' and len(coil_ids) >= 1:
            per_branch = float(amps) / max(len(coil_ids), 1)
            for cid in coil_ids:
                branch_I[cid] = per_branch
        else:
            for cid in coil_ids:
                branch_I[cid] = float(amps) * float(signs.get(cid, 1))

        # Two-pass update:
        #   Pass 1 — write every circuit member's branch current into the
        #            params map + multi_env. Each `update_coil_params`
        #            marks every known coil stale as a side effect, so
        #            doing this up front means we don't interleave
        #            re-staling with per-coil rescales.
        #   Pass 2 — rescale every member's full engine in place.
        #   Finally — clear staleness for every circuit member (they're all
        #             now consistent with the circuit's current). Coils
        #             OUTSIDE the circuit correctly remain stale because
        #             their B_ext from these members has changed.
        for cid in coil_ids:
            if cid not in self._coil_params_map:
                continue
            p = self._coil_params_map[cid]
            p['current'] = float(branch_I.get(cid, 0.0))
            self._multi_env.update_coil_params(
                cid, winds=p['winds'], current=p['current'],
                thickness=p['thickness'], width=p['width'],
            )

        rescaled_ids: list = []
        for cid in coil_ids:
            engine = self._coil_engines.get(cid)
            if engine is None:
                # Coil was never analyzed — can't rescale. Leave it stale
                # so Re-analyze All picks it up.
                continue
            target_I = float(branch_I.get(cid, 0.0))
            try:
                # force=True: circuit-level current changes scale every
                # member by the same ratio, so the stored B_ext component
                # rescales correctly alongside B_self — the
                # _analyzed_with_ext safety check doesn't apply here.
                if engine.rescale_to_current(target_I, force=True):
                    self.workspace.add_force_layer(engine, cid)
                    self.workspace.add_stress_layer(engine, cid)
                    self.workspace.add_axis_layer(engine, cid)
                    rescaled_ids.append(cid)
            except Exception:
                pass

        self.workspace.rescale_all_force_layers()

        # Clear staleness ONLY for coils whose engines actually rescaled.
        # Un-analyzed grouped coils stay stale so Re-analyze All finds them.
        for cid in rescaled_ids:
            self._multi_env.mark_fresh(cid)
            for nm in ('Forces', 'Stress', 'B Axis',
                       'Field Lines', 'Cross Section'):
                self.browser.mark_layer_stale(cid, nm, False)

        self._propagate_staleness()
        # If a member coil is currently displayed, show its actual branch
        # current (which differs from the circuit total for parallel wiring).
        if self._active_coil_id in coil_ids:
            actual = float(branch_I.get(self._active_coil_id, amps))
            self.props.set_coil_current_editable(
                False,
                inherited_from=g.get('name', group_id),
                inherited_value=actual,
            )
            self._refresh_summary_for(self._active_coil_id)

    def _on_group_as_series(self) -> None:
        self._create_circuit_group('series')

    def _on_group_as_parallel(self) -> None:
        self._create_circuit_group('parallel')

    def _create_circuit_group(self, kind: str) -> None:
        """Build a new circuit group from the current multi-selection.
        Each coil may only belong to one group; any previous group
        membership is dissolved first. Reparents the member coils
        underneath a new folder-like header in the browser."""
        sel = self.browser.selected_coil_ids()
        if len(sel) < 2:
            return
        # Dissolve any existing groups touched by this selection so coils
        # don't belong to two groups at once.
        touched_groups = {self._coil_group_map[cid]
                          for cid in sel if cid in self._coil_group_map}
        for gid in touched_groups:
            self._dissolve_group(gid)

        self._circuit_counter += 1
        gid = f"circuit_{self._circuit_counter}"
        color = self._next_group_color()
        # Initial circuit current inherits from the first selected coil's
        # current (reasonable default — user can change it via the circuit
        # view and it'll propagate to all members).
        first_current = float(
            self._coil_params_map.get(sel[0], {}).get('current', 0.0)
        )
        self._circuit_groups[gid] = {
            'kind':    kind,
            'coil_ids': list(sel),
            'signs':   {cid: 1 for cid in sel},   # all + for MVP
            'color':   color,
            'name':    f"Circuit {self._circuit_counter}",
            'current': first_current,
        }
        for cid in sel:
            self._coil_group_map[cid] = gid
        # Update the browser: create the header above the first member
        # (so it reads like a folder), then badge each member coil.
        name = self._circuit_groups[gid]['name']
        self.browser.add_circuit_header(gid, name, kind, color,
                                          insert_above=list(sel))
        for cid in sel:
            self.browser.move_coil_under_circuit(cid, gid)
        # Compute the correct branch-current split (handles parallel → I/N
        # or L-matrix-weighted, and series → signed total) and push to all
        # member coils, engines, and visualizations in one place.
        self._on_circuit_current_changed(gid, first_current)
        self._main.ribbon.set_circuit_enabled(group_ok=True, ungroup_ok=True)

    def _on_ungroup_selection(self) -> None:
        """Dissolve every circuit group that contains a currently-selected coil."""
        sel = self.browser.selected_coil_ids() or (
            [self._active_coil_id] if self._active_coil_id else []
        )
        touched = {self._coil_group_map[cid]
                   for cid in sel if cid in self._coil_group_map}
        for gid in touched:
            self._dissolve_group(gid)
        if self._active_coil_id:
            # Coil is no longer in a circuit — current is editable again
            if self._active_coil_id not in self._coil_group_map:
                self.props.set_coil_current_editable(True)
            self._refresh_summary_for(self._active_coil_id)
        self._main.ribbon.set_circuit_enabled(
            group_ok=len(sel) >= 2,
            ungroup_ok=False,
        )

    def _dissolve_group(self, group_id: str) -> None:
        g = self._circuit_groups.pop(group_id, None)
        if not g:
            return
        for cid in g['coil_ids']:
            self._coil_group_map.pop(cid, None)
        # Browser: reparent member coils back to COILS root, delete header
        self.browser.remove_circuit_header(group_id)

    def _compute_circuit_inductance(self, group_id: str) -> float | None:
        """Return the effective inductance of a circuit group, in Henries.

        Series (with per-coil signs s_i = ±1):
            L = Σ_i Σ_j s_i s_j  M_ij

        Parallel (tight-coupling aware — NOT `1 / Σ(1/L_i)`):
            L = 1 / (1ᵀ · M⁻¹ · 1)

        Requires every coil in the group to have a fresh engine (i.e. the
        user has run analysis on each). Returns None if unavailable or
        ill-conditioned."""
        g = self._circuit_groups.get(group_id)
        if not g:
            return None
        coil_ids = [cid for cid in g['coil_ids'] if cid in self._coil_engines]
        if len(coil_ids) != len(g['coil_ids']):
            return None   # some coils not analyzed yet
        try:
            M_full = self._multi_env.compute_mutual_inductance_matrix()
        except Exception:
            return None
        full_ids = M_full.get('coil_ids', [])
        L = M_full.get('L_matrix')
        if L is None:
            return None
        try:
            idx = [full_ids.index(cid) for cid in coil_ids]
        except ValueError:
            return None
        M_sub = L[np.ix_(idx, idx)]
        signs = np.array([g['signs'].get(cid, 1) for cid in coil_ids],
                         dtype=np.float64)
        kind = g.get('kind', 'series')
        try:
            if kind == 'parallel':
                # L_parallel = 1 / (1ᵀ M⁻¹ 1)
                Minv = np.linalg.inv(M_sub)
                ones = np.ones(len(idx), dtype=np.float64)
                denom = float(ones @ Minv @ ones)
                if denom <= 0:
                    return None
                return 1.0 / denom
            # series (default)
            return float(signs @ M_sub @ signs)
        except np.linalg.LinAlgError:
            return None

    def _schedule_l_matrix_precompute(self) -> None:
        """Start (or skip) a background L-matrix compute so the cache is warm
        before the user clicks a circuit header. No-op when:
          - no circuit groups exist,
          - fewer than 2 engines are registered (nothing to compute),
          - cache is already warm for the current engine set,
          - a precompute is already running.
        """
        if not self._circuit_groups:
            return
        env = self._multi_env
        try:
            engine_ids = list(env._engines.keys())
        except Exception:
            return
        if len(engine_ids) < 1:
            return
        cache = getattr(env, '_L_cache', None)
        if cache is not None and cache.get('coil_ids') == engine_ids:
            return
        if self._lmx_thread is not None and self._lmx_thread.isRunning():
            return
        self._lmx_thread = QThread(self)
        self._lmx_worker = LMatrixWorker(env)
        self._lmx_worker.moveToThread(self._lmx_thread)
        self._lmx_worker.finished.connect(self._on_l_matrix_ready)
        self._lmx_worker.finished.connect(self._lmx_thread.quit)
        self._lmx_worker.finished.connect(self._lmx_worker.deleteLater)
        self._lmx_thread.finished.connect(self._lmx_thread.deleteLater)
        self._lmx_thread.started.connect(self._lmx_worker.run)
        self._lmx_thread.start()

    def _on_l_matrix_ready(self, coil_ids) -> None:
        """Background L-matrix compute finished — cache is now warm. Refresh
        the Properties panel if a circuit view or grouped coil is showing so
        its L reading updates from the placeholder."""
        self._lmx_thread = None
        self._lmx_worker = None
        if coil_ids is None:
            return
        gid = getattr(self, '_active_circuit_id', None)
        if gid and gid in self._circuit_groups:
            try:
                L_c = self._compute_circuit_inductance(gid)
            except Exception:
                L_c = None
            if L_c is not None:
                self.props.update_circuit_inductance(gid, L_c)
        elif self._active_coil_id \
                and self._active_coil_id in self._coil_group_map:
            self._refresh_summary_for(self._active_coil_id)

    def _on_coil_param_changed(self) -> None:
        """A spinbox value changed — apply it to the active coil (or all coils
        in a multi-edit selection). When the change is current-only and the
        coil was analyzed without an external B-field, rescale the engine's
        results in place (O(n) refresh) instead of marking it stale for a
        full Biot-Savart re-sum. Otherwise fall back to the stale path."""
        ui_params = self.props.get_params()
        if self._multi_edit_ids:
            targets = list(self._multi_edit_ids)
        elif self._active_coil_id:
            targets = [self._active_coil_id]
        else:
            return

        geom_keys = ('winds', 'thickness', 'width', 'axis_num')
        for cid in targets:
            old = dict(self._coil_params_map.get(cid, {}))
            new = dict(old)
            new.update(ui_params)
            # If this coil is part of a circuit, current is owned by the
            # circuit — never let the per-coil UI overwrite it here.
            gid = self._coil_group_map.get(cid)
            if gid and gid in self._circuit_groups:
                new['current'] = float(
                    self._circuit_groups[gid].get('current', old.get('current', 0.0))
                )
            self._coil_params_map[cid] = new

            only_current_changed = (
                old.get('current') != new.get('current')
                and all(old.get(k) == new.get(k) for k in geom_keys)
            )
            engine = self._coil_engines.get(cid)
            rescaled = False
            if only_current_changed and engine is not None:
                try:
                    rescaled = bool(engine.rescale_to_current(new['current']))
                except Exception:
                    rescaled = False

            # Keep MultiCoilEnvironment in sync — this also marks every
            # coil stale (including this one, which we clear below if the
            # rescale succeeded).
            self._multi_env.update_coil_params(
                cid, winds=new['winds'], current=new['current'],
                thickness=new['thickness'], width=new['width'],
                winding_growth=new.get('stack_growth', 'symmetric'),
            )

            if rescaled:
                # Push the rescaled engine data into the visualization in
                # place — no tube rebuild needed (geometry unchanged).
                self.workspace.add_force_layer(engine, cid)
                self.workspace.add_stress_layer(engine, cid)
                self.workspace.add_axis_layer(engine, cid)
                self.workspace.rescale_all_force_layers()
                # This coil's analysis is still current after the rescale.
                self._multi_env.mark_fresh(cid)
                for nm in ('Forces', 'Stress', 'B Axis',
                           'Field Lines', 'Cross Section'):
                    self.browser.mark_layer_stale(cid, nm, False)
                # Refresh Properties summary if it's the visible coil
                if (not self._multi_edit_ids and cid == self._active_coil_id) \
                        or (targets and cid == targets[0]):
                    self.props.update_summary(engine)
            else:
                # Geometry changed or external field present — rebuild the
                # tube mesh and let the user click Re-analyze to recompute.
                total_t = new['thickness'] * 1e-6 * new['winds']
                tape_w  = new['width'] * 1e-3
                self.workspace.update_coil_mesh(
                    cid, total_t, tape_w,
                    tape_normals=new.get('tape_normals'),
                    winding_growth=new.get('stack_growth', 'symmetric'),
                )
        self._propagate_staleness()

    def _load_coil_params(self, coil_id: str) -> None:
        """Load per-coil parameters into PropertiesPanel spinboxes.

        Honors the per-coil saved unit choices (thickness_unit, width_unit)
        so the displayed value matches what the user typed last time.
        Stack growth is also restored.
        """
        p = self._coil_params_map.get(coil_id)
        if not p:
            return
        # Block signals on every editable widget so programmatic loads don't
        # trigger _on_coil_param_changed mid-update.
        widgets = (
            self.props.spin_winds, self.props.dspin_current,
            self.props.dspin_thick, self.props.dspin_width,
            self.props.spin_axis_pts,
            self.props.cmb_thick_unit, self.props.cmb_width_unit,
            self.props.cmb_stack,
        )
        for w in widgets:
            w.blockSignals(True)
        self.props.spin_winds.setValue(p['winds'])
        self.props.dspin_current.setValue(p['current'])
        # Apply the saved display units, then convert canonical → displayed.
        t_idx = int(p.get('thickness_unit', 0))
        w_idx = int(p.get('width_unit', 0))
        self.props.cmb_thick_unit.setCurrentIndex(t_idx)
        self.props._thick_unit_idx = t_idx
        self.props.cmb_width_unit.setCurrentIndex(w_idx)
        self.props._width_unit_idx = w_idx
        t_factor = self.props._thick_units[t_idx][1]
        w_factor = self.props._width_units[w_idx][1]
        self.props.dspin_thick.setValue(p['thickness'] / t_factor)
        self.props.dspin_width.setValue(p['width'] / w_factor)
        self.props.spin_axis_pts.setValue(p['axis_num'])
        # Stack growth: 'symmetric' → 0, 'up' → 1.
        self.props.cmb_stack.setCurrentIndex(
            1 if p.get('stack_growth') == 'up' else 0
        )
        for w in widgets:
            w.blockSignals(False)
        # Mark the Properties panel as displaying THIS coil — the save-back
        # gate in _on_coil_selected uses this to refuse to write the UI into
        # any coil it wasn't actually showing.
        self._props_showing_coil_id = coil_id

    def _on_coil_renamed(self, coil_id: str, new_name: str) -> None:
        self._coil_names[coil_id] = new_name

    def _on_coil_recolored(self, coil_id: str, color: str) -> None:
        self.workspace.set_coil_color(coil_id, color)

    def _on_run_analysis(self) -> None:
        if self._coords is None:
            QMessageBox.information(
                self, "No Coil", "Load a coil CSV before running analysis."
            )
            return
        # Save spinbox values only if the user is looking at this coil
        # (during re-analyze-all, active_coil_id is temporarily changed
        # but the spinboxes still show a different coil's values).
        cid_save = self._active_coil_id
        # Save UI back into the active coil's stored params ONLY when:
        #   • the active coil is real and known
        #   • we're not in a reanalyze flow (queue empty), AND
        #   • the Properties panel is actually showing this coil's values.
        # The last condition is the critical one — Reanalyze All temporarily
        # flips _active_coil_id between coils without updating the UI, so
        # saving here would overwrite the freshly-analyzed coil's params
        # with whatever coil the user was viewing.
        if (cid_save
                and cid_save in self._coil_params_map
                and not getattr(self, '_reanalyze_queue', None)
                and self._props_showing_coil_id == cid_save):
            ui_params = self.props.get_params()
            # Current is owned by the circuit when this coil is grouped —
            # don't overwrite the branch current with a stale UI value.
            if cid_save in self._coil_group_map:
                ui_params.pop('current', None)
            self._coil_params_map[cid_save].update(ui_params)

        self._main.ribbon.set_run_enabled(False)
        self._main.ribbon.set_inspect_enabled(False)

        # Clear only THIS coil's previous analysis — other coils keep theirs
        cid = self._active_coil_id
        self.workspace.clear_analysis_layers(cid)
        self.browser.remove_all_analysis_from_coil(cid)
        self._analyzed_coil_id = cid

        # Use world-space coords so physics sees the coil where it actually is
        _wc = self.workspace.get_transformed_coords(cid)
        world_coords = _wc if _wc is not None else self._coords

        # Use stored params (not spinbox) — during re-analyze-all the
        # spinboxes may show a different coil's values.
        params = self._coil_params_map.get(cid, self.props.get_params())
        self.reporter = ProgressReporter(self, title="Running Analysis…")
        self.reporter.start()

        # Build the external field callback (superposition from other coils)
        B_ext = self._multi_env.make_external_field_func(cid)

        self._a_thread = QThread(self)
        self._a_worker = AnalysisWorker(
            world_coords,
            params['winds'], params['current'],
            params['thickness'], params['width'],
            axis_num=params['axis_num'],
            B_ext=B_ext,
            tape_normals=params.get('tape_normals'),
            winding_growth=params.get('stack_growth', 'symmetric'),
        )
        self._a_worker.moveToThread(self._a_thread)
        # Scale worker progress to 0-85%; remaining 15% is layer building
        self._a_worker.progress.connect(lambda pct: self.reporter.report(int(pct * 0.85)))
        self._a_worker.stage.connect(self.reporter.set_stage)
        self._a_worker.finished.connect(self._on_analysis_done)
        self._a_worker.finished.connect(self._a_thread.quit)
        self._a_worker.finished.connect(self._a_worker.deleteLater)
        self._a_thread.finished.connect(self._a_thread.deleteLater)
        self._a_thread.started.connect(self._a_worker.run)
        self._a_thread.start()

    def _on_analysis_done(self, engine) -> None:
        self._a_thread = None
        self._a_worker = None

        cid = self._analyzed_coil_id
        self._coil_engines[cid] = engine
        self._multi_env.mark_fresh(cid)
        # Clear stale marks for this coil — its analysis is now current
        for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
            self.browser.mark_layer_stale(cid, nm, False)

        self.reporter.set_stage("Computing per-vertex forces…")
        self.reporter.report(87)

        def _force_progress(done, total):
            self.reporter.report(87 + int(8 * done / max(total, 1)))

        force_scalars = self.workspace.add_force_layer(
            engine, cid, progress_callback=_force_progress,
        )
        # Cache the per-tube-vertex scalars so session save can embed
        # them in the .calcsx. On reload, add_force_layer_from_scalars
        # bypasses the per-vertex Biot-Savart sweep — that's the one
        # step in load that scales with tube_mesh resolution × n_fil
        # and drives the 82→100% wall time on a large coil.
        if force_scalars is not None:
            self._coil_inspect_cache.setdefault(cid, {})['force_scalars'] = \
                np.asarray(force_scalars, dtype=np.float32)

        self.reporter.set_stage("Building stress & field layers…")
        self.reporter.report(95)

        self.workspace.add_stress_layer(engine, cid)
        self.workspace.add_axis_layer(engine, cid)

        # Unify force colour scale — defer during re-analyze-all so the
        # global range accounts for ALL coils, not just those done so far.
        if not getattr(self, '_reanalyze_queue', None):
            self.workspace.rescale_all_force_layers()

        self.reporter.finish()

        # Snapshot the transform state at analysis time so apply_coil_transform
        # can compute the correct delta when the coil is moved later.
        self.workspace.mark_analysis_transform(cid)

        # When analysis was auto-triggered by an inspect button (not Run Analysis),
        # hide the physics layers — user only wanted field lines / cross section.
        auto = self._analysis_auto_triggered
        self._analysis_auto_triggered = False

        if cid:
            for nm in ('Forces', 'Stress', 'B Axis'):
                self.browser.add_layer_to_coil(cid, nm, visible=not auto)

        self.props.update_summary(engine)
        self._main.ribbon.set_run_enabled(True)
        self._main.ribbon.set_inspect_enabled(True)
        # If global field mode is active, keep per-coil field lines disabled
        if self._main.ribbon._btn_global_field.isChecked():
            self._main.ribbon._btn_field_lines.set_action_enabled(False)
        # Enable "Re-analyze All" when there are stale coils in the environment
        self._main.ribbon._btn_reanalyze.set_action_enabled(
            bool(self._multi_env.get_stale_coils())
        )

        # If inspect was requested before analysis existed, chain it using the
        # coil that was just analyzed (not _active_coil_id which may have changed).
        if self._pending_inspect == 'field_lines':
            self._pending_inspect = None
            self._run_field_lines_for(cid, engine)
        elif self._pending_inspect == 'cross_section':
            self._pending_inspect = None
            self._run_cross_section_for(cid, engine)
        # Continue re-analyze-all queue if active
        elif getattr(self, '_reanalyze_queue', None):
            self._reanalyze_next()
        else:
            # Warm the L-matrix cache in the background so the first
            # circuit-header click doesn't block the UI.
            self._schedule_l_matrix_precompute()

        # Refresh the System Energy meter (if installed) — analysis-done
        # marks fresh, which _propagate_staleness doesn't see, so we hit it
        # here explicitly. Stray arrays follow the same logic.
        self._update_system_energy_readout()
        self._update_all_stray_arrays()

    def _on_reanalyze_all(self) -> None:
        """Re-run analysis on ALL coils sequentially."""
        self._reanalyze_queue = [
            cid for cid in self._coil_coords if cid in self._coil_params_map
        ]
        self._reanalyze_next()

    def _reanalyze_next(self) -> None:
        """Pop the next stale coil and run its analysis."""
        if not getattr(self, '_reanalyze_queue', None):
            self._main.ribbon._btn_reanalyze.set_action_enabled(False)
            # All coils done — now apply the global force colour scale
            self.workspace.rescale_all_force_layers()
            # Warm the L-matrix cache once the full queue is drained.
            self._schedule_l_matrix_precompute()
            return
        if self._a_thread is not None and self._a_thread.isRunning():
            return  # wait for current analysis to finish
        cid = self._reanalyze_queue.pop(0)
        # Temporarily switch active coil context for analysis
        saved_active = self._active_coil_id
        self._active_coil_id = cid
        self._coords = self._coil_coords.get(cid)
        self._on_run_analysis()
        self._active_coil_id = saved_active
        self._coords = self._coil_coords.get(saved_active)

    def _on_global_field_toggled(self, checked: bool) -> None:
        """Toggle global field lines — mutually exclusive with per-coil field lines."""
        if checked:
            # Snapshot current eye states before hiding
            self._pre_global_eye_state = {}
            for cid in list(self._coil_coords.keys()):
                eye_on = self.browser.get_layer_eye_state(cid, 'Field Lines')
                self._pre_global_eye_state[cid] = eye_on
                if self.workspace.has_layer(cid, 'Field Lines'):
                    self.workspace.set_layer_visible(cid, 'Field Lines', False)
                self.browser.set_layer_eye_locked(cid, 'Field Lines', locked=True)
            # Disable per-coil field lines button while global is on
            self._main.ribbon._btn_field_lines.set_action_enabled(False)
            # Use cached result if available, environment unchanged, and
            # seed count matches what was used for the cache.
            n_seeds = self.props.get_field_seeds()
            cache_valid = (not self._global_fl_dirty
                           and self._global_fl_cache is not None
                           and getattr(self, '_global_fl_cache_seeds', 0) == n_seeds)
            if cache_valid:
                lines, B_mags = self._global_fl_cache
                self.workspace.add_field_lines_layer(lines, B_mags, 'global')
                self.workspace.rescale_all_field_line_layers()
            else:
                self._compute_global_field_lines()
        else:
            # Remove global field lines layer
            self.workspace.clear_field_lines_layer('global')
            # Restore per-coil field lines to their pre-global eye state
            self._main.ribbon._btn_field_lines.set_action_enabled(True)
            for cid in list(self._coil_coords.keys()):
                was_on = self._pre_global_eye_state.get(cid, True)
                self.browser.set_layer_eye_unlocked(cid, 'Field Lines', restore_checked=was_on)

    def _compute_global_field_lines(self) -> None:
        """Compute field lines through the superposed B-field of all coils."""
        if self._i_thread is not None and self._i_thread.isRunning():
            return
        B_total = self._multi_env.make_total_field_func()
        coil_infos = self._multi_env.get_coil_infos()
        if B_total is None or not coil_infos:
            return
        n_seeds = self.props.get_field_seeds()
        self._inspect_reporter = ProgressReporter(self, title="Computing Global Field Lines…")
        self._inspect_reporter.start()
        self._i_thread = QThread(self)
        self._i_worker = GlobalFieldLinesWorker(B_total, coil_infos, n_seeds)
        self._i_worker.moveToThread(self._i_thread)
        self._i_worker.progress.connect(self._inspect_reporter.report)
        self._i_worker.finished.connect(self._on_global_field_lines_done)
        self._i_worker.finished.connect(self._i_thread.quit)
        self._i_worker.finished.connect(self._i_worker.deleteLater)
        self._i_thread.finished.connect(self._i_thread.deleteLater)
        self._i_thread.started.connect(self._i_worker.run)
        self._i_thread.start()

    def _on_global_field_lines_done(self, data) -> None:
        self._inspect_reporter.finish()
        self._i_thread = None
        self._i_worker = None
        lines, B_mags = data
        self._global_fl_cache = (lines, B_mags)
        self._global_fl_cache_seeds = self.props.get_field_seeds()
        self._global_fl_dirty = False
        self.workspace.add_field_lines_layer(lines, B_mags, 'global')
        self.workspace.rescale_all_field_line_layers()

    def _on_compute_field_lines(self) -> None:
        if self._i_thread is not None and self._i_thread.isRunning():
            return
        cid = self._active_coil_id
        engine = self._coil_engines.get(cid)
        if engine is None:
            self._pending_inspect = 'field_lines'
            self._analysis_auto_triggered = True
            self._on_run_analysis()
            return
        # Reuse cached field lines when the coil is fresh and the seed
        # count matches — matches the "persistent" behaviour of Global Field.
        ic = self._coil_inspect_cache.get(cid, {})
        stale_cids = set(self._multi_env.get_stale_coils())
        cached_seeds = ic.get('field_seeds')
        n_seeds = int(self.props.get_field_seeds())
        have_lines = ic.get('field_lines') is not None
        have_mags  = ic.get('field_mags')  is not None
        # If seeds weren't recorded (older session file), assume the current
        # UI value is the user's intent; otherwise require an exact match.
        seeds_ok = (cached_seeds is None) or (cached_seeds == n_seeds)
        if (cid not in stale_cids and have_lines and have_mags and seeds_ok):
            lines  = ic['field_lines']
            B_mags = ic['field_mags']
            self.workspace.add_field_lines_layer(lines, B_mags, cid)
            self.workspace.rescale_all_field_line_layers()
            self.browser.add_layer_to_coil(cid, 'Field Lines', deletable=True)
            if self._main.ribbon._btn_global_field.isChecked():
                self.workspace.set_layer_visible(cid, 'Field Lines', False)
                self.browser.set_layer_eye_locked(cid, 'Field Lines', locked=True)
            return
        self._run_field_lines_for(cid, engine)

    def _run_field_lines_for(self, cid: str, engine) -> None:
        """Start field-line computation for the given coil+engine."""
        self.workspace.clear_field_lines_layer(cid)
        if cid:
            self.browser.remove_layer_from_coil(cid, 'Field Lines')
        self._inspect_coil_id = cid
        n_seeds = self.props.get_field_seeds()
        self._inspect_reporter = ProgressReporter(self, title="Computing Field Lines…")
        self._inspect_reporter.start()
        self._i_thread = QThread(self)
        self._i_worker = FieldLinesWorker(engine, n_seeds)
        self._i_worker.moveToThread(self._i_thread)
        self._i_worker.progress.connect(self._inspect_reporter.report)
        self._i_worker.finished.connect(self._on_field_lines_done)
        self._i_worker.finished.connect(self._i_thread.quit)
        self._i_worker.finished.connect(self._i_worker.deleteLater)
        self._i_thread.finished.connect(self._i_thread.deleteLater)
        self._i_thread.started.connect(self._i_worker.run)
        self._i_thread.start()

    def _on_field_lines_done(self, data) -> None:
        self._inspect_reporter.finish()
        self._i_thread = None
        self._i_worker = None
        lines, B_mags = data
        cid = self._inspect_coil_id
        self.workspace.add_field_lines_layer(lines, B_mags, cid)
        self.workspace.rescale_all_field_line_layers()
        if cid:
            # Cache for session save/restore
            cache = self._coil_inspect_cache.setdefault(cid, {})
            cache['field_lines'] = [np.asarray(l, dtype=float) for l in lines]
            cache['field_mags']  = [np.asarray(m, dtype=float) for m in B_mags]
            cache['field_seeds'] = int(self.props.get_field_seeds())
            self.browser.add_layer_to_coil(cid, 'Field Lines', deletable=True)
            # If global field mode is active, hide per-coil lines immediately
            if self._main.ribbon._btn_global_field.isChecked():
                self.workspace.set_layer_visible(cid, 'Field Lines', False)
                self.browser.set_layer_eye_locked(cid, 'Field Lines', locked=True)

    def _on_compute_cross_section(self) -> None:
        if self._i_thread is not None and self._i_thread.isRunning():
            return
        cid = self._active_coil_id
        engine = self._coil_engines.get(cid)
        if engine is None:
            self._pending_inspect = 'cross_section'
            self._analysis_auto_triggered = True
            self._on_run_analysis()
            return
        # Reuse cached cross section when the coil is fresh and the
        # section-offset matches the current UI value.
        ic = self._coil_inspect_cache.get(cid, {})
        stale_cids = set(self._multi_env.get_stale_coils())
        cs_cache = ic.get('cross_section')
        cur_offset = float(self.props.get_cs_offset())
        cached_offset = cs_cache.get('offset') if cs_cache else None
        # Treat a missing cached offset (older sessions) as a match.
        offset_ok = (cached_offset is None) or (abs(cached_offset - cur_offset) < 1e-9)
        if cid not in stale_cids and cs_cache is not None and offset_ok:
            self.workspace.add_cross_section_layer(
                cs_cache['X'], cs_cache['Y'], cs_cache['B_plane'],
                cs_cache['e1'], cs_cache['e2'], cs_cache['center'],
                cs_cache['R'], cid,
            )
            self.browser.add_layer_to_coil(cid, 'Cross Section', deletable=True)
            return
        self._run_cross_section_for(cid, engine)

    def _run_cross_section_for(self, cid: str, engine) -> None:
        """Start cross-section computation for the given coil+engine."""
        self.workspace.clear_cross_section_layer(cid)
        if cid:
            self.browser.remove_layer_from_coil(cid, 'Cross Section')
        self._inspect_coil_id = cid
        axis_offset = self.props.get_cs_offset()
        self._inspect_reporter = ProgressReporter(self, title="Computing Cross Section…")
        self._inspect_reporter.start()
        self._i_thread = QThread(self)
        self._i_worker = CrossSectionWorker(engine, axis_offset=axis_offset)
        self._i_worker.moveToThread(self._i_thread)
        self._i_worker.progress.connect(self._inspect_reporter.report)
        self._i_worker.finished.connect(self._on_cross_section_done)
        self._i_worker.finished.connect(self._i_thread.quit)
        self._i_worker.finished.connect(self._i_worker.deleteLater)
        self._i_thread.finished.connect(self._i_thread.deleteLater)
        self._i_thread.started.connect(self._i_worker.run)
        self._i_thread.start()

    def _on_cross_section_done(self, data) -> None:
        self._inspect_reporter.finish()
        self._i_thread = None
        self._i_worker = None
        X, Y, B_plane, e1, e2, center, R = data
        cid = self._inspect_coil_id
        self.workspace.add_cross_section_layer(X, Y, B_plane, e1, e2, center, R, cid)
        if cid:
            cache = self._coil_inspect_cache.setdefault(cid, {})
            cache['cross_section'] = {
                'X': np.asarray(X, dtype=float),
                'Y': np.asarray(Y, dtype=float),
                'B_plane': np.asarray(B_plane, dtype=float),
                'e1': np.asarray(e1, dtype=float),
                'e2': np.asarray(e2, dtype=float),
                'center': np.asarray(center, dtype=float),
                'R': float(R),
                'offset': float(self.props.get_cs_offset()),
            }
            self.browser.add_layer_to_coil(cid, 'Cross Section', deletable=True)

    def _on_translate_toggled(self, checked: bool) -> None:
        # Pin guard: pinned coils can't be translated via the gizmo.
        if checked and self._active_coil_id in self._pinned_coils:
            self._main.ribbon._btn_translate.blockSignals(True)
            self._main.ribbon._btn_translate.setChecked(False)
            self._main.ribbon._btn_translate.blockSignals(False)
            QMessageBox.information(
                self, "Coil Pinned",
                "The active coil is pinned. Unpin it in the CONSTRUCT "
                "tab before moving it.",
            )
            return
        if checked:
            self.workspace.show_gizmo('T')
        else:
            self.workspace.hide_gizmo()

    def _on_rotate_toggled(self, checked: bool) -> None:
        if checked and self._active_coil_id in self._pinned_coils:
            self._main.ribbon._btn_rotate.blockSignals(True)
            self._main.ribbon._btn_rotate.setChecked(False)
            self._main.ribbon._btn_rotate.blockSignals(False)
            QMessageBox.information(
                self, "Coil Pinned",
                "The active coil is pinned. Unpin it in the CONSTRUCT "
                "tab before rotating it.",
            )
            return
        if checked:
            self.workspace.show_gizmo('R')
        else:
            self.workspace.hide_gizmo()

    def _on_reset_transform(self) -> None:
        # Uncheck both toggles (fires hide_gizmo via toggled signal if either was active).
        self._main.ribbon._btn_translate.setChecked(False)
        self._main.ribbon._btn_rotate.setChecked(False)
        # Always hide gizmo explicitly — covers the case where neither button
        # was checked but gizmo is somehow still visible.
        self.workspace.hide_gizmo()
        self.workspace.reset_coil_transform()
        # Clear stale marks: transform was reset, layers are back in original position
        cid = self._active_coil_id
        if cid:
            for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                self.browser.mark_layer_stale(cid, nm, False)

    def _on_layers_stale(self) -> None:
        """Called by workspace whenever the active coil or probe is transformed."""
        # If probe is the gizmo target, just update readout
        if self.workspace.get_gizmo_target() == 'probe':
            pid = self.workspace._active_probe_id
            if pid:
                self._update_single_probe_readout(pid)
            return
        cid = self._active_coil_id
        if cid and cid in self._coil_engines:
            for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                self.browser.mark_layer_stale(cid, nm, True)
        # Keep the CONSTRUCT-tab VALUES spinboxes in lockstep with gizmo drags.
        self._sync_ribbon_transform_ui(cid)
        # Update superposition environment with new world-space coords
        if cid:
            wc = self.workspace.get_transformed_coords(cid)
            if wc is not None:
                self._multi_env.update_coil_coords(cid, wc)
            self._propagate_staleness()

    def _export_vtk_layers(self, parent_dlg: QDialog) -> None:
        """Prompt for output folder, then export coils/analysis as .vtp."""
        out_dir = QFileDialog.getExistingDirectory(
            parent_dlg, "Select VTK Export Folder", "",
            QFileDialog.ShowDirsOnly,
        )
        if not out_dir:
            return
        exported = self.workspace.export_vtk_layers(out_dir)
        if exported:
            QMessageBox.information(
                parent_dlg, "VTK Export Complete",
                f"Exported {len(exported)} .vtp file(s) to:\n{out_dir}\n\n"
                "Open in ParaView to visualize.",
            )
        else:
            QMessageBox.warning(
                parent_dlg, "Nothing Exported",
                "No meshes found to export. Load coils and run analysis first.",
            )

    def _export_web_layers(self, parent_dlg: QDialog) -> None:
        """Export dark + light web demo layers as glTF files."""
        out_dir = QFileDialog.getExistingDirectory(
            parent_dlg, "Select Output Folder", "",
            QFileDialog.ShowDirsOnly,
        )
        if not out_dir:
            return

        name, ok = QInputDialog.getText(
            parent_dlg, "Demo Name",
            "Enter a short name for this demo (e.g. tokamak, solenoid):",
            text="demo",
        )
        if not ok or not name.strip():
            return
        name = name.strip().lower().replace(' ', '-')

        from CalcSX_app.gui.gui_utils import get_theme_name
        original_theme = get_theme_name()
        total = []

        for theme in ('dark', 'light'):
            folder = os.path.join(out_dir, f"weblayers-{name}-{theme}")
            exported = self.workspace.export_web_layers(folder, theme)
            total.extend(exported)

        # Restore original theme (the export doesn't change the live theme,
        # but guard against future changes)
        if get_theme_name() != original_theme:
            self._main._apply_theme(original_theme)

        if total:
            QMessageBox.information(
                parent_dlg, "Web Export Complete",
                f"Exported {len(total)} glTF file(s) to:\n{out_dir}\n\n"
                f"  weblayers-{name}-dark/\n  weblayers-{name}-light/",
            )
        else:
            QMessageBox.warning(
                parent_dlg, "Nothing Exported",
                "No meshes found. Load coils and run analysis first.",
            )

    def _clear_all(self) -> None:
        """Remove every coil, bobbin, layer, and engine — reset to empty state."""
        # Delete all coils (which also removes their analysis layers)
        for cid in list(self._coil_coords.keys()):
            self.workspace.clear_analysis_layers(cid)
            self.workspace.remove_coil(cid)
            self.browser.remove_coil_item(cid)
        # Delete bobbin layers
        for (bid, lname) in list(self.workspace._layers.keys()):
            if lname == 'Bobbin':
                for actor in self.workspace._layers[(bid, lname)].actors:
                    try:
                        self.workspace._plotter.remove_actor(actor, render=False)
                    except Exception:
                        pass
                del self.workspace._layers[(bid, lname)]
                self.browser.remove_coil_item(bid)
        # Clear global field lines
        self.workspace.clear_field_lines_layer('global')
        if self._main.ribbon._btn_global_field.isChecked():
            self._main.ribbon._btn_global_field.blockSignals(True)
            self._main.ribbon._btn_global_field.setChecked(False)
            self._main.ribbon._btn_global_field.blockSignals(False)
        # Reset state
        self._coil_coords.clear()
        self._coil_paths.clear()
        self._coil_names.clear()
        self._coil_params_map.clear()
        self._coil_engines.clear()
        self._coil_inspect_cache.clear()
        self._global_fl_cache = None
        self._global_fl_dirty = True
        # Remove all Hall probes
        for pid in list(self.workspace._probe_entries.keys()):
            self.workspace.remove_hall_probe(pid)
            self.browser.remove_probe_item(pid)
        self._probe_state.clear()
        if self._probe_timer is not None:
            self._probe_timer.stop()
            self._probe_timer = None
        self._multi_env = MultiCoilEnvironment()
        self._circuit_groups.clear()
        self._coil_group_map.clear()
        self._circuit_counter = 0
        self._pinned_coils.clear()
        # Drop the System Energy meter row (the underlying multi_env is being
        # reset; the meter would read 0 / error otherwise).
        if self._has_system_energy:
            self.browser.remove_system_energy_item()
            self._has_system_energy = False
        # Drop every stray-field array (workspace actors + browser rows).
        for aid in list(self._stray_arrays.keys()):
            self.workspace.remove_stray_array(aid)
            self.browser.remove_stray_array_item(aid)
        self._stray_arrays.clear()
        self._stray_array_counter = 0
        self._props_showing_coil_id = None
        self._active_coil_id = None
        self._analyzed_coil_id = None
        self._coords = None
        self._coil_counter = 0
        self._main.ribbon.set_run_enabled(False)
        self._main.ribbon.set_inspect_enabled(False)
        self._main.ribbon.set_construct_enabled(False)
        self.props.hide()
        if self.workspace._plotter:
            self.workspace._plotter.render()

    def _save_session(self, parent=None) -> bool:
        """Export coil arrangement to a .calcsx file for later reload.

        Returns True if the save completed, False if the user cancelled the
        file dialog. MainWindow.close-tab flow uses this to distinguish
        "user saved, ok to close" from "user cancelled — don't close".
        """
        import json
        parent = parent or self
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Session", "session.calcsx",
            "CalcSX Session (*.calcsx)",
        )
        if not path:
            return False

        def _jsonable(v):
            """Convert numpy types to JSON-native Python types."""
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            return v

        coils = []
        for coil_id in self._coil_coords:
            entry = self.workspace._coil_entries.get(coil_id, {})
            raw_params = self._coil_params_map.get(coil_id, {})
            save_params = {k: _jsonable(v) for k, v in raw_params.items()}
            coords = self._coil_coords[coil_id]

            # Pickle the full CoilAnalysis engine (with the B_ext closure
            # nulled, since it references MultiCoilEnvironment and isn't
            # meaningfully picklable). B_ext is re-attached on load.
            engine_b64 = None
            engine = self._coil_engines.get(coil_id)
            if engine is not None:
                saved_B_ext = getattr(engine, '_B_ext', None)
                try:
                    engine._B_ext = None
                    engine_b64 = base64.b64encode(
                        pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL)
                    ).decode('ascii')
                except Exception as exc:
                    sys.stderr.write(
                        f"[save_session] engine pickle failed for {coil_id}: "
                        f"{type(exc).name}: {exc}\n"
                    )
                finally:
                    engine._B_ext = saved_B_ext

            # Serialize cached inspection results (field lines, cross section,
            # per-tube-vertex force scalars — saving the latter avoids the
            # expensive per-vertex Biot-Savart recompute on session load.)
            ic = self._coil_inspect_cache.get(coil_id)
            inspect = None
            if ic is not None:
                inspect = {}
                if 'field_lines' in ic and 'field_mags' in ic:
                    inspect['field_lines'] = [np.asarray(l).tolist() for l in ic['field_lines']]
                    inspect['field_mags']  = [np.asarray(m).tolist() for m in ic['field_mags']]
                    if 'field_seeds' in ic:
                        inspect['field_seeds'] = int(ic['field_seeds'])
                if 'cross_section' in ic:
                    cs = ic['cross_section']
                    inspect['cross_section'] = {k: _jsonable(v) for k, v in cs.items()}
                if 'force_scalars' in ic:
                    fs = np.asarray(ic['force_scalars'], dtype=np.float32)
                    inspect['force_scalars'] = fs.tolist()
                if not inspect:
                    inspect = None

            coils.append({
                'coil_id':    coil_id,
                'csv_path':   self._coil_paths.get(coil_id, ''),
                'name':       self._coil_names.get(coil_id, ''),
                'color':      entry.get('color', ''),
                'coords':     coords.tolist(),
                'params':     save_params,
                'xfm_params': list(entry['xfm_params']) if entry.get('xfm_params') else None,
                'pinned':     coil_id in self._pinned_coils,
                'engine_b64': engine_b64,
                'inspect':    inspect,
            })

        # ── Serialize bobbin meshes ──
        bobbins = []
        try:
            import pyvista as pv
        except ImportError:
            pv = None
        for (bid, lname), layer in self.workspace._layers.items():
            if lname != 'Bobbin':
                continue
            for actor in layer.actors:
                try:
                    mesh = pv.wrap(actor.GetMapper().GetInput())
                    verts = np.asarray(mesh.points, dtype=np.float64).tolist()
                    # Extract triangle indices from VTK face array
                    raw = np.asarray(mesh.faces)
                    tri = raw.reshape(-1, 4)[:, 1:].tolist()
                    bobbins.append({
                        'bobbin_id': bid,
                        'vertices': verts,
                        'faces': tri,
                    })
                except Exception:
                    pass

        # ── Serialize global field lines (if computed) ──
        global_fl = None
        if self._global_fl_cache is not None:
            lines, B_mags = self._global_fl_cache
            global_fl = {
                'lines':  [np.asarray(l).tolist() for l in lines],
                'B_mags': [np.asarray(m).tolist() for m in B_mags],
                'seeds':  int(getattr(self, '_global_fl_cache_seeds', 0)),
            }

        # ── Serialize Hall probes ──
        probes = []
        for pid, entry in self.workspace._probe_entries.items():
            pos = self.workspace.get_probe_position(pid)
            st = self._probe_state.get(pid, {})
            uvw = st.get('uvw') or (0.0, 0.0, 0.0)
            probes.append({
                'probe_id': pid,
                'name':     st.get('name', pid),
                'color':    entry.get('color'),
                'position': list(map(float, (pos if pos is not None else entry['position']))),
                'mode':     st.get('mode', 'xyz'),
                'coil_ref': st.get('coil_ref'),
                'uvw':      [float(uvw[0]), float(uvw[1]), float(uvw[2])],
            })

        # Serialise circuit groups (kind, coil_ids, signs, color, name, current)
        circuits = []
        for gid, g in self._circuit_groups.items():
            circuits.append({
                'group_id': gid,
                'kind':     g.get('kind', 'series'),
                'coil_ids': list(g.get('coil_ids', [])),
                'signs':    {cid: int(s) for cid, s in g.get('signs', {}).items()},
                'color':    g.get('color'),
                'name':     g.get('name', gid),
                'current':  float(g.get('current', 0.0)),
            })

        # Stray-field probe arrays — geometry only; readouts recompute on load.
        stray_arrays = [
            {
                'array_id': aid,
                'name':     entry['name'],
                'positions': np.asarray(entry['positions']).tolist(),
            }
            for aid, entry in self._stray_arrays.items()
        ]

        with open(path, 'w') as f:
            json.dump({
                'version': 3,
                'coils': coils,
                'bobbins': bobbins,
                'global_field_lines': global_fl,
                'hall_probes': probes,
                'circuits': circuits,
                'system_energy_meter': bool(self._has_system_energy),
                'stray_arrays': stray_arrays,
            }, f, indent=2)

        n_xsec = sum(
            1 for c in self._coil_inspect_cache.values()
            if 'cross_section' in c
        )
        parts = [f"{len(coils)} coil(s)"]
        if bobbins:
            parts.append(f"{len(bobbins)} bobbin(s)")
        if probes:
            parts.append(f"{len(probes)} probe(s)")
        if n_xsec:
            parts.append(f"{n_xsec} cross section(s)")
        QMessageBox.information(
            parent, "Session Saved",
            f"Saved {', '.join(parts)} to:\n{path}",
        )
        return True

    def _load_session(self, parent=None) -> None:
        """Restore a coil arrangement from a previously saved .calcsx session."""
        import json
        parent = parent or self
        path, _ = QFileDialog.getOpenFileName(
            parent, "Load Session", "",
            "CalcSX Session (*.calcsx);;JSON files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(parent, "Load Error", str(exc))
            return

        # Defer the heavy work so the native file dialog fully closes
        # before we tear down the workspace and rebuild.
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._apply_loaded_session(data))

    def _apply_loaded_session(self, data: dict) -> None:
        # Clear the entire workspace before loading
        self._clear_all()

        # ── Loading progress dialog ──
        coil_entries = list(data.get('coils', []))
        bobbin_entries = list(data.get('bobbins') or [])
        probe_entries  = list(data.get('hall_probes') or [])
        circuit_entries = list(data.get('circuits') or [])
        n_coils   = len(coil_entries)
        n_bobbins_planned = len(bobbin_entries)
        n_probes_planned  = len(probe_entries)
        self._load_reporter = ProgressReporter(self, title="Loading Session…")
        self._load_reporter.start()
        self._load_reporter.set_stage("Loading coils…")
        self._load_reporter.report(0)
        QApplication.processEvents()

        file_ver = data.get('version', 1)
        failed = []
        loaded = 0
        for entry in coil_entries:
            # ── Resolve coordinates ──
            # v2+: coords embedded; v1 fallback: read from CSV
            if 'coords' in entry:
                coords = np.array(entry['coords'], dtype=np.float64)
            else:
                csv_path = entry.get('csv_path', '')
                if not os.path.isfile(csv_path):
                    failed.append(csv_path)
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    coords = (
                        df[['x', 'y', 'z']].values
                        if {'x', 'y', 'z'}.issubset(df.columns)
                        else df.iloc[:, :3].values
                    )
                except Exception as exc:
                    failed.append(f"{csv_path} ({exc})")
                    continue

            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack((coords, coords[0]))

            # ── Create coil ──
            self._coil_counter += 1
            coil_id = entry.get('coil_id', f"coil_{self._coil_counter}")
            color   = entry.get('color') or _get_coil_colors()[
                (self._coil_counter - 1) % len(_get_coil_colors())
            ]
            name    = entry.get('name', coil_id)
            params  = entry.get('params', self.props.get_params())

            # Restore tape_normals from list → ndarray
            tn = params.get('tape_normals')
            if tn is not None and not isinstance(tn, np.ndarray):
                params['tape_normals'] = np.array(tn, dtype=np.float64)

            self._coords               = coords
            self._active_coil_id       = coil_id
            self._coil_coords[coil_id] = coords
            self._coil_paths[coil_id]  = entry.get('csv_path', '')
            self._coil_names[coil_id]  = name
            self._coil_params_map[coil_id] = params

            total_t = params['thickness'] * 1e-6 * params['winds']
            tape_w  = params['width'] * 1e-3
            growth  = params.get('stack_growth', 'symmetric')
            self.workspace.add_coil(coords, coil_id, color=color,
                                     total_thickness=total_t, tape_width=tape_w,
                                     tape_normals=params.get('tape_normals'),
                                     winding_growth=growth)
            self.browser.add_coil_item(coil_id, name, color)

            self._multi_env.register_coil(
                coil_id, coords,
                winds=params['winds'], current=params['current'],
                thickness=params['thickness'], width=params['width'],
                tape_normals=params.get('tape_normals'),
                winding_growth=growth,
            )

            # Restore transform if present
            xfm = entry.get('xfm_params')
            if xfm is not None:
                xfm = tuple(xfm)
                self.workspace._coil_entries[coil_id]['xfm_params'] = xfm
                try:
                    try:
                        from vtkmodules.vtkCommonTransforms import vtkTransform
                    except ImportError:
                        import vtk as _v
                        vtkTransform = _v.vtkTransform
                    tx, ty, tz, rx, ry, rz, cx, cy, cz = xfm
                    t = vtkTransform()
                    t.PostMultiply()
                    t.Translate(-cx, -cy, -cz)
                    t.RotateX(rx); t.RotateY(ry); t.RotateZ(rz)
                    t.Translate(cx + tx, cy + ty, cz + tz)
                    for actor in self.workspace._coil_entries[coil_id]['actors']:
                        actor.SetUserTransform(t)
                except Exception:
                    pass

                wc = self.workspace.get_transformed_coords(coil_id)
                if wc is not None:
                    self._multi_env.update_coil_coords(coil_id, wc)

            # Restore pinned state (new in v2.3.2; older sessions default to False).
            if entry.get('pinned', False):
                self._pinned_coils.add(coil_id)

            # ── Restore the full engine from pickle (if saved) ──
            engine_b64 = entry.get('engine_b64')
            if engine_b64:
                try:
                    engine = pickle.loads(base64.b64decode(engine_b64))
                    engine._B_ext = self._multi_env.make_external_field_func(coil_id)
                    # Ensure the rescale-metadata attributes exist on engines
                    # loaded from .calcsx files saved before these were added.
                    if not hasattr(engine, '_analysis_current'):
                        engine._analysis_current = float(
                            getattr(engine, 'current', 0.0) or 0.0
                        )
                    if not hasattr(engine, '_analyzed_with_ext'):
                        engine._analyzed_with_ext = (engine._B_ext is not None)
                    self._coil_engines[coil_id] = engine
                    # Rebuild the standard analysis layers. For Forces,
                    # prefer pre-computed per-vertex scalars stored in the
                    # .calcsx (saved at analysis time) to skip the O(n_verts
                    # × n_sources × n_fil) per-vertex Biot-Savart sweep —
                    # the dominant cost of session load on large coils.
                    inspect_data = entry.get('inspect') or {}
                    fs_saved = inspect_data.get('force_scalars')
                    if fs_saved is not None:
                        fs_arr = np.asarray(fs_saved, dtype=np.float32)
                        self.workspace.add_force_layer_from_scalars(
                            coil_id, fs_arr,
                        )
                        self._coil_inspect_cache.setdefault(
                            coil_id, {})['force_scalars'] = fs_arr
                    else:
                        self.workspace.add_force_layer(engine, coil_id)
                    self.workspace.add_stress_layer(engine, coil_id)
                    self.workspace.add_axis_layer(engine, coil_id)
                    self.browser.add_layer_to_coil(coil_id, 'Forces')
                    self.browser.add_layer_to_coil(coil_id, 'Stress')
                    self.browser.add_layer_to_coil(coil_id, 'B Axis')
                    # Re-apply the coil's transform to the new layers
                    self.workspace.reapply_coil_transform(coil_id)
                except Exception as exc:
                    sys.stderr.write(
                        f"[load_session] engine unpickle failed for {coil_id}: "
                        f"{type(exc).name}: {exc}\n"
                    )
                    import traceback
                    traceback.print_exc()

            # ── Restore saved inspection layers (field lines, cross section) ──
            inspect = entry.get('inspect')
            if inspect:
                ic = {}
                fl = inspect.get('field_lines'); fm = inspect.get('field_mags')
                if fl and fm:
                    lines  = [np.asarray(l, dtype=float) for l in fl]
                    B_mags = [np.asarray(m, dtype=float) for m in fm]
                    self.workspace.add_field_lines_layer(lines, B_mags, coil_id)
                    self.browser.add_layer_to_coil(coil_id, 'Field Lines', deletable=True)
                    ic['field_lines'] = lines
                    ic['field_mags']  = B_mags
                    if inspect.get('field_seeds') is not None:
                        ic['field_seeds'] = int(inspect['field_seeds'])
                cs = inspect.get('cross_section')
                if cs:
                    X  = np.asarray(cs['X'], dtype=float)
                    Y  = np.asarray(cs['Y'], dtype=float)
                    Bp = np.asarray(cs['B_plane'], dtype=float)
                    e1 = np.asarray(cs['e1'], dtype=float)
                    e2 = np.asarray(cs['e2'], dtype=float)
                    ct = np.asarray(cs['center'], dtype=float)
                    R  = float(cs['R'])
                    self.workspace.add_cross_section_layer(
                        X, Y, Bp, e1, e2, ct, R, coil_id,
                    )
                    self.browser.add_layer_to_coil(coil_id, 'Cross Section', deletable=True)
                    ic['cross_section'] = {
                        'X': X, 'Y': Y, 'B_plane': Bp,
                        'e1': e1, 'e2': e2, 'center': ct, 'R': R,
                        'offset': float(cs.get('offset', 0.0)),
                    }
                if ic:
                    self._coil_inspect_cache[coil_id] = ic

            loaded += 1
            # Coils drive the bulk of load time — give them 0..80% of the bar
            if n_coils:
                self._load_reporter.report(int(80 * loaded / n_coils))
            QApplication.processEvents()

        # ── Restore bobbin meshes ──
        if n_bobbins_planned:
            self._load_reporter.set_stage("Loading bobbins…")
            self._load_reporter.report(82)
            QApplication.processEvents()
        n_bobbins = 0
        for bentry in bobbin_entries:
            try:
                import pyvista as pv
                bid = bentry['bobbin_id']
                verts = np.array(bentry['vertices'], dtype=np.float32)
                tri = np.array(bentry['faces'], dtype=np.int64)
                pv_faces = np.column_stack([
                    np.full(len(tri), 3, dtype=np.int64), tri
                ]).ravel()
                mesh = pv.PolyData(verts, pv_faces)
                self.workspace.add_bobbin_mesh(bid, mesh)
                self.browser.add_coil_item(bid, "Bobbin", '#888888')
                n_bobbins += 1
            except Exception:
                pass

        self._propagate_staleness()
        # Any coil whose full engine was restored is considered fresh
        for cid in self._coil_engines:
            self._multi_env.mark_fresh(cid)
            for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                self.browser.mark_layer_stale(cid, nm, False)

        # ── Restore global field lines if saved ──
        gfl = data.get('global_field_lines')
        any_field_lines_restored = any(
            'field_lines' in c for c in self._coil_inspect_cache.values()
        )
        if gfl and gfl.get('lines') and gfl.get('B_mags'):
            self._load_reporter.set_stage("Restoring global field lines…")
            self._load_reporter.report(90)
            QApplication.processEvents()
            lines  = [np.asarray(l, dtype=float) for l in gfl['lines']]
            B_mags = [np.asarray(m, dtype=float) for m in gfl['B_mags']]
            self._global_fl_cache = (lines, B_mags)
            self._global_fl_cache_seeds = int(gfl.get('seeds', 0))
            self._global_fl_dirty = False
            self.workspace.add_field_lines_layer(lines, B_mags, 'global')
            any_field_lines_restored = True

        # ── Restore Hall probes ──
        if n_probes_planned:
            self._load_reporter.set_stage("Restoring Hall probes…")
            self._load_reporter.report(95)
            QApplication.processEvents()
        probes_data = probe_entries
        n_probes = 0
        max_probe_counter = self._probe_counter
        for pentry in probes_data:
            try:
                pid     = pentry['probe_id']
                name    = pentry.get('name', pid)
                color   = pentry.get('color')
                position = np.asarray(pentry['position'], dtype=np.float64)
                mode    = pentry.get('mode', 'xyz')
                coil_ref = pentry.get('coil_ref')
                uvw_raw  = pentry.get('uvw') or (0.0, 0.0, 0.0)
                uvw      = (float(uvw_raw[0]), float(uvw_raw[1]), float(uvw_raw[2]))
                self.workspace.add_hall_probe(pid, position, color=color)
                self.browser.add_probe_item(pid, name)
                self._probe_state[pid] = {
                    'mode':     mode,
                    'coil_ref': coil_ref,
                    'uvw':      uvw,
                    'name':     name,
                }
                self.browser.update_probe_parent_label(pid, coil_ref, mode)
                # Bump counter past the loaded probe's numeric id so newly
                # created probes don't collide.
                try:
                    n = int(pid.rsplit('_', 1)[-1])
                    max_probe_counter = max(max_probe_counter, n)
                except ValueError:
                    pass
                n_probes += 1
            except Exception:
                pass
        self._probe_counter = max_probe_counter
        # Start the probe readout timer if any probes were restored
        if n_probes and self._probe_timer is None:
            from PyQt5.QtCore import QTimer
            self._probe_timer = QTimer(self)
            self._probe_timer.setInterval(200)
            self._probe_timer.timeout.connect(self._update_all_probe_readouts)
            self._probe_timer.start()

        # Restore circuit groups
        self._circuit_groups = {}
        self._coil_group_map = {}
        max_circuit_counter = 0
        for centry in circuit_entries:
            try:
                gid = centry['group_id']
                coil_ids = [cid for cid in centry.get('coil_ids', [])
                            if cid in self._coil_coords]
                if len(coil_ids) < 2:
                    continue
                signs_in = centry.get('signs', {}) or {}
                signs = {cid: int(signs_in.get(cid, 1)) for cid in coil_ids}
                # Restore circuit current from file, fallback to first coil's
                first_current = float(
                    centry.get('current',
                        self._coil_params_map.get(coil_ids[0], {}).get('current', 0.0))
                )
                color = centry.get('color', self._next_group_color())
                name = centry.get('name', gid)
                kind = centry.get('kind', 'series')
                self._circuit_groups[gid] = {
                    'kind':     kind,
                    'coil_ids': coil_ids,
                    'signs':    signs,
                    'color':    color,
                    'name':     name,
                    'current':  first_current,
                }
                for cid in coil_ids:
                    self._coil_group_map[cid] = gid
                    # Unify every member's per-coil current to the circuit's
                    self._coil_params_map.setdefault(cid, {})['current'] = first_current
                # Rebuild the browser folder header + badge member coils
                self.browser.add_circuit_header(gid, name, kind, color,
                                                  insert_above=list(coil_ids))
                for cid in coil_ids:
                    self.browser.move_coil_under_circuit(cid, gid)
                try:
                    max_circuit_counter = max(
                        max_circuit_counter, int(gid.rsplit('_', 1)[-1])
                    )
                except ValueError:
                    pass
            except Exception:
                pass
        self._circuit_counter = max(self._circuit_counter, max_circuit_counter)

        # Unify colour scales across restored layers
        self._load_reporter.set_stage("Finalizing…")
        self._load_reporter.report(99)
        QApplication.processEvents()
        if self._coil_engines:
            self.workspace.rescale_all_force_layers()
        if any_field_lines_restored:
            self.workspace.rescale_all_field_line_layers()
        self._main.ribbon.set_inspect_enabled(bool(self._coil_coords))
        self._main.ribbon.set_construct_enabled(bool(self._coil_coords))
        if self._active_coil_id:
            self._load_coil_params(self._active_coil_id)
            # _refresh_summary_for now defers the L-matrix compute to the
            # background worker when the cache is cold, so it's safe to
            # call synchronously even for grouped coils.
            self._refresh_summary_for(self._active_coil_id)
        self.props.show()
        if self.workspace._plotter:
            self.workspace._plotter.reset_camera()
            self.workspace._plotter.render()
        self._load_reporter.finish()
        self._load_reporter = None
        # Restore the System Energy meter if the saved session had one.
        if bool(data.get('system_energy_meter', False)):
            self._on_add_system_energy()
        # Restore stray-field probe arrays.
        for arr in (data.get('stray_arrays') or []):
            try:
                aid  = str(arr.get('array_id') or '')
                name = str(arr.get('name') or 'Stray')
                pos  = np.asarray(arr.get('positions') or [], dtype=np.float64)
                if not aid or pos.ndim != 2 or pos.shape[1] != 3:
                    continue
                # Bump counter past the loaded id so subsequent adds don't collide.
                try:
                    n = int(aid.rsplit('_', 1)[-1])
                    self._stray_array_counter = max(self._stray_array_counter, n)
                except ValueError:
                    pass
                self._stray_arrays[aid] = {'name': name, 'positions': pos}
                self.workspace.add_stray_array(aid, pos)
                self.browser.add_stray_array_item(aid, name)
                self._update_stray_array_readout(aid)
            except Exception:
                pass
        # Kick off the L-matrix precompute in the background so the first
        # circuit-header click after load is instant.
        self._schedule_l_matrix_precompute()

        n_xsec_loaded = sum(
            1 for c in self._coil_inspect_cache.values()
            if 'cross_section' in c
        )
        parts = [f"{loaded} coil(s)"]
        if n_bobbins:
            parts.append(f"{n_bobbins} bobbin(s)")
        if n_probes:
            parts.append(f"{n_probes} probe(s)")
        if n_xsec_loaded:
            parts.append(f"{n_xsec_loaded} cross section(s)")
        msg = "Loaded " + ", ".join(parts) + "."
        if failed:
            msg += f"\n\nFailed to load {len(failed)} coil(s):\n" + "\n".join(failed)
        QMessageBox.information(self, "Session Loaded", msg)

    # ── System Energy instrument ─────────────────────────────────────────────

    def _on_add_system_energy(self) -> None:
        """Toggle the singleton System Energy meter on. No-op if already added."""
        if self._has_system_energy:
            return
        if not self._coil_coords:
            QMessageBox.information(
                self, "System Energy",
                "Load at least one coil before adding the System Energy meter.",
            )
            return
        self.browser.add_system_energy_item()
        self._has_system_energy = True
        self._update_system_energy_readout()

    def _on_system_energy_delete(self) -> None:
        """Remove the System Energy meter from the browser."""
        if not self._has_system_energy:
            return
        self.browser.remove_system_energy_item()
        self._has_system_energy = False

    def _update_system_energy_readout(self) -> None:
        """Recompute ½·Iᵀ·L·I and refresh the meter's displayed scalar.

        Cheap when geometry hasn't changed — MultiCoilEnvironment caches the
        L-matrix; pure-current updates just re-evaluate the quadratic form.
        Silent no-op when the meter isn't installed or there are no coils.
        """
        if not self._has_system_energy:
            return
        if not self._coil_coords:
            self.browser.update_system_energy_readout("—")
            return
        try:
            result = self._multi_env.compute_mutual_inductance_matrix()
            total_J = float(result.get('total_energy', 0.0))
        except Exception as exc:
            self.browser.update_system_energy_readout(f"error ({type(exc).__name__})")
            return
        self.browser.update_system_energy_readout(self._fmt_energy(total_J))

    @staticmethod
    def _fmt_energy(joules: float) -> str:
        """Auto-scale a stored-energy scalar: J / kJ / MJ / GJ."""
        a = abs(joules)
        if a >= 1e9: return f"{joules/1e9:.3f} GJ"
        if a >= 1e6: return f"{joules/1e6:.3f} MJ"
        if a >= 1e3: return f"{joules/1e3:.3f} kJ"
        return f"{joules:.3f} J"

    # ── Stray-field probe array instrument ──────────────────────────────────

    def _on_add_stray_array(self) -> None:
        """Open the StrayArrayDialog and register a new array on accept."""
        if not self._coil_coords:
            QMessageBox.information(
                self, "Stray Array",
                "Load at least one coil before adding a stray-field array.",
            )
            return
        self._stray_array_counter += 1
        default_name = f"Stray {self._stray_array_counter}"
        dlg = StrayArrayDialog(self, default_name)
        if dlg.exec_() != QDialog.Accepted:
            # User cancelled — rewind the counter so the next dialog reuses N.
            self._stray_array_counter -= 1
            return
        name, positions = dlg.result()
        array_id = f"stray_{self._stray_array_counter}"
        self._stray_arrays[array_id] = {
            'name': name,
            'positions': np.asarray(positions, dtype=np.float64),
        }
        self.workspace.add_stray_array(array_id, positions)
        self.browser.add_stray_array_item(array_id, name)
        self._update_stray_array_readout(array_id)

    def _on_stray_array_delete(self, array_id: str) -> None:
        """Remove a stray-field array from scene + browser."""
        self._stray_arrays.pop(array_id, None)
        self.workspace.remove_stray_array(array_id)
        self.browser.remove_stray_array_item(array_id)

    def _update_stray_array_readout(self, array_id: str) -> None:
        """Recompute mean(|B|), max(|B|), B_rms for one array."""
        entry = self._stray_arrays.get(array_id)
        if entry is None:
            return
        B_func = self._multi_env.make_total_field_func()
        if B_func is None:
            self.browser.update_stray_array_readout(array_id, "—")
            return
        try:
            B = B_func(entry['positions'])
            B = np.atleast_2d(np.asarray(B, dtype=float))
            mags = np.linalg.norm(B[:, :3], axis=1)
            b_rms = float(np.sqrt(np.mean(mags ** 2)))
            b_max = float(np.max(mags))
        except Exception as exc:
            self.browser.update_stray_array_readout(
                array_id, f"error ({type(exc).__name__})",
            )
            return
        # Format both in shared auto-units for readability.
        unit = _b_field_unit(b_rms)
        text = (
            f"B_rms = {_fmt_b(b_rms, unit)}  •  "
            f"max = {_fmt_b(b_max, unit)}"
        )
        self.browser.update_stray_array_readout(array_id, text)

    def _update_all_stray_arrays(self) -> None:
        for aid in self._stray_arrays:
            self._update_stray_array_readout(aid)

    def _on_add_hall_probe(self) -> None:
        """Add a new Hall probe at the active coil centroid (or origin)."""
        from PyQt5.QtCore import QTimer

        self._probe_counter += 1
        probe_id = f"probe_{self._probe_counter}"

        position = None
        if self._active_coil_id:
            wc = self.workspace.get_transformed_coords(self._active_coil_id)
            c = wc if wc is not None else self._coil_coords.get(self._active_coil_id)
            if c is not None:
                position = np.mean(c, axis=0)

        self.workspace.add_hall_probe(probe_id, position)
        self.browser.add_probe_item(probe_id, f"Probe {self._probe_counter}")

        # Initialize probe metadata: default XYZ mode, associated with the
        # currently active coil (so PCA mode is available).
        self._probe_state[probe_id] = {
            'mode':     'xyz',
            'coil_ref': self._active_coil_id,
            'uvw':      (0.0, 0.0, 0.0),
            'name':     f"Probe {self._probe_counter}",
        }
        self.browser.update_probe_parent_label(
            probe_id, self._active_coil_id, 'xyz'
        )

        # Start the shared readout timer if not already running
        if self._probe_timer is None:
            self._probe_timer = QTimer(self)
            self._probe_timer.setInterval(200)
            self._probe_timer.timeout.connect(self._update_all_probe_readouts)
            self._probe_timer.start()

        # Immediately update this probe
        self._update_single_probe_readout(probe_id)

    def _update_all_probe_readouts(self) -> None:
        """Update readouts for all active probes."""
        for pid in list(self.workspace._probe_entries.keys()):
            self._update_single_probe_readout(pid)

    def _update_single_probe_readout(self, probe_id: str) -> None:
        pos = self.workspace.get_probe_position(probe_id)
        if pos is None:
            return
        # Reflect gizmo-driven moves in the Properties spinboxes — but not
        # while the user is actively editing them.
        if probe_id == self._active_probe_id() and self.props._probe_w.isVisible():
            if not any(s.hasFocus() for s in (
                self.props.dspin_probe_x,
                self.props.dspin_probe_y,
                self.props.dspin_probe_z,
                self.props.dspin_probe_u,
                self.props.dspin_probe_v,
                self.props.dspin_probe_w,
            )):
                self.props.update_probe_position_display(pos)
                st = self._probe_state.get(probe_id, {})
                if st.get('mode') == 'pca' and st.get('coil_ref'):
                    uvw = self._xyz_to_pca(st['coil_ref'], pos)
                    st['uvw'] = uvw
                    self.props.update_probe_pca_display(*uvw)
        B_func = self._multi_env.make_total_field_func()
        if B_func is None:
            # No field available yet — show position only
            self.browser.update_probe_readout(
                probe_id, 0.0, 0.0, 0.0, 0.0,
                float(pos[0]), float(pos[1]), float(pos[2]),
            )
            return
        try:
            B = np.atleast_1d(B_func(pos.reshape(1, 3))).flatten()
            if B.shape[0] >= 3:
                self.browser.update_probe_readout(
                    probe_id,
                    float(B[0]), float(B[1]), float(B[2]),
                    float(np.linalg.norm(B[:3])),
                    float(pos[0]), float(pos[1]), float(pos[2]),
                )
        except Exception:
            pass

    def _on_probe_selected(self, probe_id: str) -> None:
        """User clicked a probe in the browser — switch gizmo target."""
        self.workspace.set_gizmo_target('probe')
        self.workspace.set_active_probe(probe_id)
        pos = self.workspace.get_probe_position(probe_id)
        if pos is not None and self.workspace._gizmo is not None:
            scale = 0.1
            if self._coil_coords:
                some = next(iter(self._coil_coords.values()))
                arr = np.asarray(some)
                scale = float((arr.max(axis=0) - arr.min(axis=0)).max()) * 0.15
            self.workspace._gizmo.load(pos, scale)
        if self._main.ribbon._btn_translate.isChecked() or self._main.ribbon._btn_rotate.isChecked():
            mode = 'T' if self._main.ribbon._btn_translate.isChecked() else 'R'
            self.workspace.show_gizmo(mode)
        self._update_single_probe_readout(probe_id)
        # Show the probe-controls view in the Properties panel
        st = self._probe_state.setdefault(probe_id, {
            'mode': 'xyz', 'coil_ref': self._active_coil_id,
            'uvw': (0.0, 0.0, 0.0), 'name': probe_id,
        })
        coil_ref = st.get('coil_ref') or self._active_coil_id
        st['coil_ref'] = coil_ref
        uvw = st.get('uvw') or (0.0, 0.0, 0.0)
        # If we have a coil reference, refresh UVW to match the current XYZ
        # so the display is coherent even when the probe was moved via gizmo.
        if coil_ref and pos is not None:
            uvw = self._xyz_to_pca(coil_ref, pos)
            st['uvw'] = uvw
        self.props.show_probe_controls(
            position=pos, mode=st['mode'], coil_ref=coil_ref, uvw=uvw,
        )

    def _on_probe_delete(self, probe_id: str) -> None:
        """Delete a specific Hall probe."""
        self.workspace.remove_hall_probe(probe_id)
        self.browser.remove_probe_item(probe_id)
        self._probe_state.pop(probe_id, None)
        # Stop timer if no probes remain
        if not self.workspace._probe_entries:
            if self._probe_timer is not None:
                self._probe_timer.stop()
                self._probe_timer = None
        # If no probes remain, return to coil view
        if not self.workspace._probe_entries:
            self.props.show_coil_controls()

    # ── Probe position handlers ──────────────────────────────────────────────

    def _coil_pca_frame(self, coil_id: str):
        """Return (mean, axes) where axes is a (3, 3) matrix with the coil's
        three PCA component vectors as rows (in order of descending variance),
        computed from the current world-space coords. None on failure."""
        from sklearn.decomposition import PCA
        coords = self.workspace.get_transformed_coords(coil_id)
        if coords is None:
            coords = self._coil_coords.get(coil_id)
        if coords is None or len(coords) < 3:
            return None
        coords = np.asarray(coords, dtype=float)
        try:
            pca = PCA(n_components=3).fit(coords)
            axes = np.asarray(pca.components_, dtype=float)
            mean = np.asarray(pca.mean_,       dtype=float)
            return mean, axes
        except Exception:
            return None

    def _xyz_to_pca(self, coil_id: str, position) -> tuple:
        """Project a world-space position onto the coil's PCA frame; returns
        (u, v, w) offsets along axis 1, 2, 3 from the coil centroid."""
        frame = self._coil_pca_frame(coil_id)
        if frame is None:
            return (0.0, 0.0, 0.0)
        mean, axes = frame
        d = np.asarray(position, dtype=float) - mean
        return (float(d @ axes[0]), float(d @ axes[1]), float(d @ axes[2]))

    def _pca_to_xyz(self, coil_id: str, uvw) -> np.ndarray | None:
        """Convert PCA offsets (u, v, w) to a world-space position."""
        frame = self._coil_pca_frame(coil_id)
        if frame is None:
            return None
        mean, axes = frame
        u, v, w = float(uvw[0]), float(uvw[1]), float(uvw[2])
        return mean + u * axes[0] + v * axes[1] + w * axes[2]

    def _active_probe_id(self) -> str | None:
        return self.workspace._active_probe_id

    def _on_probe_xyz_edit(self, x: float, y: float, z: float) -> None:
        pid = self._active_probe_id()
        if pid is None:
            return
        self.workspace.set_probe_position(pid, np.array([x, y, z], dtype=float))
        # In PCA mode, also refresh the U/V/W spinboxes to reflect the new XYZ.
        st = self._probe_state.get(pid, {})
        if st.get('mode') == 'pca' and st.get('coil_ref'):
            uvw = self._xyz_to_pca(st['coil_ref'], [x, y, z])
            st['uvw'] = uvw
            self.props.update_probe_pca_display(*uvw)
        self._update_single_probe_readout(pid)

    def _on_probe_pca_edit(self, u: float, v: float, w: float) -> None:
        pid = self._active_probe_id()
        if pid is None:
            return
        st = self._probe_state.get(pid, {})
        coil_ref = st.get('coil_ref')
        if not coil_ref:
            return
        pt = self._pca_to_xyz(coil_ref, (u, v, w))
        if pt is None:
            return
        st['uvw'] = (float(u), float(v), float(w))
        self.workspace.set_probe_position(pid, pt)
        self.props.update_probe_position_display(pt)
        self._update_single_probe_readout(pid)

    def _on_probe_mode_change(self, mode: str) -> None:
        pid = self._active_probe_id()
        if pid is None:
            return
        st = self._probe_state.setdefault(pid, {})
        st['mode'] = mode
        # When switching to PCA, refresh the UVW spinboxes from current XYZ.
        if mode == 'pca' and st.get('coil_ref'):
            pos = self.workspace.get_probe_position(pid)
            if pos is not None:
                uvw = self._xyz_to_pca(st['coil_ref'], pos)
                st['uvw'] = uvw
                self.props.update_probe_pca_display(*uvw)
        self.browser.update_probe_parent_label(pid, st.get('coil_ref'), mode)

    def _propagate_staleness(self) -> None:
        """Reflect MultiCoilEnvironment staleness into browser stale markers."""
        stale = self._multi_env.get_stale_coils()
        for stale_cid in stale:
            if stale_cid in self._coil_engines:
                for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                    self.browser.mark_layer_stale(stale_cid, nm, True)
        self._main.ribbon._btn_reanalyze.set_action_enabled(bool(stale))
        self._global_fl_dirty = True

        # Auto-clear global field lines — they're invalid now
        if self._main.ribbon._btn_global_field.isChecked():
            self.workspace.clear_field_lines_layer('global')
            self._main.ribbon._btn_global_field.blockSignals(True)
            self._main.ribbon._btn_global_field.setChecked(False)
            self._main.ribbon._btn_global_field.blockSignals(False)
            # Restore per-coil field line controls
            self._main.ribbon._btn_field_lines.set_action_enabled(True)
            for cid in list(self._coil_coords.keys()):
                was_on = getattr(self, '_pre_global_eye_state', {}).get(cid, True)
                self.browser.set_layer_eye_unlocked(cid, 'Field Lines', restore_checked=was_on)

        # System Energy meter (if installed) follows every scene change —
        # _propagate_staleness is the common rendezvous for register / unregister
        # / param-update / coord-update, so one hook here covers them all.
        self._update_system_energy_readout()
        # Stray-field probe arrays update on the same cadence.
        self._update_all_stray_arrays()

