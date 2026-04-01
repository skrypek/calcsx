# results/results_page.py
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox,
)
from gui.gui_utils import make_canvas, make_canvas_with_toolbar, ViewSwitcherBar, THEME
from plots.plot_defs import (
    plot_3d_filament,
    plot_stress_vs_arc,
    plot_bfield_cross_section,
)
from tabs.tab_defs import make_forces_view, make_bfield_calcs_tab
from views.slice_viewer import BFieldSliceWidget


class ResultsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._engine   = None
        self._switcher = None

        self.btn_save = QPushButton("💾  Save Results")
        self.btn_save.setEnabled(False)
        self.btn_save.setFixedHeight(28)

        # ── Top bar: [ViewSwitcherBar ... | Save button] ─────────
        self._bar = QWidget()
        self._bar.setFixedHeight(36)
        self._bar.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
        )
        self._bar_lay = QHBoxLayout(self._bar)
        self._bar_lay.setContentsMargins(0, 4, 8, 4)
        self._bar_lay.setSpacing(0)
        self._bar_lay.addStretch()
        self._bar_lay.addWidget(self.btn_save)

        # ── Central view stack ────────────────────────────────────
        self._stack = QStackedWidget()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._bar)
        lay.addWidget(self._stack, stretch=1)

        self.btn_save.clicked.connect(self._save_results)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def setup(self, engine, show_b_along_axis: bool = False):
        self._engine = engine
        self.btn_save.setEnabled(True)

        # Remove old switcher from bar
        if self._switcher is not None:
            self._bar_lay.removeWidget(self._switcher)
            self._switcher.setParent(None)
            self._switcher = None

        # Clear view stack
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.setParent(None)

        # ── Build views ──────────────────────────────────────────
        labels  = []
        widgets = []

        # 1. 3D Geometry
        labels.append("3D Geometry")
        widgets.append(
            make_canvas_with_toolbar(plot_3d_filament, engine, projection='3d')
        )

        # 2. Forces (split: 3D quiver + force vs arc)
        labels.append("Forces")
        widgets.append(make_forces_view(engine))

        # 3. Stress
        labels.append("Stress")
        widgets.append(
            make_canvas_with_toolbar(plot_stress_vs_arc, engine)
        )

        # 4. Magnetics (on-axis B-field)
        labels.append("Magnetics")
        widgets.append(make_bfield_calcs_tab(engine))

        # 5. Cross-Section (optional planar heat map)
        if show_b_along_axis:
            try:
                if (not hasattr(engine, "cross_section_data")
                        or engine.cross_section_data is None):
                    n_grid = getattr(engine, '_n_grid', 120)
                    engine.cross_section_data = engine.sample_cross_section(n=n_grid)
            except Exception:
                engine.cross_section_data = None

            labels.append("Cross-Section")
            widgets.append(
                make_canvas_with_toolbar(
                    lambda ax, ctx: plot_bfield_cross_section(
                        ax, ctx,
                        threshold=getattr(ctx, "cross_section_threshold", None),
                    ),
                    engine,
                )
            )

        # 6. Field Slicer (always available, lazy-loaded on demand)
        labels.append("Field Slicer")
        widgets.append(BFieldSliceWidget(engine))

        for w in widgets:
            self._stack.addWidget(w)

        # ── Install view switcher ────────────────────────────────
        self._switcher = ViewSwitcherBar(labels)
        self._bar_lay.insertWidget(0, self._switcher)
        self._switcher.view_changed.connect(self._stack.setCurrentIndex)

    # ─────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────

    def _save_results(self):
        eng = self._engine
        if eng is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "calcsx_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            arc    = np.asarray(eng.arc_mid)
            force  = np.asarray(eng.F_mags)
            stress = np.asarray(eng.hoop_stress)
            n_arc  = len(arc)

            has_axis = (
                eng.bfield_axis_z is not None
                and eng.bfield_axis_mag is not None
            )
            if has_axis:
                axis_z = np.asarray(eng.bfield_axis_z)
                axis_B = np.asarray(eng.bfield_axis_mag)
                n_axis = len(axis_z)
            else:
                n_axis = 0

            n_rows = max(n_arc, n_axis)

            def _pad(arr, length):
                if len(arr) >= length:
                    return arr[:length]
                return np.concatenate([arr, np.full(length - len(arr), np.nan)])

            data = {
                "arc_position_m":        _pad(arc,    n_rows),
                "force_density_N_per_m": _pad(force,  n_rows),
                "hoop_stress_Pa":        _pad(stress, n_rows),
            }
            if has_axis:
                data["axis_z_m"] = _pad(axis_z, n_rows)
                data["B_axis_T"] = _pad(axis_B, n_rows)

            pd.DataFrame(data).to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Results exported to:\n{path}")

        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
