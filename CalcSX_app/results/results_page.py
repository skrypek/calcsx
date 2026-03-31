# results_page.py
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox
)
from gui.gui_utils import make_canvas
from plots.plot_defs  import (
    plot_3d_filament,
    plot_stress_vs_arc,
    plot_bfield_cross_section
)
from tabs.tab_defs import make_force_vs_arc_tab, make_lorentz_toggle_tab, make_bfield_calcs_tab

class ResultsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._engine = None

        self.btn_back = QPushButton("Return")
        self.btn_save = QPushButton("Save Results…")
        self.btn_save.setEnabled(False)
        self.tabs = QTabWidget()

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_save)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(btn_row)
        self.layout().addWidget(self.tabs)

        self.btn_save.clicked.connect(self._save_results)

    def setup(self, engine, show_b_along_axis=False):
        self._engine = engine
        self.btn_save.setEnabled(True)

        # 1) 3D filament
        self.tabs.clear()
        self.tabs.addTab(
            make_canvas(plot_3d_filament, engine, projection='3d'),
            "3D Filament"
        )
        # 2) Lorentz vectors
        self.tabs.addTab(
            make_lorentz_toggle_tab(engine),
            "Lorentz Vectors"
        )
        # 3) Force vs Arc
        self.tabs.addTab(
            make_force_vs_arc_tab(engine),
            "Lorentz vs Arc"
        )
        # 4) Stress vs Arc
        self.tabs.addTab(
            make_canvas(plot_stress_vs_arc, engine),
            "Stress vs Arc"
        )
        # 5) B‑field Calcs (optional)
        self.tabs.addTab(
            make_bfield_calcs_tab(engine),
            "On-Axis B‑field"
        )
        if show_b_along_axis:

            # Ensure cross-section data exists (worker already tried; fallback respects n_grid)
            try:
                if not hasattr(engine, "cross_section_data") or engine.cross_section_data is None:
                    n_grid = getattr(engine, '_n_grid', 120)
                    engine.cross_section_data = engine.sample_cross_section(n=n_grid)
            except Exception:
                engine.cross_section_data = None
        
            self.tabs.addTab(
                make_canvas(
                    lambda ax, ctx: plot_bfield_cross_section(
                        ax, ctx,
                        threshold=getattr(ctx, "cross_section_threshold", None)
                    ),
                    engine
                ),
                "Planar B‑field"
            )

    def _save_results(self):
        """Export computed arrays (arc data + on-axis B-field) to a CSV file."""
        eng = self._engine
        if eng is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "calcsx_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            # Arc-based arrays
            arc   = np.asarray(eng.arc_mid)       # (n,)
            force = np.asarray(eng.F_mags)         # (n,)
            stress = np.asarray(eng.hoop_stress)   # (n,)
            n_arc = len(arc)

            # On-axis arrays (may be None)
            has_axis = (eng.bfield_axis_z is not None and eng.bfield_axis_mag is not None)
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
                "arc_position_m":       _pad(arc,   n_rows),
                "force_density_N_per_m": _pad(force, n_rows),
                "hoop_stress_Pa":        _pad(stress, n_rows),
            }
            if has_axis:
                data["axis_z_m"] = _pad(axis_z, n_rows)
                data["B_axis_T"] = _pad(axis_B, n_rows)

            pd.DataFrame(data).to_csv(path, index=False)

            QMessageBox.information(
                self, "Saved",
                f"Results exported to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
