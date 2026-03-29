# results_page.py
from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QPushButton
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
        self.btn_back = QPushButton("Return")
        self.tabs     = QTabWidget()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.btn_back)
        self.layout().addWidget(self.tabs)

    def setup(self, engine, show_b_along_axis=False):
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

            # Ensure cross‑section data exists (worker already tried; fallback here)
            try:
                if not hasattr(engine, "cross_section_data") or engine.cross_section_data is None:
                    engine.cross_section_data = engine.sample_cross_section(n=120)
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
