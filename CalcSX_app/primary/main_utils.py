# main.py
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QStackedWidget,
    QCheckBox,
    QToolButton,
    QTextBrowser,
    QDialog,
    QLabel,
    QHBoxLayout
)

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from physics.physics_utils import CoilAnalysis
from gui.gui_utils import make_canvas, LogoMixin, ProgressReporter
from results.results_page import ResultsPage

from string import Template
from version import __version__ as version_module
__version__ = getattr(version_module, "__version__", "UNKNOWN")

# turn off interactive mode so canvases render offscreen
plt.ioff()

class AnalysisWorker(QObject):
    progress = pyqtSignal(int)
    stage    = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, coords, winds, current, thickness, width,
                 compute_bfield, use_gauss, n_grid=120, axis_num=200):
        super().__init__()
        self.coords    = coords
        self.winds     = winds
        self.current   = current
        self.thickness = thickness
        self.width     = width
        self.compute_b = compute_bfield
        self.use_gauss = use_gauss
        self.n_grid    = int(n_grid)
        self.axis_num  = int(axis_num)

    @pyqtSlot()
    def run(self):
        engine = CoilAnalysis(
            self.coords,
            self.winds,
            self.current,
            self.thickness,
            self.width
        )
        engine.run_analysis(
            compute_bfield=self.compute_b,
            use_gauss=self.use_gauss,
            n_grid=self.n_grid,
            axis_num=self.axis_num,
            progress_callback=self.progress.emit,
            stage_callback=self.stage.emit
        )
        self.finished.emit(engine)

class LandingPage(LogoMixin, QWidget):
    def __init__(self, logo_path):
        QWidget.__init__(self)
        LogoMixin.__init__(self, logo_path)

        # === Input form ===
        form = QFormLayout()
        self.spin_winds    = QSpinBox();      self.spin_winds.setRange(1,10000);  self.spin_winds.setValue(200)
        self.dspin_current = QDoubleSpinBox();self.dspin_current.setRange(0,1e6); self.dspin_current.setDecimals(1); self.dspin_current.setValue(300)
        self.dspin_thick   = QDoubleSpinBox();self.dspin_thick.setRange(0,1e6);   self.dspin_thick.setDecimals(1);   self.dspin_thick.setValue(80.0)
        self.dspin_width   = QDoubleSpinBox();self.dspin_width.setRange(0.1,100); self.dspin_width.setDecimals(2); self.dspin_width.setValue(4.00)
        form.addRow("Number of Winds:",     self.spin_winds)
        form.addRow("Current (A):",         self.dspin_current)
        form.addRow("Tape Thickness (µm):", self.dspin_thick)
        form.addRow("Tape Width (mm):",     self.dspin_width)

        # On-axis sample count (always visible)
        self.spin_axis_pts = QSpinBox()
        self.spin_axis_pts.setRange(50, 1000)
        self.spin_axis_pts.setSingleStep(50)
        self.spin_axis_pts.setValue(200)
        form.addRow("On-Axis Samples:", self.spin_axis_pts)

        self.chk_bdist = QCheckBox("Calculate B‑field Cross-section")
        form.addRow("", self.chk_bdist)

        # --- Threshold (spin box only) ---
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.01, 20.00)
        self.spin_thresh.setSingleStep(0.01)
        self.spin_thresh.setDecimals(2)
        self.spin_thresh.setValue(5.00)
        self.spin_thresh.setEnabled(False)

        self.lbl_thresh = QLabel("Cross‑section max |B|:")
        self.lbl_thresh.setEnabled(False)

        # --- Grid resolution (only enabled with cross-section) ---
        self.spin_grid_res = QSpinBox()
        self.spin_grid_res.setRange(32, 512)
        self.spin_grid_res.setSingleStep(8)
        self.spin_grid_res.setValue(120)
        self.spin_grid_res.setEnabled(False)

        self.lbl_grid_res = QLabel("Cross‑section Grid (pts/axis):")
        self.lbl_grid_res.setEnabled(False)

        # Enable/disable cross-section controls when checkbox toggles
        def _toggle_thresh(state):
            enabled = (state == Qt.Checked)
            self.spin_thresh.setEnabled(enabled)
            self.lbl_thresh.setEnabled(enabled)
            self.spin_grid_res.setEnabled(enabled)
            self.lbl_grid_res.setEnabled(enabled)

        self.chk_bdist.stateChanged.connect(_toggle_thresh)

        # Threshold row
        h_thresh = QHBoxLayout()
        h_thresh.addWidget(self.lbl_thresh)
        h_thresh.addWidget(self.spin_thresh)
        form.addRow("", h_thresh)

        # Grid resolution row
        h_grid = QHBoxLayout()
        h_grid.addWidget(self.lbl_grid_res)
        h_grid.addWidget(self.spin_grid_res)
        form.addRow("", h_grid)

        self.chk_gauss = QCheckBox("Use Gaussian Quadrature")
        form.addRow("", self.chk_gauss)

        # === Buttons ===
        self.btn_load    = QPushButton("Load CSV...")
        self.btn_preview = QPushButton("Preview Curve"); self.btn_preview.setEnabled(False)
        self.btn_next    = QPushButton("Generate");      self.btn_next.setEnabled(False)

        # === Assemble layout ===
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.btn_load)
        lay.addWidget(self.btn_preview)
        lay.addWidget(self.btn_next)

        # placeholder 3D canvas
        self._add_placeholder()

        # connect signals
        self.btn_load.clicked.connect(self.load_csv)
        self.btn_preview.clicked.connect(self.preview_curve)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        self.paint_logo(painter, self)

    def _add_placeholder(self):
        def placeholder(ax, ctx=None):
            ax.text2D(0.5, 0.5, "No preview",
                      ha='center', va='center',
                      transform=ax.transAxes, fontsize=16)
            ax.axis('off')

        self.placeholder = make_canvas(placeholder, None, projection='3d')
        self.layout().addWidget(self.placeholder)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select coil CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        df = pd.read_csv(path)
        if set(['x','y','z']).issubset(df.columns):
            coords = df[['x','y','z']].values
        else:
            coords = df.iloc[:, :3].values

        # close the loop if needed
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack((coords, coords[0]))

        self.coords        = coords
        self.current_fname = os.path.basename(path)
        self.btn_load.setText(f"Uploaded: {self.current_fname}")
        self.btn_preview.setEnabled(True)
        self.btn_next.setEnabled(True)

    def preview_curve(self):
        if not hasattr(self, 'coords'):
            return
        # remove old canvases
        for w in self.findChildren(FigureCanvas):
            self.layout().removeWidget(w)
            w.deleteLater()
        def simple_preview(ax, ctx=None):
            ax.plot(self.coords[:,0], self.coords[:,1], self.coords[:,2],
                    color='black', lw=2)
            ax.axis('off')
            name = getattr(self, 'current_fname', 'Coil')
            fs = max(8, min(12, 200 // len(name)))
            ax.set_title(name, fontsize=fs)
            
            # Equal axis scaling
            try:
                coords = self.coords
                x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
                y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
                z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
                x_range = max(x_max - x_min, 1e-10)
                y_range = max(y_max - y_min, 1e-10)
                z_range = max(z_max - z_min, 1e-10)
                max_range = max(x_range, y_range, z_range)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                z_center = (z_min + z_max) / 2
                half_range = (max_range + max_range * 0.1) / 2
                ax.set_xlim(x_center - half_range, x_center + half_range)
                ax.set_ylim(y_center - half_range, y_center + half_range)
                ax.set_zlim(z_center - half_range, z_center + half_range)
                ax.set_box_aspect((1,1,1))
            except:
                pass
        canvas = make_canvas(simple_preview, None, projection='3d')
        self.layout().insertWidget(4, canvas)
        
    def reset(self):
        # remove ALL canvases (including placeholder)
        for cv in self.findChildren(FigureCanvas):
            self.layout().removeWidget(cv)
            cv.deleteLater()
        # add back exactly one placeholder
        self._add_placeholder()
        # reset controls
        self.btn_load.setText("Load CSV...")
        self.btn_preview.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.spin_axis_pts.setValue(200)
        self.chk_bdist.setChecked(False)
        self.spin_thresh.setValue(5.00)
        self.spin_thresh.setEnabled(False)
        self.lbl_thresh.setEnabled(False)
        self.spin_grid_res.setValue(120)
        self.spin_grid_res.setEnabled(False)
        self.lbl_grid_res.setEnabled(False)
        self.chk_gauss.setChecked(False)
        # drop stored data
        if hasattr(self, 'coords'):
            del self.coords
        if hasattr(self, 'current_fname'):
            del self.current_fname
            
    def get_cross_section_threshold(self) -> float:
        return self.spin_thresh.value()

    def get_grid_resolution(self) -> int:
        return self.spin_grid_res.value()

    def get_axis_samples(self) -> int:
        return self.spin_axis_pts.value()


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(600, 450)
        layout = QVBoxLayout(self)
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        
        window_color = QApplication.palette().color(QPalette.Window).name()
        browser.setStyleSheet(f"QTextBrowser {{ background: {window_color}; border: none; }}")
        
        if getattr(sys, "frozen", False):
            BASE_DIR = Path(sys._MEIPASS)          # PyInstaller temp dir
        else:
            BASE_DIR = Path(__file__).resolve().parent
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        HELP_FILE = BASE_DIR / "resources" / "html" / "help.txt"
                
        raw_html = HELP_FILE.read_text(encoding="utf-8-sig")
        
        tmpl = Template(raw_html)
        HELP_HTML = tmpl.safe_substitute(VERSION=__version__)

        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CalcSX™ – v{__version__}")
        
        if getattr(sys, "frozen", False):
            BASE_DIR = Path(sys._MEIPASS)          # PyInstaller temp dir
        else:
            BASE_DIR = Path(__file__).resolve().parent.parent
        
        LOGO_PATH = BASE_DIR / "resources" / "images" / "CFRC_white.png"
        self.landing = LandingPage(str(LOGO_PATH))
        
        # Central UI (replace with your existing widget)
        self.resize(800, 600)
        
        self.stack   = QStackedWidget()

        self.results = ResultsPage()

        self.stack.addWidget(self.landing)
        self.stack.addWidget(self.results)
        self.setCentralWidget(self.stack)

        # navigation
        self.landing.btn_next.clicked.connect(self.show_results)
        self.results.btn_back.clicked.connect(self.show_landing)


        # (Optional) add menus here
        # file_menu = menubar.addMenu("File")
        # help_menu = menubar.addMenu("Help")
    
        self.help_btn = QToolButton(self)
        self.help_btn.setText("?")
        self.help_btn.setToolTip("Help")
        self.help_btn.clicked.connect(self.show_help_dialog)
        self.help_btn.setFixedSize(24, 24)
        self.help_btn.setStyleSheet(
            "QToolButton {"
            "  background: rgba(255,255,255,0.7);"
            "  border: 1px solid #660;"
            "  border-radius: 6px;"
            "  font-weight: bold;"
            "}"
            "QToolButton:hover { background: rgba(255,255,255,0.9); }"
        )
        self.help_btn.raise_()  # ensure it stays on top
        
        self.stack.currentChanged.connect(self._update_help_visibility)
        self._update_help_visibility(self.stack.currentIndex())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Place button at top-right of client area
        margin = 6
        x = self.width() - self.help_btn.width() - margin
        y = margin
        self.help_btn.move(x, y)

    def show_help_dialog(self):
        dlg = HelpDialog(self)
        dlg.exec_()

    def show_results(self):
        # disable Generate button
        self.landing.btn_next.setEnabled(False)

        # start progress dialog
        self.reporter = ProgressReporter(self, title="Generating Plots…")
        self.reporter.start()

        # gather parameters
        coords    = self.landing.coords
        winds     = self.landing.spin_winds.value()
        current   = self.landing.dspin_current.value()
        thick     = self.landing.dspin_thick.value()
        width     = self.landing.dspin_width.value()
        want_b    = self.landing.chk_bdist.isChecked()
        want_gq   = self.landing.chk_gauss.isChecked()
        n_grid    = self.landing.get_grid_resolution()
        axis_num  = self.landing.get_axis_samples()

        # set up worker thread
        self.thread = QThread(self)
        self.worker = AnalysisWorker(
            coords, winds, current, thick, width,
            want_b, want_gq, n_grid, axis_num
        )
        self.worker.moveToThread(self.thread)

        # connect signals
        self.worker.progress.connect(self.reporter.report)
        self.worker.stage.connect(self.reporter.set_stage)
        self.worker.finished.connect(lambda eng: self.on_analysis_done(eng, want_b))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # start analysis
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def on_analysis_done(self, engine, show_b):
        # finish progress dialog
        self.reporter.finish()
        
        if show_b:
            engine.cross_section_threshold = self.landing.get_cross_section_threshold()

        # show results
        self.results.setup(engine, show_b)
        self.stack.setCurrentWidget(self.results)

        # re-enable Generate
        self.landing.btn_next.setEnabled(True)

    def show_landing(self):
        self.landing.reset()
        self.stack.setCurrentWidget(self.landing)
        
    def _update_help_visibility(self, index: int):
        # Show only when landing page is active
        self.help_btn.setVisible(self.stack.widget(index) is self.landing)
