# primary/main_utils.py
"""
Fusion 360-style workbench layout for CalcSX.

Structure
─────────
  MainWindow
  └── QWidget (central)
        ├── RibbonBar          (top, ~88 px) — tabbed tool groups
        └── QSplitter (H)
              ├── QWidget (left, ~250 px)
              │     ├── BrowserPanel   — collapsible layer tree, eye-icon toggles
              │     └── PropertiesPanel — coil parameters + results summary
              └── Workspace3DView    — single persistent matplotlib Axes3D
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from string import Template

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QCheckBox,
    QTextBrowser,
    QDialog,
    QLabel,
    QFrame,
    QScrollArea,
    QSplitter,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QPixmap

from physics.physics_utils import CoilAnalysis
from gui.gui_utils import ProgressReporter, THEME
from views.workspace_3d import Workspace3DView

from version import __version__ as version_module
__version__ = getattr(version_module, "__version__", "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    progress = pyqtSignal(int)
    stage    = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, coords, winds, current, thickness, width,
                 use_gauss, n_grid=120, axis_num=200):
        super().__init__()
        self.coords    = coords
        self.winds     = winds
        self.current   = current
        self.thickness = thickness
        self.width     = width
        self.use_gauss = use_gauss
        self.n_grid    = int(n_grid)
        self.axis_num  = int(axis_num)

    @pyqtSlot()
    def run(self):
        engine = CoilAnalysis(
            self.coords, self.winds, self.current,
            self.thickness, self.width,
        )
        engine.run_analysis(
            compute_bfield=False,
            use_gauss=self.use_gauss,
            n_grid=self.n_grid,
            axis_num=self.axis_num,
            progress_callback=self.progress.emit,
            stage_callback=self.stage.emit,
        )
        self.finished.emit(engine)



class FieldLinesWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)   # (lines, B_mags) tuple

    def __init__(self, engine, n_seeds: int):
        super().__init__()
        self.engine  = engine
        self.n_seeds = int(n_seeds)

    @pyqtSlot()
    def run(self):
        lines, B_mags = self.engine.compute_field_lines(
            n_seeds=self.n_seeds,
            progress_callback=self.progress.emit,
        )
        self.finished.emit((lines, B_mags))


class CrossSectionWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)   # (X, Y, B_plane, e1, e2, center, R) tuple

    def __init__(self, engine, axis_offset: float = 0.0):
        super().__init__()
        self.engine      = engine
        self.axis_offset = float(axis_offset)

    @pyqtSlot()
    def run(self):
        data = self.engine.compute_bfield_midplane(
            grid_size=80,
            axis_offset=self.axis_offset,
            progress_callback=self.progress.emit,
        )
        self.finished.emit(data)


# ─────────────────────────────────────────────────────────────────────────────
# Ribbon toolbar
# ─────────────────────────────────────────────────────────────────────────────

class _RibbonBtn(QFrame):
    """
    54×54 px icon-style ribbon button.
    Symbol (large) sits above a text label.
    Hover / press feedback via background fill.
    """
    clicked = pyqtSignal()

    def __init__(self, symbol: str, label: str,
                 enabled: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self._enabled = enabled
        self._pressed = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 6, 2, 3)
        lay.setSpacing(1)

        self._sym_lbl = QLabel(symbol)
        self._sym_lbl.setAlignment(Qt.AlignCenter)

        self._txt_lbl = QLabel(label)
        self._txt_lbl.setAlignment(Qt.AlignCenter)
        self._txt_lbl.setWordWrap(True)

        lay.addWidget(self._sym_lbl, stretch=1)
        lay.addWidget(self._txt_lbl)

        self._set_enabled_style(enabled)
        self._set_bg("transparent")

        if enabled:
            self.setCursor(Qt.PointingHandCursor)

    # ── Enable/disable ────────────────────────────────────────────────────────

    def set_action_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._set_enabled_style(enabled)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        if not enabled:
            self._set_bg("transparent")

    def _set_enabled_style(self, enabled: bool) -> None:
        sym_c = THEME['text']      if enabled else '#4a4a4a'
        lbl_c = THEME['text_dim'] if enabled else '#3a3a3a'
        self._sym_lbl.setStyleSheet(
            f"font-size:15pt; color:{sym_c}; background:transparent;"
        )
        self._txt_lbl.setStyleSheet(
            f"font-size:7pt; color:{lbl_c}; background:transparent;"
        )

    def _set_bg(self, color: str) -> None:
        self.setStyleSheet(
            f"QFrame {{ border-radius:3px; background:{color}; }}"
        )

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if self._enabled and e.button() == Qt.LeftButton:
            self._pressed = True
            self._set_bg(THEME['hi_blue'])

    def mouseReleaseEvent(self, e):
        if self._enabled and self._pressed:
            self._pressed = False
            if self.rect().contains(e.pos()):
                self._set_bg(THEME['input'])
                self.clicked.emit()
            else:
                self._set_bg("transparent")

    def enterEvent(self, e):
        if self._enabled and not self._pressed:
            self._set_bg(THEME['input'])

    def leaveEvent(self, e):
        if not self._pressed:
            self._set_bg("transparent")


def _vbar() -> QFrame:
    """Thin vertical separator between ribbon groups."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"color:{THEME['border']};")
    return f


def _ribbon_group(label: str, buttons: list) -> QWidget:
    """Labeled group of ribbon buttons separated by a group label below."""
    w   = QWidget()
    out = QVBoxLayout(w)
    out.setContentsMargins(4, 4, 4, 0)
    out.setSpacing(0)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    for b in buttons:
        row.addWidget(b)

    glbl = QLabel(label)
    glbl.setAlignment(Qt.AlignCenter)
    glbl.setStyleSheet(
        f"color:{THEME['text_dim']}; font-size:7pt; "
        f"border-top:1px solid {THEME['border']}; padding-top:1px;"
    )

    out.addLayout(row, stretch=1)
    out.addWidget(glbl)
    return w


class RibbonBar(QWidget):
    """
    Two-row ribbon.
    Row 1 — tab strip (SIMULATION | INSPECT | CONSTRUCT | UTILITIES)
    Row 2 — tool groups for the active tab
    """
    load_csv             = pyqtSignal()
    run_analysis         = pyqtSignal()
    compute_field_lines  = pyqtSignal()
    compute_cross_section = pyqtSignal()
    show_help            = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        self.setStyleSheet(
            f"background:{THEME['panel']}; "
            f"border-bottom:1px solid {THEME['border']};"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Tab strip ─────────────────────────────────────────────────────────
        tab_bar = QWidget()
        tab_bar.setFixedHeight(24)
        tab_bar.setStyleSheet(
            f"background:{THEME['bg']}; border-bottom:1px solid {THEME['border']};"
        )
        tb_lay = QHBoxLayout(tab_bar)
        tb_lay.setContentsMargins(8, 0, 0, 0)
        tb_lay.setSpacing(0)

        app_lbl = QLabel(f"  CalcSX™ v{__version__}  ")
        app_lbl.setStyleSheet(
            f"color:{THEME['accent']}; font-size:8pt; font-weight:bold; "
            f"border-right:1px solid {THEME['border']}; padding-right:8px;"
        )
        tb_lay.addWidget(app_lbl)

        self._tabs: dict = {}
        self._active_tab = "SIMULATION"
        for i, name in enumerate(("SIMULATION", "INSPECT", "CONSTRUCT", "UTILITIES")):
            btn = QPushButton(name)
            btn.setFlat(True)
            btn.setFixedHeight(24)
            btn.setStyleSheet(self._tab_style(active=(i == 0)))
            btn.clicked.connect(lambda _, n=name: self._activate(n))
            self._tabs[name] = btn
            tb_lay.addWidget(btn)
        tb_lay.addStretch(1)
        root.addWidget(tab_bar)

        # ── Tool groups area ──────────────────────────────────────────────────
        tool_area = QWidget()
        ta_lay = QHBoxLayout(tool_area)
        ta_lay.setContentsMargins(4, 0, 4, 0)
        ta_lay.setSpacing(0)

        # Build tool buttons (each is its own object — no shared widget refs)
        self._btn_load = _RibbonBtn("▲", "Load\nCSV")
        self._btn_run  = _RibbonBtn("▶", "Run\nAnalysis")
        btn_help_sim   = _RibbonBtn("?", "Help")

        # SIMULATION groups
        self._sim_w = QWidget()
        sl = QHBoxLayout(self._sim_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(_ribbon_group("FILE",     [self._btn_load]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("ANALYSIS", [self._btn_run]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("HELP",     [btn_help_sim]))
        sl.addStretch(1)

        # INSPECT tab
        self._btn_field_lines = _RibbonBtn("∿", "Field\nLines",   enabled=False)
        self._btn_cross_sec   = _RibbonBtn("⊡", "Cross\nSection", enabled=False)
        self._inspect_w = QWidget()
        il = QHBoxLayout(self._inspect_w)
        il.setContentsMargins(8, 0, 0, 0)
        il.setSpacing(4)
        il.addWidget(_ribbon_group("FIELD", [
            self._btn_field_lines,
            self._btn_cross_sec,
        ]))
        il.addStretch(1)
        self._inspect_w.hide()

        # CONSTRUCT (Tier 2 stub)
        self._construct_w = QWidget()
        cl = QHBoxLayout(self._construct_w)
        cl.setContentsMargins(8, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(_ribbon_group("TIER 2", [
            _RibbonBtn("+", "New\nCoil",       enabled=False),
            _RibbonBtn("⬡", "Tape\nThickness", enabled=False),
            _RibbonBtn("⊕", "Edit\nGeometry",  enabled=False),
        ]))
        cl.addStretch(1)
        self._construct_w.hide()

        # UTILITIES (stub)
        self._util_w = QWidget()
        ul = QHBoxLayout(self._util_w)
        ul.setContentsMargins(8, 0, 0, 0)
        ul.setSpacing(4)
        btn_help_util = _RibbonBtn("?", "Help")
        btn_help_util.clicked.connect(self.show_help)
        ul.addWidget(_ribbon_group("HELP", [btn_help_util]))
        ul.addStretch(1)
        self._util_w.hide()

        self._tab_widgets = {
            "SIMULATION": self._sim_w,
            "INSPECT":    self._inspect_w,
            "CONSTRUCT":  self._construct_w,
            "UTILITIES":  self._util_w,
        }
        for w in self._tab_widgets.values():
            ta_lay.addWidget(w)

        root.addWidget(tool_area, stretch=1)

        # Wire signals
        self._btn_load.clicked.connect(self.load_csv)
        self._btn_run.clicked.connect(self.run_analysis)
        self._btn_field_lines.clicked.connect(self.compute_field_lines)
        self._btn_cross_sec.clicked.connect(self.compute_cross_section)
        btn_help_sim.clicked.connect(self.show_help)

    # ── Public ────────────────────────────────────────────────────────────────

    def set_run_enabled(self, on: bool) -> None:
        self._btn_run.set_action_enabled(on)

    def set_inspect_enabled(self, on: bool) -> None:
        self._btn_field_lines.set_action_enabled(on)
        self._btn_cross_sec.set_action_enabled(on)

    # ── Private ───────────────────────────────────────────────────────────────

    def _activate(self, name: str) -> None:
        if name == self._active_tab:
            return
        self._tab_widgets[self._active_tab].hide()
        self._active_tab = name
        self._tab_widgets[name].show()
        for n, btn in self._tabs.items():
            btn.setStyleSheet(self._tab_style(active=(n == name)))

    @staticmethod
    def _tab_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background:transparent; color:{THEME['text']}; "
                f"border:none; border-bottom:2px solid {THEME['accent']}; "
                f"padding:0 14px; font-size:8pt; font-weight:600; }}"
            )
        return (
            f"QPushButton {{ background:transparent; color:{THEME['text_dim']}; "
            f"border:none; border-bottom:2px solid transparent; "
            f"padding:0 14px; font-size:8pt; }}"
            f"QPushButton:hover {{ color:{THEME['text']}; "
            f"background:{THEME['input']}; }}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Browser panel  (Fusion 360-style layer tree)
# ─────────────────────────────────────────────────────────────────────────────

_GROUPS = ("Coils", "Analysis")

_LAYER_META: dict = {
    'Coil':          ('Coils',    THEME['accent']),
    'Forces':        ('Analysis', THEME['accent2']),
    'Stress':        ('Analysis', '#e05050'),
    'B Axis':        ('Analysis', '#80d8ff'),
    'Field Lines':   ('Analysis', '#80ffff'),
    'Cross Section': ('Analysis', '#ff9800'),
}


class _EyeBtn(QPushButton):
    """18×18 flat toggle: ● visible / ○ hidden."""
    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFlat(True)
        self.setFixedSize(18, 18)
        self._refresh()
        self.toggled.connect(self._refresh)

    def _refresh(self):
        c = THEME['text'] if self.isChecked() else THEME['text_dim']
        self.setStyleSheet(
            f"QPushButton {{ color:{c}; font-size:9pt; border:none; "
            f"background:transparent; }}"
        )


class BrowserPanel(QWidget):
    """
    Fusion 360-style browser: collapsible groups with eye-icon toggles.
    Emits layer_toggled(name: str, visible: bool).
    """
    layer_toggled = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QLabel("  BROWSER")
        hdr.setFixedHeight(22)
        hdr.setStyleSheet(
            f"background:{THEME['bg']}; color:{THEME['text_dim']}; "
            f"font-size:7pt; letter-spacing:2px; "
            f"border-bottom:1px solid {THEME['border']};"
        )
        root.addWidget(hdr)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setIndentation(14)
        self._tree.setAnimated(True)
        self._tree.setStyleSheet(f"""
            QTreeWidget {{
                background:{THEME['panel']};
                border:none;
                outline:none;
            }}
            QTreeWidget::item {{
                height:22px;
                padding:0;
            }}
            QTreeWidget::item:selected {{
                background:{THEME['hi_blue']};
            }}
        """)
        root.addWidget(self._tree, stretch=1)

        # Group items (always present)
        self._group_items: dict = {}
        self._layer_items: dict = {}
        self._eye_btns:    dict = {}
        self._group_eyes:  dict = {}

        for grp in _GROUPS:
            g = QTreeWidgetItem()
            g.setFlags(Qt.ItemIsEnabled)
            self._tree.addTopLevelItem(g)
            g.setExpanded(True)

            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(2, 0, 2, 0)
            lay.setSpacing(4)

            g_eye = _EyeBtn()
            g_eye.toggled.connect(
                lambda checked, gname=grp: self._group_eye_toggled(gname, checked)
            )
            self._group_eyes[grp] = g_eye
            lay.addWidget(g_eye)

            g_lbl = QLabel(grp.upper())
            g_lbl.setStyleSheet(
                f"color:{THEME['text_dim']}; font-size:8pt; "
                f"font-weight:600; letter-spacing:1px;"
            )
            lay.addWidget(g_lbl, stretch=1)

            self._tree.setItemWidget(g, 0, w)
            g.setSizeHint(0, QSize(0, 22))
            self._group_items[grp] = g

    # ── Public ────────────────────────────────────────────────────────────────

    def add_layer(self, name: str) -> None:
        if name in self._layer_items or name not in _LAYER_META:
            return
        grp, color = _LAYER_META[name]
        parent = self._group_items[grp]

        item = QTreeWidgetItem(parent)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(4)

        eye = _EyeBtn()
        eye.toggled.connect(
            lambda checked, n=name: self.layer_toggled.emit(n, checked)
        )
        self._eye_btns[name] = eye
        lay.addWidget(eye)

        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background:{color}; border-radius:1px;")
        lay.addWidget(swatch)

        n_lbl = QLabel(name)
        n_lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        lay.addWidget(n_lbl, stretch=1)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._layer_items[name] = item
        parent.setExpanded(True)

    def remove_layer(self, name: str) -> None:
        if name not in self._layer_items:
            return
        item = self._layer_items.pop(name)
        self._eye_btns.pop(name, None)
        grp = _LAYER_META[name][0]
        self._group_items[grp].removeChild(item)

    def remove_all_layers(self) -> None:
        for name in list(self._layer_items.keys()):
            self.remove_layer(name)

    # ── Private ───────────────────────────────────────────────────────────────

    def _group_eye_toggled(self, group: str, checked: bool) -> None:
        for name, (grp, _) in _LAYER_META.items():
            if grp == group and name in self._eye_btns:
                btn = self._eye_btns[name]
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)
                self.layer_toggled.emit(name, checked)


# ─────────────────────────────────────────────────────────────────────────────
# Properties panel  (coil parameters + results summary)
# ─────────────────────────────────────────────────────────────────────────────

class PropertiesPanel(QScrollArea):
    """
    Bottom section of the left sidebar.
    Coil parameters (always shown after CSV load) + results summary (after analysis).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        self.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        hdr = QLabel("  PROPERTIES")
        hdr.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:7pt; letter-spacing:2px;"
        )
        lay.addWidget(hdr)

        # Coil parameter form
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(4)
        form.setContentsMargins(0, 2, 0, 0)

        self.spin_winds    = QSpinBox()
        self.spin_winds.setRange(1, 10_000);   self.spin_winds.setValue(200)

        self.dspin_current = QDoubleSpinBox()
        self.dspin_current.setRange(0, 1e6);   self.dspin_current.setDecimals(1);  self.dspin_current.setValue(300.0)

        self.dspin_thick   = QDoubleSpinBox()
        self.dspin_thick.setRange(0, 1e6);     self.dspin_thick.setDecimals(1);    self.dspin_thick.setValue(80.0)

        self.dspin_width   = QDoubleSpinBox()
        self.dspin_width.setRange(0.1, 100);   self.dspin_width.setDecimals(2);    self.dspin_width.setValue(4.00)

        self.spin_axis_pts = QSpinBox()
        self.spin_axis_pts.setRange(50, 1000); self.spin_axis_pts.setSingleStep(50); self.spin_axis_pts.setValue(200)

        self.chk_gauss            = QCheckBox("Gaussian Quadrature")
        self.chk_normalize_forces = QCheckBox("Normalize Force Vectors")

        form.addRow("Winds:",           self.spin_winds)
        form.addRow("Current (A):",     self.dspin_current)
        form.addRow("Tape Thick (µm):", self.dspin_thick)
        form.addRow("Tape Width (mm):", self.dspin_width)
        form.addRow("Axis Samples:",    self.spin_axis_pts)
        form.addRow("",                 self.chk_gauss)
        form.addRow("",                 self.chk_normalize_forces)
        lay.addLayout(form)

        # INSPECT — field line seeds
        lay.addWidget(_hdivider())
        irow = QHBoxLayout()
        irow.setContentsMargins(0, 0, 0, 0)
        irow.setSpacing(6)
        ilbl = QLabel("Field Seeds:")
        ilbl.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8pt;")
        self.spin_field_seeds = QSpinBox()
        self.spin_field_seeds.setRange(8, 60)
        self.spin_field_seeds.setSingleStep(4)
        self.spin_field_seeds.setValue(20)
        irow.addWidget(ilbl)
        irow.addWidget(self.spin_field_seeds, stretch=1)
        lay.addLayout(irow)

        srow = QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(6)
        slbl = QLabel("Section Pos. (m):")
        slbl.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8pt;")
        self.dspin_cs_offset = QDoubleSpinBox()
        self.dspin_cs_offset.setRange(-5.0, 5.0)
        self.dspin_cs_offset.setSingleStep(0.01)
        self.dspin_cs_offset.setDecimals(3)
        self.dspin_cs_offset.setValue(0.0)
        srow.addWidget(slbl)
        srow.addWidget(self.dspin_cs_offset, stretch=1)
        lay.addLayout(srow)

        lay.addWidget(_hdivider())

        # Results summary (shown after analysis)
        self._sum_w = QWidget()
        self._sum_w.hide()
        sl = QVBoxLayout(self._sum_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)

        sl.addWidget(_section_lbl("RESULTS"))
        self._sum_lbls: dict = {}
        for key in ("B cent.", "B axial", "Peak |B|", "Peak F", "Peak σ", "Arc len."):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            kl = QLabel(key + ":")
            kl.setStyleSheet(
                f"color:{THEME['text_dim']}; font-size:8pt; min-width:52px;"
            )
            vl = QLabel("—")
            vl.setStyleSheet(
                f"color:{THEME['accent']}; font-size:8pt; font-weight:bold;"
            )
            self._sum_lbls[key] = vl
            row.addWidget(kl)
            row.addWidget(vl, stretch=1)
            sl.addLayout(row)

        lay.addWidget(self._sum_w)
        lay.addStretch()

    # ── Public ────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            'winds':     self.spin_winds.value(),
            'current':   self.dspin_current.value(),
            'thickness': self.dspin_thick.value(),
            'width':     self.dspin_width.value(),
            'use_gauss': self.chk_gauss.isChecked(),
            'axis_num':  self.spin_axis_pts.value(),
        }

    def get_field_seeds(self) -> int:
        return self.spin_field_seeds.value()

    def get_cs_offset(self) -> float:
        return self.dspin_cs_offset.value()

    def update_summary(self, engine) -> None:
        def _set(key, text):
            self._sum_lbls[key].setText(text)
        try:  _set("B cent.",  f"{engine.B_magnitude:.4f} T")
        except Exception: pass
        try:  _set("B axial",  f"{engine.B_axial:.4f} T")
        except Exception: pass
        try:
            if engine.bfield_axis_mag is not None:
                mag = np.asarray(engine.bfield_axis_mag)
                z   = np.asarray(engine.bfield_axis_z)
                idx = int(np.argmax(mag))
                _set("Peak |B|", f"{mag[idx]:.4f} T @ {z[idx]:+.3f} m")
        except Exception: pass
        try:
            F = np.asarray(engine.F_mags)
            _set("Peak F",  f"{F.max()/1000:.2f} kN/m")
        except Exception: pass
        try:
            s = np.asarray(engine.hoop_stress)
            _set("Peak σ",  f"{s.max()/1e6:.2f} MPa")
        except Exception: pass
        try:
            total = float(np.sum(
                np.linalg.norm(np.diff(engine.coords, axis=0), axis=1)
            ))
            _set("Arc len.", f"{total:.3f} m")
        except Exception: pass
        self._sum_w.show()


# ─────────────────────────────────────────────────────────────────────────────
# Help dialog
# ─────────────────────────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(620, 480)
        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            f"QTextBrowser {{ background:{THEME['panel']}; border:none; }}"
        )
        BASE_DIR = (
            Path(sys._MEIPASS) if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent.parent
        )
        try:
            raw = (BASE_DIR / "resources" / "html" / "help.txt").read_text(
                encoding="utf-8-sig"
            )
            browser.setHtml(Template(raw).safe_substitute(VERSION=__version__))
        except Exception:
            browser.setPlainText("Help file not found.")
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CalcSX™ – v{__version__}")
        self.resize(1280, 800)

        self._coords   = None
        self._engine   = None
        self._a_thread = None
        self._a_worker = None
        self._i_thread = None
        self._i_worker = None

        # ── Central widget ────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── Ribbon ────────────────────────────────────────────────────────────
        self.ribbon = RibbonBar()
        main_lay.addWidget(self.ribbon)

        # ── Workspace splitter ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_lay.addWidget(splitter, stretch=1)

        # Left panel
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

        # ── Wire signals ──────────────────────────────────────────────────────
        self.ribbon.load_csv.connect(self._on_load_csv)
        self.ribbon.run_analysis.connect(self._on_run_analysis)
        self.ribbon.compute_field_lines.connect(self._on_compute_field_lines)
        self.ribbon.compute_cross_section.connect(self._on_compute_cross_section)
        self.ribbon.show_help.connect(lambda: HelpDialog(self).exec_())
        self.browser.layer_toggled.connect(self.workspace.set_layer_visible)
        self.props.chk_normalize_forces.stateChanged.connect(
            self._on_normalize_forces_toggled
        )

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select coil CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            coords = (
                df[['x', 'y', 'z']].values
                if {'x', 'y', 'z'}.issubset(df.columns)
                else df.iloc[:, :3].values
            )
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack((coords, coords[0]))
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return

        fname = os.path.basename(path)
        self._coords = coords
        self._engine = None

        self.browser.remove_all_layers()
        self.workspace.load_coil(coords, fname)
        self.browser.add_layer('Coil')
        self.ribbon.set_inspect_enabled(False)
        self.props.show()

    def _on_run_analysis(self) -> None:
        if self._coords is None:
            QMessageBox.information(
                self, "No Coil", "Load a coil CSV before running analysis."
            )
            return

        self.ribbon.set_run_enabled(False)
        self.workspace.clear_analysis_layers()
        self.workspace.clear_inspect_layers()
        for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
            self.browser.remove_layer(nm)
        self.ribbon.set_inspect_enabled(False)

        params = self.props.get_params()
        self.reporter = ProgressReporter(self, title="Running Analysis…")
        self.reporter.start()

        self._a_thread = QThread(self)
        self._a_worker = AnalysisWorker(
            self._coords,
            params['winds'], params['current'],
            params['thickness'], params['width'],
            params['use_gauss'],
            axis_num=params['axis_num'],
        )
        self._a_worker.moveToThread(self._a_thread)
        self._a_worker.progress.connect(self.reporter.report)
        self._a_worker.stage.connect(self.reporter.set_stage)
        self._a_worker.finished.connect(self._on_analysis_done)
        self._a_worker.finished.connect(self._a_thread.quit)
        self._a_worker.finished.connect(self._a_worker.deleteLater)
        self._a_thread.finished.connect(self._a_thread.deleteLater)
        self._a_thread.started.connect(self._a_worker.run)
        self._a_thread.start()

    def _on_analysis_done(self, engine) -> None:
        self.reporter.finish()
        self._engine   = engine
        self._a_thread = None
        self._a_worker = None

        norm = self.props.chk_normalize_forces.isChecked()
        self.workspace.add_force_layer(engine, normalized=norm)
        self.workspace.add_stress_layer(engine)
        self.workspace.add_axis_layer(engine)

        for nm in ('Forces', 'Stress', 'B Axis'):
            self.browser.add_layer(nm)

        self.props.update_summary(engine)
        self.ribbon.set_run_enabled(True)
        self.ribbon.set_inspect_enabled(True)

    def _on_normalize_forces_toggled(self) -> None:
        """Re-render the Forces layer whenever the normalize checkbox changes."""
        if self._engine is None:
            return
        norm = self.props.chk_normalize_forces.isChecked()
        self.workspace.add_force_layer(self._engine, normalized=norm)

    def _on_compute_field_lines(self) -> None:
        if self._engine is None:
            QMessageBox.warning(self, "No Analysis",
                                "Run a full analysis first.")
            return
        if self._i_thread is not None and self._i_thread.isRunning():
            return
        self.workspace.clear_field_lines_layer()
        self.browser.remove_layer('Field Lines')
        n_seeds = self.props.get_field_seeds()
        self._inspect_reporter = ProgressReporter(self, title="Computing Field Lines…")
        self._inspect_reporter.start()
        self._i_thread = QThread(self)
        self._i_worker = FieldLinesWorker(self._engine, n_seeds)
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
        self.workspace.add_field_lines_layer(lines, B_mags)
        self.browser.add_layer('Field Lines')

    def _on_compute_cross_section(self) -> None:
        if self._engine is None:
            QMessageBox.warning(self, "No Analysis",
                                "Run a full analysis first.")
            return
        if self._i_thread is not None and self._i_thread.isRunning():
            return
        self.workspace.clear_cross_section_layer()
        self.browser.remove_layer('Cross Section')
        axis_offset = self.props.get_cs_offset()
        self._inspect_reporter = ProgressReporter(self, title="Computing Cross Section…")
        self._inspect_reporter.start()
        self._i_thread = QThread(self)
        self._i_worker = CrossSectionWorker(self._engine, axis_offset=axis_offset)
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
        self.workspace.add_cross_section_layer(X, Y, B_plane, e1, e2, center, R)
        self.browser.add_layer('Cross Section')


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hdivider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.HLine)
    d.setFrameShadow(QFrame.Sunken)
    d.setStyleSheet(f"color:{THEME['border']};")
    return d


def _section_lbl(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(
        f"color:{THEME['text_dim']}; font-size:7pt; letter-spacing:1.5px;"
    )
    return l
