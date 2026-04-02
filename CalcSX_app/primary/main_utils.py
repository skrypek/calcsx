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
    QTextBrowser,
    QColorDialog,
    QDialog,
    QInputDialog,
    QLabel,
    QFrame,
    QScrollArea,
    QSplitter,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QPixmap

from physics.physics_utils import CoilAnalysis
from physics.superposition import MultiCoilEnvironment
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
                 n_grid=120, axis_num=200, B_ext=None):
        super().__init__()
        self.coords    = coords
        self.winds     = winds
        self.current   = current
        self.thickness = thickness
        self.width     = width
        self.n_grid    = int(n_grid)
        self.axis_num  = int(axis_num)
        self.B_ext     = B_ext

    @pyqtSlot()
    def run(self):
        engine = CoilAnalysis(
            self.coords, self.winds, self.current,
            self.thickness, self.width,
            B_ext=self.B_ext,
        )
        engine.run_analysis(
            compute_bfield=False,
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


class GlobalFieldLinesWorker(QObject):
    """Compute field lines through the superposed B-field of ALL coils."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)   # (lines, B_mags) tuple

    def __init__(self, B_total, coil_infos: list, n_seeds: int):
        super().__init__()
        self.B_total    = B_total
        self.coil_infos = coil_infos
        self.n_seeds    = int(n_seeds)

    @pyqtSlot()
    def run(self):
        from physics.superposition import compute_global_field_lines
        lines, B_mags = compute_global_field_lines(
            self.B_total,
            self.coil_infos,
            n_seeds=self.n_seeds,
            progress_callback=self.progress.emit,
        )
        self.finished.emit((lines, B_mags))


# ─────────────────────────────────────────────────────────────────────────────
# Ribbon toolbar
# ─────────────────────────────────────────────────────────────────────────────

class _RibbonBtn(QFrame):
    """
    54×54 px icon-style ribbon button.
    Symbol (large) sits above a text label.
    Hover / press feedback via background fill.
    Optional checkable toggle mode (checkable=True).
    """
    clicked = pyqtSignal()
    toggled = pyqtSignal(bool)   # only fires when checkable=True

    def __init__(self, symbol: str, label: str,
                 enabled: bool = True, checkable: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self._enabled   = enabled
        self._pressed   = False
        self._checkable = checkable
        self._checked   = False

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

    def setChecked(self, checked: bool) -> None:
        if not self._checkable:
            return
        self._checked = checked
        self._set_bg(THEME['hi_blue'] if checked else "transparent")

    def isChecked(self) -> bool:
        return self._checked

    def set_action_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._set_enabled_style(enabled)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        if not enabled:
            self._checked = False
            self._set_bg("transparent")

    def _set_enabled_style(self, enabled: bool) -> None:
        sym_c = THEME['text']      if enabled else THEME['text_disabled']
        lbl_c = THEME['text_dim'] if enabled else THEME['text_dis_dim']
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

    def refresh_theme(self) -> None:
        """Re-apply current state using fresh THEME values."""
        self._set_enabled_style(self._enabled)
        self._set_bg(THEME['hi_blue'] if self._checked else "transparent")

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if self._enabled and e.button() == Qt.LeftButton:
            self._pressed = True
            self._set_bg(THEME['hi_blue'])

    def mouseReleaseEvent(self, e):
        if self._enabled and self._pressed:
            self._pressed = False
            if self.rect().contains(e.pos()):
                if self._checkable:
                    self._checked = not self._checked
                    self._set_bg(THEME['hi_blue'] if self._checked else "transparent")
                    self.toggled.emit(self._checked)
                else:
                    self._set_bg(THEME['input'])
                self.clicked.emit()
            else:
                self._set_bg(THEME['hi_blue'] if self._checked else "transparent")

    def enterEvent(self, e):
        if self._enabled and not self._pressed:
            self._set_bg(THEME['input'])

    def leaveEvent(self, e):
        if not self._pressed:
            self._set_bg(THEME['hi_blue'] if self._checked else "transparent")


def _vbar() -> QFrame:
    """Thin vertical separator between ribbon groups."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1)
    f.setObjectName('vbar')
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
    glbl.setObjectName('ribbon_grp_lbl')

    out.addLayout(row, stretch=1)
    out.addWidget(glbl)
    return w


class RibbonBar(QWidget):
    """
    Two-row ribbon.
    Row 1 — tab strip (SIMULATION | INSPECT | CONSTRUCT | UTILITIES)
    Row 2 — tool groups for the active tab
    """
    load_csv                 = pyqtSignal()
    run_analysis             = pyqtSignal()
    reanalyze_all            = pyqtSignal()
    compute_field_lines      = pyqtSignal()
    compute_cross_section    = pyqtSignal()
    global_field_toggled     = pyqtSignal(bool)
    translate_toggled        = pyqtSignal(bool)
    rotate_toggled           = pyqtSignal(bool)
    reset_transform          = pyqtSignal()
    show_help                = pyqtSignal()
    normalize_forces_toggled = pyqtSignal(bool)
    open_settings            = pyqtSignal()

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
        self._tab_bar = QWidget()
        self._tab_bar.setFixedHeight(24)
        self._tab_bar.setStyleSheet(
            f"background:{THEME['bg']}; border-bottom:1px solid {THEME['border']};"
        )
        tb_lay = QHBoxLayout(self._tab_bar)
        tb_lay.setContentsMargins(8, 0, 0, 0)
        tb_lay.setSpacing(0)

        self._app_lbl = QLabel(f"  CalcSX™ v{__version__}  ")
        self._app_lbl.setStyleSheet(
            f"color:{THEME['accent']}; font-size:8pt; font-weight:bold; "
            f"border-right:1px solid {THEME['border']}; padding-right:8px;"
        )
        tb_lay.addWidget(self._app_lbl)

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
        root.addWidget(self._tab_bar)

        # ── Tool groups area ──────────────────────────────────────────────────
        tool_area = QWidget()
        ta_lay = QHBoxLayout(tool_area)
        ta_lay.setContentsMargins(4, 0, 4, 0)
        ta_lay.setSpacing(0)

        # Build tool buttons (each is its own object — no shared widget refs)
        self._btn_load       = _RibbonBtn("▲", "Load\nCSV")
        self._btn_run        = _RibbonBtn("▶", "Run\nAnalysis")
        self._btn_reanalyze  = _RibbonBtn("⟳", "Re-analyze\nAll", enabled=False)
        btn_help_sim         = _RibbonBtn("?", "Help")

        # SIMULATION groups
        self._sim_w = QWidget()
        sl = QHBoxLayout(self._sim_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(_ribbon_group("FILE",     [self._btn_load]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("ANALYSIS", [self._btn_run, self._btn_reanalyze]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("HELP",     [btn_help_sim]))
        sl.addStretch(1)

        # INSPECT tab
        self._btn_field_lines  = _RibbonBtn("∿", "Field\nLines",   enabled=False)
        self._btn_cross_sec    = _RibbonBtn("⊡", "Cross\nSection", enabled=False)
        self._btn_global_field = _RibbonBtn("⊛", "Global\nField",  enabled=False, checkable=True)
        self._inspect_w = QWidget()
        il = QHBoxLayout(self._inspect_w)
        il.setContentsMargins(8, 0, 0, 0)
        il.setSpacing(4)
        il.addWidget(_ribbon_group("PER-COIL", [
            self._btn_field_lines,
            self._btn_cross_sec,
        ]))
        il.addWidget(_vbar())
        il.addWidget(_ribbon_group("GLOBAL", [
            self._btn_global_field,
        ]))
        il.addStretch(1)
        self._inspect_w.hide()

        # CONSTRUCT — coil positioning tools
        self._btn_translate = _RibbonBtn("⇱", "Translate", enabled=False, checkable=True)
        self._btn_rotate    = _RibbonBtn("↻", "Rotate",    enabled=False, checkable=True)
        self._btn_reset_xfm = _RibbonBtn("⌂", "Reset\nTransform", enabled=False)
        self._construct_w = QWidget()
        cl = QHBoxLayout(self._construct_w)
        cl.setContentsMargins(8, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(_ribbon_group("POSITION", [
            self._btn_translate,
            self._btn_rotate,
            self._btn_reset_xfm,
        ]))
        cl.addWidget(_vbar())
        cl.addWidget(_ribbon_group("ASSEMBLY", [
            _RibbonBtn("+", "New\nCoil",      enabled=False),
            _RibbonBtn("⊕", "Join\nAnalysis", enabled=False),
        ]))
        cl.addStretch(1)
        self._construct_w.hide()

        # UTILITIES
        self._btn_normalize = _RibbonBtn("⇈", "Normalize\nForces", checkable=True)
        self._btn_settings  = _RibbonBtn("⚙", "Settings")
        self._util_w = QWidget()
        ul = QHBoxLayout(self._util_w)
        ul.setContentsMargins(8, 0, 0, 0)
        ul.setSpacing(4)
        ul.addWidget(_ribbon_group("FORCES", [self._btn_normalize]))
        ul.addWidget(_vbar())
        btn_help_util = _RibbonBtn("?", "Help")
        btn_help_util.clicked.connect(self.show_help)
        ul.addWidget(_ribbon_group("APP", [self._btn_settings, btn_help_util]))
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
        self._btn_reanalyze.clicked.connect(self.reanalyze_all)
        self._btn_field_lines.clicked.connect(self.compute_field_lines)
        self._btn_cross_sec.clicked.connect(self.compute_cross_section)
        self._btn_global_field.toggled.connect(self.global_field_toggled)
        self._btn_translate.toggled.connect(self._on_translate_toggled)
        self._btn_rotate.toggled.connect(self._on_rotate_toggled)
        self._btn_reset_xfm.clicked.connect(self.reset_transform)
        btn_help_sim.clicked.connect(self.show_help)
        self._btn_normalize.toggled.connect(self.normalize_forces_toggled)
        self._btn_settings.clicked.connect(self.open_settings)

    # ── Public ────────────────────────────────────────────────────────────────

    def set_run_enabled(self, on: bool) -> None:
        self._btn_run.set_action_enabled(on)

    def set_inspect_enabled(self, on: bool) -> None:
        self._btn_field_lines.set_action_enabled(on)
        self._btn_cross_sec.set_action_enabled(on)
        self._btn_global_field.set_action_enabled(on)

    def set_construct_enabled(self, on: bool) -> None:
        self._btn_translate.set_action_enabled(on)
        self._btn_rotate.set_action_enabled(on)
        self._btn_reset_xfm.set_action_enabled(on)

    def _on_translate_toggled(self, checked: bool) -> None:
        """Mutual exclusion: turning on Translate turns off Rotate."""
        if checked:
            self._btn_rotate.setChecked(False)
        self.translate_toggled.emit(checked)

    def _on_rotate_toggled(self, checked: bool) -> None:
        """Mutual exclusion: turning on Rotate turns off Translate."""
        if checked:
            self._btn_translate.setChecked(False)
        self.rotate_toggled.emit(checked)

    # ── Private ───────────────────────────────────────────────────────────────

    def refresh_theme(self) -> None:
        """Re-apply all explicit styles after a theme switch."""
        self.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
        )
        self._tab_bar.setStyleSheet(
            f"background:{THEME['bg']}; border-bottom:1px solid {THEME['border']};"
        )
        self._app_lbl.setStyleSheet(
            f"color:{THEME['accent']}; font-size:8pt; font-weight:bold; "
            f"border-right:1px solid {THEME['border']}; padding-right:8px;"
        )
        for n, btn in self._tabs.items():
            btn.setStyleSheet(self._tab_style(active=(n == self._active_tab)))
        # Refresh all _RibbonBtn instances in every tab widget
        for tab_w in self._tab_widgets.values():
            for rb in tab_w.findChildren(_RibbonBtn):
                rb.refresh_theme()

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

def _get_layer_color(name: str) -> str:
    """Return the current theme-appropriate colour for a layer."""
    _map = {
        'Forces':        THEME.get('lyr_forces',    THEME['accent2']),
        'Stress':        THEME.get('lyr_stress',    '#e05050'),
        'B Axis':        THEME.get('lyr_baxis',     '#80d8ff'),
        'Field Lines':   THEME.get('lyr_fieldlines','#80ffff'),
        'Cross Section': THEME.get('lyr_xsection',  '#ff9800'),
    }
    return _map.get(name, THEME['text_dim'])

def _get_coil_colors() -> list:
    """Return the current theme-appropriate coil colour cycle."""
    return THEME.get('coil_colors', [THEME['accent']])


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
    Fusion 360-style browser.

    Tree structure
    --------------
    COILS (group)
      └── Coil_N  [eye | swatch | name (dbl-click→rename) | × delete]
            ├── Forces        [eye | swatch | name]
            ├── Stress
            ├── B Axis
            ├── Field Lines
            └── Cross Section

    Analysis layers are children of their coil — deleting the coil removes them.

    Signals
    -------
    layer_toggled(name, visible)          — analysis layer visibility
    coil_visibility_toggled(id, visible)  — coil show/hide
    coil_delete_requested(id)             — × button clicked
    coil_selected(id)                     — coil row clicked
    coil_renamed(id, new_name)            — rename confirmed
    """
    layer_toggled            = pyqtSignal(str, str, bool)   # (coil_id, name, visible)
    coil_visibility_toggled  = pyqtSignal(str, bool)
    coil_delete_requested    = pyqtSignal(str)
    coil_selected            = pyqtSignal(str)
    coil_renamed             = pyqtSignal(str, str)
    coil_recolored           = pyqtSignal(str, str)         # (coil_id, hex_color)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._hdr = QLabel("  BROWSER")
        self._hdr.setFixedHeight(22)
        self._hdr.setStyleSheet(
            f"background:{THEME['bg']}; color:{THEME['text_dim']}; "
            f"font-size:7pt; letter-spacing:2px; "
            f"border-bottom:1px solid {THEME['border']};"
        )
        root.addWidget(self._hdr)

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
        self._tree.itemClicked.connect(self._on_item_clicked)
        root.addWidget(self._tree, stretch=1)

        # Single "COILS" top-level group
        self._coils_group = QTreeWidgetItem()
        self._coils_group.setFlags(Qt.ItemIsEnabled)
        self._tree.addTopLevelItem(self._coils_group)
        self._coils_group.setExpanded(True)

        grp_w = QWidget()
        gl = QHBoxLayout(grp_w)
        gl.setContentsMargins(2, 0, 2, 0)
        gl.setSpacing(4)

        self._coils_eye = _EyeBtn()
        self._coils_eye.toggled.connect(self._group_eye_toggled)
        gl.addWidget(self._coils_eye)

        self._grp_lbl = QLabel("COILS")
        self._grp_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt; "
            f"font-weight:600; letter-spacing:1px;"
        )
        gl.addWidget(self._grp_lbl, stretch=1)
        self._tree.setItemWidget(self._coils_group, 0, grp_w)
        self._coils_group.setSizeHint(0, QSize(0, 22))

        # coil_id → {'tree_item', 'name_label', 'analysis': {layer_name: (item, eye)}}
        self._coil_data: dict = {}

    # ── Public: coil items ────────────────────────────────────────────────────

    def add_coil_item(self, coil_id: str, display_name: str, color: str) -> None:
        if coil_id in self._coil_data:
            return

        item = QTreeWidgetItem(self._coils_group)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(3)

        eye = _EyeBtn()
        eye.toggled.connect(
            lambda checked, cid=coil_id: self.coil_visibility_toggled.emit(cid, checked)
        )
        lay.addWidget(eye)

        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background:{color}; border-radius:1px;")
        swatch.setCursor(Qt.PointingHandCursor)
        swatch.setToolTip("Click to change coil color")
        swatch.mousePressEvent = lambda _e, cid=coil_id: self._recolor_coil(cid)
        lay.addWidget(swatch)

        n_lbl = QLabel(display_name)
        n_lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        n_lbl.mouseDoubleClickEvent = lambda _e, cid=coil_id: self._rename_coil(cid)
        lay.addWidget(n_lbl, stretch=1)

        del_btn = QPushButton("×")
        del_btn.setFixedSize(16, 16)
        del_btn.setFlat(True)
        del_btn.setObjectName('coil_del')
        del_btn.clicked.connect(lambda _, cid=coil_id: self.coil_delete_requested.emit(cid))
        lay.addWidget(del_btn)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._coil_data[coil_id] = {
            'tree_item':  item,
            'name_label': n_lbl,
            'swatch':     swatch,
            'color':      color,
            'analysis':   {},
        }
        self._coils_group.setExpanded(True)
        item.setExpanded(True)

    def remove_coil_item(self, coil_id: str) -> None:
        """Remove coil and ALL its analysis children from the tree."""
        if coil_id not in self._coil_data:
            return
        item = self._coil_data.pop(coil_id)['tree_item']
        self._coils_group.removeChild(item)

    # ── Public: analysis layers (nested under a coil) ─────────────────────────

    def add_layer_to_coil(self, coil_id: str, layer_name: str,
                          visible: bool = True) -> None:
        """Add an analysis layer as a child of the given coil."""
        if coil_id not in self._coil_data:
            return
        entry = self._coil_data[coil_id]
        if layer_name in entry['analysis']:
            return
        color = _get_layer_color(layer_name)

        coil_item = entry['tree_item']
        child = QTreeWidgetItem(coil_item)
        child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 2, 0)
        lay.setSpacing(4)

        eye = _EyeBtn()
        eye.toggled.connect(
            lambda checked, cid=coil_id, n=layer_name: self.layer_toggled.emit(cid, n, checked)
        )
        lay.addWidget(eye)

        sw = QLabel()
        sw.setFixedSize(10, 10)
        sw.setObjectName('layer_swatch')
        sw.setStyleSheet(f"background:{color}; border-radius:1px;")
        lay.addWidget(sw)

        lbl = QLabel(layer_name)
        lbl.setObjectName('layer_label')
        lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        lay.addWidget(lbl, stretch=1)

        self._tree.setItemWidget(child, 0, w)
        child.setSizeHint(0, QSize(0, 22))
        entry['analysis'][layer_name] = (child, eye)
        coil_item.setExpanded(True)

        if not visible:
            eye.setChecked(False)  # fires toggled → workspace.set_layer_visible(False)

    def remove_layer_from_coil(self, coil_id: str, layer_name: str) -> None:
        if coil_id not in self._coil_data:
            return
        analysis = self._coil_data[coil_id]['analysis']
        if layer_name not in analysis:
            return
        child_item, _ = analysis.pop(layer_name)
        self._coil_data[coil_id]['tree_item'].removeChild(child_item)

    def remove_all_analysis_from_coil(self, coil_id: str) -> None:
        if coil_id not in self._coil_data:
            return
        entry = self._coil_data[coil_id]
        for child_item, _ in list(entry['analysis'].values()):
            entry['tree_item'].removeChild(child_item)
        entry['analysis'].clear()

    def mark_layer_stale(self, coil_id: str, layer_name: str, stale: bool) -> None:
        """Show ⚠ on a layer label when the coil was moved after analysis."""
        entry = self._coil_data.get(coil_id)
        if not entry:
            return
        data = entry['analysis'].get(layer_name)
        if not data:
            return
        child_item, _eye = data
        w = self._tree.itemWidget(child_item, 0)
        if w is None:
            return
        lbl = w.findChild(QLabel, 'layer_label')
        if lbl is None:
            return
        if stale:
            lbl.setText(f"{layer_name} ⚠")
            lbl.setStyleSheet(f"color:{THEME['warning']}; font-size:8pt;")
            lbl.setToolTip("Coil moved after analysis — results may be outdated")
        else:
            lbl.setText(layer_name)
            lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
            lbl.setToolTip("")

    # ── Private ───────────────────────────────────────────────────────────────

    def refresh_theme(self) -> None:
        """Re-apply explicit styles after a theme switch."""
        self._hdr.setStyleSheet(
            f"background:{THEME['bg']}; color:{THEME['text_dim']}; "
            f"font-size:7pt; letter-spacing:2px; "
            f"border-bottom:1px solid {THEME['border']};"
        )
        self._tree.setStyleSheet(f"""
            QTreeWidget {{
                background:{THEME['panel']};
                border:none; outline:none;
            }}
            QTreeWidget::item {{ height:22px; padding:0; }}
            QTreeWidget::item:selected {{ background:{THEME['hi_blue']}; }}
        """)
        self._grp_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt; "
            f"font-weight:600; letter-spacing:1px;"
        )
        # Refresh all eye buttons
        for eb in self._tree.findChildren(_EyeBtn):
            eb._refresh()
        # Refresh coil name labels, layer labels, and layer swatch colours
        for cid, data in self._coil_data.items():
            data['name_label'].setStyleSheet(
                f"color:{THEME['text']}; font-size:8pt;"
            )
            for lname, (child_item, _eye) in data['analysis'].items():
                w = self._tree.itemWidget(child_item, 0)
                if w is None:
                    continue
                lbl = w.findChild(QLabel, 'layer_label')
                if lbl:
                    lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
                sw = w.findChild(QLabel, 'layer_swatch')
                if sw:
                    sw.setStyleSheet(
                        f"background:{_get_layer_color(lname)}; border-radius:1px;"
                    )

    def _on_item_clicked(self, item, _col) -> None:
        for cid, data in self._coil_data.items():
            if data['tree_item'] is item:
                self.coil_selected.emit(cid)
                return
            # Also select the parent coil when clicking an analysis layer child
            for child_item, _eye in data['analysis'].values():
                if child_item is item:
                    self.coil_selected.emit(cid)
                    return

    def _rename_coil(self, coil_id: str) -> None:
        if coil_id not in self._coil_data:
            return
        current = self._coil_data[coil_id]['name_label'].text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Coil", "New name:", text=current
        )
        if ok and new_name.strip():
            new_name = new_name.strip()
            self._coil_data[coil_id]['name_label'].setText(new_name)
            self.coil_renamed.emit(coil_id, new_name)

    def get_layer_eye_state(self, coil_id: str, layer_name: str) -> bool:
        """Return current checked state of a layer's eye button."""
        entry = self._coil_data.get(coil_id)
        if not entry:
            return True
        data = entry['analysis'].get(layer_name)
        if not data:
            return True
        _child_item, eye = data
        return eye.isChecked()

    def set_layer_eye_locked(self, coil_id: str, layer_name: str,
                              locked: bool) -> None:
        """Lock (disable + uncheck) a layer's eye button."""
        entry = self._coil_data.get(coil_id)
        if not entry:
            return
        data = entry['analysis'].get(layer_name)
        if not data:
            return
        _child_item, eye = data
        if locked:
            eye.setChecked(False)  # uncheck → fires toggled → hides layer
            eye.setEnabled(False)

    def set_layer_eye_unlocked(self, coil_id: str, layer_name: str,
                                restore_checked: bool = True) -> None:
        """Unlock a layer's eye button and restore it to the given state."""
        entry = self._coil_data.get(coil_id)
        if not entry:
            return
        data = entry['analysis'].get(layer_name)
        if not data:
            return
        _child_item, eye = data
        eye.setEnabled(True)
        eye.setChecked(restore_checked)  # fires toggled → sets visibility to match

    def _recolor_coil(self, coil_id: str) -> None:
        if coil_id not in self._coil_data:
            return
        from PyQt5.QtGui import QColor
        entry = self._coil_data[coil_id]
        initial = QColor(entry['color'])
        color = QColorDialog.getColor(initial, self, "Coil Color")
        if color.isValid():
            hex_color = color.name()
            entry['color'] = hex_color
            entry['swatch'].setStyleSheet(f"background:{hex_color}; border-radius:1px;")
            self.coil_recolored.emit(coil_id, hex_color)

    def _group_eye_toggled(self, checked: bool) -> None:
        """Toggle visibility of all coils and their analysis layers."""
        for cid in self._coil_data:
            self.coil_visibility_toggled.emit(cid, checked)
            for layer_name in self._coil_data[cid]['analysis']:
                self.layer_toggled.emit(cid, layer_name, checked)


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

        self._hdr = QLabel("  PROPERTIES")
        self._hdr.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:7pt; letter-spacing:2px;"
        )
        lay.addWidget(self._hdr)

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

        form.addRow("Winds:",           self.spin_winds)
        form.addRow("Current (A):",     self.dspin_current)
        form.addRow("Tape Thick (µm):", self.dspin_thick)
        form.addRow("Tape Width (mm):", self.dspin_width)
        form.addRow("Axis Samples:",    self.spin_axis_pts)
        lay.addLayout(form)

        # INSPECT — field line seeds
        lay.addWidget(_hdivider())
        irow = QHBoxLayout()
        irow.setContentsMargins(0, 0, 0, 0)
        irow.setSpacing(6)
        ilbl = QLabel("Field Seeds:")
        ilbl.setObjectName('dim_label')
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
        slbl.setObjectName('dim_label')
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

        self._results_hdr = _section_lbl("RESULTS")
        sl.addWidget(self._results_hdr)
        self._sum_lbls:  dict = {}   # key → value QLabel
        self._sum_keys:  list = []   # key QLabels (for theme refresh)
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
            self._sum_keys.append(kl)
            row.addWidget(kl)
            row.addWidget(vl, stretch=1)
            sl.addLayout(row)

        lay.addWidget(self._sum_w)
        lay.addStretch()

    # ── Public ────────────────────────────────────────────────────────────────

    def refresh_theme(self) -> None:
        """Re-apply explicit styles after a theme switch."""
        self._hdr.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:7pt; letter-spacing:2px;"
        )
        self._results_hdr.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:7pt; letter-spacing:2px;"
        )
        for kl in self._sum_keys:
            kl.setStyleSheet(
                f"color:{THEME['text_dim']}; font-size:8pt; min-width:52px;"
            )
        for vl in self._sum_lbls.values():
            vl.setStyleSheet(
                f"color:{THEME['accent']}; font-size:8pt; font-weight:bold;"
            )

    def get_params(self) -> dict:
        return {
            'winds':     self.spin_winds.value(),
            'current':   self.dspin_current.value(),
            'thickness': self.dspin_thick.value(),
            'width':     self.dspin_width.value(),
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


class SettingsDialog(QDialog):
    """Application settings — theme toggle."""
    theme_changed = pyqtSignal(str)   # 'dark' | 'light'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(320, 160)
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        from gui.gui_utils import get_theme_name
        current = get_theme_name()

        # Theme toggle row
        row = QHBoxLayout()
        lbl = QLabel("Theme:")
        lbl.setStyleSheet(f"color:{THEME['text']}; font-size:9pt;")
        row.addWidget(lbl)

        self._btn_dark  = QPushButton("Dark")
        self._btn_light = QPushButton("Light")
        for btn in (self._btn_dark, self._btn_light):
            btn.setCheckable(True)
            btn.setFixedHeight(28)
        self._btn_dark.setChecked(current == 'dark')
        self._btn_light.setChecked(current == 'light')
        self._btn_dark.clicked.connect(lambda: self._set('dark'))
        self._btn_light.clicked.connect(lambda: self._set('light'))
        row.addWidget(self._btn_dark)
        row.addWidget(self._btn_light)
        lay.addLayout(row)
        lay.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignRight)

    def _set(self, name: str) -> None:
        self._btn_dark.setChecked(name == 'dark')
        self._btn_light.setChecked(name == 'light')
        self.theme_changed.emit(name)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CalcSX™ – v{__version__}")
        self.resize(1280, 800)

        self._coords   = None
        self._multi_env = MultiCoilEnvironment()
        self._a_thread = None
        self._a_worker         = None
        self._i_thread         = None
        self._i_worker         = None
        self._inspect_reporter = None

        # Multi-coil tracking
        self._coil_counter   = 0                # auto-increment for unique IDs
        self._coil_names:    dict = {}          # coil_id → display name
        self._coil_coords:   dict = {}          # coil_id → np.ndarray
        self._active_coil_id:    str | None = None
        self._analyzed_coil_id:  str | None = None   # coil that owns the in-progress analysis
        self._coil_engines:      dict       = {}      # coil_id → CoilAnalysis engine
        self._coil_params_map:   dict       = {}      # coil_id → {winds, current, thickness, width, axis_num}
        self._inspect_coil_id:   str | None = None    # coil being currently inspected
        self._pending_inspect:        str | None = None   # 'field_lines' | 'cross_section'
        self._analysis_auto_triggered: bool = False        # True when analysis kicked off by inspect
        self._global_fl_cache:        tuple | None = None  # (lines, B_mags) or None
        self._global_fl_dirty:        bool = True          # True when environment changed since last global FL
        self._pre_global_eye_state:   dict       = {}      # coil_id → bool (eye state before global mode)

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
        self.ribbon.reanalyze_all.connect(self._on_reanalyze_all)
        self.ribbon.compute_field_lines.connect(self._on_compute_field_lines)
        self.ribbon.compute_cross_section.connect(self._on_compute_cross_section)
        self.ribbon.global_field_toggled.connect(self._on_global_field_toggled)
        self.ribbon.translate_toggled.connect(self._on_translate_toggled)
        self.ribbon.rotate_toggled.connect(self._on_rotate_toggled)
        self.ribbon.reset_transform.connect(self._on_reset_transform)
        self.ribbon.show_help.connect(lambda: HelpDialog(self).exec_())
        self.browser.layer_toggled.connect(self.workspace.set_layer_visible)
        self.browser.coil_visibility_toggled.connect(self.workspace.set_coil_visible)
        self.browser.coil_delete_requested.connect(self._on_coil_delete)
        self.browser.coil_selected.connect(self._on_coil_selected)
        self.browser.coil_renamed.connect(self._on_coil_renamed)
        self.browser.coil_recolored.connect(self._on_coil_recolored)
        self.ribbon.normalize_forces_toggled.connect(self._on_normalize_forces_toggled)
        self.ribbon.open_settings.connect(self._on_open_settings)
        self.workspace.set_stale_callback(self._on_layers_stale)

        # Listen to per-coil parameter changes → mark active coil stale
        for w in (self.props.spin_winds, self.props.dspin_current,
                  self.props.dspin_thick, self.props.dspin_width,
                  self.props.spin_axis_pts):
            w.valueChanged.connect(self._on_coil_param_changed)

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
        self._coil_names[coil_id]  = base_name

        # Store per-coil parameters (snapshot from current spinbox values)
        self._coil_params_map[coil_id] = self.props.get_params()

        params = self._coil_params_map[coil_id]
        total_t = params['thickness'] * 1e-6 * params['winds']  # µm→m × winds
        tape_w  = params['width'] * 1e-3                        # mm→m
        self.workspace.add_coil(coords, coil_id, color=color,
                                 total_thickness=total_t, tape_width=tape_w)
        self.browser.add_coil_item(coil_id, base_name, color)

        # Register in superposition environment — marks existing coils stale
        self._multi_env.register_coil(
            coil_id, coords,
            winds=params['winds'], current=params['current'],
            thickness=params['thickness'], width=params['width'],
        )
        self._propagate_staleness()

        self.ribbon.set_inspect_enabled(True)   # enabled — analysis runs automatically if needed
        self.ribbon.set_construct_enabled(True)
        self.ribbon._btn_translate.setChecked(False)
        self.ribbon._btn_rotate.setChecked(False)
        self.props.show()

    def _on_coil_delete(self, coil_id: str) -> None:
        # Clear analysis layers and engine for this coil
        self.workspace.clear_analysis_layers(coil_id)
        self._coil_engines.pop(coil_id, None)
        self._multi_env.unregister_coil(coil_id)
        self._propagate_staleness()
        if coil_id == self._analyzed_coil_id:
            self._analyzed_coil_id = None
        # Remove coil from workspace and browser (browser removes nested analysis too)
        self.workspace.remove_coil(coil_id)
        self.browser.remove_coil_item(coil_id)
        self._coil_coords.pop(coil_id, None)
        self._coil_names.pop(coil_id, None)
        self._coil_params_map.pop(coil_id, None)
        if self._active_coil_id == coil_id:
            new_id = self.workspace._active_coil_id
            self._active_coil_id = new_id
            self._coords = self._coil_coords.get(new_id) if new_id else None
        if not self._coil_coords:
            self.ribbon.set_construct_enabled(False)
            self.ribbon.set_inspect_enabled(False)

    def _on_coil_selected(self, coil_id: str) -> None:
        # Save current coil's spinbox values before switching
        old_cid = self._active_coil_id
        if old_cid and old_cid in self._coil_params_map:
            self._coil_params_map[old_cid] = self.props.get_params()
        self._active_coil_id = coil_id
        self._coords = self._coil_coords.get(coil_id)
        # Load the new coil's params into spinboxes
        self._load_coil_params(coil_id)
        self.workspace.set_active_coil(coil_id)
        # If gizmo is active, seamlessly move it to the new coil
        if self.ribbon._btn_translate.isChecked():
            self.workspace.show_gizmo('T')
        elif self.ribbon._btn_rotate.isChecked():
            self.workspace.show_gizmo('R')

    def _on_coil_param_changed(self) -> None:
        """A spinbox value changed — mark the active coil's analysis stale
        and rebuild its tube mesh to reflect the new winding dimensions."""
        cid = self._active_coil_id
        if not cid:
            return
        # Update stored params
        self._coil_params_map[cid] = self.props.get_params()
        # Update superposition environment (marks all coils stale)
        p = self._coil_params_map[cid]
        self._multi_env.update_coil_params(
            cid, winds=p['winds'], current=p['current'],
            thickness=p['thickness'], width=p['width'],
        )
        self._propagate_staleness()
        # Rebuild tube mesh with new dimensions
        total_t = p['thickness'] * 1e-6 * p['winds']
        tape_w  = p['width'] * 1e-3
        self.workspace.update_coil_mesh(cid, total_t, tape_w)

    def _load_coil_params(self, coil_id: str) -> None:
        """Load per-coil parameters into PropertiesPanel spinboxes."""
        p = self._coil_params_map.get(coil_id)
        if not p:
            return
        # Block signals so programmatic updates don't trigger side effects
        for w in (self.props.spin_winds, self.props.dspin_current,
                  self.props.dspin_thick, self.props.dspin_width,
                  self.props.spin_axis_pts):
            w.blockSignals(True)
        self.props.spin_winds.setValue(p['winds'])
        self.props.dspin_current.setValue(p['current'])
        self.props.dspin_thick.setValue(p['thickness'])
        self.props.dspin_width.setValue(p['width'])
        self.props.spin_axis_pts.setValue(p['axis_num'])
        for w in (self.props.spin_winds, self.props.dspin_current,
                  self.props.dspin_thick, self.props.dspin_width,
                  self.props.spin_axis_pts):
            w.blockSignals(False)

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
        # Save latest spinbox values for the active coil
        cid_save = self._active_coil_id
        if cid_save:
            self._coil_params_map[cid_save] = self.props.get_params()

        self.ribbon.set_run_enabled(False)
        self.ribbon.set_inspect_enabled(False)

        # Clear only THIS coil's previous analysis — other coils keep theirs
        cid = self._active_coil_id
        self.workspace.clear_analysis_layers(cid)
        self.browser.remove_all_analysis_from_coil(cid)
        self._analyzed_coil_id = cid

        # Use world-space coords so physics sees the coil where it actually is
        _wc = self.workspace.get_transformed_coords(cid)
        world_coords = _wc if _wc is not None else self._coords

        params = self.props.get_params()
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
        self._a_thread = None
        self._a_worker = None

        cid = self._analyzed_coil_id
        self._coil_engines[cid] = engine
        self._multi_env.mark_fresh(cid)
        # Clear stale marks for this coil — its analysis is now current
        for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
            self.browser.mark_layer_stale(cid, nm, False)

        norm = self.ribbon._btn_normalize.isChecked()
        self.workspace.add_force_layer(engine, cid, normalized=norm)
        self.workspace.add_stress_layer(engine, cid)
        self.workspace.add_axis_layer(engine, cid)

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
        self.ribbon.set_run_enabled(True)
        self.ribbon.set_inspect_enabled(True)
        # If global field mode is active, keep per-coil field lines disabled
        if self.ribbon._btn_global_field.isChecked():
            self.ribbon._btn_field_lines.set_action_enabled(False)
        # Enable "Re-analyze All" when there are stale coils in the environment
        self.ribbon._btn_reanalyze.set_action_enabled(
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

    def _on_reanalyze_all(self) -> None:
        """Re-run analysis on all stale coils sequentially."""
        stale = self._multi_env.get_stale_coils()
        # Build a queue of coils to analyze; process one at a time via chaining
        self._reanalyze_queue = [
            cid for cid in stale if cid in self._coil_coords
        ]
        self._reanalyze_next()

    def _reanalyze_next(self) -> None:
        """Pop the next stale coil and run its analysis."""
        if not getattr(self, '_reanalyze_queue', None):
            self.ribbon._btn_reanalyze.set_action_enabled(False)
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
            self.ribbon._btn_field_lines.set_action_enabled(False)
            # Use cached result if available and environment hasn't changed
            if not self._global_fl_dirty and self._global_fl_cache is not None:
                lines, B_mags = self._global_fl_cache
                self.workspace.add_field_lines_layer(lines, B_mags, '__global__')
            else:
                self._compute_global_field_lines()
        else:
            # Remove global field lines layer
            self.workspace.clear_field_lines_layer('__global__')
            # Restore per-coil field lines to their pre-global eye state
            self.ribbon._btn_field_lines.set_action_enabled(True)
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
        self._global_fl_dirty = False
        self.workspace.add_field_lines_layer(lines, B_mags, '__global__')

    def _on_normalize_forces_toggled(self, checked: bool = False) -> None:
        cid = self._active_coil_id
        engine = self._coil_engines.get(cid)
        if engine is None:
            return
        self.workspace.add_force_layer(engine, cid, normalized=checked)
        self.workspace.reapply_coil_transform(cid)

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
        if cid:
            self.browser.add_layer_to_coil(cid, 'Field Lines')
            # If global field mode is active, hide per-coil lines immediately
            if self.ribbon._btn_global_field.isChecked():
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
            self.browser.add_layer_to_coil(cid, 'Cross Section')

    def _on_translate_toggled(self, checked: bool) -> None:
        if checked:
            self.workspace.show_gizmo('T')
        else:
            self.workspace.hide_gizmo()

    def _on_rotate_toggled(self, checked: bool) -> None:
        if checked:
            self.workspace.show_gizmo('R')
        else:
            self.workspace.hide_gizmo()

    def _on_reset_transform(self) -> None:
        # Uncheck both toggles (fires hide_gizmo via toggled signal if either was active).
        self.ribbon._btn_translate.setChecked(False)
        self.ribbon._btn_rotate.setChecked(False)
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
        """Called by workspace whenever the active coil is transformed."""
        cid = self._active_coil_id
        if cid and cid in self._coil_engines:
            for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                self.browser.mark_layer_stale(cid, nm, True)
        # Update superposition environment with new world-space coords
        if cid:
            wc = self.workspace.get_transformed_coords(cid)
            if wc is not None:
                self._multi_env.update_coil_coords(cid, wc)
            self._propagate_staleness()

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.theme_changed.connect(self._apply_theme)
        dlg.exec_()

    def _apply_theme(self, name: str) -> None:
        from gui.gui_utils import apply_theme_to_app
        apply_theme_to_app(name)
        # Refresh every panel that uses explicit setStyleSheet calls
        self.ribbon.refresh_theme()
        self.browser.refresh_theme()
        self.props.refresh_theme()
        # Update 3D viewport (background, floor, ViewCube)
        self.workspace.apply_theme()

    def _propagate_staleness(self) -> None:
        """Reflect MultiCoilEnvironment staleness into browser stale markers."""
        stale = self._multi_env.get_stale_coils()
        for stale_cid in stale:
            if stale_cid in self._coil_engines:
                for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                    self.browser.mark_layer_stale(stale_cid, nm, True)
        self.ribbon._btn_reanalyze.set_action_enabled(bool(stale))
        self._global_fl_dirty = True


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hdivider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.HLine)
    d.setFrameShadow(QFrame.Sunken)
    d.setObjectName('hdivider')
    return d


def _section_lbl(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName('section_lbl')
    return l
