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
    QCheckBox,
    QComboBox,
    QListWidget,
    QProgressDialog,
    QStackedWidget,
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QPixmap

from physics.physics_utils import CoilAnalysis
from physics.superposition import MultiCoilEnvironment
from gui.gui_utils import ProgressReporter, THEME, get_app_icon
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
                 n_grid=120, axis_num=200, B_ext=None, tape_normals=None):
        super().__init__()
        self.coords    = coords
        self.winds     = winds
        self.current   = current
        self.thickness = thickness
        self.width     = width
        self.n_grid    = int(n_grid)
        self.axis_num  = int(axis_num)
        self.B_ext     = B_ext
        self.tape_normals = tape_normals

    @pyqtSlot()
    def run(self):
        engine = CoilAnalysis(
            self.coords, self.winds, self.current,
            self.thickness, self.width,
            B_ext=self.B_ext,
            tape_normals=self.tape_normals,
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
    import_bobbin            = pyqtSignal()
    run_analysis             = pyqtSignal()
    reanalyze_all            = pyqtSignal()
    compute_field_lines      = pyqtSignal()
    compute_cross_section    = pyqtSignal()
    global_field_toggled     = pyqtSignal(bool)
    translate_toggled        = pyqtSignal(bool)
    rotate_toggled           = pyqtSignal(bool)
    reset_transform          = pyqtSignal()
    generate_coil            = pyqtSignal()
    show_help                = pyqtSignal()
    normalize_forces_toggled = pyqtSignal(bool)
    open_settings            = pyqtSignal()
    save_session             = pyqtSignal()
    load_session             = pyqtSignal()
    # Inspect — probe
    add_hall_probe           = pyqtSignal()

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
        self._btn_load       = _RibbonBtn("▲", "Load\nCoil")
        self._btn_bobbin     = _RibbonBtn("⬡", "Import\nBobbin")
        self._btn_run        = _RibbonBtn("▶", "Run\nAnalysis")
        self._btn_reanalyze  = _RibbonBtn("⟳", "Re-analyze\nAll", enabled=False)
        btn_help_sim         = _RibbonBtn("?", "Help")

        # SIMULATION groups
        self._sim_w = QWidget()
        sl = QHBoxLayout(self._sim_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(_ribbon_group("FILE",     [self._btn_load, self._btn_bobbin]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("ANALYSIS", [self._btn_run, self._btn_reanalyze]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("HELP",     [btn_help_sim]))
        sl.addStretch(1)

        # INSPECT tab
        self._btn_field_lines  = _RibbonBtn("∿", "Field\nLines",   enabled=False)
        self._btn_cross_sec    = _RibbonBtn("⊡", "Cross\nSection", enabled=False)
        self._btn_global_field = _RibbonBtn("⊛", "Global\nField",  enabled=False, checkable=True)
        self._btn_hall_probe   = _RibbonBtn("⊙", "Hall\nProbe")
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
        il.addWidget(_vbar())
        il.addWidget(_ribbon_group("MEASUREMENT", [
            self._btn_hall_probe,
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
        self._btn_generate = _RibbonBtn("+", "Generate\nCoil")
        cl.addWidget(_vbar())
        cl.addWidget(_ribbon_group("GENERATORS", [
            self._btn_generate,
        ]))
        cl.addStretch(1)
        self._construct_w.hide()

        # UTILITIES
        self._btn_normalize = _RibbonBtn("⇈", "Normalize\nForces", checkable=True)
        self._btn_save_ses = _RibbonBtn("⬇", "Save\nSession")
        self._btn_load_ses = _RibbonBtn("⬆", "Load\nSession")
        self._btn_settings  = _RibbonBtn("⚙", "Settings")
        self._btn_save_ses.clicked.connect(self.save_session)
        self._btn_load_ses.clicked.connect(self.load_session)
        self._util_w = QWidget()
        ul = QHBoxLayout(self._util_w)
        ul.setContentsMargins(8, 0, 0, 0)
        ul.setSpacing(4)
        ul.addWidget(_ribbon_group("FORCES", [self._btn_normalize]))
        ul.addWidget(_vbar())
        ul.addWidget(_ribbon_group("SESSION", [self._btn_save_ses, self._btn_load_ses]))
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
        self._btn_bobbin.clicked.connect(self.import_bobbin)
        self._btn_run.clicked.connect(self.run_analysis)
        self._btn_reanalyze.clicked.connect(self.reanalyze_all)
        self._btn_field_lines.clicked.connect(self.compute_field_lines)
        self._btn_cross_sec.clicked.connect(self.compute_cross_section)
        self._btn_global_field.toggled.connect(self.global_field_toggled)
        self._btn_translate.toggled.connect(self._on_translate_toggled)
        self._btn_rotate.toggled.connect(self._on_rotate_toggled)
        self._btn_reset_xfm.clicked.connect(self.reset_transform)
        self._btn_generate.clicked.connect(self.generate_coil)
        btn_help_sim.clicked.connect(self.show_help)
        self._btn_normalize.toggled.connect(self.normalize_forces_toggled)
        self._btn_settings.clicked.connect(self.open_settings)
        self._btn_hall_probe.clicked.connect(self.add_hall_probe)

    # ── Public ────────────────────────────────────────────────────────────────

    def set_run_enabled(self, on: bool) -> None:
        self._btn_run.set_action_enabled(on)

    def set_inspect_enabled(self, on: bool) -> None:
        self._btn_field_lines.set_action_enabled(on)
        self._btn_cross_sec.set_action_enabled(on)
        self._btn_global_field.set_action_enabled(on)
        # Hall probe button is always enabled (no coil dependency)

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
    probe_selected           = pyqtSignal(str)                # probe_id
    probe_delete_requested   = pyqtSignal(str)                # probe_id
    probe_recolored          = pyqtSignal(str, str)           # (probe_id, hex_color)

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

        # "INSTRUMENTS" top-level group (Hall Probe, etc.)
        self._instruments_group = QTreeWidgetItem()
        self._instruments_group.setFlags(Qt.ItemIsEnabled)
        self._tree.addTopLevelItem(self._instruments_group)
        self._instruments_group.setExpanded(True)

        ig_w = QWidget()
        ig_l = QHBoxLayout(ig_w)
        ig_l.setContentsMargins(2, 0, 2, 0)
        ig_l.setSpacing(4)
        self._inst_lbl = QLabel("INSTRUMENTS")
        self._inst_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt; "
            f"font-weight:600; letter-spacing:1px;"
        )
        ig_l.addWidget(self._inst_lbl, stretch=1)
        self._tree.setItemWidget(self._instruments_group, 0, ig_w)
        self._instruments_group.setSizeHint(0, QSize(0, 22))
        self._instruments_group.setHidden(True)  # hidden until instruments added

        # probe_id → {'tree_item', 'readout_items': {key: QLabel}}
        self._probe_data: dict = {}

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
        self._inst_lbl.setStyleSheet(
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

    # ── Public: instrument items ────────────────────────────────────────────────

    def add_probe_item(self, probe_id: str, display_name: str = "Hall Probe") -> None:
        """Add a Hall Probe entry under INSTRUMENTS."""
        if probe_id in self._probe_data:
            return
        self._instruments_group.setHidden(False)
        item = QTreeWidgetItem(self._instruments_group)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 2, 0)
        lay.setSpacing(3)

        eye = _EyeBtn()
        eye.toggled.connect(
            lambda checked, pid=probe_id: self.layer_toggled.emit(pid, 'Hall Probe', checked)
        )
        lay.addWidget(eye)

        sw = QLabel()
        sw.setFixedSize(10, 10)
        color = THEME.get('lyr_probe', '#e040fb')
        sw.setStyleSheet(f"background:{color}; border-radius:1px;")
        sw.setCursor(Qt.PointingHandCursor)
        sw.setToolTip("Click to change probe color")
        sw.mousePressEvent = lambda _e, pid=probe_id: self._recolor_probe(pid)
        lay.addWidget(sw)

        lbl = QLabel(display_name)
        lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        lay.addWidget(lbl, stretch=1)

        del_btn = QPushButton("×")
        del_btn.setFixedSize(16, 16)
        del_btn.setFlat(True)
        del_btn.setObjectName('coil_del')
        del_btn.clicked.connect(lambda _, pid=probe_id: self.probe_delete_requested.emit(pid))
        lay.addWidget(del_btn)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._instruments_group.setExpanded(True)
        item.setExpanded(True)

        # Readout child items
        readout_items = {}
        readout_color = THEME.get('probe_readout', THEME['accent'])
        for key in ("Bx", "By", "Bz", "|B|", "X", "Y", "Z"):
            child = QTreeWidgetItem(item)
            child.setFlags(Qt.ItemIsEnabled)
            cw = QWidget()
            cl = QHBoxLayout(cw)
            cl.setContentsMargins(18, 0, 4, 0)
            cl.setSpacing(4)
            kl = QLabel(f"{key}:")
            kl.setStyleSheet(
                f"color:{THEME['text_dim']}; font-size:7pt; min-width:20px;"
            )
            vl = QLabel("—")
            vl.setStyleSheet(
                f"color:{readout_color}; font-size:7pt; font-weight:bold;"
            )
            cl.addWidget(kl)
            cl.addWidget(vl, stretch=1)
            self._tree.setItemWidget(child, 0, cw)
            child.setSizeHint(0, QSize(0, 18))
            readout_items[key] = vl

        self._probe_data[probe_id] = {
            'tree_item': item,
            'readout_items': readout_items,
            'swatch': sw,
            'color': color,
        }

    def update_probe_readout(self, probe_id: str, Bx: float, By: float, Bz: float,
                             B_mag: float, px: float, py: float, pz: float) -> None:
        entry = self._probe_data.get(probe_id)
        if entry is None:
            return
        r = entry['readout_items']
        r["Bx"].setText(f"{Bx:.6f} T")
        r["By"].setText(f"{By:.6f} T")
        r["Bz"].setText(f"{Bz:.6f} T")
        r["|B|"].setText(f"{B_mag:.6f} T")
        r["X"].setText(f"{px:.4f} m")
        r["Y"].setText(f"{py:.4f} m")
        r["Z"].setText(f"{pz:.4f} m")

    def remove_probe_item(self, probe_id: str) -> None:
        entry = self._probe_data.pop(probe_id, None)
        if entry is None:
            return
        self._instruments_group.removeChild(entry['tree_item'])
        if self._instruments_group.childCount() == 0:
            self._instruments_group.setHidden(True)

    def _recolor_probe(self, probe_id: str) -> None:
        entry = self._probe_data.get(probe_id)
        if entry is None:
            return
        from PyQt5.QtGui import QColor
        initial = QColor(entry['color'])
        color = QColorDialog.getColor(initial, self, "Probe Color")
        if color.isValid():
            hex_color = color.name()
            entry['color'] = hex_color
            entry['swatch'].setStyleSheet(f"background:{hex_color}; border-radius:1px;")
            self.probe_recolored.emit(probe_id, hex_color)

    # ── Private ──────────────────────────────────────────────────────────────

    def _on_item_clicked(self, item, _col) -> None:
        # Check if a probe was clicked
        for pid, pdata in self._probe_data.items():
            if pdata['tree_item'] is item:
                self.probe_selected.emit(pid)
                return
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
        self.spin_field_seeds.setRange(8, 100)
        self.spin_field_seeds.setSingleStep(4)
        self.spin_field_seeds.setValue(24)
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
        for key in ("B cent.", "B axial", "Peak |B|", "Peak F", "Peak σ", "Arc len.",
                    "Induct.", "Energy"):
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
        # ── New metrics: inductance, Ic, quench ──
        try:
            if engine.self_inductance is not None:
                L = engine.self_inductance
                if L >= 1.0:
                    _set("Induct.", f"{L:.3f} H")
                elif L >= 1e-3:
                    _set("Induct.", f"{L*1e3:.3f} mH")
                else:
                    _set("Induct.", f"{L*1e6:.2f} µH")
        except Exception: pass
        try:
            if engine.stored_energy is not None:
                E = engine.stored_energy
                if E >= 1e6:
                    _set("Energy", f"{E/1e6:.2f} MJ")
                elif E >= 1e3:
                    _set("Energy", f"{E/1e3:.2f} kJ")
                else:
                    _set("Energy", f"{E:.2f} J")
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
    """Application settings — theme toggle + dev settings."""
    theme_changed              = pyqtSignal(str)    # 'dark' | 'light'
    export_vtk_requested       = pyqtSignal()
    export_web_layers_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(340, 220)
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

        # ── Dev Settings ──────────────────────────────────────────────
        self._chk_dev = QCheckBox("Dev Settings")
        self._chk_dev.setStyleSheet(f"color:{THEME['text']}; font-size:9pt;")
        lay.addWidget(self._chk_dev)

        self._dev_frame = QFrame()
        dev_lay = QVBoxLayout(self._dev_frame)
        dev_lay.setContentsMargins(16, 4, 0, 4)

        self._btn_export_vtk = QPushButton("Export VTK (ParaView)…")
        self._btn_export_vtk.setFixedHeight(28)
        self._btn_export_vtk.setToolTip(
            "Export coil geometry and analysis data as .vtp files for ParaView"
        )
        self._btn_export_vtk.clicked.connect(self.export_vtk_requested.emit)
        dev_lay.addWidget(self._btn_export_vtk)

        self._btn_export_web = QPushButton("Export Web Layers…")
        self._btn_export_web.setFixedHeight(28)
        self._btn_export_web.setToolTip(
            "Export dark + light mode web demo layers (glTF) for calc.sx"
        )
        self._btn_export_web.clicked.connect(self.export_web_layers_requested.emit)
        dev_lay.addWidget(self._btn_export_web)

        self._dev_frame.setVisible(False)
        lay.addWidget(self._dev_frame)
        self._chk_dev.toggled.connect(self._dev_frame.setVisible)

        lay.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignRight)

    def _set(self, name: str) -> None:
        self._btn_dark.setChecked(name == 'dark')
        self._btn_light.setChecked(name == 'light')
        self.theme_changed.emit(name)


class CoilGeneratorDialog(QDialog):
    """Parametric coil generator — produces (N, 3) coords from a shape recipe."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Coil")
        self.resize(360, 340)
        self._coords = None

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # Shape selector
        row = QHBoxLayout()
        row.addWidget(QLabel("Shape:"))
        self._combo = QComboBox()
        self._combo.addItems(["Solenoid", "Princeton Dee", "Saddle", "CCT"])
        self._combo.currentIndexChanged.connect(self._on_shape_changed)
        row.addWidget(self._combo, stretch=1)
        lay.addLayout(row)

        # Parameter form (rebuilt per shape)
        self._form_w = QWidget()
        self._form_lay = QFormLayout(self._form_w)
        self._form_lay.setLabelAlignment(Qt.AlignRight)
        self._form_lay.setSpacing(4)
        lay.addWidget(self._form_w)

        self._spins: dict = {}
        self._build_params("Solenoid")

        # Info label
        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt;"
        )
        lay.addWidget(self._info)
        lay.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_gen = QPushButton("Generate")
        btn_gen.clicked.connect(self._generate)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_gen)
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

    def get_coords(self):
        return self._coords

    def _on_shape_changed(self, _idx):
        self._build_params(self._combo.currentText())

    def _build_params(self, shape: str):
        # Clear existing
        while self._form_lay.rowCount() > 0:
            self._form_lay.removeRow(0)
        self._spins.clear()

        def _dspin(label, val, lo, hi, dec=3, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
            if suffix:
                s.setSuffix(suffix)
            self._form_lay.addRow(label, s)
            self._spins[label] = s

        def _ispin(label, val, lo, hi, step=1):
            s = QSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step); s.setValue(val)
            self._form_lay.addRow(label, s)
            self._spins[label] = s

        if shape == "Solenoid":
            _dspin("Radius (m):", 0.05, 0.001, 50.0, suffix=" m")
            _dspin("Pitch (m):", 0.003, 0.0001, 1.0, suffix=" m")
            _ispin("Turns:", 6, 1, 10000)
            _ispin("Pts/turn:", 60, 20, 1000, step=20)
        elif shape == "Princeton Dee":
            _dspin("R inner (m):", 0.025, 0.001, 10.0, suffix=" m")
            _dspin("R outer (m):", 0.078, 0.001, 20.0, suffix=" m")
            _dspin("Height (m):", 0.132, 0.001, 20.0, suffix=" m")
            _dspin("Corner R (m):", 0.0, 0.0, 5.0, suffix=" m")
            _ispin("Points:", 300, 100, 5000, step=100)
        elif shape == "Saddle":
            _dspin("Radius (m):", 0.5, 0.001, 10.0, suffix=" m")
            _dspin("Length (m):", 1.0, 0.01, 20.0, suffix=" m")
            _dspin("Angle span:", 120.0, 10.0, 350.0, dec=1, suffix=" deg")
            _ispin("Points:", 400, 100, 5000, step=100)
        elif shape == "CCT":
            _dspin("Radius (m):", 0.3, 0.001, 10.0, suffix=" m")
            _dspin("Pitch (m):", 0.02, 0.001, 1.0, suffix=" m")
            _ispin("Turns:", 20, 1, 10000)
            _dspin("Tilt angle:", 30.0, 1.0, 80.0, dec=1, suffix=" deg")
            _ispin("Pts/turn:", 100, 20, 1000, step=20)

    def _generate(self):
        from physics.geometry import (
            generate_solenoid, generate_princeton_dee,
            generate_saddle_coil, generate_cct,
        )
        shape = self._combo.currentText()
        s = self._spins
        try:
            if shape == "Solenoid":
                self._coords = generate_solenoid(
                    radius=s["Radius (m):"].value(),
                    pitch=s["Pitch (m):"].value(),
                    n_turns=s["Turns:"].value(),
                    n_pts_per_turn=s["Pts/turn:"].value(),
                )
            elif shape == "Princeton Dee":
                self._coords = generate_princeton_dee(
                    R_inner=s["R inner (m):"].value(),
                    R_outer=s["R outer (m):"].value(),
                    height=s["Height (m):"].value(),
                    corner_radius=s["Corner R (m):"].value(),
                    n_pts=s["Points:"].value(),
                )
            elif shape == "Saddle":
                self._coords = generate_saddle_coil(
                    radius=s["Radius (m):"].value(),
                    length=s["Length (m):"].value(),
                    angle_span=s["Angle span:"].value(),
                    n_pts=s["Points:"].value(),
                )
            elif shape == "CCT":
                self._coords = generate_cct(
                    radius=s["Radius (m):"].value(),
                    pitch=s["Pitch (m):"].value(),
                    n_turns=s["Turns:"].value(),
                    tilt_angle=s["Tilt angle:"].value(),
                    n_pts_per_turn=s["Pts/turn:"].value(),
                )
            n = len(self._coords) if self._coords is not None else 0
            self._info.setText(f"Generated {shape}: {n} points")
            if self._coords is not None:
                self.accept()
        except Exception as exc:
            self._info.setText(f"Error: {exc}")
            self._info.setStyleSheet(f"color:#ff4444; font-size:8pt;")


class MeshChannelPickerDialog(QDialog):
    """
    Interactive channel picker for STL/OBJ mesh bobbins.

    Modes
    -----
    - **Auto-Detect**: curvature-based groove detection — click to select
    - **Manual Pick**: right-click surface points, geodesic path tracing
    """

    def __init__(self, mesh, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Channel on Mesh")
        self.resize(850, 650)
        self._mesh = mesh
        self._result_coords = None
        self._result_normals = None

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except ImportError:
            QMessageBox.warning(self, "Error", "PyVista required")
            return

        # Clean mesh and compute normals
        self._nmesh = mesh.clean().compute_normals(
            cell_normals=True, point_normals=True, inplace=False,
        )

        # State
        self._mode = None          # 'auto' | 'manual'
        self._picked_pts = []      # snapped (x,y,z)
        self._picked_vtx = []      # vertex indices
        self._sphere_actors = []
        self._line_actors = []
        self._channel_meshes = []  # auto-detect sub-meshes
        self._channel_actors = []
        self._selected_ch = -1
        self._vtx_tree = None

        # ── Layout ────────────────────────────────────────────
        lay = QVBoxLayout(self)

        # Mode buttons + sensitivity
        top = QHBoxLayout()
        self._btn_auto = QPushButton("Auto-Detect Channels")
        self._btn_auto.clicked.connect(self._run_auto_detect)
        self._btn_manual = QPushButton("Manual Pick (Geodesic)")
        self._btn_manual.clicked.connect(self._switch_to_manual)
        top.addWidget(self._btn_auto)
        top.addWidget(self._btn_manual)
        lbl_a = QLabel("Angle:")
        lbl_a.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8pt;")
        top.addWidget(lbl_a)
        self._spin_angle = QSpinBox()
        self._spin_angle.setRange(10, 75)
        self._spin_angle.setValue(30)
        self._spin_angle.setSuffix("°")
        self._spin_angle.setToolTip(
            "Feature angle threshold — faces whose normals differ\n"
            "by more than this are considered separate patches.\n"
            "Lower = more patches (finer segmentation).\n"
            "Higher = fewer patches (coarser).")
        top.addWidget(self._spin_angle)
        top.addStretch()
        lay.addLayout(top)

        # Channel list (auto mode) + plotter
        mid = QHBoxLayout()
        self._ch_list = QListWidget()
        self._ch_list.setMaximumWidth(180)
        self._ch_list.currentRowChanged.connect(self._on_channel_row_changed)
        self._ch_list.hide()
        mid.addWidget(self._ch_list)

        self._plotter = QtInteractor(self)
        self._plotter.set_background(THEME['bg'])
        mid.addWidget(self._plotter, stretch=1)
        lay.addLayout(mid, stretch=1)

        self._plotter.add_mesh(
            self._nmesh, color='#cccccc', opacity=1.0,
            show_edges=True, edge_color='#555555',
        )

        self._info = QLabel("Choose Auto-Detect or Manual Pick to begin.")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(f"color:{THEME['text']}; font-size:9pt;")
        lay.addWidget(self._info)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{THEME['accent']}; font-size:8pt;")
        lay.addWidget(self._status)

        # Turns
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Turns:"))
        self.spin_winds = QSpinBox()
        self.spin_winds.setRange(1, 10000)
        self.spin_winds.setValue(200)
        wrow.addWidget(self.spin_winds, stretch=1)
        lay.addLayout(wrow)

        # Buttons
        brow = QHBoxLayout()
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.clicked.connect(self._undo)
        self._btn_undo.setEnabled(False)
        btn_done = QPushButton("Done")
        btn_done.clicked.connect(self._finish)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        brow.addWidget(self._btn_undo)
        brow.addStretch()
        brow.addWidget(btn_done)
        brow.addWidget(btn_cancel)
        lay.addLayout(brow)

    # ── Auto-detect ───────────────────────────────────────────────────────

    def _run_auto_detect(self):
        self._mode = 'auto'
        self._btn_undo.setEnabled(False)
        self._ch_list.show()
        self._ch_list.clear()
        self._clear_all()

        angle_deg = self._spin_angle.value()
        self._status.setText("Segmenting mesh …")
        QApplication.processEvents()

        try:
            labels, n_patches = self._segment_mesh(
                self._nmesh, angle_deg=angle_deg)
        except Exception as exc:
            self._status.setText(f"Segmentation failed: {exc}")
            return

        # ── Classify patches as channel candidates ────────────
        centers = np.asarray(
            self._nmesh.cell_centers().points, dtype=np.float64)
        mesh_diag = max(float(self._nmesh.length), 1e-10)

        # Score each patch: elongated + narrow → likely channel
        scored = []
        for pid in range(n_patches):
            mask = labels == pid
            nc = int(mask.sum())
            if nc < 8:
                continue
            pc = centers[mask]
            centered = pc - pc.mean(axis=0)
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            length = S[0]
            width = max(S[1], 1e-10)
            elongation = length / width
            rel_width = width / mesh_diag

            # Channel candidate: elongated strip that isn't the entire
            # outer surface.  Wall patches are also elongated but very
            # thin; keep them — user can distinguish visually.
            if elongation > 2.5 and nc < labels.size * 0.4:
                scored.append({
                    'pid': pid, 'mask': mask, 'nc': nc,
                    'elongation': elongation, 'rel_width': rel_width,
                    'length': length, 'width': width,
                })

        # Sort: high elongation first
        scored.sort(key=lambda s: s['elongation'], reverse=True)
        # Cap to avoid overwhelming the list
        scored = scored[:30]

        if not scored:
            self._status.setText(
                "No elongated patches found. Adjust Angle or use Manual.")
            return

        colors = ['#ff4444', '#44ff44', '#4488ff', '#ffff00',
                  '#ff44ff', '#44ffff', '#ff8844', '#88ff44',
                  '#ff6688', '#66ff88', '#6688ff', '#ffaa44']
        self._channel_meshes = []
        self._channel_actors = []

        for i, s in enumerate(scored):
            sub = self._nmesh.extract_cells(np.where(s['mask'])[0])
            c = colors[i % len(colors)]
            a = self._plotter.add_mesh(
                sub, color=c, opacity=0.8, reset_camera=False)
            self._channel_meshes.append(sub)
            self._channel_actors.append(a)
            self._ch_list.addItem(
                f"Patch {i+1}  ({s['nc']} faces, "
                f"elong {s['elongation']:.1f}×)")

        self._info.setText(
            "Select a channel from the list, then press Done.\n"
            "Adjust Angle and re-run Auto-Detect if needed.")
        self._status.setText(
            f"Found {len(scored)} candidate patch(es) "
            f"at {angle_deg}° threshold.")

    def _on_channel_row_changed(self, row):
        self._selected_ch = row
        for i, a in enumerate(self._channel_actors):
            try:
                a.GetProperty().SetOpacity(1.0 if i == row else 0.35)
            except Exception:
                pass
        self._plotter.render()

    # ── Manual mode ───────────────────────────────────────────────────────

    def _switch_to_manual(self):
        self._mode = 'manual'
        self._btn_undo.setEnabled(True)
        self._ch_list.hide()
        self._clear_all()

        self._info.setText(
            "Right-click (two-finger) to place points along the channel.\n"
            "Geodesic paths are traced between consecutive points.\n"
            "Place at least 2 points, then press Done."
        )

        from scipy.spatial import cKDTree
        self._vtx_tree = cKDTree(np.asarray(self._nmesh.points))

        # Surface-aware picking — only visible face, no through-mesh picks
        try:
            self._plotter.enable_surface_point_picking(
                callback=self._on_pick, show_message=False,
                left_clicking=False, show_point=False, tolerance=0.025,
            )
        except AttributeError:
            # Older PyVista fallback
            self._plotter.enable_point_picking(
                callback=self._on_pick, show_message=False,
                left_clicking=False, show_point=False, tolerance=0.05,
            )

    def _on_pick(self, *args):
        if not args:
            return
        point = args[0]
        # enable_surface_point_picking may return mesh instead of point
        if hasattr(point, 'points'):
            if point.n_points == 0:
                return
            point = point.points[0]
        if point is None:
            return
        pt = np.asarray(point, dtype=np.float64).ravel()
        if len(pt) < 3:
            return
        pt = pt[:3]
        if np.allclose(pt, 0.0) and not self._picked_pts:
            return

        # Snap to nearest mesh vertex
        _, vid = self._vtx_tree.query(pt)
        snap = np.asarray(self._nmesh.points[vid], dtype=np.float64)

        self._picked_pts.append(snap.copy())
        self._picked_vtx.append(int(vid))

        # Marker sphere
        import pyvista as pv
        r = float(self._mesh.length) * 0.006
        sphere = pv.Sphere(radius=r, center=snap)
        a_s = self._plotter.add_mesh(
            sphere, color='#ff4444', reset_camera=False)
        self._sphere_actors.append(a_s)

        # Geodesic line to previous pick
        if len(self._picked_vtx) >= 2:
            v0, v1 = self._picked_vtx[-2], self._picked_vtx[-1]
            try:
                lm = self._nmesh.geodesic(v0, v1)
            except Exception:
                lm = pv.Line(self._picked_pts[-2], snap)
            a_l = self._plotter.add_mesh(
                lm, color='#ffff00', line_width=3.0, reset_camera=False,
            )
            self._line_actors.append(a_l)

        self._status.setText(f"{len(self._picked_pts)} points placed")

    def _undo(self):
        if not self._picked_pts:
            return
        self._picked_pts.pop()
        self._picked_vtx.pop()
        # Remove the sphere for this point
        if self._sphere_actors:
            try:
                self._plotter.remove_actor(self._sphere_actors.pop())
            except Exception:
                pass
        # Remove the connecting line (N points → N-1 lines)
        if self._line_actors:
            try:
                self._plotter.remove_actor(self._line_actors.pop())
            except Exception:
                pass
        self._status.setText(
            f"{len(self._picked_pts)} points placed"
            if self._picked_pts else "0 points placed"
        )

    # ── Finish ────────────────────────────────────────────────────────────

    def _finish(self):
        if self._mode == 'auto':
            self._finish_auto()
        elif self._mode == 'manual':
            self._finish_manual()
        else:
            self._status.setText("Choose Auto-Detect or Manual Pick first.")

    def _finish_auto(self):
        if (self._selected_ch < 0
                or self._selected_ch >= len(self._channel_meshes)):
            self._status.setText("Select a channel from the list first.")
            return

        ch = self._channel_meshes[self._selected_ch]
        pts = np.asarray(ch.points, dtype=np.float64)
        norms = ch.point_data.get('Normals')
        if norms is None:
            ch2 = ch.compute_normals(
                point_normals=True, cell_normals=False, inplace=False)
            norms = ch2.point_data.get('Normals')
        norms = (np.asarray(norms, dtype=np.float64)
                 if norms is not None else None)

        cl, cl_n = self._extract_centerline(pts, norms)
        self._result_coords = cl
        self._result_normals = cl_n
        self._plotter.close()
        self.accept()

    def _finish_manual(self):
        if len(self._picked_vtx) < 2:
            self._status.setText("Need at least 2 points.")
            return

        # Trace geodesic between consecutive picks
        from scipy.spatial import cKDTree
        if self._vtx_tree is None:
            self._vtx_tree = cKDTree(np.asarray(self._nmesh.points))

        all_ids = []
        for i in range(len(self._picked_vtx) - 1):
            v0, v1 = self._picked_vtx[i], self._picked_vtx[i + 1]
            try:
                geo = self._nmesh.geodesic(v0, v1)
                gp = np.asarray(geo.points, dtype=np.float64)
                _, ids = self._vtx_tree.query(gp)
                all_ids.extend(ids.tolist())
            except Exception:
                all_ids.append(v0)
        all_ids.append(self._picked_vtx[-1])

        # Deduplicate preserving order
        seen = set()
        unique = []
        for v in all_ids:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        unique = np.array(unique)

        self._result_coords = np.asarray(
            self._nmesh.points[unique], dtype=np.float64)
        na = self._nmesh.point_data.get('Normals')
        self._result_normals = (
            np.asarray(na[unique], dtype=np.float64)
            if na is not None else None
        )
        self._plotter.close()
        self.accept()

    # ── Centerline extraction (auto-detect) ───────────────────────────────

    @staticmethod
    def _extract_centerline(pts, normals, n_target=500):
        """
        Order groove vertices into a smooth centerline via PCA-based
        binning.  Detects ring-like vs strip-like groove shapes and
        bins by angle or projection accordingly.
        """
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)

        ratio = S[1] / max(S[0], 1e-10)
        is_ring = ratio > 0.3

        if is_ring:
            proj2 = centered @ Vt[:2].T
            param = np.arctan2(proj2[:, 1], proj2[:, 0])
            lo, hi = -np.pi, np.pi
        else:
            param = centered @ Vt[0]
            lo, hi = float(param.min()), float(param.max())

        n_bins = min(n_target, max(60, len(pts) // 5))
        edges = np.linspace(lo, hi, n_bins + 1)

        cl_p, cl_n = [], []
        for i in range(n_bins):
            mask = (param >= edges[i]) & (param < edges[i + 1])
            if not mask.any():
                continue
            cl_p.append(pts[mask].mean(axis=0))
            if normals is not None:
                nm = normals[mask].mean(axis=0)
                nm /= max(np.linalg.norm(nm), 1e-10)
                cl_n.append(nm)

        cl_p = np.array(cl_p, dtype=np.float64)
        cl_n = (np.array(cl_n, dtype=np.float64)
                if normals is not None else None)

        # Smooth with moving average
        if len(cl_p) > 10:
            try:
                from scipy.ndimage import uniform_filter1d
                k = 5
                mode = 'wrap' if is_ring else 'nearest'
                for ax in range(3):
                    cl_p[:, ax] = uniform_filter1d(
                        cl_p[:, ax], k, mode=mode)
                if cl_n is not None:
                    for ax in range(3):
                        cl_n[:, ax] = uniform_filter1d(
                            cl_n[:, ax], k, mode=mode)
                    cl_n /= np.maximum(
                        np.linalg.norm(cl_n, axis=1, keepdims=True), 1e-10)
            except ImportError:
                pass

        return cl_p, cl_n

    # ── Mesh segmentation ────────────────────────────────────────────────

    @staticmethod
    def _segment_mesh(mesh, angle_deg=30.0):
        """
        Segment a triangle mesh into smooth patches separated by sharp
        edges (dihedral angle > *angle_deg*).

        Algorithm
        ---------
        1. Build face-adjacency from shared edges.
        2. BFS flood-fill: grow a patch to adjacent faces whose normals
           differ by less than the threshold.  Sharp transitions act as
           natural barriers, so channel walls/floors/lands become separate
           patches automatically.

        Returns ``(labels, n_patches)`` where *labels* is an int32 array
        of length ``mesh.n_cells``.
        """
        from collections import defaultdict

        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        n_cells = mesh.n_cells
        cos_thresh = np.cos(np.radians(angle_deg))

        # ── Build face adjacency from edge topology ───────────
        faces_flat = np.asarray(mesh.faces)
        edge_to_cells = defaultdict(list)
        idx = 0
        ci = 0
        while idx < len(faces_flat):
            nv = int(faces_flat[idx])
            verts = faces_flat[idx + 1: idx + 1 + nv]
            for j in range(nv):
                a = int(verts[j])
                b = int(verts[(j + 1) % nv])
                edge_to_cells[(min(a, b), max(a, b))].append(ci)
            idx += nv + 1
            ci += 1

        # Adjacency list (only across shared edges, not vertices)
        adj = [[] for _ in range(n_cells)]
        for cids in edge_to_cells.values():
            if len(cids) == 2:
                adj[cids[0]].append(cids[1])
                adj[cids[1]].append(cids[0])

        # ── BFS flood fill ────────────────────────────────────
        labels = -np.ones(n_cells, dtype=np.int32)
        current_label = 0
        for start in range(n_cells):
            if labels[start] >= 0:
                continue
            queue = [start]
            labels[start] = current_label
            head = 0
            while head < len(queue):
                face = queue[head]
                head += 1
                fn = face_normals[face]
                for nb in adj[face]:
                    if labels[nb] >= 0:
                        continue
                    if np.dot(fn, face_normals[nb]) >= cos_thresh:
                        labels[nb] = current_label
                        queue.append(nb)
            current_label += 1

        return labels, current_label

    # ── Helpers ───────────────────────────────────────────────────────────

    def _clear_all(self):
        for a in (self._channel_actors
                  + self._sphere_actors + self._line_actors):
            try:
                self._plotter.remove_actor(a)
            except Exception:
                pass
        self._channel_meshes = []
        self._channel_actors = []
        self._sphere_actors = []
        self._line_actors = []
        self._picked_pts = []
        self._picked_vtx = []
        self._selected_ch = -1

    def get_coords(self) -> np.ndarray | None:
        return self._result_coords

    def get_normals(self) -> np.ndarray | None:
        return self._result_normals

    def get_winds(self) -> int:
        return self.spin_winds.value()

    def closeEvent(self, event):
        try:
            self._plotter.close()
        except Exception:
            pass
        super().closeEvent(event)


class BobbinChannelDialog(QDialog):
    """Legacy — kept for STEP import fallback. Not used in normal flow."""

    _CHAN_COLORS = [
        '#ff4444', '#44ff44', '#4488ff', '#ffff00',
        '#ff44ff', '#44ffff', '#ff8844', '#88ff44',
    ]

    def __init__(self, importer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bobbin Channel Wizard")
        self.resize(1100, 720)
        self._imp = importer

        import pyvista as pv
        from pyvistaqt import QtInteractor

        self._mesh = None
        self._highlight_actor = None
        self._selected_faces: set[int] = set()

        # Stage 1 collected items: list of {face_indices, winds, actor}
        self._collected: list[dict] = []

        # Final output
        self._result: list[dict] = []

        # Follow-channel mode state (stage 2)
        self._follow_mode = False
        self._follow_combined_row: int | None = None

        # Build adjacency for region growing
        self._imp.build_adjacency()

        # ── Layout ────────────────────────────────────────────────────
        lay = QVBoxLayout(self)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(f"color:{THEME['text']}; font-size:9pt;")
        lay.addWidget(self._info)

        mid = QHBoxLayout()

        # Left panel — stacked widget for stages
        self._left_stack = QStackedWidget()
        self._left_stack.setMaximumWidth(300)

        self._build_stage1_panel()
        self._build_stage2_panel()
        self._build_stage3_panel()

        mid.addWidget(self._left_stack)

        # 3D plotter (shared)
        self._plotter = QtInteractor(self)
        self._plotter.set_background(THEME['bg'])
        mid.addWidget(self._plotter, stretch=1)
        lay.addLayout(mid, stretch=1)

        # Status bar
        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color:{THEME['accent']}; font-size:8pt;")
        lay.addWidget(self._status)

        # Init mesh + enter stage 1
        self._load_mesh()
        self._enter_stage(1)

        # Right-click picking
        try:
            self._plotter.enable_surface_point_picking(
                callback=self._on_pick, show_message=False,
                left_clicking=False, show_point=False,
            )
        except AttributeError:
            self._plotter.enable_point_picking(
                callback=self._on_pick, show_message=False,
                left_clicking=False, show_point=False,
            )

    # ── Panel builders ───────────────────────────────────────────────

    def _build_stage1_panel(self):
        w = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(4, 4, 4, 4)

        ll.addWidget(QLabel("Stage 1 — Collect Faces"))
        self._s1_list = QListWidget()
        ll.addWidget(self._s1_list, stretch=1)

        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Turns:"))
        self._s1_winds = QSpinBox()
        self._s1_winds.setRange(1, 10000)
        self._s1_winds.setValue(200)
        wrow.addWidget(self._s1_winds, stretch=1)
        ll.addLayout(wrow)

        btn_add = QPushButton("Add Selected Face(s)")
        btn_add.clicked.connect(self._s1_add)
        btn_rm = QPushButton("Remove")
        btn_rm.clicked.connect(self._s1_remove)
        ll.addWidget(btn_add)
        ll.addWidget(btn_rm)

        ll.addStretch()
        btn_next = QPushButton("Next →  Refine / Split")
        btn_next.clicked.connect(lambda: self._enter_stage(2))
        ll.addWidget(btn_next)

        self._left_stack.addWidget(w)

    def _build_stage2_panel(self):
        w = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(4, 4, 4, 4)

        ll.addWidget(QLabel("Stage 2 — Refine / Split"))
        self._s2_list = QListWidget()
        ll.addWidget(self._s2_list, stretch=1)

        # Equal split controls
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Split into:"))
        self._s2_nparts = QSpinBox()
        self._s2_nparts.setRange(2, 20)
        self._s2_nparts.setValue(4)
        nrow.addWidget(self._s2_nparts, stretch=1)
        ll.addLayout(nrow)

        btn_split = QPushButton("Split Equal")
        btn_split.clicked.connect(self._s2_split)
        ll.addWidget(btn_split)

        # Follow channel controls
        self._btn_follow = QPushButton("Follow Channel")
        self._btn_follow.setToolTip(
            "Select a combined face in the list, then click\n"
            "this button and pick a properly isolated channel\n"
            "as reference.  The groove width and alignment\n"
            "are matched automatically.")
        self._btn_follow.clicked.connect(self._s2_enter_follow)
        ll.addWidget(self._btn_follow)

        self._btn_cancel_follow = QPushButton("Cancel Follow")
        self._btn_cancel_follow.clicked.connect(self._s2_cancel_follow)
        self._btn_cancel_follow.hide()
        ll.addWidget(self._btn_cancel_follow)

        btn_del = QPushButton("Delete Selected")
        btn_del.clicked.connect(self._s2_delete)
        ll.addWidget(btn_del)

        ll.addStretch()
        brow = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.clicked.connect(lambda: self._enter_stage(1))
        btn_next = QPushButton("Next →  Extract")
        btn_next.clicked.connect(lambda: self._enter_stage(3))
        brow.addWidget(btn_back)
        brow.addWidget(btn_next)
        ll.addLayout(brow)

        self._left_stack.addWidget(w)

    def _build_stage3_panel(self):
        w = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(4, 4, 4, 4)

        ll.addWidget(QLabel("Stage 3 — Review Channels"))
        self._s3_list = QListWidget()
        ll.addWidget(self._s3_list, stretch=1)

        btn_del = QPushButton("Remove Selected")
        btn_del.clicked.connect(self._s3_remove)
        ll.addWidget(btn_del)

        ll.addStretch()
        brow = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.clicked.connect(lambda: self._enter_stage(2))
        btn_done = QPushButton("Done")
        btn_done.clicked.connect(self._finish)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        brow.addWidget(btn_back)
        brow.addWidget(btn_done)
        brow.addWidget(btn_cancel)
        ll.addLayout(brow)

        self._left_stack.addWidget(w)

    # ── Stage transitions ────────────────────────────────────────────

    def _enter_stage(self, stage: int):
        self._left_stack.setCurrentIndex(stage - 1)
        self._selected_faces.clear()
        self._clear_highlight()

        if stage == 1:
            self._info.setText(
                "Right-click a groove face to select it.\n"
                "Shift-click to add adjacent faces.  "
                "Region-growing auto-extends along the groove.")
            self._render_bobbin()
            self._render_collected_tubes()
            self._status.setText("Select groove faces, then Add.")

        elif stage == 2:
            self._follow_mode = False
            self._btn_cancel_follow.hide()
            self._info.setText(
                "Select an item and Split or Follow Channel.\n"
                "Delete unwanted segments.")
            self._s2_populate()
            self._render_collected_only()
            self._status.setText(
                "Select a combined face to split, or Next.")

        elif stage == 3:
            self._info.setText(
                "Extracting centerlines and normals…")
            self._s3_extract()

    # ── Mesh rendering ───────────────────────────────────────────────

    def _load_mesh(self):
        self._mesh = self._imp.get_face_tagged_mesh()
        if self._mesh is None or self._mesh.n_cells == 0:
            self._mesh = self._imp.get_mesh()

    def _render_bobbin(self):
        self._plotter.clear()
        if self._mesh is not None and self._mesh.n_cells > 0:
            self._plotter.add_mesh(
                self._mesh, color='#aaaaaa', opacity=1.0,
                show_edges=True, edge_color='#555555',
                line_width=0.5, smooth_shading=True,
            )

    def _clear_highlight(self):
        if self._highlight_actor is not None:
            try:
                self._plotter.remove_actor(self._highlight_actor)
            except Exception:
                pass
            self._highlight_actor = None

    def _render_collected_tubes(self):
        """Re-render tube previews for all collected items."""
        import pyvista as pv
        for i, item in enumerate(self._collected):
            if item.get('_actor') is not None:
                try:
                    self._plotter.remove_actor(item['_actor'])
                except Exception:
                    pass
                item['_actor'] = None
            flist = item['face_indices']
            coords, normals = self._imp.discretize_face_group(flist)
            if coords is not None and len(coords) >= 3:
                if (self._imp._backend == 'gmsh'
                        and self._imp._gmsh_active):
                    result = self._imp._project_normals_gmsh(coords)
                    if result is not None:
                        coords, normals = result
                item['coords'] = coords
                item['normals'] = normals
                try:
                    arc = float(np.linalg.norm(
                        np.diff(coords, axis=0), axis=1).sum())
                    spline = pv.Spline(
                        coords, n_points=min(len(coords), 500))
                    tube = spline.tube(
                        radius=max(arc * 0.003, 0.0005))
                    color = self._CHAN_COLORS[
                        i % len(self._CHAN_COLORS)]
                    a = self._plotter.add_mesh(
                        tube, color=color, opacity=1.0,
                        reset_camera=False)
                    item['_actor'] = a
                except Exception:
                    pass
        self._plotter.render()

    # ── Picking ──────────────────────────────────────────────────────

    def _on_pick(self, *args):
        # Follow-channel pick in Stage 2
        if (self._left_stack.currentIndex() == 1
                and self._follow_mode):
            if not args:
                return
            point = args[0]
            if hasattr(point, 'points'):
                point = point.points[0] if point.n_points else None
            if point is None:
                return
            pt = np.asarray(point, dtype=np.float64).ravel()[:3]
            if not np.allclose(pt, 0.0):
                self._s2_on_follow_pick(pt)
            return

        if self._left_stack.currentIndex() != 0:
            return  # picking only in stage 1
        if not args or self._mesh is None:
            return
        point = args[0]
        if hasattr(point, 'points'):
            point = point.points[0] if point.n_points else None
        if point is None:
            return
        pt = np.asarray(point, dtype=np.float64).ravel()[:3]
        if np.allclose(pt, 0.0):
            return

        fids = self._mesh.cell_data.get('FaceId')
        if fids is None:
            return
        try:
            cell_idx = int(self._mesh.find_closest_cell(pt))
        except Exception:
            from scipy.spatial import cKDTree
            cc = self._mesh.cell_centers().points
            _, cell_idx = cKDTree(cc).query(pt)
            cell_idx = int(cell_idx)

        face_idx = int(fids[cell_idx])

        # Always toggle: click a highlighted face to deselect it,
        # click an unhighlighted face to select it.
        # Shift-click adds without clearing others.
        modifiers = QApplication.keyboardModifiers()
        if face_idx in self._selected_faces:
            self._selected_faces.discard(face_idx)
        elif modifiers & Qt.ShiftModifier:
            self._selected_faces.add(face_idx)
        else:
            self._selected_faces = {face_idx}

        self._update_highlight()

    def _update_highlight(self):
        self._clear_highlight()
        if not self._selected_faces or self._mesh is None:
            self._plotter.render()
            return

        fids = self._mesh.cell_data.get('FaceId')
        if fids is None:
            return
        mask = np.isin(fids, sorted(self._selected_faces))
        if not mask.any():
            self._plotter.render()
            return

        sub = self._mesh.extract_cells(np.where(mask)[0])
        color = self._CHAN_COLORS[
            len(self._collected) % len(self._CHAN_COLORS)]
        self._highlight_actor = self._plotter.add_mesh(
            sub, color=color, opacity=0.85,
            show_edges=False, smooth_shading=True,
            reset_camera=False,
        )
        try:
            mapper = self._highlight_actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                -2.0, -2.0)
        except Exception:
            pass
        self._plotter.render()
        n = len(self._selected_faces)
        self._status.setText(
            f"{n} face(s) selected.  Add or shift-click more.")

    # ── Stage 1 — Collect ────────────────────────────────────────────

    def _s1_add(self):
        if not self._selected_faces:
            self._status.setText("Select at least one face first.")
            return
        winds = self._s1_winds.value()
        flist = sorted(self._selected_faces)
        self._collected.append({
            'face_indices': flist,
            'winds': winds,
            'coords': None,
            'normals': None,
            '_actor': None,
        })
        nf = len(flist)
        label = (f"Face group ({nf} face{'s' if nf > 1 else ''}"
                 f", {winds} turns)")
        self._s1_list.addItem(label)
        self._selected_faces.clear()
        self._clear_highlight()
        self._render_collected_tubes()
        self._status.setText(
            f"Added.  {len(self._collected)} item(s) collected.")

    def _s1_remove(self):
        row = self._s1_list.currentRow()
        if row < 0 or row >= len(self._collected):
            return
        item = self._collected.pop(row)
        self._s1_list.takeItem(row)
        if item.get('_actor') is not None:
            try:
                self._plotter.remove_actor(item['_actor'])
                self._plotter.render()
            except Exception:
                pass

    # ── Stage 2 — Refine / Split ─────────────────────────────────────

    def _s2_populate(self):
        self._s2_list.clear()
        ref_span = self._get_reference_groove_span()
        for i, item in enumerate(self._collected):
            flist = item['face_indices']
            nf = len(flist)
            tag = item.get('_label', '')
            label = f"{i+1}. {nf} face(s), {item['winds']} turns"
            if tag:
                label += f"  [{tag}]"
            elif nf == 1 and ref_span is not None and ref_span > 0:
                span = self._imp.get_face_cross_span(flist[0])
                est = max(1, round(span / ref_span))
                if est > 1:
                    label += f"  [~{est} grooves]"
            self._s2_list.addItem(label)

    def _get_reference_groove_span(self) -> float | None:
        """Get the cross-groove parametric span of the smallest
        single-face item (likely a single groove)."""
        spans = []
        for item in self._collected:
            if len(item['face_indices']) == 1:
                fi = item['face_indices'][0]
                spans.append(self._imp.get_face_cross_span(fi))
        return min(spans) if spans else None

    def _s2_split(self):
        row = self._s2_list.currentRow()
        if row < 0 or row >= len(self._collected):
            self._status.setText("Select an item to split.")
            return
        item = self._collected[row]
        if len(item['face_indices']) != 1:
            self._status.setText(
                "Can only split a single-face item.")
            return

        face_idx = item['face_indices'][0]
        n_parts = self._s2_nparts.value()

        self._status.setText(f"Splitting face into {n_parts}…")
        QApplication.processEvents()

        try:
            new_indices = self._imp.split_face_equal(
                face_idx, n_parts)
        except Exception as exc:
            self._status.setText(f"Split failed: {exc}")
            return

        # Regenerate mesh after OCC modification
        self._load_mesh()

        # Replace the single collected item with N items
        winds = item['winds']
        if item.get('_actor') is not None:
            try:
                self._plotter.remove_actor(item['_actor'])
            except Exception:
                pass
        self._collected.pop(row)
        for ni in new_indices:
            self._collected.insert(row, {
                'face_indices': [ni],
                'winds': winds,
                'coords': None,
                'normals': None,
                '_actor': None,
            })
            row += 1

        self._s2_populate()
        self._render_collected_only()
        self._status.setText(
            f"Split into {len(new_indices)} faces.")

    def _s2_delete(self):
        row = self._s2_list.currentRow()
        if row < 0 or row >= len(self._collected):
            return
        item = self._collected.pop(row)
        if item.get('_actor') is not None:
            try:
                self._plotter.remove_actor(item['_actor'])
            except Exception:
                pass
        self._s2_populate()
        self._render_collected_only()

    def _render_collected_only(self):
        """Render only the collected faces — strip away the full bobbin."""
        self._plotter.clear()
        if self._mesh is None:
            return

        fids = self._mesh.cell_data.get('FaceId')
        if fids is None:
            return

        # Gather all face indices across all collected items
        all_faces = set()
        for item in self._collected:
            all_faces.update(item['face_indices'])

        if not all_faces:
            self._plotter.render()
            return

        mask = np.isin(fids, sorted(all_faces))
        sub = self._mesh.extract_cells(np.where(mask)[0])
        if sub.n_cells > 0:
            self._plotter.add_mesh(
                sub, color='#aaaaaa', opacity=1.0,
                show_edges=True, edge_color='#555555',
                line_width=0.5, smooth_shading=True,
            )

        # Re-render tubes for items that have coords
        import pyvista as pv
        for i, item in enumerate(self._collected):
            if item.get('_actor') is not None:
                try:
                    self._plotter.remove_actor(item['_actor'])
                except Exception:
                    pass
                item['_actor'] = None
            coords = item.get('coords')
            if coords is not None and len(coords) >= 3:
                try:
                    arc = float(np.linalg.norm(
                        np.diff(coords, axis=0), axis=1).sum())
                    spline = pv.Spline(
                        coords, n_points=min(len(coords), 500))
                    tube = spline.tube(
                        radius=max(arc * 0.003, 0.0005))
                    color = self._CHAN_COLORS[
                        i % len(self._CHAN_COLORS)]
                    a = self._plotter.add_mesh(
                        tube, color=color, opacity=1.0,
                        reset_camera=False)
                    item['_actor'] = a
                except Exception:
                    pass

        self._plotter.render()

    # ── Follow Channel mode (Stage 2) ────────────────────────────────

    def _s2_enter_follow(self):
        row = self._s2_list.currentRow()
        if row < 0 or row >= len(self._collected):
            self._status.setText("Select a combined face first.")
            return
        item = self._collected[row]
        if len(item['face_indices']) != 1:
            self._status.setText(
                "Follow Channel works on single-face items only.")
            return
        self._follow_mode = True
        self._follow_combined_row = row
        self._btn_follow.setEnabled(False)
        self._btn_cancel_follow.show()
        self._status.setText(
            "Click a properly isolated channel as reference.")

    def _s2_cancel_follow(self):
        self._follow_mode = False
        self._follow_combined_row = None
        self._btn_follow.setEnabled(True)
        self._btn_cancel_follow.hide()
        self._status.setText("Follow cancelled.")

    def _s2_on_follow_pick(self, pt):
        """User clicked a reference channel while in follow mode."""
        fids = self._mesh.cell_data.get('FaceId')
        if fids is None:
            return
        try:
            cell_idx = int(self._mesh.find_closest_cell(pt))
        except Exception:
            from scipy.spatial import cKDTree
            cc = self._mesh.cell_centers().points
            _, cell_idx = cKDTree(cc).query(pt)
            cell_idx = int(cell_idx)

        ref_face_idx = int(fids[cell_idx])

        # Find which collected item this face belongs to
        ref_item = None
        for item in self._collected:
            if ref_face_idx in item['face_indices']:
                ref_item = item
                break

        if ref_item is None:
            self._status.setText(
                "Click on a collected face (highlighted area).")
            return

        if len(ref_item['face_indices']) != 1:
            self._status.setText(
                "Reference must be a single-face channel, "
                "not a multi-face group.")
            return

        comb_row = self._follow_combined_row
        comb_item = self._collected[comb_row]
        comb_face_idx = comb_item['face_indices'][0]

        if ref_face_idx == comb_face_idx:
            self._status.setText(
                "Reference must be different from the "
                "face being split.")
            return

        self._status.setText("Extracting channel strip…")
        QApplication.processEvents()

        try:
            new_indices, strip_idx = \
                self._imp.extract_channel_strip(
                    comb_face_idx, ref_face_idx)
        except Exception as exc:
            self._status.setText(f"Follow failed: {exc}")
            self._s2_cancel_follow()
            return

        # Regenerate mesh
        self._load_mesh()

        # Replace the combined item with the new face parts
        winds = comb_item['winds']
        if comb_item.get('_actor') is not None:
            try:
                self._plotter.remove_actor(comb_item['_actor'])
            except Exception:
                pass
        self._collected.pop(comb_row)
        insert_at = comb_row
        for ni in new_indices:
            label = "strip" if ni == strip_idx else "margin"
            self._collected.insert(insert_at, {
                'face_indices': [ni],
                'winds': winds,
                'coords': None,
                'normals': None,
                '_actor': None,
                '_label': label,
            })
            insert_at += 1

        self._s2_cancel_follow()
        self._s2_populate()
        self._render_collected_only()
        n = len(new_indices)
        self._status.setText(
            f"Split into {n} parts.  "
            "Delete the margins, keep the strip.")

    # ── Stage 3 — Extract & Review ───────────────────────────────────

    def _s3_extract(self):
        import pyvista as pv

        self._result.clear()
        self._s3_list.clear()
        self._render_bobbin()

        self._status.setText("Extracting centerlines…")
        QApplication.processEvents()

        for i, item in enumerate(self._collected):
            flist = item['face_indices']
            winds = item['winds']
            coords, normals = self._imp.discretize_face_group(flist)
            if coords is None or len(coords) < 3:
                continue

            # Exact parametric normals
            if (self._imp._backend == 'gmsh'
                    and self._imp._gmsh_active):
                result = self._imp._project_normals_gmsh(coords)
                if result is not None:
                    coords, normals = result

            self._result.append({
                'coords': coords,
                'normals': normals,
                'winds': winds,
            })

            # Render tube
            color = self._CHAN_COLORS[i % len(self._CHAN_COLORS)]
            try:
                arc = float(np.linalg.norm(
                    np.diff(coords, axis=0), axis=1).sum())
                spline = pv.Spline(
                    coords, n_points=min(len(coords), 500))
                tube = spline.tube(
                    radius=max(arc * 0.003, 0.0005))
                self._plotter.add_mesh(
                    tube, color=color, opacity=1.0,
                    reset_camera=False)
            except Exception:
                pass

            # Normal arrows (every ~20th point)
            step = max(1, len(coords) // 20)
            arrow_pts = coords[::step]
            arrow_norms = normals[::step]
            try:
                glyph_src = pv.Arrow()
                arrows = pv.PolyData(arrow_pts)
                arrows["vec"] = arrow_norms
                scale = arc * 0.015
                glyphs = arrows.glyph(
                    orient="vec", scale=False,
                    factor=scale, geom=glyph_src)
                self._plotter.add_mesh(
                    glyphs, color=color, opacity=0.8,
                    reset_camera=False)
            except Exception:
                pass

            self._s3_list.addItem(
                f"Channel {i+1}  ({len(coords)} pts, "
                f"{winds} turns)")

        self._plotter.render()
        self._info.setText(
            "Review channels and normals.  "
            "Remove unwanted channels, then Done.")
        self._status.setText(
            f"{len(self._result)} channel(s) ready.")

    def _s3_remove(self):
        row = self._s3_list.currentRow()
        if row < 0 or row >= len(self._result):
            return
        self._result.pop(row)
        self._collected.pop(row)
        self._s3_list.takeItem(row)
        # Re-render
        self._s3_extract()

    # ── Finish ───────────────────────────────────────────────────────

    def _finish(self):
        if not self._result:
            self._status.setText("No channels to export.")
            return
        try:
            self._plotter.close()
        except Exception:
            pass
        self.accept()

    def get_selected(self) -> list[dict]:
        return self._result

    def closeEvent(self, event):
        try:
            self._plotter.close()
        except Exception:
            pass
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CalcSX™ – v{__version__}")
        self.setWindowIcon(get_app_icon())
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
        self._coil_paths:    dict = {}          # coil_id → CSV file path (absolute)
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

        # Hall probes (multiple)
        self._probe_counter = 0
        self._probe_timer   = None

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
        self.ribbon.import_bobbin.connect(self._on_import_bobbin)
        self.ribbon.run_analysis.connect(self._on_run_analysis)
        self.ribbon.reanalyze_all.connect(self._on_reanalyze_all)
        self.ribbon.compute_field_lines.connect(self._on_compute_field_lines)
        self.ribbon.compute_cross_section.connect(self._on_compute_cross_section)
        self.ribbon.global_field_toggled.connect(self._on_global_field_toggled)
        self.ribbon.translate_toggled.connect(self._on_translate_toggled)
        self.ribbon.rotate_toggled.connect(self._on_rotate_toggled)
        self.ribbon.reset_transform.connect(self._on_reset_transform)
        self.ribbon.generate_coil.connect(self._on_generate_coil)
        self.ribbon.show_help.connect(lambda: HelpDialog(self).exec_())
        self.browser.layer_toggled.connect(self.workspace.set_layer_visible)
        self.browser.coil_visibility_toggled.connect(self.workspace.set_coil_visible)
        self.browser.coil_delete_requested.connect(self._on_coil_delete)
        self.browser.coil_selected.connect(self._on_coil_selected)
        self.browser.coil_renamed.connect(self._on_coil_renamed)
        self.browser.coil_recolored.connect(self._on_coil_recolored)
        self.ribbon.normalize_forces_toggled.connect(self._on_normalize_forces_toggled)
        self.ribbon.open_settings.connect(self._on_open_settings)
        self.ribbon.save_session.connect(lambda: self._save_session(self))
        self.ribbon.load_session.connect(lambda: self._load_session(self))
        self.ribbon.add_hall_probe.connect(self._on_add_hall_probe)
        self.browser.probe_selected.connect(self._on_probe_selected)
        self.browser.probe_delete_requested.connect(self._on_probe_delete)
        self.browser.probe_recolored.connect(
            lambda pid, c: self.workspace.set_probe_color(pid, c)
        )
        self.workspace.set_stale_callback(self._on_layers_stale)

        # Listen to per-coil parameter changes → mark active coil stale
        for w in (self.props.spin_winds, self.props.dspin_current,
                  self.props.dspin_thick, self.props.dspin_width,
                  self.props.spin_axis_pts):
            w.valueChanged.connect(self._on_coil_param_changed)

    # ── Signal handlers ───────────────────────────────────────────────────────

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
                from physics.geometry import import_step_centerline
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
        self.workspace.add_coil(coords, coil_id, color=color,
                                 total_thickness=total_t, tape_width=tape_w)
        self.browser.add_coil_item(coil_id, self._coil_names[coil_id], color)

        self._multi_env.register_coil(
            coil_id, coords,
            winds=params['winds'], current=params['current'],
            thickness=params['thickness'], width=params['width'],
        )
        self._propagate_staleness()

        self.ribbon.set_inspect_enabled(True)
        self.ribbon.set_construct_enabled(True)
        self.props.show()

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
            self._coil_params_map[coil_id] = p

            total_t = p['thickness'] * 1e-6 * winds
            tape_w = p['width'] * 1e-3
            self.workspace.add_coil(coords, coil_id, color=color,
                                     total_thickness=total_t,
                                     tape_width=tape_w,
                                     tape_normals=normals)
            self.browser.add_coil_item(coil_id, name, color)

            self._multi_env.register_coil(
                coil_id, coords,
                winds=winds, current=p['current'],
                thickness=p['thickness'], width=p['width'],
                tape_normals=normals,
            )

        self._propagate_staleness()
        self.ribbon.set_inspect_enabled(True)
        self.ribbon.set_construct_enabled(True)
        self.props.show()
        if self.workspace._plotter:
            self.workspace._plotter.reset_camera()
            self.workspace._plotter.render()

    def _on_coil_delete(self, coil_id: str) -> None:
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
            return

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
        self._coil_paths.pop(coil_id, None)
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
            self._coil_params_map[old_cid].update(self.props.get_params())
        self._active_coil_id = coil_id
        self._coords = self._coil_coords.get(coil_id)
        # Switch gizmo target back to coil
        self.workspace.set_gizmo_target('coil')
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
        # Update stored params — merge UI values into existing dict
        # so that non-UI fields (tape_normals, etc.) are preserved.
        prev = self._coil_params_map.get(cid, {})
        prev.update(self.props.get_params())
        self._coil_params_map[cid] = prev
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
        self.workspace.update_coil_mesh(cid, total_t, tape_w,
                                        tape_normals=p.get('tape_normals'))

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
        # Save spinbox values only if the user is looking at this coil
        # (during re-analyze-all, active_coil_id is temporarily changed
        # but the spinboxes still show a different coil's values).
        cid_save = self._active_coil_id
        if (cid_save and cid_save in self._coil_params_map
                and not getattr(self, '_reanalyze_queue', None)):
            self._coil_params_map[cid_save].update(self.props.get_params())

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

        norm = self.ribbon._btn_normalize.isChecked()
        self.workspace.add_force_layer(engine, cid, normalized=norm,
                                        show_arrows=norm,
                                        progress_callback=_force_progress)

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
        """Re-run analysis on ALL coils sequentially."""
        self._reanalyze_queue = [
            cid for cid in self._coil_coords if cid in self._coil_params_map
        ]
        self._reanalyze_next()

    def _reanalyze_next(self) -> None:
        """Pop the next stale coil and run its analysis."""
        if not getattr(self, '_reanalyze_queue', None):
            self.ribbon._btn_reanalyze.set_action_enabled(False)
            # All coils done — now apply the global force colour scale
            self.workspace.rescale_all_force_layers()
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
            # Use cached result if available, environment unchanged, and
            # seed count matches what was used for the cache.
            n_seeds = self.props.get_field_seeds()
            cache_valid = (not self._global_fl_dirty
                           and self._global_fl_cache is not None
                           and getattr(self, '_global_fl_cache_seeds', 0) == n_seeds)
            if cache_valid:
                lines, B_mags = self._global_fl_cache
                self.workspace.add_field_lines_layer(lines, B_mags, '__global__')
                self.workspace.rescale_all_field_line_layers()
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
        self._global_fl_cache_seeds = self.props.get_field_seeds()
        self._global_fl_dirty = False
        self.workspace.add_field_lines_layer(lines, B_mags, '__global__')
        self.workspace.rescale_all_field_line_layers()

    def _on_normalize_forces_toggled(self, checked: bool = False) -> None:
        cid = self._active_coil_id
        engine = self._coil_engines.get(cid)
        if engine is None:
            return
        self.workspace.add_force_layer(engine, cid, normalized=checked,
                                        show_arrows=checked)
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
        self.workspace.rescale_all_field_line_layers()
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
        # Update superposition environment with new world-space coords
        if cid:
            wc = self.workspace.get_transformed_coords(cid)
            if wc is not None:
                self._multi_env.update_coil_coords(cid, wc)
            self._propagate_staleness()

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.theme_changed.connect(self._apply_theme)
        dlg.export_vtk_requested.connect(lambda: self._export_vtk_layers(dlg))
        dlg.export_web_layers_requested.connect(lambda: self._export_web_layers(dlg))
        dlg.exec_()

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

        from gui.gui_utils import get_theme_name
        original_theme = get_theme_name()
        total = []

        for theme in ('dark', 'light'):
            folder = os.path.join(out_dir, f"weblayers-{name}-{theme}")
            exported = self.workspace.export_web_layers(folder, theme)
            total.extend(exported)

        # Restore original theme (the export doesn't change the live theme,
        # but guard against future changes)
        if get_theme_name() != original_theme:
            self._apply_theme(original_theme)

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
        self.workspace.clear_field_lines_layer('__global__')
        if self.ribbon._btn_global_field.isChecked():
            self.ribbon._btn_global_field.blockSignals(True)
            self.ribbon._btn_global_field.setChecked(False)
            self.ribbon._btn_global_field.blockSignals(False)
        # Reset state
        self._coil_coords.clear()
        self._coil_paths.clear()
        self._coil_names.clear()
        self._coil_params_map.clear()
        self._coil_engines.clear()
        self._multi_env = MultiCoilEnvironment()
        self._active_coil_id = None
        self._analyzed_coil_id = None
        self._coords = None
        self._coil_counter = 0
        self.ribbon.set_run_enabled(False)
        self.ribbon.set_inspect_enabled(False)
        self.ribbon.set_construct_enabled(False)
        self.props.hide()
        if self.workspace._plotter:
            self.workspace._plotter.render()

    def _save_session(self, parent=None) -> None:
        """Export coil arrangement to a .csx file for later reload."""
        import json
        parent = parent or self
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Session", "session.csx",
            "CalcSX Session (*.csx)",
        )
        if not path:
            return

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
            coils.append({
                'coil_id':    coil_id,
                'csv_path':   self._coil_paths.get(coil_id, ''),
                'name':       self._coil_names.get(coil_id, ''),
                'color':      entry.get('color', ''),
                'coords':     coords.tolist(),
                'params':     save_params,
                'xfm_params': list(entry['xfm_params']) if entry.get('xfm_params') else None,
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

        with open(path, 'w') as f:
            json.dump({'version': 2, 'coils': coils, 'bobbins': bobbins},
                      f, indent=2)

        n_total = len(coils) + len(bobbins)
        QMessageBox.information(
            parent, "Session Saved",
            f"Saved {len(coils)} coil(s) and {len(bobbins)} bobbin(s) to:\n{path}",
        )

    def _load_session(self, parent=None) -> None:
        """Restore a coil arrangement from a previously saved .csx session."""
        import json
        parent = parent or self
        path, _ = QFileDialog.getOpenFileName(
            parent, "Load Session", "",
            "CalcSX Session (*.csx);;JSON files (*.json)",
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

        file_ver = data.get('version', 1)
        failed = []
        loaded = 0
        for entry in data.get('coils', []):
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
            self.workspace.add_coil(coords, coil_id, color=color,
                                     total_thickness=total_t, tape_width=tape_w,
                                     tape_normals=params.get('tape_normals'))
            self.browser.add_coil_item(coil_id, name, color)

            self._multi_env.register_coil(
                coil_id, coords,
                winds=params['winds'], current=params['current'],
                thickness=params['thickness'], width=params['width'],
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

            loaded += 1

        # ── Restore bobbin meshes ──
        n_bobbins = 0
        for bentry in data.get('bobbins', []):
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
        self.ribbon.set_inspect_enabled(bool(self._coil_coords))
        self.ribbon.set_construct_enabled(bool(self._coil_coords))
        if self._active_coil_id:
            self._load_coil_params(self._active_coil_id)
        self.props.show()
        if self.workspace._plotter:
            self.workspace._plotter.reset_camera()
            self.workspace._plotter.render()

        msg = f"Loaded {loaded} coil(s)"
        if n_bobbins:
            msg += f" and {n_bobbins} bobbin(s)"
        msg += "."
        if failed:
            msg += f"\n\nFailed to load {len(failed)} coil(s):\n" + "\n".join(failed)
        QMessageBox.information(self, "Session Loaded", msg)

    def _apply_theme(self, name: str) -> None:
        from gui.gui_utils import apply_theme_to_app
        apply_theme_to_app(name)
        # Refresh every panel that uses explicit setStyleSheet calls
        self.ribbon.refresh_theme()
        self.browser.refresh_theme()
        self.props.refresh_theme()
        # Update 3D viewport (background, floor, ViewCube)
        self.workspace.apply_theme()
        # Swap window + application icon to match the new theme
        themed_icon = get_app_icon(name)
        self.setWindowIcon(themed_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(themed_icon)

    # ── Hall probe handlers ──────────────────────────────────────────────────

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
        if self.ribbon._btn_translate.isChecked() or self.ribbon._btn_rotate.isChecked():
            mode = 'T' if self.ribbon._btn_translate.isChecked() else 'R'
            self.workspace.show_gizmo(mode)
        self._update_single_probe_readout(probe_id)

    def _on_probe_delete(self, probe_id: str) -> None:
        """Delete a specific Hall probe."""
        self.workspace.remove_hall_probe(probe_id)
        self.browser.remove_probe_item(probe_id)
        # Stop timer if no probes remain
        if not self.workspace._probe_entries:
            if self._probe_timer is not None:
                self._probe_timer.stop()
                self._probe_timer = None

    def _propagate_staleness(self) -> None:
        """Reflect MultiCoilEnvironment staleness into browser stale markers."""
        stale = self._multi_env.get_stale_coils()
        for stale_cid in stale:
            if stale_cid in self._coil_engines:
                for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
                    self.browser.mark_layer_stale(stale_cid, nm, True)
        self.ribbon._btn_reanalyze.set_action_enabled(bool(stale))
        self._global_fl_dirty = True

        # Auto-clear global field lines — they're invalid now
        if self.ribbon._btn_global_field.isChecked():
            self.workspace.clear_field_lines_layer('__global__')
            self.ribbon._btn_global_field.blockSignals(True)
            self.ribbon._btn_global_field.setChecked(False)
            self.ribbon._btn_global_field.blockSignals(False)
            # Restore per-coil field line controls
            self.ribbon._btn_field_lines.set_action_enabled(True)
            for cid in list(self._coil_coords.keys()):
                was_on = getattr(self, '_pre_global_eye_state', {}).get(cid, True)
                self.browser.set_layer_eye_unlocked(cid, 'Field Lines', restore_checked=was_on)


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
