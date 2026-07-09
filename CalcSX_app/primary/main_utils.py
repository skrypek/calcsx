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
import base64
import pickle
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
    QAbstractItemView,
    QMenu,
    QCheckBox,
    QComboBox,
    QListWidget,
    QProgressDialog,
    QStackedWidget,
    QRadioButton,
    QButtonGroup,
    QTabBar,
    QToolButton,
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtWidgets import QShortcut

from CalcSX_app.physics.physics_utils import CoilAnalysis
from CalcSX_app.physics.superposition import MultiCoilEnvironment
from CalcSX_app.gui.gui_utils import ProgressReporter, THEME, get_app_icon
from CalcSX_app.views.workspace_3d import Workspace3DView

from CalcSX_app.version import __version__ as version_module
__version__ = getattr(version_module, "__version__", "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    progress = pyqtSignal(int)
    stage    = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, coords, winds, current, thickness, width,
                 n_grid=120, axis_num=200, B_ext=None, tape_normals=None,
                 winding_growth='symmetric'):
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
        self.winding_growth = winding_growth

    @pyqtSlot()
    def run(self):
        engine = CoilAnalysis(
            self.coords, self.winds, self.current,
            self.thickness, self.width,
            B_ext=self.B_ext,
            tape_normals=self.tape_normals,
            winding_growth=self.winding_growth,
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
        from CalcSX_app.physics.superposition import compute_global_field_lines
        lines, B_mags = compute_global_field_lines(
            self.B_total,
            self.coil_infos,
            n_seeds=self.n_seeds,
            progress_callback=self.progress.emit,
        )
        self.finished.emit((lines, B_mags))


class LMatrixWorker(QObject):
    """Warm ``MultiCoilEnvironment._L_cache`` off the UI thread so the first
    circuit-header click doesn't freeze the app on the N×N Neumann double sum."""
    finished = pyqtSignal(object)   # coil_ids list used, or None on failure

    def __init__(self, multi_env):
        super().__init__()
        self._env = multi_env

    @pyqtSlot()
    def run(self):
        try:
            result = self._env.compute_mutual_inductance_matrix()
            self.finished.emit(result.get('coil_ids'))
        except Exception:
            self.finished.emit(None)




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


class _DropdownRibbonBtn(QWidget):
    """Fusion360-style split button: icon+label button on top showing the
    current default action, with a small chevron strip at the bottom that
    pops up a menu of all available actions. Clicking an action in the menu
    sets it as the new default AND fires it immediately.

    Each action has its own enable/disable state via ``set_action_enabled``.
    The primary button follows the enabled state of the currently-selected
    default; disabled actions in the dropdown appear greyed out.

    Constructor takes a list of ``(symbol, label, callback)`` tuples; the
    first entry is the initial default. All entries (including the default)
    appear in the dropdown.
    """
    def __init__(self, actions: list, parent=None):
        super().__init__(parent)
        self._actions = list(actions)                 # [(sym, label, cb), ...]
        self._enabled_flags = [True] * len(self._actions)
        self._default_idx = 0

        sym, lbl, _ = self._actions[0]
        self._main_btn = _RibbonBtn(sym, lbl, enabled=True)
        self._main_btn.clicked.connect(self._fire_default)

        self._chev_btn = QPushButton("▾")
        self._chev_btn.setFixedHeight(10)
        self._chev_btn.setFlat(True)
        self._chev_btn.setCursor(Qt.PointingHandCursor)
        # Use an objectName-scoped selector so the app-wide QPushButton QSS
        # (which sets a 1px border + padding on every button) doesn't leak
        # through. Without this override there's a visible horizontal line
        # where the chevron's top/bottom border meets the main button below
        # the label — readable as a "middle bar" across each dropdown tile.
        self._chev_btn.setObjectName('ribbon_chevron')
        self._chev_btn.setStyleSheet(
            "QPushButton#ribbon_chevron {"
            f" color: {THEME['text_dim']}; font-size: 8pt;"
            " border: 0px; border-width: 0; outline: 0;"
            " background: transparent; padding: 0; margin: 0;"
            " }"
            "QPushButton#ribbon_chevron:hover {"
            f" color: {THEME['text']};"
            f" background: {THEME['input']};"
            " border: 0px; }"
        )
        self._chev_btn.clicked.connect(self._show_menu)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._main_btn)
        lay.addWidget(self._chev_btn)

        self.setFixedWidth(self._main_btn.width())

    def _fire_default(self) -> None:
        if not self._enabled_flags[self._default_idx]:
            return
        try:
            self._actions[self._default_idx][2]()
        except Exception:
            pass

    def _show_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background:{THEME['panel']};"
            f" color:{THEME['text']}; border:1px solid {THEME['border']}; }}"
            f"QMenu::item {{ padding:6px 14px; }}"
            f"QMenu::item:disabled {{ color:{THEME['text_dim']}; }}"
            f"QMenu::item:selected {{ background:{THEME['hi_blue']};"
            f" color:{THEME['text']}; }}"
        )
        for i, (sym, label, _) in enumerate(self._actions):
            text = f"{sym}   {label.replace(chr(10), ' ')}"
            act = menu.addAction(text)
            act.setData(i)
            act.setEnabled(self._enabled_flags[i])
        chosen = menu.exec_(self._chev_btn.mapToGlobal(
            self._chev_btn.rect().bottomLeft()))
        if chosen is not None:
            idx = chosen.data()
            if isinstance(idx, int) and 0 <= idx < len(self._actions) \
                    and self._enabled_flags[idx]:
                self._set_default(idx)
                self._fire_default()

    def _set_default(self, idx: int) -> None:
        self._default_idx = idx
        sym, lbl, _ = self._actions[idx]
        self._main_btn._sym_lbl.setText(sym)
        self._main_btn._txt_lbl.setText(lbl)
        self._main_btn.set_action_enabled(self._enabled_flags[idx])

    def set_action_enabled(self, idx: int, enabled: bool) -> None:
        """Enable / disable a specific action by its index."""
        if 0 <= idx < len(self._enabled_flags):
            self._enabled_flags[idx] = bool(enabled)
            if idx == self._default_idx:
                self._main_btn.set_action_enabled(bool(enabled))

    def action_proxy(self, idx: int):
        """Return a lightweight proxy with ``set_action_enabled(bool)``
        targeting this dropdown's index-`idx` action — lets call-sites
        treat a specific dropdown action like the old standalone button
        (e.g. `_btn_reanalyze.set_action_enabled(True)`)."""
        parent = self
        class _Proxy:
            def set_action_enabled(self, on: bool) -> None:
                parent.set_action_enabled(idx, on)
            # Kept for legacy callers that checked state
            def isChecked(self) -> bool:
                return False
            def setChecked(self, _on: bool) -> None:
                pass
            def refresh_theme(self) -> None:
                parent._main_btn.refresh_theme()
        return _Proxy()

    def refresh_theme(self) -> None:
        self._main_btn.refresh_theme()
        # Use an objectName-scoped selector so the app-wide QPushButton QSS
        # (which sets a 1px border + padding on every button) doesn't leak
        # through. Without this override there's a visible horizontal line
        # where the chevron's top/bottom border meets the main button below
        # the label — readable as a "middle bar" across each dropdown tile.
        self._chev_btn.setObjectName('ribbon_chevron')
        self._chev_btn.setStyleSheet(
            "QPushButton#ribbon_chevron {"
            f" color: {THEME['text_dim']}; font-size: 8pt;"
            " border: 0px; border-width: 0; outline: 0;"
            " background: transparent; padding: 0; margin: 0;"
            " }"
            "QPushButton#ribbon_chevron:hover {"
            f" color: {THEME['text']};"
            f" background: {THEME['input']};"
            " border: 0px; }"
        )


def _vbar() -> QFrame:
    """Thin vertical separator between ribbon groups."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1)
    f.setObjectName('vbar')
    return f


def _ribbon_group(label: str, buttons: list) -> QWidget:
    """Group of ribbon buttons. The ``label`` argument is accepted for
    API compatibility but no longer rendered — per-button labels plus the
    chevron strip on dropdown buttons convey grouping visually without
    the extra row of small-caps text below."""
    w   = QWidget()
    out = QVBoxLayout(w)
    out.setContentsMargins(4, 4, 4, 0)
    out.setSpacing(0)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    for b in buttons:
        row.addWidget(b)

    out.addLayout(row, stretch=1)
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
    open_settings            = pyqtSignal()
    save_session             = pyqtSignal()
    load_session             = pyqtSignal()
    # Inspect — probe
    add_hall_probe           = pyqtSignal()
    add_system_energy        = pyqtSignal()
    add_stray_array          = pyqtSignal()
    # Circuits — group/ungroup selected coils into a wired circuit
    group_as_series          = pyqtSignal()
    group_as_parallel        = pyqtSignal()
    ungroup_selection        = pyqtSignal()
    # Construct — position fix + relative distance + type-in values
    pin_toggled              = pyqtSignal(bool)  # pin/unpin ACTIVE coil
    relative_distance_requested = pyqtSignal()
    # Emitted when the user edits any of the Tx/Ty/Tz/Rx/Ry/Rz spinboxes.
    # Values are (tx m, ty m, tz m, rx deg, ry deg, rz deg).
    transform_values_changed = pyqtSignal(float, float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        # Scope every stylesheet to the widget's own object name so
        # `border-bottom` doesn't cascade to descendants. Without the
        # scoping, Qt applies the bottom border to every child widget,
        # which renders a stacked 1-px line under each ribbon button
        # (the artifact that was crossing every tile).
        self.setObjectName('ribbon_bar_widget')
        self.setStyleSheet(
            f"#ribbon_bar_widget {{"
            f"  background:{THEME['panel']};"
            f"  border-bottom:1px solid {THEME['border']};"
            f"}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Tab strip ─────────────────────────────────────────────────────────
        self._tab_bar = QWidget()
        self._tab_bar.setFixedHeight(24)
        self._tab_bar.setObjectName('ribbon_tab_bar')
        self._tab_bar.setStyleSheet(
            f"#ribbon_tab_bar {{"
            f"  background:{THEME['bg']};"
            f"  border-bottom:1px solid {THEME['border']};"
            f"}}"
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

        # FILE dropdown — Load Coil (primary) / Import Bobbin / Load Environment
        self._file_dropdown = _DropdownRibbonBtn([
            ("▲", "Load\nCoil",        lambda: self.load_csv.emit()),
            ("⬡", "Import\nBobbin",    lambda: self.import_bobbin.emit()),
            ("⬆", "Load\nEnvironment", lambda: self.load_session.emit()),
        ])
        # ANALYSIS dropdown — Run Analysis (primary) / Re-analyze All.
        # Re-analyze starts disabled; gets enabled when there are stale coils.
        self._analysis_dropdown = _DropdownRibbonBtn([
            ("▶", "Run\nAnalysis",     lambda: self.run_analysis.emit()),
            ("⟳", "Re-analyze\nAll",   lambda: self.reanalyze_all.emit()),
        ])
        self._analysis_dropdown.set_action_enabled(1, False)

        # Legacy handles: proxies targeting specific actions within each
        # dropdown so existing ``_btn_run.set_action_enabled(bool)`` etc.
        # calls keep working without changes at the call sites.
        self._btn_load      = self._file_dropdown.action_proxy(0)
        self._btn_bobbin    = self._file_dropdown.action_proxy(1)
        self._btn_run       = self._analysis_dropdown.action_proxy(0)
        self._btn_reanalyze = self._analysis_dropdown.action_proxy(1)
        btn_help_sim        = _RibbonBtn("?", "Help")

        # SIMULATION groups
        self._sim_w = QWidget()
        sl = QHBoxLayout(self._sim_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(_ribbon_group("FILE",     [self._file_dropdown]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("ANALYSIS", [self._analysis_dropdown]))
        sl.addWidget(_vbar())
        sl.addWidget(_ribbon_group("HELP",     [btn_help_sim]))
        sl.addStretch(1)

        # INSPECT tab
        self._btn_field_lines  = _RibbonBtn("∿", "Field\nLines",   enabled=False)
        self._btn_cross_sec    = _RibbonBtn("⊡", "Cross\nSection", enabled=False)
        self._btn_global_field = _RibbonBtn("⊛", "Global\nField",  enabled=False, checkable=True)
        self._btn_hall_probe   = _RibbonBtn("⊙", "Hall\nProbe")
        self._btn_sys_energy   = _RibbonBtn("⚡", "System\nEnergy")
        self._btn_stray_array  = _RibbonBtn("📊", "Stray\nArray")
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
            self._btn_sys_energy,
            self._btn_stray_array,
        ]))
        il.addStretch(1)
        self._inspect_w.hide()

        # CONSTRUCT — coil positioning tools
        self._btn_translate = _RibbonBtn("⇱", "Translate", enabled=False, checkable=True)
        self._btn_rotate    = _RibbonBtn("↻", "Rotate",    enabled=False, checkable=True)
        self._btn_reset_xfm = _RibbonBtn("⌂", "Reset\nTransform", enabled=False)
        self._btn_pin       = _RibbonBtn("⚲", "Pin\nPosition", enabled=False, checkable=True)
        self._btn_rel_dist  = _RibbonBtn("⇄", "Relative\nDistance", enabled=False)
        self._construct_w = QWidget()
        cl = QHBoxLayout(self._construct_w)
        cl.setContentsMargins(8, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(_ribbon_group("POSITION", [
            self._btn_translate,
            self._btn_rotate,
            self._btn_reset_xfm,
            self._btn_pin,
            self._btn_rel_dist,
        ]))

        # VALUES — type-in XYZ translation and XYZ rotation for the active coil.
        # Two rows of three spinboxes. Bidirectional with the gizmo; editing a
        # value immediately re-applies the coil transform.
        self._values_grp = self._build_values_group()
        cl.addWidget(_vbar())
        cl.addWidget(self._values_grp)

        self._btn_generate = _RibbonBtn("+", "Generate\nCoil")
        cl.addWidget(_vbar())
        cl.addWidget(_ribbon_group("GENERATORS", [
            self._btn_generate,
        ]))
        # Circuit wiring — group multi-selected coils into series / parallel
        self._btn_series    = _RibbonBtn("⇌", "Group as\nSeries",   enabled=False)
        self._btn_parallel  = _RibbonBtn("∥", "Group as\nParallel", enabled=False)
        self._btn_ungroup   = _RibbonBtn("⊘", "Ungroup",            enabled=False)
        cl.addWidget(_vbar())
        cl.addWidget(_ribbon_group("CIRCUITS", [
            self._btn_series,
            self._btn_parallel,
            self._btn_ungroup,
        ]))
        cl.addStretch(1)
        self._construct_w.hide()

        # UTILITIES
        self._btn_save_ses = _RibbonBtn("⬇", "Save\nSession")
        self._btn_load_ses = _RibbonBtn("⬆", "Load\nSession")
        self._btn_settings  = _RibbonBtn("⚙", "Settings")
        self._btn_save_ses.clicked.connect(self.save_session)
        self._btn_load_ses.clicked.connect(self.load_session)
        self._util_w = QWidget()
        ul = QHBoxLayout(self._util_w)
        ul.setContentsMargins(8, 0, 0, 0)
        ul.setSpacing(4)
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

        # Wire signals. File and Analysis buttons are now dropdown widgets
        # that emit their own signals via lambdas (wired at construction),
        # so no separate .clicked hookup needed here.
        self._btn_field_lines.clicked.connect(self.compute_field_lines)
        self._btn_cross_sec.clicked.connect(self.compute_cross_section)
        self._btn_global_field.toggled.connect(self.global_field_toggled)
        self._btn_translate.toggled.connect(self._on_translate_toggled)
        self._btn_rotate.toggled.connect(self._on_rotate_toggled)
        self._btn_reset_xfm.clicked.connect(self.reset_transform)
        self._btn_generate.clicked.connect(self.generate_coil)
        btn_help_sim.clicked.connect(self.show_help)
        self._btn_settings.clicked.connect(self.open_settings)
        self._btn_hall_probe.clicked.connect(self.add_hall_probe)
        self._btn_sys_energy.clicked.connect(self.add_system_energy)
        self._btn_stray_array.clicked.connect(self.add_stray_array)
        self._btn_series.clicked.connect(self.group_as_series)
        self._btn_parallel.clicked.connect(self.group_as_parallel)
        self._btn_ungroup.clicked.connect(self.ungroup_selection)
        self._btn_pin.toggled.connect(self.pin_toggled)
        self._btn_rel_dist.clicked.connect(self.relative_distance_requested)

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
        self._btn_pin.set_action_enabled(on)
        self._btn_rel_dist.set_action_enabled(on)
        # Value spinboxes follow the same gate as the transform buttons.
        for s in (self._spin_tx, self._spin_ty, self._spin_tz,
                  self._spin_rx, self._spin_ry, self._spin_rz):
            s.setEnabled(on)

    def set_pin_state(self, pinned: bool) -> None:
        """Reflect the active coil's pinned state on the Pin toggle without
        triggering the pin_toggled signal."""
        if self._btn_pin.isChecked() == pinned:
            return
        self._btn_pin.blockSignals(True)
        self._btn_pin.setChecked(pinned)
        self._btn_pin.blockSignals(False)
        # Editing disabled while pinned — gizmo drag and type-in both blocked.
        locked = pinned
        for s in (self._spin_tx, self._spin_ty, self._spin_tz,
                  self._spin_rx, self._spin_ry, self._spin_rz):
            s.setEnabled(not locked)
        self._btn_translate.set_action_enabled(not locked)
        self._btn_rotate.set_action_enabled(not locked)

    def set_transform_values(self, tx: float, ty: float, tz: float,
                             rx: float, ry: float, rz: float) -> None:
        """Load the active coil's transform into the spinboxes without
        emitting transform_values_changed."""
        for s, v in ((self._spin_tx, tx), (self._spin_ty, ty), (self._spin_tz, tz),
                     (self._spin_rx, rx), (self._spin_ry, ry), (self._spin_rz, rz)):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)

    def get_transform_values(self) -> tuple:
        return (
            float(self._spin_tx.value()), float(self._spin_ty.value()),
            float(self._spin_tz.value()),
            float(self._spin_rx.value()), float(self._spin_ry.value()),
            float(self._spin_rz.value()),
        )

    def _build_values_group(self) -> QWidget:
        """Build the VALUES ribbon group — six spinboxes for Tx/Ty/Tz + Rx/Ry/Rz."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(2)

        header = QLabel("VALUES")
        header.setObjectName('ribbon_grp_lbl')
        header.setAlignment(Qt.AlignHCenter)
        outer.addWidget(header)

        grid = QFormLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)

        def _tspin(suffix: str) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-1000.0, 1000.0)
            s.setDecimals(4)
            s.setSingleStep(0.01)
            s.setSuffix(suffix)
            s.setFixedWidth(86)
            s.setEnabled(False)
            return s

        self._spin_tx = _tspin(" m")
        self._spin_ty = _tspin(" m")
        self._spin_tz = _tspin(" m")
        self._spin_rx = _tspin(" °"); self._spin_rx.setRange(-3600.0, 3600.0); self._spin_rx.setDecimals(2); self._spin_rx.setSingleStep(1.0)
        self._spin_ry = _tspin(" °"); self._spin_ry.setRange(-3600.0, 3600.0); self._spin_ry.setDecimals(2); self._spin_ry.setSingleStep(1.0)
        self._spin_rz = _tspin(" °"); self._spin_rz.setRange(-3600.0, 3600.0); self._spin_rz.setDecimals(2); self._spin_rz.setSingleStep(1.0)

        # Layout: two columns — translation on the left, rotation on the right.
        row_xyz = QHBoxLayout()
        row_xyz.setSpacing(2)
        row_xyz.addWidget(QLabel("Tx"))
        row_xyz.addWidget(self._spin_tx)
        row_xyz.addWidget(QLabel("Rx"))
        row_xyz.addWidget(self._spin_rx)

        row_y = QHBoxLayout()
        row_y.setSpacing(2)
        row_y.addWidget(QLabel("Ty"))
        row_y.addWidget(self._spin_ty)
        row_y.addWidget(QLabel("Ry"))
        row_y.addWidget(self._spin_ry)

        row_z = QHBoxLayout()
        row_z.setSpacing(2)
        row_z.addWidget(QLabel("Tz"))
        row_z.addWidget(self._spin_tz)
        row_z.addWidget(QLabel("Rz"))
        row_z.addWidget(self._spin_rz)

        outer.addLayout(row_xyz)
        outer.addLayout(row_y)
        outer.addLayout(row_z)

        for s in (self._spin_tx, self._spin_ty, self._spin_tz,
                  self._spin_rx, self._spin_ry, self._spin_rz):
            s.valueChanged.connect(self._emit_transform_values)

        return container

    def _emit_transform_values(self) -> None:
        self.transform_values_changed.emit(*self.get_transform_values())

    def set_circuit_enabled(self, group_ok: bool, ungroup_ok: bool) -> None:
        """Enable grouping buttons when ≥2 coils are multi-selected; enable
        the ungroup button when the current selection contains any grouped
        coils."""
        self._btn_series.set_action_enabled(group_ok)
        self._btn_parallel.set_action_enabled(group_ok)
        self._btn_ungroup.set_action_enabled(ungroup_ok)

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
            f"#ribbon_bar_widget {{"
            f"  background:{THEME['panel']};"
            f"  border-bottom:1px solid {THEME['border']};"
            f"}}"
        )
        self._tab_bar.setStyleSheet(
            f"#ribbon_tab_bar {{"
            f"  background:{THEME['bg']};"
            f"  border-bottom:1px solid {THEME['border']};"
            f"}}"
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
            # Also re-style chevron strips on dropdown buttons
            for db in tab_w.findChildren(_DropdownRibbonBtn):
                db.refresh_theme()

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
    layer_delete_requested   = pyqtSignal(str, str)         # (coil_id, layer_name)
    coil_visibility_toggled  = pyqtSignal(str, bool)
    coil_delete_requested    = pyqtSignal(str)
    coil_selected            = pyqtSignal(str)
    coils_multi_selected     = pyqtSignal(list)             # [coil_id, ...] when >1 selected
    coil_renamed             = pyqtSignal(str, str)
    coil_recolored           = pyqtSignal(str, str)         # (coil_id, hex_color)
    probe_selected           = pyqtSignal(str)                # probe_id
    probe_delete_requested   = pyqtSignal(str)                # probe_id
    probe_recolored          = pyqtSignal(str, str)           # (probe_id, hex_color)
    circuit_selected         = pyqtSignal(str)                # group_id
    # System Energy meter (singleton instrument)
    system_energy_delete_requested = pyqtSignal()
    # Stray-field probe arrays (one per array_id)
    stray_array_delete_requested   = pyqtSignal(str)

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
        # Allow Shift/⌘-click multi-selection for bulk coil edits + circuit grouping
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
            QTreeWidget::item:selected,
            QTreeWidget::item:selected:active,
            QTreeWidget::item:selected:!active {{
                background:{THEME['hi_blue']};
                color:{THEME['text']};
            }}
        """)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
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
        self._system_energy_item = None
        self._system_energy_label = None
        self._stray_array_items: dict = {}   # array_id → {tree_item, label}
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

        # group_id → {'tree_item', 'name_label', 'kind_label', 'color'}
        self._circuit_data: dict = {}

    # ── Public: circuit (folder-like) headers ─────────────────────────────────

    def add_circuit_header(self, group_id: str, name: str,
                             kind: str, color: str,
                             insert_above: list | None = None) -> None:
        """Add a circuit-family header row under COILS. Implementation note:
        Qt's ``setItemWidget`` bindings don't survive reparenting tree items
        (takeChild → addChild crashes), so we don't actually nest grouped
        coils under the header. Instead the header is a sibling of the
        coils at the COILS-group level, and member coils are visually
        grouped by a colored left border painted on their row widget.

        ``insert_above`` is an optional list of member coil IDs; when
        provided, the header is inserted at the index of the first-listed
        member so it reads naturally above its coils in the tree."""
        if group_id in self._circuit_data:
            return
        # Compute insertion index so the header lands ABOVE its first member
        insert_idx = self._coils_group.childCount()  # default: end
        if insert_above:
            for mid in insert_above:
                mcoil = self._coil_data.get(mid)
                if mcoil:
                    try:
                        idx = self._coils_group.indexOfChild(mcoil['tree_item'])
                        if idx >= 0 and idx < insert_idx:
                            insert_idx = idx
                    except Exception:
                        pass
        item = QTreeWidgetItem()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._coils_group.insertChild(insert_idx, item)

        w = QWidget()
        w.setObjectName('circuit_row')
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(4)

        bar = QLabel()
        bar.setFixedSize(4, 18)
        bar.setStyleSheet(f"background:{color}; border-radius:1px;")
        lay.addWidget(bar)

        icon = QLabel("▾")
        icon.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:9pt; padding-left:1px;"
        )
        lay.addWidget(icon)

        n_lbl = QLabel(name)
        n_lbl.setStyleSheet(
            f"color:{THEME['text']}; font-size:8pt; font-weight:600;"
        )
        lay.addWidget(n_lbl)

        k_lbl = QLabel(f"· {kind.capitalize()}")
        k_lbl.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8pt;")
        lay.addWidget(k_lbl, stretch=1)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._circuit_data[group_id] = {
            'tree_item':  item,
            'name_label': n_lbl,
            'kind_label': k_lbl,
            'color_bar':  bar,
            'color':      color,
            'members':    set(),   # coil_ids currently badged with this color
        }

    def remove_circuit_header(self, group_id: str) -> None:
        """Dissolve a circuit: clear the left-border color badge from member
        coil rows, then delete the header item."""
        entry = self._circuit_data.pop(group_id, None)
        if not entry:
            return
        for cid in entry.get('members', set()):
            self._set_coil_group_badge(cid, None)
        header = entry['tree_item']
        idx = self._coils_group.indexOfChild(header)
        if idx >= 0:
            self._coils_group.takeChild(idx)

    def move_coil_under_circuit(self, coil_id: str, group_id: str) -> None:
        """Badge a coil row with the circuit's color to visually group it.
        No tree reparenting — the coil stays as a direct child of the
        COILS group (see ``add_circuit_header`` for the design rationale)."""
        coil = self._coil_data.get(coil_id)
        grp  = self._circuit_data.get(group_id)
        if not coil or not grp:
            return
        # If the coil was in another circuit, drop it from there first
        for gid, gdata in self._circuit_data.items():
            if gid != group_id and coil_id in gdata.get('members', set()):
                gdata['members'].discard(coil_id)
        grp.setdefault('members', set()).add(coil_id)
        self._set_coil_group_badge(coil_id, grp['color'])

    def move_coil_to_root(self, coil_id: str) -> None:
        """Strip the circuit badge from a coil row (coil leaves its circuit).
        No tree reparenting — just a visual update."""
        for gid, gdata in self._circuit_data.items():
            gdata.get('members', set()).discard(coil_id)
        self._set_coil_group_badge(coil_id, None)

    def _set_coil_group_badge(self, coil_id: str, color: str | None) -> None:
        """Apply / remove a colored left-border on a coil row widget to
        indicate circuit-family membership. The selector is scoped to
        ``#coil_row`` so the border only appears on the row frame, not on
        every descendant widget (eye, swatch, labels, delete button)."""
        coil = self._coil_data.get(coil_id)
        if not coil:
            return
        w = coil.get('row_widget')
        if w is None:
            return
        if color:
            w.setStyleSheet(
                f"QWidget#coil_row {{ border-left:3px solid {color}; }}"
            )
        else:
            w.setStyleSheet("")

    def update_circuit_header(self, group_id: str, name: str | None = None,
                               kind: str | None = None,
                               color: str | None = None) -> None:
        entry = self._circuit_data.get(group_id)
        if not entry:
            return
        if name is not None:
            entry['name_label'].setText(name)
        if kind is not None:
            entry['kind_label'].setText(f"· {kind.capitalize()}")
        if color is not None:
            entry['color'] = color
            entry['color_bar'].setStyleSheet(
                f"background:{color}; border-radius:1px;"
            )

    # ── Public: coil items ────────────────────────────────────────────────────

    def add_coil_item(self, coil_id: str, display_name: str, color: str) -> None:
        if coil_id in self._coil_data:
            return

        item = QTreeWidgetItem(self._coils_group)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        w.setObjectName('coil_row')   # scope stylesheet to this widget only
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
            'row_widget': w,          # kept so we can re-attach after reparenting
            'name_label': n_lbl,
            'swatch':     swatch,
            'color':      color,
            'analysis':   {},         # layer_name → (child_item, eye)
            'analysis_widgets': {},   # layer_name → row_widget (for re-attach)
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
                          visible: bool = True,
                          deletable: bool = False) -> None:
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

        if deletable:
            del_btn = QPushButton("×")
            del_btn.setFixedSize(16, 16)
            del_btn.setFlat(True)
            del_btn.setObjectName('coil_del')
            del_btn.clicked.connect(
                lambda _, cid=coil_id, n=layer_name:
                    self.layer_delete_requested.emit(cid, n)
            )
            lay.addWidget(del_btn)

        self._tree.setItemWidget(child, 0, w)
        child.setSizeHint(0, QSize(0, 22))
        entry['analysis'][layer_name] = (child, eye)
        entry.setdefault('analysis_widgets', {})[layer_name] = w
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
        lbl.setToolTip(display_name)
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
            'name_label': lbl,
            'base_name': display_name,
        }

    def update_probe_parent_label(self, probe_id: str,
                                   coil_ref: str | None,
                                   mode: str) -> None:
        """Update the probe's browser label to show its parent coil and mode,
        so a reparent is obvious at a glance (e.g. 'Probe 1 → coil_1 · PCA')."""
        entry = self._probe_data.get(probe_id)
        if entry is None:
            return
        base = entry.get('base_name', 'Probe')
        lbl = entry.get('name_label')
        if lbl is None:
            return
        if coil_ref:
            text = f"{base} → {coil_ref} · {mode.upper()}"
        else:
            text = base
        lbl.setText(text)
        lbl.setToolTip(text)

    def update_probe_readout(self, probe_id: str, Bx: float, By: float, Bz: float,
                             B_mag: float, px: float, py: float, pz: float) -> None:
        entry = self._probe_data.get(probe_id)
        if entry is None:
            return
        r = entry['readout_items']
        # Pick a single unit from |B| so the four B-components stay coherent;
        # otherwise Bx might read in nT and |B| in mT and the user can't add
        # them in their head.
        unit = _b_field_unit(B_mag)
        r["Bx"].setText(_fmt_b(Bx, unit, decimals=4))
        r["By"].setText(_fmt_b(By, unit, decimals=4))
        r["Bz"].setText(_fmt_b(Bz, unit, decimals=4))
        r["|B|"].setText(_fmt_b(B_mag, unit, decimals=4))
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

    # ── System Energy meter (singleton) ─────────────────────────────────────

    def add_system_energy_item(self) -> None:
        """Insert the singleton ⚡ System Energy entry under INSTRUMENTS.

        No-op if already present. Reads as "⚡ System Energy: —" until
        ProjectView calls update_system_energy_readout() with a numeric value.
        """
        if getattr(self, '_system_energy_item', None) is not None:
            return
        self._instruments_group.setHidden(False)
        item = QTreeWidgetItem(self._instruments_group)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 2, 0)
        lay.setSpacing(4)

        sw = QLabel("⚡")
        sw.setStyleSheet(
            f"color:{THEME.get('accent', '#ffaa00')}; font-size:10pt;"
        )
        sw.setFixedWidth(14)
        lay.addWidget(sw)

        lbl = QLabel("System Energy: —")
        lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        lbl.setToolTip("Total stored magnetic energy ½·Iᵀ·L·I across every coil")
        lay.addWidget(lbl, stretch=1)

        del_btn = QPushButton("×")
        del_btn.setFixedSize(16, 16)
        del_btn.setFlat(True)
        del_btn.setObjectName('coil_del')
        del_btn.clicked.connect(lambda _: self.system_energy_delete_requested.emit())
        lay.addWidget(del_btn)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._instruments_group.setExpanded(True)

        self._system_energy_item = item
        self._system_energy_label = lbl

    def update_system_energy_readout(self, text: str) -> None:
        """Set the displayed text on the System Energy meter (no-op if absent)."""
        lbl = getattr(self, '_system_energy_label', None)
        if lbl is not None:
            lbl.setText(f"System Energy: {text}")

    def remove_system_energy_item(self) -> None:
        """Remove the System Energy meter from the browser (no-op if absent)."""
        item = getattr(self, '_system_energy_item', None)
        if item is None:
            return
        self._instruments_group.removeChild(item)
        self._system_energy_item = None
        self._system_energy_label = None
        if self._instruments_group.childCount() == 0:
            self._instruments_group.setHidden(True)

    def has_system_energy_item(self) -> bool:
        return getattr(self, '_system_energy_item', None) is not None

    # ── Stray-field probe arrays (one row per array) ────────────────────────

    def add_stray_array_item(self, array_id: str, name: str,
                              color: str = '#80c0ff') -> None:
        """Insert a stray-field array entry under INSTRUMENTS."""
        if array_id in self._stray_array_items:
            return
        self._instruments_group.setHidden(False)
        item = QTreeWidgetItem(self._instruments_group)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 2, 0)
        lay.setSpacing(4)

        sw = QLabel("📊")
        sw.setStyleSheet(f"color:{color}; font-size:10pt;")
        sw.setFixedWidth(14)
        lay.addWidget(sw)

        lbl = QLabel(f"{name}: —")
        lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")
        lbl.setToolTip(
            "Stray-field probe array. Readout is sqrt(mean(|B|²)) across "
            "every point — the B²_stray metric used by TEAM 22's OF."
        )
        lay.addWidget(lbl, stretch=1)

        del_btn = QPushButton("×")
        del_btn.setFixedSize(16, 16)
        del_btn.setFlat(True)
        del_btn.setObjectName('coil_del')
        del_btn.clicked.connect(
            lambda _, aid=array_id: self.stray_array_delete_requested.emit(aid)
        )
        lay.addWidget(del_btn)

        self._tree.setItemWidget(item, 0, w)
        item.setSizeHint(0, QSize(0, 22))
        self._instruments_group.setExpanded(True)

        self._stray_array_items[array_id] = {
            'tree_item': item,
            'label': lbl,
            'base_name': name,
        }

    def update_stray_array_readout(self, array_id: str, text: str) -> None:
        entry = self._stray_array_items.get(array_id)
        if entry is None:
            return
        entry['label'].setText(f"{entry['base_name']}: {text}")

    def remove_stray_array_item(self, array_id: str) -> None:
        entry = self._stray_array_items.pop(array_id, None)
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
        # Circuit folder header
        for gid, gdata in self._circuit_data.items():
            if gdata['tree_item'] is item:
                self.circuit_selected.emit(gid)
                return
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

    def _on_selection_changed(self) -> None:
        """Authoritative view of the current multi-selection. Emits
        ``coils_multi_selected`` with the list of selected coil IDs whenever
        the set of selected rows changes — including the single-item and
        empty cases (MainWindow uses this to clear multi-edit state only
        when the selection truly collapses, not on every coil click, so a
        shift-built selection survives navigation elsewhere).

        Selecting a circuit folder header counts as selecting all of its
        member coils, so a user can group/regroup or bulk-edit whole
        circuits at once."""
        selected = self._tree.selectedItems()
        coil_ids: list[str] = []
        seen: set = set()
        for item in selected:
            # Direct coil row
            for cid, data in self._coil_data.items():
                if data['tree_item'] is item:
                    if not cid.startswith('bobbin_') and cid not in seen:
                        coil_ids.append(cid); seen.add(cid)
                    break
            # Circuit header → expand to its tracked member coils
            for gid, gdata in self._circuit_data.items():
                if gdata['tree_item'] is item:
                    for cid2 in gdata.get('members', set()):
                        if cid2 not in seen and cid2 in self._coil_data:
                            coil_ids.append(cid2); seen.add(cid2)
                    break
        self.coils_multi_selected.emit(coil_ids)

    def selected_coil_ids(self) -> list:
        """Return the list of currently selected coil IDs (excluding bobbins).
        Ordered as in selection; single-selection returns a 1-list."""
        out: list = []
        for item in self._tree.selectedItems():
            for cid, data in self._coil_data.items():
                if data['tree_item'] is item and not cid.startswith('bobbin_'):
                    out.append(cid)
                    break
        return out

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

    # Emitted when the user starts/stops editing the Current spinbox.
    # ProjectView uses these to show/hide red current-direction arrows on
    # the active coil so the polarity convention is unambiguous before run.
    current_edit_started  = pyqtSignal()
    current_edit_finished = pyqtSignal()

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

        # ── Coil parameters container (hidden when a probe is selected) ──
        self._coil_w = QWidget()
        coil_lay = QVBoxLayout(self._coil_w)
        coil_lay.setContentsMargins(0, 0, 0, 0)
        coil_lay.setSpacing(4)

        # Multi-edit banner — shown when >1 coil is selected in the browser
        self._multi_banner = QLabel()
        self._multi_banner.setStyleSheet(
            f"background:{THEME.get('hi_blue', '#3b5a8a')}; "
            f"color:{THEME['text']}; font-size:8pt; font-weight:600; "
            f"padding:3px 6px; border-radius:2px;"
        )
        self._multi_banner.setAlignment(Qt.AlignCenter)
        self._multi_banner.hide()
        coil_lay.addWidget(self._multi_banner)

        # Circuit banner — shown when the active coil is part of a wired circuit
        self._circuit_banner = QLabel()
        self._circuit_banner.setStyleSheet(
            f"color:{THEME['text']}; font-size:8pt; font-weight:600; "
            f"padding:2px 4px; border-left:3px solid "
            f"{THEME.get('accent', '#e06a2a')};"
        )
        self._circuit_banner.hide()
        coil_lay.addWidget(self._circuit_banner)

        # Coil parameter form
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(4)
        form.setContentsMargins(0, 2, 0, 0)

        self.spin_winds    = QSpinBox()
        self.spin_winds.setRange(1, 10_000);   self.spin_winds.setValue(200)

        self.dspin_current = QDoubleSpinBox()
        # Allow negative currents — needed for SMES/TEAM-22-style configurations
        # where one coil's current flows opposite to the other (J = ±22.5 A/mm²).
        self.dspin_current.setRange(-1e6, 1e6); self.dspin_current.setDecimals(1);  self.dspin_current.setValue(300.0)
        # Focus on this spinbox flips on the red current-direction arrows in
        # the workspace; defocus removes them.
        self.dspin_current.installEventFilter(self)

        # Tape thickness — value + unit selector. Internal canonical unit is
        # µm (kept that way so downstream physics doesn't change). Conversion
        # tuples: (label, factor_to_µm).
        self._thick_units = [('µm', 1.0), ('mm', 1e3), ('cm', 1e4), ('m', 1e6)]
        self.dspin_thick   = QDoubleSpinBox()
        self.dspin_thick.setRange(0, 1e6);     self.dspin_thick.setDecimals(4);    self.dspin_thick.setValue(80.0)
        self.cmb_thick_unit = QComboBox()
        for lbl, _ in self._thick_units: self.cmb_thick_unit.addItem(lbl)
        self.cmb_thick_unit.setCurrentIndex(0)
        self._thick_unit_idx = 0
        self.cmb_thick_unit.currentIndexChanged.connect(self._on_thick_unit_changed)
        thick_row = QHBoxLayout()
        thick_row.setContentsMargins(0, 0, 0, 0); thick_row.setSpacing(2)
        thick_row.addWidget(self.dspin_thick); thick_row.addWidget(self.cmb_thick_unit)
        thick_w = QWidget(); thick_w.setLayout(thick_row)

        # Tape width — same pattern. Canonical unit is mm.
        self._width_units = [('mm', 1.0), ('cm', 10.0), ('m', 1000.0)]
        self.dspin_width   = QDoubleSpinBox()
        # Range expanded so TEAM-22-scale solenoids (h up to ~1.6 m = 1600 mm) fit.
        self.dspin_width.setRange(0.0001, 1e6); self.dspin_width.setDecimals(4); self.dspin_width.setValue(4.0)
        self.cmb_width_unit = QComboBox()
        for lbl, _ in self._width_units: self.cmb_width_unit.addItem(lbl)
        self.cmb_width_unit.setCurrentIndex(0)
        self._width_unit_idx = 0
        self.cmb_width_unit.currentIndexChanged.connect(self._on_width_unit_changed)
        width_row = QHBoxLayout()
        width_row.setContentsMargins(0, 0, 0, 0); width_row.setSpacing(2)
        width_row.addWidget(self.dspin_width); width_row.addWidget(self.cmb_width_unit)
        width_w = QWidget(); width_w.setLayout(width_row)

        # Stack growth direction — symmetric (centerline at middle of pack)
        # or up from centerline (centerline at bottom edge). Default is 'up'
        # because that's the convention used by every import path
        # (CSV/STEP/bobbin/.bobsx) and matches manufactured-tape-on-bobbin
        # geometry. The TEAM 22 symmetric case explicitly overrides via the
        # saved .calcsx file.
        self.cmb_stack = QComboBox()
        self.cmb_stack.addItems(["Symmetric", "Up from centerline"])
        self.cmb_stack.setCurrentIndex(1)

        self.spin_axis_pts = QSpinBox()
        self.spin_axis_pts.setRange(50, 1000); self.spin_axis_pts.setSingleStep(50); self.spin_axis_pts.setValue(200)

        form.addRow("Winds:",        self.spin_winds)
        form.addRow("Current (A):",  self.dspin_current)
        form.addRow("Tape Thick:",   thick_w)
        form.addRow("Tape Width:",   width_w)
        form.addRow("Stack:",        self.cmb_stack)
        form.addRow("Axis Samples:", self.spin_axis_pts)
        coil_lay.addLayout(form)
        # Keep a reference to the "Current (A):" label so we can hide the
        # whole row (label + spinbox) when the coil is driven by a circuit.
        self._current_form_label = form.labelForField(self.dspin_current)

        # Inherited-current note (shown INSTEAD of the Current row when the
        # coil belongs to a circuit family — current is owned by the circuit)
        self._inherited_current_lbl = QLabel()
        self._inherited_current_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt; "
            f"padding:2px 4px; font-style:italic;"
        )
        self._inherited_current_lbl.setWordWrap(True)
        self._inherited_current_lbl.hide()
        coil_lay.addWidget(self._inherited_current_lbl)

        # INSPECT — field line seeds
        coil_lay.addWidget(_hdivider())
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
        coil_lay.addLayout(irow)

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
        coil_lay.addLayout(srow)

        lay.addWidget(self._coil_w)

        # ── Probe controls (shown only when a Hall probe is selected) ──
        self._build_probe_controls()
        lay.addWidget(self._probe_w)
        self._probe_w.hide()

        # ── Circuit controls (shown only when a circuit header is selected) ──
        self._build_circuit_controls()
        lay.addWidget(self._circuit_w)
        self._circuit_w.hide()

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
        self._sum_rows:  dict = {}   # key → row QWidget (for show/hide)
        for key in ("B cent.", "B axial", "Peak |B|", "Peak F", "Peak σ", "Arc len.",
                    "Induct.", "Circuit L", "Energy"):
            row_w = QWidget()
            row = QHBoxLayout(row_w)
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
            self._sum_rows[key] = row_w
            row.addWidget(kl)
            row.addWidget(vl, stretch=1)
            sl.addWidget(row_w)

        lay.addWidget(self._sum_w)

        lay.addStretch()

    # ── Probe controls ────────────────────────────────────────────────────────

    probe_position_changed = pyqtSignal(float, float, float)  # x, y, z (world, metres)
    probe_pca_changed      = pyqtSignal(float, float, float)  # u, v, w offsets along PCA axes 1, 2, 3
    probe_mode_changed     = pyqtSignal(str)                  # 'xyz' or 'pca'

    def _build_probe_controls(self) -> None:
        self._probe_w = QWidget()
        pv_lay = QVBoxLayout(self._probe_w)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(4)

        hdr = _section_lbl("HALL PROBE")
        self._probe_hdr = hdr
        pv_lay.addWidget(hdr)

        self._probe_coil_lbl = QLabel("Coil: —")
        self._probe_coil_lbl.setObjectName('dim_label')
        pv_lay.addWidget(self._probe_coil_lbl)

        # Mode toggle (mutually exclusive radios)
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(6)
        self._probe_mode_grp = QButtonGroup(self._probe_w)
        self._probe_xyz_radio = QRadioButton("XYZ")
        self._probe_pca_radio = QRadioButton("PCA axes")
        self._probe_mode_grp.addButton(self._probe_xyz_radio, 0)
        self._probe_mode_grp.addButton(self._probe_pca_radio, 1)
        self._probe_xyz_radio.setChecked(True)
        mode_row.addWidget(self._probe_xyz_radio)
        mode_row.addWidget(self._probe_pca_radio)
        pv_lay.addLayout(mode_row)

        def _mk_coord():
            s = QDoubleSpinBox()
            s.setRange(-1e9, 1e9)     # effectively unbounded
            s.setDecimals(4)
            s.setSingleStep(0.01)
            return s

        # XYZ spinboxes
        self._probe_xyz_w = QWidget()
        xyz_form = QFormLayout(self._probe_xyz_w)
        xyz_form.setLabelAlignment(Qt.AlignRight)
        xyz_form.setSpacing(4)
        xyz_form.setContentsMargins(0, 0, 0, 0)
        self.dspin_probe_x = _mk_coord()
        self.dspin_probe_y = _mk_coord()
        self.dspin_probe_z = _mk_coord()
        xyz_form.addRow("X (m):", self.dspin_probe_x)
        xyz_form.addRow("Y (m):", self.dspin_probe_y)
        xyz_form.addRow("Z (m):", self.dspin_probe_z)
        pv_lay.addWidget(self._probe_xyz_w)

        # PCA offset spinboxes (U along axis 1, V along axis 2, W along axis 3)
        self._probe_pca_w = QWidget()
        pca_form = QFormLayout(self._probe_pca_w)
        pca_form.setLabelAlignment(Qt.AlignRight)
        pca_form.setSpacing(4)
        pca_form.setContentsMargins(0, 0, 0, 0)
        self.dspin_probe_u = _mk_coord()
        self.dspin_probe_v = _mk_coord()
        self.dspin_probe_w = _mk_coord()
        pca_form.addRow("Axis 1 (m):", self.dspin_probe_u)
        pca_form.addRow("Axis 2 (m):", self.dspin_probe_v)
        pca_form.addRow("Axis 3 (m):", self.dspin_probe_w)
        pv_lay.addWidget(self._probe_pca_w)
        self._probe_pca_w.hide()

        # Wire signals
        self._probe_xyz_radio.toggled.connect(self._on_probe_mode_toggled)
        for s in (self.dspin_probe_x, self.dspin_probe_y, self.dspin_probe_z):
            s.valueChanged.connect(self._emit_probe_xyz)
        for s in (self.dspin_probe_u, self.dspin_probe_v, self.dspin_probe_w):
            s.valueChanged.connect(self._emit_probe_pca)

    def _on_probe_mode_toggled(self, xyz_checked: bool) -> None:
        mode = 'xyz' if xyz_checked else 'pca'
        self._probe_xyz_w.setVisible(mode == 'xyz')
        self._probe_pca_w.setVisible(mode == 'pca')
        self.probe_mode_changed.emit(mode)

    def _emit_probe_xyz(self) -> None:
        self.probe_position_changed.emit(
            float(self.dspin_probe_x.value()),
            float(self.dspin_probe_y.value()),
            float(self.dspin_probe_z.value()),
        )

    def _emit_probe_pca(self) -> None:
        self.probe_pca_changed.emit(
            float(self.dspin_probe_u.value()),
            float(self.dspin_probe_v.value()),
            float(self.dspin_probe_w.value()),
        )

    def show_probe_controls(self, position, mode: str,
                             coil_ref: str | None,
                             uvw: tuple) -> None:
        """Populate + reveal the probe-controls panel."""
        # Swap panels: show probe UI, hide coil UI, hide summary
        self._coil_w.hide()
        self._probe_w.show()
        self._sum_w.hide()

        controls = (self.dspin_probe_x, self.dspin_probe_y, self.dspin_probe_z,
                    self.dspin_probe_u, self.dspin_probe_v, self.dspin_probe_w,
                    self._probe_xyz_radio, self._probe_pca_radio)
        for s in controls:
            s.blockSignals(True)
        try:
            if position is not None:
                self.dspin_probe_x.setValue(float(position[0]))
                self.dspin_probe_y.setValue(float(position[1]))
                self.dspin_probe_z.setValue(float(position[2]))
            if uvw is not None:
                self.dspin_probe_u.setValue(float(uvw[0]))
                self.dspin_probe_v.setValue(float(uvw[1]))
                self.dspin_probe_w.setValue(float(uvw[2]))
            pca_on = (mode == 'pca')
            self._probe_xyz_radio.setChecked(not pca_on)
            self._probe_pca_radio.setChecked(pca_on)
            self._probe_xyz_w.setVisible(not pca_on)
            self._probe_pca_w.setVisible(pca_on)
            if coil_ref:
                mode_tag = "PCA" if pca_on else "XYZ"
                self._probe_coil_lbl.setText(
                    f"Parent: {coil_ref}  •  {mode_tag}"
                )
            else:
                self._probe_coil_lbl.setText("Parent: — (select a coil first)")
            self._probe_pca_radio.setEnabled(coil_ref is not None)
        finally:
            for s in controls:
                s.blockSignals(False)

    # ── Circuit controls ──────────────────────────────────────────────────

    circuit_current_changed  = pyqtSignal(str, float)   # (group_id, amps)
    circuit_renamed          = pyqtSignal(str, str)     # (group_id, new_name)

    def _build_circuit_controls(self) -> None:
        self._circuit_w = QWidget()
        cv = QVBoxLayout(self._circuit_w)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)

        self._circuit_hdr = _section_lbl("CIRCUIT")
        cv.addWidget(self._circuit_hdr)

        self._cv_name_lbl = QLabel("—")
        self._cv_name_lbl.setStyleSheet(
            f"color:{THEME['text']}; font-size:10pt; font-weight:600; "
            f"padding:2px 0;"
        )
        cv.addWidget(self._cv_name_lbl)

        self._cv_kind_lbl = QLabel("")
        self._cv_kind_lbl.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:8pt;"
        )
        cv.addWidget(self._cv_kind_lbl)

        cf = QFormLayout()
        cf.setLabelAlignment(Qt.AlignRight)
        cf.setSpacing(4)
        cf.setContentsMargins(0, 4, 0, 0)
        self._cv_dspin_current = QDoubleSpinBox()
        self._cv_dspin_current.setRange(-1e6, 1e6)
        self._cv_dspin_current.setDecimals(1)
        self._cv_dspin_current.setValue(0.0)
        self._cv_dspin_current.valueChanged.connect(self._emit_cv_current)
        cf.addRow("Current (A):", self._cv_dspin_current)
        cv.addLayout(cf)

        cv.addWidget(_hdivider())
        self._cv_members_hdr = _section_lbl("MEMBERS")
        cv.addWidget(self._cv_members_hdr)
        self._cv_members_lbl = QLabel("")
        self._cv_members_lbl.setStyleSheet(
            f"color:{THEME['text']}; font-size:8pt; padding:2px 0;"
        )
        self._cv_members_lbl.setWordWrap(True)
        cv.addWidget(self._cv_members_lbl)

        cv.addWidget(_hdivider())
        self._cv_L_hdr = _section_lbl("CIRCUIT INDUCTANCE")
        cv.addWidget(self._cv_L_hdr)
        self._cv_L_lbl = QLabel("—")
        self._cv_L_lbl.setStyleSheet(
            f"color:{THEME['accent']}; font-size:10pt; font-weight:bold;"
        )
        cv.addWidget(self._cv_L_lbl)

        self._cv_group_id: str | None = None

    def _emit_cv_current(self, amps: float) -> None:
        if self._cv_group_id is not None:
            self.circuit_current_changed.emit(self._cv_group_id, float(amps))

    def show_circuit_view(self, group_id: str, group_info: dict,
                           member_names: list, L_henries: float | None) -> None:
        """Show the circuit-level Properties view for `group_id`."""
        self._probe_w.hide()
        self._coil_w.hide()
        self._circuit_w.show()
        self._sum_w.hide()
        self._cv_group_id = group_id
        self._cv_name_lbl.setText(group_info.get('name', group_id))
        kind = group_info.get('kind', 'series').capitalize()
        color = group_info.get('color', THEME.get('accent', '#e06a2a'))
        self._cv_kind_lbl.setText(f"{kind} wiring")
        # Accent the header with the group's color
        self._circuit_hdr.setStyleSheet(
            f"color:{color}; font-size:7pt; font-weight:600; letter-spacing:2px;"
        )
        # Block-signal-set the current spinbox to the group's value
        self._cv_dspin_current.blockSignals(True)
        self._cv_dspin_current.setValue(float(group_info.get('current', 0.0)))
        self._cv_dspin_current.blockSignals(False)
        # Member list
        n = len(member_names)
        if member_names:
            text = f"({n} coils)  " + ", ".join(member_names)
        else:
            text = "(no members)"
        self._cv_members_lbl.setText(text)
        # Inductance
        self.update_circuit_inductance(group_id, L_henries)

    def update_circuit_inductance(self, group_id: str,
                                    L_henries: float | None) -> None:
        """Update just the circuit-view L reading. Used by the background
        L-matrix worker to swap the "computing…" placeholder for the real
        value without re-rendering the whole circuit panel."""
        if getattr(self, '_cv_group_id', None) != group_id:
            return
        if L_henries is None:
            self._cv_L_lbl.setText("computing…")
        elif L_henries >= 1.0:
            self._cv_L_lbl.setText(f"{L_henries:.3f} H")
        elif L_henries >= 1e-3:
            self._cv_L_lbl.setText(f"{L_henries*1e3:.3f} mH")
        else:
            self._cv_L_lbl.setText(f"{L_henries*1e6:.2f} µH")

    def set_coil_current_editable(self, editable: bool,
                                    inherited_from: str | None = None,
                                    inherited_value: float | None = None) -> None:
        """When a coil is in a circuit, the Current row must not be edited
        per-coil (it's driven by the circuit). Hide the spinbox + label and
        show an informational note instead."""
        if editable:
            self._current_form_label.setVisible(True)
            self.dspin_current.setVisible(True)
            self.dspin_current.setEnabled(True)
            self._inherited_current_lbl.hide()
        else:
            self._current_form_label.setVisible(False)
            self.dspin_current.setVisible(False)
            val = inherited_value if inherited_value is not None else 0.0
            src = inherited_from or "circuit"
            self._inherited_current_lbl.setText(
                f"Current {val:.1f} A — inherited from {src}"
            )
            self._inherited_current_lbl.show()

    def show_coil_controls(self) -> None:
        """Swap back to coil-params view (summary visibility is caller's choice)."""
        self._probe_w.hide()
        self._circuit_w.hide()
        self._coil_w.show()

    def set_circuit_banner(self, group_info: dict | None) -> None:
        """Show a "Circuit: NAME (kind, N coils)" banner at the top of the
        Properties panel when the active coil belongs to a circuit group.
        Pass None to hide."""
        if not group_info:
            self._circuit_banner.hide()
            return
        name = group_info.get('name', 'Circuit')
        kind = group_info.get('kind', 'series').capitalize()
        n    = len(group_info.get('coil_ids', []))
        color = group_info.get('color', THEME.get('accent', '#e06a2a'))
        self._circuit_banner.setStyleSheet(
            f"color:{THEME['text']}; font-size:8pt; font-weight:600; "
            f"padding:2px 4px; border-left:3px solid {color};"
        )
        self._circuit_banner.setText(f"  {name}  •  {kind}  •  {n} coils")
        self._circuit_banner.show()

    def set_summary_row_visible(self, key: str, visible: bool) -> None:
        """Show or hide a single row of the results summary by key name.
        Used to hide per-coil Induct. when a coil is part of a circuit
        family (the circuit's L is the meaningful value there)."""
        row = self._sum_rows.get(key)
        if row is not None:
            row.setVisible(visible)

    def set_multi_edit_banner(self, n_coils: int, mixed_keys: list | None = None) -> None:
        """Show a banner indicating that multiple coils are being edited in
        bulk. Pass n_coils=0 (or 1) to clear. mixed_keys lists parameter names
        whose values differ across the selection; the banner names them so the
        user knows what will be overwritten on any spinbox change."""
        if n_coils < 2:
            self._multi_banner.hide()
            return
        if mixed_keys:
            detail = "  •  differs: " + ", ".join(mixed_keys)
        else:
            detail = ""
        self._multi_banner.setText(f"Editing {n_coils} coils{detail}")
        self._multi_banner.show()

    def show_bobbin_view(self) -> None:
        """Bobbins are visual-only imports — hide all physics parameter
        controls and the results summary. Browser entry stays interactive
        (name, colour, visibility, delete, transform)."""
        self._probe_w.hide()
        self._coil_w.hide()
        self._circuit_w.hide()
        self._sum_w.hide()

    def update_probe_position_display(self, position) -> None:
        """Live-update the XYZ spinboxes without emitting signals (used when
        gizmo dragging moves the probe)."""
        if position is None:
            return
        for s in (self.dspin_probe_x, self.dspin_probe_y, self.dspin_probe_z):
            s.blockSignals(True)
        try:
            self.dspin_probe_x.setValue(float(position[0]))
            self.dspin_probe_y.setValue(float(position[1]))
            self.dspin_probe_z.setValue(float(position[2]))
        finally:
            for s in (self.dspin_probe_x, self.dspin_probe_y, self.dspin_probe_z):
                s.blockSignals(False)

    def update_probe_pca_display(self, u: float, v: float, w: float) -> None:
        """Update the PCA offset spinboxes without emitting signals."""
        for s in (self.dspin_probe_u, self.dspin_probe_v, self.dspin_probe_w):
            s.blockSignals(True)
        try:
            self.dspin_probe_u.setValue(float(u))
            self.dspin_probe_v.setValue(float(v))
            self.dspin_probe_w.setValue(float(w))
        finally:
            for s in (self.dspin_probe_u, self.dspin_probe_v, self.dspin_probe_w):
                s.blockSignals(False)

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

    def eventFilter(self, obj, event):
        # Focus in/out on dspin_current → emit edit-started/finished signals
        # so ProjectView can show/hide the red current-direction arrows.
        # QScrollArea (our base) routes its own viewport / scrollbar events
        # through eventFilter during setup — before dspin_current is bound —
        # so we look it up defensively rather than asserting attribute
        # existence.
        target = getattr(self, 'dspin_current', None)
        if target is not None and obj is target:
            from PyQt5.QtCore import QEvent
            if event.type() == QEvent.FocusIn:
                self.current_edit_started.emit()
            elif event.type() == QEvent.FocusOut:
                self.current_edit_finished.emit()
        return super().eventFilter(obj, event)

    def get_params(self) -> dict:
        # Convert displayed thickness/width back to canonical units (µm, mm)
        # so the rest of the pipeline (workspace mesh build, AnalysisWorker,
        # superposition, save/load) keeps its existing assumptions.
        t_factor = self._thick_units[self._thick_unit_idx][1]   # µm per displayed unit
        w_factor = self._width_units[self._width_unit_idx][1]   # mm per displayed unit
        return {
            'winds':     self.spin_winds.value(),
            'current':   self.dspin_current.value(),
            'thickness': self.dspin_thick.value() * t_factor,
            'width':     self.dspin_width.value() * w_factor,
            'axis_num':  self.spin_axis_pts.value(),
            # Display preferences — preserved across save/load so the user's
            # chosen units stick.
            'thickness_unit': self._thick_unit_idx,
            'width_unit':     self._width_unit_idx,
            'stack_growth':   'up' if self.cmb_stack.currentIndex() == 1 else 'symmetric',
        }

    def _on_thick_unit_changed(self, new_idx: int) -> None:
        """Convert displayed thickness so the physical value (in µm) is preserved."""
        old_factor = self._thick_units[self._thick_unit_idx][1]
        new_factor = self._thick_units[new_idx][1]
        canonical_um = self.dspin_thick.value() * old_factor
        self._thick_unit_idx = new_idx
        self.dspin_thick.blockSignals(True)
        self.dspin_thick.setValue(canonical_um / new_factor)
        self.dspin_thick.blockSignals(False)
        # Trigger the param-changed pipeline so canonical thickness is re-read
        # through get_params (it's unchanged, but the stored snapshot needs to
        # know the new unit choice).
        self.dspin_thick.valueChanged.emit(self.dspin_thick.value())

    def _on_width_unit_changed(self, new_idx: int) -> None:
        """Convert displayed width so the physical value (in mm) is preserved."""
        old_factor = self._width_units[self._width_unit_idx][1]
        new_factor = self._width_units[new_idx][1]
        canonical_mm = self.dspin_width.value() * old_factor
        self._width_unit_idx = new_idx
        self.dspin_width.blockSignals(True)
        self.dspin_width.setValue(canonical_mm / new_factor)
        self.dspin_width.blockSignals(False)
        self.dspin_width.valueChanged.emit(self.dspin_width.value())

    def get_field_seeds(self) -> int:
        return self.spin_field_seeds.value()

    def get_cs_offset(self) -> float:
        return self.dspin_cs_offset.value()

    def update_summary(self, engine) -> None:
        def _set(key, text):
            self._sum_lbls[key].setText(text)
        try:  _set("B cent.",  _fmt_b(engine.B_magnitude))
        except Exception: pass
        try:  _set("B axial",  _fmt_b(engine.B_axial))
        except Exception: pass
        try:
            if engine.bfield_axis_mag is not None:
                mag = np.asarray(engine.bfield_axis_mag)
                z   = np.asarray(engine.bfield_axis_z)
                idx = int(np.argmax(mag))
                _set("Peak |B|", f"{_fmt_b(mag[idx])} @ {z[idx]:+.3f} m")
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
        def _fmt_L(L: float) -> str:
            if L >= 1.0:
                return f"{L:.3f} H"
            if L >= 1e-3:
                return f"{L*1e3:.3f} mH"
            return f"{L*1e6:.2f} µH"
        try:
            if engine.self_inductance is not None:
                _set("Induct.", _fmt_L(engine.self_inductance))
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

        from CalcSX_app.gui.gui_utils import get_theme_name
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
        self._combo.addItems(["Loop", "Solenoid", "Princeton Dee", "Saddle", "CCT"])
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
        # Build params for whichever shape is currently selected in the combo
        # (not hardcoded to "Solenoid") so adding a new first-position shape
        # doesn't orphan the form against the combo's default.
        self._build_params(self._combo.currentText())

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

        if shape == "Loop":
            # Flat single circle; axial height is supplied later via the
            # Properties panel (tape_width) + volumetric model.
            _dspin("Radius (m):", 2.0, 0.001, 50.0, suffix=" m")
            _ispin("Points:", 400, 50, 5000, step=50)
        elif shape == "Solenoid":
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
        from CalcSX_app.physics.geometry import (
            generate_solenoid, generate_princeton_dee,
            generate_saddle_coil, generate_cct, generate_circular_loop,
        )
        shape = self._combo.currentText()
        s = self._spins
        try:
            if shape == "Loop":
                self._coords = generate_circular_loop(
                    radius=s["Radius (m):"].value(),
                    n_pts=s["Points:"].value(),
                )
            elif shape == "Solenoid":
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


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CalcSX™ – v{__version__}")
        self.setWindowIcon(get_app_icon())
        self.resize(1280, 800)

        # ── Central widget: ribbon + tab bar + stack of ProjectViews ────────
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        self.ribbon = RibbonBar()
        main_lay.addWidget(self.ribbon)

        # Tab bar sits between ribbon and content area. QTabBar is standalone
        # (no setCornerWidget — that's QTabWidget), so pair it with a trailing
        # "+" QToolButton inside an HBox.
        tab_row = QWidget()
        tab_row_lay = QHBoxLayout(tab_row)
        tab_row_lay.setContentsMargins(0, 0, 0, 0)
        tab_row_lay.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setTabsClosable(True)
        self._tab_bar.setExpanding(False)
        self._tab_bar.setDrawBase(False)
        self._tab_bar.setDocumentMode(True)
        self._tab_bar.setUsesScrollButtons(True)
        self._tab_bar.setMovable(True)
        self._tab_bar.tabCloseRequested.connect(self._on_close_project)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        self._tab_bar.tabMoved.connect(self._on_tab_moved)
        self._tab_bar.tabBarDoubleClicked.connect(self._on_rename_project)
        tab_row_lay.addWidget(self._tab_bar)

        new_btn = QToolButton()
        new_btn.setText("+")
        new_btn.setAutoRaise(True)
        new_btn.setToolTip("New Project")
        new_btn.clicked.connect(lambda: self._new_project())
        tab_row_lay.addWidget(new_btn)
        tab_row_lay.addStretch(1)

        main_lay.addWidget(tab_row)

        # Stack holds the actual ProjectView widgets; the tab bar drives which
        # one is shown.
        self._stack = QStackedWidget()
        main_lay.addWidget(self._stack, stretch=1)

        # Per-tab project list; kept in lockstep with _stack + _tab_bar.
        self._projects: list = []

        # ── Ribbon → active project (resolved at emit time) ─────────────────
        # Each connection goes through self.current_project so a tab switch
        # automatically routes subsequent ribbon emits to the newly active
        # project without needing to disconnect/reconnect.
        self.ribbon.load_csv.connect(lambda: self.current_project._on_load_csv())
        self.ribbon.import_bobbin.connect(lambda: self.current_project._on_import_bobbin())
        self.ribbon.run_analysis.connect(lambda: self.current_project._on_run_analysis())
        self.ribbon.reanalyze_all.connect(lambda: self.current_project._on_reanalyze_all())
        self.ribbon.compute_field_lines.connect(lambda: self.current_project._on_compute_field_lines())
        self.ribbon.compute_cross_section.connect(lambda: self.current_project._on_compute_cross_section())
        self.ribbon.global_field_toggled.connect(lambda checked: self.current_project._on_global_field_toggled(checked))
        self.ribbon.translate_toggled.connect(lambda checked: self.current_project._on_translate_toggled(checked))
        self.ribbon.rotate_toggled.connect(lambda checked: self.current_project._on_rotate_toggled(checked))
        self.ribbon.reset_transform.connect(lambda: self.current_project._on_reset_transform())
        self.ribbon.generate_coil.connect(lambda: self.current_project._on_generate_coil())
        self.ribbon.group_as_series.connect(lambda: self.current_project._on_group_as_series())
        self.ribbon.group_as_parallel.connect(lambda: self.current_project._on_group_as_parallel())
        self.ribbon.ungroup_selection.connect(lambda: self.current_project._on_ungroup_selection())
        self.ribbon.add_hall_probe.connect(lambda: self.current_project._on_add_hall_probe())
        self.ribbon.add_system_energy.connect(lambda: self.current_project._on_add_system_energy())
        self.ribbon.add_stray_array.connect(lambda: self.current_project._on_add_stray_array())
        self.ribbon.pin_toggled.connect(lambda checked: self.current_project._on_pin_toggled(checked))
        self.ribbon.relative_distance_requested.connect(lambda: self.current_project._on_relative_distance())
        self.ribbon.transform_values_changed.connect(
            lambda tx, ty, tz, rx, ry, rz:
            self.current_project._on_transform_values_changed(tx, ty, tz, rx, ry, rz)
        )
        self.ribbon.save_session.connect(lambda: self.current_project._save_session(self))
        # Load: routes through MainWindow so it can decide "reuse current blank
        # tab" vs "spawn a new tab and load there".
        self.ribbon.load_session.connect(self._on_load_session)

        # ── Ribbon → MainWindow (global chrome) ─────────────────────────────
        self.ribbon.show_help.connect(lambda: HelpDialog(self).exec_())
        self.ribbon.open_settings.connect(self._on_open_settings)

        # Create the initial project tab (always keep ≥1 tab alive).
        self._new_project()

        # ── Keyboard shortcuts (PyQt maps Ctrl → Cmd on macOS) ──────────────
        # Cmd+T / Cmd+W / Cmd+Shift+] / Cmd+Shift+[ — standard tab gestures.
        QShortcut(QKeySequence("Ctrl+T"), self,
                  activated=lambda: self._new_project())
        QShortcut(QKeySequence("Ctrl+W"), self,
                  activated=lambda: self._on_close_project(self._tab_bar.currentIndex()))
        QShortcut(QKeySequence("Ctrl+Shift+]"), self,
                  activated=lambda: self._cycle_tab(+1))
        QShortcut(QKeySequence("Ctrl+Shift+["), self,
                  activated=lambda: self._cycle_tab(-1))

    # ── Tab / project lifecycle ────────────────────────────────────────────

    @property
    def current_project(self):
        """The ProjectView on the currently active tab.

        Never returns None under normal operation: _new_project() is always
        called during __init__ and _on_close_project() guarantees at least
        one tab remains open.
        """
        idx = self._tab_bar.currentIndex()
        if 0 <= idx < len(self._projects):
            return self._projects[idx]
        # Defensive fallback — should not happen in practice.
        return self._projects[0] if self._projects else None

    def _new_project(self):
        """Create a fresh blank ProjectView, append a tab for it, and focus it."""
        from CalcSX_app.primary.project_view import ProjectView
        proj = ProjectView(self)
        # Sequential default name; user can rename in Step 3.
        proj.name = f"Project {len(self._projects) + 1}"
        self._projects.append(proj)
        self._stack.addWidget(proj)
        # setCurrentIndex fires currentChanged → _on_tab_changed → stack sync.
        new_idx = self._tab_bar.addTab(proj.name)
        self._tab_bar.setCurrentIndex(new_idx)
        return proj

    def _on_tab_changed(self, idx: int) -> None:
        """Keep the stack in lockstep with the tab bar.

        Uses setCurrentWidget rather than setCurrentIndex so reordering
        (which shuffles the _projects list) keeps pointing at the right
        ProjectView even when the stack's own indices no longer match.
        """
        if 0 <= idx < len(self._projects):
            proj = self._projects[idx]
            self._stack.setCurrentWidget(proj)
            # macOS + multiple QtInteractor instances occasionally show a
            # black viewport on tab switch until something triggers a render.
            plotter = getattr(proj.workspace, '_plotter', None)
            if plotter is not None:
                try:
                    plotter.render()
                except Exception:
                    pass

    def _on_tab_moved(self, from_idx: int, to_idx: int) -> None:
        """Keep self._projects in sync when the user drags a tab."""
        if from_idx == to_idx:
            return
        if not (0 <= from_idx < len(self._projects)):
            return
        proj = self._projects.pop(from_idx)
        self._projects.insert(to_idx, proj)

    def _on_rename_project(self, idx: int) -> None:
        """Double-click on a tab → rename prompt."""
        if not (0 <= idx < len(self._projects)):
            return
        proj = self._projects[idx]
        new_name, ok = QInputDialog.getText(
            self, "Rename Project", "Project name:",
            text=proj.name or self._tab_bar.tabText(idx),
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        proj.name = new_name
        self._tab_bar.setTabText(idx, new_name)

    def _on_close_project(self, idx: int) -> None:
        """Close a tab.

        Blank projects (nothing loaded) close silently. Dirty projects get a
        Save / Discard / Cancel prompt. If the user picks Save and then
        cancels the save dialog, the close is aborted.

        Worker threads + probe timer are stopped cleanly before the widget
        is deleted.

        If this was the last tab, a fresh blank "Project 1" is spawned so
        the window never sits empty.
        """
        if not (0 <= idx < len(self._projects)):
            return
        proj = self._projects[idx]

        if proj.is_dirty():
            reply = QMessageBox.question(
                self, "Close Project",
                f"'{proj.name}' has unsaved scene data.\n\n"
                "Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                saved = proj._save_session(self)
                if not saved:
                    # User cancelled the file dialog — abort the close too.
                    return

        # Shut workers down cleanly before teardown.
        proj.shutdown_workers()

        self._projects.pop(idx)
        self._stack.removeWidget(proj)
        self._tab_bar.removeTab(idx)
        proj.deleteLater()

        # Never leave the window empty — spawn a fresh blank project.
        if not self._projects:
            self._new_project()

    def _cycle_tab(self, delta: int) -> None:
        """Advance (delta=+1) or retreat (-1) through the open tabs, wrapping."""
        n = self._tab_bar.count()
        if n < 2:
            return
        new_idx = (self._tab_bar.currentIndex() + delta) % n
        self._tab_bar.setCurrentIndex(new_idx)

    def closeEvent(self, event) -> None:
        """Window-close handler. Prompts before discarding unsaved scene data.

        Walks every open project; if any are dirty, shows a single Save All /
        Discard All / Cancel prompt with the dirty project names listed. On
        Save All, each dirty project's _save_session runs (with its own file
        dialog); a cancel inside any of those file dialogs aborts the whole
        close. Worker threads are shut down on every project before the
        window goes away.
        """
        dirty = [p for p in self._projects if p.is_dirty()]
        if dirty:
            names = "\n  • ".join(p.name or "(unnamed)" for p in dirty)
            reply = QMessageBox.question(
                self, "Close CalcSX",
                f"{len(dirty)} project(s) have unsaved scene data:\n\n"
                f"  • {names}\n\n"
                "Save before closing?",
                QMessageBox.SaveAll | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.SaveAll,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.SaveAll:
                for proj in dirty:
                    # Show that tab so the user sees what they're saving.
                    if proj in self._projects:
                        self._tab_bar.setCurrentIndex(self._projects.index(proj))
                    saved = proj._save_session(self)
                    if not saved:
                        # Cancelled the file dialog mid-save — abort close.
                        event.ignore()
                        return
        # Stop worker threads + probe timers on every project before teardown.
        for proj in self._projects:
            try:
                proj.shutdown_workers()
            except Exception:
                pass
        event.accept()

    # ── MainWindow-level handlers (route scene ops to the active project) ───

    def _on_load_session(self) -> None:
        """Prompt for a .calcsx and load it.

        If the current tab is blank (is_dirty() == False), load in place so
        the user doesn't accumulate orphaned "Project 1" tabs. Otherwise
        spawn a fresh tab and load there. The tab is renamed to the file's
        basename so multi-file sessions are self-documenting.
        """
        import json, os
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "CalcSX Session (*.calcsx);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        cur = self.current_project
        if cur is not None and not cur.is_dirty():
            target = cur
        else:
            target = self._new_project()

        # Rename the target tab after the file so the user can tell them apart.
        base = os.path.splitext(os.path.basename(path))[0]
        if base:
            target.name = base
            idx = self._projects.index(target)
            self._tab_bar.setTabText(idx, base)

        # Defer the heavy apply so the file dialog fully closes first.
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: target._apply_loaded_session(data))

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.theme_changed.connect(self._apply_theme)
        dlg.export_vtk_requested.connect(lambda: self.current_project._export_vtk_layers(dlg))
        dlg.export_web_layers_requested.connect(lambda: self.current_project._export_web_layers(dlg))
        dlg.exec_()

    def _apply_theme(self, name: str) -> None:
        from CalcSX_app.gui.gui_utils import apply_theme_to_app
        apply_theme_to_app(name)
        # Ribbon refreshes once; every open project refreshes its own panels
        # (Step 5 work landed here in Step 2 so multi-tab theme switching works
        # as soon as tabs exist).
        self.ribbon.refresh_theme()
        for proj in self._projects:
            proj.browser.refresh_theme()
            proj.props.refresh_theme()
            proj.workspace.apply_theme()
        # Swap window + application icon to match the new theme
        themed_icon = get_app_icon(name)
        self.setWindowIcon(themed_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(themed_icon)
        # Persist choice across launches
        from PyQt5.QtCore import QSettings
        QSettings().setValue("theme", name)


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


def _b_field_unit(magnitude_T: float) -> tuple:
    """Pick a sensible (multiplier, label) for displaying a B-field magnitude.

    Returns a (factor, unit_label) pair where ``factor * value_in_T`` yields
    the displayed number. Choosing the unit from the magnitude (not from each
    component) keeps Bx/By/Bz/|B| readable as a coherent vector.
    """
    m = abs(float(magnitude_T))
    if m >= 1.0:    return (1.0,   'T')
    if m >= 1e-3:   return (1e3,   'mT')
    if m >= 1e-6:   return (1e6,   'µT')
    if m >= 1e-9:   return (1e9,   'nT')
    if m >= 1e-12:  return (1e12,  'pT')
    return (1.0, 'T')


def _fmt_b(value_T: float, unit: tuple = None, decimals: int = 4) -> str:
    """Format a single B-field scalar in Tesla into a string with auto-units."""
    if unit is None:
        unit = _b_field_unit(value_T)
    factor, label = unit
    return f"{value_T * factor:.{decimals}f} {label}"
