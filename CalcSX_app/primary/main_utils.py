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
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QPixmap

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
    # Circuits — group/ungroup selected coils into a wired circuit
    group_as_series          = pyqtSignal()
    group_as_parallel        = pyqtSignal()
    ungroup_selection        = pyqtSignal()

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
        self._btn_series.clicked.connect(self.group_as_series)
        self._btn_parallel.clicked.connect(self.group_as_parallel)
        self._btn_ungroup.clicked.connect(self.ungroup_selection)

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
        self._cv_dspin_current.setRange(0, 1e6)
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
        from CalcSX_app.physics.geometry import (
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
        self._multi_edit_ids:    list = []          # IDs of coils in a bulk-edit selection (len>=2); empty otherwise
        # Circuit groups: each group_id → {kind: 'series'|'parallel',
        #                                   coil_ids: list,
        #                                   signs: {coil_id → ±1},
        #                                   color: hex, name: str}
        # Coils not in any group are treated as their own 1-coil circuit.
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
        self.browser.layer_delete_requested.connect(self._on_layer_delete)
        self.browser.coil_selected.connect(self._on_coil_selected)
        self.browser.coils_multi_selected.connect(self._on_coils_multi_selected)
        self.browser.coil_renamed.connect(self._on_coil_renamed)
        self.ribbon.group_as_series.connect(self._on_group_as_series)
        self.ribbon.group_as_parallel.connect(self._on_group_as_parallel)
        self.ribbon.ungroup_selection.connect(self._on_ungroup_selection)
        self.browser.circuit_selected.connect(self._on_circuit_selected)
        self.props.circuit_current_changed.connect(self._on_circuit_current_changed)
        self.browser.coil_recolored.connect(self._on_coil_recolored)
        self.ribbon.open_settings.connect(self._on_open_settings)
        self.ribbon.save_session.connect(lambda: self._save_session(self))
        self.ribbon.load_session.connect(lambda: self._load_session(self))
        self.ribbon.add_hall_probe.connect(self._on_add_hall_probe)
        self.browser.probe_selected.connect(self._on_probe_selected)
        self.browser.probe_delete_requested.connect(self._on_probe_delete)
        self.browser.probe_recolored.connect(
            lambda pid, c: self.workspace.set_probe_color(pid, c)
        )
        self.props.probe_position_changed.connect(self._on_probe_xyz_edit)
        self.props.probe_pca_changed.connect(self._on_probe_pca_edit)
        self.props.probe_mode_changed.connect(self._on_probe_mode_change)
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
            self.ribbon.set_construct_enabled(False)
            self.ribbon.set_inspect_enabled(False)
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

    def _on_coil_selected(self, coil_id: str) -> None:
        # Save current coil's spinbox values before switching.
        old_cid = self._active_coil_id
        if old_cid and old_cid in self._coil_params_map and not self._multi_edit_ids:
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
            if self.ribbon._btn_translate.isChecked():
                self.workspace.show_gizmo('T')
            elif self.ribbon._btn_rotate.isChecked():
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
        # If gizmo is active, seamlessly move it to the new coil
        if self.ribbon._btn_translate.isChecked():
            self.workspace.show_gizmo('T')
        elif self.ribbon._btn_rotate.isChecked():
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
            # Save any in-flight edits on the currently-displayed coil first
            if self._active_coil_id and self._active_coil_id in self._coil_params_map \
                    and not self._multi_edit_ids:
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
            self.ribbon.set_circuit_enabled(group_ok=True,
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
        self.ribbon.set_circuit_enabled(group_ok=False, ungroup_ok=grouped)

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
        self.ribbon.set_circuit_enabled(group_ok=False, ungroup_ok=True)

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
        self.ribbon.set_circuit_enabled(group_ok=True, ungroup_ok=True)

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
        self.ribbon.set_circuit_enabled(
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
                )
        self._propagate_staleness()

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
            ui_params = self.props.get_params()
            # Current is owned by the circuit when this coil is grouped —
            # don't overwrite the branch current with a stale UI value.
            if cid_save in self._coil_group_map:
                ui_params.pop('current', None)
            self._coil_params_map[cid_save].update(ui_params)

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
        else:
            # Warm the L-matrix cache in the background so the first
            # circuit-header click doesn't block the UI.
            self._schedule_l_matrix_precompute()

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
            self.ribbon._btn_field_lines.set_action_enabled(False)
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
            if self.ribbon._btn_global_field.isChecked():
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
        self.workspace.clear_field_lines_layer('global')
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
        """Export coil arrangement to a .calcsx file for later reload."""
        import json
        parent = parent or self
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Session", "session.calcsx",
            "CalcSX Session (*.calcsx)",
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

        with open(path, 'w') as f:
            json.dump({
                'version': 3,
                'coils': coils,
                'bobbins': bobbins,
                'global_field_lines': global_fl,
                'hall_probes': probes,
                'circuits': circuits,
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
        self.ribbon.set_inspect_enabled(bool(self._coil_coords))
        self.ribbon.set_construct_enabled(bool(self._coil_coords))
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

    def _apply_theme(self, name: str) -> None:
        from CalcSX_app.gui.gui_utils import apply_theme_to_app
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
        # Persist choice across launches
        from PyQt5.QtCore import QSettings
        QSettings().setValue("theme", name)

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
        if self.ribbon._btn_translate.isChecked() or self.ribbon._btn_rotate.isChecked():
            mode = 'T' if self.ribbon._btn_translate.isChecked() else 'R'
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
        self.ribbon._btn_reanalyze.set_action_enabled(bool(stale))
        self._global_fl_dirty = True

        # Auto-clear global field lines — they're invalid now
        if self.ribbon._btn_global_field.isChecked():
            self.workspace.clear_field_lines_layer('global')
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
