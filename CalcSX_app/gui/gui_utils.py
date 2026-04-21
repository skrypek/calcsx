# gui_utils.py
import sys
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QObject, QSize
from PyQt5.QtGui import QPalette, QColor, QFont, QFontDatabase, QIcon, QPixmap
import random
import time
from PyQt5.QtWidgets import (
    QProgressDialog, QApplication, QLabel,
)

# ─────────────────────────────────────────────────────────────
# Colour palette (single source of truth for all modules)
# ─────────────────────────────────────────────────────────────
_DARK_THEME = {
    'bg':            '#1e1e1e',   # VS-Code dark — main window / figure background
    'panel':         '#252526',   # sidebar / plot area background
    'input':         '#3c3c3c',   # spin-box / text-field background
    'border':        '#474747',   # separator lines
    'text':          '#d4d4d4',   # primary text
    'text_dim':      '#808080',   # secondary / disabled text
    'text_disabled': '#4a4a4a',   # disabled button symbol
    'text_dis_dim':  '#3a3a3a',   # disabled button label
    'accent':        '#4dd0e1',   # cyan — coil lines, active borders, highlight
    'accent2':       '#ce9178',   # warm orange — secondary plot lines
    'success':       '#4ec9b0',   # teal — "done" states
    'warning':       '#dcdcaa',   # yellow — warning states
    'hi_blue':       '#264f78',   # selection highlight
    'gen_btn':       '#007acc',   # generate button face
    'gen_btn_hover': '#1a8ad4',   # generate button hover
    'floor':         '#606060',   # 3-D floor grid colour
    'edge':          '#404040',   # tube mesh edge colour
    'del_hover':     '#ff6b6b',   # delete button hover
    'btn_dis_bdr':   '#333333',   # disabled button border
    'sb_text':       '#ffffff',   # scalar bar text
    'tool_bg':       'rgba(255,255,255,0.08)',   # tool button background
    'tool_bg_hover': 'rgba(255,255,255,0.15)',   # tool button hover
    'cmap_forces':   'plasma',
    'cmap_stress':   'YlOrRd',
    'cmap_field':    'cool',
    'cmap_section':  'inferno',
    # Layer designation colours (bright for dark bg)
    'lyr_forces':    '#ce9178',
    'lyr_stress':    '#e05050',
    'lyr_baxis':     '#80d8ff',
    'lyr_fieldlines':'#80ffff',
    'lyr_xsection':  '#ff9800',
    'lyr_probe':     '#e040fb',   # hall probe (purple)
    'probe_readout': '#00e5ff',   # probe readout accent (cyan)
    # Coil colour cycle
    'coil_colors':   ['#4dd0e1','#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff'],
    'mode':          'dark',
}

_LIGHT_THEME = {
    'bg':            '#f5f5f5',   # near-white background
    'panel':         '#e8e8ec',   # light-gray sidebars
    'input':         '#ffffff',   # white input fields
    'border':        '#c0c4cc',   # medium-gray borders
    'text':          '#1a1a2e',   # near-black text
    'text_dim':      '#6b7280',   # muted gray text
    'text_disabled': '#b0b0b0',   # disabled button symbol (light)
    'text_dis_dim':  '#c8c8c8',   # disabled button label (light)
    'accent':        '#21918c',   # viridis teal
    'accent2':       '#5ec962',   # viridis green
    'success':       '#31a354',   # green
    'warning':       '#b8860b',   # dark goldenrod
    'hi_blue':       '#c8ddf0',   # soft blue selection
    'gen_btn':       '#3b528b',   # viridis indigo
    'gen_btn_hover': '#4c6aaf',   # viridis indigo hover
    'floor':         '#b0b0b0',   # light-gray floor grid
    'edge':          '#c0c0c0',   # tube mesh edge colour (light)
    'del_hover':     '#e53935',   # delete button hover (light)
    'btn_dis_bdr':   '#d0d0d0',   # disabled button border (light)
    'sb_text':       '#1a1a2e',   # scalar bar text (dark on light bg)
    'tool_bg':       'rgba(0,0,0,0.06)',       # tool button background (light)
    'tool_bg_hover': 'rgba(0,0,0,0.12)',       # tool button hover (light)
    'cmap_forces':   'viridis',
    'cmap_stress':   'YlOrRd',
    'cmap_field':    'viridis',
    'cmap_section':  'viridis',
    # Layer designation colours (muted for light bg)
    'lyr_forces':    '#8b5e3c',
    'lyr_stress':    '#b03030',
    'lyr_baxis':     '#2070a0',
    'lyr_fieldlines':'#0d7377',
    'lyr_xsection':  '#c06000',
    'lyr_probe':     '#7b1fa2',   # hall probe (dark purple)
    'probe_readout': '#006064',   # probe readout accent (teal)
    # Coil colour cycle (deeper tones for light bg)
    'coil_colors':   ['#21918c','#c62828','#f9a825','#2e7d32','#1565c0','#7b1fa2'],
    'mode':          'light',
}

# Mutable dict — updated in-place by set_theme() so all importers see changes
THEME: dict = dict(_DARK_THEME)

_current_theme_name = 'dark'

def set_theme(name: str) -> None:
    """Switch theme in place.  *name* is 'dark' or 'light'."""
    global _current_theme_name
    src = _LIGHT_THEME if name == 'light' else _DARK_THEME
    THEME.clear()
    THEME.update(src)
    _current_theme_name = name

def get_theme_name() -> str:
    return _current_theme_name


_IMG_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _resources_dir() -> Path:
    """Resolve the resources/images folder whether running frozen or from source."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "resources" / "images"
    return Path(__file__).resolve().parent.parent / "resources" / "images"


def get_app_icon(theme: str | None = None) -> QIcon:
    """Return the multi-size app QIcon for the given theme (defaults to current)."""
    theme = theme if theme in ('light', 'dark') else _current_theme_name
    base = _resources_dir()
    icon = QIcon()
    for sz in _IMG_SIZES:
        fp = base / f"app_{theme}-{sz}.png"
        if fp.exists():
            icon.addFile(str(fp), QSize(sz, sz))
    return icon

# ─────────────────────────────────────────────────────────────
# Application-level dark setup helpers
# ─────────────────────────────────────────────────────────────

def build_palette() -> QPalette:
    """Return a QPalette matching the current THEME for use with Fusion style."""
    p = QPalette()
    bg      = QColor(THEME['bg'])
    panel   = QColor(THEME['panel'])
    inp     = QColor(THEME['input'])
    txt     = QColor(THEME['text'])
    dim     = QColor(THEME['text_dim'])
    hi      = QColor(THEME['hi_blue'])
    white   = QColor('#ffffff')

    p.setColor(QPalette.Window,           bg)
    p.setColor(QPalette.WindowText,       txt)
    p.setColor(QPalette.Base,             inp)
    p.setColor(QPalette.AlternateBase,    panel)
    p.setColor(QPalette.ToolTipBase,      panel)
    p.setColor(QPalette.ToolTipText,      txt)
    p.setColor(QPalette.Text,             txt)
    p.setColor(QPalette.Button,           panel)
    p.setColor(QPalette.ButtonText,       txt)
    p.setColor(QPalette.BrightText,       white)
    p.setColor(QPalette.Highlight,        hi)
    p.setColor(QPalette.HighlightedText,  white)
    p.setColor(QPalette.Link,             QColor(THEME['accent']))

    p.setColor(QPalette.Disabled, QPalette.Text,       dim)
    p.setColor(QPalette.Disabled, QPalette.ButtonText,  dim)
    p.setColor(QPalette.Disabled, QPalette.WindowText,  dim)
    return p


def pick_mono_font(size: int = 9) -> QFont:
    """Return the best available monospace font for the terminal aesthetic."""
    families = set(QFontDatabase().families())
    candidates = [
        "JetBrains Mono", "Cascadia Code", "Fira Code", "Source Code Pro",
        "SF Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New",
    ]
    for name in candidates:
        if name in families:
            return QFont(name, size)
    f = QFont()
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    return f


def build_app_qss() -> str:
    """Generate the global QSS stylesheet from the current THEME."""
    return f"""
QWidget {{
    background-color: {THEME['bg']};
    color: {THEME['text']};
}}
/* ── scrollbar ── */
QScrollBar:vertical {{
    background: {THEME['panel']}; width: 8px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {THEME['border']}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {THEME['panel']}; height: 8px; border: none;
}}
QScrollBar::handle:horizontal {{
    background: {THEME['border']}; border-radius: 4px; min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
/* ── splitter ── */
QSplitter::handle {{
    background: {THEME['border']}; width: 2px; height: 2px;
}}
/* ── tabs ── */
QTabWidget::pane {{
    border: 1px solid {THEME['border']}; background: {THEME['bg']};
}}
QTabBar::tab {{
    background: {THEME['panel']}; color: {THEME['text_dim']};
    padding: 6px 14px; border: 1px solid {THEME['border']}; border-bottom: none;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {THEME['bg']}; color: {THEME['text']};
    border-bottom: 2px solid {THEME['accent']};
}}
QTabBar::tab:hover {{ color: {THEME['text']}; }}
/* ── buttons ── */
QPushButton {{
    background: {THEME['panel']}; border: 1px solid {THEME['border']};
    border-radius: 3px; padding: 5px 10px; color: {THEME['text']};
}}
QPushButton:hover {{ background: {THEME['input']}; border-color: {THEME['accent']}; }}
QPushButton:pressed {{ background: {THEME['hi_blue']}; }}
QPushButton:disabled {{ color: {THEME['text_dim']}; border-color: {THEME['btn_dis_bdr']}; background: {THEME['panel']}; }}
/* generate button gets accent face */
QPushButton#GenerateButton {{
    background: {THEME['gen_btn']}; color: {THEME['sb_text']};
    border-color: {THEME['gen_btn']}; font-weight: bold;
}}
QPushButton#GenerateButton:hover {{ background: {THEME['gen_btn_hover']}; }}
QPushButton#GenerateButton:disabled {{
    background: {THEME['panel']}; color: {THEME['text_dim']};
    border-color: {THEME['border']};
}}
/* ── spin boxes / inputs ── */
QSpinBox, QDoubleSpinBox {{
    background: {THEME['input']}; border: 1px solid {THEME['border']};
    border-radius: 3px; padding: 2px 4px; color: {THEME['text']};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {THEME['accent']}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {THEME['border']}; border: none; width: 14px;
}}
/* ── checkboxes ── */
QCheckBox {{ color: {THEME['text']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    background: {THEME['input']}; border: 1px solid {THEME['border']}; border-radius: 2px;
}}
QCheckBox::indicator:checked {{
    background: {THEME['accent']}; border-color: {THEME['accent']};
}}
/* ── labels ── */
QLabel {{ color: {THEME['text']}; background: transparent; }}
/* ── summary card ── */
QFrame#SummaryCard {{
    background: {THEME['panel']}; border: 1px solid {THEME['border']};
    border-radius: 4px;
}}
/* ── parameter panel ── */
QWidget#ParameterPanel {{
    background: {THEME['panel']}; border-right: 1px solid {THEME['border']};
}}
/* ── progress dialog ── */
QProgressDialog {{
    background: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 6px;
}}
QProgressBar {{
    background: {THEME['input']}; border: 1px solid {THEME['border']};
    border-radius: 3px; text-align: center; color: {THEME['text']};
}}
QProgressBar::chunk {{ background: {THEME['accent']}; border-radius: 3px; }}
/* ── slider ── */
QSlider::groove:horizontal {{
    background: {THEME['input']}; height: 4px; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {THEME['accent']}; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {THEME['accent']}; border-radius: 2px; }}
/* ── tool button (help ?) ── */
QToolButton {{
    background: {THEME['tool_bg']}; border: 1px solid {THEME['border']};
    border-radius: 6px; font-weight: bold; color: {THEME['text']};
}}
QToolButton:hover {{ background: {THEME['tool_bg_hover']}; border-color: {THEME['accent']}; }}
/* ── structural helpers (objectName-based, auto-refresh on theme switch) ── */
QFrame#hdivider {{ color: {THEME['border']}; }}
QFrame#vbar     {{ color: {THEME['border']}; }}
QLabel#section_lbl {{ color: {THEME['text_dim']}; font-size:7pt; letter-spacing: 1.5px; }}
QLabel#ribbon_grp_lbl {{
    color: {THEME['text_dim']}; font-size:7pt;
    border-top: 1px solid {THEME['border']}; padding-top: 1px;
}}
QLabel#dim_label {{ color: {THEME['text_dim']}; font-size:8pt; }}
QPushButton#coil_del {{
    color: {THEME['text_dim']}; font-size:10pt; border: none;
    background: transparent; padding: 0;
}}
QPushButton#coil_del:hover {{ color: {THEME['del_hover']}; }}
"""

# Pre-build for backward compatibility (importers that use APP_QSS at load time)
APP_QSS = build_app_qss()


def apply_theme_to_app(name: str) -> None:
    """Switch the live application between 'dark' and 'light' themes."""
    global APP_QSS
    set_theme(name)
    APP_QSS = build_app_qss()
    app = QApplication.instance()
    if app is not None:
        app.setPalette(build_palette())
        app.setStyleSheet(APP_QSS)



# ─────────────────────────────────────────────────────────────
# Progress reporter
# ─────────────────────────────────────────────────────────────

class ProgressReporter(QObject):
    """
    Dynamic loading screen: quippy cycling messages + computation-stage subtitle.
    """

    DEFAULT_MESSAGES = [
        "Crunching numbers",       "Smoothing splines",        "Aligning magnetic fields",
        "Doing the math",          "Winding HTS",              "Acting busy",
        "Spinning the gimbal",     "Charging capacitors",      "Calculating Lorentz forces",
        "Applying stress model",   "Dividing by zero",         "Working overtime",
        "Thinking very hard",      "Vectorizing field equations", "Batching the Biot-Savart",
        "Querying the matrix",     "Racing electrons",         "Consulting Maxwell",
        "Solving in parallel",     "Flipping the parity",      "Inverting the Jacobian",
        "Aligning flux quanta",    "Cooling the coil",         "Integrating by parts",
        "Commuting the operators", "Calibrating the torus",
    ]

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.dlg = QProgressDialog("Booting up", None, 0, 100, parent)
        self.dlg.setWindowTitle(title)
        self.dlg.setWindowModality(Qt.WindowModal)
        self.dlg.setAutoReset(False)
        self.dlg.setAutoClose(False)
        self.dlg.setCancelButton(None)
        self.dlg.setMinimumWidth(360)

        label = self.dlg.findChild(QLabel)
        if label:
            label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            label.setTextFormat(Qt.RichText)
            label.setMinimumHeight(52)

        self._msgs          = []
        self._base_message  = "Booting up"
        self._stage_text    = ""
        self._dot_cycle     = ["", ".", "..", "..."]
        self._dot_index     = 0
        self._count         = 0
        self._interval      = 0
        self._booting       = True
        self._last_msg_time = 0.0
        self._min_msg_sec   = 4.0

        self._dot_timer  = QTimer(self)
        self._dot_timer.timeout.connect(self._advance_dots)
        self._boot_timer = QTimer(self)
        self._boot_timer.setSingleShot(True)
        self._boot_timer.timeout.connect(self._end_boot)

    def start(self):
        self._msgs         = list(self.DEFAULT_MESSAGES)
        random.shuffle(self._msgs)
        self._base_message = "Booting up"
        self._stage_text   = ""
        self._dot_index    = 0
        self._booting      = True
        self.dlg.setRange(0, 100)
        self.dlg.setValue(0)
        self._update_label()
        self.dlg.show()
        QApplication.processEvents()
        self._dot_timer.start(700)
        self._boot_timer.start(1800)
        self._interval      = random.randint(12, 20)
        self._count         = 0
        self._last_msg_time = time.time()

    def report(self, pct: int):
        self.dlg.setValue(pct)
        if not self._booting:
            self._count += 1
            now = time.time()
            if self._count >= self._interval and (now - self._last_msg_time) >= self._min_msg_sec:
                self._next_message()
                self._count         = 0
                self._interval      = random.randint(12, 20)
                self._last_msg_time = now
        QApplication.processEvents()

    def set_stage(self, msg: str):
        """Update computation-stage subtitle (called via cross-thread signal)."""
        self._stage_text = msg
        self._update_label()

    def finish(self):
        self._dot_timer.stop()
        self._boot_timer.stop()
        self.dlg.setValue(100)
        self._base_message = "Done"
        self._stage_text   = ""
        self._dot_index    = 0
        self._update_label()
        QTimer.singleShot(600, self.dlg.close)

    # ── internal ──────────────────────────────────────────────

    def _end_boot(self):
        self._booting = False
        self._next_message()
        self._last_msg_time = time.time()

    def _next_message(self):
        if not self._msgs:
            self._msgs = list(self.DEFAULT_MESSAGES)
            random.shuffle(self._msgs)
        self._base_message = self._msgs.pop()
        self._update_label()

    def _update_label(self):
        dots    = self._dot_cycle[self._dot_index]
        base    = self._base_message.rstrip('.')
        padding = '&nbsp;' * (3 - len(dots))
        quippy  = f"{base}{dots}{padding}"
        if self._stage_text:
            html = (
                f'<div style="text-align:center;line-height:1.6">'
                f'{quippy}<br>'
                f'<span style="font-size:85%;color:{THEME["text_dim"]}">'
                f'{self._stage_text}</span>'
                f'</div>'
            )
        else:
            html = f'<div style="text-align:center">{quippy}</div>'
        self.dlg.setLabelText(html)

    def _advance_dots(self):
        self._dot_index = (self._dot_index + 1) % len(self._dot_cycle)
        self._update_label()
