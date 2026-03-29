# gui_utils.py
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QTimer, QObject
from PyQt5.QtGui import QPainter, QPixmap
import random
import time
from PyQt5.QtWidgets import QProgressDialog, QApplication, QLabel
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# load and cache the pixmap once
_LOGO_PATH = Path(__file__).parent.parent / "resources" / "images" / "powered_by.png"
_WATERMARK_PM = QPixmap(str(_LOGO_PATH))

class WatermarkedCanvas(FigureCanvasQTAgg):
    """
    FigureCanvas that draws a transparent watermark pixmap in its paintEvent, after Matplotlib 
    has rendered. Now preserves 3D view state properly.
    """
    
    def __init__(self, figure, zoom=0.10, pad=0.02, alpha=0.4):
        super().__init__(figure)
        self._zoom = zoom  # fraction of widget width
        self._pad = pad    # fraction padding from right/bottom
        self._alpha = alpha
        
        # Store 3D view state to prevent view changes during repaints
        self._3d_view_state = {}
        self._store_3d_view_state()
    
    def _store_3d_view_state(self):
        """Store current 3D view state for all 3D axes"""
        self._3d_view_state = {}
        for i, ax in enumerate(self.figure.get_axes()):
            if hasattr(ax, 'zaxis'):  # It's a 3D axis
                self._3d_view_state[i] = {
                    'elev': ax.elev,
                    'azim': ax.azim,
                    'xlim': ax.get_xlim(),
                    'ylim': ax.get_ylim(),
                    'zlim': ax.get_zlim()
                }
    
    def _restore_3d_view_state(self):
        """Restore 3D view state for all 3D axes"""
        for i, ax in enumerate(self.figure.get_axes()):
            if hasattr(ax, 'zaxis') and i in self._3d_view_state:
                state = self._3d_view_state[i]
                ax.view_init(elev=state['elev'], azim=state['azim'])
                ax.set_xlim(state['xlim'])
                ax.set_ylim(state['ylim'])
                ax.set_zlim(state['zlim'])

    def paintEvent(self, event):
        # Store current 3D view state before matplotlib renders
        self._store_3d_view_state()
        
        # Let Matplotlib draw the figure contents
        super().paintEvent(event)
        
        # Restore 3D view state after matplotlib render (prevents drift)
        self._restore_3d_view_state()

        # Overlay the watermark via Qt (this should not affect matplotlib state)
        painter = QPainter(self)
        painter.setOpacity(self._alpha)

        w = self.width()
        h = self.height()
        # target watermark width in pixels
        target_w = int(w * self._zoom)
        scaled = _WATERMARK_PM.scaledToWidth(
            target_w, Qt.SmoothTransformation
        )

        # pad in pixels
        pad_x = int(w * self._pad)
        pad_y = int(h * self._pad)

        # bottom‑right corner
        x = w - scaled.width() - pad_x
        y = h - scaled.height() - pad_y

        painter.drawPixmap(x, y, scaled)
        painter.end()

    def mousePressEvent(self, event):
        """Override to update stored view state after user interaction"""
        super().mousePressEvent(event)
        # Update stored state after any mouse interaction
        self._store_3d_view_state()
    
    def mouseReleaseEvent(self, event):
        """Override to update stored view state after user interaction"""
        super().mouseReleaseEvent(event)
        # Update stored state after any mouse interaction
        self._store_3d_view_state()
    
    def wheelEvent(self, event):
        """Override to update stored view state after wheel zoom"""
        super().wheelEvent(event)
        # Update stored state after any wheel interaction
        self._store_3d_view_state()

def make_canvas(plot_fn, ctx, figsize=(8, 6), projection=None):
    """
    Returns a WatermarkedCanvas, which will draw the watermark in the Qt layer, not in Matplotlib.
    """
    
    fig = plt.figure(figsize=figsize)
    if projection:
        ax = fig.add_subplot(111, projection=projection)
    else:
        ax = fig.add_subplot(111)

    plot_fn(ax, ctx)

    plt.close(fig)
    # wrap it in our canvas subclass
    canvas = WatermarkedCanvas(fig)
    
    # Store initial state right after plot creation
    canvas._store_3d_view_state()
    
    return canvas

class LogoMixin:
    
    def __init__(self, logo_path, logo_pct=0.2, margin=10):
        self._logo_path = logo_path
        self._logo_pct = logo_pct
        self._margin = margin
    
    def paint_logo(self, painter, widget):
        pix = QPixmap(self._logo_path)
        if pix.isNull(): 
            return
        tw = int(widget.width() * self._logo_pct)
        scaled = pix.scaledToWidth(tw, Qt.SmoothTransformation)
        x = self._margin
        y = self._margin
        painter.drawPixmap(x, y, scaled)

class ProgressReporter(QObject):
    """
    Enables dynamic loading screen
    """
    
    DEFAULT_MESSAGES = [
        "Crunching numbers...", "Smoothing splines...", "Aligning magnetic fields...", "Doing the math...",
        "Winding HTS...", "Acting busy...", "Spinning gimbal...", "Charging capacitors...", "Calculating Lorentz forces...",
        "Applying stress model...", "Dividing by zero...", "Working overtime...", "Finishing up...", "Thinking..."
    ]

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.dlg = QProgressDialog("Booting up...", None, 0, 100, parent)
        self.dlg.setWindowTitle(title)
        self.dlg.setWindowModality(Qt.WindowModal)
        self.dlg.setAutoReset(False)
        self.dlg.setAutoClose(False)
        self.dlg.setCancelButton(None)

        # center label text
        label = self.dlg.findChild(QLabel)
        if label:
            label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # state
        self._msgs = []
        self._base_message = "Booting up..."
        self._dot_cycle = ["", ".", "..", "..."]
        self._dot_index = 0

        # progress-driven switching
        self._count = 0
        self._interval = 0
        self._booting = True

        # timing to prevent too-frequent message switches
        self._last_msg_time = 0.0
        self._min_msg_seconds = 5.0  # minimum seconds between message switches

        # timers
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._advance_dots)

        self._boot_timer = QTimer(self)
        self._boot_timer.setSingleShot(True)
        self._boot_timer.timeout.connect(self._end_boot)

    def start(self):
        self._msgs = [m for m in self.DEFAULT_MESSAGES if m != "Booting up..."]
        self._base_message = "Booting up..."
        self._dot_index = 0
        self._booting = True

        self.dlg.setValue(0)
        self._update_label()
        self.dlg.show()
        QApplication.processEvents()

        self._dot_timer.start(1000)  # dot animation (1s)
        self._boot_timer.start(2000)  # "Booting up" lasts 2.0s

        self._interval = random.randint(16, 22)
        self._count = 0
        self._last_msg_time = time.time()

    def report(self, pct: int):
        """Call this with current percent (0–100)."""
        
        self.dlg.setValue(pct)

        if not self._booting:
            self._count += 1
            now = time.time()
            if (self._count >= self._interval and
                    (now - self._last_msg_time) >= self._min_msg_seconds):
                self._next_message()
                self._count = 0
                self._interval = random.randint(16, 22)
                self._last_msg_time = now

        QApplication.processEvents()

    def finish(self):
        self._dot_timer.stop()
        self._boot_timer.stop()
        self.dlg.setValue(100)
        self._base_message = "Done"
        self._dot_index = 0
        self._update_label()
        QTimer.singleShot(800, self.dlg.close)

    # ---- internal helpers ----
    def _end_boot(self):
        self._booting = False
        self._next_message()
        self._last_msg_time = time.time()

    def _next_message(self):
        if self._msgs:
            self._base_message = self._msgs.pop(random.randrange(len(self._msgs)))
        else:
            self._base_message = "Checking work..."
        # DO NOT reset self._dot_index here — lets dots continue cycling
        self._update_label()

    def _update_label(self):
        dots = self._dot_cycle[self._dot_index]
        base = self._base_message.rstrip('.')  # remove baked-in dots (self-check)
        text = base + dots if dots else base
        text += ' ' * (3 - len(dots))  # reserve space for max three dots
        self.dlg.setLabelText(text)

    def _advance_dots(self):
        self._dot_index = (self._dot_index + 1) % len(self._dot_cycle)
        self._update_label()