# gui_utils.py
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QTimer, QObject
from PyQt5.QtGui import QPainter, QPixmap
import random
import time
from PyQt5.QtWidgets import QProgressDialog, QApplication, QLabel
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# load and cache the pixmap once (deferred until QApplication exists)
_LOGO_PATH = Path(__file__).parent.parent / "resources" / "images" / "powered_by.png"
_WATERMARK_PM = None

def _get_watermark_pm():
    global _WATERMARK_PM
    if _WATERMARK_PM is None:
        _WATERMARK_PM = QPixmap(str(_LOGO_PATH))
    return _WATERMARK_PM

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
        scaled = _get_watermark_pm().scaledToWidth(
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
    Dynamic loading screen with quippy cycling messages and a stage subtitle
    that reflects the actual computation step in progress.
    """

    DEFAULT_MESSAGES = [
        "Crunching numbers", "Smoothing splines", "Aligning magnetic fields",
        "Doing the math", "Winding HTS", "Acting busy", "Spinning the gimbal",
        "Charging capacitors", "Calculating Lorentz forces", "Applying stress model",
        "Dividing by zero", "Working overtime", "Thinking very hard",
        "Vectorizing field equations", "Batching the Biot-Savart",
        "Querying the matrix", "Racing electrons", "Consulting Maxwell",
        "Solving in parallel", "Flipping the parity", "Inverting the Jacobian",
        "Aligning flux quanta", "Cooling the coil", "Integrating by parts",
    ]

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.dlg = QProgressDialog("Booting up", None, 0, 100, parent)
        self.dlg.setWindowTitle(title)
        self.dlg.setWindowModality(Qt.WindowModal)
        self.dlg.setAutoReset(False)
        self.dlg.setAutoClose(False)
        self.dlg.setCancelButton(None)

        # Enable rich text on the inner label so HTML renders correctly
        label = self.dlg.findChild(QLabel)
        if label:
            label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            label.setTextFormat(Qt.RichText)

        # state
        self._msgs         = []
        self._base_message = "Booting up"
        self._stage_text   = ""          # current computation stage (subtitle)
        self._dot_cycle    = ["", ".", "..", "..."]
        self._dot_index    = 0

        # progress-driven switching
        self._count    = 0
        self._interval = 0
        self._booting  = True

        # timing to prevent too-frequent message switches
        self._last_msg_time  = 0.0
        self._min_msg_seconds = 4.0

        # timers
        self._dot_timer = QTimer(self)
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

        self.dlg.setValue(0)
        self._update_label()
        self.dlg.show()
        QApplication.processEvents()

        self._dot_timer.start(700)       # dot animation every 700 ms
        self._boot_timer.start(1800)     # boot phase lasts 1.8 s

        self._interval       = random.randint(12, 20)
        self._count          = 0
        self._last_msg_time  = time.time()

    def report(self, pct: int):
        """Update progress bar; may cycle to the next quippy message."""
        self.dlg.setValue(pct)

        if not self._booting:
            self._count += 1
            now = time.time()
            if (self._count >= self._interval and
                    (now - self._last_msg_time) >= self._min_msg_seconds):
                self._next_message()
                self._count    = 0
                self._interval = random.randint(12, 20)
                self._last_msg_time = now

        QApplication.processEvents()

    def set_stage(self, msg: str):
        """Show the current computation stage as a subtitle (called from worker thread via signal)."""
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

    # ---- internal helpers ----

    def _end_boot(self):
        self._booting = False
        self._next_message()
        self._last_msg_time = time.time()

    def _next_message(self):
        if self._msgs:
            self._base_message = self._msgs.pop()
        else:
            # Replenish and reshuffle when exhausted
            self._msgs = list(self.DEFAULT_MESSAGES)
            random.shuffle(self._msgs)
            self._base_message = self._msgs.pop()
        self._update_label()

    def _update_label(self):
        dots = self._dot_cycle[self._dot_index]
        base = self._base_message.rstrip('.')
        # Reserve horizontal space for up to 3 dots using non-breaking spaces
        padding = '&nbsp;' * (3 - len(dots))
        quippy  = f"{base}{dots}{padding}"

        if self._stage_text:
            html = (
                f'<div style="text-align:center; line-height:1.5">'
                f'{quippy}<br>'
                f'<span style="font-size:85%; color:#888">{self._stage_text}</span>'
                f'</div>'
            )
        else:
            html = f'<div style="text-align:center">{quippy}</div>'

        self.dlg.setLabelText(html)

    def _advance_dots(self):
        self._dot_index = (self._dot_index + 1) % len(self._dot_cycle)
        self._update_label()