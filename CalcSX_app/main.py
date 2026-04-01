# main.py
import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── dark palette ──────────────────────────────────────────
    from gui.gui_utils import build_dark_palette, pick_mono_font, apply_mpl_dark_theme, APP_QSS
    app.setPalette(build_dark_palette())
    app.setFont(pick_mono_font(size=9))
    app.setStyleSheet(APP_QSS)

    # ── matplotlib dark theme (must be called before any figure is created) ──
    apply_mpl_dark_theme()

    # ── main window ───────────────────────────────────────────
    from primary.main_utils import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
