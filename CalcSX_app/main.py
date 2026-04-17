# main.py
import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── dark palette ──────────────────────────────────────────
    from gui.gui_utils import build_palette, pick_mono_font, APP_QSS, get_app_icon
    app.setPalette(build_palette())
    app.setFont(pick_mono_font(size=9))
    app.setStyleSheet(APP_QSS)
    app.setWindowIcon(get_app_icon())

    # ── main window ───────────────────────────────────────────
    from primary.main_utils import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
