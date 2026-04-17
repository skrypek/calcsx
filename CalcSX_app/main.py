# main.py
import sys
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    # Identity for QSettings persistence (plist in ~/Library/Preferences/)
    app.setOrganizationName("CalcSX")
    app.setOrganizationDomain("calc.sx")
    app.setApplicationName("CalcSX")
    app.setStyle("Fusion")

    # ── Load saved theme (default: dark) ──────────────────────
    settings = QSettings()
    saved_theme = settings.value("theme", "dark", type=str)
    if saved_theme not in ("dark", "light"):
        saved_theme = "dark"

    from CalcSX_app.gui.gui_utils import (
        apply_theme_to_app, pick_mono_font, get_app_icon,
    )
    apply_theme_to_app(saved_theme)     # sets THEME, palette, and QSS
    app.setFont(pick_mono_font(size=9))
    app.setWindowIcon(get_app_icon())

    # ── main window ───────────────────────────────────────────
    from CalcSX_app.primary.main_utils import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
