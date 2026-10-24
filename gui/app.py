from PyQt6.QtWidgets import QApplication
from gui import theme


def create_app(argv):
    app = QApplication(argv)
    app.setApplicationName("Parrot.py")
    app.setStyle("Fusion")
    theme.apply(app)

    # Imported after QApplication: the window pulls in config, which imports
    # pyautogui, and that claims Windows DPI awareness if it gets there first.
    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    # Strong reference: `window` is only a local, and once create_app() returns
    # GC would delete the C++ widget (and every child) under the event loop.
    app._main_window = window

    return app
