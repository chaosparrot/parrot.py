from PyQt6.QtWidgets import QMainWindow, QToolBar, QStackedWidget, QWidget
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtCore import Qt
from gui.state import AppState
from gui.windows.home import HomePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parrot.py")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        self.app_state = AppState()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage(self.app_state, self)
        self.home_page.navigate.connect(self._go_to_page)
        self.pages = {
            "Home": self.home_page,
            "Sounds": QWidget(),
            "Models": QWidget(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        toolbar = QToolBar("Navigation")
        toolbar.setMovable(False)
        # Without this, right-clicking the toolbar opens a Qt menu with a
        # "Navigation" checkbox. Unticking it hides the tabs, with no way to
        # get them back short of restarting.
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        group = QActionGroup(self)
        group.setExclusive(True)

        self.nav_actions = {}
        for name in self.pages:
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked, n=name: self._show_page(n))
            group.addAction(action)
            toolbar.addAction(action)
            # Qt default is NoFocus. TabFocus so a click doesn't leave a ring
            toolbar.widgetForAction(action).setFocusPolicy(Qt.FocusPolicy.TabFocus)
            self.nav_actions[name] = action
        self.nav_actions["Home"].setChecked(True)

    def _go_to_page(self, name):
        """Navigation from page buttons: keeps the toolbar checked state in sync."""
        self.nav_actions[name].setChecked(True)
        self._show_page(name)

    def _show_page(self, name):
        self.stack.setCurrentWidget(self.pages[name])
