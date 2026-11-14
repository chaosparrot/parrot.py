from PyQt6.QtWidgets import (QMainWindow, QToolBar, QStackedWidget, QWidget, QApplication,
                             QStatusBar, QLabel)
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtCore import Qt
from gui import theme
from gui.state import AppState
from gui.pages.home import HomePage
from gui.pages.sounds import SoundsPage


class MainWindow(QMainWindow):
    """
    ┌ toolbar ─────────────────────────────────────────────────┐
    │ [ Home ] [ Sounds ] ...    nav_actions                   │
    ├ page_stack ──────────────────────────────────────────────┤
    │ individual page shown by _show_page()                    │
    │                                                          │
    │                                                          │
    │                                                          │
    │                                                          │
    ├ status_bar ──────────────────────────────────────────────┤
    │ keyboard shortcuts if applicable                         │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parrot.py")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        self.app_state = AppState()

        self.page_stack = QStackedWidget()
        self.setCentralWidget(self.page_stack)

        self.home_page = HomePage(self.app_state, self)
        self.home_page.navigate.connect(self._go_to_page)
        self.pages = {
            "Home": self.home_page,
            "Sounds": SoundsPage(self.app_state, self),
            "Models": QWidget(),
        }
        for page in self.pages.values():
            self.page_stack.addWidget(page)

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
            group.addAction(action)
            toolbar.addAction(action)
            button = toolbar.widgetForAction(action)
            # Qt default is NoFocus. TabFocus so a click doesn't leave a ring
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            action.triggered.connect(
                lambda _checked, n=name, b=button: self._show_page(n, focus_page=b.hasFocus()))
            self.nav_actions[name] = action
        self.nav_actions["Home"].setChecked(True)

        # Permanent widgets only: showMessage() would hide them
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.keys_label = QLabel("")
        self.keys_label.setStyleSheet(f"color: {theme.colors()['text_dim']}; padding-right: 8px;")
        self.status_bar.addPermanentWidget(self.keys_label)
        self._refresh_keys_label()

    def _go_to_page(self, name):
        """Navigation from page buttons: keeps the toolbar checked state in sync."""
        self.nav_actions[name].setChecked(True)
        focused = QApplication.focusWidget()
        self._show_page(name, focus_page=focused is not None
                        and self.page_stack.currentWidget().isAncestorOf(focused))

    def _show_page(self, name, focus_page=False):
        """focus_page: navigated by keyboard. Focus follows into the page, so a
        screen reader announces where it landed instead of just "checked"."""
        page = self.pages[name]
        self.page_stack.setCurrentWidget(page)
        self._refresh_keys_label()
        if focus_page and page.focusProxy():
            page.setFocus(Qt.FocusReason.TabFocusReason)

    def _refresh_keys_label(self):
        """The shown page's keys, from its keybinding_hint(), if it has one."""
        hint = getattr(self.page_stack.currentWidget(), "keybinding_hint", None)
        self.keys_label.setText(hint() if callable(hint) else "")
