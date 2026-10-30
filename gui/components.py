"""Shared UI primitives.

Imports nothing from ``pages/``, so anything may import it.
Colours and type sizes come from ``theme``.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui import theme


def heading_style(rank="card"):
    """One of the ranks in ``theme.TYPE_SCALE``."""
    t = theme.colors()
    size = theme.TYPE_SCALE[rank]
    return f"font-size: {size}px; font-weight: bold; color: {t['text_bright']};"


def primary_button_style():
    """Accent-filled call to action. One per screen."""
    t = theme.colors()
    return (f"QPushButton#primaryAction {{ background-color: {t['accent']}; "
            f"color: {t['accent_text']}; font-weight: bold; border: none; "
            f"border-radius: {t['radius']}; padding: 6px 18px; }} "
            # Else the app-wide QPushButton:hover greys it out.
            f"QPushButton#primaryAction:hover {{ "
            f"background-color: {t['accent']}; }} "
            f"QPushButton#primaryAction:focus {{ "
            f"border: 2px solid {t['focus']}; padding: 4px 16px; }} "
            f"QPushButton#primaryAction:disabled {{ "
            f"background-color: {t['disabled_bg']}; "
            f"color: {t['disabled_text']}; }}")


def ghost_button_style():
    """Borderless and dim, for a quiet secondary action."""
    t = theme.colors()
    return (f"QPushButton#secondaryAction {{ color: {t['text_dim']}; "
            f"border: none; background: transparent; padding: 3px 8px; }} "
            f"QPushButton#secondaryAction:hover {{ color: {t['text_bright']}; }} "
            f"QPushButton#secondaryAction:focus {{ "
            f"border: 2px solid {t['focus']}; padding: 1px 6px; }}")


class EmptyState(QWidget):
    """Centered title and body, for a view with nothing to show."""

    # Measured line length for the body copy. Wider reads as a paragraph.
    BODY_WIDTH = 460

    def __init__(self, title, body=""):
        super().__init__()
        t = theme.colors()
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(8)

        self.title = QLabel(title)
        self.title.setStyleSheet(heading_style("section"))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.title)

        self.body = QLabel(body)
        self.body.setWordWrap(True)
        self.body.setStyleSheet(f"color: {t['text_dim']};")
        self.body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.setFixedWidth(self.BODY_WIDTH)
        # A word-wrapped label reports a one-line sizeHint, so the layout
        # clips it unless told the height
        self.body.setMinimumHeight(max(0, self.body.heightForWidth(self.BODY_WIDTH)))
        self.body.setVisible(bool(body))
        v.addWidget(self.body, 0, Qt.AlignmentFlag.AlignHCenter)


def center_in(layout, panel):
    """Center `panel` vertically in an otherwise empty container: springs above
    and below, ahead of the container's own trailing stretch."""
    layout.insertStretch(layout.count() - 1)
    layout.insertWidget(layout.count() - 1, panel, 0,
                        Qt.AlignmentFlag.AlignHCenter)
