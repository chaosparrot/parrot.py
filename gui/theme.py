"""The app's colours and type sizes, and the stylesheet built from them.

Widgets read colours via ``colors()``; ``apply()`` sets the stylesheet and
palette on the app once, at startup.
"""
from string import Template
from PyQt6.QtGui import QPalette, QColor, QFont


DARK = {
    # Used when installed, Qt's default otherwise.
    "font": "Inter",
    "window": "#23272e",
    "base": "#15181c",
    "panel": "#2d323b",
    "card": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #363c46, stop:1 #2b303a)",
    "toolbar": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2f343d, stop:1 #262a31)",
    "text": "#dde2e8",
    "text_dim": "#a3abb6",
    "text_bright": "#ffffff",
    "accent": "#41d97f",
    "accent_text": "#0c1f14",
    "border": "#5b6372",
    "control_border": "#8a929e",
    "button": "#4c5462",
    "button_hover": "#5c6577",
    "focus": "#ffffff",
    "radius": "5px",
    "radius_card": "8px",
    "disabled_text": "#828a96",
    "disabled_bg": "#2f343c",
}

# Type scale. Every size in the app comes from here.
#
#   hero      the Home landing title. One in the app.
#   card      a heading inside a card or panel.
#   body      running copy, and the floor.
TYPE_SCALE = {
    "hero": 28,
    "card": 16,
    "body": 14,
}


def colors():
    return DARK


_STYLESHEET = Template("""
    * { font-size: ${fs_body}px; }
    QMainWindow, QWidget { background-color: $window; color: $text; }
    /* QWidget rule matches subclasses: without this, every QLabel paints
       an opaque box over gradient panels */
    QLabel { background-color: transparent; }
    QToolBar {
        background-color: $toolbar;
        border-bottom: 1px solid $border;
        spacing: 4px; padding: 4px 8px;
    }
    QToolBar QToolButton {
        color: $text_dim; padding: 6px 16px;
        border-radius: $radius; font-weight: bold; outline: none;
    }
    QToolBar QToolButton:hover { background-color: $button_hover; color: $text_bright; }
    QToolBar QToolButton:checked { background-color: $accent; color: $accent_text; }
    /* Border comes out of padding so focus doesn't resize. No :focus-visible
       in Qt, TabFocus does that */
    QToolBar QToolButton:focus { border: 2px solid $focus; padding: 4px 14px; }
    QPushButton {
        padding: 6px 16px; border-radius: $radius;
        background-color: $button; border: 1px solid $control_border; color: $text;
        outline: none;
    }
    QPushButton:focus { border: 2px solid $focus; padding: 5px 15px; }
    QPushButton:hover { background-color: $button_hover; }
    QPushButton:disabled {
        background-color: $disabled_bg; color: $disabled_text;
        border-color: $border;
    }
    QScrollBar:vertical { background-color: $window; width: 12px; border: none; }
    QScrollBar::handle:vertical { background-color: $control_border; border-radius: 5px; min-height: 20px; }
    QScrollBar::handle:vertical:hover { background-color: $accent; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar:horizontal { background-color: $window; height: 12px; border: none; }
    QScrollBar::handle:horizontal { background-color: $control_border; border-radius: 5px; min-width: 24px; }
    QScrollBar::handle:horizontal:hover { background-color: $accent; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
""")


def _palette(t):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(t["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(t["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(t["base"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(t["panel"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(t["panel"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(t["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(t["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(t["button"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(t["text"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(t["accent"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(t["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(t["accent_text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(t["disabled_text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(t["disabled_text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(t["disabled_text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(t["disabled_bg"]))
    return palette


def apply(app):
    """Set the font, palette and stylesheet on the whole application."""
    t = colors()
    font = QFont(t["font"], 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    app.setPalette(_palette(t))
    subs = dict(t, **{f"fs_{k}": v for k, v in TYPE_SCALE.items()})
    app.setStyleSheet(_STYLESHEET.substitute(subs))
