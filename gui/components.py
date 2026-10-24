"""Shared UI primitives.

Imports nothing from ``pages/``, so anything may import it.
Colours and type sizes come from ``theme``.
"""
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
