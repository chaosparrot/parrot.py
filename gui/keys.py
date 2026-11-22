"""Key checks and labels for pages that read key presses themselves."""
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence


def has_no_modifiers(event):
    # Bitwise: keep the modifiers but exclude the keypad modifier (the flag Qt
    # sets on number pad keys, e.g. numpad 4 with Num Lock off).
    # macOS reports every arrow key with the keypad modifier set, even with nothing
    # held, so without this, Left and Right fail the no-modifiers check and do nothing.
    modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
    return modifiers == Qt.KeyboardModifier.NoModifier


def unmodified(key):
    """A check for `key` pressed with no modifiers."""
    return lambda event: event.key() == key and has_no_modifiers(event)


def is_go_to_start(event):
    # Each platform's own keys: Home and Ctrl+Home on Windows / Linux,
    # Cmd+Left, Cmd+Up and Home on macOS, whose laptops have no Home key
    return (unmodified(Qt.Key.Key_Home)(event)
            or event.matches(QKeySequence.StandardKey.MoveToStartOfLine)
            or event.matches(QKeySequence.StandardKey.MoveToStartOfDocument))


def go_to_start_key_name():
    """What the status bar calls is_go_to_start's keys."""
    return "⌘←" if sys.platform == "darwin" else "Home"
