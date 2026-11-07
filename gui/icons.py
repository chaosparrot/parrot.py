"""Icons

Prefer painted paths over text glyphs (▶ ⏸) or emoji, because of:
a) same height in every font, so swapping play/pause doesn't shift the button
b) theme colours, including focus and disabled
c) sharp at any size or screen scale
"""
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QApplication

from gui import theme


def _render(draw, colour, size):
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    pixmap = QPixmap(int(size * dpr), int(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(size, size)          # draw in a 0..1 box
    draw(painter, QColor(colour))
    painter.end()
    return pixmap


def _button_icon(draw, size=14):
    t = theme.colors()
    icon = QIcon()
    for colour, mode in ((t["text"], QIcon.Mode.Normal),
                         (t["text_bright"], QIcon.Mode.Active),
                         (t["disabled_text"], QIcon.Mode.Disabled)):
        icon.addPixmap(_render(draw, colour, size), mode, QIcon.State.Off)
    return icon


def _draw_play(painter, colour):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    path = QPainterPath()
    path.moveTo(0.30, 0.18)
    path.lineTo(0.82, 0.50)
    path.lineTo(0.30, 0.82)
    path.closeSubpath()
    painter.drawPath(path)


def _draw_pause(painter, colour):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    for x in (0.26, 0.58):
        painter.drawRoundedRect(QRectF(x, 0.20, 0.16, 0.60), 0.05, 0.05)


def play():
    return _button_icon(_draw_play)


def pause():
    return _button_icon(_draw_pause)
