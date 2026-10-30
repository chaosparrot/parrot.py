import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from gui.widgets.audio_graph import AudioGraph, GRAPH_DISPLAY_WAVEFORM
from gui import theme
from gui.formatters import recorded_at, duration_text
from lib.recordings import get_recorded_at
from lib.wav import wav_duration, UnreadableWav


def _unreadable(exc):
    if exc.reason == UnreadableWav.NOT_16_BIT:
        return f"{exc.bits}-bit audio, parrot.py records 16-bit"
    return {
        UnreadableWav.CANT_OPEN: "The file couldn't be opened",
        UnreadableWav.NOT_PCM: "Unsupported wav format, parrot.py records 16-bit",
        UnreadableWav.EMPTY: "The recording is empty",
    }.get(exc.reason, "Not a wav file, or a damaged one")


class RecordingCard(QFrame):
    """One recording: when it was made, how long it is, and a waveform or
    spectrogram with detection overlaid."""

    AUDIO_GRAPH_HEIGHT = 150

    def __init__(self, name, label, wav_path, srt_path, parent=None):
        super().__init__(parent)
        # The audio graph inside asks to grow even at a fixed height, so without
        # this a lone card soaks up the page's spare height around its title
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.label = label
        self.wav_path = wav_path
        self.srt_path = srt_path

        t = theme.colors()
        self.setStyleSheet(
            "RecordingCard { border: 1px solid %s; border-radius: %s; "
            "background-color: %s; }" % (t["border"], t["radius"], t["card"]))

        self.audio_graph = None
        self._audio_graph_tried = False
        self._graph_display = GRAPH_DISPLAY_WAVEFORM
        self._normalized = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 6, 10, 6)
        self._layout.setSpacing(4)
        self._layout.addLayout(self._build_header(name))

        self._placeholder = QLabel("…")
        self._placeholder.setFixedHeight(self.AUDIO_GRAPH_HEIGHT)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {t['text_dim']}; border: none; background: transparent;")
        self._layout.addWidget(self._placeholder)

    def _build_header(self, name):
        row = QHBoxLayout()
        t = theme.colors()
        # Example:
        # name  = "mici_1__1693668038"
        # title = "Sep 02, 2023, 10:20 AM", or the name if it has no timestamp
        stamp = get_recorded_at(name)
        title = QLabel(recorded_at(stamp) if stamp else name)
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; border: none; background: transparent;")
        row.addWidget(title)

        length = wav_duration(self.wav_path)
        if length:
            meta = QLabel(duration_text(length))
            meta.setStyleSheet(f"color: {t['text_dim']}; border: none; background: transparent;")
            row.addWidget(meta)

        # Otherwise a recording with no srt looks like one where nothing was detected
        self._detection_note = QLabel("Not segmented yet")
        self._detection_note.setStyleSheet(f"color: {t['text_dim']}; border: none; background: transparent;")
        self._detection_note.setVisible(not self.srt_path)
        row.addWidget(self._detection_note)

        row.addStretch()
        return row

    def build_audio_graph(self):
        """Build it, or say the file can't be read. Runs once."""
        if self._audio_graph_tried:
            return
        self._audio_graph_tried = True
        graph = AudioGraph()
        graph.setFixedHeight(self.AUDIO_GRAPH_HEIGHT)
        problem = graph.load(self.wav_path, self.srt_path, self.label)
        if problem:
            graph.deleteLater()
            self._detection_note.setVisible(False)
            # The name, not the date: it's the file to go and look at
            name = os.path.basename(self.wav_path)
            self._placeholder.setText(f"Can't read {name}\n{_unreadable(problem)}")
            return
        if graph.regions_unreadable:
            self._detection_note.setText(f"Can't read {os.path.basename(self.srt_path)}")
            self._detection_note.setVisible(True)
        self.audio_graph = graph
        self.audio_graph.set_graph_display(self._graph_display)
        self.audio_graph.set_normalized(self._normalized)
        self._layout.replaceWidget(self._placeholder, self.audio_graph)
        self._placeholder.setParent(None)
        self._placeholder.deleteLater()
        self._placeholder = None

    def set_graph_display(self, graph_display):
        # Remembered, so an audio graph built later starts in it
        self._graph_display = graph_display
        if self.audio_graph is not None:
            self.audio_graph.set_graph_display(graph_display)

    def set_normalized(self, normalized):
        self._normalized = normalized
        if self.audio_graph is not None:
            self.audio_graph.set_normalized(normalized)

    def cleanup(self):
        """Release plot items before deletion."""
        if self.audio_graph is not None:
            self.audio_graph.cleanup()
