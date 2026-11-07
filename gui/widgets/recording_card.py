import os
from PyQt6.QtCore import Qt, QEvent, QTimer, QElapsedTimer, QSize, pyqtSignal
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QPushButton
from gui.widgets.audio_graph import AudioGraph, GRAPH_DISPLAY_WAVEFORM
from gui.services import playback
from gui import icons, theme
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
    """One recording: when it was made, how long it is, a waveform or
    spectrogram with detection overlaid, and playback."""

    AUDIO_GRAPH_HEIGHT = 150

    playback_started = pyqtSignal(object)
    playback_failed = pyqtSignal(str)
    picked = pyqtSignal(object)

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

        self._samples = None
        self._sample_rate = None
        self._duration = 0.0
        self._playhead_position = 0.0   # kept across pause
        self._play_start_position = 0.0
        self._playing = False
        self._output_latency = 0.0     # of the current play

        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(16)
        self._playhead_timer.timeout.connect(self._tick)
        self._play_clock = QElapsedTimer()

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
        self._play_icon = icons.play()
        self._pause_icon = icons.pause()
        self.play_btn = QPushButton(self._play_icon, " Play")
        self.play_btn.setIconSize(QSize(14, 14))
        self.play_btn.setFixedWidth(90)
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.play_btn.setToolTip("Play or pause (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.installEventFilter(self)
        row.addWidget(self.play_btn)

        # Example:
        # name  = "mici_1__1693668038"
        # title = "Sep 02, 2023, 10:20 AM", or the name if it has no timestamp
        stamp = get_recorded_at(name)
        title_text = recorded_at(stamp) if stamp else name
        title = QLabel(title_text)
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
        self.play_btn.setAccessibleName(
            f"{title_text}, {duration_text(length)}" if length else title_text)
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
            self.play_btn.setEnabled(False)
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
        self.audio_graph.seek_requested.connect(self._on_seek_requested)
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

    def playable(self):
        self.build_audio_graph()
        return self.audio_graph is not None

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self, from_position=None):
        self.build_audio_graph()
        if self.audio_graph is None:
            return      # unreadable, the card already says why
        if self._samples is None:
            self._samples, self._sample_rate = self.audio_graph.playback_audio()
            self._duration = self.audio_graph.duration()

        if from_position is not None:
            self._playhead_position = from_position
        if self._playhead_position >= self._duration:
            self._playhead_position = 0.0
        self._playhead_position = max(0.0, self._playhead_position)
        # Emit before playback.play(), so another card stops first and doesn't stop this stream
        self.playback_started.emit(self)

        self._play_start_position = self._playhead_position
        rest_of_clip = self._samples[int(self._playhead_position * self._sample_rate):]
        try:
            self._output_latency = playback.play(rest_of_clip, self._sample_rate)
        except playback.PlaybackError:
            self._show_stopped()
            self.playback_failed.emit("The default playback device would not open.")
            return

        self._playing = True
        self.play_btn.setIcon(self._pause_icon)
        self.play_btn.setText(" Pause")
        self.audio_graph.set_playhead(self._playhead_position)
        self._play_clock.restart()
        self._playhead_timer.start()

    def pause(self):
        if self._playing:
            playback.stop()
            self._playhead_position = min(self._heard_position(), self._duration)
        self._show_stopped()
        self.audio_graph.set_playhead(self._playhead_position)

    def stop(self):
        """Stop, leaving the playhead where it is."""
        if self._playing:
            playback.stop()
        self._show_stopped()

    def _show_stopped(self):
        self._playing = False
        self._playhead_timer.stop()
        self.play_btn.setIcon(self._play_icon)
        self.play_btn.setText(" Play")

    def eventFilter(self, obj, event):
        if obj is self.play_btn and event.type() == QEvent.Type.FocusIn:
            self.picked.emit(self)
        return super().eventFilter(obj, event)

    def _on_seek_requested(self, position):
        self.picked.emit(self)
        self._playhead_position = position
        self.audio_graph.set_playhead(position)
        if self._playing or playback.is_playing():
            self.play(from_position=position)

    def _heard_position(self):
        """Where playback has reached, in seconds. Audio leaves the device a
        buffer after play() is called, so without subtracting that the
        playhead runs ahead of what you hear."""
        seconds_since_play_started = self._play_clock.elapsed() / 1000.0
        heard_seconds = max(0.0, seconds_since_play_started - self._output_latency)
        return self._play_start_position + heard_seconds

    def _tick(self):
        heard_position = self._heard_position()
        if heard_position >= self._duration:
            # The playback thread finishes the stream itself. Aborting here would race it
            self._show_stopped()
            self._playhead_position = 0.0
            self.audio_graph.set_playhead(0.0)
            return
        self._playhead_position = heard_position
        self.audio_graph.set_playhead(heard_position)
        if playback.failed_mid_play():
            self._show_stopped()

    def cleanup(self):
        """Stop, and release plot items before deletion."""
        self.stop()
        if self.audio_graph is not None:
            self.audio_graph.cleanup()
