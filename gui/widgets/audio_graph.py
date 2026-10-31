"""For waveform and spectrogram views

Terms:
    Spectrogram:
        FFT       Fast Fourier Transform - transform frame of samples to strength/freq (np.fft.rfft)
        frame     32 ms (SPECTROGRAM_FRAME_S) - one spectrogram column (nfft samples)
        hop       4 ms (SPECTROGRAM_HOP_S) - time between frame starts (frames overlap). wider on takes over ~4 min
        bin       31.25 Hz apart (bin_hz) - one freq slot of FFT output (1 / frame length)
        window    fade curve, 0 at both ends (Hann / np.hanning) - multiplied into each frame
        dB        20 * log10(strength)
        mel       pitch scale modelled on hearing (hz_to_mel)
        row       one freq band, evenly spaced in mel (SPECTROGRAM_ROWS)
        levels    pg ImageItem colour range - dB drawn dark to light, top 60 dB (SPECTROGRAM_RANGE_DB)

    Waveform:
        envelope  min/max per view slice, filled between (_max_curve, _min_curve, _fill)

    Both:
        ViewBox   pg's zoomable plot area (_vb). items placed in data coords (seconds, amplitude or mel), not pixels
"""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QScrollBar
from lib.mfsc import hz_to_mel, mel_to_hz
from lib.srt import get_label_ranges_in_srt
from lib.wav import read_wav, UnreadableWav
from config.config import MS_PER_FRAME
from gui import theme
from gui.timing import debounced

GRAPH_DISPLAY_WAVEFORM = "waveform"
GRAPH_DISPLAY_SPECTROGRAM = "spectrogram"

pg.setConfigOptions(antialias=True)


def _mel_bands(sample_rate, nfft, rows):
    """Where each of `rows` evenly spaced mel rows falls among the FFT bins:
    its centre, and the whole bins inside it."""
    bin_hz = sample_rate / nfft
    last_bin = nfft // 2
    mel_edges = np.linspace(0.0, hz_to_mel(sample_rate / 2.0), rows + 1)
    edges = mel_to_hz(mel_edges) / bin_hz
    centres = np.minimum(mel_to_hz((mel_edges[:-1] + mel_edges[1:]) / 2.0) / bin_hz, last_bin)
    starts = np.ceil(edges[:-1]).astype(int)
    ends = np.minimum(np.floor(edges[1:]).astype(int), last_bin) + 1
    return centres, starts, ends


def _to_mel_rows(db, bands):
    """Each row is the loudest bin in its band, so a tone between two row
    centres still shows. Rows with no whole bin inside interpolate instead."""
    centres, starts, ends = bands
    lo = np.floor(centres).astype(int)
    hi = np.minimum(lo + 1, db.shape[1] - 1)
    frac = (centres - lo).astype(np.float32)
    rows = db[:, lo] * (1.0 - frac) + db[:, hi] * frac
    for row in np.flatnonzero(ends > starts):
        rows[:, row] = db[:, starts[row]:ends[row]].max(axis=1)
    return rows


class AudioGraph(QWidget):
    """A recording as a waveform or a spectrogram, with detection shaded over
    it. Both views share the time axis.

    Zoom and pan are X only. The waveform is re-rendered as a min/max envelope
    whenever the view changes, so detail sharpens on zoom instead of aliasing.
    """

    SPECTROGRAM_FRAME_S = 0.032
    SPECTROGRAM_HOP_S = 0.004
    # A long take gets wider columns rather than an image too big to keep per card
    SPECTROGRAM_MAX_COLUMNS = 60_000
    # Mel-spaced rows: 0-2 kHz gets half the height instead of a quarter,
    # 4-8 kHz a quarter instead of half
    SPECTROGRAM_ROWS = 160
    SPECTROGRAM_RANGE_DB = 60
    SPECTROGRAM_TICKS_HZ = ((1000, "1k"), (2000, "2k"), (4000, "4k"), (8000, "8k"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.regions_unreadable = False
        self._samples = None      # mono float32 in [-1, 1]
        self._sample_rate = None
        self._duration = 0.0
        self._peak = 0.0          # max abs sample, for normalizing the view
        self._normalized = False
        self._graph_display = GRAPH_DISPLAY_WAVEFORM
        self._spectrogram = None  # cached (image, levels, nfft, hop)

        t = theme.colors()
        self._colors = t

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(background=t["plot_bg"])
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        # Nothing to do with the keyboard, and no ring to show it had focus
        self.plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=False, alpha=t["grid_alpha"])
        layout.addWidget(self.plot)

        # Shown only when zoomed in. A fixed-height row, so showing it never
        # shifts the plot above.
        self._scroll_scale = 1000.0  # view seconds <-> scrollbar units (ms)
        self._updating_scroll = False
        self._scroll_row = QWidget()
        self._scroll_row.setFixedHeight(14)
        scroll_layout = QVBoxLayout(self._scroll_row)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self.hscroll.setVisible(False)
        self.hscroll.valueChanged.connect(self._on_hscroll)
        scroll_layout.addWidget(self.hscroll)
        layout.addWidget(self._scroll_row)

        self._vb = self.plot.getViewBox()
        # Drag pan off, drag reserved for selecting a range. Pan via scrollbar
        self._vb.mouseDragEvent = lambda event, axis=None: event.ignore()
        # Wheel zooms time over the left axis too, so there's no dead zone
        self.plot.getAxis("left").wheelEvent = self._vb.wheelEvent

        # Filled waveform: the min/max envelope between two curves
        pen = pg.mkPen(QColor(*t["wave"]), width=1)
        self._max_curve = self.plot.plot(pen=pen)
        self._min_curve = self.plot.plot(pen=pen)
        self._fill = pg.FillBetweenItem(self._max_curve, self._min_curve,
                                        brush=QColor(*t["wave_fill"]))
        self.plot.addItem(self._fill)

        # Averages columns when there are more than pixels, instead of
        # dropping most of them and thinning out every pop
        self._image = pg.ImageItem(autoDownsample=True)
        self._image.setZValue(-20)  # under the detection regions
        self._image.setVisible(False)
        self.plot.addItem(self._image)

        self._regions = []

        # Hover crosshair and time readout
        cursor_pen = pg.mkPen(QColor(t["text_dim"]), width=1, style=Qt.PenStyle.DashLine)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=cursor_pen)
        self.cursor.setZValue(15)
        self.cursor.setVisible(False)
        self.plot.addItem(self.cursor)
        self.readout = pg.TextItem(color=QColor(t["text_bright"]), anchor=(0, 0))
        self.readout.setZValue(16)
        self.readout.setVisible(False)
        self.plot.addItem(self.readout)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Redraw once zooming or panning settles
        self._render_when_settled = debounced(
            self._render_waveform, wait_ms=15, owner=self)
        self._vb.sigXRangeChanged.connect(self._render_when_settled)
        self._vb.sigXRangeChanged.connect(lambda *_: self._sync_scrollbar())

        self.plot.scene().sigMouseClicked.connect(self._on_clicked)

    # ---- loading -------------------------------------------------------

    def load(self, wav_path, srt_path, label):
        """The UnreadableWav if it can't be read, or None once it's on the plot."""
        self._spectrogram = None
        try:
            self._samples, self._sample_rate = read_wav(wav_path)
        except UnreadableWav as exc:
            return exc
        self._peak = float(np.abs(self._samples).max()) if len(self._samples) else 0.0
        self._duration = len(self._samples) / self._sample_rate if self._sample_rate else 0.0

        self.plot.setLimits(xMin=0, xMax=self._duration)
        self._vb.setXRange(0.0, self._duration, padding=0)
        self._apply_y_range()
        self.load_regions(srt_path, label)
        self._render_waveform()
        return None

    def load_regions(self, srt_path, label):
        """Draw the detection overlay."""
        for region in self._regions:
            self.plot.removeItem(region)
        self._regions = []
        self.regions_unreadable = False
        if not srt_path:
            return
        try:
            ranges = get_label_ranges_in_srt(label, srt_path, MS_PER_FRAME)
        except (OSError, ValueError, IndexError):
            self.regions_unreadable = True
            return
        for start_ms, end_ms in ranges:
            region = pg.LinearRegionItem(
                values=[start_ms / 1000.0, end_ms / 1000.0], movable=False,
                brush=QColor(*self._colors["detect_brush"]),
                pen=pg.mkPen(QColor(*self._colors["detect_pen"])),
            )
            region.setZValue(-10)
            self.plot.addItem(region)
            self._regions.append(region)

    # ---- graph display -------------------------------------------------

    def set_graph_display(self, graph_display):
        if graph_display == self._graph_display:
            return
        self._graph_display = graph_display
        is_wave = graph_display == GRAPH_DISPLAY_WAVEFORM
        self._max_curve.setVisible(is_wave)
        self._min_curve.setVisible(is_wave)
        self._fill.setVisible(is_wave)
        self._image.setVisible(not is_wave)
        if is_wave:
            self.plot.getAxis("left").setTicks(None)
            self._apply_y_range()
            self._render_waveform()
        else:
            self._render_spectrogram()

    def set_normalized(self, normalized):
        self._normalized = normalized
        if self._graph_display == GRAPH_DISPLAY_WAVEFORM:
            self._apply_y_range()

    def _apply_y_range(self):
        if self._normalized and self._peak > 0:
            lim = max(self._peak, 0.02)
        else:
            lim = 1.0
        self.plot.setLimits(yMin=-lim * 1.08, yMax=lim * 1.08)
        self._vb.setYRange(-lim, lim, padding=0)

    def _render_spectrogram(self):
        if self._samples is None or self._sample_rate is None:
            return
        if self._spectrogram is None:
            self._spectrogram = self._compute_spectrogram()
        if self._spectrogram is None:
            return
        img, levels, nfft, hop = self._spectrogram
        sr = self._sample_rate
        top_mel = float(hz_to_mel(sr / 2.0))
        self._image.setImage(img, autoLevels=False, levels=levels)
        # Each column centred on its frame, so it lines up with the time
        # axis and the detection regions
        x0 = (nfft - hop) / 2.0 / sr
        self._image.setRect(QRectF(x0, 0, len(img) * hop / sr, top_mel))
        self._image.setLookupTable(pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256))
        # The y axis is in mel, labelled in kHz
        self.plot.getAxis("left").setTicks([[
            (float(hz_to_mel(hz)), text) for hz, text in self.SPECTROGRAM_TICKS_HZ
            if hz <= sr / 2.0]])
        self.plot.setLimits(yMin=0, yMax=top_mel)
        self._vb.setYRange(0, top_mel, padding=0)

    def _compute_spectrogram(self):
        samples, sr = self._samples, self._sample_rate
        if samples is None or not len(samples):
            return None
        nfft = max(2, int(round(sr * self.SPECTROGRAM_FRAME_S)))
        if len(samples) < nfft:
            samples = np.pad(samples, (0, nfft - len(samples)))
        hop = max(int(round(sr * self.SPECTROGRAM_HOP_S)),
                  int(np.ceil((len(samples) - nfft) / self.SPECTROGRAM_MAX_COLUMNS)), 1)
        # samples (n,) -> frames (cols, nfft) -> rfft (cols, nfft/2+1) -> dB -> mel rows (cols, ROWS)
        frames = np.lib.stride_tricks.sliding_window_view(samples, nfft)[::hop]
        window = np.hanning(nfft).astype(np.float32)
        bands = _mel_bands(sr, nfft, self.SPECTROGRAM_ROWS)
        db = np.empty((len(frames), self.SPECTROGRAM_ROWS), np.float32)
        # In chunks: a whole long take at once peaks at ~20x the image's size
        chunk = 4096
        for i in range(0, len(frames), chunk):
            mag = np.abs(np.fft.rfft(frames[i:i + chunk] * window, axis=1))
            db[i:i + chunk] = _to_mel_rows(20.0 * np.log10(mag + 1e-6), bands)
        top = float(db.max())
        return db, (top - self.SPECTROGRAM_RANGE_DB, top), nfft, hop

    # ---- waveform envelope --------------------------------------------

    def _render_waveform(self):
        if self._graph_display != GRAPH_DISPLAY_WAVEFORM or self._samples is None or self._sample_rate is None:
            return
        x0, x1 = self._vb.viewRange()[0]
        x0 = max(0.0, x0)
        x1 = min(self._duration, x1)
        if x1 <= x0:
            return

        sr = self._sample_rate
        i0 = int(x0 * sr)
        i1 = min(len(self._samples), int(x1 * sr))
        seg = self._samples[i0:i1]
        if len(seg) == 0:
            return

        width_px = max(self.plot.width(), 800)
        buckets = min(len(seg), width_px * 2)

        if len(seg) <= buckets:
            t = (i0 + np.arange(len(seg))) / sr
            self._max_curve.setData(t, seg)
            self._min_curve.setData(t, seg)
            return

        edges = np.linspace(0, len(seg), buckets + 1).astype(int)
        env_max = np.maximum.reduceat(seg, edges[:-1])
        env_min = np.minimum.reduceat(seg, edges[:-1])
        t = (i0 + (edges[:-1] + edges[1:]) / 2.0) / sr
        self._max_curve.setData(t, env_max)
        self._min_curve.setData(t, env_min)

    # ---- zoom ----------------------------------------------------------

    def fit_full(self):
        if self._duration <= 0:
            return
        self._vb.setXRange(0.0, self._duration, padding=0)
        if self._graph_display == GRAPH_DISPLAY_WAVEFORM:
            self._apply_y_range()

    # ---- horizontal scrollbar -----------------------------------------

    def _sync_scrollbar(self):
        """Mirror the visible window onto the scrollbar; hidden when the whole
        clip is in view."""
        if self._updating_scroll or self._duration <= 0:
            return
        x0, x1 = self._vb.viewRange()[0]
        span = x1 - x0
        if span >= self._duration - 1e-6:
            self.hscroll.setVisible(False)
            return
        s = self._scroll_scale
        self._updating_scroll = True
        self.hscroll.setMinimum(0)
        self.hscroll.setMaximum(max(0, int((self._duration - span) * s)))
        self.hscroll.setPageStep(max(1, int(span * s)))
        self.hscroll.setSingleStep(max(1, int(span * s / 20)))
        self.hscroll.setValue(int(max(0.0, x0) * s))
        self.hscroll.setVisible(True)
        self._updating_scroll = False

    def _on_hscroll(self, value):
        if self._updating_scroll:
            return
        x0, x1 = self._vb.viewRange()[0]
        span = x1 - x0
        new_x0 = value / self._scroll_scale
        self._updating_scroll = True
        self._vb.setXRange(new_x0, new_x0 + span, padding=0)
        self._updating_scroll = False

    def cleanup(self):
        """Tear down plot items before deletion, so pyqtgraph never paints an
        item whose ViewBox has already gone."""
        self._render_when_settled.cancel()
        for signal in (self._vb.sigXRangeChanged, self.plot.scene().sigMouseMoved,
                       self.plot.scene().sigMouseClicked):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
        self.plot.clear()

    def leaveEvent(self, event):
        # Leaving the widget fires no scene move, so the hover would linger
        self.cursor.setVisible(False)
        self.readout.setVisible(False)
        super().leaveEvent(event)

    def _on_mouse_moved(self, scene_pos):
        if self._samples is None:
            return
        if not self._vb.sceneBoundingRect().contains(scene_pos):
            self.cursor.setVisible(False)
            self.readout.setVisible(False)
            return
        point = self._vb.mapSceneToView(scene_pos)
        x = max(0.0, min(point.x(), self._duration))
        self.cursor.setPos(x)
        y_top = self._vb.viewRange()[1][1]
        self.readout.setText(f"{x:.3f}s")
        self.readout.setPos(x, y_top)
        self.cursor.setVisible(True)
        self.readout.setVisible(True)

    def _on_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.double():
            self.fit_full()
