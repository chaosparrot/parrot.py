import os
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QFrame, QPushButton, QRadioButton,
    QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from gui.widgets.recording_card import RecordingCard
from gui.widgets.audio_graph import GRAPH_DISPLAY_WAVEFORM, GRAPH_DISPLAY_SPECTROGRAM
from gui.services import library_ops
from gui import components, theme
from gui.formatters import duration_text, spoken_duration
from gui.timing import debounced, staggered, next_tick
from lib.recordings import get_label_dir
from lib.print_status import get_quantity_rating, QUANTITIES
from lib.wav import wav_duration

SOUND_COLUMN, DATA_COLUMN, DETECTED_COLUMN = 0, 1, 2


class SoundsPage(QWidget):
    """
    ┌ sound_list ─────────┐┌ sound_header ────────────────────────────────────────────┐
    │ Sound  Data  Detect.││ sound_title                                              │
    │ pop    Good  1m 04s ││ sound_stats                                              │
    │ ss     Good   51.0s ││ sound_quantity                                           │
    │                     ││ open_folder_btn   _graph_display_buttons   normalize_btn │
    │                     │├ cards_scroll ────────────────────────────────────────────┤
    │                     ││ ┌ cards_container ─────────────────────────────────────┐ │
    │                     ││ │ ┌ RecordingCard ───────────────────────────────────┐ │ │
    │                     ││ │ │ Aug 08, 2026, 10:55 AM   32.4s                   │ │ │
    │                     ││ │ │ .:|:..:||:..:|:.. audio_graph ..:|:..:||:..:|:.. │ │ │
    │                     ││ │ └──────────────────────────────────────────────────┘ │ │
    │                     ││ │ ┌ RecordingCard ───────────────────────────────────┐ │ │
    │                     ││ │ │ Aug 08, 2026, 11:02 AM   18.9s                   │ │ │
    │                     ││ │ │ .:|:..:||:..:|:.. audio_graph ..:|:..:||:..:|:.. │ │ │
    │                     ││ │ └──────────────────────────────────────────────────┘ │ │
    │                     ││ └──────────────────────────────────────────────────────┘ │
    └─────────────────────┘└──────────────────────────────────────────────────────────┘
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._cards = []
        self._graph_display = GRAPH_DISPLAY_WAVEFORM
        self._normalized = True

        # Stagger audio graph builds to not stall the main thread (50-220 ms each)
        self._audio_graphs_build = staggered(RecordingCard.build_audio_graph, owner=self)
        # Same for switching display: a long take's spectrogram is ~60 ms
        self._graph_displays_set = staggered(
            lambda card: card.set_graph_display(self._graph_display), owner=self)

        self._setup_ui()
        self._sound_list_populate()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 8, 12)
        self.left_title = QLabel("Sounds")
        left_layout.addWidget(self.left_title)
        self.sound_list = QTreeWidget()
        self.sound_list.setAccessibleName("Sounds")
        self.sound_list.setHeaderLabels(["Sound", "Data", "Detected"])
        self.sound_list.setRootIsDecorated(False)
        self.sound_list.setUniformRowHeights(True)
        header = self.sound_list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(SOUND_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(DATA_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(DETECTED_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        # Debounce sound change in case arrow key is held
        self.sound_list.currentItemChanged.connect(
            debounced(self._cards_show_selected, wait_ms=15, owner=self))
        left_layout.addWidget(self.sound_list)
        # Where keyboard navigation to this page lands
        self.setFocusProxy(self.sound_list)

        left.setMinimumWidth(280)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._sound_header_build())

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_scroll.setObjectName("cardsScroll")
        self.cards_scroll.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.cards_scroll.setAccessibleName("Recordings")
        # Always on: wheel scroll over a graph zooms it instead of scrolling the page
        self.cards_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.cards_scroll.setWidget(self._cards_container_new()[0])
        right_layout.addWidget(self.cards_scroll)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        t = theme.colors()
        self.left_title.setStyleSheet(components.heading_style("card"))
        self.sound_title.setStyleSheet(components.heading_style("title"))
        self.sound_stats.setStyleSheet(f"color: {t['text_dim']};")
        self.sound_quantity.setStyleSheet(f"color: {t['text_dim']}; margin-top: 2px;")
        self.sound_header.setStyleSheet(
            f"QFrame#soundHeader {{ background-color: {t['toolbar']}; "
            f"border-bottom: 1px solid {t['border']}; }}")
        self.open_folder_btn.setStyleSheet(components.ghost_button_style())
        setting_toggle_style = (
            # t['text'], not t['text_dim'], for color contrast
            f"#settingToggle {{ background-color: {t['button']}; "
            f"color: {t['text']}; border: 1px solid {t['control_border']}; "
            f"border-radius: {t['radius']}; padding: 5px 14px; spacing: 0px; }} "
            f"#settingToggle::indicator {{ width: 0px; height: 0px; }} "
            f"#settingToggle:checked {{ background-color: {t['panel']}; "
            f"color: {t['text_bright']}; border: 1px solid {t['accent']}; }} "
            f"#settingToggle:hover {{ color: {t['text_bright']}; }} "
            f"#settingToggle:focus {{ "
            f"border: 2px solid {t['focus']}; padding: 4px 13px; }} "
            f"#settingToggle:disabled {{ background-color: {t['disabled_bg']}; "
            f"color: {t['disabled_text']}; border: 1px solid {t['control_border']}; }}")
        for b in list(self._graph_display_buttons.values()) + [self.normalize_btn]:
            b.setStyleSheet(setting_toggle_style)
        self.graph_display_label.setStyleSheet(f"color: {t['text_dim']};")
        self.cards_scroll.setStyleSheet(
            f"QScrollBar:vertical {{ width: 16px; background: {t['base']}; }} "
            # Transparent until focused, so the ring doesn't shift the cards
            f"QScrollArea#cardsScroll {{ border: 2px solid transparent; }} "
            f"QScrollArea#cardsScroll:focus {{ border: 2px solid {t['focus']}; }}")

    def _sound_header_build(self):
        header = QFrame()
        header.setObjectName("soundHeader")
        self.sound_header = header
        v = QVBoxLayout(header)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(2)

        self.sound_title = QLabel("")
        v.addWidget(self.sound_title)
        self.sound_stats = QLabel("")
        v.addWidget(self.sound_stats)
        self.sound_quantity = QLabel("")
        v.addWidget(self.sound_quantity)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 10, 0, 0)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.setObjectName("secondaryAction")
        self.open_folder_btn.setFlat(True)
        self.open_folder_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.open_folder_btn.clicked.connect(self._on_open_sound_folder)
        actions.addWidget(self.open_folder_btn)
        actions.addStretch()

        self.graph_display_label = QLabel("View:")
        actions.addWidget(self.graph_display_label)
        # Radio buttons styled as toggles: a screen reader hears one choice of
        # two, and arrow keys switch. Siblings in one parent are exclusive.
        graph_display_box = QWidget()
        graph_display_box.setAccessibleName("View")
        box_layout = QHBoxLayout(graph_display_box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self._graph_display_buttons = {}
        for graph_display, text in ((GRAPH_DISPLAY_WAVEFORM, "Waveform"),
                                    (GRAPH_DISPLAY_SPECTROGRAM, "Spectrogram")):
            btn = QRadioButton(text)
            btn.setObjectName("settingToggle")
            btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            btn.setChecked(graph_display == self._graph_display)
            btn.toggled.connect(
                lambda checked, m=graph_display: checked and self._cards_set_graph_display(m))
            self._graph_display_buttons[graph_display] = btn
            box_layout.addWidget(btn)
        actions.addWidget(graph_display_box)

        self.normalize_btn = QPushButton("Normalize")
        self.normalize_btn.setObjectName("settingToggle")
        self.normalize_btn.setCheckable(True)
        self.normalize_btn.setChecked(self._normalized)
        self.normalize_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.normalize_btn.setToolTip("Scale each waveform to its loudest point")
        self.normalize_btn.toggled.connect(self._cards_set_normalized)
        actions.addSpacing(12)
        actions.addWidget(self.normalize_btn)
        v.addLayout(actions)
        return header

    # ---- data ----------------------------------------------------------

    def _sound_list_populate(self):
        self.sound_list.blockSignals(True)
        self.sound_list.clear()
        for label in self.app_state.get_sound_labels():
            duration_ms = self.app_state.get_label_duration_ms(label)
            duration = duration_text(duration_ms / 1000.0)
            quantity, _pct = get_quantity_rating(duration_ms)
            item = QTreeWidgetItem([label, quantity, duration])
            item.setData(SOUND_COLUMN, Qt.ItemDataRole.UserRole, label)
            # A screen reader reads only the first column otherwise
            item.setData(SOUND_COLUMN, Qt.ItemDataRole.AccessibleTextRole,
                         f"{label}, {quantity} data, {spoken_duration(duration_ms / 1000.0)} detected")
            color = theme.QUANTITY_COLORS.get(quantity)
            if color:
                item.setForeground(DATA_COLUMN, QColor(color))
            item.setTextAlignment(DETECTED_COLUMN, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.sound_list.addTopLevelItem(item)
        self.sound_list.blockSignals(False)

        if self.sound_list.topLevelItemCount() == 0:
            self._cards_populate(None)
            return
        # Fires the selection handler, which shows the sound
        self.sound_list.setCurrentItem(self.sound_list.topLevelItem(0))

    # ---- cards: one per recording of the selected sound -----------------

    def _cards_show_selected(self):
        self._cards_populate(self.sound_list.currentItem())

    def _cards_populate(self, item):
        """Builds a new container of placeholder cards off-screen and swaps it
        in, so the sound shows in one repaint. Audio graphs fill in after, and
        the old container is freed once the new one is on screen.

        sound_list change (debounced) -> _cards_show_selected -> here
            _sound_header_update
            _cards_build
                _cards_container_new
                _card_new, per recording
            _cards_swap_in
            _audio_graphs_build (staggered)
            _cards_destroy_later -> _cards_destroy
        """
        self._audio_graphs_build.cancel()
        self._graph_displays_set.cancel()
        label = item.data(SOUND_COLUMN, Qt.ItemDataRole.UserRole) if item else None
        recordings = self.app_state.get_recordings_for_label(label) if label else []
        self._sound_header_update(label, recordings)
        old_cards = self._cards
        cards_container, self._cards = self._cards_build(label, recordings)
        old_cards_container = self._cards_swap_in(cards_container)
        self._audio_graphs_build(self._cards)
        self._cards_destroy_later(old_cards, old_cards_container)

    def _cards_build(self, label, recordings):
        """A placeholder card per recording, no audio graphs. Or the empty state."""
        cards_container, layout = self._cards_container_new()
        if label is None:
            components.center_in(layout, components.EmptyState("No sounds yet"))
        elif not recordings:
            components.center_in(layout, components.EmptyState(
                f"“{label}” has no recordings yet"))

        cards = [self._card_new(label, rec) for rec in recordings]
        for card in cards:
            layout.insertWidget(layout.count() - 1, card)
        return cards_container, cards

    def _card_new(self, label, rec):
        """One recording's card, in the current graph display and normalize setting."""
        base = os.path.splitext(rec["filename"])[0]
        card = RecordingCard(base, label, rec["wav_path"], rec["srt_path"])
        card.set_graph_display(self._graph_display)
        card.set_normalized(self._normalized)
        return card

    def _cards_container_new(self):
        """The empty container the cards stack in, top-aligned. Goes in cards_scroll."""
        cards_container = QWidget()
        layout = QVBoxLayout(cards_container)
        layout.setContentsMargins(20, 12, 24, 12)
        layout.setSpacing(10)
        layout.addStretch()
        return cards_container, layout

    def _cards_swap_in(self, cards_container):
        """Returns the container it replaced."""
        old = self.cards_scroll.takeWidget()
        self.cards_scroll.setWidget(cards_container)
        return old

    def _cards_destroy_later(self, cards, cards_container):
        """Once the new cards have painted: a built audio graph takes ~10 ms to free,
        and switching sounds shouldn't wait on the last one's cleanup."""
        next_tick(lambda: self._cards_destroy(cards, cards_container))

    def _cards_destroy(self, cards, cards_container):
        for card in cards:
            card.cleanup()
        cards_container.setParent(None)
        cards_container.deleteLater()

    # ---- header --------------------------------------------------------

    def _sound_header_update(self, label, recordings):
        self.sound_header.setVisible(label is not None)
        if label is None:
            self.cards_scroll.setAccessibleName("Recordings")
            return
        self.sound_title.setText(label)
        self.open_folder_btn.setAccessibleName(f"Open folder for {label}")
        self.open_folder_btn.setToolTip(os.path.normpath(os.path.abspath(get_label_dir(label))))
        count = len(recordings)
        recorded_s = sum(wav_duration(r["wav_path"]) for r in recordings)
        detected_ms = self.app_state.get_label_duration_ms(label)
        noun = "recording" if count == 1 else "recordings"
        self.cards_scroll.setAccessibleName(f"{count} {noun} of {label}")
        self.sound_stats.setText(f"{count} {noun}   ·   {duration_text(recorded_s)} recorded")
        self.sound_quantity.setVisible(count > 0)
        if count:
            self._sound_quantity_update(detected_ms)

    def _sound_quantity_update(self, detected_ms):
        quantity, percent_to_next = get_quantity_rating(detected_ms)
        color = theme.QUANTITY_COLORS.get(quantity, theme.colors()["text"])
        tail = ""
        if quantity != QUANTITIES["excellent"]:
            names = list(QUANTITIES.values())
            next_quantity = names[names.index(quantity) + 1]
            tail = f"   ·   {int(percent_to_next)}% toward “{next_quantity}”"
        self.sound_quantity.setText(
            f"Data quantity:  <b style='color:{color};'>{quantity}</b>"
            f"   ({duration_text(detected_ms / 1000.0)} of detected sound){tail}"
        )

    def _cards_set_graph_display(self, graph_display):
        self._graph_display = graph_display
        # Normalize scales the waveform's height. Spectrograms are always scaled
        # to each take's loudest bin
        self.normalize_btn.setEnabled(graph_display == GRAPH_DISPLAY_WAVEFORM)
        self._graph_displays_set(self._cards)

    def _cards_set_normalized(self, normalized):
        self._normalized = normalized
        for card in self._cards:
            card.set_normalized(normalized)

    def _on_open_sound_folder(self):
        item = self.sound_list.currentItem()
        if item is None:
            return
        label = item.data(SOUND_COLUMN, Qt.ItemDataRole.UserRole)
        try:
            library_ops.open_in_file_manager(get_label_dir(label))
        except library_ops.LibraryOpError as exc:
            QMessageBox.warning(self, "Couldn't open folder", str(exc))
