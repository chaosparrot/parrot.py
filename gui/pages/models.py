import html
import os
import threading
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QSplitter, QFrame, QPushButton, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from config.config import BACKGROUND_LABEL, CLASSIFIER_FOLDER
from gui.services import library_ops
from gui import components, theme
from gui.formatters import trained_when, file_size_text, percent_span

MODEL_COLUMN, TRAINED_COLUMN, SOUNDS_COLUMN, NETS_COLUMN = 0, 1, 2, 3
SOUND_COLUMN, ACCURACY_COLUMN, NOTE_COLUMN = 0, 1, 2


class ModelsPage(QWidget):
    """
    ┌ model_list ──────────────────────┐┌ model_header ───────────────────────────────────┐
    │                                  ││ model_title                                     │
    │                                  ││ model_stats                                     │
    │                                  │├ details ────────────────────────────────────────┤
    │                                  ││ ┌ sound_tree ────────────────┐┌ facts ─────────┐│
    │                                  ││ │                            ││                ││
    │                                  ││ │                            ││                ││
    │                                  ││ │                            ││                ││
    │                                  ││ └────────────────────────────┘└────────────────┘│
    └──────────────────────────────────┘└─────────────────────────────────────────────────┘
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.reader = ModelDetailsReader(app_state, self)
        self.reader.model_read.connect(self._on_model_read)
        self.reader.all_read.connect(self._on_all_read)
        self._items = {}
        self._setup_ui()
        self._model_show(None)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 8, 12)
        self.left_title = QLabel("Models")
        left_layout.addWidget(self.left_title)
        self.model_list = QTreeWidget()
        self.model_list.setAccessibleName("Models")
        self.model_list.setHeaderLabels(["Model", "Trained", "Sounds", "Nets"])
        self.model_list.setRootIsDecorated(False)
        self.model_list.setUniformRowHeights(True)
        header = self.model_list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(MODEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        for column in (TRAINED_COLUMN, SOUNDS_COLUMN, NETS_COLUMN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.model_list.currentItemChanged.connect(self._on_model_selected)
        left_layout.addWidget(self.model_list)
        self.setFocusProxy(self.model_list)

        left.setMinimumWidth(340)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._model_header_build())
        right_layout.addWidget(self._details_build(), 1)
        self.empty_state = QWidget()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.addStretch()
        components.center_in(empty_layout, components.EmptyState("No models yet"))
        right_layout.addWidget(self.empty_state, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 820])

        t = theme.colors()
        self.left_title.setStyleSheet(components.heading_style("card"))
        self.model_title.setStyleSheet(components.heading_style("title"))
        self.model_stats.setStyleSheet(f"color: {t['text_dim']};")
        self.model_header.setStyleSheet(
            f"QFrame#modelHeader {{ background-color: {t['toolbar']}; "
            f"border-bottom: 1px solid {t['border']}; }}")
        self.open_folder_btn.setStyleSheet(components.ghost_button_style())

    def _model_header_build(self):
        header = QFrame()
        header.setObjectName("modelHeader")
        self.model_header = header
        v = QVBoxLayout(header)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(2)

        self.model_title = QLabel("")
        self.model_title.setTextFormat(Qt.TextFormat.PlainText)
        v.addWidget(self.model_title)
        self.model_stats = QLabel("")
        v.addWidget(self.model_stats)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 10, 0, 0)
        self.open_folder_btn = QPushButton("Open models folder")
        self.open_folder_btn.setObjectName("secondaryAction")
        self.open_folder_btn.setFlat(True)
        self.open_folder_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.open_folder_btn.setToolTip(os.path.normpath(os.path.abspath(CLASSIFIER_FOLDER)))
        self.open_folder_btn.clicked.connect(self._on_open_models_folder)
        actions.addWidget(self.open_folder_btn)
        actions.addStretch()
        v.addLayout(actions)
        return header

    def _details_build(self):
        """sound_tree carries the width, facts beside it is fixed."""
        self.details = QWidget()
        columns = QHBoxLayout(self.details)
        columns.setContentsMargins(20, 16, 24, 16)
        columns.setSpacing(22)
        columns.addWidget(self._sound_tree_build(), 1)
        self.facts = QLabel("")
        self.facts.setTextFormat(Qt.TextFormat.RichText)
        self.facts.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.facts.setFixedWidth(300)
        columns.addWidget(self.facts, 0)
        return self.details

    def _sound_tree_build(self):
        """A tree rather than rich text, so accuracy sorts."""
        tree = QTreeWidget()
        self.sound_tree = tree
        tree.setAccessibleName("Sounds in this model")
        tree.setHeaderLabels(["Sound", "Accuracy", ""])
        tree.setRootIsDecorated(False)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        # Nothing in it to act on
        tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tree.setSortingEnabled(True)
        tree.sortByColumn(SOUND_COLUMN, Qt.SortOrder.AscendingOrder)
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(SOUND_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(ACCURACY_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(NOTE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        return tree

    # ---- model list ----------------------------------------------------

    def showEvent(self, event):
        """A model trained from the command line while the app is open shows
        on the next visit."""
        super().showEvent(event)
        self._model_list_refresh()

    def _model_list_refresh(self):
        """Only if the models or their order changed."""
        models_newest_first = self.app_state.get_models_newest_first()
        names = [name for name, _trained_at in models_newest_first]
        if names != list(self._items):
            self._model_list_populate(models_newest_first)

    def _model_list_populate(self, models_newest_first):
        current = self._current_model_name()
        now = datetime.now()
        self.model_list.blockSignals(True)
        self.model_list.clear()
        self._items = {}
        for name, trained_at in models_newest_first:
            item = QTreeWidgetItem([name, "", "", ""])
            item.setData(MODEL_COLUMN, Qt.ItemDataRole.UserRole, name)
            # Long names are cut off in the column
            item.setToolTip(MODEL_COLUMN, name)
            for column in (SOUNDS_COLUMN, NETS_COLUMN):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._items[name] = item
            self._model_row_update(name, now, trained_at)
            self.model_list.addTopLevelItem(item)
        self.model_list.blockSignals(False)

        if not self._items:
            self._model_show(None)
            return
        self._reader_start(list(self._items))
        # Fires the selection handler, which shows the model
        self.model_list.setCurrentItem(self._items.get(current) or self.model_list.topLevelItem(0))

    def _model_row_update(self, name, now, trained_at=(None, None)):
        """Sounds is "…" until the checkpoints are read, "?" if they can't be.
        trained_at (replay csv or file date) shows until then."""
        item = self._items[name]
        details = self.app_state.get_model_details_if_read(name)
        when, source = details.trained_at if details else trained_at
        net_count = details.net_count if details else self.app_state.get_net_count(name)
        if details is None:
            sounds = "…"
        else:
            sounds = str(_sound_count(details.labels)) if details.labels else "?"
        trained = trained_when(when, source, now) if when else ""
        item.setText(TRAINED_COLUMN, trained)
        item.setText(SOUNDS_COLUMN, sounds)
        item.setText(NETS_COLUMN, str(net_count) if net_count else "")
        # A screen reader reads only the first column otherwise
        item.setData(MODEL_COLUMN, Qt.ItemDataRole.AccessibleTextRole,
                     f"{name}, trained {trained or 'unknown'}, {sounds} sounds, {net_count} nets")

    def _reader_start(self, names):
        # Before the reader's first torch.load: torch imported off the main
        # thread has segfaulted alongside Qt. Already loaded on Windows.
        try:
            import torch  # noqa: F401
        except Exception:
            pass  # Models with checkpoints then read as unreadable
        self.reader.read([name for name in names
                          if self.app_state.get_model_details_if_read(name) is None])

    def _current_model_name(self):
        item = self.model_list.currentItem()
        return item.data(MODEL_COLUMN, Qt.ItemDataRole.UserRole) if item else None

    def _on_model_selected(self, item, _previous=None):
        name = item.data(MODEL_COLUMN, Qt.ItemDataRole.UserRole) if item else None
        if name:
            self.reader.read_first(name)
        self._model_show(name)

    def _on_model_read(self, name):
        if name in self._items:
            self._model_row_update(name, datetime.now())
        if name == self._current_model_name():
            self._model_show(name)

    def _on_all_read(self):
        """Checkpoint dates can reorder the list. Once, at the end, so rows
        don't jump while being read."""
        self._model_list_refresh()

    # ---- details -------------------------------------------------------

    def _model_show(self, name):
        self.model_header.setVisible(name is not None)
        self.details.setVisible(name is not None)
        self.empty_state.setVisible(name is None)
        if name is None:
            return
        self.model_title.setText(name)
        details = self.app_state.get_model_details_if_read(name)
        if details is None:
            self.model_stats.setText("Reading…")
            self.facts.setText("")
            self.sound_tree.clear()
        elif not details.labels:
            self.model_stats.setText("Couldn't read this model")
            self.facts.setText(self._facts_html(name, details))
            self.sound_tree.clear()
        else:
            self.model_stats.setText(self._model_stats_text(details))
            self.facts.setText(self._facts_html(name, details))
            self._sound_tree_fill(details)

    def _model_stats_text(self, details):
        count = _sound_count(details.labels)
        sounds = f"{count} {'sound' if count == 1 else 'sounds'}"
        if BACKGROUND_LABEL in details.labels:
            sounds += f" + {BACKGROUND_LABEL}"
        if not details.net_count:
            return sounds
        nets = f"{details.net_count} {'net' if details.net_count == 1 else 'nets'}"
        return f"{sounds}   ·   {nets}"

    def _facts_rows(self, name, details):
        """[(label, value)], only the ones the model's files record."""
        rows = []
        accuracy = _model_accuracy(details.checkpoints, details.net_count)
        if accuracy is not None:
            rows.append(("Accuracy", percent_span([accuracy], 1)))
        when, source = details.trained_at
        if when:
            trained = trained_when(when, source, datetime.now())
            rows.append(("Trained", f"{trained} (file date)" if source == "mtime" else trained))
        rows.append(("File", html.escape(f"{name}.pkl")))
        if details.pkl_bytes:
            rows.append(("Size", file_size_text(details.pkl_bytes)))
        net_accuracies = _net_accuracies(details.checkpoints)
        if len(net_accuracies) > 1:
            rows.append(("Each net", percent_span(net_accuracies, 1)))
        epochs = _best_epochs_text(details.checkpoints)
        if epochs:
            rows.append(("Best epoch", epochs))
        return rows

    def _facts_html(self, name, details):
        t = theme.colors()
        return "<table cellspacing='0' cellpadding='3'>" + "".join(
            f"<tr><td style='color:{t['text_dim']}; padding-right:20px;'>{label}</td>"
            f"<td style='color:{t['text']};'>{value}</td></tr>"
            for label, value in self._facts_rows(name, details)) + "</table>"

    def _sound_tree_fill(self, details):
        t = theme.colors()
        accuracies = _sound_accuracies(details.checkpoints)
        # Off while filling, or every insert re-sorts
        self.sound_tree.setSortingEnabled(False)
        self.sound_tree.clear()
        for label in details.labels:
            scores = accuracies.get(label)
            note = "from all your recordings" if label == BACKGROUND_LABEL else ""
            item = SoundItem([label, percent_span(scores, 0) if scores else "", note])
            item.setTextAlignment(ACCURACY_COLUMN, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if scores:
                item.setData(ACCURACY_COLUMN, Qt.ItemDataRole.UserRole, min(scores))
            item.setForeground(NOTE_COLUMN, QColor(t["text_dim"]))
            self.sound_tree.addTopLevelItem(item)
        self.sound_tree.setSortingEnabled(True)

    def _on_open_models_folder(self):
        try:
            library_ops.open_in_file_manager(CLASSIFIER_FOLDER)
        except library_ops.LibraryOpError as exc:
            QMessageBox.warning(self, "Couldn't open folder", str(exc))


def _sound_count(labels):
    """Silence is a class the trainer makes, not a sound anyone recorded."""
    return len([label for label in labels if label != BACKGROUND_LABEL])


def _model_accuracy(checkpoints, net_count):
    """combined_accuracy from the net whose best epoch came last, else a lone
    net's accuracy. None if not recorded."""
    scored = [c for c in checkpoints
              if c.get("combined_accuracy") is not None and c.get("epoch") is not None]
    if scored:
        return max(scored, key=lambda c: c["epoch"])["combined_accuracy"]
    if net_count == 1 and checkpoints:
        return checkpoints[0].get("accuracy")
    return None


def _net_accuracies(checkpoints):
    return [c["accuracy"] for c in checkpoints if c.get("accuracy") is not None]


def _best_epochs_text(checkpoints):
    """'12' or '9-14'. +1: the trainer counts epochs from zero."""
    epochs = [c["epoch"] + 1 for c in checkpoints if c.get("epoch") is not None]
    if not epochs:
        return ""
    low, high = min(epochs), max(epochs)
    return str(low) if low == high else f"{low}-{high}"


def _sound_accuracies(checkpoints):
    """{label: [each net's accuracy on it]}, on the samples the trainer held
    back. Empty for models trained before label_accuracy was saved."""
    scores = {}
    for checkpoint in checkpoints:
        for label, accuracy in (checkpoint.get("label_accuracy") or {}).items():
            scores.setdefault(label, []).append(accuracy)
    return scores


class SoundItem(QTreeWidgetItem):
    """Sorts the accuracy column by its lowest net. As text, 100% sorts below 97%."""

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        mine = self.data(column, Qt.ItemDataRole.UserRole)
        theirs = other.data(column, Qt.ItemDataRole.UserRole)
        if mine is None or theirs is None:
            return super().__lt__(other)
        return mine < theirs


class ModelDetailsReader(QThread):
    """Reads models' checkpoints off the UI thread, one model at a time so rows
    fill in from the top. read() queues more, read_first() jumps the queue."""
    model_read = pyqtSignal(str)
    all_read = pyqtSignal()

    def __init__(self, app_state, parent):
        super().__init__(parent)
        self.app_state = app_state
        self._pending = []
        self._lock = threading.Lock()
        self.finished.connect(self._on_finished)
        QApplication.instance().aboutToQuit.connect(self._stop)

    def read(self, names):
        with self._lock:
            self._pending += [name for name in names if name not in self._pending]
        if names and not self.isRunning():
            self.start()

    def read_first(self, name):
        with self._lock:
            if name in self._pending:
                self._pending.remove(name)
                self._pending.insert(0, name)

    def run(self):
        while (name := self._pending_pop()) is not None:
            self.app_state.get_model_details(name)
            self.model_read.emit(name)

    def _pending_pop(self):
        with self._lock:
            return self._pending.pop(0) if self._pending else None

    def _on_finished(self):
        # read() can arrive after the last pop, while still running, and not restart
        with self._lock:
            more = bool(self._pending)
        if more:
            self.start()
        else:
            self.all_read.emit()

    def _stop(self):
        """The read in progress finishes, the rest are dropped. A QThread
        deleted while running aborts the app."""
        with self._lock:
            self._pending.clear()
        self.wait()
