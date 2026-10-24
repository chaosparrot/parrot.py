"""Home landing page. Re-orients a user returning after months away: the
workflow steps, each with where it stands."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QSizePolicy, QMessageBox
)

from config.config import CLASSIFIER_FOLDER, DATA_DIR
from gui import components, theme
from gui.services import library_ops


class _StepCard(QFrame):
    """One numbered bubble in the Record -> Train -> Use strip."""

    def __init__(self, number, title, action_text, parent=None):
        super().__init__(parent)
        self.setObjectName("stepCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 18, 18, 16)
        v.setSpacing(6)

        self.bubble = QLabel(str(number))
        self.bubble.setObjectName("stepBubble")
        self.bubble.setFixedSize(36, 36)
        self.bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bubble_row = QHBoxLayout()
        bubble_row.setSpacing(10)
        bubble_row.addWidget(self.bubble)
        self.number = number
        self.title = QLabel(title)
        self.title.setWordWrap(True)
        bubble_row.addWidget(self.title, 1,
                             Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(bubble_row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch()

        self.action = QPushButton(action_text)
        self.action.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        v.addWidget(self.action)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_state(self, done, current, status_text, action_primary=False):
        """action_primary keeps the bright styling on the button even when the
        step is done - the ongoing main action."""
        t = theme.colors()
        self.bubble.setText("✓" if done else str(self.number))
        bubble_bg = t["accent"] if done else (t["button"] if not current else t["panel"])
        bubble_fg = t["accent_text"] if done else t["text_bright"]
        bubble_border = t["accent"] if (done or current) else t["border"]
        self.bubble.setStyleSheet(
            f"QLabel#stepBubble {{ background-color: {bubble_bg}; color: {bubble_fg}; "
            f"border: 2px solid {bubble_border}; border-radius: 18px; "
            f"font-size: {theme.TYPE_SCALE['card']}px; font-weight: bold; }}")
        border = t["accent"] if current else t["border"]
        self.setStyleSheet(
            f"QFrame#stepCard {{ background-color: {t['card']}; "
            f"border: 1px solid {border}; "
            f"border-radius: {t['radius_card']}; }}")
        self.title.setStyleSheet(
            components.heading_style("card") + " border: none;")
        self.status.setStyleSheet(
            f"color: {t['accent'] if done else t['text']}; border: none;")
        self.status.setText(status_text)
        # Screen readers skip the label, so the button carries the status
        self.action.setAccessibleDescription(status_text)
        self.action.setObjectName(
            "primaryAction" if (current or action_primary) else "")
        self.action.setStyleSheet(
            components.primary_button_style()
            if (current or action_primary) else "")


class HomePage(QWidget):
    """
    ┌──────────────────────────────────────────────────────────┐
    │ hero_title                                               │
    │ hero_sub                                                 │
    │                                                          │
    │ ┌ step_record ───┐ ┌ step_train ────┐ ┌ step_use ──────┐ │
    │ │ _StepCard      │ │ _StepCard      │ │ _StepCard      │ │
    │ └────────────────┘ └────────────────┘ └────────────────┘ │
    └──────────────────────────────────────────────────────────┘
    """
    navigate = pyqtSignal(str)   # tab name, handled by MainWindow

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state

        self._setup_ui()

        # deferred so the window paints before the first scan of the data dir
        QTimer.singleShot(0, self._refresh)

    # ---- ui --------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setAccessibleName("Home")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        wrapper = QWidget()
        wrap_layout = QHBoxLayout(wrapper)
        wrap_layout.setContentsMargins(32, 28, 32, 28)
        body = QWidget()
        body.setMaximumWidth(980)
        # heavy body stretch: side gutters absorb only the space past maxWidth,
        # and collapse to ~0 when the window narrows
        wrap_layout.addStretch(1)
        wrap_layout.addWidget(body, 1000)
        wrap_layout.addStretch(1)
        scroll.setWidget(wrapper)

        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(16)

        self.hero_title = QLabel("Parrot.py")
        v.addWidget(self.hero_title)
        self.hero_sub = QLabel(
            "Turn noises into instant actions on your computer.")
        self.hero_sub.setWordWrap(True)
        v.addWidget(self.hero_sub)

        steps_row = QHBoxLayout()
        steps_row.setSpacing(12)
        self.step_record = _StepCard(1, "Record sounds", "Open sounds")
        self.step_train = _StepCard(2, "Train a model", "Open models")
        self.step_use = _StepCard(3, "Use your model", "Open models folder")
        self.step_record.action.clicked.connect(lambda: self.navigate.emit("Sounds"))
        self.step_train.action.clicked.connect(lambda: self.navigate.emit("Models"))
        self.step_use.action.clicked.connect(self._on_open_models_folder)
        self.step_use.action.setToolTip(
            "The folder holding your .pkl files, to copy one into an integration")
        for card in (self.step_record, self.step_train, self.step_use):
            steps_row.addWidget(card, 1)
        v.addLayout(steps_row)

        data_row = QHBoxLayout()
        data_row.addStretch()
        open_data_btn = QPushButton("Open data folder")
        open_data_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        open_data_btn.setToolTip("Recordings and models, in one folder")
        open_data_btn.clicked.connect(self._on_open_data_folder)
        data_row.addWidget(open_data_btn)
        v.addLayout(data_row)

        # Non-zero factor: leftover height goes to the bottom of the page, not
        # into the step cards.
        v.addStretch(1)

        t = theme.colors()
        self.hero_title.setStyleSheet(components.heading_style("hero"))
        self.hero_sub.setStyleSheet(
            f"font-size: {theme.TYPE_SCALE['card']}px; color: {t['text_dim']};")

    def _on_open_data_folder(self):
        self._open_folder(DATA_DIR)

    def _on_open_models_folder(self):
        self._open_folder(CLASSIFIER_FOLDER)

    def _open_folder(self, path):
        try:
            library_ops.open_in_file_manager(path)
        except library_ops.LibraryOpError as exc:
            QMessageBox.warning(self, "Couldn't open folder", str(exc))

    # ---- content ----------------------------------------------------------

    def _refresh(self):
        labels = self.app_state.get_sound_labels()
        model_names = self.app_state.get_model_names()

        step1_done = len(labels) >= 2
        if not labels:
            s1 = "No sounds yet."
        elif len(labels) == 1:
            s1 = "1 sound. A model needs 2."
        else:
            s1 = f"{len(labels)} sounds recorded."

        step2_done = bool(model_names)
        if not model_names:
            s2 = "No models yet."
        else:
            noun = "model" if len(model_names) == 1 else "models"
            s2 = f"{len(model_names)} {noun} trained."

        # Nothing outside the app can be observed, so having a model to hand
        # over is as far as this step can honestly get.
        step3_done = step2_done
        s3 = ("Copy a .pkl from the models folder into your integration."
              if step3_done else "Train a model first.")

        current = next((i for i, done in enumerate(
            (step1_done, step2_done, step3_done)) if not done), None)
        self.step_record.set_state(step1_done, current == 0, s1)
        self.step_train.set_state(step2_done, current == 1, s2)
        # Bright once done, but only when no earlier step is unfinished:
        # one accent button at a time.
        self.step_use.set_state(
            step3_done, current == 2, s3,
            action_primary=step3_done and current is None)
        self.step_use.action.setEnabled(step3_done)
