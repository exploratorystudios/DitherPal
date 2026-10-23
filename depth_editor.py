"""A small neural-depth preview with one shared depth-strength control."""
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QSlider, QVBoxLayout

from ditherpal_depth import image_and_depth, scale_depth


class DepthPreviewJob(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def update_status(self, message):
        if self.isInterruptionRequested():
            raise InterruptedError()
        self.status.emit(message)

    def run(self):
        try:
            _, near = image_and_depth(self.source, self.update_status)
            if not self.isInterruptionRequested():
                self.ready.emit(near)
        except Exception as error:
            if not self.isInterruptionRequested():
                self.failed.emit(str(error))


class DepthMapDialog(QDialog):
    strength_changed = pyqtSignal(int)

    def __init__(self, source, strength=100, parent=None):
        super().__init__(parent)
        self.source = str(source)
        self.raw_depth = None
        self.closing = False
        self.setWindowTitle('Neural Depth — '+Path(source).name)
        self.resize(450, 520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('White = near · Black = far'))
        self.preview = QLabel('Estimating neural depth…')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(300, 300)
        self.preview.setStyleSheet('background: #1e293b; border-radius: 6px;')
        layout.addWidget(self.preview, 1)
        self.value_label = QLabel()
        layout.addWidget(self.value_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 200)
        self.slider.setValue(strength)
        self.slider.valueChanged.connect(self.strength_updated)
        layout.addWidget(self.slider)
        note = QLabel('0% = flat · 100% = neural depth · 200% = stronger depth')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel('Depth is shared by pixel art and 3D camera rendering. Videos preview their first frame.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.strength_updated(strength)
        self.job = DepthPreviewJob(source)
        self.job.ready.connect(self.depth_ready)
        self.job.failed.connect(self.failed)
        self.job.status.connect(self.status.setText)
        self.job.finished.connect(self.job_finished)
        self.job.finished.connect(self.job.deleteLater)
        self.job.start()

    def strength_updated(self, value):
        self.value_label.setText(f'Depth strength: {value}%')
        self.strength_changed.emit(value)
        self.draw_depth()

    def depth_ready(self, near):
        if self.closing:
            return
        self.raw_depth = near
        self.draw_depth()
        self.status.setText('Ready. This strength is used for the next render; moving the slider does not repeat inference.')

    def draw_depth(self):
        if self.raw_depth is None:
            return
        gray = np.uint8(np.round(scale_depth(self.raw_depth, self.slider.value())*255))
        image = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format.Format_Grayscale8)
        picture = QPixmap.fromImage(image).scaled(410, 350, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(picture)

    def failed(self, message):
        self.preview.setText('Depth unavailable')
        self.status.setText(message)

    def job_finished(self):
        self.job = None
        if self.closing:
            self.close()

    def closeEvent(self, event):
        self.closing = True
        if self.job is not None:
            self.job.requestInterruption()
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)
