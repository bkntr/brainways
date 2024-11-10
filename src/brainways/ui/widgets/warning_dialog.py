from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QTextEdit,
    QVBoxLayout,
)


class WarningDialog(QDialog):
    def __init__(self, text: str, title: str = "Warnings", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        # Get screen size
        screen = QApplication.primaryScreen()
        assert screen is not None, "No screen found"
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Set dialog size to 50% of screen size
        self.resize(screen_width // 2, screen_height // 2)

        self._layout = QVBoxLayout(self)

        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        icon = QMessageBox.standardIcon(QMessageBox.Warning)
        self.icon_label.setPixmap(icon)  # Adjust size as needed

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self.text_edit.setPlainText(text)

        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.addWidget(self.icon_label, alignment=Qt.AlignTop)  # type: ignore[attr-defined]
        self.horizontal_layout.addWidget(self.text_edit)

        self._layout.addLayout(self.horizontal_layout)

        self.button_layout = QHBoxLayout()

        # Add spacer item to push buttons to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.button_layout.addItem(spacer)

        style = self.style()
        assert style is not None, "No style found"

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setIcon(style.standardIcon(QStyle.SP_DialogCancelButton))  # type: ignore[attr-defined]
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.setIcon(style.standardIcon(QStyle.SP_DialogOkButton))  # type: ignore[attr-defined]
        self.ok_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.ok_button)

        self._layout.addLayout(self.button_layout)


def show_warning_dialog(text: str) -> bool:
    dialog = WarningDialog(text)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return True
    else:
        return False
