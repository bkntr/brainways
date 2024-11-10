from typing import Callable, Dict, Tuple

from qtpy.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget


class TpsWidget(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.controller.reset_params)
        self.elastix_button = QPushButton("Elastix")
        self.elastix_button.clicked.connect(self.controller.run_elastix_async)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.reset_button)
        self.layout().addWidget(self.elastix_button)

    def show_help(self, key_bindings: Dict[str, Tuple[Callable, str]]):
        message = "\n".join(
            f"{f'[{key}]':<15} {help}" for key, (func, help) in key_bindings.items()
        )
        message = f"<pre>{message}</pre>"
        QMessageBox.about(self, "Keys", message)
