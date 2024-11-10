from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class CellViewerWidget(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self.set_2d_mode_button = QPushButton("2D View Mode")
        self.set_2d_mode_button.setCheckable(True)
        self.set_2d_mode_button.setChecked(True)
        self.set_2d_mode_button.clicked.connect(self.set_2d_mode)
        self.set_3d_mode_button = QPushButton("3D View Mode")
        self.set_3d_mode_button.setCheckable(True)
        self.set_3d_mode_button.clicked.connect(self.set_3d_mode)

        self.mode_buttons = QWidget()
        self.mode_buttons.setLayout(QHBoxLayout())
        self.mode_buttons.layout().addWidget(self.set_2d_mode_button)
        self.mode_buttons.layout().addWidget(self.set_3d_mode_button)

        self.load_full_res_button = QPushButton("Full Resolution Image")
        self.load_full_res_button.clicked.connect(
            self.controller.load_full_res_image_async
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mode_buttons)
        self.layout().addWidget(self.load_full_res_button)

    def set_2d_mode(self, _=None):
        self.set_2d_mode_button.setChecked(True)
        self.set_3d_mode_button.setChecked(False)
        self.controller.set_2d_mode()

    def set_3d_mode(self, _=None):
        self.set_3d_mode_button.setChecked(True)
        self.set_2d_mode_button.setChecked(False)
        self.controller.set_3d_mode()
