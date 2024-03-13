from typing import TYPE_CHECKING, Callable, Dict, Tuple

from magicgui import magicgui
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QMessageBox, QPushButton, QStyle, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from napari_brainways.controllers.registration_controller import (
        RegistrationController,
    )


class RegistrationView(QWidget):
    def __init__(self, controller: "RegistrationController"):
        super().__init__()

        self.controller = controller

        self.registration_params_widget = magicgui(
            self.registration_params,
            auto_call=True,
            ap={
                "label": "AP",
                "widget_type": "FloatSlider",
                "max": 1,
            },
            rot_horizontal={
                "label": "H-Rot",
                "widget_type": "FloatSlider",
                "min": -15,
                "max": 15,
            },
            rot_sagittal={
                "label": "S-Rot",
                "widget_type": "FloatSlider",
                "min": -15,
                "max": 15,
            },
            hemisphere={
                "label": "Hem",
                "widget_type": "RadioButtons",
                "orientation": "horizontal",
                "choices": [
                    ("Left", "left"),
                    ("Both", "both"),
                    ("Right", "right"),
                ],
            },
        )
        for widget in self.registration_params_widget.native.findChildren(QWidget):
            widget.installEventFilter(self)

        self.run_model_button = QPushButton("Automatic Registration")
        self.run_model_button.clicked.connect(self.controller.on_run_model_button_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.registration_params_widget.native)
        self.layout().addWidget(self.run_model_button)

    def show_help(self, key_bindings: Dict[str, Tuple[Callable, str]]):
        message = "\n".join(
            f"{f'[{key}]':<15} {help}" for key, (func, help) in key_bindings.items()
        )
        message = f"<pre>{message}</pre>"
        QMessageBox.about(self, "Keys", message)

    def modify_ap(self, _=None, value: int = 0):
        self.registration_params_widget.ap.value += value

    def modify_rot_horizontal(self, _=None, value: int = 0):
        self.registration_params_widget.rot_horizontal.value += value

    def modify_rot_sagittal(self, _=None, value: int = 0):
        self.registration_params_widget.rot_sagittal.value += value

    def modify_hemisphere(self, _=None, value: str = "both"):
        self.registration_params_widget.hemisphere.value = value

    def eventFilter(self, object, event) -> bool:
        if event.type() == QtCore.QEvent.KeyPress:
            self.controller.widget.event(event)
            return True
        return super().eventFilter(object, event)

    def registration_params(
        self,
        ap: float,
        rot_horizontal: float,
        rot_sagittal: float,
        hemisphere: str,
    ):
        self.controller.on_params_changed(
            ap=ap,
            rot_horizontal=rot_horizontal,
            rot_sagittal=rot_sagittal,
            hemisphere=hemisphere,
        )

    def set_registration_params(
        self,
        ap: float,
        rot_horizontal: float,
        rot_sagittal: float,
        hemisphere: str,
    ):
        widget = self.registration_params_widget
        widget._auto_call = False
        widget.ap.value = ap
        widget.rot_horizontal.value = rot_horizontal
        widget.rot_sagittal.value = rot_sagittal
        widget.hemisphere.value = hemisphere
        widget._auto_call = True

    @property
    def ap_limits(self):
        return (
            self.registration_params_widget.ap.min,
            self.registration_params_widget.ap.max,
        )

    @ap_limits.setter
    def ap_limits(self, value: Tuple[int, int]):
        self.registration_params_widget.ap.min = value[0]
        self.registration_params_widget.ap.max = value[1]

    def update_model(self, ap_min: int, ap_max: int):
        self.ap_limits = (ap_min, ap_max)
        if self.controller.model_available():
            self.run_model_button.setIcon(QtGui.QIcon())
            self.run_model_button.setToolTip("")
        else:
            self.run_model_button.setIcon(
                self.style().standardIcon(QStyle.SP_MessageBoxWarning)
            )
            self.run_model_button.setToolTip(
                "Automatic registration model is not installed"
            )
