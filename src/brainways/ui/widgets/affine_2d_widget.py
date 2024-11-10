from typing import Callable, Dict, Tuple

from magicgui import magicgui
from qtpy import QtCore
from qtpy.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget


class Affine2DWidget(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self.params_widget = magicgui(
            self.params,
            auto_call=True,
            angle={
                "label": "Angle",
                "widget_type": "FloatSlider",
                "min": -90,
                "max": 90,
            },
            tx={
                "label": "Left-Right",
                "widget_type": "FloatSlider",
                "max": 1,
            },
            ty={
                "label": "Up-Down",
                "widget_type": "FloatSlider",
                "max": 1,
            },
            sx={
                "label": "Horizontal Scale",
                "widget_type": "FloatSlider",
                "min": 0.01,
                "max": 3,
            },
            sy={
                "label": "Vertical Scale",
                "widget_type": "FloatSlider",
                "min": 0.01,
                "max": 3,
            },
        )
        self.params_widget.native.layout().setContentsMargins(0, 0, 0, 0)
        for widget in self.params_widget.native.findChildren(QWidget):
            widget.installEventFilter(self)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.controller.reset_params)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.params_widget.native)
        self.layout().addWidget(self.reset_button)

    def eventFilter(self, object, event) -> bool:
        if event.type() == QtCore.QEvent.KeyPress:
            self.controller.widget.event(event)
            return True
        return super().eventFilter(object, event)

    def show_help(self, key_bindings: Dict[str, Tuple[Callable, str]]):
        message = "\n".join(
            f"{f'[{key}]':<15} {help}" for key, (func, help) in key_bindings.items()
        )
        message = f"<pre>{message}</pre>"
        QMessageBox.about(self, "Keys", message)

    def params(
        self,
        angle: float,
        tx: float,
        ty: float,
        sx: float,
        sy: str,
    ):
        self.controller.on_params_changed(
            angle=angle,
            tx=tx,
            ty=ty,
            sx=sx,
            sy=sy,
        )

    def set_params(
        self,
        angle: float,
        tx: float,
        ty: float,
        sx: float,
        sy: float,
    ):
        self.params_widget._auto_call = False
        self.params_widget.angle.value = angle
        self.params_widget.tx.value = tx
        self.params_widget.ty.value = ty
        self.params_widget.sx.value = sx
        self.params_widget.sy.value = sy
        self.params_widget._auto_call = True

    def set_ranges(self, tx: Tuple[float, float], ty: Tuple[float, float]):
        self.params_widget.tx.min = tx[0]
        self.params_widget.tx.max = tx[1]
        self.params_widget.ty.min = ty[0]
        self.params_widget.ty.max = ty[1]
