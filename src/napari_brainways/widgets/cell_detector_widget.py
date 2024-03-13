from typing import Tuple

import magicgui
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class CellDetectorWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.stardist_label = QLabel(
            text='by <a href="https://github.com/stardist/stardist">StarDist</a>'
        )
        self.stardist_label.setOpenExternalLinks(True)

        self.cell_detector_params_widget = magicgui.magicgui(
            self.controller.on_params_changed,
            normalizer={
                "label": "Normalizer",
                "widget_type": "RadioButtons",
                "orientation": "horizontal",
                "choices": [
                    ("Quantile", "quantile"),
                    # ("Value", "value"),
                    ("CLAHE", "clahe"),
                    ("None", "none"),
                ],
            },
            # normalizer_range={
            #     "label": "Range",
            #     "widget_type": "RangeSlider",
            #     "min": 0,
            #     "max": 1000,
            #     "step": 1,
            # },
            min_value={
                "label": "Low",
                "widget_type": "Slider",
                "min": 0,
                "max": 1000,
                "step": 1,
            },
            max_value={
                "label": "High",
                "widget_type": "Slider",
                "min": 0,
                "max": 1000,
                "step": 1,
            },
            auto_call=True,
        )
        self.cell_detector_params_widget.native.layout().setContentsMargins(0, 0, 0, 0)

        self.run_preview_button = QPushButton("Run on preview")
        self.run_preview_button.clicked.connect(
            self.controller.run_cell_detector_preview_async
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.stardist_label)
        self.layout().addWidget(self.cell_detector_params_widget.native)
        self.layout().addWidget(self.run_preview_button)

    def set_cell_detector_params(
        self,
        normalizer: str,
        normalizer_range: Tuple[float, float],
        unique: bool,
    ):
        widget = self.cell_detector_params_widget
        widget._auto_call = False
        widget.normalizer.value = normalizer
        widget.min_value.value = int(normalizer_range[0] * 1000)
        widget.max_value.value = int(normalizer_range[1] * 1000)
        widget.unique.value = unique
        widget._auto_call = True
