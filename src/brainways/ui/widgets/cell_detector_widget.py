from typing import TYPE_CHECKING, Tuple

import magicgui
from magicgui.widgets import request_values
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from brainways.project.info_classes import MaskFileFormat, SliceSelection
from brainways.ui.widgets.warning_dialog import show_warning_dialog

if TYPE_CHECKING:
    from brainways.ui.controllers.cell_detector_controller import CellDetectorController


class CellDetectorWidget(QWidget):
    def __init__(self, controller: "CellDetectorController"):
        super().__init__()
        self.controller = controller
        stardist_label = QLabel(
            text='by <a href="https://github.com/stardist/stardist">StarDist</a>'
        )
        stardist_label.setOpenExternalLinks(True)

        self.cell_detector_params_widget = magicgui.magicgui(
            self.controller.on_params_changed,
            normalizer={
                "label": "Normalizer",
                "widget_type": "RadioButtons",
                "orientation": "horizontal",
                "choices": [
                    ("Quantile", "quantile"),
                    ("Value", "value"),
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
            min_value={"label": "Normalizer Low Value", "widget_type": "LineEdit"},
            max_value={"label": "Normalizer High Value", "widget_type": "LineEdit"},
            min_cell_size_value={"label": "Min Cell Size (um)"},
            max_cell_size_value={"label": "Max Cell Size (um)"},
            auto_call=True,
        )
        self.cell_detector_params_widget.native.layout().setContentsMargins(0, 0, 0, 0)

        run_preview_button = QPushButton("Preview cell detector")
        run_preview_button.clicked.connect(
            self.controller.run_cell_detector_preview_async
        )

        run_cell_detector_button = QPushButton("Run cell detector on...")
        run_cell_detector_button.clicked.connect(self.run_cell_detector)

        self.setLayout(QVBoxLayout())
        layout = self.layout()
        assert layout is not None
        layout.addWidget(stardist_label)
        layout.addWidget(self.cell_detector_params_widget.native)
        layout.addWidget(run_preview_button)
        layout.addWidget(run_cell_detector_button)

    def set_cell_detector_params(
        self,
        normalizer: str,
        normalizer_range: Tuple[float, float],
        cell_size_range: Tuple[float, float],
        unique: bool,
    ):
        widget = self.cell_detector_params_widget
        widget._auto_call = False
        widget.normalizer.value = normalizer
        widget.min_value.value = str(normalizer_range[0])
        widget.max_value.value = str(normalizer_range[1])
        widget.min_cell_size_value.value = cell_size_range[0]
        widget.max_cell_size_value.value = cell_size_range[1]
        widget.unique.value = unique
        widget._auto_call = True

    def run_cell_detector(self):
        DONT_SAVE_CELL_DETECTION_MASK_VALUE = "Don't Save"
        values = request_values(
            title="Run Cell Detector",
            slice_selection=dict(
                value=SliceSelection.CURRENT_SLICE.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in SliceSelection],
                    tooltip="Which slices to run the cell detector on",
                ),
                annotation=str,
                label="Slice Selection",
            ),
            save_cell_detection_masks_file_format=dict(
                value=DONT_SAVE_CELL_DETECTION_MASK_VALUE,
                widget_type="ComboBox",
                options=dict(
                    choices=[DONT_SAVE_CELL_DETECTION_MASK_VALUE]
                    + [e.value for e in MaskFileFormat],
                    tooltip="File format to save the cell detection masks to",
                ),
                annotation=str,
                label="Cell Detection Masks File Format",
            ),
            resume=dict(
                value=True,
                widget_type="CheckBox",
                options=dict(tooltip="Resume the cell detection if it was interrupted"),
                annotation=bool,
                label="Resume Previous Run",
            ),
        )
        if values is None:
            return

        slice_selection = SliceSelection(values["slice_selection"])
        if not values["resume"]:
            if not show_warning_dialog(
                f'Unchecking "Resume Previous Run" will delete all previous cell detections in {slice_selection.value.lower()}.\n\nDo you want to continue?'
            ):
                return

        save_cell_detection_masks_file_format = (
            None
            if values["save_cell_detection_masks_file_format"]
            == DONT_SAVE_CELL_DETECTION_MASK_VALUE
            else MaskFileFormat(values["save_cell_detection_masks_file_format"])
        )

        self.controller.run_cell_detector_async(
            slice_selection=slice_selection,
            resume=values["resume"],
            save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
        )
