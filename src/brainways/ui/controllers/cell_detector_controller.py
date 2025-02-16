from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple, Union

import napari
import napari.layers
import numpy as np
from PyQt5.QtWidgets import QApplication

from brainways.pipeline.brainways_params import BrainwaysParams, CellDetectorParams
from brainways.pipeline.cell_detector import CellDetector
from brainways.project.info_classes import MaskFileFormat, SliceSelection
from brainways.ui.controllers.base import Controller
from brainways.ui.utils.general_utils import update_layer_contrast_limits
from brainways.ui.widgets.cell_detector_widget import CellDetectorWidget

if TYPE_CHECKING:
    from brainways.ui.brainways_ui import BrainwaysUI


class CellDetectorController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.model: CellDetector | None = None
        self.widget = CellDetectorWidget(self)
        self.widget.hide()

        self.input_layer = None
        self.preview_box_layer: napari.layers.Points | None = None
        self.crop_layer: napari.layers.Image | None = None
        self.normalized_crop_layer: napari.layers.Image | None = None
        self.cell_mask_layer: napari.layers.Image | None = None
        self._run_lock = False
        self._params: BrainwaysParams | None = None
        self._crop = None

    @property
    def name(self) -> str:
        return "Cell Detection"

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return params.cell is not None

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return True

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        return params

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params

        if self._params.cell is not None:
            cell_detector_params = self._params.cell
        else:
            cell_detector_params = self.ui.project.settings.default_cell_detector_params

        if image is not None:
            self.input_layer.data = image
            self.ui.update_layer_contrast_limits(self.input_layer)

            x0, y0, w, h = self.selected_bounding_box(
                image, point=(0.5 * image.shape[0], 0.5 * image.shape[1])
            )
            x1 = x0 + w
            y1 = y0 + h
            self.preview_box_layer.data = (
                np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
                * self.input_layer.data.shape
            )

        if not from_ui:
            self.widget.set_cell_detector_params(
                normalizer=cell_detector_params.normalizer,
                normalizer_range=cell_detector_params.normalizer_range,
                cell_size_range=cell_detector_params.cell_size_range,
                unique=self._params.cell is not None,
            )

        self.on_click()
        self.set_preview_affine()

        self.ui.viewer.reset_view()

    def load_model(self) -> None:
        if (
            self.model is None
            or self.model.custom_model_dir
            != self.ui.project.settings.cell_detector_custom_model_dir
        ):
            self.model = self.ui.project.get_cell_detector()

    def open(self) -> None:
        if self._is_open:
            return

        self.input_layer = self.ui.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Input",
        )
        self.input_layer.events.contrast_limits.connect(self.ui.set_contrast_limits)
        self.input_layer.translate = (0, 0)

        self.preview_box_layer = self.ui.viewer.add_shapes(
            name="Region selector",
            face_color="#ffffff00",
            edge_color="red",
        )
        self.preview_box_layer.mouse_double_click_callbacks.append(self.on_click)

        self.crop_layer = self.ui.viewer.add_image(
            np.zeros((100, 100), np.uint8),
            name="Preview",
        )
        self.crop_layer.events.contrast_limits.connect(self.ui.set_contrast_limits)
        self.normalized_crop_layer = self.ui.viewer.add_image(
            np.zeros((100, 100), np.uint8),
            name="Preview (Normalized)",
            visible=False,
        )
        self.normalized_crop_layer.events.contrast_limits.connect(
            self.ui.set_contrast_limits
        )
        self.cell_mask_layer = self.ui.viewer.add_labels(
            np.zeros((10, 10), np.uint8), name="Cells"
        )
        self.ui.viewer.layers.selection.active = self.preview_box_layer

        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        self.widget.hide()
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.preview_box_layer)
        self.ui.viewer.layers.remove(self.crop_layer)
        self.ui.viewer.layers.remove(self.normalized_crop_layer)
        self.ui.viewer.layers.remove(self.cell_mask_layer)
        QApplication.instance().processEvents()

        self._image = None
        self._params = None
        self._crop = None

        self.input_layer = None
        self.preview_box_layer = None
        self.crop_layer = None
        self.normalized_crop_layer = None
        self.cell_mask_layer = None
        self._image_reader = None
        self._is_open = False

    @property
    def _preview_translate(self):
        self._check_is_open()
        ty = 0
        tx = self.input_layer.data.shape[1]
        return ty, tx

    @property
    def _preview_scale(self):
        self._check_is_open()
        scale = self.input_layer.data.shape[0] / self.crop_layer.data.shape[0]
        return scale, scale

    def set_preview_affine(self):
        self.crop_layer.translate = self._preview_translate
        self.crop_layer.scale = self._preview_scale
        self.normalized_crop_layer.translate = self._preview_translate
        self.normalized_crop_layer.scale = self._preview_scale
        self.cell_mask_layer.translate = self._preview_translate
        self.cell_mask_layer.scale = self._preview_scale

    def _on_cell_detector_returned(self, mask: np.ndarray):
        assert self.model is not None
        if self._params.cell is not None:
            cell_detector_params = self._params.cell
        else:
            cell_detector_params = self.ui.project.settings.default_cell_detector_params
        normalizer = self.model.get_normalizer(cell_detector_params)
        if normalizer is not None:
            self.normalized_crop_layer.data = normalizer.before(
                self._crop, axes=None
            ).squeeze()
            update_layer_contrast_limits(
                self.normalized_crop_layer, contrast_limits_quantiles=(0.0, 1.0)
            )
            self.crop_layer.visible = False
            self.normalized_crop_layer.visible = True
        self.cell_mask_layer.data = mask
        self.cell_mask_layer.visible = True
        self.ui.viewer.layers.selection = {self.preview_box_layer}

    def on_params_changed(
        self,
        normalizer: str,
        min_value: Union[float, str],
        max_value: Union[float, str],
        min_cell_size_value: float,
        max_cell_size_value: float,
        unique: bool = False,
    ):
        min_value = float(min_value)
        max_value = float(max_value)

        if max_value <= min_value:
            max_value = min_value + 0.001
        normalizer_range = (min_value, max_value)
        cell_size_range = (min_cell_size_value, max_cell_size_value)
        cell_detector_params = CellDetectorParams(
            normalizer=normalizer,
            normalizer_range=normalizer_range,
            cell_size_range=cell_size_range,
        )
        if unique:
            # in unique mode, set cell detector params unique to current slice
            self._params = replace(self._params, cell=cell_detector_params)
        elif self._params.cell is not None:
            # if changing from unique mode to non-unique mode, remove unique and set
            # default params
            self._params = replace(self._params, cell=None)
            p = self.ui.project.settings.default_cell_detector_params
            self.widget.set_cell_detector_params(
                normalizer=p.normalizer,
                normalizer_range=p.normalizer_range,
                cell_size_range=p.cell_size_range,
                unique=False,
            )
        else:
            # in default mode, change project default settings
            self.ui.project.settings = replace(
                self.ui.project.settings,
                default_cell_detector_params=cell_detector_params,
            )

    def _run_cell_detector_on_preview(self):
        self.load_model()
        self.ui.save_subject()

        if self._params.cell is not None:
            cell_detector_params = self._params.cell
        else:
            cell_detector_params = self.ui.project.settings.default_cell_detector_params
        return self.model.run_cell_detector(
            image=self._crop,
            params=cell_detector_params,
            physical_pixel_sizes=self.ui.current_document.physical_pixel_sizes,
        )

    def run_cell_detector_preview_async(self):
        self._check_is_open()

        self.ui.do_work_async(
            self._run_cell_detector_on_preview,
            return_callback=self._on_cell_detector_returned,
            progress_label="Running cell detector on preview...",
        )

    def run_cell_detector_async(
        self,
        slice_selection: SliceSelection,
        resume: bool,
        save_cell_detection_masks_file_format: Optional[MaskFileFormat],
    ):
        assert self.ui.project is not None

        slice_infos = self.ui.get_slice_selection(slice_selection)

        return self.ui.do_work_async(
            self.ui.project.run_cell_detector_iter,
            slice_infos=slice_infos,
            resume=resume,
            save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
            progress_label=f"Running Cell Detector on {slice_selection.value}...",
            progress_max_value=len(slice_infos),
        )

    @property
    def params(self) -> BrainwaysParams:
        return self._params

    def selected_bounding_box(
        self, image: np.ndarray | None = None, point: Tuple[float, float] | None = None
    ):
        """

        :return: x, y, w, h
        """
        if image is None:
            image = self.input_layer.data

        if point is None:
            box = self.preview_box_layer.data[-1] / image.shape
            x = box[0, 1]
            y = box[0, 0]
            w = box[1, 1] - box[0, 1]
            h = box[2, 0] - box[0, 0]
            return x, y, w, h

        image_height = image.shape[0]
        image_width = image.shape[1]

        y = point[0] / image_height
        x = point[1] / image_width

        w = min(4096 / self.ui.current_document.image_size[1], 1)
        h = min(4096 / self.ui.current_document.image_size[0], 1)

        x0 = min(max(x - w / 2, 0), 1 - w)
        y0 = min(max(y - h / 2, 0), 1 - h)

        return x0, y0, w, h

    def on_click(self, layer=None, event=None):
        if self._run_lock:
            with self.preview_box_layer.events.data.blocker():
                self.preview_box_layer.selected_data = {
                    self.preview_box_layer.data.shape[0] - 1
                }
                self.preview_box_layer.remove_selected()
            return

        if event is not None:
            with self.preview_box_layer.events.data.blocker():
                x, y, w, h = self.selected_bounding_box(point=event.position)
                self.preview_box_layer.data = (
                    np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]])
                    * self.input_layer.data.shape
                )

        x, y, w, h = self.selected_bounding_box()
        x0 = int(round(x * self.ui.current_document.image_size[1]))
        y0 = int(round(y * self.ui.current_document.image_size[0]))
        x1 = int(round((x + w) * self.ui.current_document.image_size[1]))
        y1 = int(round((y + h) * self.ui.current_document.image_size[0]))

        highres_crop = (
            self.ui.current_document.image_reader()
            .get_image_dask_data(
                "YX",
                X=slice(x0, x1),
                Y=slice(y0, y1),
                C=self.ui.project.settings.channel,
            )
            .compute()
        )

        # if self._params.cell is not None:
        #     cell_detector_params = self._params.cell
        # else:
        #     cell_detector_params = self.ui.project.settings.default_cell_detector_params
        # normalizer = get_normalizer(
        #     name=cell_detector_params.normalizer,
        #     range=cell_detector_params.normalizer_range,
        # )
        # if normalizer is not None:
        #     highres_crop = normalizer.before(highres_crop).squeeze()

        self._crop = highres_crop
        self.crop_layer.data = self._crop
        self.crop_layer.visible = True
        self.normalized_crop_layer.visible = False
        self.ui.update_layer_contrast_limits(self.crop_layer)
        self.cell_mask_layer.data = np.zeros_like(self._crop, dtype=np.uint8)
        self.set_preview_affine()
