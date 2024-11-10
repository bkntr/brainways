from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import napari.layers
import numpy as np
from napari.qt.threading import FunctionWorker
from PyQt5.QtWidgets import QApplication

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.ui.controllers.base import Controller
from brainways.ui.widgets.cell_viewer_widget import CellViewerWidget
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cells import get_cell_struct_ids, get_struct_colors

if TYPE_CHECKING:
    from brainways.ui.brainways_ui import BrainwaysUI


class Cell3DViewerController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.input_layer: napari.layers.Image | None = None
        self.points_layer: napari.layers.Points | None = None
        self.atlas_layer: napari.layers.Image | None = None
        self._params: BrainwaysParams | None = None
        self._image: np.ndarray | None = None
        self._atlas: BrainwaysAtlas | None = None
        # TODO: ability to switch between modes
        self._3d_view_mode = False
        self.widget = CellViewerWidget(self)

    def set_2d_mode(self):
        if not self._3d_view_mode:
            return
        self._3d_view_mode = False
        self.show(params=self._params, image=self._image)

    def set_3d_mode(self):
        if self._3d_view_mode:
            return
        self._3d_view_mode = True
        self.show(params=self._params, image=self._image)

    def open(self) -> None:
        if self._is_open:
            return

        self._atlas = self.ui.project.atlas
        self.points_layer = self.ui.viewer.add_points(
            size=1, ndim=3, name="Detected Cells"
        )
        self.ui.viewer.dims.ndisplay = 3
        self.ui.viewer.reset_view()
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        if self._3d_view_mode:
            self.ui.viewer.layers.remove(self.atlas_layer)
        else:
            self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.points_layer)
        QApplication.instance().processEvents()
        self.input_layer = None
        self.atlas_layer = None
        self.points_layer = None
        self.ui.viewer.dims.ndisplay = 2
        self._atlas = None
        self._params = None
        self._image = None
        self._is_open = False

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params
        if image is not None:
            self._image = image

        if self._3d_view_mode:
            self.show_3d()
        else:
            self.show_2d(image=self._image, from_ui=from_ui)

    def show_3d(self):
        if self.input_layer is not None:
            self.ui.viewer.layers.remove(self.input_layer)
            self.input_layer = None
        self.atlas_layer = self.ui.viewer.add_image(
            self._atlas.reference.numpy(),
            name="Atlas",
            rendering="attenuated_mip",
            attenuation=0.5,
        )
        self.ui.viewer.dims.ndisplay = 3

        self.ui.viewer.layers.remove(self.points_layer)
        self.points_layer = self.ui.viewer.add_points(
            size=2, ndim=3, name="Detected Cells"
        )

        subject = self.ui.current_subject
        all_cells = subject.get_cells_on_atlas()
        if all_cells is not None:
            # TODO: get struct ids from annotation, not from brainglobe
            struct_ids = get_cell_struct_ids(all_cells, self._atlas.brainglobe_atlas)
            colors = get_struct_colors(struct_ids, self._atlas.brainglobe_atlas)
            self.points_layer.data = all_cells[["z", "y", "x"]].values
            self.points_layer.face_color = colors
            self.points_layer.border_color = colors
            self.points_layer.selected_data = set()
        else:
            self.points_layer.data = []

        # TODO: points color not updating in napari 0.4.18, this is a workaround
        self.points_layer.visible = False
        self.points_layer.visible = True

        self.ui.viewer.reset_view()

    def show_2d(self, image: np.ndarray | None = None, from_ui: bool = False):
        if image is None:
            return

        if self.atlas_layer is not None:
            self.ui.viewer.layers.remove(self.atlas_layer)
            self.atlas_layer = None
        if self.input_layer is not None:
            self.ui.viewer.layers.remove(self.input_layer)
        self.input_layer = self.ui.viewer.add_image(image, name="Image")
        self.input_layer.events.contrast_limits.connect(self.ui.set_contrast_limits)
        self.ui.update_layer_contrast_limits(self.input_layer)
        self.ui.viewer.dims.ndisplay = 2

        self.ui.viewer.layers.remove(self.points_layer)
        self.points_layer = self.ui.viewer.add_points(
            size=max(image.shape) * 0.004, ndim=2, name="Detected Cells"
        )

        subject = self.ui.current_subject
        document = self.ui.current_document

        # TODO: read image from disk with options for other channels

        cells_atlas = subject.get_cells_on_atlas([document])
        if cells_atlas is not None:
            cells = subject.get_valid_cells(document)
            # TODO: get struct ids from annotation, not from brainglobe
            struct_ids = get_cell_struct_ids(
                cells_atlas, subject.atlas.brainglobe_atlas
            )
            colors = get_struct_colors(struct_ids, subject.atlas.brainglobe_atlas)
            cell_xy = cells[["y", "x"]].values * [image.shape[0], image.shape[1]]
            self.points_layer.data = cell_xy
            self.points_layer.face_color = colors
            self.points_layer.border_color = colors
            self.points_layer.selected_data = set()
        else:
            self.points_layer.data = []

        # TODO: points color not updating in napari 0.4.18, this is a workaround
        self.points_layer.visible = False
        self.points_layer.visible = True

        self.ui.viewer.reset_view()

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        return params

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return True

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return params.tps is not None

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    def _load_full_res_image(self):
        self._image = self.ui.current_subject.read_highres_image(
            self.ui.current_document
        )

    def _load_full_res_image_returned(self):
        self.show(params=self._params, image=self._image)

    def load_full_res_image_async(self) -> FunctionWorker:
        return self.ui.do_work_async(
            self._load_full_res_image,
            return_callback=self._load_full_res_image_returned,
            progress_label="Loading full resolution image...",
        )

    @property
    def params(self) -> BrainwaysParams:
        return self._params

    @property
    def name(self) -> str:
        return "Cell Viewer"
