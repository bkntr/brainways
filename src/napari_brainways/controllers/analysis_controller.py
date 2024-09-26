from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import napari
import napari.layers
import numpy as np
import pandas as pd
import scipy.stats
from napari.qt.threading import FunctionWorker
from napari.utils.colormaps.colormap import Colormap
from PyQt5.QtWidgets import QApplication

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.info_classes import (
    ExcelMode,
    RegisteredAnnotationFileFormat,
    SliceSelection,
)
from napari_brainways.controllers.base import Controller
from napari_brainways.utils.general_utils import update_layer_contrast_limits
from napari_brainways.widgets.analysis_widget import AnalysisWidget

if TYPE_CHECKING:
    from napari_brainways.brainways_ui import BrainwaysUI


class AnalysisController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.atlas_layer: napari.layers.Image | None = None
        self.annotations_layer: napari.layers.Image | None = None
        self._params: BrainwaysParams | None = None
        self._condition: str | None = None
        self._anova_df: pd.DataFrame | None = None
        self._posthoc_df: pd.DataFrame | None = None
        self._show_mode: str | None = None
        self._contrast: str | None = None
        self.widget = AnalysisWidget(self)

    @property
    def name(self) -> str:
        return "Analysis"

    def default_params(self, image: np.ndarray, params: BrainwaysParams):
        return params

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return True

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return True

    def open(self) -> None:
        if self._is_open:
            return

        # remove the sample project helper layer
        for layer in self.ui.viewer.layers:
            if "__brainways__" in layer.metadata:
                self.ui.viewer.layers.remove(layer)

        self.atlas_layer = self.ui.viewer.add_image(
            self.ui.project.atlas.reference.numpy(),
            name="Atlas",
            rendering="attenuated_mip",
            attenuation=0.5,
            visible=False,
        )
        self._annotations = self.ui.project.atlas.annotation.numpy().astype(np.int32)
        colors = {i: "white" for i in self.ui.project.atlas.brainglobe_atlas.structures}
        colors[0] = "black"
        self.annotations_layer = self.ui.viewer.add_labels(
            self._annotations,
            name="Structures",
            # color=colors,
            opacity=1.0,
        )
        self.annotations_layer.contour = 1
        mpl_colors = plt.get_cmap("hot")(np.linspace(0, 1, 256))
        colormap = Colormap(name="hot", display_name="hot", colors=mpl_colors)
        self.contrast_layer = self.ui.viewer.add_image(
            np.zeros_like(self.ui.project.atlas.annotation),
            name="Contrast",
            rendering="attenuated_mip",
            attenuation=0.5,
            colormap=colormap,
            blending="additive",
        )

        self.atlas_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self.annotations_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self.contrast_layer.mouse_move_callbacks.append(self.on_mouse_move)

        self.ui.viewer.text_overlay.visible = True
        self.ui.viewer.text_overlay.font_size = 16
        self.ui.viewer.text_overlay.color = (0.0, 0.8, 0.0, 1.0)
        self.ui.viewer.text_overlay.position = "top_center"

        self._is_open = True

    def on_mouse_move(self, _layer, event):
        _ = self.annotations_layer.extent
        data_position = self.annotations_layer.world_to_data(event.position)
        data_position = tuple(int(round(c)) for c in data_position)
        if all(0 <= c < s for c, s in zip(data_position, self._annotations.shape)):
            struct_id = self._annotations[data_position]
        else:
            struct_id = 0

        string = ""

        if struct_id and struct_id in self.pipeline.atlas.brainglobe_atlas.structures:
            struct_name = self.pipeline.atlas.brainglobe_atlas.structures[struct_id][
                "name"
            ]
            string = struct_name

        if self.current_show_mode:
            tvalue = self.contrast_layer.get_value(event.position, world=True)
            if tvalue:
                if self._show_mode == "anova":
                    string += f" (F={tvalue:.2f})"
                else:
                    pvalue = 1 - scipy.stats.norm.cdf(tvalue).round(5)
                    string += f" (p={pvalue:.5})"

        self.ui.viewer.text_overlay.text = string

    def close(self) -> None:
        self.ui.viewer.layers.remove(self.atlas_layer)
        self.ui.viewer.layers.remove(self.annotations_layer)
        self.ui.viewer.layers.remove(self.contrast_layer)
        QApplication.instance().processEvents()
        self.atlas_layer = None
        self.annotations_layer = None
        self.contrast_layer = None
        self._params = None
        self._condition = None
        self._anova_df = None
        self._posthoc_df = None
        self._show_mode = None
        self._contrast = None
        self._is_open = False

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params

    def show_anova(self):
        assert self._anova_df is not None

        atlas = self.ui.project.atlas
        annotation = self.ui.project.atlas.annotation.numpy()
        annotation_anova = np.zeros_like(annotation)
        for structure, row in self._anova_df[self._anova_df["reject"]].iterrows():
            struct_id = atlas.brainglobe_atlas.structures[structure]["id"]
            struct_mask = annotation == struct_id
            annotation_anova[struct_mask] = row["F"]
        self.contrast_layer.data = annotation_anova
        update_layer_contrast_limits(self.contrast_layer)

        self.contrast_layer.visible = True
        self.annotations_layer.visible = False

        self._show_mode = "anova"
        self.widget.set_label()

    def show_posthoc(self, contrast: str, pvalue: float):
        assert self._posthoc_df is not None

        atlas = self.ui.project.atlas
        annotation = self.ui.project.atlas.annotation.numpy()
        annotation_anova = np.zeros_like(annotation)
        for structure, row in self._posthoc_df[
            self._posthoc_df[contrast] <= pvalue
        ].iterrows():
            struct_id = atlas.brainglobe_atlas.structures[structure]["id"]
            struct_mask = annotation == struct_id
            tvalue = scipy.stats.norm.ppf(1 - row[contrast])
            annotation_anova[struct_mask] = tvalue
        self.contrast_layer.data = annotation_anova
        self.annotations_layer.data[annotation_anova == 0] = 0
        update_layer_contrast_limits(self.contrast_layer)

        # TODO: insert this nicely
        # import matplotlib as mpl

        # figure = Figure(figsize=(1, 8))
        # mpl_widget = FigureCanvas()
        # ax = mpl_widget.figure.subplots()
        # self.ui.viewer.window.add_dock_widget(mpl_widget)
        # norm = mpl.colors.Normalize(
        #     vmin=self.contrast_layer.contrast_limits[0],
        #     vmax=self.contrast_layer.contrast_limits[1],
        # )
        # cbar = figure.colorbar(
        #     mpl.cm.ScalarMappable(norm=norm, cmap="hot"),
        #     ax=ax,
        #     pad=0.05,
        #     fraction=1,
        # )
        # ax.axis("off")
        # cbar.set_label("t score")

        self._show_mode = "posthoc"
        self.contrast_layer.visible = True

    def run_calculate_results_async(
        self,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
        excel_mode: ExcelMode = ExcelMode.ROW_PER_SUBJECT,
    ) -> FunctionWorker:
        return self.ui.do_work_async(
            self.ui.project.calculate_results_iter,
            min_region_area_um2=min_region_area_um2,
            cells_per_area_um2=cells_per_area_um2,
            min_cell_size_um=min_cell_size_um,
            max_cell_size_um=max_cell_size_um,
            excel_mode=excel_mode,
            progress_label="Calculating Brainways Results...",
            progress_max_value=len(self.ui.project.subjects),
        )

    def run_contrast_analysis_async(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        pvalue: float,
        multiple_comparisons_method: str,
    ) -> FunctionWorker:
        self._condition = condition_col
        self.ui.do_work_async(
            self._run_contrast_analysis,
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
            return_callback=self.show_anova,
        )

    def _run_contrast_analysis(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        pvalue: float,
        multiple_comparisons_method: str,
    ):
        self._anova_df, self._posthoc_df = self.ui.project.calculate_contrast(
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
        )

    def run_pls_analysis_async(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        alpha: float,
        n_perm: int = 1000,
        n_boot: int = 1000,
    ):
        self.ui.do_work_async(
            self.ui.project.calculate_pls_analysis,
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            alpha=alpha,
            n_perm=n_perm,
            n_boot=n_boot,
        )

    def run_network_analysis_async(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        alpha: float,
    ):
        self.ui.do_work_async(
            self.ui.project.calculate_network_graph,
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            alpha=alpha,
        )

    def export_registration_masks_async(
        self,
        output_path: Path,
        slice_selection: SliceSelection,
        file_format: RegisteredAnnotationFileFormat,
    ):
        assert self.ui.project is not None

        if slice_selection == SliceSelection.CURRENT_SLICE:
            slice_infos = [self.ui.current_document]
        elif slice_selection == SliceSelection.CURRENT_SUBJECT:
            slice_infos = [
                slice_info for _, slice_info in self.ui.current_subject.valid_documents
            ]
        else:
            slice_infos = [
                slice_info
                for subject in self.ui.project.subjects
                for _, slice_info in subject.valid_documents
            ]

        self.ui.do_work_async(
            self.ui.project.export_registration_masks_async,
            progress_label="Exporting Registered Annotation Masks...",
            progress_max_value=len(slice_infos),
            output_dir=output_path,
            slice_infos=slice_infos,
            file_format=file_format,
        )

    @property
    def current_condition(self) -> str | None:
        return self._condition

    @property
    def current_show_mode(self) -> str | None:
        return self._show_mode

    @property
    def current_contrast(self) -> str | None:
        return self._contrast

    @property
    def possible_contrasts(self) -> List[str]:
        result = self.ui.project.possible_contrasts(self._condition)
        return ["-".join(c) for c in result]

    @property
    def params(self) -> BrainwaysParams:
        return self._params
