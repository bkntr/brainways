from pathlib import Path
from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox, request_values
from qtpy.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from brainways.project.info_classes import (
    ExcelMode,
    MaskFileFormat,
    RegisteredPixelValues,
    SliceSelection,
)
from brainways.ui.widgets.structure_selection_dialog import StructureSelectionDialog

if TYPE_CHECKING:
    from brainways.ui.controllers.analysis_controller import AnalysisController


class AnalysisWidget(QWidget):
    def __init__(self, controller: "AnalysisController"):
        super().__init__()
        self.controller = controller

        self.label = QLabel()
        self.set_label()

        calculate_results_button = QPushButton("Calculate results")
        calculate_results_button.clicked.connect(self.on_run_calculate_results_clicked)

        contrast_analysis_button = QPushButton("Run contrast analysis (ANOVA)")
        contrast_analysis_button.clicked.connect(self.on_run_contrast_analysis_clicked)

        pls_analysis_button = QPushButton("Run PLS analysis")
        pls_analysis_button.clicked.connect(self.on_run_pls_analysis_clicked)

        network_analysis_button = QPushButton("Run network analysis")
        network_analysis_button.clicked.connect(self.on_run_network_analysis_clicked)

        show_anova_button = QPushButton("Show ANOVA")
        show_anova_button.clicked.connect(self.on_show_anova_clicked)

        show_posthoc_button = QPushButton("Show Posthoc")
        show_posthoc_button.clicked.connect(self.on_show_posthoc_clicked)

        export_registration_masks_button = QPushButton(
            "Export Registered Annotation Masks"
        )
        export_registration_masks_button.clicked.connect(
            self.on_export_registration_masks_clicked
        )

        export_slice_locations_button = QPushButton("Export Slice Locations")
        export_slice_locations_button.clicked.connect(
            self.on_export_slice_locations_clicked
        )

        export_annotated_region_images_button = QPushButton(
            "Export Annotated Region Images"
        )
        export_annotated_region_images_button.clicked.connect(
            self.on_export_annotated_region_images_clicked
        )

        self.setLayout(QVBoxLayout())  # Initialize layout first
        layout = self.layout()
        assert layout is not None

        layout.addWidget(self.label)
        layout.addWidget(calculate_results_button)
        layout.addWidget(contrast_analysis_button)
        layout.addWidget(pls_analysis_button)
        layout.addWidget(network_analysis_button)
        layout.addWidget(show_anova_button)
        layout.addWidget(show_posthoc_button)
        layout.addWidget(export_registration_masks_button)
        layout.addWidget(export_slice_locations_button)
        layout.addWidget(export_annotated_region_images_button)

    def on_run_calculate_results_clicked(self, _=None):
        if not self.controller.ui.prompt_user_slices_have_missing_params(
            check_cells=True
        ):
            return

        values = request_values(
            title="Excel Parameters",
            min_region_area_um2=dict(
                value=250,
                annotation=int,
                label="Min Structure Square Area (μm)",
                options=dict(
                    tooltip="Filter out structures with an area smaller than this value"
                ),
            ),
            cells_per_area_um2=dict(
                value=250,
                annotation=int,
                label="Cells Per Square Area (μm)",
                options=dict(
                    tooltip="Normalize number of cells to number of cells per area unit"
                ),
            ),
            min_cell_size_um=dict(
                value=0,
                annotation=int,
                label="Min Cell Area (μm)",
                options=dict(
                    tooltip=(
                        "Filter out detected cells with area smaller than this value"
                    )
                ),
            ),
            max_cell_size_um=dict(
                value=0,
                annotation=int,
                label="Max Cell Area (μm)",
                options=dict(
                    tooltip="Filter out detected cells with area larger than this value"
                ),
            ),
            excel_mode=dict(
                value=ExcelMode.ROW_PER_SUBJECT.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in ExcelMode],
                    tooltip="How to format the excel file",
                ),
                annotation=str,
                label="Detail Level",
            ),
        )
        if values is None:
            return

        self.controller.run_calculate_results_async(
            min_region_area_um2=values["min_region_area_um2"],
            cells_per_area_um2=values["cells_per_area_um2"],
            min_cell_size_um=values["min_cell_size_um"],
            max_cell_size_um=values["max_cell_size_um"],
            excel_mode=ExcelMode(values["excel_mode"]),
        )

    def on_run_contrast_analysis_clicked(self, _=None):
        conditions = self.controller.ui.project.settings.condition_names
        # cell_types = self.controller.ui.project.cell_types

        values = request_values(
            title="Run Contrast",
            condition_col=dict(
                value=conditions[0],
                widget_type="ComboBox",
                options=dict(choices=conditions),
                annotation=str,
                label="Condition",
            ),
            values_col=dict(
                value="cells",
                # widget_type="ComboBox",
                # options=dict(choices=cell_types),
                annotation=str,
                label="Cell Type",
            ),
            min_group_size=dict(
                value=3,
                annotation=int,
                label="Min Group Size",
                options=dict(
                    tooltip="Minimal number of animals to consider an area for contrast"
                ),
            ),
            pvalue=dict(
                value=0.05,
                annotation=float,
                label="P Value",
                options=dict(tooltip="P value cutoff for posthoc"),
            ),
            multiple_comparisons_method=dict(
                value="fdr_bh",
                annotation=str,
                label="Multiple Comparisons",
                options=dict(
                    tooltip="Method to use when adjusting for multiple comparisons"
                ),
            ),
        )
        if values is None:
            return

        self.controller.run_contrast_analysis_async(
            condition_col=values["condition_col"],
            values_col=values["values_col"],
            min_group_size=values["min_group_size"],
            pvalue=values["pvalue"],
            multiple_comparisons_method=values["multiple_comparisons_method"],
        )

    def on_run_pls_analysis_clicked(self, _=None):
        conditions = self.controller.ui.project.settings.condition_names
        # cell_types = self.controller.ui.project.cell_types

        values = request_values(
            title="Run PLS Analysis",
            condition_col=dict(
                value=conditions[0],
                widget_type="ComboBox",
                options=dict(choices=conditions),
                annotation=str,
                label="Condition",
            ),
            values_col=dict(
                value="cells",
                # widget_type="ComboBox",
                # options=dict(choices=cell_types),
                annotation=str,
                label="Cell Type",
            ),
            min_group_size=dict(
                value=3,
                annotation=int,
                label="Min Group Size",
                options=dict(
                    tooltip="Minimal number of animals to consider an area for contrast"
                ),
            ),
            alpha=dict(
                value=0.05,
                annotation=float,
                label="alpha",
                options=dict(tooltip="alpha for plots"),
            ),
        )
        if values is None:
            return

        self.controller.run_pls_analysis_async(
            condition_col=values["condition_col"],
            values_col=values["values_col"],
            min_group_size=values["min_group_size"],
            alpha=values["alpha"],
        )
        self.set_label()

    def on_run_network_analysis_clicked(self, _=None):
        conditions = self.controller.ui.project.settings.condition_names
        # cell_types = self.controller.ui.project.cell_types

        values = request_values(
            title="Run Network Analysis",
            condition_col=dict(
                value=conditions[0],
                widget_type="ComboBox",
                options=dict(choices=conditions),
                annotation=str,
                label="Condition",
            ),
            values_col=dict(
                value="cells",
                # widget_type="ComboBox",
                # options=dict(choices=cell_types),
                annotation=str,
                label="Cell Type",
            ),
            min_group_size=dict(
                value=3,
                annotation=int,
                label="Min Group Size",
                options=dict(
                    tooltip="Minimal number of animals to consider an area for contrast"
                ),
            ),
            n_bootstraps=dict(
                value=1000,
                annotation=int,
                label="Number of Bootstraps",
                options=dict(
                    tooltip="Number of bootstrap iterations to perform for the null network analysis"
                ),
            ),
            multiple_comparison_correction_method=dict(
                value="fdr_bh",
                annotation=str,
                label="Multiple Comparisons",
                options=dict(
                    tooltip="Method to use when adjusting for multiple comparisons"
                ),
            ),
            output_path=dict(
                value="",
                annotation=Path,
                label="Output Path",
                options=dict(
                    mode="w",
                    tooltip="Path to save the network analysis results to",
                    filter="GraphML files (*.graphml)",
                ),
            ),
        )
        if values is None:
            return

        if not values["output_path"].parent.exists():
            raise ValueError("Output directory does not exist")

        self.controller.run_network_analysis_async(
            condition_col=values["condition_col"],
            values_col=values["values_col"],
            min_group_size=values["min_group_size"],
            n_bootstraps=values["n_bootstraps"],
            multiple_comparison_correction_method=values[
                "multiple_comparison_correction_method"
            ],
            output_path=values["output_path"],
        )
        self.set_label()

    def on_show_anova_clicked(self, _=None):
        self.controller.show_anova()

    def on_show_posthoc_clicked(self, _=None):
        if self.controller.current_condition is None:
            return

        possible_contrasts = self.controller.possible_contrasts
        values = request_values(
            title="Show Posthoc",
            contrast=dict(
                value=possible_contrasts[0],
                widget_type="ComboBox",
                options=dict(choices=self.controller.possible_contrasts),
                annotation=str,
                label="Contrast",
            ),
            pvalue=dict(
                value=0.05,
                annotation=float,
                label="P Value",
                options=dict(tooltip="P value cutoff for posthoc"),
            ),
        )
        if values is None:
            return

        self.controller.show_posthoc(
            contrast=values["contrast"], pvalue=values["pvalue"]
        )

    def on_export_registration_masks_clicked(self, _=None):
        if not self.controller.ui.prompt_user_slices_have_missing_params():
            return

        values = request_values(
            title="Export Registered Annotation Masks",
            pixel_value_mode=dict(
                value=RegisteredPixelValues.STRUCTURE_IDS.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in RegisteredPixelValues],
                    tooltip="What to export to each pixel in the masks",
                ),
                annotation=str,
                label="Pixel Values",
            ),
            output_path=dict(
                value="",
                annotation=Path,
                label="Output Directory",
                options=dict(
                    mode="d",
                    tooltip="Directory to save the registered annotation masks to",
                ),
            ),
            slice_selection=dict(
                value=SliceSelection.CURRENT_SLICE.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in SliceSelection],
                    tooltip="Which slices to export",
                ),
                annotation=str,
                label="Slice Selection",
            ),
            file_format=dict(
                value=MaskFileFormat.NPZ.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in MaskFileFormat],
                    tooltip="File format to save the masks to",
                ),
                annotation=str,
                label="File Format",
            ),
        )
        if values is None:
            return

        self.controller.export_registration_masks_async(
            output_path=values["output_path"],
            pixel_value_mode=RegisteredPixelValues(values["pixel_value_mode"]),
            slice_selection=SliceSelection(values["slice_selection"]),
            file_format=MaskFileFormat(values["file_format"]),
        )

    def on_export_slice_locations_clicked(self, _=None):
        if not self.controller.ui.prompt_user_slices_have_missing_params():
            return

        values = request_values(
            title="Export Slice Locations",
            output_path=dict(
                value="",
                annotation=Path,
                label="Output Path",
                options=dict(
                    mode="w",
                    tooltip="Path to save the slice locations to",
                    filter="CSV files (*.csv)",
                ),
            ),
            slice_selection=dict(
                value=SliceSelection.CURRENT_SLICE.value,
                widget_type="ComboBox",
                options=dict(
                    choices=[e.value for e in SliceSelection][::-1],
                    tooltip="Which slices to export",
                ),
                annotation=str,
                label="Slice Selection",
            ),
        )
        if values is None:
            return

        if values["output_path"].suffix != ".csv":
            values["output_path"] = values["output_path"].with_suffix(".csv")

        self.controller.export_slice_locations(
            output_path=values["output_path"],
            slice_selection=SliceSelection(values["slice_selection"]),
        )

    def on_export_annotated_region_images_clicked(self, _=None):
        if not self.controller.ui.prompt_user_slices_have_missing_params():
            return

        values = request_values(
            title="Export Annotated Region Images",
            output_path=dict(
                annotation=Path,
                label="Output Directory",
                options=dict(
                    mode="d",
                    tooltip="Directory to save the annotated region images to",
                ),
            ),
            draw_cells=dict(
                annotation=bool,
                label="Draw Cells",
                value=True,
                options=dict(tooltip="Overlay detected cell locations on the images"),
            ),
            slice_selection=dict(
                widget_type=ComboBox,
                value=SliceSelection.CURRENT_SLICE.value,
                options=dict(
                    choices=[e.value for e in SliceSelection],
                    tooltip="Which slices to export images for",
                ),
                annotation=str,
                label="Slice Selection",
            ),
        )
        if values is None:
            return

        structures = self.controller.ui.project.atlas.brainglobe_atlas.structures

        dialog = StructureSelectionDialog(structures, parent=self)
        if dialog.exec_():
            selected_acronyms = dialog.get_selected_structures()
            if not selected_acronyms:
                QMessageBox.warning(
                    self, "No Selection", "No structures were selected."
                )
                return

            self.controller.export_annotated_region_images_async(
                output_path=values["output_path"],
                structure_acronyms=selected_acronyms,
                draw_cells=values["draw_cells"],
                slice_selection=SliceSelection(values["slice_selection"]),
            )

    def set_label(self):
        if self.controller.current_show_mode is None:
            self.label.setText(
                "Click 'Run contrast analysis' to be able to see contrast."
            )
        else:
            lines = [
                f"Currently showing: {self.controller.current_show_mode}",
                f"Condition: {self.controller.current_condition}",
            ]
            if self.controller.current_show_mode == "posthoc":
                lines += [f"Contrast: {self.controller.current_contrast}"]

            self.label.setText("\n".join(lines))
