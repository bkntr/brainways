from typing import TYPE_CHECKING

from magicgui.widgets import request_values
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from napari_brainways.controllers.analysis_controller import AnalysisController


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

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addWidget(calculate_results_button)
        self.layout().addWidget(contrast_analysis_button)
        self.layout().addWidget(pls_analysis_button)
        self.layout().addWidget(network_analysis_button)
        self.layout().addWidget(show_anova_button)
        self.layout().addWidget(show_posthoc_button)

    def on_run_calculate_results_clicked(self, _=None):
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
        )
        if values is None:
            return

        self.controller.run_calculate_results_async(
            min_region_area_um2=values["min_region_area_um2"],
            cells_per_area_um2=values["cells_per_area_um2"],
            min_cell_size_um=values["min_cell_size_um"],
            max_cell_size_um=values["max_cell_size_um"],
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
            alpha=dict(
                value=0.05,
                annotation=float,
                label="alpha",
                options=dict(tooltip="alpha for plots"),
            ),
        )
        if values is None:
            return

        self.controller.run_network_analysis_async(
            condition_col=values["condition_col"],
            values_col=values["values_col"],
            min_group_size=values["min_group_size"],
            alpha=values["alpha"],
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
