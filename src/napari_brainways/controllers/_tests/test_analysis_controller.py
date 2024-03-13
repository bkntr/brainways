from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pytest import fixture

from napari_brainways.brainways_ui import BrainwaysUI
from napari_brainways.controllers.analysis_controller import AnalysisController


@fixture
def app_on_analysis(opened_app: BrainwaysUI) -> Tuple[BrainwaysUI, AnalysisController]:
    tps_step_index = [
        isinstance(step, AnalysisController) for step in opened_app.steps
    ].index(True)
    opened_app.set_step_index_async(tps_step_index)
    controller: AnalysisController = opened_app.current_step

    return opened_app, controller


def test_analysis_controller_run_pls_analysis(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    for subject in app.project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition1": [subject.subject_info.conditions["condition1"]] * n,
                    "animal_id": [subject.subject_info.name] * n,
                    "acronym": [str(i) for i in range(n)],
                    "name": ["a"] * n,
                    "is_parent_structure": [False] * n,
                    "is_gray_matter": [True] * n,
                    "total_area_um2": [10.0] * n,
                    "cells": [10.0] * n,
                }
            )
        )

    controller.run_calculate_results_async()
    controller.run_pls_analysis_async(
        condition_col="condition1",
        values_col="cells",
        min_group_size=1,
        alpha=1.0,
        n_perm=10,
        n_boot=10,
    )


def test_analysis_controller_run_contrast_analysis(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    for subject in app.project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition1": [subject.subject_info.conditions["condition1"]] * n,
                    "animal_id": [subject.subject_info.name] * n,
                    "acronym": ["TEST"] * n,
                    "name": ["a"] * n,
                    "is_parent_structure": [False] * n,
                    "is_gray_matter": [True] * n,
                    "total_area_um2": [10.0] * n,
                    "cells": np.random.randint(0, 100, n),
                }
            )
        )

    controller.run_calculate_results_async()
    controller.run_contrast_analysis_async(
        condition_col="condition1",
        values_col="cells",
        min_group_size=1,
        pvalue=1.0,
        multiple_comparisons_method="fdr_bh",
    )


def test_analysis_controller_run_network_analysis(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    for subject in app.project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition1": [subject.subject_info.conditions["condition1"]] * n,
                    "animal_id": [subject.subject_info.name] * n,
                    "acronym": [str(i) for i in range(n)],
                    "name": ["a"] * n,
                    "is_parent_structure": [False] * n,
                    "is_gray_matter": [True] * n,
                    "total_area_um2": [10.0] * n,
                    "cells": np.random.randint(0, 100, n),
                }
            )
        )

    graph_path = (
        app.project.path.parent
        / "__outputs__"
        / "network_graph"
        / "Condition=condition1,Values=cells.graphml"
    )

    assert not graph_path.exists()

    controller.run_calculate_results_async()
    controller.run_network_analysis_async(
        condition_col="condition1",
        values_col="cells",
        min_group_size=1,
        alpha=1.0,
    )

    assert graph_path.exists()
