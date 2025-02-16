from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from pytest import fixture

from brainways.project.info_classes import (
    MaskFileFormat,
    RegisteredPixelValues,
    SliceSelection,
)
from brainways.ui.brainways_ui import BrainwaysUI
from brainways.ui.controllers.analysis_controller import AnalysisController


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
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController],
    tmp_path: Path,
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

    controller.run_calculate_results_async()
    controller.run_network_analysis_async(
        condition_col="condition1",
        values_col="cells",
        min_group_size=1,
        n_bootstraps=100,
        multiple_comparison_correction_method="fdr_bh",
        output_path=tmp_path / "network_graph",
    )

    assert (tmp_path / "network_graph.graphml").exists()


def test_export_registration_masks_async(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    output_dir = Path("/fake/path")
    pixel_value_mode = RegisteredPixelValues.PIXEL_COORDINATES
    slice_selection = SliceSelection.CURRENT_SUBJECT
    file_format = MaskFileFormat.NPZ

    with (
        patch.object(app.project, "export_registration_masks_async") as mock_export,
        patch.object(app, "get_slice_selection", return_value="mock_slice_infos"),
    ):
        controller.export_registration_masks_async(
            output_dir, pixel_value_mode, slice_selection, file_format
        )
        mock_export.assert_called_once()
        assert mock_export.call_args[1]["output_dir"] == output_dir
        assert mock_export.call_args[1]["pixel_value_mode"] == pixel_value_mode
        assert mock_export.call_args[1]["slice_infos"] == "mock_slice_infos"
        assert mock_export.call_args[1]["file_format"] == file_format


def test_export_slice_locations(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    output_path = Path("/fake/path")
    slice_selection = SliceSelection.CURRENT_SUBJECT

    with (
        patch.object(app.project, "export_slice_locations") as mock_export,
        patch.object(app, "get_slice_selection", return_value="mock_slice_infos"),
    ):
        controller.export_slice_locations(output_path, slice_selection)
        mock_export.assert_called_once()
        assert mock_export.call_args[0][0] == output_path
        assert mock_export.call_args[0][1] == "mock_slice_infos"
