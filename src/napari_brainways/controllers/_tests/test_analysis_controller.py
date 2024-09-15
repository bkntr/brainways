from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from pytest import fixture

from brainways.project.info_classes import (
    RegisteredAnnotationFileFormat,
    SliceSelection,
)
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


def test_export_registration_masks_async_current_slice(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    output_dir = Path("/fake/path")
    slice_selection = SliceSelection.CURRENT_SLICE
    file_format = RegisteredAnnotationFileFormat.CSV

    with patch.object(
        controller.ui.project, "export_registration_masks_async"
    ) as mock_export:
        controller.export_registration_masks_async(
            output_dir, slice_selection, file_format
        )
        mock_export.assert_called_once()
        assert mock_export.call_args[1]["output_dir"] == output_dir
        assert mock_export.call_args[1]["slice_infos"] == [
            controller.ui.current_document
        ]
        assert mock_export.call_args[1]["file_format"] == file_format


def test_export_registration_masks_async_current_subject(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    output_dir = Path("/fake/path")
    slice_selection = SliceSelection.CURRENT_SUBJECT
    file_format = RegisteredAnnotationFileFormat.NPZ

    with patch.object(
        controller.ui.project, "export_registration_masks_async"
    ) as mock_export:
        controller.export_registration_masks_async(
            output_dir, slice_selection, file_format
        )
        mock_export.assert_called_once()
        expected_slice_infos = [
            slice_info for _, slice_info in app.current_subject.valid_documents
        ]
        assert mock_export.call_args[1]["output_dir"] == output_dir
        assert mock_export.call_args[1]["slice_infos"] == expected_slice_infos
        assert mock_export.call_args[1]["file_format"] == file_format


def test_export_registration_masks_async_all_subjects(
    app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    assert app.project is not None
    output_dir = Path("/fake/path")
    slice_selection = SliceSelection.ALL_SUBJECTS
    file_format = RegisteredAnnotationFileFormat.CSV

    with patch.object(
        controller.ui.project, "export_registration_masks_async"
    ) as mock_export:
        controller.export_registration_masks_async(
            output_dir, slice_selection, file_format
        )
        mock_export.assert_called_once()
        expected_slice_infos = [
            slice_info
            for subject in app.project.subjects
            for _, slice_info in subject.valid_documents
        ]
        assert mock_export.call_args[1]["output_dir"] == output_dir
        assert mock_export.call_args[1]["slice_infos"] == expected_slice_infos
        assert mock_export.call_args[1]["file_format"] == file_format
