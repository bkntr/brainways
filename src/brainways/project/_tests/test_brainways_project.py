from dataclasses import replace
from pathlib import Path
from typing import List
from unittest.mock import Mock

import numpy as np
import pandas as pd

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import ProjectSettings, SliceInfo, SubjectInfo


def test_brainways_project_create_excel(brainways_project: BrainwaysProject):
    cells_df = pd.DataFrame({"x": [0.5], "y": [0.5]})
    subject = brainways_project.subjects[0]
    _, slice_info = subject.valid_documents[0]
    cells_df.to_csv(subject.cell_detections_path(slice_info.path), index=False)
    assert not brainways_project._results_path.exists()
    brainways_project.calculate_results()
    assert brainways_project._results_path.exists()


def test_brainways_project_move_images(brainways_project: BrainwaysProject):
    for subject in brainways_project.subjects:
        subject.move_images_root = Mock()
    brainways_project.move_images_directory(
        new_images_root=Path(), old_images_root=Path()
    )
    for subject in brainways_project.subjects:
        subject.move_images_root.assert_called_once()


def test_open_brainways_project(
    project_path: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
):
    project = BrainwaysProject.open(project_path, lazy_init=True)
    assert project.settings == mock_project_settings
    assert len(project.subjects) == 1
    assert project.subjects[0].documents == mock_subject_documents


def test_open_brainways_project_v0_1_1(
    brainways_project_path_v0_1_1: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
):
    project = BrainwaysProject.open(brainways_project_path_v0_1_1, lazy_init=True)
    assert project.settings.atlas == mock_project_settings.atlas
    assert len(project.subjects) == 1
    assert (
        project.subjects[0].documents[0].params.atlas
        == mock_subject_documents[0].params.atlas
    )
    assert (
        project.subjects[0].documents[0].params.affine
        == mock_subject_documents[0].params.affine
    )


def test_open_brainways_project_v0_1_4(
    brainways_project_path_v0_1_4: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
):
    project = BrainwaysProject.open(brainways_project_path_v0_1_4, lazy_init=True)
    assert project.settings.atlas == mock_project_settings.atlas
    assert len(project.subjects) == 1
    assert (
        project.subjects[0].documents[0].params.atlas
        == mock_subject_documents[0].params.atlas
    )
    assert (
        project.subjects[0].documents[0].params.affine
        == mock_subject_documents[0].params.affine
    )


def test_add_subject(brainways_project: BrainwaysProject):
    brainways_project.add_subject(
        SubjectInfo(name="subject3", conditions={"condition": "a"})
    )
    assert brainways_project.subjects[-1].atlas == brainways_project.atlas
    assert brainways_project.subjects[-1].pipeline == brainways_project.pipeline


def test_next_slice_missing_params_none_missing(brainways_project: BrainwaysProject):
    assert brainways_project.next_slice_missing_params() is None


def test_next_slice_missing_params_ignored_missing(brainways_project: BrainwaysProject):
    params_missing = replace(
        brainways_project.subjects[1].documents[2].params, tps=None
    )
    brainways_project.subjects[1].documents[1] = replace(
        brainways_project.subjects[1].documents[1], params=params_missing, ignore=True
    )
    assert brainways_project.next_slice_missing_params() is None


def test_next_slice_missing_params_has_missing(brainways_project: BrainwaysProject):
    params_missing = replace(
        brainways_project.subjects[1].documents[2].params, tps=None
    )
    brainways_project.subjects[1].documents[1] = replace(
        brainways_project.subjects[1].documents[1], params=params_missing, ignore=True
    )
    brainways_project.subjects[1].documents[2] = replace(
        brainways_project.subjects[1].documents[2], params=params_missing
    )
    assert brainways_project.next_slice_missing_params() == (1, 2)


def test_can_calculate_results(brainways_project: BrainwaysProject):
    assert brainways_project.can_calculate_results()


def test_cant_calculate_results(brainways_project: BrainwaysProject):
    params_missing = replace(
        brainways_project.subjects[1].documents[2].params, tps=None
    )
    brainways_project.subjects[1].documents[2] = replace(
        brainways_project.subjects[1].documents[2], params=params_missing
    )
    assert not brainways_project.can_calculate_results()


def test_can_calculate_contrast(brainways_project: BrainwaysProject):
    assert brainways_project.can_calculate_contrast("condition")


def test_cant_calculate_contrast_only_one_condition(
    brainways_project: BrainwaysProject,
):
    for subject_idx, subject in enumerate(brainways_project.subjects):
        subject.subject_info = replace(
            subject.subject_info, conditions={"condition": "a"}
        )
    assert not brainways_project.can_calculate_contrast("condition")


def test_cant_calculate_contrast_missing_conditions(
    brainways_project: BrainwaysProject,
):
    brainways_project.subjects[0].subject_info = replace(
        brainways_project.subjects[0].subject_info, conditions=None
    )
    assert not brainways_project.can_calculate_contrast("condition")


def test_calculate_contrast(
    brainways_project: BrainwaysProject,
):
    """
    need to add mock cells for this test to work
    """

    for subject in brainways_project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition": [subject.subject_info.conditions["condition"]] * n,
                    "animal_id": [subject.subject_info.name] * n,
                    "acronym": ["a"] * n,
                    "name": ["a"] * n,
                    "is_parent_structure": [False] * n,
                    "is_gray_matter": [True] * n,
                    "total_area_um2": [10.0] * n,
                    "cells": np.random.randint(0, 100, n),
                }
            )
        )

    brainways_project.calculate_results()
    brainways_project.calculate_contrast(
        condition_col="condition", values_col="cells", min_group_size=1, pvalue=1.0
    )


def test_pls_analysis(
    brainways_project: BrainwaysProject,
):
    for subject in brainways_project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition": [subject.subject_info.conditions["condition"]] * n,
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

    brainways_project.calculate_results()
    brainways_project.calculate_pls_analysis(
        condition_col="condition",
        values_col="cells",
        min_group_size=1,
        alpha=1.0,
        n_perm=1,
        n_boot=1,
    )


def test_network_analysis(
    brainways_project: BrainwaysProject,
):
    """
    need to add mock cells for this test to work
    """

    for subject in brainways_project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition": [subject.subject_info.conditions["condition"]] * n,
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

    brainways_project.calculate_results()
    brainways_project.calculate_network_graph(
        condition_col="condition", values_col="cells", min_group_size=1, alpha=1.0
    )

    graph_path = (
        brainways_project.path.parent
        / "__outputs__"
        / "network_graph"
        / "Condition=condition,Values=cells.graphml"
    )

    assert graph_path.exists()
