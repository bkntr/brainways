from dataclasses import replace
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import (
    MaskFileFormat,
    ProjectSettings,
    RegisteredPixelValues,
    SliceInfo,
    SubjectInfo,
)
from brainways.utils.io_utils.image_path import ImagePath


def test_brainways_project_create_excel(brainways_project: BrainwaysProject):
    cells_df = pd.DataFrame({"x": [0.5], "y": [0.5], "area_um": [10.0]})
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


def test_cant_calculate_contrast_missing_conditions_parameter(
    brainways_project: BrainwaysProject,
):
    brainways_project.subjects[0].subject_info = replace(
        brainways_project.subjects[0].subject_info, conditions=None
    )
    assert not brainways_project.can_calculate_contrast("condition")


def test_cant_calculate_contrast_missing_one_condition(
    brainways_project: BrainwaysProject,
):
    brainways_project.subjects[0].subject_info.conditions.pop("condition")
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
    tmp_path,
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
        condition_col="condition",
        values_col="cells",
        min_group_size=1,
        n_bootstraps=100,
        multiple_comparison_correction_method="fdr_bh",
        output_path=tmp_path / "network_graph",
    )

    assert (tmp_path / "network_graph.graphml").exists()


@pytest.mark.parametrize(
    "file_format",
    [
        MaskFileFormat.NPZ,
        MaskFileFormat.CSV,
        MaskFileFormat.MAT,
    ],
)
@pytest.mark.parametrize(
    "pixel_value_mode",
    [
        pytest.param(RegisteredPixelValues.STRUCTURE_IDS, id="struct_ids"),
        pytest.param(RegisteredPixelValues.PIXEL_COORDINATES, id="pixel_coords"),
        pytest.param(RegisteredPixelValues.MICRON_COORDINATES, id="micron_coords"),
    ],
)
@patch("brainways.project.brainways_project.open_directory")
@patch("brainways.project.brainways_project.export_mask")
def test_export_registration_masks_async(
    export_mask_mock: Mock,
    open_directory_mock: Mock,
    brainways_project: BrainwaysProject,
    pixel_value_mode: RegisteredPixelValues,
    file_format: MaskFileFormat,
    tmp_path,
):
    brainways_project._pipeline = Mock()
    if pixel_value_mode == RegisteredPixelValues.STRUCTURE_IDS:
        values = np.array([[1, 2], [3, 4]])
    else:
        values = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    brainways_project.pipeline.get_registered_values_on_image = Mock(
        return_value=values
    )
    slice_infos = brainways_project.subjects[0].documents
    assert len(slice_infos) > 0
    output_dir = tmp_path / "output"

    generator = brainways_project.export_registration_masks_async(
        output_dir=output_dir,
        pixel_value_mode=pixel_value_mode,
        slice_infos=slice_infos,
        file_format=file_format,
    )

    if (
        file_format == MaskFileFormat.CSV
        and pixel_value_mode != RegisteredPixelValues.STRUCTURE_IDS
    ):
        with pytest.raises(
            ValueError,
            match="CSV format is only supported for structure IDs pixel values",
        ):
            for _ in generator:
                pass
        return
    else:
        for _ in generator:
            pass

    assert brainways_project.pipeline.get_registered_values_on_image.call_count == len(
        slice_infos
    )
    for slice_info in slice_infos:
        brainways_project.pipeline.get_registered_values_on_image.assert_any_call(
            slice_info, pixel_value_mode=pixel_value_mode
        )
        output_file = (
            output_dir
            / f"{Path(str(slice_info.path)).name}_{pixel_value_mode.name.lower()}"
        )
        export_mask_mock.assert_any_call(
            data=values, path=output_file, file_format=file_format
        )


@patch("brainways.project.brainways_project.open_directory")
def test_export_slice_locations(
    open_directory_mock, brainways_project: BrainwaysProject, tmp_path
):
    # Create mock slice_infos and subject_infos
    slice_infos = [
        SliceInfo(
            path=ImagePath("slice1"),
            params=Mock(
                atlas=AtlasRegistrationParams(
                    ap=1.0, rot_frontal=2.0, rot_horizontal=3.0, rot_sagittal=4.0
                )
            ),
            image_size=(0, 0),
            lowres_image_size=(0, 0),
        ),
        SliceInfo(
            path=ImagePath("slice2"),
            params=Mock(
                atlas=AtlasRegistrationParams(
                    ap=5.0, rot_frontal=6.0, rot_horizontal=7.0, rot_sagittal=8.0
                )
            ),
            image_size=(0, 0),
            lowres_image_size=(0, 0),
        ),
    ]
    subject_infos = [
        SubjectInfo(name="subject1", conditions={"condition": "a"}),
        SubjectInfo(name="subject2", conditions={"condition": "b"}),
    ]

    # Mock the subjects in the brainways_project
    brainways_project.subjects = [
        Mock(documents=[slice_infos[0]], subject_info=subject_infos[0]),
        Mock(documents=[slice_infos[1]], subject_info=subject_infos[1]),
    ]

    # Define the output path
    output_path = tmp_path / "slice_locations.csv"

    # Call the method
    brainways_project.export_slice_locations(output_path, slice_infos)

    # Read the output CSV
    slice_locations_df = pd.read_csv(output_path)

    # Define the expected DataFrame
    expected_df = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "condition": "a",
                "slice": "slice1",
                "AP (μm)": 10,
                "Frontal rotation": 2.0,
                "Horizontal rotation": 3.0,
                "Sagittal rotation": 4.0,
            },
            {
                "subject": "subject2",
                "condition": "b",
                "slice": "slice2",
                "AP (μm)": 50,
                "Frontal rotation": 6.0,
                "Horizontal rotation": 7.0,
                "Sagittal rotation": 8.0,
            },
        ]
    )

    # Assert the DataFrame is as expected
    pd.testing.assert_frame_equal(slice_locations_df, expected_df)


def test_get_subjects_unknown_slice(brainways_project: BrainwaysProject, tmp_path):
    # Create invalid slice info
    invalid_slice = SliceInfo(
        path=ImagePath("nonexistent.jpg"),
        params=Mock(atlas=AtlasRegistrationParams(ap=100)),
        image_size=(0, 0),
        lowres_image_size=(0, 0),
    )

    with pytest.raises(
        ValueError, match="Slice nonexistent.jpg not found in any subject"
    ):
        brainways_project._get_subjects([invalid_slice])


@patch("brainways.project.brainways_project.open_directory")
def test_export_slice_locations_missing_params(
    open_directory_mock, brainways_project: BrainwaysProject, tmp_path
):
    """Test handling of slices with missing atlas parameters"""
    output_path = tmp_path / "slice_locations.csv"
    slice_info = brainways_project.subjects[0].documents[0]

    # Create slice with missing atlas params
    slice_with_missing = replace(
        slice_info, params=replace(slice_info.params, atlas=None)
    )
    brainways_project.subjects[0].documents[0] = slice_with_missing

    brainways_project.export_slice_locations(output_path, [slice_with_missing])

    df = pd.read_csv(output_path)
    assert pd.isna(df["AP (μm)"].iloc[0])
    assert pd.isna(df["Frontal rotation"].iloc[0])
    assert pd.isna(df["Horizontal rotation"].iloc[0])
    assert pd.isna(df["Sagittal rotation"].iloc[0])


def test_export_slice_locations_empty_list(
    brainways_project: BrainwaysProject, tmp_path
):
    """Test that exporting empty list of slices raises ValueError"""
    output_path = tmp_path / "slice_locations.csv"

    with pytest.raises(ValueError, match="No slices to export"):
        brainways_project.export_slice_locations(output_path, [])


@pytest.mark.parametrize(
    "resume",
    [
        pytest.param(True, id="resume"),
        pytest.param(False, id="no_resume"),
    ],
)
@pytest.mark.parametrize(
    "save_cell_detection_masks_file_format",
    [
        pytest.param(MaskFileFormat.NPZ, id="npz"),
        pytest.param(None, id="none"),
    ],
)
@patch("brainways.project.brainways_project.open_directory")
def test_run_cell_detector_iter(
    open_directory_mock,
    brainways_project: BrainwaysProject,
    resume: bool,
    save_cell_detection_masks_file_format: MaskFileFormat,
):
    slice_infos = [Mock(spec=SliceInfo), Mock(spec=SliceInfo)]
    subjects = [Mock(), Mock()]

    brainways_project.get_cell_detector = Mock()
    brainways_project._get_subjects = Mock(return_value=subjects)

    generator = brainways_project.run_cell_detector_iter(
        slice_infos,
        resume=resume,
        save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
    )

    for _ in generator:
        pass

    brainways_project.get_cell_detector.assert_called_once()
    brainways_project._get_subjects.assert_called_once_with(slice_infos)

    for subject, slice_info in zip(subjects, slice_infos):
        if resume:
            subject.clear_cell_detection.assert_not_called()
        else:
            subject.clear_cell_detection.assert_called_once_with(slice_info)
        subject.run_cell_detector.assert_called_once_with(
            slice_info=slice_info,
            cell_detector=brainways_project.get_cell_detector(),
            default_params=brainways_project.settings.default_cell_detector_params,
            save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
        )
