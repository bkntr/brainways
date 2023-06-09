from pathlib import Path
from typing import List
from unittest.mock import Mock

import pandas as pd
import pytest

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


@pytest.fixture
def brainways_project(
    brainways_subject: BrainwaysSubject,
    mock_project_settings: ProjectSettings,
    mock_atlas: BrainwaysAtlas,
) -> BrainwaysProject:
    brainways_project = BrainwaysProject(
        subjects=[brainways_subject], settings=mock_project_settings, lazy_init=True
    )
    brainways_project.atlas = mock_atlas
    brainways_project.load_pipeline()
    return brainways_project


def test_brainways_project_create_excel(brainways_project: BrainwaysProject, tmpdir):
    excel_path = Path(tmpdir / "excel.xlsx")
    cells_df = pd.DataFrame({"x": [0.5], "y": [0.5]})
    subject = brainways_project.subjects[0]
    _, slice_info = subject.valid_documents[0]
    cells_df.to_csv(subject.cell_detections_path(slice_info.path), index=False)
    brainways_project.create_excel(excel_path)
    assert excel_path.exists()


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
