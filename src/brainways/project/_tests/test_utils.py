from pathlib import Path

from brainways.project._utils import update_project_from_previous_versions
from brainways.project.brainways_project import BrainwaysProject


def test_update_project_from_v0_1_1(brainways_project_path_v0_1_1: Path):
    update_project_from_previous_versions(brainways_project_path_v0_1_1)
    BrainwaysProject.open(brainways_project_path_v0_1_1, lazy_init=True)


def test_update_project_from_v0_1_4(brainways_project_path_v0_1_4: Path):
    update_project_from_previous_versions(brainways_project_path_v0_1_4)
    project = BrainwaysProject.open(brainways_project_path_v0_1_4, lazy_init=True)
    assert project.subjects[0].documents[0].physical_pixel_sizes is not None


def test_update_project_from_v0_1_5(
    brainways_project_path_v0_1_5: Path, mock_subject_documents
):
    update_project_from_previous_versions(brainways_project_path_v0_1_5)
    project = BrainwaysProject.open(brainways_project_path_v0_1_5, lazy_init=True)
    assert project.subjects[0].documents[0].params == mock_subject_documents[0].params
    assert project.subjects[0].subject_info.name == project.subjects[0]._save_dir.name
