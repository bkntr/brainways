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
