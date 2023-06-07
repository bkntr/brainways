from pathlib import Path

from brainways.project._utils import update_project_from_previous_versions
from brainways.project.brainways_project import BrainwaysProject


def test_update_project_from_v0_1_1(brainways_project_path_v0_1_1: Path):
    update_project_from_previous_versions(brainways_project_path_v0_1_1)
    BrainwaysProject.open(brainways_project_path_v0_1_1, lazy_init=True)
