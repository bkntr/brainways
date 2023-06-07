import os
import shutil
from pathlib import Path
from typing import List

import pytest

from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.io_utils import ImagePath


@pytest.fixture
def brainways_tmp_subject(
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
    mock_image_path: ImagePath,
) -> BrainwaysSubject:
    brainways_subject = BrainwaysSubject(
        settings=mock_project_settings, documents=mock_subject_documents
    )
    brainways_subject.add_image(path=mock_image_path)
    return brainways_subject


@pytest.fixture
def brainways_project_path_v0_1_1(tmpdir) -> Path:
    v0_1_1_project_path = (
        Path(os.path.realpath(__file__)).parent / "test_projects/v0.1.1/project.bwp"
    )
    tmp_v0_1_1_dir = Path(tmpdir / "v0.1.1")
    shutil.copytree(v0_1_1_project_path.parent, tmp_v0_1_1_dir)
    return tmp_v0_1_1_dir / "project.bwp"
