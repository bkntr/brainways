import os
import pickle
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
def brainways_project_path_v0_1_1(mock_image_path: ImagePath, tmpdir) -> Path:
    project_path = (
        Path(os.path.realpath(__file__)).parent / "test_projects/v0.1.1/project.bwp"
    )
    tmp_project_dir = Path(tmpdir / "v0.1.1")
    shutil.copytree(project_path.parent, tmp_project_dir)
    rewrite_image_path(project_path=tmp_project_dir, image_path=mock_image_path)
    return tmp_project_dir / "project.bwp"


@pytest.fixture
def brainways_project_path_v0_1_4(mock_image_path: ImagePath, tmpdir) -> Path:
    project_path = (
        Path(os.path.realpath(__file__)).parent / "test_projects/v0.1.4/project.bwp"
    )
    tmp_project_dir = Path(tmpdir / "v0.1.4")
    shutil.copytree(project_path.parent, tmp_project_dir)
    rewrite_image_path(project_path=tmp_project_dir, image_path=mock_image_path)
    return tmp_project_dir / "project.bwp"


@pytest.fixture
def brainways_project_path_v0_1_5(mock_image_path: ImagePath, tmpdir) -> Path:
    project_path = (
        Path(os.path.realpath(__file__)).parent / "test_projects/v0.1.5/project.bwp"
    )
    tmp_project_dir = Path(tmpdir / "v0.1.5")
    shutil.copytree(project_path.parent, tmp_project_dir)
    rewrite_image_path(project_path=tmp_project_dir, image_path=mock_image_path)
    return tmp_project_dir / "project.bwp"


def rewrite_image_path(project_path: Path, image_path: ImagePath):
    brainways_subject_paths = project_path.parent.rglob("brainways.bin")
    for brainways_subject_path in brainways_subject_paths:
        with open(brainways_subject_path, "rb") as f:
            serialized_settings, serialized_slice_infos = pickle.load(f)
        for serialized_slice_info in serialized_slice_infos:
            serialized_slice_info["path"]["filename"] = image_path.filename
        with open(brainways_subject_path, "wb") as f:
            pickle.dump((serialized_settings, serialized_slice_infos), f)
