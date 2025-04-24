import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, create_autospec

import napari
import numpy as np
import pytest
import torch
from brainglobe_atlasapi.structure_class import StructuresDict
from PIL import Image
from pytest import MonkeyPatch, fixture
from qtpy.QtWidgets import QApplication

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    CellDetectorParams,
    TPSTransformParams,
)
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import ProjectSettings, SliceInfo, SubjectInfo
from brainways.ui.brainways_ui import BrainwaysUI
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.paths import get_brainways_dir
from brainways.utils.setup import BrainwaysSetup


@fixture(scope="session", autouse=True)
def env_config():
    """
    Configure environment variables needed for the test session
    """

    # This makes QT render everything offscreen and thus prevents
    # any Modals / Dialogs or other Widgets being rendered on the screen while running unit tests
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    yield

    os.environ.pop("QT_QPA_PLATFORM")


@fixture(autouse=True)
def setup_qt(qapp: QApplication):
    # the pytestqt.qapp fixture sets up the QApplication required to run QT code
    # see https://pytest-qt.readthedocs.io/en/latest/reference.html
    yield


@fixture
def napari_viewer(make_napari_viewer_proxy) -> napari.Viewer:
    return make_napari_viewer_proxy()


@fixture
def app(
    napari_viewer: napari.Viewer,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_atlas: BrainwaysAtlas,
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(BrainwaysAtlas, "load", Mock(return_value=mock_atlas))
    monkeypatch.setattr(
        AtlasRegistration,
        "download_model",
        Mock(side_effect=Exception("don't download model in tests")),
    )
    monkeypatch.setattr(
        AtlasRegistration,
        "is_model_available",
        Mock(return_value=False),
    )
    monkeypatch.setattr(
        BrainwaysSetup,
        "run",
        Mock(side_effect=Exception("don't setup in tests")),
    )
    get_brainways_dir()  # TODO: remove after brainways 0.10.1
    BrainwaysSetup.set_initialized()
    app = BrainwaysUI(napari_viewer, async_disabled=True)
    yield app


@fixture
def opened_app(
    app: BrainwaysUI,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_project: BrainwaysProject,
):
    app.open_project_async(mock_project.path)
    return app


@fixture(autouse=True)
def seed():
    np.random.seed(0)


@fixture
def mock_atlas(test_data: Tuple[np.ndarray, AtlasSlice]) -> BrainwaysAtlas:
    test_image, test_atlas_slice = test_data
    ATLAS_SIZE = 2
    ATLAS_DEPTH = 512
    mock_atlas = create_autospec(BrainwaysAtlas)
    mock_atlas.bounding_box = Mock(return_value=(0, 0, ATLAS_SIZE, ATLAS_SIZE))
    mock_atlas.shape = (ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.reference = torch.rand(ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.annotation = torch.rand(ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.brainglobe_atlas = Mock()
    mock_atlas.brainglobe_atlas.structure_from_coords = Mock(return_value=10)
    mock_atlas.brainglobe_atlas.resolution = (1, 2, 3)
    mock_atlas.brainglobe_atlas.atlas_name = "MOCK_ATLAS"
    structures_list = [
        {
            "name": "root",
            "acronym": "root",
            "id": 1,
            "structure_id_path": [1],
            "rgb_triplet": [0, 0, 0],
            "mesh_filename": Path("/"),
        },
        {
            "name": "test_region",
            "acronym": "TEST",
            "id": 10,
            "structure_id_path": [1, 10],
            "rgb_triplet": [255, 255, 255],
            "mesh_filename": Path("/"),
        },
    ]
    structures = StructuresDict(structures_list=structures_list)
    mock_atlas.brainglobe_atlas.structures = structures
    mock_atlas.brainglobe_atlas.reference = torch.zeros((3, 3, 3))
    mock_atlas.slice = Mock(return_value=test_atlas_slice)
    return mock_atlas


@fixture(scope="session")
def test_data() -> Tuple[np.ndarray, AtlasSlice]:
    npz = np.load(
        str(Path(__file__).parent.parent.parent.parent / "data/test_data.npz")
    )
    input = npz["input"]
    reference = npz["atlas_slice_reference"]
    annotation = npz["atlas_slice_annotation"]
    hemispheres = npz["atlas_slice_hemispheres"]
    atlas_slice = AtlasSlice(
        reference=torch.as_tensor(reference),
        annotation=torch.as_tensor(annotation),
        hemispheres=torch.as_tensor(hemispheres),
    )
    return input, atlas_slice


@fixture(scope="session")
def test_image_size(test_data: Tuple[np.ndarray, AtlasSlice]) -> ImageSizeHW:
    input, atlas_size = test_data
    return input.shape


@pytest.fixture
def mock_image_path(
    test_data: Tuple[np.ndarray, AtlasSlice], tmp_path: Path
) -> ImagePath:
    image, _ = test_data
    image_path = ImagePath(str(tmp_path / "image.jpg"), scene=0)
    Image.fromarray(image).save(image_path.filename)
    return image_path


@pytest.fixture
def mock_subject_documents(
    mock_image_path: ImagePath, test_data: Tuple[np.ndarray, AtlasSlice]
) -> List[SliceInfo]:
    test_image, test_atlas_slice = test_data
    image_height = test_image.shape[0]
    image_width = test_image.shape[1]
    tps_points = (np.random.rand(10, 2) * (image_width, image_height)).astype(
        np.float32
    )

    params = BrainwaysParams(
        atlas=AtlasRegistrationParams(ap=5),
        affine=AffineTransform2DParams(),
        tps=TPSTransformParams(
            points_src=tps_points.tolist(),
            points_dst=tps_points.tolist(),
        ),
        cell=CellDetectorParams(
            normalizer="clahe",
        ),
    )
    documents = []
    for i in range(3):
        doc_image_filename_name = f"{Path(mock_image_path.filename).stem}_{i}.jpg"
        doc_image_filename = Path(mock_image_path.filename).with_name(
            doc_image_filename_name
        )
        shutil.copy(mock_image_path.filename, doc_image_filename)
        doc_image_path = replace(mock_image_path, filename=str(doc_image_filename))
        documents.append(
            SliceInfo(
                path=doc_image_path,
                image_size=(image_height, image_width),
                lowres_image_size=(image_height, image_width),
                params=params,
                ignore=i == 0,
            )
        )
    return documents


@pytest.fixture
def mock_project_settings() -> ProjectSettings:
    return ProjectSettings(
        atlas="MOCK_ATLAS", channel=0, condition_names=["condition1", "condition2"]
    )


@pytest.fixture
def mock_project(
    tmp_path: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
) -> BrainwaysProject:
    project_path = tmp_path / "project/project.bwp"
    project_path.parent.mkdir()
    project = BrainwaysProject.create(
        project_path, settings=mock_project_settings, lazy_init=True
    )
    subject1 = project.add_subject(
        SubjectInfo(
            name="subject1", conditions={"condition1": "c11", "condition2": "c21"}
        )
    )
    subject1.documents = mock_subject_documents
    subject1.save()
    subject2 = project.add_subject(
        SubjectInfo(
            name="subject2", conditions={"condition1": "c12", "condition2": "c22"}
        )
    )
    subject2.documents = mock_subject_documents
    subject2.save()
    return project
