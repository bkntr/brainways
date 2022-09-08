import pickle
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest
import torch
from bg_atlasapi.structure_class import StructuresDict
from PIL import Image
from pytest import fixture

from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    TPSTransformParams,
)
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_project_settings import (
    ProjectDocument,
    ProjectSettings,
)
from brainways.utils.atlas.duracell_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers.base import ImageReader


@fixture(autouse=True)
def seed():
    np.random.seed(0)


@fixture
def mock_atlas(test_data: Tuple[np.ndarray, AtlasSlice]) -> BrainwaysAtlas:
    test_image, test_atlas_slice = test_data
    ATLAS_SIZE = 32
    ATLAS_DEPTH = 10
    mock_atlas = create_autospec(BrainwaysAtlas)
    mock_atlas.bounding_boxes = [(0, 0, ATLAS_SIZE, ATLAS_SIZE)] * ATLAS_DEPTH
    mock_atlas.shape = (ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.reference = torch.rand(ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.atlas = Mock()
    mock_atlas.atlas.structure_from_coords = Mock(return_value=10)
    mock_atlas.atlas.resolution = (1, 2, 3)
    mock_atlas.atlas.atlas_name = "MOCK_ATLAS"
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
        {
            "name": "test_region2",
            "acronym": "TEST2",
            "id": 11,
            "structure_id_path": [1, 11],
            "rgb_triplet": [255, 255, 255],
            "mesh_filename": Path("/"),
        },
    ]
    structures = StructuresDict(structures_list=structures_list)
    mock_atlas.atlas.structures = structures
    mock_atlas.slice = Mock(return_value=test_atlas_slice)
    return mock_atlas


@fixture(scope="session")
def test_data() -> Tuple[np.ndarray, AtlasSlice]:
    npz = np.load(str(Path(__file__).parent.parent.parent / "data/test_data.npz"))
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
def test_image_path() -> Path:
    return Path(__file__).parent.parent.parent / "data/test_image.jpg"


@fixture(scope="session")
def test_image_size(test_data: Tuple[np.ndarray, AtlasSlice]) -> ImageSizeHW:
    input, atlas_size = test_data
    return input.shape


@fixture(autouse=True, scope="session")
def image_reader_mock(test_data: Tuple[np.ndarray, AtlasSlice]):
    mock_image_reader = create_autospec(ImageReader)
    test_image, test_atlas_slice = test_data
    HEIGHT = test_image.shape[0]
    WIDTH = test_image.shape[1]
    mock_image_reader.read_image.return_value = test_image
    mock_image_reader.scene_bb = (0, 0, WIDTH, HEIGHT)

    mock_get_scenes = Mock(return_value=[0])

    with patch(
        "brainways.utils.io_utils.readers.get_reader", return_value=mock_image_reader
    ), patch(
        "brainways.utils.io_utils.readers.get_scenes", return_value=mock_get_scenes
    ):
        yield


@pytest.fixture
def mock_image_path(test_data: Tuple[np.ndarray, AtlasSlice], tmpdir) -> ImagePath:
    image, _ = test_data
    image_path = ImagePath(str(tmpdir / "image.jpg"), scene=0)
    Image.fromarray(image).save(image_path.filename)
    return image_path


@pytest.fixture
def mock_project_documents(
    mock_image_path: ImagePath, test_data: Tuple[np.ndarray, AtlasSlice]
) -> List[ProjectDocument]:
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
            points_src=tps_points,
            points_dst=tps_points,
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
            ProjectDocument(
                path=doc_image_path,
                image_size=(image_height, image_width),
                lowres_image_size=(image_height, image_width),
                params=params,
                cells=np.array([[0.5, 0.5]]),
                ignore=i == 0,
            )
        )
    return documents


@pytest.fixture
def mock_project_settings() -> ProjectSettings:
    return ProjectSettings(atlas="MOCK_ATLAS", channel=0)


@pytest.fixture
def project_path(
    tmpdir,
    mock_project_settings: ProjectSettings,
    mock_project_documents: List[ProjectDocument],
) -> Path:
    project_path = Path(tmpdir) / "project/brainways.bin"
    project_path.parent.mkdir()
    serialized_project_settings = asdict(mock_project_settings)
    serialized_project_documents = [asdict(doc) for doc in mock_project_documents]
    with open(project_path, "wb") as f:
        pickle.dump((serialized_project_settings, serialized_project_documents), f)
    yield project_path


@pytest.fixture
def brainways_project(
    project_path: Path,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_atlas: BrainwaysAtlas,
) -> BrainwaysProject:
    brainways_project = BrainwaysProject.open(project_path)
    brainways_project.atlas = mock_atlas
    for document in brainways_project.documents:
        brainways_project.read_lowres_image(document)
    return brainways_project
