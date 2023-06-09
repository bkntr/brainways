import json
import pickle
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest
import torch
from aicsimageio.types import PhysicalPixelSizes
from bg_atlasapi.structure_class import StructuresDict
from PIL import Image
from pytest import fixture

from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    CellDetectorParams,
    TPSTransformParams,
)
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers.base import ImageReader
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


@fixture(autouse=True)
def seed():
    np.random.seed(0)


@fixture
def mock_atlas(test_data: Tuple[np.ndarray, AtlasSlice]) -> BrainwaysAtlas:
    test_image, test_atlas_slice = test_data
    test_atlas_slice.annotation[test_atlas_slice.annotation > 0] = 10
    ATLAS_SIZE = 32
    ATLAS_DEPTH = 10
    mock_atlas = create_autospec(BrainwaysAtlas)
    mock_atlas.bounding_boxes = [(0, 0, ATLAS_SIZE, ATLAS_SIZE)] * ATLAS_DEPTH
    mock_atlas.shape = (ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.reference = torch.rand(ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
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
    mock_atlas.brainglobe_atlas.structures = structures
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
    QupathReader.physical_pixel_sizes = PhysicalPixelSizes(Z=None, Y=10.0, X=10.0)
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
            points_src=tps_points,
            points_dst=tps_points,
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
                physical_pixel_sizes=(10.0, 10.0),
            )
        )
    return documents


@pytest.fixture
def mock_project_settings() -> ProjectSettings:
    return ProjectSettings(atlas="MOCK_ATLAS", channel=0)


@pytest.fixture
def subject_path(
    tmpdir,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
) -> Path:
    subject_path = Path(tmpdir) / "project/subject1/brainways.bin"
    subject_path.parent.mkdir(parents=True)
    serialized_subject_settings = asdict(mock_project_settings)
    serialized_subject_documents = [asdict(doc) for doc in mock_subject_documents]
    with open(subject_path, "wb") as f:
        pickle.dump((serialized_subject_settings, serialized_subject_documents), f)
    yield subject_path


@pytest.fixture
def project_path(
    subject_path: Path,
    mock_project_settings: ProjectSettings,
) -> Path:
    project_path = subject_path.parent.parent / "project.bwp"
    serialized_project_settings = asdict(mock_project_settings)
    with open(project_path, "w") as f:
        json.dump(serialized_project_settings, f)
    yield project_path


@pytest.fixture
def brainways_subject(
    subject_path: Path,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_atlas: BrainwaysAtlas,
) -> BrainwaysSubject:
    brainways_subject = BrainwaysSubject.open(subject_path)
    brainways_subject.atlas = mock_atlas
    for document in brainways_subject.documents:
        brainways_subject.read_lowres_image(document)
    return brainways_subject
