import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from aicsimageio.types import PhysicalPixelSizes
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.structure_class import StructuresDict
from PIL import Image
from pytest import fixture

from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    CellDetectorParams,
    TPSTransformParams,
)
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import (
    ProjectSettings,
    SliceInfo,
    SubjectFileFormat,
    SubjectInfo,
)
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


@fixture(autouse=True)
def seed():
    np.random.seed(0)


@fixture(autouse=True)
def safeguards(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        BrainGlobeAtlas,
        "download_extract_file",
        value=Mock(side_effect=Exception("don't download atlas in test")),
    )
    yield


@fixture
def mock_atlas(test_data: Tuple[np.ndarray, AtlasSlice]) -> BrainwaysAtlas:
    test_image, test_atlas_slice = test_data
    test_atlas_slice.annotation[test_atlas_slice.annotation > 0] = 10
    ATLAS_SIZE = 32
    ATLAS_DEPTH = 10

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

    mock_brainglobe_atlas = Mock()
    mock_brainglobe_atlas.structure_from_coords = Mock(return_value=10)
    mock_brainglobe_atlas.resolution = (10, 20, 30)
    mock_brainglobe_atlas.atlas_name = "MOCK_ATLAS"
    mock_brainglobe_atlas.reference = np.random.rand(
        ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE
    )
    mock_brainglobe_atlas.annotation = np.random.rand(
        ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE
    )
    mock_brainglobe_atlas.hemispheres = np.random.rand(
        ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE
    )
    mock_brainglobe_atlas.shape = (ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_brainglobe_atlas.structures = structures

    mock_atlas = BrainwaysAtlas(
        brainglobe_atlas=mock_brainglobe_atlas, exclude_regions=None
    )
    mock_atlas.slice = Mock(return_value=test_atlas_slice)
    return mock_atlas


@pytest.fixture
def mock_rat_atlas(mock_atlas: BrainwaysAtlas) -> BrainwaysAtlas:
    mock_atlas.brainglobe_atlas.atlas_name = "whs_sd_rat_39um"
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


@pytest.fixture
def mock_image_path(
    test_data: Tuple[np.ndarray, AtlasSlice], tmp_path: Path
) -> ImagePath:
    image, _ = test_data
    image_path = ImagePath(str(tmp_path / "image.jpg"), scene=0)
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
    tps_points = (
        (np.random.rand(10, 2) * (image_width, image_height))
        .astype(np.float32)
        .tolist()
    )

    params = BrainwaysParams(
        atlas=AtlasRegistrationParams(ap=5),
        affine=AffineTransform2DParams(),
        tps=TPSTransformParams(
            points_src=tps_points,
            points_dst=tps_points,
        ),
        cell=CellDetectorParams(normalizer="clahe", normalizer_range=(0.98, 0.997)),
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
    tmp_path: Path,
    mock_subject_file_format: SubjectFileFormat,
) -> Path:
    subject_path = tmp_path / "test_subject/data.bws"
    subject_path.parent.mkdir(parents=True)
    serialized_subject_file_format = asdict(mock_subject_file_format)
    with open(subject_path, "w") as f:
        json.dump(serialized_subject_file_format, f)
    yield subject_path


@pytest.fixture
def mock_subject_info() -> SubjectInfo:
    return SubjectInfo(name="subject1", conditions={"condition": "a"})


@pytest.fixture
def mock_subject_file_format(
    mock_subject_info: SubjectInfo, mock_subject_documents: List[SliceInfo]
) -> SubjectFileFormat:
    return SubjectFileFormat(
        subject_info=mock_subject_info, slice_infos=mock_subject_documents
    )


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
def brainways_project(
    mock_subject_info: SubjectInfo,
    mock_subject_documents: List[SliceInfo],
    mock_project_settings: ProjectSettings,
    mock_atlas: BrainwaysAtlas,
    test_data: Tuple[np.ndarray, AtlasSlice],
    tmp_path: Path,
) -> BrainwaysProject:
    project_path = tmp_path / "project/project.bwp"
    project_path.parent.mkdir()
    brainways_project = BrainwaysProject(
        subjects=[],
        settings=mock_project_settings,
        path=project_path,
        lazy_init=True,
    )
    brainways_project._atlas = mock_atlas
    brainways_project.load_pipeline()

    # add mock subject to project
    brainways_project.add_subject(mock_subject_info)
    brainways_project.add_subject(
        SubjectInfo(name="subject2", conditions={"condition": "b"})
    )
    for brainways_subject in brainways_project.subjects:
        brainways_subject.documents = deepcopy(mock_subject_documents)
        for document in brainways_subject.documents:
            brainways_subject.read_lowres_image(document)
    return brainways_project


@pytest.fixture
def brainways_subject(brainways_project: BrainwaysProject) -> BrainwaysSubject:
    return brainways_project.subjects[0]
