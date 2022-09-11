from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_project_settings import (
    ProjectDocument,
    ProjectSettings,
)
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.io_utils import ImagePath


@pytest.fixture
def brainways_tmp_project(
    mock_project_settings: ProjectSettings,
    mock_project_documents: List[ProjectDocument],
    mock_image_path: ImagePath,
) -> BrainwaysProject:
    brainways_project = BrainwaysProject(
        settings=mock_project_settings, documents=mock_project_documents
    )
    brainways_project.add_image(path=mock_image_path)
    return brainways_project


def test_new_project_saves_in_tmpdir(
    brainways_tmp_project: BrainwaysProject,
):
    assert brainways_tmp_project.project_path == Path(
        brainways_tmp_project._tmpdir.name
    )


def test_thumbnails_root(brainways_project: BrainwaysProject):
    assert (
        brainways_project.thumbnails_root
        == brainways_project.project_path / "thumbnails"
    )


def test_open_project(
    project_path: Path,
    mock_project_settings: ProjectSettings,
    mock_project_documents: List[ProjectDocument],
):
    brainways_project = BrainwaysProject.open(project_path)
    assert brainways_project.project_path == project_path.parent
    assert brainways_project.settings == mock_project_settings
    assert brainways_project.documents == mock_project_documents
    assert brainways_project.atlas is None
    assert brainways_project.pipeline is None


def test_open_project_with_atlas_and_pipeline(
    brainways_project: BrainwaysProject,
    mock_atlas: BrainwaysAtlas,
    project_path: Path,
    mock_project_settings: ProjectSettings,
    mock_project_documents: List[ProjectDocument],
):
    mock_pipeline = Mock()
    brainways_project = BrainwaysProject.open(
        project_path, atlas=mock_atlas, pipeline=mock_pipeline
    )
    assert brainways_project.project_path == project_path.parent
    assert brainways_project.settings == mock_project_settings
    assert brainways_project.documents == mock_project_documents
    assert brainways_project.atlas == mock_atlas
    assert brainways_project.pipeline == mock_pipeline


def test_import_cells(brainways_project: BrainwaysProject, tmpdir):
    cells = np.random.rand(len(brainways_project.documents), 3, 2)

    # create cells csvs
    root = Path(tmpdir)
    for i, document in enumerate(brainways_project.documents):
        csv_filename = (
            f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
        )
        df = pd.DataFrame(
            {
                "centroid-0": cells[i, :, 0],
                "centroid-1": cells[i, :, 1],
                "area": 400,
            }
        )
        df.to_csv(root / csv_filename)
    brainways_project.import_cells(root)

    for i, document in brainways_project.valid_documents:
        expected = cells[i, :, ::-1]
        assert np.allclose(document.cells, expected)


def test_cell_count_summary(brainways_project: BrainwaysProject):
    summary = brainways_project.cell_count_summary()
    expected = pd.DataFrame(
        [
            {
                "id": 10,
                "acronym": "TEST",
                "name": "test_region",
                "cell_count": 2,
                "total_area_um2": 0,
                "cells_per_um2": 2.0,
            },
            {
                "id": 1,
                "acronym": "root",
                "name": "root",
                "cell_count": 2,
                "total_area_um2": 0,
                "cells_per_um2": 2.0,
            },
        ]
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_add_image_adds_document(
    brainways_project: BrainwaysProject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
):
    image, _ = test_data
    brainways_project.add_image(path=mock_image_path)
    expected_document = ProjectDocument(
        path=mock_image_path,
        image_size=image.shape,
        lowres_image_size=image.shape,
    )
    assert brainways_project.documents[-1] == expected_document


def test_add_image_saves_lowres_image(
    brainways_project: BrainwaysProject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
):
    image, _ = test_data
    thumbnail_path = brainways_project.thumbnail_path(mock_image_path)
    brainways_project.add_image(path=mock_image_path)
    thumbnail_image = np.array(Image.open(thumbnail_path))
    assert image.shape == thumbnail_image.shape


def test_close_project(brainways_project: BrainwaysProject):
    brainways_project.close()
    assert brainways_project.documents == []
    assert brainways_project.settings is None
    assert brainways_project.atlas is None
    assert brainways_project.pipeline is None
    assert brainways_project.project_path == Path(brainways_project._tmpdir.name)


def test_close_project_clears_tmpdir(brainways_tmp_project: BrainwaysProject):
    project_tmp_dir = Path(brainways_tmp_project._tmpdir.name)
    assert len(list(project_tmp_dir.glob("*"))) > 0
    brainways_tmp_project.close()
    assert len(list(project_tmp_dir.glob("*"))) == 0


def test_read_lowres_image_reads_from_thumbnail_cache(
    brainways_project: BrainwaysProject,
    mock_image_path: ImagePath,
):
    thumbnail_path = brainways_project.thumbnail_path(mock_image_path, channel=0)
    random_image = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    Image.fromarray(random_image).save(thumbnail_path)
    cached_thumbnail_image = np.array(Image.open(thumbnail_path))
    opened_image = brainways_project.read_lowres_image(
        ProjectDocument(
            mock_image_path,
            image_size=cached_thumbnail_image.shape,
            lowres_image_size=cached_thumbnail_image.shape,
        )
    )
    assert np.allclose(opened_image, cached_thumbnail_image)


def test_save_moves_project(
    brainways_project: BrainwaysProject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
    tmpdir,
):
    image, _ = test_data
    save_path = Path(tmpdir) / "save_project"
    brainways_project.save(save_path)
    assert brainways_project.project_path == save_path
    assert brainways_project.thumbnail_path(
        brainways_project.documents[0].path, channel=0
    ).exists()


def test_init_brainways_with_wrong_atlas_raises_exception(
    mock_project_settings: ProjectSettings,
):
    mock_atlas = Mock()
    mock_atlas.atlas.atlas_name = "test"
    with pytest.raises(ValueError):
        BrainwaysProject(settings=mock_project_settings, documents=[], atlas=mock_atlas)


def test_load_pipeline(brainways_project: BrainwaysProject, mock_atlas: BrainwaysAtlas):
    assert brainways_project.pipeline is None
    brainways_project.load_pipeline()
    assert brainways_project.pipeline is not None
