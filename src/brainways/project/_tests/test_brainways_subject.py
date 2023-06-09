from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.cell_detection_importer.brainways_cell_detection_importer import (
    BrainwaysCellDetectionsImporter,
)
from brainways.utils.io_utils import ImagePath


def test_new_subject_saves_in_tmpdir(
    brainways_tmp_subject: BrainwaysSubject,
):
    assert brainways_tmp_subject.subject_path == Path(
        brainways_tmp_subject._tmpdir.name
    )


def test_thumbnails_root(brainways_subject: BrainwaysSubject):
    assert (
        brainways_subject.thumbnails_root
        == brainways_subject.subject_path / "thumbnails"
    )


def test_open_subject(
    subject_path: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
):
    brainways_subject = BrainwaysSubject.open(subject_path)
    assert brainways_subject.subject_path == subject_path.parent
    assert brainways_subject.settings == mock_project_settings
    assert brainways_subject.documents == mock_subject_documents
    assert brainways_subject.atlas is None
    assert brainways_subject.pipeline is None


def test_open_subject_with_atlas_and_pipeline(
    brainways_subject: BrainwaysSubject,
    mock_atlas: BrainwaysAtlas,
    subject_path: Path,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
):
    mock_pipeline = Mock()
    brainways_subject = BrainwaysSubject.open(
        subject_path, atlas=mock_atlas, pipeline=mock_pipeline
    )
    assert brainways_subject.subject_path == subject_path.parent
    assert brainways_subject.settings == mock_project_settings
    assert brainways_subject.documents == mock_subject_documents
    assert brainways_subject.atlas == mock_atlas
    assert brainways_subject.pipeline == mock_pipeline


def test_import_cells(brainways_subject: BrainwaysSubject, tmpdir):
    cells = np.random.rand(len(brainways_subject.documents), 3, 2)

    # create cells csvs
    root = Path(tmpdir)
    for i, document in enumerate(brainways_subject.documents):
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
    brainways_subject.import_cell_detections(
        root, cell_detection_importer=BrainwaysCellDetectionsImporter()
    )

    for i, document in brainways_subject.valid_documents:
        assert brainways_subject.cell_detections_path(document.path).exists()
        # expected_df = pd.DataFrame(
        #     {
        #         "x": cells[i, :, 1],
        #         "y": cells[i, :, 0],
        #     }
        # )
        # cell_detections = pd.read_csv(
        #     brainways_subject.cell_detections_path(document.path)
        # )
        # pd.testing.assert_frame_equal(cell_detections, expected_df)


def test_cell_detections_path(brainways_subject: BrainwaysSubject):
    assert (
        brainways_subject.cell_detections_path(brainways_subject.documents[0].path)
        == brainways_subject.cell_detections_root / "image_0.jpg [Scene #0].csv"
    )


def test_add_image_adds_document(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
):
    image, _ = test_data
    brainways_subject.add_image(path=mock_image_path)
    expected_document = SliceInfo(
        path=mock_image_path,
        image_size=image.shape,
        lowres_image_size=(788, 1024),
        physical_pixel_sizes=(10.0, 10.0),
    )
    assert brainways_subject.documents[-1] == expected_document


def test_add_image_saves_lowres_image(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
):
    image, _ = test_data
    thumbnail_path = brainways_subject.thumbnail_path(mock_image_path)
    brainways_subject.add_image(path=mock_image_path)
    thumbnail_image = np.array(Image.open(thumbnail_path))
    assert thumbnail_image.shape == (788, 1024)


def test_close_subject(brainways_subject: BrainwaysSubject):
    brainways_subject.close()
    assert brainways_subject.documents == []
    assert brainways_subject.settings is None
    assert brainways_subject.atlas is None
    assert brainways_subject.pipeline is None
    assert brainways_subject.subject_path is None


def test_close_subject_clears_tmpdir(brainways_tmp_subject: BrainwaysSubject):
    subject_tmp_dir = Path(brainways_tmp_subject._tmpdir.name)
    assert len(list(subject_tmp_dir.glob("*"))) > 0
    brainways_tmp_subject.close()
    assert len(list(subject_tmp_dir.glob("*"))) == 0


def test_read_lowres_image_reads_from_thumbnail_cache(
    brainways_subject: BrainwaysSubject,
    mock_image_path: ImagePath,
):
    thumbnail_path = brainways_subject.thumbnail_path(mock_image_path, channel=0)
    random_image = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    Image.fromarray(random_image).save(thumbnail_path)
    cached_thumbnail_image = np.array(Image.open(thumbnail_path))
    opened_image = brainways_subject.read_lowres_image(
        SliceInfo(
            mock_image_path,
            image_size=cached_thumbnail_image.shape,
            lowres_image_size=cached_thumbnail_image.shape,
        )
    )
    assert np.allclose(opened_image, cached_thumbnail_image)


def test_save_moves_subject(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
    tmpdir,
):
    image, _ = test_data
    save_path = Path(tmpdir) / "save_subject"
    brainways_subject.save(save_path)
    assert brainways_subject.subject_path == save_path
    assert brainways_subject.thumbnail_path(
        brainways_subject.documents[0].path, channel=0
    ).exists()


def test_save_to_empty_directory(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
    tmpdir,
):
    image, _ = test_data
    save_path = Path(tmpdir) / "save_subject"
    save_path.mkdir()
    brainways_subject.save(save_path)
    assert brainways_subject.subject_path == save_path
    assert brainways_subject.thumbnail_path(
        brainways_subject.documents[0].path, channel=0
    ).exists()


def test_save_to_already_existing_directory(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
    tmpdir,
):
    image, _ = test_data
    save_path = Path(tmpdir) / "save_subject"
    save_path.mkdir()
    (save_path / "test").touch()
    with pytest.raises(FileExistsError):
        brainways_subject.save(save_path)


def test_init_brainways_with_wrong_atlas_raises_exception(
    mock_project_settings: ProjectSettings,
):
    mock_atlas = Mock()
    mock_atlas.brainglobe_atlas.atlas_name = "test"
    with pytest.raises(ValueError):
        BrainwaysSubject(settings=mock_project_settings, documents=[], atlas=mock_atlas)


def test_init_subject_with_nonempty_directory(
    mock_project_settings: ProjectSettings, tmpdir
):
    (Path(tmpdir) / "test").touch()
    with pytest.raises(FileExistsError):
        BrainwaysSubject(settings=mock_project_settings, subject_path=Path(tmpdir))


def test_move_images_root(brainways_subject: BrainwaysSubject, tmpdir):
    new_images_root = Path(tmpdir) / "new"
    new_images_root.mkdir()
    new_image_path = (
        new_images_root / Path(brainways_subject.documents[0].path.filename).name
    )
    new_image_path.touch()
    brainways_subject.move_images_root(new_images_root)
    assert brainways_subject.documents[0].path.filename == str(new_image_path)


def test_move_images_root_with_base(brainways_subject: BrainwaysSubject, tmpdir):
    old_filename = Path(brainways_subject.documents[0].path.filename)
    old_images_root = old_filename.parent.parent
    new_images_root = Path(tmpdir) / "new"
    new_images_root.mkdir()
    (new_images_root / old_filename.parent.name).mkdir()
    new_image_path = new_images_root / old_filename.parent.name / old_filename.name
    new_image_path.touch()
    brainways_subject.move_images_root(new_images_root, old_images_root=old_images_root)
    assert brainways_subject.documents[0].path.filename == str(new_image_path)
