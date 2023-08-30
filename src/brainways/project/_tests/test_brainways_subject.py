from dataclasses import replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainways.pipeline.brainways_params import CellDetectorParams
from brainways.pipeline.cell_detector import CellDetector
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo, SubjectInfo
from brainways.utils.atlas.brainways_atlas import AtlasSlice
from brainways.utils.cell_detection_importer.brainways_cell_detection_importer import (
    BrainwaysCellDetectionsImporter,
)
from brainways.utils.io_utils import ImagePath


def test_create_subject(brainways_project: BrainwaysProject):
    subject = BrainwaysSubject.create(
        subject_info=SubjectInfo(name="test_subject", conditions={"condition": "a"}),
        project=brainways_project,
    )
    assert (subject._save_dir / "data.bws").exists()


def test_open_subject(
    subject_path: Path,
    mock_subject_info: SubjectInfo,
    mock_subject_documents: List[SliceInfo],
    brainways_project: BrainwaysProject,
):
    brainways_subject = BrainwaysSubject.open(subject_path, project=brainways_project)
    assert brainways_subject.subject_info == mock_subject_info
    assert brainways_subject.documents == mock_subject_documents
    assert brainways_subject.project == brainways_project


def test_thumbnails_root(brainways_subject: BrainwaysSubject):
    assert (
        brainways_subject.thumbnails_root == brainways_subject._save_dir / "thumbnails"
    )


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


def test_save(
    brainways_subject: BrainwaysSubject,
):
    brainways_subject.subject_info = replace(
        brainways_subject.subject_info, name="test_save"
    )
    brainways_subject.save()
    opened_subject = BrainwaysSubject.open(
        brainways_subject._save_dir / "data.bws", project=brainways_subject.project
    )
    assert opened_subject.subject_info == brainways_subject.subject_info


def test_create_in_already_existing_directory(brainways_project: BrainwaysProject):
    with pytest.raises(FileExistsError):
        BrainwaysSubject.create(
            subject_info=brainways_project.subjects[0].subject_info,
            project=brainways_project,
        )


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


def test_run_cell_detector(
    brainways_subject: BrainwaysSubject,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_image_path: ImagePath,
):
    cell_detector = create_autospec(CellDetector)
    cell_detector.cells.return_value = pd.DataFrame({"test": ["test"]})
    default_params = CellDetectorParams()
    brainways_subject.run_cell_detector(
        cell_detector=cell_detector, default_params=default_params
    )
    for i, document in brainways_subject.valid_documents:
        assert brainways_subject.cell_detections_path(document.path).exists()
