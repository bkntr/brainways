from dataclasses import replace
from pathlib import Path
from typing import List, Tuple, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainways.pipeline.brainways_params import (
    AtlasRegistrationParams,
    BrainwaysParams,
    CellDetectorParams,
)
from brainways.pipeline.cell_detector import CellDetector
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import MaskFileFormat, SliceInfo, SubjectInfo
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


def test_empty_cell_count_summary_no_valid_documents(
    brainways_subject: BrainwaysSubject,
):
    brainways_subject.documents = []
    summary = brainways_subject.cell_count_summary()
    assert summary is None


def test_set_rotation_updates_subject_info():
    # Create a BrainwaysSubject instance
    brainways_subject = BrainwaysSubject(
        subject_info=SubjectInfo(name="test_subject"),
        slice_infos=[],
        project=MagicMock(),
    )

    # Initial rotation values
    initial_rotation = brainways_subject.subject_info.rotation
    new_rotation = (45.0, 30.0)

    # Set new rotation
    brainways_subject.set_rotation(*new_rotation)

    # Verify subject_info rotation is updated
    assert brainways_subject.subject_info.rotation == new_rotation
    assert brainways_subject.subject_info.rotation != initial_rotation


def test_set_rotation_updates_slice_infos():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        SliceInfo(
            path=ImagePath("test_image.jpg"),
            image_size=(100, 100),
            lowres_image_size=(100, 100),
            params=BrainwaysParams(atlas=AtlasRegistrationParams()),
        )
        for _ in range(3)
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    new_rotation = (45.0, 30.0)

    # Set new rotation
    brainways_subject.set_rotation(*new_rotation)

    # Verify each document's atlas_params rotation is updated
    for document in brainways_subject.documents:
        assert document.params.atlas.rot_horizontal == new_rotation[0]
        assert document.params.atlas.rot_sagittal == new_rotation[1]


def test_set_rotation_with_none_atlas_params():
    # Create a BrainwaysSubject instance with documents having None atlas_params
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        SliceInfo(
            path=ImagePath("test_image.jpg"),
            image_size=(100, 100),
            lowres_image_size=(100, 100),
            params=BrainwaysParams(atlas=AtlasRegistrationParams()),
        ),
        SliceInfo(
            path=ImagePath("test_image.jpg"),
            image_size=(100, 100),
            lowres_image_size=(100, 100),
            params=BrainwaysParams(atlas=None),
        ),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    new_rotation = (45.0, 30.0)

    # Set new rotation
    brainways_subject.set_rotation(*new_rotation)

    # Verify subject_info rotation is updated
    assert brainways_subject.subject_info.rotation == new_rotation

    # Verify documents with atlas_params are updated
    assert brainways_subject.documents[0].params.atlas.rot_horizontal == new_rotation[0]
    assert brainways_subject.documents[0].params.atlas.rot_sagittal == new_rotation[1]

    # Verify documents without atlas_params are not updated
    assert slice_infos[1].params.atlas is None


def create_slice_info(ap_value=None, ignore: bool = False):
    return SliceInfo(
        path=ImagePath("test_image.jpg"),
        image_size=(100, 100),
        lowres_image_size=(100, 100),
        params=BrainwaysParams(
            atlas=AtlasRegistrationParams(ap=ap_value) if ap_value is not None else None
        ),
        ignore=ignore,
    )


def test_evenly_space_slices_on_ap_axis():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(20),
        create_slice_info(30),
        create_slice_info(40),
        create_slice_info(50),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    expected_aps = [10, 20, 30, 40, 50]
    assert len(brainways_subject.valid_documents) == len(expected_aps)
    for i, (_, document) in enumerate(brainways_subject.valid_documents):
        assert document.params.atlas.ap == expected_aps[i]


def test_evenly_space_unordered_slices_on_ap_axis():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(100),
        create_slice_info(90),
        create_slice_info(80),
        create_slice_info(50),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    expected_aps = [10, 20, 30, 40, 50]
    assert len(brainways_subject.valid_documents) == len(expected_aps)
    for i, (_, document) in enumerate(brainways_subject.valid_documents):
        assert document.params.atlas.ap == expected_aps[i]


def test_evenly_space_slices_on_ap_axis_with_ignored_documents():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(20),
        create_slice_info(30, ignore=True),
        create_slice_info(40),
        create_slice_info(70),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    expected_aps = [10, 30, 50, 70]
    assert len(brainways_subject.valid_documents) == len(expected_aps)
    for i, (_, document) in enumerate(brainways_subject.valid_documents):
        assert document.params.atlas.ap == expected_aps[i]


def test_evenly_space_slices_on_ap_axis_two_or_fewer_documents():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(20),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    assert brainways_subject.documents[0].params.atlas.ap == 10
    assert brainways_subject.documents[1].params.atlas.ap == 20


def test_evenly_space_slices_on_ap_axis_missing_atlas_params():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(),
        create_slice_info(0),
        create_slice_info(),
        create_slice_info(50),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    expected_aps = [10, 20, 30, 40, 50]
    assert len(brainways_subject.valid_documents) == len(expected_aps)
    for i, (_, document) in enumerate(brainways_subject.valid_documents):
        assert document.params.atlas.ap == expected_aps[i]


def test_evenly_spaced_slices_on_ap_axis_no_valid_documents():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(ignore=True),
        create_slice_info(ignore=True),
        create_slice_info(ignore=True),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    assert len(brainways_subject.valid_documents) == 0


def test_evenly_spaced_slices_on_ap_axis_no_slices():
    subject_info = SubjectInfo(name="test_subject")
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=[], project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    assert len(brainways_subject.valid_documents) == 0


def test_evenly_spaced_slices_on_ap_axis_adds_subject_rotation():
    subject_info = SubjectInfo(name="test_subject", rotation=(45.0, 30.0))
    slice_infos = [
        create_slice_info(10),
        create_slice_info(None),
        create_slice_info(50),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    brainways_subject.evenly_space_slices_on_ap_axis()

    tested_slice_info = slice_infos[1]
    assert tested_slice_info.params.atlas.rot_horizontal == 45.0
    assert tested_slice_info.params.atlas.rot_sagittal == 30.0


def test_evenly_spaced_slices_on_ap_axis_raises_error_if_last_slice_missing_params():
    subject_info = SubjectInfo(name="test_subject")
    slice_infos = [
        create_slice_info(10),
        create_slice_info(20),
        create_slice_info(None),
    ]
    brainways_subject = BrainwaysSubject(
        subject_info=subject_info, slice_infos=slice_infos, project=MagicMock()
    )

    with pytest.raises(ValueError):
        brainways_subject.evenly_space_slices_on_ap_axis()


@pytest.mark.parametrize(
    "cell_params",
    [
        pytest.param(CellDetectorParams(), id="use_custom_params"),
        pytest.param(None, id="use_default_params"),
    ],
)
@pytest.mark.parametrize(
    "save_cell_detection_masks_file_format",
    [
        pytest.param(None, id="dont_export_mask"),
        pytest.param(MaskFileFormat.NPZ, id="export_mask"),
    ],
)
@patch("brainways.project.brainways_subject.export_mask")
def test_run_cell_detector(
    export_mask_mock: Mock,
    tmp_path,
    brainways_subject: BrainwaysSubject,
    cell_params: Union[CellDetectorParams, None],
    save_cell_detection_masks_file_format: Union[MaskFileFormat, None],
):
    # Mock SliceInfo
    slice_info = MagicMock(spec=SliceInfo)
    slice_info.path = tmp_path / "slice_info"
    slice_info.image_reader.return_value.get_image_dask_data.return_value.compute.return_value = np.random.rand(
        100, 100
    )
    slice_info.params.cell = cell_params

    # Mock CellDetector
    mock_cell_mask = np.random.randint(0, 2, (100, 100))
    cell_detector = MagicMock(spec=CellDetector)
    cell_detector.run_cell_detector.return_value = mock_cell_mask
    cell_detector.cells.return_value.to_csv = MagicMock()

    # Mock default params
    default_params = CellDetectorParams(normalizer="test_default")

    # Run the method
    brainways_subject.run_cell_detector(
        slice_info=slice_info,
        cell_detector=cell_detector,
        default_params=default_params,
        save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
    )

    # Assertions
    expected_cell_params = cell_params if cell_params is not None else default_params
    cell_detector.run_cell_detector.assert_called_once_with(
        slice_info.image_reader().get_image_dask_data().compute(),
        params=expected_cell_params,
        physical_pixel_sizes=slice_info.physical_pixel_sizes,
    )
    cell_detector.cells.return_value.to_csv.assert_called_once_with(
        brainways_subject.cell_detections_path(slice_info.path), index=False
    )
    if save_cell_detection_masks_file_format is None:
        export_mask_mock.assert_not_called()
    else:
        expected_mask_path = (
            brainways_subject.project.path.parent
            / "__outputs__"
            / "cell_detection_masks"
            / brainways_subject.subject_info.name
            / Path(str(slice_info.path)).name
        )
        export_mask_mock.assert_called_once_with(
            data=mock_cell_mask,
            path=expected_mask_path,
            file_format=save_cell_detection_masks_file_format,
        )


def test_clear_cell_detection_file_exists(tmp_path):
    # Create a BrainwaysSubject instance
    brainways_subject = BrainwaysSubject(
        subject_info=SubjectInfo(name="test_subject"),
        slice_infos=[],
        project=MagicMock(),
    )

    # Create a mock SliceInfo instance
    slice_info = MagicMock(spec=SliceInfo)
    slice_info.path = tmp_path / "slice_info"

    # Mock the cell_detections_path method to return a specific path
    cell_detections_path = tmp_path / "cell_detections.csv"
    cell_detections_path.touch()  # Create the file

    brainways_subject.cell_detections_path = MagicMock(
        return_value=cell_detections_path
    )

    # Call the method
    brainways_subject.clear_cell_detection(slice_info)

    # Assert that the file was deleted
    assert not cell_detections_path.exists()


def test_clear_cell_detection_file_does_not_exist(tmp_path):
    # Create a BrainwaysSubject instance
    brainways_subject = BrainwaysSubject(
        subject_info=SubjectInfo(name="test_subject"),
        slice_infos=[],
        project=MagicMock(),
    )

    # Create a mock SliceInfo instance
    slice_info = MagicMock(spec=SliceInfo)
    slice_info.path = tmp_path / "slice_info"

    # Mock the cell_detections_path method to return a specific path
    cell_detections_path = tmp_path / "cell_detections.csv"

    brainways_subject.cell_detections_path = MagicMock(
        return_value=cell_detections_path
    )

    # Call the method
    brainways_subject.clear_cell_detection(slice_info)

    # Assert that the file does not exist
    assert not cell_detections_path.exists()
