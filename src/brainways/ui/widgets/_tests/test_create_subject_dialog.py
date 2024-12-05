from dataclasses import replace
from unittest.mock import Mock

from aicsimageio.types import PhysicalPixelSizes
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QCheckBox

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import SliceInfo, SubjectInfo
from brainways.ui.utils.test_utils import worker_join
from brainways.ui.widgets.create_subject_dialog import CreateSubjectDialog
from brainways.utils.image import ImageSizeHW, get_resize_size
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import QupathReader


@fixture
def create_subject_dialog(
    qtbot: QtBot,
    mock_project: BrainwaysProject,
    mock_image_path: ImagePath,
    test_image_size: ImageSizeHW,
) -> CreateSubjectDialog:
    QupathReader.physical_pixel_sizes = PhysicalPixelSizes(Z=None, Y=10.0, X=10.0)
    create_subject_dialog = CreateSubjectDialog(mock_project, async_disabled=True)
    create_subject_dialog.new_subject(
        subject_id="test_subject", conditions={"condition1": "c1", "condition2": "c2"}
    )
    create_subject_dialog.add_filenames_async([str(mock_image_path.filename)])
    return create_subject_dialog


@fixture
def create_subject_document(
    qtbot: QtBot, mock_image_path: ImagePath, test_image_size: ImageSizeHW
) -> SliceInfo:
    return SliceInfo(
        path=mock_image_path,
        image_size=test_image_size,
        lowres_image_size=get_resize_size(
            test_image_size, (1024, 1024), keep_aspect=True
        ),
        physical_pixel_sizes=(10.0, 10.0),
    )


def test_documents(
    create_subject_dialog: CreateSubjectDialog,
    create_subject_document: SliceInfo,
):
    documents = create_subject_dialog.subject.documents
    expected = [create_subject_document]
    assert documents == expected


def test_ignore(
    create_subject_dialog: CreateSubjectDialog,
    create_subject_document: SliceInfo,
):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    documents = create_subject_dialog.subject.documents
    expected = [replace(create_subject_document, ignore=True)]
    assert documents == expected


def test_edit_subject(qtbot: QtBot, mock_project: BrainwaysProject, tmpdir):
    dialog = CreateSubjectDialog(project=mock_project, async_disabled=True)
    worker = dialog.edit_subject_async(subject_index=1, document_index=1)
    worker_join(worker, qtbot)
    assert dialog.subject == mock_project.subjects[1]
    assert dialog.files_table.rowCount() == len(mock_project.subjects[1].documents)
    selected_row = dialog.files_table.selectionModel().selectedRows()[0].row()
    assert selected_row == 1
    assert dialog.conditions_widget[0].value == "c12"
    assert dialog.conditions_widget[1].value == "c22"


def test_uncheck_check(create_subject_dialog: CreateSubjectDialog):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    checkbox.setChecked(True)
    assert create_subject_dialog.subject.documents[0].ignore is False


def test_conditions_initialized(create_subject_dialog: CreateSubjectDialog):
    assert create_subject_dialog.conditions_widget[0].value == "c1"
    assert create_subject_dialog.conditions_widget[1].value == "c2"


def test_conditions_changed_works(create_subject_dialog: CreateSubjectDialog):
    assert create_subject_dialog.subject is not None
    create_subject_dialog.conditions_widget[0].value = "mod1"
    create_subject_dialog.conditions_widget[1].value = "mod2"
    assert create_subject_dialog.subject.subject_info.conditions["condition1"] == "mod1"
    assert create_subject_dialog.subject.subject_info.conditions["condition2"] == "mod2"


def test_add_cell_detection_channels_checkboxes():
    mock_project = Mock()
    mock_project.subjects = []
    mock_project.settings.condition_names = []
    create_subject_dialog = CreateSubjectDialog(mock_project, async_disabled=True)
    create_subject_dialog.new_subject(
        subject_id="test_subject", conditions={"condition1": "c1", "condition2": "c2"}
    )
    assert create_subject_dialog.subject is not None

    # Setup
    cell_detection_channels = ["Channel 1", "Channel 2", "Channel 3"]
    create_subject_dialog.subject.subject_info.cell_detection_channels = [0, 2]

    # Execute
    create_subject_dialog._add_cell_detection_channels_checkboxes(
        cell_detection_channels
    )

    # Verify
    assert len(create_subject_dialog.cell_detection_channels_checkboxes) == 3
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[0].text()
        == "Channel 1"
    )
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[1].text()
        == "Channel 2"
    )
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[2].text()
        == "Channel 3"
    )
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[0].isChecked() is True
    )
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[1].isChecked() is False
    )
    assert (
        create_subject_dialog.cell_detection_channels_checkboxes[2].isChecked() is True
    )


def test_cell_detection_channel_checkbox_state_change():
    mock_project = Mock()
    mock_project.subjects = []
    mock_project.settings.condition_names = []
    create_subject_dialog = CreateSubjectDialog(mock_project, async_disabled=True)
    create_subject_dialog.new_subject(
        subject_id="test_subject", conditions={"condition1": "c1", "condition2": "c2"}
    )
    assert create_subject_dialog.subject is not None

    # Setup
    cell_detection_channels = ["Channel 1", "Channel 2", "Channel 3"]
    create_subject_dialog.subject.subject_info.cell_detection_channels = [0, 2]

    create_subject_dialog._add_cell_detection_channels_checkboxes(
        cell_detection_channels
    )

    # Execute
    checkbox = create_subject_dialog.cell_detection_channels_checkboxes[1]
    assert checkbox.isChecked() is False
    assert create_subject_dialog.subject.subject_info.cell_detection_channels == [
        0,
        2,
    ]
    checkbox.setChecked(True)

    # Verify
    assert checkbox.isChecked() is True
    assert create_subject_dialog.subject.subject_info.cell_detection_channels == [
        0,
        1,
        2,
    ]


def test_new_subject_with_existing_subjects():
    # Setup
    mock_project = Mock()
    mock_project.settings.condition_names = ["a", "b"]
    mock_subject = Mock()
    mock_subject.subject_info.registration_channel = 1
    mock_subject.subject_info.cell_detection_channels = [1, 2]
    mock_project.subjects = [mock_subject]
    create_subject_dialog = CreateSubjectDialog(mock_project, async_disabled=True)

    # Execute
    create_subject_dialog.new_subject(
        subject_id="new_subject", conditions={"a": "1", "b": "2"}
    )

    # Verify
    mock_project.add_subject.assert_called_once_with(
        SubjectInfo(
            name="new_subject",
            registration_channel=1,
            cell_detection_channels=[1, 2],
            conditions={"a": "1", "b": "2"},
        )
    )


def test_new_subject_without_existing_subjects():
    # Setup
    mock_project = Mock()
    mock_project.settings.condition_names = ["a", "b"]
    mock_project.subjects = []
    create_subject_dialog = CreateSubjectDialog(mock_project, async_disabled=True)

    # Execute
    create_subject_dialog.new_subject(
        subject_id="new_subject", conditions={"a": "1", "b": "2"}
    )

    # Verify
    mock_project.add_subject.assert_called_once_with(
        SubjectInfo(
            name="new_subject",
            registration_channel=0,
            cell_detection_channels=[0],
            conditions={"a": "1", "b": "2"},
        )
    )
