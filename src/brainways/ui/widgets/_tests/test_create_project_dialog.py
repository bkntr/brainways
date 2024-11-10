from dataclasses import replace

from aicsimageio.types import PhysicalPixelSizes
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QCheckBox

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import SliceInfo
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
    create_subject_dialog.conditions_widget[0].value = "mod1"
    create_subject_dialog.conditions_widget[1].value = "mod2"
    assert create_subject_dialog.subject.subject_info.conditions["condition1"] == "mod1"
    assert create_subject_dialog.subject.subject_info.conditions["condition2"] == "mod2"
