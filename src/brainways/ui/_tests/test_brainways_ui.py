import shutil
from dataclasses import replace
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo, SliceSelection
from brainways.ui.brainways_ui import BrainwaysUI
from brainways.ui.controllers.base import Controller
from brainways.ui.utils.test_utils import randomly_modified_params
from brainways.utils.io_utils import ImagePath


@fixture(params=[0, 1])
def image_index(request):
    return request.param


@fixture(params=[0, 1])
def subject_index(request):
    return request.param


STEP_INDICES = [0, 1, 2, 3, 4, 5, 6]


@fixture(params=STEP_INDICES)
def step_index(opened_app: BrainwaysUI, request) -> int:
    assert len(STEP_INDICES) == len(opened_app.steps)
    return request.param


@fixture
def step(opened_app: BrainwaysUI, step_index: int):
    return opened_app.steps[step_index]


@fixture
def subject_doc() -> SliceInfo:
    return SliceInfo(
        path=ImagePath("/"),
        image_size=(3840, 5120),
        lowres_image_size=(384, 512),
        params=BrainwaysParams(),
    )


def test_app_init(app: BrainwaysUI):
    pass


def test_steps_are_loading(opened_app: BrainwaysUI, step: Controller):
    opened_app.set_step_index_async(opened_app.steps.index(step))


def test_steps_are_loading_twice(opened_app: BrainwaysUI, step: Controller):
    opened_app.set_step_index_async(opened_app.steps.index(step))
    opened_app.set_step_index_async(
        (opened_app.steps.index(step) + 1) % len(opened_app.steps)
    )
    opened_app.set_step_index_async(opened_app.steps.index(step))


def test_steps_are_loading_for_new_subject(opened_app: BrainwaysUI):
    opened_app.current_document = replace(
        opened_app.current_document, params=BrainwaysParams()
    )
    opened_app.set_step_index_async(0, force=True, save_subject=False)
    for step_index in range(len(opened_app.steps)):
        opened_app.set_step_index_async(step_index)


def test_next_image_prev_image_keeps_changed_params(
    opened_app: BrainwaysUI, step: Controller
):
    # set step
    opened_app.set_step_index_async(opened_app.steps.index(step))

    # modify params
    current_params = opened_app.current_params
    first_modification = randomly_modified_params(current_params)
    assert current_params != first_modification
    step.show(first_modification)

    # go next image
    opened_app.next_image()

    # modify params again
    second_modification = randomly_modified_params(opened_app.current_params)
    step.show(second_modification)

    # go prev image
    opened_app.prev_image()

    # assert that params of first image didn't change
    opened_app.persist_current_params()
    assert opened_app.current_params == first_modification


@pytest.mark.skip
def test_run_workflow(opened_app: BrainwaysUI):
    opened_app.run_workflow_async()


def test_open_project(
    app: BrainwaysUI,
    mock_project: BrainwaysProject,
):
    assert app._project is None
    app.open_project_async(mock_project.path)
    assert isinstance(app.project, BrainwaysProject)
    assert isinstance(app.current_subject, BrainwaysSubject)


def test_open_project_without_subjects(
    app: BrainwaysUI, mock_project: BrainwaysProject
):
    for subject_dir in mock_project.path.parent.glob("subject*"):
        shutil.rmtree(subject_dir)
    assert app._project is None
    app.open_project_async(mock_project.path)
    assert isinstance(app.project, BrainwaysProject)
    assert app._current_valid_subject_index is None


def test_open_project_filters_subjects_without_documents(
    app: BrainwaysUI, mock_project: BrainwaysProject
):
    mock_project.subjects[0].documents = []
    mock_project.subjects[0].save()
    app.open_project_async(mock_project.path)
    assert len(app.project.subjects) == 1


def test_set_subject_index_async(
    opened_app: BrainwaysUI,
    subject_index: int,
):
    opened_app.set_subject_index_async(subject_index)
    assert opened_app.current_subject == opened_app.project.subjects[subject_index]


@pytest.mark.skip
def test_save_load_subject(
    opened_app: BrainwaysUI,
    step_index: int,
    image_index: int,
    tmpdir,
):
    opened_app.set_step_index_async(step_index)
    opened_app.set_document_index_async(image_index)
    save_path = Path(tmpdir) / "test"
    docs = opened_app.current_subject.documents
    opened_app.save_subject()
    opened_app.current_subject.documents = []
    opened_app.open_subject_async(save_path)
    assert opened_app.current_subject.documents == docs


@pytest.mark.skip
def test_save_after_run_workflow(
    opened_app: BrainwaysUI,
    tmpdir,
):
    opened_app.run_workflow_async()
    save_path = Path(tmpdir) / "test"
    docs = opened_app.current_subject.documents
    opened_app.save_subject()
    opened_app.all_documents = []
    opened_app.open_subject_async(save_path)
    assert opened_app.current_subject.documents == docs


@fixture
def app_batch_run_model(
    opened_app: BrainwaysUI,
    step: Controller,
    step_index: int,
    image_index: int,
) -> Tuple[BrainwaysUI, BrainwaysParams]:
    opened_app.set_step_index_async(step_index)
    opened_app.set_document_index_async(image_index)
    modified_params = randomly_modified_params(opened_app.current_params)
    step.run_model = Mock(return_value=modified_params)
    opened_app.batch_run_model_async()
    return opened_app, modified_params


@pytest.mark.skip
def test_batch_run_model_works(
    app_batch_run_model: Tuple[BrainwaysUI, BrainwaysParams], step: Controller
):
    app, modified_params = app_batch_run_model
    for doc in app.documents:
        assert doc.params == modified_params


@pytest.mark.skip
def test_batch_run_model_ends_with_last_image(
    app_batch_run_model: Tuple[BrainwaysUI, BrainwaysParams],
):
    app, modified_params = app_batch_run_model
    assert app._current_valid_document_index == len(app.documents) - 1


@pytest.mark.skip
def test_export_cells_to_csv(opened_app: BrainwaysUI, tmpdir):
    cells = np.array([[0, 0, 0], [1, 1, 0]])
    opened_app.all_documents = [
        SliceInfo(
            path=ImagePath("/a"),
            image_size=(10, 10),
            lowres_image_size=(10, 10),
            params=BrainwaysParams(),
            region_areas={0: 1},
            cells=cells,
        ),
        SliceInfo(
            path=ImagePath("/b"),
            image_size=(10, 10),
            lowres_image_size=(10, 10),
            params=BrainwaysParams(),
            region_areas={0: 1},
            cells=cells,
        ),
    ]

    cells_path = Path(tmpdir) / "cells.csv"
    opened_app.export_cells(cells_path)
    df = pd.read_csv(cells_path)
    assert df.shape == (2, 2)


def test_autosave_on_set_image_index(opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    opened_app.set_document_index_async(image_index=1)
    opened_app.save_subject.assert_called_once()


def test_autosave_on_set_step_index(opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    opened_app.set_step_index_async(step_index=1)
    opened_app.save_subject.assert_called_once()


@pytest.mark.skip
def test_autosave_on_close(opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    opened_app.viewer.close()
    opened_app.save_subject.assert_called_once()


@pytest.mark.skip
def test_import_cells(opened_app: BrainwaysUI, tmpdir):
    # for document in opened_app.documents:
    #     assert document.cells is None

    cells = np.random.rand(len(opened_app.documents), 3, 2)

    # create cells csvs
    root = Path(tmpdir)
    for i, document in enumerate(opened_app.documents):
        csv_filename = (
            f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
        )
        df = pd.DataFrame({"centroid-0": cells[i, :, 0], "centroid-1": cells[i, :, 1]})
        df.to_csv(root / csv_filename)
    opened_app.import_cells_async(root)

    for i, document in enumerate(opened_app.documents):
        assert np.allclose(document.cells, cells[i])


@pytest.fixture
def brainways_ui():
    viewer = MagicMock()
    return BrainwaysUI(viewer)


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=True)
def test_no_subjects(mock_show_warning_dialog, brainways_ui):
    brainways_ui._project = MagicMock(subjects=[])
    assert brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_not_called()


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=True)
def test_no_valid_documents(mock_show_warning_dialog, brainways_ui):
    subject = MagicMock(valid_documents=[])
    brainways_ui._project = MagicMock(subjects=[subject])
    assert brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_not_called()


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=True)
def test_all_params_present(mock_show_warning_dialog, brainways_ui):
    params = BrainwaysParams(atlas="atlas", affine="affine", tps="tps", cell="cell")
    slice_info = MagicMock(params=params)
    subject = MagicMock(valid_documents=[(0, slice_info)])
    brainways_ui._project = MagicMock(subjects=[subject])
    assert brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_not_called()


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=True)
def test_some_params_missing(mock_show_warning_dialog, brainways_ui):
    params = BrainwaysParams(atlas=None, affine="affine", tps="tps", cell="cell")
    slice_info = MagicMock(params=params)
    subject = MagicMock(valid_documents=[(0, slice_info)])
    brainways_ui._project = MagicMock(subjects=[subject])
    assert brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_called_once()


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=True)
def test_all_params_missing(mock_show_warning_dialog, brainways_ui):
    params = BrainwaysParams(atlas=None, affine=None, tps=None, cell=None)
    slice_info = MagicMock(params=params)
    subject = MagicMock(valid_documents=[(0, slice_info)])
    brainways_ui._project = MagicMock(subjects=[subject])
    assert brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_called_once()


@patch("brainways.ui.brainways_ui.show_warning_dialog", return_value=False)
def test_all_params_missing_user_rejects(mock_show_warning_dialog, brainways_ui):
    params = BrainwaysParams(atlas=None, affine=None, tps=None, cell=None)
    slice_info = MagicMock(params=params)
    subject = MagicMock(valid_documents=[(0, slice_info)])
    brainways_ui._project = MagicMock(subjects=[subject])
    assert not brainways_ui.prompt_user_slices_have_missing_params()
    mock_show_warning_dialog.assert_called_once()


def test_get_slice_selection_current_slice(opened_app: BrainwaysUI):
    slice_selection = SliceSelection.CURRENT_SLICE
    expected_slice_infos = [opened_app.current_document]

    slice_infos = opened_app.get_slice_selection(slice_selection)

    assert slice_infos == expected_slice_infos


def test_get_slice_selection_current_subject(opened_app: BrainwaysUI):
    slice_selection = SliceSelection.CURRENT_SUBJECT
    expected_slice_infos = [
        slice_info for _, slice_info in opened_app.current_subject.valid_documents
    ]

    slice_infos = opened_app.get_slice_selection(slice_selection)

    assert slice_infos == expected_slice_infos


def test_get_slice_selection_all_subjects(opened_app: BrainwaysUI):
    slice_selection = SliceSelection.ALL_SUBJECTS
    expected_slice_infos = [
        slice_info
        for subject in opened_app.project.subjects
        for _, slice_info in subject.valid_documents
    ]

    slice_infos = opened_app.get_slice_selection(slice_selection)

    assert slice_infos == expected_slice_infos
