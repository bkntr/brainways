from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import napari
import numpy as np
from napari.layers import Image
from napari.qt.threading import FunctionWorker, create_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QProgressDialog, QVBoxLayout, QWidget

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo, SliceSelection
from brainways.ui.controllers.affine_2d_controller import Affine2DController
from brainways.ui.controllers.analysis_controller import AnalysisController
from brainways.ui.controllers.annotation_viewer_controller import (
    AnnotationViewerController,
)
from brainways.ui.controllers.base import Controller
from brainways.ui.controllers.cell_3d_viewer_controller import Cell3DViewerController
from brainways.ui.controllers.cell_detector_controller import CellDetectorController
from brainways.ui.controllers.registration_controller import RegistrationController
from brainways.ui.controllers.tps_controller import TpsController
from brainways.ui.utils.async_utils import do_work_async
from brainways.ui.widgets.warning_dialog import show_warning_dialog
from brainways.ui.widgets.workflow_widget import WorkflowView
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.paths import get_brainways_dir
from brainways.utils.setup import BrainwaysSetup


class BrainwaysUI(QWidget):
    progress = Signal(object)

    def __init__(self, napari_viewer: napari.Viewer, async_disabled: bool = False):
        super().__init__()

        self.viewer = napari_viewer
        self.async_disabled = async_disabled

        self.registration_controller = RegistrationController(self)
        self.affine_2d_controller = Affine2DController(self)
        self.tps_controller = TpsController(self)
        self.annotation_viewer_controller = AnnotationViewerController(self)
        self.cell_detector_controller = CellDetectorController(self)
        self.cell_viewer_controller = Cell3DViewerController(self)
        self.analysis_controller = AnalysisController(self)

        self.steps = [
            self.registration_controller,
            self.affine_2d_controller,
            self.tps_controller,
            self.annotation_viewer_controller,
            self.cell_detector_controller,
            self.cell_viewer_controller,
            self.analysis_controller,
        ]

        self._project: Optional[BrainwaysProject] = None
        self._current_valid_subject_index: Optional[int] = None
        self._current_valid_document_index: Optional[int] = None
        self._current_step_index: int = 0

        self._auto_contrast = True
        self._layer_contrast_limits: Dict[str, Tuple[float, float]] = {}

        self.widget = WorkflowView(self, steps=self.steps)

        self._set_layout()
        get_brainways_dir()  # TODO: remove after brainways 0.10.1
        self._setup_async()

        self.viewer.layers.events.inserted.connect(
            self._on_layer_inserted, position="last"
        )

    def _setup_async(self):
        if not BrainwaysSetup.is_first_launch():
            return

        progress_dialog = QProgressDialog("First time setup...", "Cancel", 0, 0, self)
        progress_dialog.setModal(True)
        progress_dialog.setWindowTitle("First time setup...")
        progress_dialog.setCancelButton(None)
        # progress_dialog.setWindowFlag(Qt.WindowType.CustomizeWindowHint)
        # progress_dialog.setWindowFlag(~Qt.WindowType.WindowCloseButtonHint)
        progress_dialog.show()

        def _progress_callback(desc: str):
            progress_dialog.setLabelText(desc)

        self.progress.connect(_progress_callback)

        def _return_callback():
            progress_dialog.close()
            self.progress.disconnect(_progress_callback)

        setup = BrainwaysSetup(
            atlas_names=["whs_sd_rat_39um", "allen_mouse_25um"],
            progress_callback=lambda desc: self.progress.emit(desc),
        )
        self.do_work_async(setup.run, return_callback=_return_callback)

    def _on_layer_inserted(self, event):
        layer = event.value
        if layer is None or "__brainways__" not in layer.metadata:
            return
        sample_project_path = Path(layer.metadata["sample_project_path"])
        self.open_project_async(sample_project_path)

    def _set_layout(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(500)

    def _register_keybinds(self):
        self.viewer.bind_key("PageDown", self.next_step, overwrite=True)
        self.viewer.bind_key("PageUp", self.prev_step, overwrite=True)
        self.viewer.bind_key("n", self.next_image, overwrite=True)
        self.viewer.bind_key("b", self.prev_image, overwrite=True)
        self.viewer.bind_key("Shift-N", self.next_subject, overwrite=True)
        self.viewer.bind_key("Shift-B", self.prev_subject, overwrite=True)
        self.viewer.bind_key(
            "Home",
            lambda _: self.set_document_index_async(image_index=0),
            overwrite=True,
        )
        self.viewer.bind_key(
            "End",
            lambda _: self.set_document_index_async(
                image_index=len(self.current_subject.valid_documents) - 1
            ),
            overwrite=True,
        )

    def reset(self, save_current_subject: bool = True):
        if self._current_valid_subject_index is not None:
            if save_current_subject:
                self.save_subject()
            self.current_step.close()
            self.widget.set_step(0)
            self.widget.set_image_index(1)
        self.widget.set_subject_index(1)
        self._current_valid_subject_index = None
        self._current_valid_document_index = 0
        self._current_step_index = 0

    def _run_workflow_single_doc(self, doc_i: int) -> None:
        raise NotImplementedError()
        # reader = brainways.utils.io_utils.readers.get_reader(self.documents[doc_i].path)
        # transform = self.current_subject.pipeline.get_image_to_atlas_transform(doc_i, reader)
        # cell_detector_result = None
        # document = self.documents[doc_i]
        #
        # for step in self.steps:
        #     cell_detector_result = step.cells(
        #         reader=reader,
        #         params=document.params,
        #         prev_cell_detector_result=cell_detector_result,
        #     )
        #
        # cells_on_atlas = transform.transform_points(cell_detector_result.cells)
        # self.set_document(replace(document, cells=cells_on_atlas), doc_i)

    def run_workflow_async(self) -> FunctionWorker:
        raise NotImplementedError()
        # self.set_step_index_async(len(self.steps) - 1)
        # view_images = ViewImages(atlas=self._atlas)
        # self.cell_viewer_controller.open(self._atlas)
        # self.widget.show_progress_bar(len(self.documents))
        # worker = create_worker(self._run_workflow)
        # worker.yielded.connect(self._on_run_workflow_yielded)
        # worker.returned.connect(self._on_run_workflow_returned)
        # worker.errored.connect(self._on_work_error)
        # worker.start()
        # return worker

    def _run_workflow(self):
        raise NotImplementedError()
        # for step in self.steps:
        #     step.load_model()
        #
        # for doc_i, doc in enumerate(self.documents):
        #     self._run_workflow_single_doc(doc_i)
        #     yield doc_i

    def _on_run_workflow_yielded(self, doc_index: int):
        raise NotImplementedError()
        # cells = np.concatenate(
        #     [doc.cells for doc in self.documents if doc.cells is not None]
        # )
        # self.cell_viewer_controller.show_cells(cells)
        # self.widget.update_progress_bar(doc_index)

    def _on_run_workflow_returned(self):
        raise NotImplementedError()
        # self.widget.hide_progress_bar()

    def open_project_async(self, path: Path) -> FunctionWorker:
        self.reset()
        return self.do_work_async(
            self._open_project,
            return_callback=self._on_project_opened,
            progress_label="Opening project...",
            path=path,
        )

    def _open_project(self, path: Path):
        yield "Opening project..."
        self._project = BrainwaysProject.open(path, lazy_init=True)
        # subjects with no valid documents are not supported in GUI
        self.project.subjects = [
            subject
            for subject in self.project.subjects
            if len(subject.valid_documents) > 0
        ]
        yield f"Loading '{self.project.settings.atlas}' atlas..."
        self.project.load_atlas()
        yield "Loading Brainways Pipeline models..."
        self.project.load_pipeline()
        if len(self.project.subjects) > 0:
            self._current_valid_subject_index = 0
            yield "Opening image..."
            self._open_image()

    def _open_image(self):
        self._image = self.current_subject.read_lowres_image(self.current_document)
        self._load_step_default_params()

    def _load_step_default_params(self):
        if not self.current_step.has_current_step_params(self.current_params):
            self.current_params = self.current_step.default_params(
                image=self._image, params=self.current_params
            )

    def _open_step(self):
        self.current_step.open()
        self.current_step.show(self.current_params, self._image)
        self.widget.update_enabled_steps()
        self._set_title()

    def _set_title(self, valid_document_index: Optional[int] = None):
        if valid_document_index is None:
            valid_document_index = self._current_valid_document_index
        _, document = self.current_subject.valid_documents[valid_document_index]
        self.viewer.title = (
            f"{self.current_subject.subject_info.name} - {document.path}"
        )

    def _on_project_opened(self):
        self.widget.on_project_changed(len(self.project.subjects))
        if len(self.project.subjects) > 0:
            self._on_subject_opened()

    def _on_subject_opened(self):
        self._set_title()
        self._register_keybinds()
        self.widget.on_subject_changed()
        for step in self.steps:
            step.pipeline_loaded()
        self.current_step.open()
        self.current_step.show(self.current_params, self._image)
        self.widget.update_enabled_steps()

    def _on_progress_returned(self):
        self.widget.hide_progress_bar()

    def persist_current_params(self):
        assert self.current_step.params is not None
        self.current_document = replace(
            self.current_document, params=self.current_step.params
        )

    def save_subject(self, persist: bool = True) -> None:
        if persist:
            self.persist_current_params()
        self.current_subject.save()
        self.project.save()

    def set_subject_index_async(
        self,
        subject_index: int,
        force: bool = False,
        save_current_subject: bool = True,
    ) -> FunctionWorker | None:
        if not force and self._current_valid_subject_index == subject_index:
            return None
        self.reset(save_current_subject=save_current_subject)
        subject_index = min(max(subject_index, 0), len(self.project.subjects) - 1)

        self._current_valid_subject_index = subject_index
        self.widget.set_subject_index(subject_index + 1)
        return self.do_work_async(
            self._open_image, return_callback=self._on_subject_opened
        )

    def set_document_index_async(
        self,
        image_index: int,
        force: bool = False,
        persist_current_params: bool = True,
    ) -> FunctionWorker | None:
        if persist_current_params:
            self.save_subject()

        image_index = min(
            max(image_index, 0), len(self.current_subject.valid_documents) - 1
        )
        if not force and self._current_valid_document_index == image_index:
            return None

        self._current_valid_document_index = image_index

        if not self.current_step.enabled(self.current_params):
            self.current_step.close()
            for step_index in reversed(range(self._current_step_index)):
                if self.current_step.enabled(self.current_params):
                    break
                self._current_step_index = step_index
            self.widget.set_step(self._current_step_index)

        self.widget.set_image_index(image_index + 1)
        return self.do_work_async(self._open_image, return_callback=self._open_step)

    def prev_subject(self, _=None) -> FunctionWorker | None:
        return self.set_subject_index_async(
            max(self._current_valid_subject_index - 1, 0)
        )

    def next_subject(self, _=None) -> FunctionWorker | None:
        return self.set_subject_index_async(
            min(
                self._current_valid_subject_index + 1,
                len(self.project.subjects) - 1,
            )
        )

    def prev_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            max(self._current_valid_document_index - 1, 0)
        )

    def next_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            min(
                self._current_valid_document_index + 1,
                len(self.current_subject.valid_documents) - 1,
            )
        )

    def set_step_index_async(
        self,
        step_index: int,
        force: bool = False,
        save_subject: bool = True,
    ) -> FunctionWorker | None:
        if not force and self._current_step_index == step_index:
            return
        if save_subject:
            self.save_subject()
        self.current_step.close()
        self.widget.set_step(step_index)
        self._current_step_index = step_index
        return self.do_work_async(
            self._load_step_default_params, return_callback=self._open_step
        )

    def prev_step(self, _=None) -> FunctionWorker | None:
        return self.set_step_index_async(max(self._current_step_index - 1, 0))

    def next_step(self, _=None) -> FunctionWorker | None:
        return self.set_step_index_async(
            min(self._current_step_index + 1, len(self.steps) - 1)
        )

    def _batch_run_model(self):
        self.widget.show_progress_bar()
        for valid_index in range(len(self.current_subject.valid_documents)):
            self._current_valid_document_index = valid_index
            self._open_image()
            self.current_params = self.current_step.run_model(
                self._image, self.current_params
            )
            self.save_subject(persist=False)
            yield valid_index, self.current_params, self._image

    def _batch_run_model_yielded(
        self, args: Tuple[int, BrainwaysParams, np.ndarray]
    ) -> None:
        valid_index, params, image = args
        self.current_step.show(params, image)
        self.widget.set_image_index(valid_index + 1)
        self.widget.update_progress_bar(valid_index + 1)
        self._set_title(valid_document_index=valid_index)

    def batch_run_model_async(self) -> FunctionWorker:
        self.widget.show_progress_bar(
            max_value=len(self.current_subject.valid_documents)
        )
        worker = create_worker(self._batch_run_model)
        worker.yielded.connect(self._batch_run_model_yielded)
        worker.returned.connect(self._progress_returned)
        worker.start()
        return worker

    def import_cell_detections_async(
        self, path: Path, importer: CellDetectionImporter
    ) -> FunctionWorker:
        return self.do_work_async(
            self.project.import_cell_detections_iter,
            importer=importer,
            cell_detections_root=path,
            progress_label="Importing Cell Detections...",
            progress_max_value=self.project.n_valid_images,
        )

    def view_brain_structure_async(
        self,
        structure_names: List[str],
        condition_type: Optional[str] = None,
        condition_value: Optional[str] = None,
        num_subjects: Optional[int] = None,
        display_channel: Optional[int] = None,
        filter_cell_type: Optional[str] = None,
    ) -> FunctionWorker:
        # return self.do_work_async(
        #     self.project.view_brain_structure,
        #     progress_label="Viewing Brain Structure...",
        #     structure_names=structure_names,
        #     condition_type=condition_type,
        #     condition_value=condition_value,
        #     num_subjects=num_subjects,
        # )
        self.project.view_brain_structure(
            structure_names=structure_names,
            condition_type=condition_type,
            condition_value=condition_value,
            num_subjects=num_subjects,
            display_channel=display_channel,
            filter_cell_type=filter_cell_type,
        )

    def show_cells_view(self):
        self.save_subject()
        self.current_step.close()
        self.cell_viewer_controller.open(self.current_subject.atlas)
        cells = np.concatenate(
            [
                doc.cells
                for i, doc in self.current_subject.valid_documents
                if doc.cells is not None
            ]
        )
        self.cell_viewer_controller.show_cells(cells)

    def _progress_returned(self) -> None:
        self.widget.hide_progress_bar()

    def do_work_async(
        self,
        function: Callable,
        return_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
        yield_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
        error_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
        progress_label: Optional[str] = None,
        progress_max_value: int = 0,
        **kwargs,
    ) -> FunctionWorker:
        self.widget.show_progress_bar(
            label=progress_label, max_value=progress_max_value
        )
        return do_work_async(
            function,
            return_callback=self._merge_callbacks(
                self._on_work_returned, return_callback
            ),
            yield_callback=self._merge_callbacks(self._on_work_yielded, yield_callback),
            error_callback=self._merge_callbacks(self._on_work_error, error_callback),
            async_disabled=self.async_disabled,
            **kwargs,
        )

    def prompt_user_slices_have_missing_params(self, check_cells: bool = False) -> bool:
        missing_param_warnings = []
        missing_cell_detction_warnings = []
        for subject_idx, subject in enumerate(self.project.subjects):
            for slice_idx, slice_info in subject.valid_documents:
                missing_params = []
                params = slice_info.params
                if params.atlas is None:
                    missing_params.append("Atlas Registration")
                if params.affine is None:
                    missing_params.append("Rigid Registration")
                if params.tps is None:
                    missing_params.append("Non-rigid Registration")
                if missing_params:
                    missing_param_warnings.append(
                        f"Subject #{subject_idx + 1}, Slice #{slice_idx + 1}: {missing_params}"
                    )
                if (
                    check_cells
                    and not subject.cell_detections_path(slice_info.path).exists()
                ):
                    missing_cell_detction_warnings.append(
                        f"Subject #{subject_idx + 1}, Slice #{slice_idx + 1}"
                    )

        warning_text = []
        if missing_param_warnings:
            warning_text += (
                [
                    "The following slices have missing parameters:",
                ]
                + [f"  {warning}" for warning in missing_param_warnings]
                + [""]
            )
        if missing_cell_detction_warnings:
            warning_text += (
                [
                    "The following slices have missing cell detections:",
                ]
                + [f"  {warning}" for warning in missing_cell_detction_warnings]
                + [""]
            )

        if warning_text:
            warning_text += [
                "These issues may lead to incorrect results. Do you want to continue?",
            ]
            return show_warning_dialog("\n".join(warning_text))
        return True

    def update_layer_contrast_limits(
        self,
        layer: Image,
        contrast_limits_quantiles: Tuple[float, float] = (0.01, 0.98),
        contrast_limits_range_quantiles: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if layer.data.dtype == np.bool_:
            return

        nonzero_mask = layer.data > 0
        if (~nonzero_mask).all():
            return

        limit_0, limit_1, limit_range_0, limit_range_1 = np.quantile(
            layer.data[nonzero_mask],
            (*contrast_limits_quantiles, *contrast_limits_range_quantiles),
        )
        with layer.events.contrast_limits.blocker():
            layer.contrast_limits_range = (limit_range_0, limit_range_1 + 1e-8)

        if self._auto_contrast:
            layer.contrast_limits = (limit_0, limit_1 + 1e-8)
        else:
            if layer.name not in self._layer_contrast_limits:
                self._layer_contrast_limits[layer.name] = layer.contrast_limits
            layer.contrast_limits = self._layer_contrast_limits[layer.name]

    def set_contrast_limits(self, event):
        layer = event.source
        self._layer_contrast_limits[layer.name] = layer.contrast_limits

    def set_auto_contrast(self, value: bool):
        self._auto_contrast = bool(value)

    @staticmethod
    def _merge_callbacks(
        callback: Callable,
        callback_or_callbacks: Optional[Union[Callable, Sequence[Callable]]],
    ):
        if callback_or_callbacks is None:
            return callback
        elif isinstance(callback_or_callbacks, Sequence):
            return [callback] + callback_or_callbacks
        else:
            return [callback, callback_or_callbacks]

    def _on_work_returned(self, *args, **kwargs):
        self.widget.hide_progress_bar()

    def _on_work_yielded(
        self, text: Optional[str] = None, value: Optional[int] = None, **kwargs
    ):
        self.widget.update_progress_bar(value=value, text=text)

    def _on_work_error(self, *args, **kwargs):
        self.widget.hide_progress_bar()

    def get_slice_selection(self, slice_selection: SliceSelection) -> List[SliceInfo]:
        if slice_selection == SliceSelection.CURRENT_SLICE:
            return [self.current_document]
        elif slice_selection == SliceSelection.CURRENT_SUBJECT:
            return [
                slice_info for _, slice_info in self.current_subject.valid_documents
            ]
        else:
            return [
                slice_info
                for subject in self.project.subjects
                for _, slice_info in subject.valid_documents
            ]

    @property
    def _current_document_index(self):
        current_document_index, _ = self.current_subject.valid_documents[
            self._current_valid_document_index
        ]
        return current_document_index

    @property
    def current_valid_document_index(self):
        return self._current_valid_document_index

    @property
    def current_subject(self) -> BrainwaysSubject:
        return self.project.subjects[self._current_valid_subject_index]

    @current_subject.setter
    def current_subject(self, value: BrainwaysSubject):
        assert self._current_valid_subject_index is not None
        self.project.subjects[self._current_valid_subject_index] = value

    @property
    def current_document(self) -> SliceInfo:
        return self.current_subject.documents[self._current_document_index]

    @current_document.setter
    def current_document(self, value: SliceInfo):
        self.current_subject.documents[self._current_document_index] = value

    @property
    def current_step(self) -> Controller:
        return self.steps[self._current_step_index]

    @property
    def current_params(self):
        return self.current_subject.documents[self._current_document_index].params

    @current_params.setter
    def current_params(self, value: BrainwaysParams):
        self.current_document = replace(self.current_document, params=value)

    @property
    def current_subject_index(self):
        return self._current_valid_subject_index

    @property
    def subject_size(self):
        return len(self.current_subject.valid_documents)

    @property
    def project(self) -> BrainwaysProject:
        if self._project is None:
            raise ValueError("Project not opened")
        return self._project
