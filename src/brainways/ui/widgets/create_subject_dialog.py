import functools
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from magicgui import widgets
from qtpy import QtCore
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo, SubjectInfo
from brainways.ui.utils.async_utils import do_work_async
from brainways.utils.image import resize_image
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import get_channels, get_scenes


class CreateSubjectDialog(QDialog):
    def __init__(
        self,
        project: BrainwaysProject,
        async_disabled: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.project = project
        self.async_disabled = async_disabled
        self.subject: Optional[BrainwaysSubject] = None
        self.create_subject_button = QPushButton("&Create", self)
        self.create_subject_button.clicked.connect(self.on_create_subject_clicked)
        self.registration_channel_combobox = QComboBox()
        self.cell_detection_channels_checkboxes: List[QCheckBox] = []
        self.add_images_button = QPushButton("&Add Image(s)...", self)
        self.add_images_button.clicked.connect(self.on_add_images_clicked)
        self.files_table = self.create_table()
        self.conditions_widget = self._create_conditions_widget()
        self.bottom_label = QLabel("")

        self._layout = QGridLayout(self)
        self.setLayout(self._layout)

        cur_row = 0
        self._layout.addWidget(QLabel("Channel:"), cur_row, 0)
        self._layout.addWidget(self.registration_channel_combobox, cur_row, 1)
        cur_row += 1  # leave room for cell detection channel checkboxes

        if self.project.settings.condition_names:
            cur_row += 1
            self._layout.addWidget(self.conditions_widget.native, cur_row, 0, 1, 3)

        cur_row += 1
        self._layout.addWidget(self.files_table, cur_row, 0, 1, 3)

        cur_row += 1
        self._layout.addWidget(self.bottom_label, cur_row, 0, 1, 2)

        cur_row += 1
        self._layout.addWidget(
            self.add_images_button, cur_row, 1, alignment=QtCore.Qt.AlignRight  # type: ignore
        )

        self._layout.addWidget(self.create_subject_button, cur_row, 2)

        if parent is not None:
            window = self.window()
            assert window is not None
            self.resize(
                int(window.width() * 0.8),
                int(window.height() * 0.8),
            )

    def edit_subject_async(self, subject_index: int, document_index: int) -> None:
        self.setWindowTitle("Edit Subject")
        self._set_subject(self.project.subjects[subject_index])
        assert self.subject is not None
        self.create_subject_button.setText("Done")
        self.add_document_rows_async(
            documents=self.subject.documents, select_document_index=document_index
        )

    def new_subject(self, subject_id: str, conditions: Dict[str, str]):
        assert subject_id is not None
        self.setWindowTitle(f"New Subject ({subject_id})")
        if self.project.subjects:
            last_subject_info = self.project.subjects[-1].subject_info
            default_registration_channel = last_subject_info.registration_channel
            default_cell_detection_channels = last_subject_info.cell_detection_channels
        else:
            default_registration_channel = 0
            default_cell_detection_channels = [0]
        self._set_subject(
            self.project.add_subject(
                SubjectInfo(
                    name=subject_id,
                    registration_channel=default_registration_channel,
                    cell_detection_channels=default_cell_detection_channels,
                    conditions=conditions,
                )
            )
        )

    def _set_subject(self, subject: BrainwaysSubject):
        self.subject = subject
        for index, condition in enumerate(self.project.settings.condition_names):
            self.conditions_widget[index].value = subject.subject_info.conditions.get(
                condition, ""
            )
            self.conditions_widget[index].changed.disconnect()
            self.conditions_widget[index].changed.connect(
                self._get_set_condition_callback(subject=subject, condition=condition)
            )

    def _get_set_condition_callback(self, subject: BrainwaysSubject, condition: str):
        def __set_condition(value: str):
            subject.subject_info.conditions[condition] = value

        return __set_condition

    def create_table(self) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setHorizontalHeaderLabels(["", "Thumbnail", "Path", "Scene"])
        horizontal_header = table.horizontalHeader()
        assert horizontal_header is not None
        vertical_header = table.verticalHeader()
        assert vertical_header is not None
        horizontal_header.setSectionResizeMode(2, QHeaderView.Stretch)
        vertical_header.hide()
        table.setShowGrid(False)
        return table

    def _create_conditions_widget(self) -> widgets.Container:
        condition_widgets = [
            widgets.LineEdit(label=condition)
            for condition in self.project.settings.condition_names
        ]
        return widgets.Container(widgets=condition_widgets)

    def add_filenames_async(self, filenames: List[str]) -> None:
        progress = QProgressDialog(
            "Loading image scenes...", "Cancel", 0, len(filenames), self
        )
        progress.setModal(True)
        progress.setValue(0)
        progress.setWindowTitle("Loading...")
        progress.show()

        def on_work_returned(documents: List[SliceInfo]):
            progress.close()
            self.add_document_rows_async(documents)

        def on_work_yielded():
            progress.setValue(progress.value() + 1)

        def work():
            documents = []
            for filename in filenames:
                if progress.wasCanceled():
                    # remove added documents if operation is cancelled
                    for document in documents:
                        self.subject.documents.remove(document)
                    return []
                for scene in range(len(get_scenes(filename))):
                    documents.append(
                        self.subject.add_image(
                            ImagePath(filename=filename, scene=scene),
                            load_thumbnail=False,
                        )
                    )
                yield
            return documents

        do_work_async(
            work,
            return_callback=on_work_returned,
            yield_callback=on_work_yielded,
            error_callback=on_work_returned,
            async_disabled=self.async_disabled,
        )

    def get_image_widget(self, thumbnail: np.ndarray) -> QWidget:
        image_widget = QLabel()
        image_widget.setPixmap(
            QPixmap(
                QImage(
                    thumbnail.data,
                    thumbnail.shape[1],
                    thumbnail.shape[0],
                    thumbnail.shape[1] * 3,
                    QImage.Format_RGB888,
                )
            )
        )
        return image_widget

    def get_thumbnail_image(self, document: SliceInfo) -> np.ndarray:
        assert self.subject is not None
        thumbnail = self.subject.read_lowres_image(document)
        thumbnail = resize_image(thumbnail, size=(256, 256), keep_aspect=True)
        thumbnail = np.tile(thumbnail[..., None], [1, 1, 3]).astype(np.float32)
        if document.ignore:
            thumbnail[..., [1, 2]] *= 0.3
        else:
            thumbnail[..., [0, 2]] *= 0.3
        thumbnail = thumbnail.astype(np.uint8)
        return thumbnail

    def add_document_rows_async(
        self, documents: List[SliceInfo], select_document_index: Optional[int] = None
    ) -> None:
        assert self.subject is not None
        progress = QProgressDialog(
            "Opening images...", "Cancel", 0, len(documents), self
        )
        progress.setModal(True)
        progress.setValue(0)
        progress.setWindowTitle("Loading...")
        progress.show()

        def on_work_returned():
            self.registration_channel_combobox.setCurrentIndex(
                self.subject.subject_info.registration_channel
            )
            for channel_index, checkbox in enumerate(
                self.cell_detection_channels_checkboxes
            ):
                checkbox.setChecked(
                    channel_index in self.subject.subject_info.cell_detection_channels
                )
            if select_document_index is not None:
                self.files_table.selectRow(select_document_index)
            progress.close()

        def on_work_yielded(result: Tuple[SliceInfo, np.ndarray]):
            document, thumbnail = result
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)
            checkbox = QCheckBox()
            checkbox.setChecked(not document.ignore)
            checkbox.stateChanged.connect(
                functools.partial(
                    self.on_check_changed, checkbox=checkbox, document_index=row
                )
            )
            self.files_table.setCellWidget(row, 0, checkbox)
            self.files_table.setCellWidget(row, 1, self.get_image_widget(thumbnail))
            self.files_table.setItem(
                row, 2, QTableWidgetItem(str(document.path.filename))
            )
            self.files_table.setItem(row, 3, QTableWidgetItem(str(document.path.scene)))
            self.files_table.resizeRowToContents(row)
            self.files_table.resizeColumnsToContents()

            channels = get_channels(document.path.filename)
            if self.registration_channel_combobox.count() == 0:
                self.registration_channel_combobox.addItems(channels)
                self.registration_channel_combobox.currentIndexChanged.connect(
                    self.on_selected_channel_changed
                )
                self._add_cell_detection_channels_checkboxes(channels)
            else:
                current_channels = [
                    self.registration_channel_combobox.itemText(i)
                    for i in range(self.registration_channel_combobox.count())
                ]
                if channels != current_channels:
                    raise ValueError(
                        f"Channels for {document.path.filename} ({channels}) do not match existing "
                        f"channels ({current_channels}), please add images with the same channels."
                    )

            progress.setValue(progress.value() + 1)

        def work():
            for document in documents:
                if progress.wasCanceled():
                    return
                thumbnail = self.get_thumbnail_image(document)
                yield document, thumbnail

        do_work_async(
            work,
            return_callback=on_work_returned,
            yield_callback=on_work_yielded,
            error_callback=on_work_returned,
            async_disabled=self.async_disabled,
        )

    @property
    def subject_path(self) -> Path:
        return Path(self.subject_location_line_edit.text())

    def _add_cell_detection_channels_checkboxes(
        self, cell_detection_channels: List[str]
    ):
        assert self.subject is not None
        assert self.cell_detection_channels_checkboxes == []
        for channel_index, channel in enumerate(cell_detection_channels):
            checkbox = QCheckBox(channel)
            checkbox.setChecked(
                channel_index in self.subject.subject_info.cell_detection_channels
            )
            checkbox.stateChanged.connect(self.on_cell_detection_channel_changed)
            self.cell_detection_channels_checkboxes.append(checkbox)

        layout = QHBoxLayout()
        for checkbox in self.cell_detection_channels_checkboxes:
            layout.addWidget(checkbox)
        self._layout.addLayout(layout, 1, 0, 1, 3)

    def on_check_changed(self, _, checkbox: QCheckBox, document_index: int):
        assert self.subject is not None
        document = replace(
            self.subject.documents[document_index],
            ignore=not checkbox.isChecked(),
        )
        thumbnail = self.get_thumbnail_image(document)
        self.files_table.setCellWidget(
            document_index, 1, self.get_image_widget(thumbnail)
        )
        self.subject.documents[document_index] = document

    def on_cell_detection_channel_changed(self, _=None):
        assert self.subject is not None
        cell_detection_channels = [
            channel_index
            for channel_index, checkbox in enumerate(
                self.cell_detection_channels_checkboxes
            )
            if checkbox.isChecked()
        ]
        self.subject.subject_info.cell_detection_channels = cell_detection_channels

    def on_selected_channel_changed(self, _=None):
        new_channel = int(self.registration_channel_combobox.currentIndex())
        self.subject.subject_info = replace(
            self.subject.subject_info, channel=new_channel
        )
        self.files_table.setRowCount(0)
        self.add_document_rows_async(self.subject.documents)

    def on_add_images_clicked(self, _=None):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image(s)",
            str(Path.home()),
        )
        self.add_filenames_async(filenames)

    def on_create_subject_clicked(self, _=None):
        self.project.save()
        self.subject.save()
        self.accept()
