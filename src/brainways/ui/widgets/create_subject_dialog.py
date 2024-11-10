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
        self.channels_combobox = QComboBox()
        self.add_images_button = QPushButton("&Add Image(s)...", self)
        self.add_images_button.clicked.connect(self.on_add_images_clicked)
        self.files_table = self.create_table()
        self.conditions_widget = self._create_conditions_widget()
        self.bottom_label = QLabel("")

        self.layout = QGridLayout(self)
        self.setLayout(self.layout)

        cur_row = 0
        self.layout.addWidget(QLabel("Channel:"), cur_row, 0)
        self.layout.addWidget(self.channels_combobox, cur_row, 1)

        if self.project.settings.condition_names:
            cur_row += 1
            self.layout.addWidget(self.conditions_widget.native, cur_row, 0, 1, 3)

        cur_row += 1
        self.layout.addWidget(self.files_table, cur_row, 0, 1, 3)

        cur_row += 1
        self.layout.addWidget(self.bottom_label, cur_row, 0, 1, 2)

        cur_row += 1
        self.layout.addWidget(
            self.add_images_button, cur_row, 1, alignment=QtCore.Qt.AlignRight
        )

        self.layout.addWidget(self.create_subject_button, cur_row, 2)

        if parent is not None:
            self.resize(
                int(parent.parent().parent().parent().width() * 0.8),
                int(parent.parent().parent().parent().height() * 0.8),
            )

    def edit_subject_async(self, subject_index: int, document_index: int) -> None:
        self.setWindowTitle("Edit Subject")
        self._set_subject(self.project.subjects[subject_index])
        self.create_subject_button.setText("Done")
        self.add_document_rows_async(
            documents=self.subject.documents, select_document_index=document_index
        )

    def new_subject(self, subject_id: str, conditions: Dict[str, str]):
        assert subject_id is not None
        self.setWindowTitle(f"New Subject ({subject_id})")
        self._set_subject(
            self.project.add_subject(
                SubjectInfo(name=subject_id, conditions=conditions)
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
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.verticalHeader().hide()
        table.setShowGrid(False)
        return table

    def _create_conditions_widget(self) -> widgets.Widget:
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
        progress = QProgressDialog(
            "Opening images...", "Cancel", 0, len(documents), self
        )
        progress.setModal(True)
        progress.setValue(0)
        progress.setWindowTitle("Loading...")
        progress.show()

        def on_work_returned():
            self.channels_combobox.setCurrentIndex(self.project.settings.channel)
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

            if self.channels_combobox.count() == 0:
                self.channels_combobox.addItems(get_channels(document.path.filename))
                self.channels_combobox.currentIndexChanged.connect(
                    self.on_selected_channel_changed
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

    def on_check_changed(
        self, _=None, checkbox: QCheckBox = None, document_index: int = None
    ):
        document = replace(
            self.subject.documents[document_index],
            ignore=not checkbox.isChecked(),
        )
        thumbnail = self.get_thumbnail_image(document)
        self.files_table.setCellWidget(
            document_index, 1, self.get_image_widget(thumbnail)
        )
        self.subject.documents[document_index] = document

    def on_selected_channel_changed(self, _=None):
        new_channel = int(self.channels_combobox.currentIndex())
        self.project.settings = replace(self.project.settings, channel=new_channel)
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
        self.subject.save()
        self.accept()
