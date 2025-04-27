from typing import List, Optional

from brainglobe_atlasapi.structure_class import StructuresDict
from qtpy.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from brainways.ui.models.structure_tree_model import StructureTreeModel


class StructureSelectionDialog(QDialog):
    def __init__(self, structures: StructuresDict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Brain Structures")

        self.filter_textbox = QLineEdit()
        self.filter_textbox.setPlaceholderText("Filter structures...")

        self.tree_view = QTreeView()
        self.source_model = StructureTreeModel(structures)

        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.source_model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy_model.setFilterKeyColumn(
            0
        )  # Filter based on the first column (structure name/acronym)
        self.proxy_model.setRecursiveFilteringEnabled(True)

        self.tree_view.setModel(self.proxy_model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.filter_textbox.textChanged.connect(self.proxy_model.setFilterFixedString)

        source_root_index = self.source_model.index(0, 0, QModelIndex())
        if source_root_index.isValid():
            proxy_root_index = self.proxy_model.mapFromSource(source_root_index)
            if proxy_root_index.isValid():
                self.tree_view.expand(proxy_root_index)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.filter_textbox)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.filter_textbox.textChanged.connect(self._expand_matching_items)

    def _expand_matching_items(self, text: str):
        """Expands items that match the filter text."""
        if not text:
            source_root_index = self.source_model.index(0, 0, QModelIndex())
            if source_root_index.isValid():
                proxy_root_index = self.proxy_model.mapFromSource(source_root_index)
                if proxy_root_index.isValid():
                    self.tree_view.expand(proxy_root_index)
            return

        for row in range(self.proxy_model.rowCount()):
            index = self.proxy_model.index(row, 0)
            if index.isValid():
                self._expand_recursive(index)

    def _expand_recursive(self, index: QModelIndex):
        """Recursively expands an item and its children if they are visible."""
        if not index.isValid() or not self.proxy_model.hasChildren(index):
            return

        should_expand = False
        for i in range(self.proxy_model.rowCount(index)):
            child_index = self.proxy_model.index(i, 0, index)
            if (
                child_index.isValid()
                and self.proxy_model.mapToSource(child_index).isValid()
            ):
                should_expand = True
                break

        if should_expand:
            self.tree_view.expand(index)
            for i in range(self.proxy_model.rowCount(index)):
                child_index = self.proxy_model.index(i, 0, index)
                self._expand_recursive(child_index)

    def get_selected_structures(self) -> List[str]:
        """Gets the acronyms of the structures selected via checkboxes."""
        checked_ids = self.source_model.get_checked_structure_ids()
        selected_structures = []
        for structure_id in checked_ids:
            structure_info = self.source_model._structures.get(structure_id)
            if structure_info:
                acronym = structure_info.get("acronym")
                if acronym:
                    selected_structures.append(acronym)
        return selected_structures
