from typing import Any, Dict, List, Optional

from brainglobe_atlasapi.structure_class import StructuresDict
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from treelib import Node


class StructureTreeModel(QAbstractItemModel):
    """A Qt item model for displaying a treelib structure tree."""

    def __init__(self, structures: StructuresDict, parent: Optional[Any] = None):
        super().__init__(parent)
        self._structures = structures
        self._tree = structures.tree
        self._check_states: Dict[Any, Qt.CheckState] = {}  # Store check states
        if not self._tree:
            self._root_node = None
        else:
            # Assuming the tree has a single root node
            root_id = self._tree.root
            if root_id is None:
                self._root_node = None  # Handle empty tree case
            else:
                self._root_node = self._tree.get_node(root_id)
                # Initialize check states (optional, default is Unchecked)
                for node_id in self._tree.expand_tree():
                    self._check_states[node_id] = Qt.Unchecked

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of columns (always 1)."""
        return 1

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of children for the given parent index."""
        if not parent.isValid():
            # Root level
            if self._root_node:
                return 1  # Only the root node itself at the top level
            else:
                return 0
        else:
            parent_node = parent.internalPointer()
            if isinstance(parent_node, Node):
                return len(self._tree.children(parent_node.identifier))
            return 0

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """Returns the data for the given index and role."""
        if not index.isValid():
            return None

        node = index.internalPointer()
        if not isinstance(node, Node):
            return None

        if role == Qt.DisplayRole:
            # Combine name and acronym for display
            structure_info = self._structures.get(node.identifier)
            if structure_info:
                name = structure_info.get("name", "Unknown")
                acronym = structure_info.get("acronym", "N/A")
                return f"{name} ({acronym})"
            return "Unknown Node"  # Fallback if structure info not found
        elif role == Qt.CheckStateRole:
            return self._check_states.get(node.identifier, Qt.Unchecked)
        # Add other roles if needed (e.g., Qt.UserRole for the ID)
        elif role == Qt.UserRole:
            return node.identifier  # Store the ID

        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        """Sets the data for the given index and role."""
        if not index.isValid() or role != Qt.CheckStateRole:
            return False

        node = index.internalPointer()
        if not isinstance(node, Node):
            return False

        node_id = node.identifier
        new_state = Qt.CheckState(value)  # Ensure value is a CheckState enum
        self._check_states[node_id] = new_state

        # # Propagate check state changes to children and parents if needed
        # # (Simple implementation: check/uncheck all children)
        # if new_state == Qt.Checked:
        #     for child_id in self._tree.is_branch(node_id):
        #         child_node = self._tree.get_node(child_id)
        #         if child_node:
        #             self._check_states[child_id] = Qt.Checked
        # elif new_state == Qt.Unchecked:
        #     for child_id in self._tree.is_branch(node_id):
        #         child_node = self._tree.get_node(child_id)
        #         if child_node:
        #             self._check_states[child_id] = Qt.Unchecked

        # Emit dataChanged signal for the modified item and potentially its children/parent
        self.dataChanged.emit(index, index, [role])

        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Returns the flags for the given index."""
        if not index.isValid():
            return Qt.NoItemFlags

        # Make items checkable
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> Any:
        """Returns the header data."""
        # No header needed for a simple tree view
        return None

    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Returns the index of the item in the model specified by the given row, column, and parent index."""
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parent_node: Optional[Node]
        if not parent.isValid():
            # Requesting index for the root node
            if row == 0 and column == 0 and self._root_node:
                parent_node = None  # Special case for root
                children = [
                    self._root_node
                ]  # Treat root as the only child of invalid parent
            else:
                return QModelIndex()  # Invalid index if not row 0 for root
        else:
            parent_node = parent.internalPointer()
            if not isinstance(parent_node, Node):
                return QModelIndex()
            children = self._tree.children(parent_node.identifier)

        if children and row < len(children):
            child_node = children[row]
            return self.createIndex(row, column, child_node)
        else:
            return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Returns the parent of the model item with the given index."""
        if not index.isValid():
            return QModelIndex()

        child_node = index.internalPointer()
        if not isinstance(child_node, Node) or child_node == self._root_node:
            return QModelIndex()  # Root has no parent in the model view

        parent_id = self._tree.parent(child_node.identifier)
        if parent_id is None:
            return QModelIndex()  # Should not happen if not root, but safety check

        parent_node = self._tree.get_node(parent_id.identifier)
        if parent_node is None or parent_node == self._root_node:
            # If parent is root, return invalid index to represent top level
            return QModelIndex()

        # Need to find the row of the parent within its own parent's children
        grandparent_id = self._tree.parent(parent_node.identifier)
        if grandparent_id is None:
            # Parent's parent is the invisible root of the tree, so parent is row 0
            # This case handles when the parent *is* the root node displayed in the view
            if parent_node == self._root_node:
                return QModelIndex()  # Root has no parent
            else:
                # This case should technically not be hit if the above root check works
                # but as a fallback, assume it's a child of the displayed root
                grandparent_children = [self._root_node] if self._root_node else []

        else:
            grandparent_node = self._tree.get_node(grandparent_id.identifier)
            if grandparent_node is None:
                return QModelIndex()  # Should not happen
            grandparent_children = self._tree.children(grandparent_node.identifier)

        try:
            parent_row = grandparent_children.index(parent_node)
            return self.createIndex(parent_row, 0, parent_node)
        except ValueError:
            return QModelIndex()  # Parent not found in grandparent's children? Error.

    def get_checked_structure_ids(self) -> List[Any]:
        """Returns a list of IDs for the checked structures."""
        return [
            node_id
            for node_id, state in self._check_states.items()
            if state == Qt.Checked
        ]
