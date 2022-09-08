from pathlib import Path
from typing import Union

from brainways.utils.io_utils.file_iterators.path import PathFileIterator
from brainways.utils.io_utils.file_iterators.qupath import QuPathFileIterator


def get_file_iterator(path: Union[str, Path]):
    if Path(path).suffix == ".qpproj":
        return QuPathFileIterator(path)
    else:
        return PathFileIterator(path)
