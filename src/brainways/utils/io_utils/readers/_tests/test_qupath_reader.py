from pathlib import Path

from brainways.utils.io_utils.readers.qupath_reader import QupathReader


def test_qupath_reader(test_image_path: Path):
    QupathReader(test_image_path)
