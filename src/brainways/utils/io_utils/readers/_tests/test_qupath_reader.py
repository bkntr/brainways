from pathlib import Path

from brainways.utils.io_utils.readers.qupath_reader import QupathReader


def test_qupath_reader(test_image_path: Path):
    reader = QupathReader(test_image_path)
    reader.get_image_data()
