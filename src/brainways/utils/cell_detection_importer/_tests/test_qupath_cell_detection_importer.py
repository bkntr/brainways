from pathlib import Path

import pytest

from brainways.utils.cell_detection_importer.keren_cell_detection_importer import (
    KerenCellDetectionsImporter,
)


# Can't test now because sample image file doesn't have physical pixel sizes
@pytest.mark.skip
def test_read_cell_detections(mock_subject_documents):
    document = mock_subject_documents[0]
    sample_file = Path(__file__).parent / "qupath_sample_file.txt"
    importer = KerenCellDetectionsImporter()
    importer.read_cells_file(sample_file, document)


def test_find_cell_detections_file(mock_subject_documents, tmpdir):
    document = mock_subject_documents[0]
    csv_path = Path(tmpdir) / f"{Path(document.path.filename).name} Detections.txt"
    csv_path.touch()
    decoy_csv_path = (
        Path(tmpdir) / f"{Path(document.path.filename).name} 2 Detections.txt"
    )
    decoy_csv_path.touch()
    importer = KerenCellDetectionsImporter(
        cfos_threshold=0,
        drd1_threshold=0,
        drd2_threshold=0,
        oxtr_threshold=0,
    )
    found_csv_path = importer.find_cell_detections_file(
        root=Path(tmpdir), document=document
    )
    assert found_csv_path == csv_path


def test_find_cell_detections_file_doest_exist(mock_subject_documents, tmpdir):
    document = mock_subject_documents[0]
    decoy_csv_path = (
        Path(tmpdir) / f"{Path(document.path.filename).name} 2 Detections.txt"
    )
    decoy_csv_path.touch()
    importer = KerenCellDetectionsImporter(
        cfos_threshold=0,
        drd1_threshold=0,
        drd2_threshold=0,
        oxtr_threshold=0,
    )
    found_csv_path = importer.find_cell_detections_file(
        root=Path(tmpdir), document=document
    )
    assert found_csv_path is None
