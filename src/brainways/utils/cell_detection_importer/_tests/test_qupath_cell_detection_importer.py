import logging
from pathlib import Path

import pytest

from brainways.utils.cell_detection_importer.qupath_cell_detection_importer import (
    QupathCellDetectionsImporter,
)


@pytest.fixture
def importer():
    return QupathCellDetectionsImporter(
        threshold_1=0,
        threshold_2=0,
        threshold_3=0,
        threshold_4=0,
        threshold_5=0,
        threshold_6=0,
        threshold_7=0,
        threshold_8=0,
        threshold_9=0,
        threshold_10=0,
    )


# Can't test now because sample image file doesn't have physical pixel sizes
@pytest.mark.skip
def test_read_cell_detections(mock_subject_documents, importer):
    document = mock_subject_documents[0]
    sample_file = Path(__file__).parent / "qupath_sample_file.txt"
    importer.read_cells_file(sample_file, document)


def test_find_cell_detections_file_single_candidate(
    mock_subject_documents, tmpdir, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    csv_path = root / f"{Path(document.path.filename).name} Detections.txt"
    csv_path.touch()
    found_csv_path = importer.find_cell_detections_file(root=root, document=document)
    assert found_csv_path == csv_path


def test_find_cell_detections_file_multiple_candidates(
    mock_subject_documents, tmpdir, caplog, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    csv_path_1 = root / f"{Path(document.path.filename).name} Detections.txt"
    csv_path_2 = root / f"{Path(document.path.filename).name} 2 Detections.txt"
    csv_path_1.touch()
    csv_path_2.touch()
    with caplog.at_level(logging.WARNING):
        found_csv_path = importer.find_cell_detections_file(
            root=root, document=document
        )
    assert found_csv_path is None
    assert "Multiple cell detection files found" in caplog.text


def test_find_cell_detections_file_no_candidates(
    mock_subject_documents, tmpdir, caplog, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    with caplog.at_level(logging.WARNING):
        found_csv_path = importer.find_cell_detections_file(
            root=root, document=document
        )
    assert found_csv_path is None
    assert "No cell detection file found" in caplog.text


def test_find_cell_detections_file_multiple_candidates_same_scene(
    mock_subject_documents, tmpdir, caplog, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    csv_path_1 = (
        root
        / f"{Path(document.path.filename).name} {document.path.scene} Detections.txt"
    )
    csv_path_2 = (
        root
        / f"{Path(document.path.filename).name} {document.path.scene} Detections duplicate.txt"
    )
    csv_path_1.touch()
    csv_path_2.touch()
    with caplog.at_level(logging.WARNING):
        found_csv_path = importer.find_cell_detections_file(
            root=root, document=document
        )
    assert found_csv_path is None
    assert "Multiple cell detection files found" in caplog.text


def test_find_cell_detections_file_single_scene_candidate(
    mock_subject_documents, tmpdir, caplog, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    csv_path_1 = root / f"{Path(document.path.filename).name} Detections.txt"
    csv_path_2 = (
        root
        / f"{Path(document.path.filename).name} {document.path.scene} Detections.txt"
    )
    csv_path_1.touch()
    csv_path_2.touch()
    found_csv_path = importer.find_cell_detections_file(root=root, document=document)
    assert found_csv_path == csv_path_2


def test_find_cell_detections_file_single_scene_candidate_complex(
    mock_subject_documents, tmpdir, caplog, importer
):
    document = mock_subject_documents[0]
    root = Path(tmpdir) / "detections"
    root.mkdir()
    csv_path_1 = (
        root
        / f"{Path(document.path.filename).name} 1{document.path.scene} Detections.txt"
    )
    csv_path_2 = (
        root
        / f"{Path(document.path.filename).name}_{document.path.scene}_Detections.txt"
    )
    csv_path_3 = (
        root
        / f"{Path(document.path.filename).name}_1{document.path.scene}_Detections.txt"
    )
    csv_path_1.touch()
    csv_path_2.touch()
    csv_path_3.touch()
    found_csv_path = importer.find_cell_detections_file(root=root, document=document)
    assert found_csv_path == csv_path_2
