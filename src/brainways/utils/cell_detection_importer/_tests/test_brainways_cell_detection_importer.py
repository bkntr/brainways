from pathlib import Path

from brainways.utils.cell_detection_importer.brainways_cell_detection_importer import (
    BrainwaysCellDetectionsImporter,
)


def test_find_cell_detections_file(mock_subject_documents, tmpdir):
    document = mock_subject_documents[0]
    csv_path = (
        Path(tmpdir)
        / f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
    )
    csv_path.touch()
    decoy_csv_path = (
        Path(tmpdir)
        / f"{Path(document.path.filename).stem}_scene{document.path.scene + 1}.csv"
    )
    decoy_csv_path.touch()
    importer = BrainwaysCellDetectionsImporter()
    found_csv_path = importer.find_cell_detections_file(
        root=Path(tmpdir), document=document
    )
    assert found_csv_path == csv_path


def test_find_cell_detections_file_doest_exist(mock_subject_documents, tmpdir):
    document = mock_subject_documents[0]
    decoy_csv_path = (
        Path(tmpdir)
        / f"{Path(document.path.filename).stem}_scene{document.path.scene + 1}.csv"
    )
    decoy_csv_path.touch()
    importer = BrainwaysCellDetectionsImporter()
    found_csv_path = importer.find_cell_detections_file(
        root=Path(tmpdir), document=document
    )
    assert found_csv_path is None
