import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from brainways.project.info_classes import SliceInfo
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)


class QupathCellDetectionsImporter(CellDetectionImporter):
    parameters = {
        f"threshold_{i}": {
            "annotation": int,
            "value": -1,
            "options": {"nullable": True, "min": -1, "max": 10000},
            "label": f"Channel {i} Threshold",
        }
        for i in range(1, 11)
    }

    def __init__(
        self,
        threshold_1: int,
        threshold_2: int,
        threshold_3: int,
        threshold_4: int,
        threshold_5: int,
        threshold_6: int,
        threshold_7: int,
        threshold_8: int,
        threshold_9: int,
        threshold_10: int,
    ):
        super().__init__()
        self.thresholds = [
            threshold_1,
            threshold_2,
            threshold_3,
            threshold_4,
            threshold_5,
            threshold_6,
            threshold_7,
            threshold_8,
            threshold_9,
            threshold_10,
        ]

    def find_cell_detections_file(
        self,
        root: Path,
        document: SliceInfo,
    ) -> Optional[Path]:
        image_filename = Path(document.path.filename).name
        image_pattern = re.compile(f"{re.escape(image_filename)}.*")
        candidates = [
            candidate
            for candidate in root.rglob("*")
            if image_pattern.search(candidate.name)
        ]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            scene_number = document.path.scene
            image_and_scene_pattern = re.compile(
                f"{re.escape(image_filename)}.*(?<!\\d){scene_number}(?!\\d)"
            )
            scene_candidates = [
                candidate
                for candidate in candidates
                if image_and_scene_pattern.search(candidate.name)
            ]
            if len(scene_candidates) == 1:
                return scene_candidates[0]
            elif len(scene_candidates) > 1:
                logging.warning(
                    f"Multiple cell detection files found for {image_filename} scene {document.path.scene}: {candidates}"
                )
                return None
            else:
                logging.warning(
                    f"Multiple cell detection files found for {image_filename}: {candidates}"
                )
                return None
        else:
            logging.warning(f"No cell detection file found for {image_filename}")
            return None

    def read_cells_file(self, path: Path, document: SliceInfo) -> pd.DataFrame:
        input_cells_df = pd.read_csv(path, sep="\t")
        if "Class" in input_cells_df.columns:
            input_cells_df = input_cells_df[
                input_cells_df["Class"].isin(("Positive", "Negative"))
            ]
        image_size_um = [
            document.image_size[1] * document.physical_pixel_sizes[1],
            document.image_size[0] * document.physical_pixel_sizes[0],
        ]
        brainways_cells_df = pd.DataFrame(
            {
                "x": input_cells_df["Centroid X µm"] / image_size_um[0],
                "y": input_cells_df["Centroid Y µm"] / image_size_um[1],
            }
        ).dropna()

        for i, threshold in enumerate(self.thresholds):
            if threshold < 0:
                continue
            brainways_cells_df[f"LABEL-Channel-{i+1}"] = (
                input_cells_df[f"Subcellular: Channel {i+1}: Num single spots"]
                > threshold
            )

        assert (brainways_cells_df[["x", "y"]].values < 1).all()
        return brainways_cells_df
