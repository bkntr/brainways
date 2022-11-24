import pickle
from pathlib import Path
from typing import Callable, List, Optional, Union

import dacite
import pandas as pd
from pandas import ExcelWriter

from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


class BrainwaysProject:
    def __init__(
        self,
        subjects: List[BrainwaysSubject],
        settings: ProjectSettings,
        path: Optional[Path] = None,
        lazy_init: bool = False,
    ):
        self.path = path
        self.subjects = subjects
        self.settings = settings

        self.atlas: Optional[BrainwaysAtlas] = None
        self.pipeline: Optional[BrainwaysPipeline] = None

        if not lazy_init:
            self.load_atlas()
            self.load_pipeline()

    @classmethod
    def open(cls, path: Union[Path, str], lazy_init: bool = False):
        if not path.exists():
            raise FileNotFoundError(f"BrainwaysProject file not found: {path}")
        if not path.suffix == ".bwp":
            raise FileNotFoundError(f"File is not a Brainways project file: {path}")

        with open(path, "rb") as f:
            serialized_settings = pickle.load(f)

        settings = dacite.from_dict(ProjectSettings, serialized_settings)
        subject_directories = [d for d in path.parent.glob("*") if d.is_dir()]
        subjects = [
            BrainwaysSubject.open(subject_path) for subject_path in subject_directories
        ]
        return cls(subjects=subjects, settings=settings, lazy_init=lazy_init)

    def load_atlas(self, load_volumes: bool = True):
        self.atlas = BrainwaysAtlas.load(
            self.settings.atlas, exclude_regions=[76, 42, 41]
        )  # TODO: from model
        # load volumes to cache
        if load_volumes:
            _ = self.atlas.reference
            _ = self.atlas.annotation
            _ = self.atlas.hemispheres

    def load_pipeline(self):
        if self.atlas is None:
            self.load_atlas()
        self.pipeline = BrainwaysPipeline(self.atlas)

    def move_images_directory(
        self, new_images_root: Path, old_images_root: Optional[Path] = None
    ):
        for subject in self.subjects:
            subject.move_images_root(
                new_images_root=new_images_root, old_images_root=old_images_root
            )
            subject.save()

    def create_excel(
        self,
        path: Union[Path, str],
        slice_info_predicate: Optional[Callable[[SliceInfo], bool]] = None,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
    ):
        cells_per_area_sheet = []
        cells_count_sheet = []
        for subject in self.subjects:
            cells_per_area_sheet.append(
                subject.cell_count_summary_co_labeling(
                    slice_info_predicate=slice_info_predicate,
                    min_region_area_um2=min_region_area_um2,
                    cells_per_area_um2=cells_per_area_um2,
                )
            )

            cells_count_sheet.append(
                subject.cell_count_summary_co_labeling(
                    slice_info_predicate=slice_info_predicate,
                    min_region_area_um2=min_region_area_um2,
                )
            )

        cells_count_sheet = pd.concat(
            [sheet for sheet in cells_count_sheet if sheet is not None], axis=0
        )
        cells_per_area_sheet = pd.concat(
            [sheet for sheet in cells_per_area_sheet if sheet is not None], axis=0
        )
        with ExcelWriter(path) as writer:
            cells_per_area_sheet.to_excel(
                writer, sheet_name=f"Cells per {cells_per_area_um2}um2", index=False
            )
            cells_count_sheet.to_excel(writer, sheet_name="Cell count", index=False)
