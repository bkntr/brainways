import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import dacite
import pandas as pd
from natsort import natsorted, ns
from pandas import ExcelWriter

from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.pipeline.cell_detector import CellDetector
from brainways.project._utils import update_project_from_previous_versions
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import (
    ExcelMode,
    ProjectSettings,
    SliceInfo,
    SubjectInfo,
)
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.contrast import calculate_contrast


class BrainwaysProject:
    def __init__(
        self,
        subjects: List[BrainwaysSubject],
        settings: ProjectSettings,
        path: Path,
        lazy_init: bool = False,
    ):
        if path.suffix != ".bwp":
            raise ValueError(f"Brainways project must be of .bwp file type, got {path}")

        self.path = path
        self._results_path = self.path.parent / "results.xlsx"
        self.subjects = subjects
        self.settings = settings

        self.atlas: Optional[BrainwaysAtlas] = None
        self.pipeline: Optional[BrainwaysPipeline] = None

        if not lazy_init:
            self.load_atlas()
            self.load_pipeline()

    @classmethod
    def create(
        cls,
        path: Union[Path, str],
        settings: ProjectSettings,
        lazy_init: bool = False,
        force: bool = False,
    ):
        path = Path(path)
        if path.suffix == ".bwp":
            project_dir = path.parent
        else:
            project_dir = path
            path = path / "brainways.bwp"

        if not force and project_dir.exists() and len(list(project_dir.glob("*"))) > 0:
            raise FileExistsError(f"Directory is not empty: {project_dir}")

        project_dir.mkdir(parents=True, exist_ok=True)
        project = cls(subjects=[], settings=settings, path=path, lazy_init=lazy_init)
        project.save()

        return project

    @classmethod
    def open(cls, path: Union[Path, str], lazy_init: bool = False):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BrainwaysProject file not found: {path}")
        if not path.suffix == ".bwp":
            raise FileNotFoundError(f"File is not a Brainways project file: {path}")

        update_project_from_previous_versions(path)

        with open(path) as f:
            serialized_settings = json.load(f)

        settings = dacite.from_dict(
            ProjectSettings, serialized_settings, config=dacite.Config(cast=[tuple])
        )

        project = cls(subjects=[], settings=settings, path=path, lazy_init=lazy_init)
        subject_directories = [d for d in path.parent.glob("*") if d.is_dir()]
        subject_directories = natsorted(
            subject_directories, alg=ns.IGNORECASE, key=lambda x: x.name
        )
        subjects = [
            BrainwaysSubject.open(path=subject_path / "data.bws", project=project)
            for subject_path in subject_directories
        ]
        project.subjects = subjects
        return project

    def save(self):
        serialized_settings = asdict(self.settings)
        with open(self.path, "w") as f:
            json.dump(serialized_settings, f)

    def add_subject(self, subject_info: SubjectInfo) -> BrainwaysSubject:
        subject = BrainwaysSubject.create(subject_info=subject_info, project=self)
        self.subjects.append(subject)
        return subject

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

    def calculate_results_iter(
        self,
        path: Optional[Union[Path, str]] = None,
        slice_info_predicate: Optional[Callable[[SliceInfo], bool]] = None,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
        excel_mode: ExcelMode = ExcelMode.ROW_PER_SUBJECT,
    ) -> Iterator:
        if path is None:
            path = self._results_path
        if not path.suffix == ".xlsx":
            path = Path(str(path) + ".xlsx")

        results = []
        for subject in self.subjects:
            results.append(
                subject.cell_count_summary(
                    slice_info_predicate=slice_info_predicate,
                    min_region_area_um2=min_region_area_um2,
                    cells_per_area_um2=cells_per_area_um2,
                    min_cell_size_um=min_cell_size_um,
                    max_cell_size_um=max_cell_size_um,
                    excel_mode=excel_mode,
                )
            )

            yield

        results = pd.concat([sheet for sheet in results if sheet is not None], axis=0)
        with ExcelWriter(path) as writer:
            results.to_excel(writer, index=False)

    def calculate_results(
        self,
        path: Optional[Union[Path, str]] = None,
        slice_info_predicate: Optional[Callable[[SliceInfo], bool]] = None,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
        excel_mode: ExcelMode = ExcelMode.ROW_PER_SUBJECT,
    ) -> None:
        for _ in self.calculate_results_iter(
            path=path,
            slice_info_predicate=slice_info_predicate,
            min_region_area_um2=min_region_area_um2,
            cells_per_area_um2=cells_per_area_um2,
            min_cell_size_um=min_cell_size_um,
            max_cell_size_um=max_cell_size_um,
            excel_mode=excel_mode,
        ):
            pass

    def import_cell_detections_iter(
        self,
        importer: CellDetectionImporter,
        cell_detections_root: Path,
    ) -> Iterator:
        for subject in self.subjects:
            yield from subject.import_cell_detections_iter(
                root=cell_detections_root,
                cell_detection_importer=importer,
            )

    def import_cell_detections(
        self,
        importer: CellDetectionImporter,
        cell_detections_root: Path,
    ) -> None:
        for subject in self.subjects:
            subject.import_cell_detections(
                root=cell_detections_root,
                cell_detection_importer=importer,
            )

    def run_cell_detector_iter(self) -> Iterator:
        cell_detector = CellDetector()
        for subject in self.subjects:
            yield from subject.run_cell_detector_iter(
                cell_detector, default_params=self.settings.default_cell_detector_params
            )

    def run_cell_detector(self) -> None:
        cell_detector = CellDetector()
        for subject in self.subjects:
            subject.run_cell_detector(
                cell_detector, default_params=self.settings.default_cell_detector_params
            )

    def calculate_contrast(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        pvalue: float,
        multiple_comparisons_method: str = "fdr_bh",
    ):
        if not self.can_calculate_contrast(condition_col):
            raise RuntimeError(
                "Can't calculate contrast, some slice has missing parameters or missing"
                " conditions"
            )

        if not self._results_path.exists():
            raise RuntimeError("Calculate results before calculating contrast")

        results_df = pd.read_excel(self._results_path)
        return calculate_contrast(
            results_df=results_df,
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
        )

    def next_slice_missing_params(self) -> Optional[Tuple[int, int]]:
        for subject_idx, subject in enumerate(self.subjects):
            for slice_idx, slice_info in subject.valid_documents:
                for field in fields(slice_info.params):
                    if getattr(slice_info.params, field.name) is None:
                        return subject_idx, slice_idx
        return None

    def can_calculate_results(self) -> bool:
        return self.next_slice_missing_params() is None

    def can_calculate_contrast(self, condition: str) -> bool:
        conditions = set()
        for subject in self.subjects:
            if subject.subject_info.conditions is not None:
                conditions.add(subject.subject_info.conditions.get(condition))
            else:
                conditions.add(None)
        return (
            self.can_calculate_results()
            and None not in conditions
            and len(conditions) > 1
        )

    @property
    def n_valid_images(self):
        return sum(len(subject.valid_documents) for subject in self.subjects)

    def __len__(self) -> int:
        return len(self.subjects)
