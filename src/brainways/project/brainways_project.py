import json
import logging
from dataclasses import asdict, fields
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import dacite
import numpy as np
import pandas as pd
import scipy.io
from natsort import natsorted, ns
from pandas import ExcelWriter

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.pipeline.cell_detector import CellDetector
from brainways.project._utils import update_project_from_previous_versions
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import (
    ExcelMode,
    ProjectSettings,
    RegisteredAnnotationFileFormat,
    RegisteredPixelValues,
    SliceInfo,
    SubjectInfo,
)
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.contrast import calculate_contrast
from brainways.utils.network_analysis import calculate_network_graph
from brainways.utils.paths import open_directory
from brainways.utils.pls_analysis import (
    get_estimated_lv_plot,
    get_lv_p_values_plot,
    get_results_df_for_pls,
    get_salience_plot,
    pls_analysis,
    save_estimated_lv_plot,
    save_lv_p_values_plot,
    save_salience_plot,
)
from brainways.utils.view_brain_structure import view_brain_structure


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
        self._results_path = self.path.parent / (
            self.path.stem + "_cell_density_per_area_per_animal.xlsx"
        )
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
        subject_directories = [
            d.parent for d in path.parent.rglob("*.bws") if d.is_file()
        ]
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
            if excel_mode == ExcelMode.ROW_PER_SUBJECT:
                path = self._results_path
            else:
                path = self.path.parent / (
                    self.path.stem + "_cell_density_per_area_per_slice.xlsx"
                )
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

    def run_cell_detector_iter(
        self, slice_infos: List[SliceInfo], resume: bool
    ) -> Iterator:
        cell_detector = self.get_cell_detector()
        subjects = self._get_subjects(slice_infos)
        if not resume:
            for subject, slice_info in zip(subjects, slice_infos):
                subject.clear_cell_detection(slice_info)

        for subject, slice_info in zip(subjects, slice_infos):
            subject.run_cell_detector(
                slice_info=slice_info,
                cell_detector=cell_detector,
                default_params=self.settings.default_cell_detector_params,
            )
            yield

    def get_cell_detector(self) -> CellDetector:
        model_path = self.settings.cell_detector_custom_model_dir
        return CellDetector(model_path)

    def view_brain_structure(
        self,
        structure_names: List[str],
        condition_type: Optional[str] = None,
        condition_value: Optional[str] = None,
        num_subjects: Optional[int] = None,
        display_channel: Optional[int] = None,
        filter_cell_type: Optional[str] = None,
    ) -> None:
        view_brain_structure(
            project=self,
            structure_names=structure_names,
            condition_type=condition_type,
            condition_value=condition_value,
            num_subjects=num_subjects,
            display_channel=display_channel,
            filter_cell_type=filter_cell_type,
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
                " conditions (check logs for more info)"
            )

        if not self._results_path.exists():
            raise RuntimeError("Calculate results before calculating contrast")

        results_df = pd.read_excel(self._results_path)
        anova_df, posthoc_df = calculate_contrast(
            results_df=results_df,
            condition_col=condition_col,
            values_col=values_col,
            posthoc_comparisons=self.possible_contrasts(condition_col),
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
        )

        contrast_path = self.path.parent / f"contrast-{condition_col}-{values_col}.xlsx"
        with ExcelWriter(contrast_path) as writer:
            anova_df.to_excel(writer, sheet_name="ANOVA")
            posthoc_df.to_excel(writer, sheet_name="Posthoc")

        return anova_df, posthoc_df

    def calculate_pls_analysis(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        alpha: float,
        conditions: Optional[List[str]] = None,
        n_perm: int = 1000,
        n_boot: int = 1000,
    ):
        if not self.can_calculate_contrast(condition_col):
            raise RuntimeError(
                "Can't calculate contrast, some slice has missing parameters or missing"
                " conditions (check logs for more info)"
            )

        if not self._results_path.exists():
            raise RuntimeError("Calculate results before calculating contrast")

        results_df = pd.read_excel(self._results_path)

        results_df_pls = get_results_df_for_pls(
            results_df,
            values=values_col,
            condition_col=condition_col,
            conditions=conditions,
            min_per_group=min_group_size,
        )

        pls_results = pls_analysis(
            results_df_pls=results_df_pls,
            condition_col=condition_col,
            n_perm=n_perm,
            n_boot=n_boot,
        )

        estimated_lv_plot = get_estimated_lv_plot(
            pls_results=pls_results,
            results_df_pls=results_df_pls,
            condition=condition_col,
        )
        lv_p_values_plot = get_lv_p_values_plot(pls_results=pls_results)
        salience_plot = get_salience_plot(
            pls_results=pls_results, results_df_pls=results_df_pls
        )

        if conditions is None:
            conditions = results_df[condition_col].unique().tolist()
        conditions_str = ",".join(conditions)
        pls_file_prefix = (
            "Condition"
            f" Type={condition_col},Conditions={conditions_str},Values={values_col}"
        )
        pls_root_path = self.path.parent / "__outputs__" / "PLS" / pls_file_prefix
        pls_root_path.mkdir(parents=True, exist_ok=True)
        pls_excel_path = pls_root_path / f"pls-{pls_file_prefix}.xlsx"
        with ExcelWriter(pls_excel_path) as writer:
            results_df_pls.to_excel(writer, sheet_name="Values")
            lv_p_values_plot.to_excel(writer, sheet_name="LV P Value", index=False)
            estimated_lv_plot.to_excel(writer, sheet_name="Estimated LV1", index=False)
            salience_plot.to_excel(writer, sheet_name="PLS Salience", index=False)

        save_estimated_lv_plot(
            pls_root_path / f"estimated_lv-{pls_file_prefix}.png", estimated_lv_plot
        )
        save_lv_p_values_plot(
            pls_root_path / f"lv_p_values-{pls_file_prefix}.png",
            lv_p_values_plot,
            alpha=alpha,
        )
        save_salience_plot(
            pls_root_path / f"salience-{pls_file_prefix}.png",
            salience_plot,
            alpha=alpha,
        )

    def calculate_network_graph(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        alpha: float,
    ):
        if not self.can_calculate_contrast(condition_col):
            raise RuntimeError(
                "Can't calculate contrast, some slice has missing parameters or missing"
                " conditions (check logs for more info)"
            )

        if not self._results_path.exists():
            raise RuntimeError("Calculate results before calculating contrast")

        results_df = pd.read_excel(self._results_path)

        cell_counts = get_results_df_for_pls(
            results_df,
            values=values_col,
            condition_col=condition_col,
            min_per_group=min_group_size,
        )

        file_prefix = f"Condition={condition_col},Values={values_col}"
        graph_root_path = (
            self.path.parent / "__outputs__" / "network_graph" / file_prefix
        )
        graph_root_path.mkdir(parents=True, exist_ok=True)
        calculate_network_graph(
            cell_counts=cell_counts,
            alpha=alpha,
            output_path=graph_root_path.with_suffix(".graphml"),
        )

    def next_slice_missing_params(self) -> Optional[Tuple[int, int]]:
        for subject_idx, subject in enumerate(self.subjects):
            for slice_idx, slice_info in subject.valid_documents:
                for field in fields(slice_info.params):
                    if (
                        getattr(slice_info.params, field.name) is None
                        and field.name != "cell"
                    ):
                        logging.warning(
                            f"Missing parameter '{field.name}' in slice '{slice_info.path}' in subject '{subject.subject_info.name}'"
                        )
                        return subject_idx, slice_idx
        return None

    def can_calculate_results(self) -> bool:
        return self.next_slice_missing_params() is None

    def can_calculate_contrast(self, condition: str) -> bool:
        conditions = set()
        missing_conditions = False
        for subject in self.subjects:
            if subject.subject_info.conditions is not None:
                if condition in subject.subject_info.conditions:
                    conditions.add(subject.subject_info.conditions[condition])
                else:
                    logging.warning(
                        f"Missing condition '{condition}' in subject '{subject.subject_info.name}'"
                    )
                    missing_conditions = True
            else:
                logging.warning(
                    f"Missing conditions parameter in subject '{subject.subject_info.name}'"
                )
                missing_conditions = True

        if len(conditions) == 1:
            logging.warning(
                f"Only one condition in all subjects, can't calculate contrast: {conditions}"
            )

        return (
            self.can_calculate_results()
            and not missing_conditions
            and len(conditions) > 1
        )

    def export_registration_masks_async(
        self,
        output_dir: Path,
        pixel_value_mode: RegisteredPixelValues,
        slice_infos: List[SliceInfo],
        file_format: RegisteredAnnotationFileFormat,
    ):
        assert self.pipeline is not None

        if (
            file_format == RegisteredAnnotationFileFormat.CSV
            and pixel_value_mode != RegisteredPixelValues.STRUCTURE_IDS
        ):
            raise ValueError(
                "CSV format is only supported for structure IDs pixel values"
            )

        for slice_info in slice_infos:
            registered_values = self.pipeline.get_registered_values_on_image(
                slice_info, pixel_value_mode=pixel_value_mode
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            file_name = (
                Path(str(slice_info.path)).name
                + f"_{pixel_value_mode.name.lower()}.{file_format.value}"
            )
            output_path = output_dir / file_name
            logging.info(f"Saving {file_format.value} file to {output_path}")
            if file_format == RegisteredAnnotationFileFormat.NPZ:
                np.savez_compressed(
                    output_path,
                    values=registered_values,
                )
            elif file_format == RegisteredAnnotationFileFormat.MAT:
                scipy.io.savemat(
                    output_path,
                    {"values": registered_values},
                    do_compression=True,
                )
            elif file_format == RegisteredAnnotationFileFormat.CSV:
                np.savetxt(
                    output_path,
                    registered_values,
                    fmt="%d",
                    delimiter=",",
                )
            yield

        open_directory(output_dir)

    def export_slice_locations(
        self, output_path: Path, slice_infos: List[SliceInfo]
    ) -> None:
        assert self.atlas is not None

        if len(slice_infos) == 0:
            raise ValueError("No slices to export")

        subject_infos = [
            subject.subject_info for subject in self._get_subjects(slice_infos)
        ]

        missing_params = AtlasRegistrationParams(
            ap=float("nan"),
            rot_frontal=float("nan"),
            rot_horizontal=float("nan"),
            rot_sagittal=float("nan"),
        )

        slice_locations = []
        for subject_info, slice_info in zip(subject_infos, slice_infos):
            atlas_reg_params = slice_info.params.atlas or missing_params
            slice_locations.append(
                {
                    "subject": subject_info.name,
                    **subject_info.conditions,
                    "slice": str(slice_info.path),
                    "AP (μm)": atlas_reg_params.ap
                    * self.atlas.brainglobe_atlas.resolution[0],
                    "Frontal rotation": atlas_reg_params.rot_frontal,
                    "Horizontal rotation": atlas_reg_params.rot_horizontal,
                    "Sagittal rotation": atlas_reg_params.rot_sagittal,
                }
            )

        slice_locations_df = pd.DataFrame(slice_locations)
        slice_locations_df.to_csv(output_path, index=False)

        open_directory(output_path.parent)

    def _get_subjects(self, slice_infos: List[SliceInfo]) -> List[BrainwaysSubject]:
        # TODO: this is inefficient, we should have a better way to do this
        subjects = []
        for slice_info in slice_infos:
            for subject in self.subjects:
                if slice_info in subject.documents:
                    subjects.append(subject)
                    break
            else:
                raise ValueError(f"Slice {slice_info.path} not found in any subject")
        return subjects

    @property
    def n_valid_images(self):
        return sum(len(subject.valid_documents) for subject in self.subjects)

    def possible_contrasts(self, condition: str) -> List[Tuple[str, str]]:
        condition_values = {
            subject.subject_info.conditions.get(condition) for subject in self.subjects
        }
        condition_values -= {None}
        possible_contrasts = list(combinations(sorted(condition_values), 2))
        return possible_contrasts

    def __len__(self) -> int:
        return len(self.subjects)
