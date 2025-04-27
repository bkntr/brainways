import json
import logging
from dataclasses import asdict, fields
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import cv2
import dacite
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from natsort import natsorted, ns
from pandas import ExcelWriter
from tqdm import tqdm

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.pipeline.cell_detector import CellDetector
from brainways.project._utils import update_project_from_previous_versions
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import (
    ExcelMode,
    MaskFileFormat,
    ProjectSettings,
    RegisteredPixelValues,
    SliceInfo,
    SubjectInfo,
)
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.contrast import calculate_contrast
from brainways.utils.export import export_mask
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

        self._atlas: Optional[BrainwaysAtlas] = None
        self._pipeline: Optional[BrainwaysPipeline] = None

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
        self._atlas = BrainwaysAtlas.load(
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
        self._pipeline = BrainwaysPipeline(self.atlas)

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
        self,
        slice_infos: List[SliceInfo],
        resume: bool,
        save_cell_detection_masks_file_format: Optional[MaskFileFormat],
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
                save_cell_detection_masks_file_format=save_cell_detection_masks_file_format,
            )
            yield

        if save_cell_detection_masks_file_format is not None:
            open_directory(self.path.parent / "__outputs__" / "cell_detection_masks")

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
        n_bootstraps: int,
        multiple_comparison_correction_method: str,
        output_path: Path,
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

        graph = calculate_network_graph(
            cell_counts=cell_counts,
            n_bootstraps=n_bootstraps,
            multiple_comparison_correction_method=multiple_comparison_correction_method,
        )
        nx.write_graphml(graph, output_path.with_suffix(".graphml"))
        open_directory(output_path.parent)

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
        file_format: MaskFileFormat,
    ):
        if (
            file_format == MaskFileFormat.CSV
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
                Path(str(slice_info.path)).name + f"_{pixel_value_mode.name.lower()}"
            )
            output_path = output_dir / file_name
            export_mask(
                data=registered_values, path=output_path, file_format=file_format
            )
            yield

        open_directory(output_dir)

    def export_slice_locations(
        self, output_path: Path, slice_infos: List[SliceInfo]
    ) -> None:
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
            if slice_info.params.atlas is not None:
                ap_um = round(
                    atlas_reg_params.ap * self.atlas.brainglobe_atlas.resolution[0]
                )
            else:
                ap_um = float("nan")

            slice_locations.append(
                {
                    "subject": subject_info.name,
                    **subject_info.conditions,
                    "slice": str(slice_info.path),
                    "AP (μm)": ap_um,
                    "Frontal rotation": atlas_reg_params.rot_frontal,
                    "Horizontal rotation": atlas_reg_params.rot_horizontal,
                    "Sagittal rotation": atlas_reg_params.rot_sagittal,
                }
            )

        slice_locations_df = pd.DataFrame(slice_locations)
        slice_locations_df.to_csv(output_path, index=False)

        open_directory(output_path.parent)

    def export_annotated_region_images(
        self,
        output_dir: Union[str, Path],
        structure_acronyms: List[str],
        draw_cells: bool = False,
        slice_info_predicate: Optional[Callable[[SliceInfo], bool]] = None,
        output_dpi: int = 150,
    ):
        """Exports images of annotated brain regions for specified structures using Matplotlib.

        Generates two types of images per structure per slice:
        1. Full slice view with the structure highlighted.
        2. Zoomed-in view focusing on the structure's bounding box, preserving crop resolution.

        Args:
            output_dir: The root directory where images will be saved.
            structure_acronyms: A list of structure acronyms (e.g., ["NAc-sh", "BLA"]).
            draw_cells: If True, detected cells within the slice will be overlaid as points.
            slice_info_predicate: An optional function to filter which slices are processed.
            output_dpi: The resolution (dots per inch) for the output images.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the custom black to green colormap
        colors = [(0, 0, 0), (0, 1, 0)]  # Black to Green
        n_bins = 256
        cmap_name = "black_green"
        black_green_cmap = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bins
        )

        struct_tree = self.atlas.brainglobe_atlas.structures.tree
        leaf_ids = {node.identifier for node in struct_tree.leaves()}

        total_iterations = 0
        for subject in self.subjects:
            for _, slice_info in subject.valid_documents:
                if slice_info_predicate is None or slice_info_predicate(slice_info):
                    if slice_info.params.atlas is not None:
                        total_iterations += len(structure_acronyms)

        pbar = tqdm(total=total_iterations, desc="Exporting region images")

        for acronym in structure_acronyms:
            try:
                struct_id = self.atlas.brainglobe_atlas.structures[acronym]["id"]
            except KeyError:
                logging.warning(
                    f"Structure acronym '{acronym}' not found in atlas. Skipping."
                )
                pbar.update(
                    sum(
                        1
                        for subject in self.subjects
                        for _, slice_info in subject.valid_documents
                        if (
                            slice_info_predicate is None
                            or slice_info_predicate(slice_info)
                        )
                        and slice_info.params.atlas is not None
                    )
                )
                continue

            display_ids = [struct_id] + [
                child_id
                for child_id in struct_tree.is_branch(struct_id)
                if child_id in leaf_ids
            ]

            for subject in self.subjects:
                subject_name = subject.subject_info.name
                for _, slice_info in subject.valid_documents:
                    if slice_info_predicate is not None and not slice_info_predicate(
                        slice_info
                    ):
                        continue
                    if slice_info.params.atlas is None:
                        logging.warning(
                            f"Skipping slice {slice_info.path} for subject {subject_name}, structure {acronym}: Missing atlas parameters."
                        )
                        pbar.update(1)
                        continue

                    try:
                        image = subject.read_highres_image(slice_info)
                        mask = np.isin(
                            self.pipeline.get_registered_values_on_image(
                                slice_info,
                                pixel_value_mode=RegisteredPixelValues.STRUCTURE_IDS,
                            ),
                            display_ids,
                        )

                        if not np.any(mask):
                            pbar.update(1)
                            continue

                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        # --- Load Cells (if needed) ---
                        cells_coords = None
                        cells_df = None  # Initialize cells_df
                        if draw_cells:
                            try:
                                cells_df = subject.read_cell_detections(slice_info)
                                # Convert normalized coords to pixel coords
                                cells_coords = cells_df[["x", "y"]].values * np.array(
                                    [slice_info.image_size[1], slice_info.image_size[0]]
                                )  # x, y order for plotting
                            except FileNotFoundError:
                                logging.warning(
                                    f"Cell detections not found for slice {slice_info.path} in subject {subject_name}. Skipping cells for {acronym}."
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error loading cells for {slice_info.path}, structure {acronym}: {e}"
                                )

                        # --- Calculate Contrast Limits ---
                        vmin, vmax = (
                            np.quantile(image[image > 0], [0.01, 0.998])
                            if np.any(image > 0)
                            else (0, 1)
                        )

                        # --- Prepare Output Paths ---
                        slice_filename_stem = Path(str(slice_info.path)).stem
                        ap_value = int(
                            round(
                                slice_info.params.atlas.ap
                                * self.atlas.brainglobe_atlas.resolution[0]
                            )
                        )
                        # Directory for full slice images
                        full_output_dir = output_dir / acronym / subject_name / "full"
                        full_output_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path_full = (
                            full_output_dir / f"{ap_value}um_{slice_filename_stem}.jpg"
                        )
                        # Directory for cropped images WITHOUT cells
                        struct_output_dir = (
                            output_dir / acronym / subject_name / "struct"
                        )
                        struct_output_dir.mkdir(parents=True, exist_ok=True)
                        # Directory for cropped images WITH cells
                        struct_cells_output_dir = (
                            output_dir / acronym / subject_name / "struct-cells"
                        )
                        if draw_cells:  # Only create if needed
                            struct_cells_output_dir.mkdir(parents=True, exist_ok=True)

                        # --- Plot Full Slice ---
                        fig_full, ax_full = plt.subplots(
                            figsize=(10, 10)
                        )  # Adjust figsize as needed
                        ax_full.imshow(
                            image, cmap=black_green_cmap, vmin=vmin, vmax=vmax
                        )  # Use custom cmap

                        # Draw contours
                        for contour in contours:
                            # cv2 contours are [N, 1, 2] with (x, y)
                            poly = Polygon(
                                contour.squeeze(),
                                closed=True,
                                edgecolor="cyan",
                                facecolor="none",
                                linewidth=1.5,
                            )  # Mimic napari label look
                            ax_full.add_patch(poly)

                        ax_full.set_axis_off()
                        ax_full.set_position([0, 0, 1, 1])  # Remove padding
                        fig_full.savefig(
                            screenshot_path_full,
                            dpi=output_dpi,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close(fig_full)

                        # --- Plot Zoomed Structure Views ---
                        largest_contours = sorted(
                            contours, key=cv2.contourArea, reverse=True
                        )
                        if largest_contours:
                            max_area = cv2.contourArea(largest_contours[0])
                            largest_contours = [
                                c
                                for c in largest_contours
                                if cv2.contourArea(c) >= 0.3 * max_area
                            ]

                        for c_id, contour in enumerate(largest_contours):
                            x, y, w, h = cv2.boundingRect(
                                contour
                            )  # x, y is top-left corner
                            # Ensure width and height are at least 1 pixel
                            w = max(1, w)
                            h = max(1, h)
                            logging.info(
                                f"{acronym} bounding box: x={x}, y={y}, w={w}, h={h}"
                            )

                            # Define view box with 10% padding
                            pad_x = int(0.1 * w)
                            pad_y = int(0.1 * h)
                            ymin = max(0, y - pad_y)
                            ymax = min(image.shape[0], y + h + pad_y)
                            xmin = max(0, x - pad_x)
                            xmax = min(image.shape[1], x + w + pad_x)

                            # Calculate figure size in inches to match crop resolution at target DPI
                            fig_width_inches = w / output_dpi
                            fig_height_inches = h / output_dpi

                            fig_struct, ax_struct = plt.subplots(
                                figsize=(fig_width_inches, fig_height_inches)
                            )
                            ax_struct.imshow(
                                image, cmap=black_green_cmap, vmin=vmin, vmax=vmax
                            )

                            # Draw the current contour
                            poly = Polygon(
                                contour.squeeze(),
                                closed=True,
                                edgecolor="cyan",
                                facecolor="none",
                                linewidth=1.5,  # Consider scaling linewidth based on fig size?
                            )
                            ax_struct.add_patch(poly)

                            # Set zoom limits (invert y-axis for imshow)
                            ax_struct.set_xlim(xmin, xmax)
                            ax_struct.set_ylim(ymax, ymin)  # Inverted y-axis
                            ax_struct.set_axis_off()
                            ax_struct.set_position([0, 0, 1, 1])  # Remove padding

                            # --- Save version WITHOUT cells ---
                            base_filename = (
                                f"{ap_value}um_{slice_filename_stem}_contour_{c_id}"
                            )
                            # Path for image without cells (uses struct_output_dir)
                            screenshot_path_struct_no_cells = (
                                struct_output_dir / f"{base_filename}.jpg"
                            )
                            fig_struct.savefig(
                                screenshot_path_struct_no_cells,
                                dpi=output_dpi,
                                bbox_inches="tight",
                                pad_inches=0,
                            )

                            # --- Save version WITH cells (if requested and available) ---
                            if (
                                draw_cells
                                and cells_coords is not None
                                and len(cells_coords) > 0
                            ):
                                cells_in_box_mask = (
                                    (cells_coords[:, 0] >= xmin)
                                    & (cells_coords[:, 0] <= xmax)
                                    & (cells_coords[:, 1] >= ymin)
                                    & (cells_coords[:, 1] <= ymax)
                                )
                                cells_in_box = cells_coords[cells_in_box_mask]
                                if len(cells_in_box) > 0:
                                    scatter_sizes_struct = 10  # Default size
                                    if (
                                        cells_df is not None
                                        and "area_pixels" in cells_df.columns
                                    ):
                                        # Select sizes corresponding to cells within the box
                                        scatter_sizes_struct = (
                                            pd.to_numeric(
                                                cells_df.loc[
                                                    cells_in_box_mask, "area_pixels"
                                                ],
                                                errors="coerce",
                                            )
                                            .fillna(10)
                                            .values
                                        )

                                    ax_struct.scatter(
                                        cells_in_box[:, 0],
                                        cells_in_box[:, 1],
                                        s=scatter_sizes_struct,
                                        edgecolor="red",
                                        facecolor="none",
                                        linewidth=0.5,
                                    )  # Use calculated sizes

                                # Path for image with cells (uses struct_cells_output_dir)
                                screenshot_path_struct_with_cells = (
                                    struct_cells_output_dir
                                    / f"{base_filename}.jpg"  # Use same base filename, different dir
                                )
                                fig_struct.savefig(
                                    screenshot_path_struct_with_cells,
                                    dpi=output_dpi,
                                    bbox_inches="tight",
                                    pad_inches=0,
                                )

                            # Close the figure after all saves for this contour are done
                            plt.close(fig_struct)

                            yield
                    except Exception as e:
                        logging.error(
                            f"Failed to process slice {slice_info.path} for subject {subject_name}, structure {acronym}: {e}",
                            exc_info=True,
                        )
                    finally:
                        pbar.update(1)

        pbar.close()
        logging.info(f"Finished exporting images to {output_dir}")
        open_directory(output_dir)

    def _get_subjects(self, slice_infos: List[SliceInfo]) -> List[BrainwaysSubject]:
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

    @property
    def atlas(self) -> BrainwaysAtlas:
        if self._atlas is None:
            raise RuntimeError("Atlas not loaded")
        return self._atlas

    @property
    def pipeline(self) -> BrainwaysPipeline:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        return self._pipeline

    def possible_contrasts(self, condition: str) -> List[Tuple[str, str]]:
        condition_values = {
            subject.subject_info.conditions.get(condition) for subject in self.subjects
        }
        condition_values -= {None}
        possible_contrasts = list(combinations(sorted(condition_values), 2))
        return possible_contrasts

    def __len__(self) -> int:
        return len(self.subjects)
