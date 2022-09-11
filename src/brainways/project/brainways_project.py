import logging
import pickle
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import dacite
import numpy as np
from pandas import DataFrame
from PIL import Image

from brainways.pipeline.brainways_pipeline import BrainwaysPipeline, PipelineStep
from brainways.project.brainways_project_settings import (
    ProjectDocument,
    ProjectSettings,
)
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cells import (
    cell_count_summary,
    filter_cells_on_annotation,
    filter_cells_on_tissue,
    get_region_areas,
    read_cells_csv,
)
from brainways.utils.image import get_resize_size, slice_to_uint8
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import get_image_size
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


class BrainwaysProject:
    def __init__(
        self,
        settings: ProjectSettings,
        documents: List[ProjectDocument] = None,
        project_path: Optional[Union[Path, str]] = None,
        atlas: Optional[BrainwaysAtlas] = None,
        pipeline: Optional[BrainwaysPipeline] = None,
    ):
        if atlas is not None:
            if atlas.atlas.atlas_name != settings.atlas:
                raise ValueError(
                    "Input atlas doesn't match atlas in project settings "
                    f"({atlas.atlas.atlas_name} != {settings.atlas})"
                )
        self.documents = documents or []
        self.settings = settings
        self.atlas = atlas
        self.pipeline = pipeline
        self._tmpdir = None

        # TODO: refactor, BrainwaysProject.create() and BrainwaysProject.open()
        if project_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.project_path = Path(self._tmpdir.name)
        else:
            self.project_path = self._get_project_dir(project_path)
            if not (self.project_path / "brainways.bin").exists():
                if self.project_path.exists():
                    if not self.project_path.is_dir() or any(
                        self.project_path.iterdir()
                    ):
                        raise FileExistsError(
                            f"New project directory {self.project_path} is not empty!"
                        )

        if not self.thumbnails_root.exists():
            self.thumbnails_root.mkdir()

    def close(self):
        self.documents = []
        self.settings = None
        self.atlas = None
        self.pipeline = None
        self.project_path = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def read_lowres_image(self, document: ProjectDocument) -> np.ndarray:
        thumbnail_path = self.thumbnail_path(
            document.path, channel=self.settings.channel
        )
        if thumbnail_path.exists():
            image = np.array(Image.open(thumbnail_path))
        else:
            reader = QupathReader(document.path.filename)
            reader.set_scene(document.path.scene)
            image = reader.get_thumbnail(
                target_size=document.lowres_image_size, channel=self.settings.channel
            )
            image = slice_to_uint8(image)
            Image.fromarray(image).save(thumbnail_path)
        return image

    def read_highres_image(
        self,
        document: ProjectDocument,
    ):
        reader = QupathReader(document.path.filename)
        reader.set_scene(document.path.scene)
        image = reader.get_image_data("YX", C=self.settings.channel)
        # image = slice_to_uint8(image)
        return image

    def load_atlas(self, load_volumes: bool = True):
        self.atlas = BrainwaysAtlas(
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

    def add_image(
        self, path: ImagePath, load_thumbnail: bool = True
    ) -> ProjectDocument:
        image_size = get_image_size(path)
        lowres_image_size = get_resize_size(
            input_size=image_size, output_size=(1024, 1024), keep_aspect=True
        )
        document = ProjectDocument(
            path=path,
            image_size=image_size,
            lowres_image_size=lowres_image_size,
        )
        if load_thumbnail:
            self.read_lowres_image(document)
        self.documents.append(document)
        return document

    @staticmethod
    def _get_project_dir(path: Union[Path, str]):
        project_dir = Path(path)
        if path.name == "brainways.bin":
            project_dir = path.parent
        return project_dir

    @classmethod
    def open(
        cls,
        path: Union[Path, str],
        atlas: Optional[BrainwaysAtlas] = None,
        pipeline: Optional[BrainwaysPipeline] = None,
    ):
        project_dir = BrainwaysProject._get_project_dir(path)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project path not found: {path}")

        with open(project_dir / "brainways.bin", "rb") as f:
            serialized_settings, serialized_documents = pickle.load(f)
        settings = dacite.from_dict(ProjectSettings, serialized_settings)
        documents = [dacite.from_dict(ProjectDocument, d) for d in serialized_documents]
        project = BrainwaysProject(
            settings=settings,
            documents=documents,
            project_path=project_dir,
            atlas=atlas,
            pipeline=pipeline,
        )
        return project

    def save(self, path: Optional[Union[Path, str]] = None):
        if path is None:
            path = self.project_path
        path = Path(path)
        project_dir = self._get_project_dir(path)
        if project_dir != self.project_path:
            if project_dir.exists():
                if project_dir.is_dir() and not any(project_dir.iterdir()):
                    shutil.rmtree(str(project_dir))
                else:
                    raise FileExistsError(
                        f"Project directory {project_dir} is not empty!"
                    )
            shutil.move(str(self.project_path), str(project_dir))
            self.project_path = project_dir
        serialized_settings = asdict(self.settings)
        serialized_documents = [asdict(d) for d in self.documents]
        with open(project_dir / "brainways.bin", "wb") as f:
            pickle.dump((serialized_settings, serialized_documents), f)

    def import_cells_yield_progress(
        self, path: Path
    ) -> Iterator[Tuple[int, ProjectDocument]]:
        for i, document in self.valid_documents:
            csv_filename = (
                f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
            )
            csv_path = path / csv_filename
            if not csv_path.exists():
                logging.warning(
                    f"found no cells for document: '{document.path}', "
                    f"csv path: {csv_path}"
                )
                continue
            cells = read_cells_csv(csv_path=csv_path, document=document)
            self.documents[i] = replace(document, cells=cells)
            yield i, document

    def import_cells(self, path: Path) -> None:
        for _ in self.import_cells_yield_progress(path):
            pass

    def get_valid_cells(self, document: ProjectDocument):
        image = self.read_lowres_image(document)
        valid_cells = filter_cells_on_tissue(cells=document.cells, image=image)
        valid_cells = filter_cells_on_annotation(
            cells=valid_cells,
            lowres_image_size=document.lowres_image_size,
            params=document.params,
            pipeline=self.pipeline,
        )
        return valid_cells

    def get_cells_on_atlas(self, documents: Optional[List[ProjectDocument]] = None):
        cells_on_atlas = []
        if documents is None:
            documents = (document for i, document in self.valid_documents)
        for document in documents:
            if document.cells is None:
                continue

            image_to_atlas_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
            )
            cells = self.get_valid_cells(document)
            cells_on_image = cells * document.lowres_image_size[::-1]
            cells_on_atlas.append(
                image_to_atlas_transform.transform_points(cells_on_image)
            )

        cells_on_atlas = np.concatenate(cells_on_atlas)
        return cells_on_atlas

    def cell_count_summary(
        self, min_region_area_um2: Optional[int] = None
    ) -> DataFrame:
        if self.pipeline is None:
            self.load_pipeline()

        all_region_areas = Counter()
        all_cells_on_atlas = []
        for _, document in self.valid_documents:
            if document.cells is None:
                raise RuntimeError(
                    f"{document.path}: missing cells, please run cell detection."
                )
            image = self.read_lowres_image(document)
            assert image.shape == document.lowres_image_size
            image_to_atlas_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
            )
            image_to_atlas_slice_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
                until_step=PipelineStep.TPS,
            )
            atlas_slice = self.pipeline.get_atlas_slice(document.params)
            cells = self.get_valid_cells(document)
            cells_on_image = cells * document.lowres_image_size[::-1]
            registered_image = image_to_atlas_slice_transform.transform_image(
                image,
                output_size=atlas_slice.shape,
            )

            cells_on_atlas = image_to_atlas_transform.transform_points(cells_on_image)

            region_areas = get_region_areas(
                annotation=atlas_slice.annotation.numpy(),
                atlas=self.atlas,
                registered_image=registered_image,
            )

            all_cells_on_atlas.append(cells_on_atlas)

            all_region_areas.update(region_areas)

        all_cells_on_atlas = np.concatenate(all_cells_on_atlas)
        df = cell_count_summary(
            cells=all_cells_on_atlas,
            region_areas=all_region_areas,
            atlas=self.atlas,
            min_region_area_um2=min_region_area_um2,
        )
        return df

    def thumbnail_path(self, image_path: ImagePath, channel: Optional[int] = None):
        if channel is None:
            channel = self.settings.channel

        suffixes = []
        if image_path.scene is not None:
            suffixes.append(f"Scene #{image_path.scene}")
        suffixes.append(f"Channel #{channel}")
        suffix = " ".join(suffixes)
        thumbnail_filename = f"{Path(image_path.filename).stem} [{suffix}].jpg"
        return self.thumbnails_root / thumbnail_filename

    @property
    def thumbnails_root(self) -> Path:
        return self.project_path / "thumbnails"

    @property
    def valid_documents(self) -> List[Tuple[int, ProjectDocument]]:
        return [
            (i, document)
            for i, document in enumerate(self.documents)
            if not document.ignore
        ]
