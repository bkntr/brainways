from functools import cached_property
from typing import Tuple

import cv2
import dask.array as da
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from brainways.utils._imports import STARDIST_AVAILABLE
from brainways.utils.image import normalize_contrast

if STARDIST_AVAILABLE:
    from csbdeep.data import Normalizer
else:
    Normalizer = object


class ClaheNormalizer(Normalizer):
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def before(self, x, axes):
        if isinstance(x, da.Array):
            x = x.compute()
        x = self.clahe.apply(x.astype(np.uint16))
        x = x.astype(np.float32)
        x = normalize_contrast(x, 0.0, 99.8)
        x = x.squeeze()
        return x[..., None]

    def after(self, mean, scale, axes):
        return mean, scale

    @property
    def do_after(self):
        return False


class MinMaxNormalizer(Normalizer):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def before(self, x, axes):
        x = np.clip(x, self.min, self.max)
        x = (x - self.min) / (self.max - self.min)
        x = x.astype(np.float32)
        x = x.squeeze()
        return x[..., None]

    def after(self, mean, scale, axes):
        return mean, scale

    @property
    def do_after(self):
        return False


class CellDetector:
    @cached_property
    def stardist(self):
        if STARDIST_AVAILABLE:
            from stardist.models import StarDist2D
        else:
            raise ImportError(
                "Tried to run cell detector model but stardist is not installed, "
                "please install by running `pip install stardist` or `pip install "
                "brainways[all]`"
            )

        return StarDist2D.from_pretrained("2D_versatile_fluo")

    def _filter_detections_by_area(self, labels: np.ndarray):
        regionprops_df = pd.DataFrame(
            regionprops_table(labels, properties=("label", "centroid", "area"))
        )

        cell_area = regionprops_df["area"]
        cell_area_mean = cell_area.mean()
        cell_area_std = cell_area.std()
        bad_cell_rows = (cell_area < cell_area_mean - 3 * cell_area_std) | (
            cell_area > cell_area_mean + 3 * cell_area_std
        )
        bad_label_idxs = regionprops_df[bad_cell_rows]["label"].to_numpy()
        bad_label_mask = (labels[..., None] == bad_label_idxs).any(axis=-1)
        labels[bad_label_mask] = 0
        return labels

    def run_cell_detector(self, image, normalizer: Normalizer, **kwargs) -> np.ndarray:
        labels, details = self.stardist.predict_instances_big(
            image,
            axes="YX",
            block_size=4096,
            min_overlap=128,
            normalizer=normalizer,
            **kwargs
        )
        # labels = self._filter_detections_by_area(labels)
        return labels

        # masks, flows, styles, diams = self.cellpose.eval(
        #     image.squeeze(), channels=[0, 0], **kwargs
        # )
        #
        # regionprops_df = pd.DataFrame(
        #     regionprops_table(
        #         masks, image, properties=("centroid", "area", "mean_intensity")
        #     )
        # )
        #
        # cells = regionprops_df[["centroid-0", "centroid-1"]].to_numpy()
        #
        # return masks

    def cells(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        physical_pixel_sizes: Tuple[float, float],
    ):
        regionprops_df = pd.DataFrame(
            regionprops_table(
                labels, image, properties=("centroid", "area", "mean_intensity")
            )
        ).astype(int)
        df = pd.DataFrame()
        df["x"] = regionprops_df["centroid-1"] / image.shape[1]
        df["y"] = regionprops_df["centroid-0"] / image.shape[0]
        df["area_pixels"] = regionprops_df["area"]
        df["area_um"] = (
            regionprops_df["area"] * physical_pixel_sizes[0] * physical_pixel_sizes[1]
        )
        df["mean_intensity"] = regionprops_df["mean_intensity"]

        return df
