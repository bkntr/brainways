from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
from brainglobe_atlasapi import BrainGlobeAtlas
from numpy.typing import NDArray

from brainways.utils.atlas.slice_atlas import slice_atlas
from brainways.utils.image import nonzero_bounding_box


class BrainwaysAtlas:
    _instances = {}
    _atlas_obj_cache = {}
    _raw_numpy_reference_cache = {}

    def __new__(
        cls,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]] = None,
    ):
        atlas_name = (
            brainglobe_atlas
            if isinstance(brainglobe_atlas, str)
            else brainglobe_atlas.atlas_name
        )
        key = (atlas_name, tuple(exclude_regions) if exclude_regions else None)
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    def __init__(
        self,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]] = None,
    ):
        # Prevent re-initialization for singleton
        if hasattr(self, "_initialized") and self._initialized:
            return

        atlas_name = (
            brainglobe_atlas
            if isinstance(brainglobe_atlas, str)
            else brainglobe_atlas.atlas_name
        )
        if atlas_name in self._atlas_obj_cache:
            self.brainglobe_atlas = self._atlas_obj_cache[atlas_name]
        else:
            if isinstance(brainglobe_atlas, str):
                obj = BrainGlobeAtlas(atlas_name, check_latest=False)
            else:
                obj = brainglobe_atlas
            self._atlas_obj_cache[atlas_name] = obj
            self.brainglobe_atlas = obj

        self.exclude_regions = exclude_regions
        self._initialized = True

    @classmethod
    def load(
        cls,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]],
    ):
        return cls(brainglobe_atlas=brainglobe_atlas, exclude_regions=exclude_regions)

    def slice(
        self,
        ap: float,
        rot_horizontal: float = 0,
        rot_sagittal: float = 0,
        hemisphere: str = "both",
    ) -> AtlasSlice:
        reference = slice_atlas(
            self.shape[1:],
            volume=self.reference,
            ap=ap,
            si=self.reference.shape[1] // 2,
            lr=self.reference.shape[2] // 2,
            rot_frontal=0.0,
            rot_horizontal=rot_horizontal,
            rot_sagittal=rot_sagittal,
        )
        annotation = slice_atlas(
            self.shape[1:],
            volume=self.annotation,
            ap=ap,
            si=self.reference.shape[1] // 2,
            lr=self.reference.shape[2] // 2,
            rot_frontal=0.0,
            rot_horizontal=rot_horizontal,
            rot_sagittal=rot_sagittal,
            interpolation="nearest",
        ).to(torch.int32)
        hemispheres = slice_atlas(
            self.shape[1:],
            volume=self.hemispheres,
            ap=ap,
            si=self.reference.shape[1] // 2,
            lr=self.reference.shape[2] // 2,
            rot_frontal=0.0,
            rot_horizontal=rot_horizontal,
            rot_sagittal=rot_sagittal,
            interpolation="nearest",
        ).to(torch.uint8)

        if hemisphere != "both":
            hemisphere_idx = 1 if hemisphere == "right" else 2
            reference *= hemispheres == hemisphere_idx
            annotation *= hemispheres == hemisphere_idx

        return AtlasSlice(
            reference=reference, annotation=annotation, hemispheres=hemispheres
        )

    def bounding_box(self, ap: int) -> tuple[int, int, int, int]:
        kernel = np.ones((5, 5), np.uint8)
        annotation_mask = (self.annotation[ap] > 0).byte().numpy()
        ann_open = cv2.morphologyEx(annotation_mask, cv2.MORPH_OPEN, kernel)
        return nonzero_bounding_box(ann_open)

    @property
    def shape(self):
        return self.brainglobe_atlas.shape

    @property
    def raw_numpy_reference(self) -> NDArray[np.float32]:
        atlas_name = self.atlas_name
        if atlas_name in self._raw_numpy_reference_cache:
            return self._raw_numpy_reference_cache[atlas_name]
        ref = self.brainglobe_atlas.reference.astype(np.float32)
        ref /= ref.max()
        self._raw_numpy_reference_cache[atlas_name] = ref
        return ref

    @cached_property
    def reference(self):
        return torch.as_tensor(self.raw_numpy_reference) * (self.annotation != 0)

    @cached_property
    def annotation(self):
        ann = self.brainglobe_atlas.annotation
        ann *= np.isin(ann, self.exclude_regions, invert=True)
        return torch.as_tensor(ann.astype(np.float32))

    @cached_property
    def hemispheres(self):
        return torch.as_tensor(self.brainglobe_atlas.hemispheres.astype(np.float32))

    @property
    def atlas_name(self) -> str:
        return self.brainglobe_atlas.atlas_name


@dataclass
class AtlasSlice:
    reference: torch.Tensor
    annotation: torch.Tensor
    hemispheres: torch.Tensor

    @property
    def shape(self):
        return self.reference.shape
