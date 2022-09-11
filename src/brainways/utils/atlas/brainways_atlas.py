from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Union

import kornia
import numpy as np
import torch
from bg_atlasapi import BrainGlobeAtlas

from brainways.utils.atlas.slice_atlas import slice_atlas
from brainways.utils.image import nonzero_bounding_box_tensor

_BRAINGLOBE_ATLAS_CACHE = {}


class BrainwaysAtlas:
    def __init__(
        self,
        atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]] = None,
    ):
        self.exclude_regions = exclude_regions
        if isinstance(atlas, str):
            cache_key = (atlas, tuple(exclude_regions))
            if cache_key in _BRAINGLOBE_ATLAS_CACHE:
                self.atlas, self.bounding_boxes = _BRAINGLOBE_ATLAS_CACHE[cache_key]
            else:
                _BRAINGLOBE_ATLAS_CACHE.clear()
                self.atlas = BrainGlobeAtlas(atlas)
                self.bounding_boxes = self._calc_bounding_boxes()
                _BRAINGLOBE_ATLAS_CACHE[cache_key] = (
                    self.atlas,
                    self.bounding_boxes,
                )
        elif isinstance(atlas, BrainGlobeAtlas):
            self.atlas = atlas
            self.bounding_boxes = self._calc_bounding_boxes()
        else:
            raise ValueError(f"Unsupported atlas type {type(atlas)}")

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

    def _calc_bounding_boxes(self):
        boxes = []
        kernel = torch.ones(5, 5)
        for ann in self.annotation:
            ann_open = kornia.morphology.opening(ann[None, None], kernel)[0, 0]
            boxes.append(nonzero_bounding_box_tensor(ann_open))
        return boxes

    @property
    def shape(self):
        return self.atlas.shape

    @cached_property
    def reference(self):
        ref = self.atlas.reference.astype(np.float32)
        ref[self.annotation == 0] = 0
        return torch.as_tensor(ref / ref.max(), dtype=torch.float32)

    @cached_property
    def annotation(self):
        ann = self.atlas.annotation.astype(np.float32)
        exclude_mask = (ann[..., None] != self.exclude_regions).all(axis=-1)
        ann = ann * exclude_mask
        return torch.as_tensor(ann.astype(float), dtype=torch.float32)

    @cached_property
    def hemispheres(self):
        return torch.as_tensor(
            self.atlas.hemispheres.astype(np.float32), dtype=torch.float32
        )


@dataclass
class AtlasSlice:
    reference: torch.Tensor
    annotation: torch.Tensor
    hemispheres: torch.Tensor

    @property
    def shape(self):
        return self.reference.shape
