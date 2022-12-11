from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Union

import kornia
import numpy as np
import torch
from bg_atlasapi import BrainGlobeAtlas

from brainways.utils.atlas.slice_atlas import slice_atlas
from brainways.utils.image import nonzero_bounding_box_tensor

_BRAINWAYS_ATLAS_CACHE = {}


class BrainwaysAtlas:
    def __init__(
        self,
        brainglobe_atlas: BrainGlobeAtlas,
        exclude_regions: Optional[Sequence[int]],
    ):
        self.brainglobe_atlas = brainglobe_atlas
        self.exclude_regions = exclude_regions
        self.bounding_boxes: List[torch.Tensor] = self._calc_bounding_boxes()

    @classmethod
    def load(
        cls,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]] = None,
    ):
        if isinstance(brainglobe_atlas, str):
            brainglobe_atlas = BrainGlobeAtlas(brainglobe_atlas)

        cache_key = (brainglobe_atlas.atlas_name, tuple(exclude_regions))

        if cache_key not in _BRAINWAYS_ATLAS_CACHE:
            _BRAINWAYS_ATLAS_CACHE.clear()
            atlas = BrainwaysAtlas(
                brainglobe_atlas=brainglobe_atlas, exclude_regions=exclude_regions
            )
            _BRAINWAYS_ATLAS_CACHE[cache_key] = atlas

        return _BRAINWAYS_ATLAS_CACHE[cache_key]

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
        ann_open = kornia.morphology.opening(self.annotation[:, None], kernel)[:, 0]
        for ann_open_slice in ann_open:
            boxes.append(nonzero_bounding_box_tensor(ann_open_slice))
        return boxes

    @property
    def shape(self):
        return self.brainglobe_atlas.shape

    @cached_property
    def reference(self):
        ref = self.brainglobe_atlas.reference.astype(np.float32)
        ref[self.annotation == 0] = 0
        return torch.as_tensor(ref / ref.max(), dtype=torch.float32)

    @cached_property
    def annotation(self):
        ann = self.brainglobe_atlas.annotation.astype(np.float32)
        exclude_mask = (ann[..., None] != self.exclude_regions).all(axis=-1)
        ann = ann * exclude_mask
        return torch.as_tensor(ann.astype(float), dtype=torch.float32)

    @cached_property
    def hemispheres(self):
        return torch.as_tensor(
            self.brainglobe_atlas.hemispheres.astype(np.float32), dtype=torch.float32
        )


@dataclass
class AtlasSlice:
    reference: torch.Tensor
    annotation: torch.Tensor
    hemispheres: torch.Tensor

    @property
    def shape(self):
        return self.reference.shape
