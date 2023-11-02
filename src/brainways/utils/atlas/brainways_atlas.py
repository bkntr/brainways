from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
from bg_atlasapi import BrainGlobeAtlas

from brainways.utils.atlas.slice_atlas import slice_atlas
from brainways.utils.image import nonzero_bounding_box


class BrainwaysAtlas:
    def __init__(
        self,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]],
    ):
        if isinstance(brainglobe_atlas, str):
            self.brainglobe_atlas = BrainGlobeAtlas(
                brainglobe_atlas, check_latest=False
            )
        else:
            self.brainglobe_atlas = brainglobe_atlas

        self.exclude_regions = exclude_regions

    @classmethod
    def load(
        cls,
        brainglobe_atlas: Union[str, BrainGlobeAtlas],
        exclude_regions: Optional[Sequence[int]],
    ):
        return BrainwaysAtlas(
            brainglobe_atlas=brainglobe_atlas, exclude_regions=exclude_regions
        )

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

    @cached_property
    def reference(self):
        ref = torch.as_tensor(self.brainglobe_atlas.reference.astype(np.float32))
        ref /= ref.max()
        ref *= self.annotation != 0
        return ref

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
