from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from brainglobe_atlasapi import BrainGlobeAtlas
from torch.utils.data import Dataset

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)
from brainways_reg_model.slice_generator.stages import stages_dict
from brainways_reg_model.utils.slice_atlas import slice_atlas


class SliceGenerator(Dataset):
    def __init__(
        self,
        atlas: BrainGlobeAtlas,
        stages: List[Union[str, Dict, Callable]],
        n: int,
        rot_frontal_limit: Tuple[float, float],
        rot_horizontal_limit: Tuple[float, float],
        rot_sagittal_limit: Tuple[float, float],
    ):
        self.atlas = atlas
        self.reference = torch.tensor(
            atlas.reference / atlas.reference.max(), dtype=torch.float32
        )
        self.annotation = torch.tensor(
            atlas.annotation.astype(float), dtype=torch.float32
        )
        self.hemispheres = torch.tensor(
            atlas.hemispheres.astype(float), dtype=torch.float32
        )
        self.stages = self._parse_generation_stages(stages)
        self.n = n
        self.rot_frontal_limit = rot_frontal_limit
        self.rot_horizontal_limit = rot_horizontal_limit
        self.rot_sagittal_limit = rot_sagittal_limit

    @staticmethod
    def _parse_generation_stages(
        stages: List[Union[str, Dict, Callable]]
    ) -> List[Callable]:
        stage_fns = []
        for stage in stages:
            if isinstance(stage, str):
                stage_fns.append(stages_dict[stage]())
            elif isinstance(stage, dict):
                assert len(stage.keys()) == 1
                name = next(iter(stage.keys()))
                params = stage[name]
                stage_fn = stages_dict[name](**params)
                stage_fns.append(stage_fn)
            else:
                stage_fns.append(stage)
        return stage_fns

    def get_sample_params(self) -> SliceGeneratorSample:
        """
        Get random synthesis parameters for a sample
        :return:
        """
        return SliceGeneratorSample(
            ap=np.random.uniform(0, self.reference.shape[0]),
            si=self.reference.shape[1] / 2,
            lr=self.reference.shape[2] / 2,
            rot_frontal=np.random.uniform(*self.rot_frontal_limit),
            rot_horizontal=np.random.uniform(*self.rot_horizontal_limit),
            rot_sagittal=np.random.uniform(*self.rot_sagittal_limit),
            hemisphere=np.random.choice(["both", "left", "right"]),
        )

    def __getitem__(self, item):
        # generate random parameters
        sample = self.get_sample_params()
        # populate image and regions
        sample = self.populate_sample_slice_images(sample)

        # apply generation stages
        for generation_stage in self.stages:
            sample = generation_stage(sample)

        return sample

    def populate_sample_slice_images(self, sample: SliceGeneratorSample):
        sample.image = slice_atlas(
            shape=self.reference.shape[1:],
            volume=self.reference,
            ap=sample.ap,
            si=sample.si,
            lr=sample.lr,
            rot_frontal=sample.rot_frontal,
            rot_horizontal=sample.rot_horizontal,
            rot_sagittal=sample.rot_sagittal,
        )
        sample.regions = slice_atlas(
            shape=self.annotation.shape[1:],
            volume=self.annotation,
            ap=sample.ap,
            si=sample.si,
            lr=sample.lr,
            rot_frontal=sample.rot_frontal,
            rot_horizontal=sample.rot_horizontal,
            rot_sagittal=sample.rot_sagittal,
            interpolation="nearest",
        )
        sample.hemispheres = slice_atlas(
            shape=self.hemispheres.shape[1:],
            volume=self.hemispheres,
            ap=sample.ap,
            si=sample.si,
            lr=sample.lr,
            rot_frontal=sample.rot_frontal,
            rot_horizontal=sample.rot_horizontal,
            rot_sagittal=sample.rot_sagittal,
            interpolation="nearest",
        )
        sample.image = sample.image * (sample.regions > 0)

        # mask single hemisphere
        if sample.hemisphere != "both":
            hemisphere_idx = 1 if sample.hemisphere == "right" else 2
            sample.image *= sample.hemispheres == hemisphere_idx
            sample.regions *= sample.hemispheres == hemisphere_idx

        return sample

    def __len__(self):
        return self.n
