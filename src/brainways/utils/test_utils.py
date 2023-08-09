from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from brainways.pipeline.brainways_params import BrainwaysParams, TPSTransformParams


def randomly_modified_params(params: BrainwaysParams):
    modified_atlas = None
    modified_affine = None
    modified_tps = None
    modified_cell = None

    if params.atlas is not None:
        modified_atlas = replace(params.atlas, ap=random.uniform(0, 1))
    if params.affine is not None:
        modified_affine = replace(
            params.affine, angle=random.randint(10, 90), sx=0.5, sy=0.5
        )
    if params.tps is not None:
        modified_tps = TPSTransformParams(
            np.random.randint(10, size=(10, 2)).tolist(),
            np.random.randint(10, size=(10, 2)).tolist(),
        )
    if params.cell is not None:
        modified_cell = replace(params.cell, normalizer="none")
    modified_params = BrainwaysParams(
        atlas=modified_atlas,
        affine=modified_affine,
        tps=modified_tps,
        cell=modified_cell,
    )
    return modified_params
