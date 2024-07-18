from typing import List, Union

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas


def structure_labels(
    structures: np.ndarray, classes: Union[List[int], List[str]], atlas: BrainGlobeAtlas
):
    labels = np.zeros_like(structures)
    for i, cls in enumerate(classes):
        structure_id = atlas.structures[cls]["id"]
        leaves = atlas.structures.tree.leaves(structure_id)
        for node in leaves:
            labels[structures == node.identifier] = i + 1
    return labels
