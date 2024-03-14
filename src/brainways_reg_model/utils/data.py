from typing import Any

import kornia
import torch

from brainways_reg_model.utils.config import LabelParams


def value_to_model_label(value: Any, label_params: LabelParams):
    if label_params.label_names is not None:
        return torch.as_tensor(label_params.label_names.index(value)).long()
    elif label_params.limits is not None:
        label = (
            (value - label_params.limits[0])
            / (label_params.limits[1] - label_params.limits[0])
            * label_params.n_classes
        )
        return torch.clip(torch.as_tensor(label).long(), 0, label_params.n_classes - 1)
    else:
        raise ValueError("Label parameters must have either limits or label_names")


def model_label_to_value(label: Any, label_params: LabelParams):
    # if label_params.label_names is not None:
    #     if isinstance(label, Tensor):
    #         return [label_params.label_names[l] for l in label]
    #     else:
    #         return label_params.label_names[label]
    # elif label_params.limits is not None:
    #     return (label / label_params.n_classes) * (
    #         label_params.limits[1] - label_params.limits[0]
    #     ) + label_params.limits[0]
    # else:
    #     raise ValueError("Label parameters must have either limits or label_names")
    if label_params.label_names is not None:
        return label
    elif label_params.limits is not None:
        return (label / label_params.n_classes) * (
            label_params.limits[1] - label_params.limits[0]
        ) + label_params.limits[0]
    else:
        raise ValueError("Label parameters must have either limits or label_names")


def get_augmentation_rotation_deg(
    augmentation: kornia.augmentation.AugmentationBase2D,
) -> torch.Tensor:
    if (
        augmentation._transform_matrix is None
        and len(augmentation._transform_matrices) != 0
    ):
        augmentation._transform_matrix = augmentation._transform_matrices[0]
        for mat in augmentation._transform_matrices[1:]:
            if mat is not None:
                augmentation._update_transform_matrix(mat)

    mat = augmentation.transform_matrix
    return kornia.geometry.rad2deg(
        torch.as_tensor(torch.atan(mat[:, 1, 0] / mat[:, 0, 0]))
    )
