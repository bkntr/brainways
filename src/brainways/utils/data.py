from typing import Any

from brainways.utils.config import LabelParams


def value_to_model_label(value: Any, label_params: LabelParams):
    if label_params.label_names is not None:
        return label_params.label_names.index(value)
    elif label_params.limits is not None:
        label = int(
            (value - label_params.limits[0])
            / (label_params.limits[1] - label_params.limits[0])
            * label_params.n_classes
        )
        return max(min(label, label_params.n_classes - 1), 0)
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
