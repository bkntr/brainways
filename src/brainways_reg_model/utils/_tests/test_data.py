import pytest

from brainways_reg_model.utils.config import LabelParams
from brainways_reg_model.utils.data import model_label_to_value, value_to_model_label


@pytest.mark.parametrize(
    ["value", "label_params"],
    [
        (110, LabelParams(n_classes=300, limits=(100, 400))),
    ],
)
def test_value_to_model_label_and_back(value, label_params):
    model_label = value_to_model_label(value=value, label_params=label_params)
    converted_value = model_label_to_value(label=model_label, label_params=label_params)
    assert value == converted_value
