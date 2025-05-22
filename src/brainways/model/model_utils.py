from pathlib import Path

import albumentations as A
import torch

from brainways.model.siamese.siamese_backbone import SiameseBackbone
from brainways.model.siamese.siamese_model import SiameseModel
from brainways.model.transforms.normalize_percentile import NormalizePercentile  # noqa: F401


def load_model(model_dir: Path) -> SiameseModel:
    """
    Load a model from the specified directory.

    Args:
        model_dir: Directory containing the model files

    Returns:
        Loaded and configured SiameseModel
    """
    model = SiameseModel(
        backbone=SiameseBackbone(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        ),
        transforms=[
            A.Resize(height=224, width=224, interpolation=3),
            NormalizePercentile(limits=(0.1, 99.8)),
            A.FromFloat(dtype="uint8"),
            A.ToRGB(),
            A.Normalize(),
            A.ToTensorV2(),
        ],
        ap_limits={
            "allen_mouse_25um": (1, 527),
            "whs_sd_rat_39um": (122, 800),
        },
        inner_dim=256,
    )

    # Load state dict
    state_dict_path = model_dir / "state_dict.pt"
    state_dict = torch.load(
        state_dict_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    model.eval()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")

    return model
