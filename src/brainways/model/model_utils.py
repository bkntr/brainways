from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser

from brainways.model.siamese.siamese_model import SiameseModel


def load_model(model_dir: Path) -> SiameseModel:
    """
    Load a model from the specified directory.

    Args:
        model_dir: Directory containing the model files

    Returns:
        Loaded and configured SiameseModel
    """
    # Load configuration
    config_path = model_dir / "pred_config.yml"
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    # Initialize model
    parser = ArgumentParser()
    parser.add_class_arguments(SiameseModel, nested_key="model", fail_untyped=False)
    cfg = parser.parse_object({"model": model_cfg})
    model = parser.instantiate_classes(cfg).model
    assert isinstance(model, SiameseModel)

    # Load state dict
    state_dict_path = model_dir / "model_state_dict.pt"
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode and move to GPU
    model.eval()
    model.to("cuda")

    return model
