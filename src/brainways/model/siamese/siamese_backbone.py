import timm  # type: ignore[import-untyped]
import torch
from torch import nn


class SiameseBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        self._model = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=True
        )

    def feature_extractor(self, image: torch.Tensor) -> torch.Tensor:
        return self._model(image)[-1]

    @property
    def feature_size(self) -> int:
        return self._model.feature_info[-1]["num_chs"]
