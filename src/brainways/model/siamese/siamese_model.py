from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from albumentations.core.composition import Compose, TransformType
from numpy.typing import NDArray

from brainways.model.siamese.siamese_backbone import SiameseBackbone
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


class ModelOutput(TypedDict):
    logits: torch.Tensor
    preds: torch.Tensor
    features: torch.Tensor


class SiameseModel(nn.Module):
    def __init__(
        self,
        backbone: SiameseBackbone,
        transforms: list[TransformType],
        ap_limits: dict[str, tuple[int, int]],
        inner_dim: int = 256,
    ) -> None:
        super().__init__()
        self._backbone = backbone
        self._transform = Compose(transforms)
        self._ap_limits = ap_limits

        downsample_op: nn.Module
        if self._backbone.feature_size > inner_dim:
            downsample_op = nn.Linear(self._backbone.feature_size, inner_dim)
        else:
            downsample_op = nn.Identity()

        self.feature_dim_reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            downsample_op,
        )

        self.classifier = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1),
        )

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> ModelOutput:
        """
        Forward pass of the model.

        Args:
            image_a (torch.Tensor): Input image A with dimensions (batch_size, channels, height, width).
            image_b (torch.Tensor): Input image B with dimensions (batch_size, channels, height, width).

        Returns:
            ModelOutput: Output of the model, including logits, predictions, and features.
        """
        embed_a = self._backbone.feature_extractor(image=image_a)
        embed_a_reduced = self.feature_dim_reduce(embed_a)
        embed_b = self._backbone.feature_extractor(image=image_b)
        embed_b_reduced = self.feature_dim_reduce(embed_b)
        linear_input = torch.cat([embed_a_reduced, embed_b_reduced], dim=1)
        logits = self.classifier(linear_input).squeeze(1)
        preds = torch.tanh(logits)

        return ModelOutput(logits=logits, preds=preds, features=linear_input)

    def predict(
        self,
        image: torch.Tensor | NDArray[np.uint8],
        atlas_name: str,
        transform_image: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the output based on the given input image and atlas.

        Args:
            image (torch.Tensor or np.ndarray): The input image to predict on.
            atlas_name (str): The name of the atlas to register to.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predicted AP value and corresponding atlas slice.
        """
        with torch.no_grad():
            if transform_image:
                image = self._batch_transform([image])

            assert isinstance(image, torch.Tensor)

            atlas = BrainwaysAtlas(atlas_name).raw_numpy_reference
            ap_limits = self._ap_limits[atlas_name]

            low = torch.full((len(image),), ap_limits[0], device=image.device)
            high = torch.full((len(image),), ap_limits[1], device=image.device)
            while (low < high).any():
                mid = (low + high) // 2
                atlas_slice = self._batch_transform(atlas[mid.cpu().numpy()])
                output = self.forward(image, atlas_slice)
                pred = output["preds"]
                high = torch.where((pred < 0) & (low < high), mid - 1, high)
                low = torch.where((pred >= 0) & (low < high), mid + 1, other=low)

            # Final prediction is the found low value
            pred_slice = atlas[low.cpu().numpy(), 0]
            pred_ap = low

        return pred_ap, pred_slice

    def _batch_transform(self, batch) -> torch.Tensor:
        """
        Applies the transformation to a batch of images.
        Args:
            batch: The batch of images to transform.
        Returns:
            torch.Tensor: The transformed batch of images.
        """
        return torch.stack([self._transform(image=t)["image"] for t in batch]).to(
            self.classifier[0].weight.device
        )
