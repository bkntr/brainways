from functools import cached_property
from typing import Dict, List, Sequence, Union

import kornia
import kornia.color
import pytorch_lightning as pl
import torch
from kornia import augmentation as K
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

from brainways_reg_model.model.approx_acc import ApproxAccuracy
from brainways_reg_model.model.atlas_registration_params import AtlasRegistrationParams
from brainways_reg_model.model.metric_dict_input_wrapper import MetricDictInputWrapper
from brainways_reg_model.utils.augmentations.random_contrast_limits import (
    RandomContrastLimits,
)
from brainways_reg_model.utils.config import BrainwaysConfig
from brainways_reg_model.utils.data import (
    get_augmentation_rotation_deg,
    model_label_to_value,
    value_to_model_label,
)


class BrainwaysRegModel(pl.LightningModule):
    def __init__(self, config: BrainwaysConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.label_params = self.config.data.label_params

        self._build_model()
        self._build_metrics()

    def _build_metrics_base(self, postfix=""):
        return {
            "ap_acc_10"
            + postfix: MetricDictInputWrapper(
                ApproxAccuracy(tolerance=10), "ap" + postfix
            ),
            "ap_acc_20"
            + postfix: MetricDictInputWrapper(
                ApproxAccuracy(tolerance=20), "ap" + postfix
            ),
            "ap_mae"
            + postfix: MetricDictInputWrapper(MeanAbsoluteError(), "ap" + postfix),
            "hem_acc"
            + postfix: MetricDictInputWrapper(
                Accuracy(
                    num_classes=self.label_params["hemisphere"].n_classes,
                    task="multiclass",
                ),
                "hemisphere" + postfix,
            ),
            "rot_frontal_acc"
            + postfix: MetricDictInputWrapper(
                ApproxAccuracy(tolerance=5),
                "rot_frontal" + postfix,
            ),
        }

    def _build_metrics(self):
        metrics = self._build_metrics_base()
        if self.config.opt.train_confidence:
            metrics.update(self._build_metrics_base(postfix="_confident"))
            if "valid" in self.config.model.outputs:
                metrics["valid_acc"] = MetricDictInputWrapper(
                    Accuracy(task="binary"), "valid"
                )
            metrics["confidence_acc"] = MetricDictInputWrapper(
                Accuracy(task="binary"), "confidence"
            )
            metrics["confident_percent"] = MetricDictInputWrapper(
                Accuracy(task="binary"), "confident_percent"
            )
        metrics = MetricCollection(metrics)

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.config.model.backbone)
        backbone = model_func(weights=ResNet50_Weights.DEFAULT)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        num_outputs = sum(
            self.label_params[output].n_classes for output in self.config.model.outputs
        )
        # Confidence output
        num_outputs += 2

        _fc_layers = [
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.nll_loss

    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        outputs = {}
        out_i = 0
        for output_name in self.config.model.outputs:
            outputs[output_name] = x[
                :, out_i : self.label_params[output_name].n_classes
            ]
        confidence = x[:, -2:]
        return outputs, confidence

    def predict(
        self, x: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> Union[AtlasRegistrationParams, List[AtlasRegistrationParams]]:
        if isinstance(x, torch.Tensor):
            x = [x]
            batch = False
        else:
            batch = True

        x = self.transform(torch.stack(x).to(self.device))
        y_logits, confidence = self(x)
        preds = self.postprocess(y_logits, confidence)

        if not batch:
            preds = preds[0]
        return preds

    def postprocess(
        self, logits: Dict[str, Tensor], confidence: Tensor
    ) -> List[AtlasRegistrationParams]:
        preds = {
            output_name: model_label_to_value(
                label=torch.argmax(logits[output_name], dim=-1),
                label_params=self.label_params[output_name],
            )
            for output_name in logits
        }
        confidence = torch.softmax(confidence, dim=-1)[:, 1].tolist()

        # Dict[List] -> List[Dict]
        list_of_dicts = [
            {output_name: float(preds[output_name][i]) for output_name in preds}
            for i in range(len(preds["ap"]))
        ]

        list_of_reg_params = [
            AtlasRegistrationParams(
                ap=d.get("ap"),
                rot_frontal=d.get("rot_frontal", 0.0),
                rot_horizontal=d.get("rot_horizontal", 0.0),
                rot_sagittal=d.get("rot_sagittal", 0.0),
                hemisphere=self.label_params["hemisphere"].label_names[
                    int(d.get("hemisphere", 0))
                ],
                confidence=confidence[i],
            )
            for i, d in enumerate(list_of_dicts)
        ]

        return list_of_reg_params

    def step(self, batch, batch_idx, phase: str, metrics: MetricCollection):
        batch_size, C, H, W = batch["image"].shape
        # Forward pass
        with torch.no_grad():
            if phase == "train":
                image = self.augmentation(batch["image"])
                self.add_augmentation_rotation_to_label(batch)
            else:
                image = self.transform(batch["image"])
        y_logits, confidence_logits = self(image)

        # mask out irrelevant outputs
        for output_name in self.config.model.outputs:
            output_mask = batch[output_name + "_mask"]
            batch[output_name] = batch[output_name][output_mask]
            y_logits[output_name] = y_logits[output_name][output_mask]

        # Compute losses and metrics
        losses = {}
        pred_values = {}
        gt_values = {}
        for output_name in self.config.model.outputs:
            if len(y_logits[output_name]) == 0:
                continue
            losses[output_name] = (
                self.loss_func(
                    input=F.log_softmax(y_logits[output_name], dim=-1),
                    target=batch[output_name],
                )
                * self.config.opt.loss_weights[output_name]
            )
            pred_values[output_name] = model_label_to_value(
                label=torch.argmax(y_logits[output_name], dim=-1),
                label_params=self.label_params[output_name],
            )
            gt_values[output_name] = model_label_to_value(
                label=batch[output_name],
                label_params=self.label_params[output_name],
            )

            # confident samples metrics
            if self.config.opt.train_confidence:
                for output_name in self.config.model.outputs:
                    (
                        pred_values[output_name + "_confident"],
                        gt_values[output_name + "_confident"],
                    ) = self.confident_samples_pred_and_label(
                        output_name=output_name,
                        batch=batch,
                        y_logits=y_logits,
                        confidence_logits=confidence_logits,
                    )
        if self.config.opt.train_confidence:
            pred_values["confident_percent"] = self.confident_mask(
                confidence_logits[batch["ap_mask"]]
            )
            gt_values["confident_percent"] = torch.ones_like(
                pred_values["confident_percent"]
            ).long()

        # confidence loss
        if self.config.opt.train_confidence:
            confidence_on_ap = confidence_logits[batch["ap_mask"]]
            confidence_loss, confidence_label = self.confidence_loss_and_label(
                batch, y_logits, confidence_on_ap
            )
            pred_values["confidence"] = confidence_on_ap
            gt_values["confidence"] = confidence_label
            losses["confidence"] = confidence_loss

        metrics_ = metrics(pred_values, gt_values)
        self.log_dict(metrics_, prog_bar=True, batch_size=batch_size)

        loss = torch.mean(torch.stack(list(losses.values())))

        if phase != "train":
            self.log(f"{phase}_loss", loss, prog_bar=True, batch_size=batch_size)

        return loss

    def add_augmentation_rotation_to_label(self, batch: Dict[str, Tensor]):
        rot_frontal_mask = batch["rot_frontal_mask"]
        if rot_frontal_mask.any():
            augmentation_deg = get_augmentation_rotation_deg(self.augmentation)
            rot_frontal_value = model_label_to_value(
                label=batch["rot_frontal"][rot_frontal_mask],
                label_params=self.label_params["rot_frontal"],
            )
            rot_frontal_value += augmentation_deg[rot_frontal_mask]
            batch["rot_frontal"][rot_frontal_mask] = value_to_model_label(
                value=rot_frontal_value,
                label_params=self.label_params["rot_frontal"],
            )

    def confident_samples_pred_and_label(
        self,
        output_name: str,
        batch: Dict[str, Tensor],
        y_logits: Dict[str, Tensor],
        confidence_logits: Tensor,
    ):
        output_mask = batch[output_name + "_mask"]
        confident_mask = self.confident_mask(confidence_logits[output_mask])
        pred = model_label_to_value(
            label=torch.argmax(y_logits[output_name][confident_mask], dim=-1),
            label_params=self.label_params[output_name],
        )
        label = model_label_to_value(
            label=batch[output_name][confident_mask],
            label_params=self.label_params[output_name],
        )

        if not confident_mask.any():
            return (
                torch.full(
                    [1],
                    model_label_to_value(
                        label=self.label_params[output_name].n_classes - 1,
                        label_params=self.label_params[output_name],
                    ),
                    device=confidence_logits.device,
                    dtype=pred.dtype,
                ),
                torch.zeros([1], device=confidence_logits.device, dtype=torch.long),
            )
        return pred, label

    def confident_mask(self, confidence_logits: Tensor):
        confidence_prob = torch.softmax(confidence_logits, dim=-1)[:, 1]
        confident_mask = confidence_prob > 0.85  # TODO: from config
        return confident_mask

    def confidence_loss_and_label(
        self, batch: Dict[str, Tensor], y_logits: Tensor, confidence: Tensor
    ):
        with torch.no_grad():
            pred_ap = model_label_to_value(
                label=torch.argmax(y_logits["ap"], dim=-1),
                label_params=self.label_params["ap"],
            ).int()
            gt_ap = model_label_to_value(
                label=batch["ap"],
                label_params=self.label_params["ap"],
            ).int()
            confidence_label = (abs(gt_ap - pred_ap) < 20).int()

        confidence_loss = self.loss_func(
            input=F.log_softmax(confidence, dim=-1),
            target=confidence_label.long(),
        )
        return confidence_loss, confidence_label

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "val", self.val_metrics)

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "test", self.test_metrics)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(
            trainable_parameters,
            lr=self.config.opt.lr,
            weight_decay=self.config.opt.weight_decay,
        )
        # scheduler = MultiStepLR(
        #     optimizer,
        #     milestones=self.config.opt.milestones,
        #     gamma=self.config.opt.lr_scheduler_gamma,
        # )
        # return [optimizer], [scheduler]
        return optimizer

    @cached_property
    def target_transform(self):
        return transforms.Lambda(lambda x: torch.LongTensor(x))

    @cached_property
    def transform(self):
        return K.AugmentationSequential(
            K.Resize(self.config.data.image_size),
            RandomContrastLimits((0.001, 0.001), (0.998, 0.998)),
            kornia.color.GrayscaleToRgb(),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ).to(self.device)

    @cached_property
    def augmentation(self):
        elastic_transform_aug = K.AugmentationSequential(
            K.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0)),
            K.RandomElasticTransform(kernel_size=(31, 31), sigma=(16.0, 16.0)),
            K.RandomElasticTransform(kernel_size=(21, 21), sigma=(12.0, 12.0)),
            random_apply=1,
        )
        random_augmentations = K.AugmentationSequential(
            # K.ColorJitter(0.2, 0.2, 0.2),
            K.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.7, 1.4),
                # shear=(20.0, 20.0),
            ),
            # K.RandomPerspective(distortion_scale=0.5),
            K.RandomBoxBlur(p=0.2),
            K.RandomErasing(),
            random_apply=(1,),
        )
        return K.AugmentationSequential(
            K.Resize(self.config.data.image_size),
            RandomContrastLimits(),
            elastic_transform_aug,
            random_augmentations,
            kornia.color.GrayscaleToRgb(),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            keepdim=True,
            same_on_batch=False,
        ).to(self.device)
