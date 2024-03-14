import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple, train_bn: bool):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        for param in pl_module.feature_extractor.parameters():
            param.requires_grad = False

    def finetune_function(
        self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer
    ):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:],
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=10.0,
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaing layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5],
                optimizer=optimizer,
                train_bn=self.train_bn,
                initial_denom_lr=10.0,
            )
