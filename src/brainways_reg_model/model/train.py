# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computer vision example on Transfer Learning.
This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs.

The training consists of three stages.

From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.

From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).

Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.

Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from brainways_reg_model.model.dataset import BrainwaysDataModule
from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.utils.config import load_config
from brainways_reg_model.utils.milestones_finetuning import MilestonesFinetuning
from brainways_reg_model.utils.paths import REAL_DATA_ZIP_PATH

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_name",
    help="Config section name.",
    required=True,
)
@click.option(
    "--train-data",
    "train_data_path",
    type=Path,
    help="Training data path.",
    required=True,
)
@click.option(
    "--output",
    type=Path,
    help="Config section name.",
    required=True,
)
@click.option(
    "--pretrained-checkpoint",
    type=Path,
    help="Pretrained model path.",
)
@click.option("--num-workers", default=32, help="Number of data workers.")
def train(
    config_name: str,
    train_data_path: Path,
    output: Path,
    pretrained_checkpoint: Optional[Path],
    num_workers: int,
):
    config = load_config(config_name)
    pl.seed_everything(config.seed, workers=True)

    # init model
    if pretrained_checkpoint is not None:
        model = BrainwaysRegModel.load_from_checkpoint(
            pretrained_checkpoint, config=config
        )
    else:
        model = BrainwaysRegModel(config)

    # init data
    # transformes are performed in the model (in gpu) so are not given to the dataset
    datamodule = BrainwaysDataModule(
        data_paths={
            "train": train_data_path,
            "val": REAL_DATA_ZIP_PATH,
            "test": REAL_DATA_ZIP_PATH,
        },
        data_config=config.data,
        num_workers=num_workers,
        target_transform=model.target_transform,
    )

    finetuning_callback = MilestonesFinetuning(
        milestones=config.opt.milestones, train_bn=config.opt.train_bn
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=config.opt.monitor.metric, mode=config.opt.monitor.mode
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=str(output),
        callbacks=[finetuning_callback, checkpoint_callback],
        accelerator="auto",
        max_epochs=config.opt.max_epochs,
        check_val_every_n_epoch=config.opt.check_val_every_n_epoch,
        # num_sanity_val_steps=0,
    )

    # Train the model ⚡
    trainer.fit(model, datamodule=datamodule)

    checkpoint_path = output / "model.ckpt"
    logs_path = output / "logs"

    Path(checkpoint_path).unlink(missing_ok=True)
    shutil.move(checkpoint_callback.best_model_path, checkpoint_path)
    shutil.rmtree(logs_path, ignore_errors=True)
    shutil.copytree(trainer.log_dir, logs_path)


if __name__ == "__main__":
    train()
