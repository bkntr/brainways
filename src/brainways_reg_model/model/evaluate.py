import json
from pathlib import Path

import click
import pytorch_lightning as pl

from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.model.train import BrainwaysDataModule
from brainways_reg_model.utils.config import load_config
from brainways_reg_model.utils.paths import REAL_DATA_ZIP_PATH, REAL_TRAINED_MODEL_ROOT


@click.command()
@click.option(
    "--checkpoint",
    default=REAL_TRAINED_MODEL_ROOT / "model.ckpt",
    type=Path,
    show_default=True,
)
@click.option(
    "--output",
    default=REAL_TRAINED_MODEL_ROOT / "scores.json",
    type=Path,
    show_default=True,
)
@click.option(
    "--config",
    default="reg",
    show_default=True,
)
@click.option("--num-workers", default=4, help="Number of data workers.")
def evaluate(checkpoint: Path, output: Path, config: str, num_workers: int):
    config = load_config(config)

    # Load model
    model = BrainwaysRegModel.load_from_checkpoint(str(checkpoint))

    # # init data
    # synth_datamodule = BrainwaysDataModule(
    #     data_paths={
    #         "train": "data/synth.zip",
    #         "val": "data/synth.zip",
    #         "test": "data/synth.zip",
    #     },
    #     data_config=config.data,
    #     num_workers=num_workers,
    #     transform=model.transform,
    #     target_transform=model.target_transform,
    # )

    real_datamodule = BrainwaysDataModule(
        data_paths={
            "train": REAL_DATA_ZIP_PATH,
            "val": REAL_DATA_ZIP_PATH,
            "test": REAL_DATA_ZIP_PATH,
        },
        data_config=config.data,
        num_workers=num_workers,
    )

    # Initialize a trainer
    trainer = pl.Trainer(logger=False, accelerator="auto", max_epochs=-1)

    # Test the model ⚡
    scores = trainer.test(
        model,
        dataloaders=[
            real_datamodule.test_dataloader(),
            # synth_datamodule.test_dataloader(),
        ],
    )

    all_scores = {}
    for k, v in scores[0].items():
        all_scores[k.replace("dataloader_idx_0", "real")] = v
    # for k, v in scores[1].items():
    #     all_scores[k.replace("dataloader_idx_1", "synth")] = v

    with open(output, "w") as fp:
        json.dump(all_scores, fp)
