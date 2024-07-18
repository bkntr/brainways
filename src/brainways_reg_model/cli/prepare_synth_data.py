import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd
import pytorch_lightning as pl
import yaml
from brainglobe_atlasapi import BrainGlobeAtlas
from lightning_fabric.utilities.seed import pl_worker_init_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainways_reg_model.slice_generator.slice_generator import SliceGenerator
from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)
from brainways_reg_model.utils.config import load_config, load_synth_config
from brainways_reg_model.utils.paths import SYNTH_DATA_ROOT


def atlas_reference(atlas: str, brainglobe: bool, axes: Tuple[int, int, int]):
    if brainglobe:
        atlas = BrainGlobeAtlas(atlas)
        reference = atlas.reference
        reference = reference.transpose(axes)
    else:
        raise NotImplementedError()
    return reference


def create_dataset(
    phase: str,
    atlas: BrainGlobeAtlas,
    stages: List[Dict],
    n: int,
    rot_frontal_limit: Tuple[float, float],
    rot_horizontal_limit: Tuple[float, float],
    rot_sagittal_limit: Tuple[float, float],
    output: Path,
    debug: bool,
):
    root_dir = output / phase
    images_dir = root_dir / "images"
    images_dir.mkdir(parents=True)

    generator = DataLoader(
        SliceGenerator(
            atlas=atlas,
            stages=stages,
            n=n,
            rot_frontal_limit=rot_frontal_limit,
            rot_horizontal_limit=rot_horizontal_limit,
            rot_sagittal_limit=rot_sagittal_limit,
        ),
        batch_size=None,
        num_workers=16 if not debug else 1,
        worker_init_fn=pl_worker_init_function,
    )
    all_labels = []
    for sample_idx, sample in enumerate(tqdm(generator, desc=phase)):
        sample: SliceGeneratorSample
        filename = f"{sample_idx}.jpg"
        sample.image.save(images_dir / filename)
        sample.regions.save(
            str(images_dir / f"{sample_idx}-structures.tif"), compression="tiff_lzw"
        )

        # save non-image parameters
        attrs = asdict(sample)
        attrs["filename"] = filename
        del attrs["image"]
        del attrs["regions"]
        del attrs["hemispheres"]
        all_labels.append(attrs)
    all_labels = pd.DataFrame(all_labels)
    all_labels.to_csv(str(root_dir / "labels.csv"), float_format="%.3f", index=False)

    metadata = {
        "atlas": atlas.atlas_name,
        "ap_size": atlas.shape[0],
        "si_size": atlas.shape[1],
        "lr_size": atlas.shape[2],
        "rot_frontal_limit": list(rot_frontal_limit),
        "rot_horizontal_limit": list(rot_horizontal_limit),
        "rot_sagittal_limit": list(rot_sagittal_limit),
    }

    with open(root_dir / "metadata.yaml", "w") as outfile:
        yaml.dump(metadata, outfile, default_flow_style=False)


def _prepare_synth_data_phase(phase: str, output_dir: Path, debug: bool):
    config = load_config()
    synth_config = load_synth_config()
    atlas = BrainGlobeAtlas(config.data.atlas.name)

    create_dataset(
        phase=phase,
        atlas=atlas,
        stages=synth_config["stages"],
        n=synth_config[phase],
        rot_frontal_limit=config.data.label_params["rot_frontal"].limits,
        rot_horizontal_limit=config.data.label_params["rot_horizontal"].limits,
        rot_sagittal_limit=config.data.label_params["rot_sagittal"].limits,
        output=output_dir,
        debug=debug,
    )


@click.command()
@click.option("--output", default=SYNTH_DATA_ROOT, type=Path, help="Output directory.")
@click.option("--debug", is_flag=True, help="Debug mode (no workers).")
def prepare_synth_data(output: Path, debug: bool):
    pl.seed_everything(load_config().seed)

    if output.exists():
        if click.confirm(f"Synthetic data already exists in {output}, overwrite?"):
            shutil.rmtree(str(output))
        else:
            return

    for phase in ("test", "val", "train"):
        _prepare_synth_data_phase(phase=phase, output_dir=output, debug=debug)

    output.parent.mkdir(exist_ok=True)
    output.with_suffix(".zip").unlink(missing_ok=True)
    shutil.make_archive(str(output), "zip", str(output.parent))


if __name__ == "__main__":
    prepare_synth_data()
