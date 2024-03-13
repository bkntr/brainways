import shutil
import tempfile
from pathlib import Path

import click
import pandas as pd
import yaml
from tqdm import tqdm

from brainways_reg_model.utils.paths import (
    REAL_DATA_ROOT,
    REAL_DATA_ZIP_PATH,
    REAL_RAW_DATA_ROOT,
)


def prepare_real_data_phase(
    phase: str,
    metadata,
    labels: pd.DataFrame,
    output_dir: Path,
):
    output_dir = output_dir / phase
    output_dir.mkdir()

    # write metadata
    with open(output_dir / "metadata.yaml", "w") as outfile:
        yaml.dump(metadata, outfile, default_flow_style=False)

    # write labels
    labels.to_csv(output_dir / "labels.csv", index=False)

    # write images
    input_images_root = REAL_RAW_DATA_ROOT / "images"
    output_images_root = output_dir / "images"
    output_images_root.mkdir()
    for image_path in tqdm(labels.filename.to_list(), desc=phase):
        src = input_images_root / image_path
        dst = output_images_root / image_path
        dst.parent.mkdir(exist_ok=True, parents=True)
        assert src.exists()
        assert not dst.exists()
        shutil.copy(src, dst)


@click.command()
def prepare_real_data():
    tmp_dir = Path(tempfile.mkdtemp())
    labels = pd.read_csv(REAL_RAW_DATA_ROOT / "labels.csv")
    with open(REAL_RAW_DATA_ROOT / "metadata.yaml") as fd:
        metadata = yaml.safe_load(fd)
    test_animal_ids = [
        "28-1",
        "28-2",
        "29-2",
        "30-1",
        "30-2",
        "31-1",
        "31-2",
        "80-2",
        "81-1",
        "82-2",
        "83-2",
        "84-1",
        "85-2",
        "86-2",
        "adan 23-1",
        "adan 24-1",
        "adan 27-1",
        "adan 27-2",
        "adan 29-1",
        "nader-32-1",
        "nader-32-2",
        "nader-33-2",
        "nader-34-1",
        "nader-79-1",
        "nader-79-2",
    ]
    val_animal_ids = [
        "Dev24",
        "Dev25",
        "Dev26",
        "Retro1",
        "Retro10",
        "Retro12",
    ]
    # test_animal_ids = ["Dev24", "Dev25", "81-1", "Retro2", "29-2"]
    # val_animal_ids = ["Dev27", "Dev28", "Retro1", "85-2"]
    cfos_channels = ["Alexa Fluor 488", "AF488", "AF647"]
    test_labels = labels.loc[
        labels.animal_id.isin(test_animal_ids) & labels.channel.isin(cfos_channels)
    ]
    val_labels = labels.loc[
        labels.animal_id.isin(val_animal_ids) & labels.channel.isin(cfos_channels)
    ]
    train_labels = labels.loc[~labels.animal_id.isin(test_animal_ids + val_animal_ids)]

    prepare_real_data_phase(
        phase="test",
        metadata=metadata,
        labels=test_labels,
        output_dir=tmp_dir,
    )

    prepare_real_data_phase(
        phase="val",
        metadata=metadata,
        labels=val_labels,
        output_dir=tmp_dir,
    )
    prepare_real_data_phase(
        phase="train",
        metadata=metadata,
        labels=train_labels,
        output_dir=tmp_dir,
    )

    Path(REAL_DATA_ZIP_PATH).unlink(missing_ok=True)
    shutil.rmtree(REAL_DATA_ROOT, ignore_errors=True)
    shutil.make_archive(REAL_DATA_ZIP_PATH.with_suffix(""), "zip", tmp_dir)
    shutil.rmtree(str(tmp_dir))
