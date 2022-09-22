import shutil
from pathlib import Path

import click
import pandas as pd
import yaml
from click import confirm
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject


@click.command()
@click.option(
    "--input",
    type=Path,
    required=True,
    help=(
        "Input project file / directory of project files to create registration model"
        " training data for."
    ),
)
@click.option("--output", type=Path, required=True, help="Output directory.")
def create_reg_model_data(input: Path, output: Path):
    if output.exists():
        confirm("Output directory already exists, overwrite?", abort=True)
        shutil.rmtree(output)

    if (input / "brainways.bin").exists():
        paths = [input]
    else:
        paths = sorted(list(input.rglob("*.bin")))

    labels = []
    metadata = None
    output_images_dir = output / "images"
    output.mkdir()
    output_images_dir.mkdir()
    for project_path in tqdm(paths):
        project = BrainwaysProject.open(project_path)
        if metadata is None:
            project.load_atlas(load_volumes=False)
            metadata = {
                "atlas": project.settings.atlas,
                "ap_size": project.atlas.atlas.shape[0],
                "si_size": project.atlas.atlas.shape[1],
                "lr_size": project.atlas.atlas.shape[2],
            }
            with open(output / "metadata.yaml", "w") as outfile:
                yaml.dump(metadata, outfile, default_flow_style=False)
        if project.settings.atlas != metadata["atlas"]:
            print(
                f"Project {project_path.parent.name} has a different atlas"
                f" {project.settings.atlas} (expected  {metadata['atlas']})"
            )
            continue
        for document in tqdm(
            project.documents, desc=project_path.parent.name, leave=False
        ):
            ap = None
            rot_frontal = None
            rot_horizontal = None
            rot_sagittal = None
            hemisphere = None
            if not document.ignore:
                ap = document.params.atlas.ap
                rot_horizontal = document.params.atlas.rot_horizontal
                rot_sagittal = document.params.atlas.rot_sagittal
                hemisphere = document.params.atlas.hemisphere
                if document.params.affine is not None:
                    rot_frontal = document.params.affine.angle
            output_image_filename = (
                project_path.parent.relative_to(input)
                / project.thumbnail_path(document.path).name
            )
            labels.append(
                {
                    "filename": str(output_image_filename),
                    "animal_id": Path(project.project_path).name,
                    "image_id": str(document.path),
                    "ap": ap,
                    "rot_frontal": rot_frontal,
                    "rot_horizontal": rot_horizontal,
                    "rot_sagittal": rot_sagittal,
                    "ignore": document.ignore,
                    "hemisphere": hemisphere,
                }
            )
            project.read_lowres_image(document)
            src_image_path = project.thumbnail_path(document.path)
            output_image_path = output_images_dir / output_image_filename
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_image_path, output_image_path)

    # write labels
    pd.DataFrame(labels).to_csv(output / "labels.csv", index=False)
