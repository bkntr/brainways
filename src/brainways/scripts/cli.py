import click

from brainways.scripts.batch_create_thumbnails import batch_create_thumbnails
from brainways.scripts.cell_detection import cell_detection
from brainways.scripts.create_excel import create_excel
from brainways.scripts.create_reg_model_data import create_reg_model_data
from brainways.scripts.display_area import display_area
from brainways.scripts.import_cell_detections_keren import import_cell_detections_keren
from brainways.scripts.import_cells import import_cell_detections
from brainways.scripts.move_images import move_images_root


@click.group()
def cli():
    pass


cli.add_command(cell_detection, name="cell-detection")
cli.add_command(create_excel, name="create-excel")
cli.add_command(display_area, name="display-area")
cli.add_command(create_reg_model_data, name="create-reg-model-data")
cli.add_command(batch_create_thumbnails, name="batch-create-thumbnails")
cli.add_command(move_images_root, name="batch-move-images")
cli.add_command(import_cell_detections, name="import-cell-detections")
cli.add_command(import_cell_detections_keren, name="import-cell-detections-keren")


if __name__ == "__main__":
    cli()
