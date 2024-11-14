import click

from brainways.scripts.batch_create_thumbnails import batch_create_thumbnails
from brainways.scripts.create_excel import create_excel
from brainways.scripts.create_reg_model_data import create_reg_model_data
from brainways.scripts.import_cell_detections_keren import import_cell_detections_keren
from brainways.scripts.import_cells import import_cell_detections
from brainways.scripts.move_images import move_images_root
from brainways.scripts.pls_analysis import run_pls_analysis
from brainways.scripts.run_ui import run_ui


@click.group()
def cli():
    pass


cli.add_command(create_excel, name="create-excel")
cli.add_command(create_reg_model_data, name="create-reg-model-data")
cli.add_command(batch_create_thumbnails, name="batch-create-thumbnails")
cli.add_command(move_images_root, name="batch-move-images")
cli.add_command(import_cell_detections, name="import-cell-detections")
cli.add_command(import_cell_detections_keren, name="import-cell-detections-keren")
cli.add_command(run_pls_analysis, name="pls-analysis")
cli.add_command(run_ui, name="ui")


if __name__ == "__main__":
    cli()
