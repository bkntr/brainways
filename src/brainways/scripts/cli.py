import click

from brainways.scripts.cell_detection import cell_detection
from brainways.scripts.create_excel import create_excel
from brainways.scripts.create_reg_model_data import create_reg_model_data
from brainways.scripts.display_area import display_area
from brainways.scripts.import_cells import import_cells


@click.group()
def cli():
    pass


cli.add_command(cell_detection, name="cell-detection")
cli.add_command(create_excel, name="create-excel")
cli.add_command(display_area, name="display-area")
cli.add_command(create_reg_model_data, name="create-reg-model-data")
cli.add_command(import_cells, name="import-cells")


if __name__ == "__main__":
    cli()
