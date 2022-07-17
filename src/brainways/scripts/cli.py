import click

from brainways.scripts.cell_detection import cell_detection
from brainways.scripts.create_excel import create_excel
from brainways.scripts.display_area import display_area


@click.group()
def cli():
    pass


cli.add_command(cell_detection, name="cell-detection")
cli.add_command(create_excel, name="create-excel")
cli.add_command(display_area, name="display-area")


if __name__ == "__main__":
    cli()
