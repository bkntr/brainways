from pathlib import Path

import click

from brainways.project.brainways_project import BrainwaysProject


@click.command()
@click.option(
    "--project",
    "project_path",
    type=Path,
    required=True,
    help="Brainways project path.",
)
@click.option("--condition", required=True)
@click.option("--values", required=True)
@click.option(
    "--min-group-size",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "--alpha",
    type=float,
    default=0.05,
    show_default=True,
)
def run_pls_analysis(
    project_path: Path,
    condition: str,
    values: str,
    min_group_size: int,
    alpha: float,
):
    project = BrainwaysProject.open(project_path)
    project.calculate_pls_analysis(
        condition_col=condition,
        values_col=values,
        min_group_size=min_group_size,
        alpha=alpha,
    )


if __name__ == "__main__":
    run_pls_analysis()
