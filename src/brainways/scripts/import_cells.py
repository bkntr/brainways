from argparse import ArgumentParser
from pathlib import Path

from duracell.project.brainways_project import BrainwaysProject
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args()

    paths = list(args.input.glob("*"))
    for project_path in tqdm(paths):
        project = BrainwaysProject.open(project_path)
        project.import_cells(args.cells)
        project.save(project_path)


if __name__ == "__main__":
    main()
