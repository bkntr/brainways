from pathlib import Path
from typing import Union


class PathFileIterator:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.is_file():
            self._paths = [self.path]
        else:
            self._paths = list(self.path.glob("*.czi"))

    def __iter__(self):
        yield from self._paths

    def __len__(self):
        return len(self._paths)
