import re
from pathlib import Path
from typing import Union

from paquo.images import ImageProvider
from paquo.projects import QuPathProject

from brainways.utils.io_utils.file_iterators.image_entry import ImageEntry
from brainways.utils.io_utils.readers.czi import CziReader


class QuPathFileIterator:
    def __init__(self, path: Union[str, Path]):
        self.subject = QuPathProject(path)

    def __iter__(self):
        for entry in self.subject.images:
            image_path = Path(ImageProvider.path_from_uri(entry.uri))
            scene_id = int(re.findall(r"Scene #(\d+)", entry.image_name)[0])
            reader = CziReader(image_path)
            image_entry = ImageEntry(
                image=reader.read_scene(scene_id),
                path=image_path,
                scene=scene_id,
            )
            yield image_entry

    def __len__(self):
        return len(self.subject.images)
