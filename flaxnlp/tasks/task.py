from os import PathLike
from typing import Union

import colt


class Task(colt.Registrable):
    def run(self, work_dir: Union[str, PathLike]) -> None:
        raise NotImplementedError
