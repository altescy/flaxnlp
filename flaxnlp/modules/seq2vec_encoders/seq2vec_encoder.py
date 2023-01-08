import abc
from typing import Any, Optional

import flax

Array = Any


class Seq2VecEncoder(abc.ABC, flax.linen.Module):
    @abc.abstractmethod
    def __call__(
        self,
        inputs: Array,
        lengths: Array,
        deterministic: Optional[bool] = None,
    ) -> Array:
        raise NotImplementedError
