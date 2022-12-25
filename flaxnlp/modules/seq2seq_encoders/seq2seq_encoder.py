import abc
from typing import Any

import flax

Array = Any


class Seq2SeqEncoder(abc.ABC, flax.linen.Module):
    @abc.abstractmethod
    def __call__(self, inputs: Array, lengths: Array, deterministic: bool) -> Array:
        raise NotImplementedError
