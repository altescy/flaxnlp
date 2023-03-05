import abc
from typing import Any, ClassVar, Dict, Optional, Set, Tuple

import colt
import flax
import jax

Array = Any


class Model(abc.ABC, flax.linen.Module, colt.Registrable):
    rngkeys: ClassVar[Set[str]] = set()
    mutables: ClassVar[Set[str]] = set()
    default_training_module: ClassVar[str] = "default"

    @abc.abstractmethod
    def __call__(
        self,
        *args: Any,
        train: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def split_rngs(
        self,
        rngs: Any,
        train: bool = False,
        additional_keys: Optional[Set[str]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        additional_keys = additional_keys or set()
        required_rngkeys = list(self.rngkeys | additional_keys)

        if not train:
            if "dropout" in required_rngkeys:
                required_rngkeys.remove("dropout")

        rngs, *new_rngs = jax.random.split(rngs, len(required_rngkeys) + 1)
        rngs_for_call = {name: rng for name, rng in zip(required_rngkeys, new_rngs)}
        return rngs, rngs_for_call
