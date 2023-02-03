import abc
from typing import Any, ClassVar, Dict, Optional, Set, Tuple

import flax
import jax

Array = Any


class Model(abc.ABC, flax.linen.Module):
    required_rngkeys: ClassVar[Set[str]] = set()

    @abc.abstractmethod
    def __call__(
        self,
        *args: Any,
        train: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_loss(
        self,
        *args: Any,
        train: bool = False,
        **kwargs: Any,
    ) -> Array:
        outputs = self(*args, train=train, **kwargs)
        loss = outputs["loss"]
        return outputs, loss

    def compute_metrics(
        self,
        *args: Any,
        train: bool = False,
        **kwargs: Any,
    ) -> Dict[str, float]:
        outputs = self(*args, train=train, **kwargs)
        metrics: Dict[str, float] = outputs["metrics"]
        return metrics

    def split_rngs(
        self,
        rngs: Any,
        train: bool = False,
        additional_keys: Optional[Set[str]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        additional_keys = additional_keys or set()
        required_rngkeys = list(self.required_rngkeys | additional_keys)

        if not train:
            if "dropout" in required_rngkeys:
                required_rngkeys.remove("dropout")

        rngs, *new_rngs = jax.random.split(rngs, len(required_rngkeys) + 1)
        rngs_for_call = {name: rng for name, rng in zip(required_rngkeys, new_rngs)}
        return rngs, rngs_for_call
