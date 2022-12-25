from typing import Any, Callable, Optional, Sequence

import flax

Array = Any


class FeedForward(flax.linen.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[Array], Array] = flax.linen.relu
    dropout: float = 0.0

    @flax.linen.compact
    def __call__(self, inputs: Array, deterministic: Optional[bool] = None) -> Array:
        for dim in self.hidden_dims[:-1]:
            inputs = flax.linen.Dense(dim)(inputs)  # type: ignore[no-untyped-call]
            inputs = self.activation(inputs)
            inputs = flax.linen.Dropout(  # type: ignore[no-untyped-call]
                rate=self.dropout,
                deterministic=deterministic,
            )(inputs)
        return flax.linen.Dense(self.hidden_dims[-1])(inputs)  # type: ignore[no-untyped-call]
