from typing import Any, Optional

import flax
import jax

from flaxnlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

Array = Any


class KeysOnlyMlpAttention(flax.linen.Module):
    hidden_dim: int

    @flax.linen.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
    ) -> Array:
        hidden = flax.linen.Dense(  # type: ignore[no-untyped-call]
            self.hidden_dim,
            name="key",
            use_bias=False,
        )(inputs)
        energy = flax.linen.tanh(hidden)
        scores = flax.linen.Dense(  # type: ignore[no-untyped-call]
            1,
            name="energy",
            use_bias=False,
        )(energy)
        scores = scores.squeeze(-1)
        scores = jax.numpy.where(mask, scores, -jax.numpy.inf)
        scores = flax.linen.softmax(scores, axis=-1)

        # Captures the scores if 'intermediates' is mutable, otherwise does nothing.
        self.sow("intermediate", "attention", scores)

        return scores


class SelfAttentiveEncoder(Seq2VecEncoder):
    hidden_dim: int
    output_dim: Optional[int] = None

    @flax.linen.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        deterministic: Optional[bool] = None,
    ) -> Array:
        attention = KeysOnlyMlpAttention(  # type: ignore[no-untyped-call]
            self.hidden_dim,
            name="attention",
        )(inputs, mask)
        output = jax.numpy.einsum("bld,bl->bd", inputs, attention)  # type: ignore[no-untyped-call]

        if self.output_dim is not None:
            output = flax.linen.Dense(  # type: ignore[no-untyped-call]
                self.output_dim,
                name="output",
            )(output)

        return output
