from typing import Any, Optional, Sequence

import flax
import jax

from flaxnlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from flaxnlp.util import sequence_mask

Array = Any


class CnnSeq2VecEncoder(Seq2VecEncoder):
    num_filters: int
    ngram_filter_sizes: Sequence[int]
    output_size: Optional[int] = None

    def setup(self) -> None:
        self._convolutions = [
            flax.linen.Conv(  # type: ignore[no-untyped-call]
                features=self.num_filters,
                kernel_size=(filter_size,),
            )
            for filter_size in self.ngram_filter_sizes
        ]
        self._dense = (
            flax.linen.Dense(  # type: ignore[no-untyped-call]
                features=self.output_size,
            )
            if self.output_size is not None
            else None
        )

    def __call__(
        self,
        inputs: Array,
        lengths: Array,
        deterministic: Optional[bool] = None,
    ) -> Array:
        mask = sequence_mask(lengths, inputs.shape[1])
        inputs = inputs * mask[:, :, None]
        output = jax.numpy.concatenate(
            [
                jax.numpy.where(mask[:, :, None], conv(inputs), -jax.numpy.inf).max(axis=1)
                for conv in self._convolutions
            ],
            axis=-1,
        )
        if self._dense:
            output = self._dense(output)
        return output
