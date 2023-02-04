from typing import Any, Callable, Optional, Tuple

import flax
import jax
import numpy

Array = Any


def sinusoidal_init(max_len: int = 2048) -> Array:
    def init(key: Array, shape: Tuple[int, int, int], dtype: Any = numpy.float32) -> Array:
        del key, dtype
        d_feature = shape[-1]
        pe = numpy.zeros((max_len, d_feature), dtype=numpy.float32)
        position = numpy.arange(0, max_len)[:, numpy.newaxis]
        div_term = numpy.exp(numpy.arange(0, d_feature, 2) * -(numpy.log(10000.0) / d_feature))
        pe[:, 0::2] = numpy.sin(position * div_term)
        pe[:, 1::2] = numpy.cos(position * div_term)
        pe = pe[numpy.newaxis, :, :]  # [1, max_len, d_feature]
        return jax.numpy.array(pe)

    return init


class AddPositionEmbedding(flax.linen.Module):
    max_length: int = 2048
    posemb_init: Optional[Callable] = None

    @flax.linen.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies AddPositionEmbedding module.
        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.
        Args:
          inputs: input data.
        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        length = inputs.shape[1]
        pos_emb_shape = (1, self.max_length, inputs.shape[-1])
        if self.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=self.max_length)(None, pos_emb_shape, None)
        else:
            pos_embedding = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        pe = pos_embedding[:, :length, :]
        return inputs + pe
