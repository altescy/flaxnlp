from typing import Any, Callable, Optional

import flax
import jax
from jax._src.numpy.lax_numpy import _ScalarMeta

from flaxnlp.modules.token_embedders.token_embedder import TokenEmbedder

Array = Any


class TokenDropout(flax.linen.Module):
    """Applies token dropout to a batch of input IDs.
    This is basically the same as `flax.linen.Dropout`, but allows specifying the
    value of dropped out items.
    """

    dropout: float
    unknown_index: int
    deterministic: Optional[bool] = None

    @flax.linen.compact
    def __call__(self, inputs: Array, deterministic: Optional[bool] = None) -> Array:
        deterministic = flax.linen.module.merge_param("deterministic", self.deterministic, deterministic)
        if deterministic or self.dropout == 0.0:
            return inputs
        rng = self.make_rng("dropout")
        mask = jax.random.bernoulli(rng, p=self.dropout, shape=inputs.shape)
        return jax.numpy.where(mask, jax.numpy.array([self.unknown_index]), inputs)


class Embedding(TokenEmbedder):
    num_embeddings: int
    embedding_dim: int
    frozen: bool = False
    dropout: float = 0.0
    unknown_index: Optional[int] = None
    deterministic: Optional[bool] = None
    embedding_init: Callable[..., Array] = flax.linen.initializers.normal(stddev=0.1)
    dtype: _ScalarMeta = jax.numpy.float32

    def setup(self) -> None:
        self.embedding = self.param(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.embedding_dim),
            self.dtype,
        )
        self.dropout_layer = flax.linen.Dropout(  # type: ignore[no-untyped-call]
            rate=self.dropout,
        )
        self.token_dropout_layer = TokenDropout(  # type: ignore[no-untyped-call]
            dropout=self.dropout,
            unknown_index=self.unknown_index,
        )

    def __call__(
        self,
        token_ids: Array,
        deterministic: Optional[bool] = None,
        **kwargs: Any,
    ) -> Array:
        """Embeds the input sequences and applies word dropout and dropout.
        Args:
          inputs: Batch of input token ID sequences <int64>[batch_size, seq_length].
          deterministic: Disables dropout when set to True.
        Returns:
          The embedded inputs, shape: <float32>[batch_size, seq_length,
          embedding_size].
        """
        deterministic = flax.linen.module.merge_param("deterministic", self.deterministic, deterministic)
        token_ids = self.token_dropout_layer(token_ids, deterministic=deterministic)
        embeddings = self.embedding[token_ids]

        if self.frozen:
            embeddings = jax.lax.stop_gradient(embeddings)

        return self.dropout_layer(embeddings, deterministic=deterministic)
