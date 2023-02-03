from typing import Any, Callable, Optional

import flax

from flaxnlp.modules.feedforward import FeedForward
from flaxnlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

Array = Any


class TransformerLayer(flax.linen.Module):
    input_dim: int
    num_heads: int
    hidden_dim: int
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-6
    activation: Callable[[Array], Array] = flax.linen.relu

    def setup(self) -> None:
        self.self_attention = flax.linen.SelfAttention(  # type: ignore[no-untyped-call]
            num_heads=self.num_heads,
            qkv_features=self.input_dim,
            dropout_rate=self.dropout,
        )
        self.feedforward = FeedForward(  # type: ignore[no-untyped-call]
            hidden_dims=[self.hidden_dim, self.input_dim],
            dropout=self.dropout,
        )
        self.layer_norm1 = flax.linen.LayerNorm(epsilon=self.layernorm_epsilon)  # type: ignore[no-untyped-call]
        self.layer_norm2 = flax.linen.LayerNorm(epsilon=self.layernorm_epsilon)  # type: ignore[no-untyped-call]

    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ) -> Array:
        h = self.layer_norm1(inputs + self.self_attention(inputs, mask=mask, deterministic=deterministic))
        h = self.layer_norm2(h + self.feedforward(h, deterministic=deterministic))
        return h


class TransformerEncoder(Seq2SeqEncoder):
    input_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    causal: bool = False
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-6
    activation: Callable[[Array], Array] = flax.linen.relu

    @flax.linen.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        deterministic: Optional[bool] = None,
    ) -> Array:
        if self.causal:
            attention_mask = flax.linen.make_causal_mask(mask)  # type: ignore[no-untyped-call]
        else:
            attention_mask = flax.linen.make_attention_mask(mask, mask)  # type: ignore[no-untyped-call]
        for _ in range(self.num_layers):
            inputs = TransformerLayer(  # type: ignore[no-untyped-call]
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                layernorm_epsilon=self.layernorm_epsilon,
                activation=self.activation,
            )(inputs, mask=attention_mask, deterministic=deterministic)
        return inputs
