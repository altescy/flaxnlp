import functools
from typing import Any, Tuple, cast

import flax
import jax

from flaxnlp import util
from flaxnlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

Array = Any


class SimpleLSTM(flax.linen.Module):
    """A simple unidirectional LSTM."""

    @functools.partial(
        flax.linen.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @flax.linen.compact
    def __call__(
        self,
        carry: Tuple[Array, Array],
        inputs: Array,
    ) -> Tuple[Tuple[Array, Array], Array]:
        return flax.linen.OptimizedLSTMCell()(carry, inputs)  # type: ignore[no-untyped-call]

    @staticmethod
    def initialize_carry(
        batch_dims: Tuple[int, ...],
        hidden_size: int,
    ) -> Tuple[Array, Array]:
        # Use fixed random key since default state init fn is just zeros.
        return cast(
            Tuple[Array, Array],
            flax.linen.OptimizedLSTMCell.initialize_carry(  # type: ignore[no-untyped-call]
                jax.random.PRNGKey(0),
                batch_dims,
                hidden_size,
            ),
        )


class LSTMLayer(flax.linen.Module):
    """A simple bi-directional LSTM."""

    hidden_size: int
    bidirectional: bool

    def setup(self) -> None:
        self.forward_lstm = SimpleLSTM()  # type: ignore[no-untyped-call]
        self.backward_lstm = SimpleLSTM() if self.bidirectional else None  # type: ignore[no-untyped-call]

    def __call__(
        self,
        inputs: Array,
        lengths: Array,
    ) -> Array:
        batch_size = inputs.shape[0]

        # Forward LSTM.
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, outputs = self.forward_lstm(initial_state, inputs)

        # Backward LSTM.
        if self.backward_lstm is not None:
            reversed_inputs = util.flip_sequences(inputs, lengths)
            initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
            _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
            backward_outputs = util.flip_sequences(backward_outputs, lengths)

            # Concatenate the forward and backward representations.
            outputs = jax.numpy.concatenate([outputs, backward_outputs], -1)

        return outputs


class LSTMEncoder(Seq2SeqEncoder):
    """A multi-layer LSTM encoder."""

    hidden_size: int
    num_layers: int
    bidirectional: bool = False
    dropout: float = 0.0

    @flax.linen.compact
    def __call__(
        self,
        inputs: Array,
        lengths: Array,
        deterministic: bool,
    ) -> Array:
        for _ in range(self.num_layers):
            inputs = LSTMLayer(hidden_size=self.hidden_size, bidirectional=self.bidirectional)(inputs, lengths)  # type: ignore[no-untyped-call]
            inputs = flax.linen.Dropout(rate=self.dropout, deterministic=deterministic)(inputs)  # type: ignore[no-untyped-call]
        return inputs
