import jax
import numpy

from flaxnlp.modules.seq2seq_encoders.lstm_encoder import LSTMEncoder
from flaxnlp.util import sequence_mask


def test_lstm_encoder() -> None:
    encoder = LSTMEncoder(hidden_dim=4, num_layers=2, bidirectional=True)  # type: ignore[no-untyped-call]

    inputs = numpy.random.RandomState(0).normal(size=(2, 3, 4))
    mask = sequence_mask(numpy.array([2, 3], dtype=numpy.int32))

    outputs, _ = encoder.init_with_output(jax.random.PRNGKey(0), inputs, mask, False)
    assert outputs.shape == (2, 3, 8)
