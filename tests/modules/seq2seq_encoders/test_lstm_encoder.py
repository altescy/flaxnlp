import jax
import numpy

from flaxnlp.modules.seq2seq_encoders.lstm_encoder import LSTMEncoder


def test_lstm_encoder() -> None:
    encoder = LSTMEncoder(hidden_size=4, num_layers=2, bidirectional=True)  # type: ignore[no-untyped-call]

    inputs = numpy.random.RandomState(0).normal(size=(2, 3, 4))
    lengths = numpy.array([2, 3], dtype=numpy.int32)

    outputs, _ = encoder.init_with_output(jax.random.PRNGKey(0), inputs, lengths, False)
    assert outputs.shape == (2, 3, 8)
