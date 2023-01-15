import jax
import numpy

from flaxnlp.modules.seq2vec_encoders.cnn_encoder import CnnSeq2VecEncoder


def test_cnn_encoder() -> None:
    encoder = CnnSeq2VecEncoder(num_filters=4, ngram_filter_sizes=[2, 3, 4, 5])  # type: ignore[no-untyped-call]
    inputs = numpy.random.RandomState(0).randn(3, 4, 5)
    lengths = numpy.array([3, 2, 1], dtype=numpy.int32)

    rng = jax.random.PRNGKey(0)
    output, _ = encoder.init_with_output(rng, inputs, lengths)

    assert output.shape == (3, 16)
