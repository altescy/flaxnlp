import jax
import numpy

from flaxnlp.modules.seq2vec_encoders.self_attentive_encoder import SelfAttentievEncoder


def test_self_attentive_encoder() -> None:
    encoder = SelfAttentievEncoder(  # type: ignore[no-untyped-call]
        hidden_dim=3,
        output_dim=2,
    )

    rng = jax.random.PRNGKey(0)
    inputs = numpy.random.RandomState(0).randn(2, 3, 4)
    lengths = numpy.array([3, 2], dtype=numpy.int32)
    output, _ = encoder.init_with_output(rng, inputs, lengths, deterministic=True)

    assert output.shape == (2, 2)
