import jax
import numpy

from flaxnlp.modules.seq2seq_encoders.transformer_encoder import TransformerEncoder


def test_transformer_encoder() -> None:
    encoder = TransformerEncoder(  # type: ignore[no-untyped-call]
        input_dim=4,
        num_layers=2,
        num_heads=2,
        hidden_dim=8,
    )

    inputs = numpy.random.RandomState(0).normal(size=(2, 3, 4))
    lengths = numpy.array([3, 2])

    outputs, _ = encoder.init_with_output(jax.random.PRNGKey(0), inputs, lengths, deterministic=True)
    assert outputs.shape == (2, 3, 4)
