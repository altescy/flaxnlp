import jax
import numpy

from flaxnlp.modules.token_embedders.embedding import Embedding


def test_embedding() -> None:
    num_embeddings = 5
    embedding_dim = 3
    embedder = Embedding(  # type: ignore[no-untyped-call]
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )
    rng = jax.random.PRNGKey(0)
    token_ids = numpy.array([[2, 4, 3], [2, 6, 3]], dtype=numpy.int32)
    output, _ = embedder.init_with_output(rng, token_ids, deterministic=True)

    assert output.shape == (2, 3, 3)
