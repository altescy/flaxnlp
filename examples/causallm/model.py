from typing import Any, ClassVar, Dict, Optional

import flax
import jax
import optax

from flaxnlp.models.model import Model
from flaxnlp.modules.position_embedding import AddPositionEmbedding
from flaxnlp.modules.seq2seq_encoders.transformer_encoder import TransformerEncoder
from flaxnlp.modules.token_embedders.embedding import Embedding

Array = Any


class CausalLM(Model):
    rngkeys: ClassVar = {"dropout"}

    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    def setup(self) -> None:
        self.embedder = Embedding(  # type: ignore[no-untyped-call]
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.encoder = TransformerEncoder(  # type: ignore[no-untyped-call]
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            causal=True,
            activation=flax.linen.gelu,
        )
        self.position_embedder = AddPositionEmbedding()  # type: ignore[no-untyped-call]
        self.classifier = flax.linen.Dense(self.vocab_size)  # type: ignore[no-untyped-call]

    def __call__(  # type: ignore[override]
        self,
        source: Dict[str, Any],
        target: Optional[Dict[str, Any]] = None,
        train: bool = False,
    ) -> Dict[str, Any]:
        deterministic = flax.linen.module.merge_param("deterministic", self.deterministic, not train)

        source_mask = self._get_mask_from_text(source)

        embeddings = self.embedder(deterministic=deterministic, **source)
        embeddings = self.position_embedder(embeddings)
        encodings = self.encoder(embeddings, source_mask, deterministic=deterministic)
        logits = self.classifier(encodings)

        output = {"logits": logits}

        if target is not None:
            target_mask = self._get_mask_from_text(target)
            labels = target["token_ids"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).sum() / len(logits)
            output["loss"] = loss

        return output

    def _get_mask_from_text(self, text: Dict[str, Any]) -> Array:
        if "mask" in text:
            return text["mask"]
        raise ValueError("Mask not found in text")

    def generate(
        self,
        rngs: Any,
        params: Any,
        token_ids: Array,
        max_new_tokens: int,
        topk: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Array:
        @jax.jit
        def sample(rng: Any, variables: Any, token_ids: Array) -> Array:
            source = {"token_ids": token_ids, "mask": jax.numpy.ones_like(token_ids)}
            logits = self.apply(variables=variables, train=False, source=source)["logits"][:, -1, :]  # type: ignore
            logits = logits / temperature
            if topk is not None:
                v = jax.numpy.sort(logits, axis=-1)[:, ::-1][:, :topk]
                logits = jax.numpy.where(logits < v[:, [-1]], -jax.numpy.inf, logits)
            new_token_ids = jax.random.categorical(rng, logits)
            return new_token_ids

        for _ in range(max_new_tokens):
            rngs, subrng = jax.random.split(rngs)
            new_token_ids = sample(subrng, params, token_ids)
            token_ids = jax.numpy.concatenate([token_ids, new_token_ids[None, :]], axis=-1)
        return token_ids
