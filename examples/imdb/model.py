from typing import Any, ClassVar, Dict, Optional, cast

import colt
import flax
import optax
from colt import Lazy

from flaxnlp.models.model import Model
from flaxnlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from flaxnlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from flaxnlp.modules.token_embedders.token_embedder import TokenEmbedder
from flaxnlp.util import sequence_mask

Array = Any


class TextClassifier(Model):  # type: ignore[misc]
    rngkeys: ClassVar = {"dropout"}
    mutables = {"batch_stats"}

    vocab_size: int
    num_classes: int
    embedder_config: Lazy[TokenEmbedder]
    vectorizer_config: Lazy[Seq2VecEncoder]
    contextualizer_config: Optional[Lazy[Seq2SeqEncoder]] = None
    deterministic: Optional[bool] = None
    dropout: float = 0.1

    @staticmethod
    def get_mask_from_text(text: Dict[str, Any]) -> Array:
        if "mask" in text:
            return text["mask"]
        raise ValueError("No mask found in text.")

    def setup(self) -> None:
        self.embedder = self.embedder_config.construct(num_embeddings=self.vocab_size)
        self.vectorizer = self.vectorizer_config.construct()
        self.classifier = flax.linen.Dense(self.num_classes)  # type: ignore[no-untyped-call]
        self.batch_norm = flax.linen.BatchNorm(use_bias=False, use_scale=False)  # type: ignore[no-untyped-call]
        self.dropout_layer = flax.linen.Dropout(rate=self.dropout)  # type: ignore[no-untyped-call]
        self.contextualizer = None
        if self.contextualizer_config is not None:
            self.contextualizer = self.contextualizer_config.construct()

    def __call__(  # type: ignore[override]
        self,
        text: Dict[str, Any],
        label: Optional[Array],
        *,
        train: bool = False,
    ) -> Dict[str, Any]:
        deterministic = flax.linen.module.merge_param("deterministic", self.deterministic, not train)
        mask = self.get_mask_from_text(text)
        embeddings = self.embedder(deterministic=deterministic, **text)
        if self.contextualizer is not None:
            embeddings = self.contextualizer(embeddings, mask, deterministic=deterministic)
        encodings = self.vectorizer(embeddings, mask, deterministic=deterministic)
        encodings = self.batch_norm(encodings, use_running_average=not train)
        logits = self.classifier(self.dropout_layer(encodings, deterministic=deterministic))
        output = {"logits": logits}
        if label is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
            accuracy = (logits.argmax(-1) == label).mean()
            output["loss"] = loss
            output["metrics"] = {"accuracy": accuracy}
        return output
