from typing import Any

from tranformers import FlaxAutoModel

from flaxnlp.modules.token_embedders.token_embedder import TokenEmbedder

Array = Any


class PretrainedTransformerEmbedder(TokenEmbedder):
    model_name: str

    def setup(self) -> None:
        self.transformer = FlaxAutoModel.from_pretrained(self.model_name)

    def __call__(self, **inputs: Any) -> Array:
        transformer_output = self.transformer(
            **inputs,
            return_dict=True,
        )
        return transformer_output.last_hidden_state
