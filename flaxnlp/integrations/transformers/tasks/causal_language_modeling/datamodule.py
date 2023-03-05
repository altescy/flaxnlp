from os import PathLike
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy
from collatable import Field, Instance, TensorField, TextField

from flaxnlp.data import Dataset
from flaxnlp.integrations.transformers.tokenizers import TransformersTokenizer

Array = Any


class DataModuleForTransformersCausalLM:
    def __init__(
        self,
        tokenizer: TransformersTokenizer,
        block_size: int = 128,
        truncation: bool = True,
        text_field_name: str = "text",
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.truncation = truncation
        self.text_field_name = text_field_name

    def get_model_kwargs(self) -> Dict[str, Any]:
        return dict(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def read_dataset(
        self,
        dataset: Union[Sequence[Mapping[str, Any]], Sequence[str]],
        cache_path: Optional[Union[str, PathLike]] = None,
    ) -> Sequence[Instance]:
        def instance_iterator() -> Iterator[Instance]:
            for example in dataset:
                if isinstance(example, Mapping):
                    for key in self.text_field_name.split("."):
                        example = example[key]  # type: ignore
                    if not isinstance(example, str):
                        raise ValueError(f"Expected a string, but got {type(example)}")

                if not isinstance(example, str):
                    raise ValueError(f"Expected a string for the text field, but got {type(example)}")

                results: Dict[str, List[Field]] = {}
                for key, values in self.tokenizer(example, truncation=self.truncation).items():
                    results[key] = []
                    for i in range(0, len(values), self.block_size):
                        start = max(0, min(i, len(values) - self.block_size))
                        end = start + self.block_size
                        block = values[start:end]

                        field: Field
                        if key == "input_ids":
                            field = TextField(block, indexer=numpy.array, padding_value=self.tokenizer.eos_token_id)
                        else:
                            field = TensorField(numpy.array(block))

                        results[key].append(field)

                for i in range(max(len(x) for x in results.values())):
                    fields = {key: values[i] for key, values in results.items()}
                    fields["labels"] = fields["input_ids"].copy()
                    yield Instance(**fields)

        if cache_path is not None:
            return Dataset.from_iterable(instance_iterator(), path=cache_path)

        return list(instance_iterator())
