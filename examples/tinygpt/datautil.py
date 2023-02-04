import pickle
from os import PathLike
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import colt
from collatable import Instance, TextField
from collatable.extras.dataset import Dataset
from collatable.extras.indexer import TokenIndexer


class Tokenizer(colt.Registrable):  # type: ignore[misc]
    def __call__(self, text: str) -> List[str]:
        raise NotImplementedError


@Tokenizer.register("character")
class CharacterTokenizer(Tokenizer):
    def __call__(self, text: str) -> List[str]:
        return list(text)


class TokenIndexerBuilder(colt.Registrable):  # type: ignore[misc]
    def __call__(self, documents: Iterable[Sequence[str]]) -> TokenIndexer[str]:
        raise NotImplementedError


@TokenIndexerBuilder.register("single_id")
class SingleIdTokenIndexer(TokenIndexerBuilder):  # type: ignore[misc]
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = {
            "specials": ["<pad>", "<unk>", "<eos>"],
            "default": "<unk>",
            **kwargs,
        }

    def __call__(self, documents: Iterable[Sequence[str]]) -> TokenIndexer[str]:
        return TokenIndexer.from_documents(documents, **self.kwargs)


class GptDataModule:
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexer: TokenIndexerBuilder,
    ) -> None:
        self.tokenizer = tokenizer
        self.token_indexer: Optional[TokenIndexer[str]] = None
        self._token_indexer_builder = token_indexer

    @property
    def vocab_size(self) -> int:
        if self.token_indexer is None:
            raise ValueError("build_vocab must be called before vocab_size.")
        return len(self.token_indexer)

    def build_vocab(self, dataset: Iterable[str]) -> None:
        if self.token_indexer is not None:
            raise ValueError("build_vocab can only be called once.")
        self.token_indexer = self._token_indexer_builder(self.tokenizer(text) for text in dataset)

    def text_to_instance(self, text: str) -> Instance:
        if self.token_indexer is None:
            raise ValueError("build_vocab must be called before text_to_instance.")
        source_tokens = self.tokenizer(text)
        target_tokens = source_tokens[1:] + ["<eos>"]
        return Instance(
            source=TextField(source_tokens, indexer=self.token_indexer, padding_value=self.token_indexer["<pad>"]),
            target=TextField(target_tokens, indexer=self.token_indexer, padding_value=self.token_indexer["<pad>"]),
        )

    def read_dataset(
        self,
        dataset: Iterable[str],
        train: bool = False,
        path: Optional[Union[str, PathLike]] = None,
    ) -> Dataset[Instance]:
        if self.token_indexer is None:
            if train:
                self.build_vocab(dataset)
            else:
                raise ValueError("build_vocab must be called before read_dataset.")
        output = Dataset.from_iterable((self.text_to_instance(data) for data in dataset), path=path)
        output.flush()
        return output

    def save(self, filename: Union[str, PathLike]) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: Union[str, PathLike]) -> "GptDataModule":
        with open(filename, "rb") as f:
            datamodule = pickle.load(f)
            assert isinstance(datamodule, cls)
            return datamodule
