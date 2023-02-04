import pickle
from os import PathLike
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import collatable
import colt
from collatable import Instance, LabelField, TextField
from collatable.extras.dataset import Dataset
from collatable.extras.indexer import TokenIndexer


class TokenIndexerBuilder(colt.Registrable):  # type: ignore[misc]
    def __call__(self, documents: Iterable[Sequence[str]]) -> TokenIndexer[str]:
        raise NotImplementedError


@TokenIndexerBuilder.register("single_id")
class SingleIdTokenIndexer(TokenIndexerBuilder):  # type: ignore[misc]
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = {
            "specials": ["<pad>", "<unk>"],
            "default": "<unk>",
            **kwargs,
        }

    def __call__(self, documents: Iterable[Sequence[str]]) -> TokenIndexer[str]:
        return TokenIndexer.from_documents(documents, **self.kwargs)


class Tokenizer(colt.Registrable):  # type: ignore[misc]
    def __call__(self, text: str) -> List[str]:
        raise NotImplementedError


@Tokenizer.register("whitespace")
class WhitespaceTokenizer(Tokenizer):  # type: ignore[misc]
    def __call__(self, text: str) -> List[str]:
        return text.lower().split()


class ImdbDataModule:
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

    @property
    def num_classes(self) -> int:
        return 2

    def build_vocab(self, dataset: Sequence[Dict[str, Any]]) -> None:
        if self.token_indexer is not None:
            raise ValueError("build_vocab can only be called once.")
        self.token_indexer = self._token_indexer_builder(self.tokenizer(data["text"]) for data in dataset)

    def text_to_instance(self, data: Dict[str, Any]) -> Instance:
        if self.token_indexer is None:
            raise ValueError("build_vocab must be called before text_to_instance.")
        return Instance(
            text=TextField(self.tokenizer(data["text"]), indexer=self.token_indexer),
            label=LabelField(data["label"]),
        )

    def read_dataset(self, dataset: Sequence[Dict[str, Any]], train: bool = False) -> List[Instance]:
        if self.token_indexer is None:
            if train:
                self.build_vocab(dataset)
            else:
                raise ValueError("build_vocab must be called before read_dataset.")
        return list(self.text_to_instance(data) for data in dataset)

    def save(self, filename: Union[str, PathLike]) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: Union[str, PathLike]) -> "ImdbDataModule":
        with open(filename, "rb") as f:
            datamodule = pickle.load(f)
            assert isinstance(datamodule, cls)
            return datamodule
