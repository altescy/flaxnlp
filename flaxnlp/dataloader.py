import math
import random
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence, TypeVar

T = TypeVar("T", bound=Mapping[str, Any])


class BatchIterator:
    def __init__(
        self,
        dataset: Sequence[T],
        collate_fn: Callable[[Sequence[T]], Dict[str, Any]],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._collate_fn = collate_fn
        self._offset = 0
        self._indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.shuffle(self._indices)

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._dataset) // self._batch_size
        return math.ceil(len(self._dataset) / self._batch_size)

    def __next__(self) -> Dict[str, Any]:
        if self._offset >= len(self._dataset):
            raise StopIteration
        if self._offset + self._batch_size > len(self._dataset):
            if self._drop_last:
                raise StopIteration
            batch_indices = self._indices[self._offset :]
        else:
            batch_indices = self._indices[self._offset : self._offset + self._batch_size]
        self._offset += self._batch_size
        return self._collate_fn([self._dataset[i] for i in batch_indices])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self


class DataLoader:
    def __init__(
        self,
        collate_fn: Callable[[Sequence[T]], Dict[str, Any]],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._collate_fn = collate_fn
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __call__(self, dataset: Sequence[T]) -> BatchIterator:
        return BatchIterator(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
        )
