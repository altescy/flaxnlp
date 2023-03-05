import functools
import json
import mmap
import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import colt

T = TypeVar("T")
Self = TypeVar("Self", bound="Dataset")


class Index(NamedTuple):
    page: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        return self.page.to_bytes(4, "little") + self.offset.to_bytes(4, "little") + self.length.to_bytes(4, "little")

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Index":
        return cls(
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
        )


class Dataset(Sequence[T], colt.Registrable):
    def __init__(
        self,
        path: Optional[Union[str, PathLike]] = None,
        pagesize: int = 1024 * 1024 * 1024,
    ) -> None:
        self._delete_on_exit = path is None
        self._path = Path(path or tempfile.TemporaryDirectory().name)
        self._pagesize = pagesize
        self._indices: List[Index] = []
        self._pageios: Dict[int, BinaryIO] = {}

        self._path.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        index_filename = self._get_index_filename()
        if not index_filename.exists():
            index_filename.touch()

        metadata_filename = self._get_metadata_filename()
        if metadata_filename.exists():
            self._load_metadata()
        else:
            self._save_metadata()

        self._indexio: BinaryIO = index_filename.open("rb+")
        if self._indexio.seek(0, 2) > 0:
            self._load_indices()

        for page, page_filename in self._iter_page_filenames():
            self._pageios[page] = page_filename.open("rb+")

    def __del__(self) -> None:
        if self._delete_on_exit:
            shutil.rmtree(self._path)

    @staticmethod
    def _encode(obj: T) -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def _decode(data: bytes) -> T:
        return cast(T, pickle.loads(data))

    @property
    def path(self) -> Path:
        return self._path

    def _get_index_filename(self) -> Path:
        return self._path / "index.bin"

    def _get_metadata_filename(self) -> Path:
        return self._path / "metadata.json"

    def _get_lock_filename(self) -> Path:
        return self._path / "lock"

    def _get_page_filename(self, page: int) -> Path:
        return self._path / f"page_{page:08d}"

    def _iter_page_filenames(self) -> Iterable[Tuple[int, Path]]:
        for page_filename in self._path.glob("page_*"):
            page = int(page_filename.stem.split("_", 1)[1])
            yield page, page_filename

    def _add_index(self, index: Index) -> None:
        self._indices.append(index)
        self._indexio.seek(0, 2)
        self._indexio.write(index.to_bytes())

    def _load_indices(self) -> None:
        if self._indices:
            raise RuntimeError("indices already loaded")
        eof = self._indexio.seek(0, 2)
        self._indexio.seek(0)
        while self._indexio.tell() < eof:
            self._indices.append(Index.from_binaryio(self._indexio))

    def _load_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("r") as f:
            metadata = json.load(f)
        self._pagesize = metadata["pagesize"]

    def _save_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("w") as f:
            json.dump({"pagesize": self._pagesize}, f)

    def append(self, obj: T) -> None:
        binary = self._encode(obj)

        pageio: BinaryIO
        if not self._pageios:
            page = 0
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio
        else:
            page = len(self._pageios) - 1
            pageio = self._pageios[page]

        offset = pageio.seek(0, 2)
        if offset + len(binary) > self._pagesize:
            page += 1
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio

        pageio.write(binary)
        self._add_index(Index(page, offset, len(binary)))

    def flush(self) -> None:
        for pageio in self._pageios.values():
            pageio.flush()
        self._indexio.flush()

    def close(self) -> None:
        for pageio in self._pageios.values():
            pageio.close()
        self._indexio.close()

    @contextmanager
    def lock(self) -> Iterator[None]:
        import fcntl

        lockfile = self._get_lock_filename().open("w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
            lockfile.close()

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> "List[T]":
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, List[T]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            index = self._indices[key]
            pageio = self._pageios[index.page]
            pageio.seek(index.offset)
            return self._decode(pageio.read(index.length))
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

    def __getstate__(self) -> Dict[str, Any]:
        if self._delete_on_exit:
            raise RuntimeError("cannot pickle a temporary database")
        return {"path": self._path}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._path = state["path"]
        self._delete_on_exit = False
        self._pagesize = 1024 * 1024 * 1024
        self._indices = []
        self._pageios = {}
        self._restore()

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[T],
        path: Optional[Union[str, PathLike]] = None,
        pagesize: int = 1024 * 1024 * 1024,
    ) -> "Dataset[T]":
        dataset = cls(path, pagesize)
        for obj in iterable:
            dataset.append(obj)
        dataset.flush()
        return dataset

    @classmethod
    def from_path(
        cls: Type[Self],
        path: Union[str, PathLike],
    ) -> Self:
        return cls(path)


class LineByLineTextDataset(Sequence[str]):
    def __init__(
        self,
        filename: Union[str, PathLike],
        encoding: str = "utf-8",
    ) -> None:
        self._filename = Path(filename)
        self._encoding = encoding
        with self._open(filename, os.O_RDWR) as fd:
            self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    @property
    def filename(self) -> Path:
        return self._filename

    @functools.lru_cache
    def _get_offsets(self) -> List[int]:
        self._mmap.seek(0)
        return [0] + [self._mmap.tell() for line in iter(self._mmap.readline, b"")]

    @property
    def _offsets(self) -> List[int]:
        return self._get_offsets()

    @contextmanager
    def _open(self, filename: Union[str, PathLike], flags: int, **kwargs: Any) -> Iterator[int]:
        try:
            fd = os.open(filename, flags, **kwargs)
            yield fd
        finally:
            os.close(fd)

    def __len__(self) -> int:
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> str:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[str]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[str, List[str]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(f"index {index} out of range")
            start = self._offsets[index]
            end = self._offsets[index + 1] - 1
            self._mmap.seek(start)
            return self._mmap.read(end - start).decode(self._encoding)
        else:
            raise TypeError(f"index must be int or slice, not {type(index)}")

    def __getstate__(self) -> Dict[str, Any]:
        return {"filename": self._filename, "encoding": self._encoding, "decoder": self._decoder}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._filename = state["filename"]
        self._encoding = state["encoding"]
        self._decoder = state["decoder"]
        self._mmap = mmap.mmap(self._filename.open("rb").fileno(), 0, access=mmap.ACCESS_READ)

    def __del__(self) -> None:
        if hasattr(self, "_mmap"):
            self._mmap.close()


class JSONLinesDataset(Sequence[Dict[str, Any]]):
    def __init__(
        self,
        filename: Union[str, PathLike],
        encoding: str = "utf-8",
    ) -> None:
        self._text_dataset = LineByLineTextDataset(filename, encoding)

    def __len__(self) -> int:
        return len(self._text_dataset)

    @overload
    def __getitem__(self, index: int) -> Dict[str, Any]:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[Dict[str, Any]]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            value = json.loads(self._text_dataset[index])
            assert isinstance(value, dict)
            return value
        else:
            raise TypeError(f"index must be int or slice, not {type(index)}")


Dataset.register("default")(Dataset)
Dataset.register("jsonlines")(JSONLinesDataset)
Dataset.register("linebylinetext")(LineByLineTextDataset)
